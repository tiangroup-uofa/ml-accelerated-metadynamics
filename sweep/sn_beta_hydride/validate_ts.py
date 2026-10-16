#!/usr/bin/env python3
"""
validate_ts.py
==============
Stage 4: Hessian-based characterization of a stationary point (normally the
Sella result), using the **mass-weighted** normal-mode analysis in
``vibrations.py`` (generalized from ``sweep/corrections/phase2_lib.py``).

Reports
  * all frequencies (cm^-1; imaginary as negative) and the lowest ten
  * number of significant imaginary modes (nu < -imag_cutoff_cm, default 50)
    plus any small imaginary modes below the cutoff (numerical noise / soft modes)
  * the dominant imaginary frequency and its unit Cartesian displacement vector
  * per-atom participation of that mode (is the transferring H dominant?)
  * overlap |c . v| of that mode with the gradient of the configured
    hydride-transfer CV (``ts_mode_reference``, default ``delta_H``)
  * +/- imaginary-mode connectivity check (``connectivity.py``; config section
    ``connectivity``): displace the TS both ways along the normalized mode,
    relax, label each side reactant-/product-like from the mapped hydride
    coordinate, compare with the relaxed endpoints (RMSD, core distances,
    torsions, energy offsets). A sanity check, NOT an IRC.
  * energetics against BOTH labelled reference states (``energetics.py``):
    ``relaxed_endpoint`` (stage-1 endpoints) and ``connected_side_minimum``
    (the +/- mode minima). No unlabeled barrier is written.
  * a single certification label (``connectivity.ts_certification``):
    CERTIFIED only with stationarity + exactly one significant imaginary mode
    aligned with the hydride transfer + valid geometry + connectivity.

A valid first-order saddle for this step has exactly one significant imaginary
mode that is dominated by the hydride moving between C2 and C1. The verdict is
reported as evidence, never forced: failures are written out, not hidden.

Partial Hessians (``hessian.atoms`` = "free" with fixed atoms, or an explicit
list) are an approximation and are flagged in the output.

Outputs (vib/): validate_ts.json, frequencies.csv, hessian.npy,
normal_modes.npz (all frequencies + mass-weighted eigenvectors + 1/sqrt(m)
weights, so any mode can be inspected later), imag_mode_vector.npy (unit
Cartesian displacement of the dominant imaginary mode, n_atoms x 3),
imag_mode.xyz (animation), connectivity/ (start + relaxed side structures,
trajectories, logs)

    python validate_ts.py --config config.json [--ts path/to/ts.xyz] [--no-connectivity]

Exit code 0 only when the label is CERTIFIED; 2 otherwise; 1 if refused.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from ase.io import read, write  # noqa: E402

from calculators import add_calc_args, model_from_config, parse_calc_settings  # noqa: E402
from config_io import fixed_atoms, index_of, load_config, output_root, section, write_json  # noqa: E402
from connectivity import (connectivity_check, options as conn_options,  # noqa: E402
                          side_minimum_reference, ts_certification)
from energetics import energetics, format_table, reference_state  # noqa: E402
from validate_structure import (ValidationError, print_report, require_valid,  # noqa: E402
                                validate_single)
from vibrations import analyze_hessian, hessian, normal_modes  # noqa: E402
from workflow import (apply_fixed, banner, cv_values, ef_record,  # noqa: E402
                      provenance, stage_dir, strip_calc)

DEFAULTS = {"delta": 0.001, "imag_cutoff_cm": 50.0, "atoms": "free",
            "min_reaction_overlap": 0.30, "stationary_fmax": 0.01,
            "mode_amplitude_A": 0.3, "mode_frames": 11}


def hessian_indices(cfg, natoms, spec):
    fixed = fixed_atoms(cfg, natoms)
    if spec == "all":
        return None, ("all atoms" + (" (NOTE: fixed atoms included)" if fixed else ""))
    if spec == "free":
        if not fixed:
            return None, "all atoms (no fixed atoms configured)"
        return [i for i in range(natoms) if i not in set(fixed)], \
            f"partial Hessian over {natoms - len(fixed)} free atoms (approximation)"
    if isinstance(spec, list):
        idx = sorted({index_of(cfg, r, natoms) for r in spec})
        return idx, f"partial Hessian over {len(idx)} selected atoms (approximation)"
    raise ValueError(f"hessian.atoms must be 'free', 'all' or a list, got {spec!r}")


def verdict(res, opts, h_index, fmax=0.0):
    base = _mode_verdict(res, opts, h_index)
    if fmax > float(opts["stationary_fmax"]):
        return (f"NOT A VERIFIED STATIONARY POINT (free-atom fmax {fmax:.2e} > "
                f"{opts['stationary_fmax']} eV/Å); Hessian indicative only — {base}")
    return base


def _mode_verdict(res, opts, h_index):
    n = res["n_imaginary"]
    ov = res["reaction_mode_overlap"]
    if n == 0:
        return "NOT A SADDLE: no significant imaginary mode (minimum-like)"
    if n > 1:
        return f"HIGHER-ORDER SADDLE: {n} significant imaginary modes"
    parts = []
    if ov is not None:
        parts.append(f"overlap with hydride-transfer coordinate {ov:.2f}")
    top = res["mode_atom_participation"][0]
    if h_index is not None:
        parts.append(f"H_transfer participation "
                     f"{next((p['fraction'] for p in res['mode_atom_participation'] if p['index'] == h_index), 0.0):.2f}")
    if ov is not None and ov < opts["min_reaction_overlap"]:
        return ("FIRST-ORDER SADDLE but its imaginary mode is NOT the configured hydride "
                f"transfer ({'; '.join(parts)}; largest atom {top['element']}{top['index']})")
    return ("FIRST-ORDER SADDLE CANDIDATE: exactly one significant imaginary mode"
            + (f" ({'; '.join(parts)})" if parts else ""))


def load_endpoints(cfg, tag) -> dict | None:
    """This model's relaxed endpoints from stage 1 (atoms + energies), or None."""
    ep = output_root(cfg, tag) / "endpoints"
    js = ep / "endpoints.json"
    if not js.exists():
        return None
    data = json.loads(js.read_text())["endpoints"]
    out = {}
    for role in ("reactant", "product"):
        xyz = ep / f"{role}_opt.xyz"
        if not xyz.exists():
            return None
        out[role] = {"atoms": read(str(xyz)), "energy_eV": data[role]["final"]["energy_eV"],
                     "converged": data[role]["converged"], "source": str(xyz)}
    return out


def run(cfg, model=None, variant=None, device=None, dtype=None, ts=None, out=None,
        handle=None, connectivity=None) -> dict:
    banner(cfg)
    opts = {**DEFAULTS, **section(cfg, "hessian")}
    copts = conn_options(cfg)
    if connectivity is not None:
        copts["enabled"] = bool(connectivity)
    tag = (handle.settings if handle else
           parse_calc_settings(cfg, model, variant, device, dtype)).tag
    ts = Path(ts) if ts else output_root(cfg, tag) / "ts" / "ts_sella.xyz"
    rep = validate_single(ts, cfg, f"TS ({ts})")
    require_valid(rep, "TS geometry")
    print_report(rep)
    at = read(str(ts))

    handle = handle or model_from_config(cfg, model, variant, device, dtype)
    handle.check_elements(at)
    out = stage_dir(cfg, tag, "vib", out)
    calc = handle.new_calc()
    at.calc = calc
    fmax_all = float(np.linalg.norm(at.get_forces(), axis=1).max())
    apply_fixed(at, cfg)            # stationarity is judged on the free atoms
    ef = ef_record(at)
    cvs = cv_values(cfg, at)
    idx, scope = hessian_indices(cfg, len(at), opts["atoms"])
    k = len(at) if idx is None else len(idx)

    ref_name = cfg.get("ts_mode_reference")
    ref_vec = None
    if ref_name:
        from reaction_coordinate import ReactionCoordinates

        rc = ReactionCoordinates.from_config(cfg, len(at), at.get_chemical_symbols(), [ref_name])
        ref_vec = rc.gradient(strip_calc(at), ref_name)
    try:
        h_index = index_of(cfg, "H_transfer", len(at))
    except Exception:
        h_index = None

    print(f"== Hessian ({tag}): {scope}; {6 * k} force calls, delta={opts['delta']} Å ==",
          flush=True)
    if ef["fmax_eV_A"] > float(opts["stationary_fmax"]):
        print(f"  WARNING: free-atom fmax = {ef['fmax_eV_A']:.3e} eV/Å — not a stationary "
              "point; frequencies are only indicative")
    t0 = time.time()
    step = max(1, (3 * k) // 10)
    prog = lambda i, n: print(f"    dof {i}/{n}", flush=True) if i % step == 0 else None
    H = hessian(strip_calc(at), calc, float(opts["delta"]), idx, prog)
    res = analyze_hessian(H, at, idx, ref_vec, float(opts["imag_cutoff_cm"]))
    np.save(out / "hessian.npy", H)
    h_idx = np.arange(len(at)) if idx is None else np.asarray(idx)
    nu_all, vecs_mw, w = normal_modes(H, at.get_masses()[h_idx])
    np.savez(out / "normal_modes.npz", frequencies_cm=nu_all, eigenvectors_mass_weighted=vecs_mw,
             inv_sqrt_mass_per_dof=w, hessian_atom_indices=h_idx,
             note="Cartesian displacement of mode k = eigenvectors_mass_weighted[:, k] * "
                  "inv_sqrt_mass_per_dof (then normalize)")
    if res["mode_displacement"] is not None:
        np.save(out / "imag_mode_vector.npy", np.array(res["mode_displacement"]))

    with open(out / "frequencies.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "frequency_cm", "significant_imaginary"])
        for i, nu in enumerate(res["frequencies_cm"]):
            w.writerow([i, f"{nu:.2f}", int(nu < -opts["imag_cutoff_cm"])])
    if res["mode_displacement"] is not None:
        disp = np.array(res["mode_displacement"])
        frames = []
        for s in np.linspace(-1, 1, int(opts["mode_frames"])):
            f = strip_calc(at)
            f.positions = f.positions + s * float(opts["mode_amplitude_A"]) * disp / np.abs(disp).max()
            frames.append(f)
        write(str(out / "imag_mode.xyz"), frames)

    v = verdict(res, opts, h_index, ef["fmax_eV_A"])
    stationary = ef["fmax_eV_A"] <= float(opts["stationary_fmax"])
    endpoints = load_endpoints(cfg, tag)

    # +/- imaginary-mode connectivity (sanity check, not an IRC)
    conn, conn_skip = None, None
    if not copts["enabled"]:
        conn_skip = "disabled"
    elif res["n_imaginary"] != 1 or res["mode_displacement"] is None:
        conn_skip = f"needs exactly one significant imaginary mode (found {res['n_imaginary']})"
    else:
        print(f"== +/- imaginary-mode connectivity ({copts['optimizer']}, fmax {copts['fmax']}, "
              f"step {copts['mode_displacement_A']} Å on the lead atom; NOT an IRC) ==", flush=True)
        conn = connectivity_check(strip_calc(at), res["mode_displacement"], handle.new_calc,
                                  cfg, endpoints, out / "connectivity", ef["energy_eV"])
        conn.pop("_atoms")
        for k, s in conn["sides"].items():
            print(f"  side {k}: {s['state']:13s} E={s['energy_eV']:.6f} eV converged="
                  f"{s['converged']}  C2-H*={s['hydride_state']['d_C2_H_A']:.3f} "
                  f"C1-H*={s['hydride_state']['d_C1_H_A']:.3f}")
        print(f"  {conn['verdict']}")

    label = ts_certification(stationary=stationary, fmax=ef["fmax_eV_A"],
                             stationary_fmax=float(opts["stationary_fmax"]),
                             n_imaginary=res["n_imaginary"],
                             overlap=res["reaction_mode_overlap"],
                             min_overlap=float(opts["min_reaction_overlap"]),
                             geometry_valid=rep["status"] != "FAIL", connectivity=conn)

    states = []
    if endpoints:
        states.append(reference_state(
            "relaxed_endpoint", endpoints["reactant"]["energy_eV"],
            endpoints["product"]["energy_eV"],
            {k: endpoints[k]["source"] for k in ("reactant", "product")}))
    else:
        states.append(reference_state("relaxed_endpoint",
                                      unavailable_reason="stage-1 endpoints.json not found"))
    e_rs, e_ps, why = side_minimum_reference(conn)
    if conn is None and conn_skip:
        why = f"connectivity check not run ({conn_skip})"
    states.append(reference_state(
        "connected_side_minimum", e_rs, e_ps,
        ({"reactant": conn["sides"][conn["reactant_side"]]["files"],
          "product": conn["sides"][conn["product_side"]]["files"]} if e_rs is not None else None),
        why))
    en = energetics(ef["energy_eV"], states, label, str(ts),
                    float((cfg.get("conformer") or {}).get("material_offset_kJ_mol", 4.184)))

    summary = {
        **provenance(cfg, handle, stage="validate_ts"),
        "structure": str(ts), "settings": opts, "hessian_scope": scope,
        "seconds": time.time() - t0,
        "energy_eV": ef["energy_eV"], "fmax_eV_A": ef["fmax_eV_A"],
        "fmax_all_atoms_incl_fixed_eV_A": fmax_all, "cvs": cvs,
        "mode_reference_cv": ref_name,
        **res,
        "verdict": v,
        "connectivity": conn, "connectivity_skipped": conn_skip,
        "ts_certification": label,
        "energetics": en,
        "files": {"hessian": str(out / "hessian.npy"), "frequencies": str(out / "frequencies.csv"),
                  "normal_modes": str(out / "normal_modes.npz"),
                  "imag_mode_vector": (str(out / "imag_mode_vector.npy")
                                       if res["mode_displacement"] else None),
                  "imag_mode": str(out / "imag_mode.xyz") if res["mode_displacement"] else None,
                  "connectivity_dir": str(out / "connectivity") if conn else None},
    }
    write_json(out / "validate_ts.json", summary)
    print(f"  significant imaginary modes: {res['n_imaginary']}  {res['imag_freqs_cm']}")
    print(f"  lowest 10 (cm^-1): {[round(x, 1) for x in res['lowest_10_cm']]}")
    if res["small_imaginary_below_cutoff_cm"]:
        print(f"  small imaginary below cutoff: "
              f"{[round(x, 1) for x in res['small_imaginary_below_cutoff_cm']]}")
    if res["mode_atom_participation"]:
        print("  dominant-mode participation: " + ", ".join(
            f"{p['element']}{p['index']}:{p['fraction']:.2f}" for p in res["mode_atom_participation"][:5]))
    print(format_table(en))
    print(f"  MODE VERDICT: {v}")
    print(f"  TS: {label}")
    print(f"  wrote {out}/validate_ts.json")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hessian / imaginary-mode validation of a TS")
    add_calc_args(ap)
    ap.add_argument("--ts", default=None, help="structure (default: ts/ts_sella.xyz)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-connectivity", action="store_true",
                    help="skip the +/- mode relaxation (the TS can then not be CERTIFIED)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    try:
        s = run(cfg, a.model, a.variant, a.device, a.dtype, a.ts, a.out,
                connectivity=False if a.no_connectivity else None)
    except (ValidationError, FileNotFoundError) as e:
        print(f"ABORTED: {e}")
        return 1
    return 0 if s["ts_certification"].startswith("CERTIFIED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
