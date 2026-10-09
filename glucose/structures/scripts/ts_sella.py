#!/usr/bin/env python3
"""
ts_sella.py
===========
Vacuum barrier of the Sn-beta C2->C1 hydride transfer with a MACE foundation
model (default MACE-POLAR-1 polar-1-s), from the relax_polar.py endpoints.

1. relaxed scan along delta = d(C2-H*) - d(C1-H*) (FixInternals bond combo),
   reactant -> product                                  -> scan_<tag>.{xyz,csv}
2. Sella order-1 saddle search from the highest scan point -> ts_<tag>.xyz
3. finite-difference Hessian (vibrations.py): one imaginary mode, overlap
   with grad(delta)
4. TS connectivity: displace +/- along the imaginary mode and relax both sides

Energies are potential energies (no ZPE / thermal terms), so they are not
directly the paper's free-energy barriers.

    $XTB_ENV/bin/python ts_sella.py [--model mace-polar] [--variant polar-1-s]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.constraints import FixInternals
from ase.io import read, write
from ase.optimize import BFGS
from sella import Sella

# Keep archived structures untouched; all rerun files live in a separate workspace.
import os
import shutil

ARCHIVE = Path(__file__).resolve().parent.parent
HERE = Path(os.environ.get("GLUCOSE_WORKDIR", str(ARCHIVE / "runs" / "polar-1-s"))).resolve()
if HERE == ARCHIVE or ARCHIVE in HERE.parents and "runs" not in HERE.relative_to(ARCHIVE).parts:
    raise ValueError("Use a separate workspace, not the structure archive or scripts directory")
HERE.mkdir(parents=True, exist_ok=True)
for pattern in ("*.xyz", "*.json", "*.csv"):
    for source in ARCHIVE.glob(pattern):
        target = HERE / source.name
        if not target.exists():
            shutil.copy2(source, target)
from calculators import CalcSettings, build_model  # noqa: E402
from geometry import kabsch_rmsd  # noqa: E402
from vibrations import vibrational_analysis  # noqa: E402

KJ = 96.485  # eV -> kJ/mol


def unit(v):
    return v / np.linalg.norm(v)


def delta(atoms, m):
    return (atoms.get_distance(m["C2"], m["H_transfer"])
            - atoms.get_distance(m["C1"], m["H_transfer"]))


def delta_gradient(atoms, m):
    X = atoms.positions
    g = np.zeros_like(X)
    u2 = unit(X[m["H_transfer"]] - X[m["C2"]])
    u1 = unit(X[m["H_transfer"]] - X[m["C1"]])
    g[m["H_transfer"]] = u2 - u1
    g[m["C2"]] = -u2
    g[m["C1"]] = u1
    return g


def set_delta(atoms, m, target, tol=1e-5):
    """Move H* along C2->C1 until delta == target."""
    for _ in range(100):
        err = target - delta(atoms, m)
        if abs(err) < tol:
            return
        u = unit(atoms.positions[m["C1"]] - atoms.positions[m["C2"]])
        atoms.positions[m["H_transfer"]] += 0.5 * err * u
    raise RuntimeError(f"could not set delta to {target}")


def relaxed_scan(react, prod, calc, m, n_points, fmax, stem):
    d0, d1 = delta(react, m), delta(prod, m)
    targets = np.linspace(d0, d1, n_points)
    combo = [[m["C2"], m["H_transfer"], 1.0], [m["C1"], m["H_transfer"], -1.0]]
    a = react.copy()
    frames, rows = [], []
    for k, t in enumerate(targets):
        a.set_constraint()
        set_delta(a, m, t)
        a.set_constraint(FixInternals(bondcombos=[(delta(a, m), combo)]))
        a.calc = calc
        BFGS(a, logfile=str(HERE / f"{stem}.log")).run(fmax=fmax, steps=500)
        e = float(a.get_potential_energy())
        f = a.copy()
        f.set_constraint()
        f.info.update(delta=delta(a, m), scan_energy_eV=e)
        frames.append(f)
        rows.append((k, delta(a, m), a.get_distance(m["C2"], m["H_transfer"]),
                     a.get_distance(m["C1"], m["H_transfer"]), e))
        print(f"scan {k:2d}  delta {rows[-1][1]:+.3f}  E-E0 "
              f"{(e - rows[0][4]) * KJ:7.1f} kJ/mol", flush=True)
    write(HERE / f"{stem}.xyz", frames)
    with open(HERE / f"{stem}.csv", "w") as fh:
        fh.write("point,delta_A,d_C2_H_A,d_C1_H_A,energy_eV\n")
        for r in rows:
            fh.write(",".join(f"{x}" for x in r) + "\n")
    return frames


def load_scan(path):
    """Reuse a written scan; older files stored the energy as the calc energy."""
    frames = read(path, ":")
    for f in frames:
        if "scan_energy_eV" not in f.info:
            f.info["scan_energy_eV"] = float(f.get_potential_energy())
        f.calc = None
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mace-polar")
    ap.add_argument("--variant", default="polar-1-s")
    ap.add_argument("--scan-points", type=int, default=15)
    ap.add_argument("--scan-fmax", type=float, default=0.03)
    ap.add_argument("--ts-fmax", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reuse-scan", action="store_true",
                    help="read scan_<tag>.xyz instead of rescanning")
    args = ap.parse_args()

    meta = json.loads((HERE / "atom_map.json").read_text())
    m = meta["atom_map"]
    settings = CalcSettings(model=args.model, variant=args.variant, dtype="float64",
                            threads=args.threads)
    handle = build_model(settings, meta["charge"], meta["spin_multiplicity"])
    tag = settings.tag
    calc = handle.new_calc()
    react = read(HERE / f"reactant_{tag}.xyz")
    prod = read(HERE / f"product_{tag}.xyz")
    e_r = float(json.loads((HERE / f"relax_{tag}.json").read_text())["reactant"]["energy_eV"])
    e_p = float(json.loads((HERE / f"relax_{tag}.json").read_text())["product"]["energy_eV"])
    t0 = time.time()

    # 1. relaxed scan
    if args.reuse_scan:
        frames = load_scan(HERE / f"scan_{tag}.xyz")
    else:
        frames = relaxed_scan(react, prod, calc, m, args.scan_points, args.scan_fmax,
                              f"scan_{tag}")
    k_max = int(np.argmax([f.info["scan_energy_eV"] for f in frames]))
    print(f"scan max at point {k_max}, delta {frames[k_max].info['delta']:+.3f}", flush=True)

    # 2. Sella saddle search
    ts = frames[k_max].copy()
    ts.info = {}
    ts.calc = calc
    n0 = calc.n_calls
    opt = Sella(ts, order=1, internal=True, trajectory=str(HERE / f"ts_{tag}.traj"),
                logfile=str(HERE / f"ts_{tag}.log"))
    sella_ok = bool(opt.run(fmax=args.ts_fmax, steps=1000))
    e_ts = float(ts.get_potential_energy())
    fmax_ts = float(np.max(np.linalg.norm(ts.get_forces(), axis=1)))
    write(HERE / f"ts_{tag}.xyz", ts)
    print(f"Sella converged={sella_ok} steps={opt.nsteps} calls={calc.n_calls - n0} "
          f"fmax={fmax_ts:.2e}", flush=True)

    # 3. Hessian
    vib = vibrational_analysis(ts, calc, reference_vector=delta_gradient(ts, m))
    print(f"imag modes {vib['imag_freqs_cm']}  overlap {vib['reaction_mode_overlap']}",
          flush=True)

    # 4. connectivity: push off the saddle along the imaginary mode both ways
    sides = {}
    if vib["mode_displacement"] is not None:
        mode = np.array(vib["mode_displacement"])
        mode *= 0.1 / np.max(np.linalg.norm(mode, axis=1))      # 0.1 A on the lead atom
        for sign in (+1, -1):
            a = ts.copy()
            a.positions += sign * mode
            a.calc = calc
            BFGS(a, logfile=str(HERE / f"irc_side_{tag}.log")).run(fmax=0.02, steps=1500)
            e = float(a.get_potential_energy())
            dl = delta(a, m)
            ref, e_ref = (react, e_r) if dl < 0 else (prod, e_p)
            key = "reactant_side" if dl < 0 else "product_side"
            write(HERE / f"ts_{key}_{tag}.xyz", a)
            sides[key] = {"delta_A": dl, "energy_eV": e,
                          "E_minus_endpoint_kJ_mol": (e - e_ref) * KJ,
                          "rmsd_to_endpoint_A": float(kabsch_rmsd(a.positions, ref.positions))}
            print(key, sides[key], flush=True)

    out = {
        "model": handle.info(), "charge": meta["charge"],
        "spin_multiplicity": meta["spin_multiplicity"],
        "scan": {"n_points": args.scan_points, "fmax": args.scan_fmax, "max_point": k_max,
                 "barrier_scan_kJ_mol": (frames[k_max].info["scan_energy_eV"] - e_r) * KJ},
        "ts": {"sella_converged": sella_ok, "steps": opt.nsteps, "fmax_eV_A": fmax_ts,
               "fmax_criterion": args.ts_fmax, "energy_eV": e_ts,
               "delta_A": delta(ts, m),
               "d_C2_H_A": ts.get_distance(m["C2"], m["H_transfer"]),
               "d_C1_H_A": ts.get_distance(m["C1"], m["H_transfer"])},
        "vib": {k: vib[k] for k in ("n_imaginary", "imag_freqs_cm", "nu_imag_cm",
                                    "reaction_mode_overlap", "lowest_10_cm",
                                    "mode_atom_participation")},
        "connectivity": sides,
        "barrier_forward_kJ_mol": (e_ts - e_r) * KJ,
        "barrier_reverse_kJ_mol": (e_ts - e_p) * KJ,
        "dE_kJ_mol": (e_p - e_r) * KJ,
        "force_calls": calc.n_calls, "wall_s": round(time.time() - t0, 1),
        "note": "potential energies of hand-built local minima; not free energies",
    }
    (HERE / f"ts_{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"barrier {out['barrier_forward_kJ_mol']:.1f} kJ/mol (fwd), "
          f"{out['barrier_reverse_kJ_mol']:.1f} (rev), dE {out['dE_kJ_mol']:.1f}  "
          f"-> ts_{tag}.json")


if __name__ == "__main__":
    main()
