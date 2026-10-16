#!/usr/bin/env python3
"""
summarize_tian_polar.py
=======================
Provenance-backed summary of Tian's saved MACE-POLAR-1-S results for the
reconstructed glucose / Sn-beta cluster (tt-test, commit b97ef46), using ONLY
the raw energies and coordinates in ``glucose/structures/``. No calculator is
run and no reported number is typed in: every barrier / reaction energy below
is recomputed from saved total energies and labelled with its reference state.

What it does
  * checks the saved files against the SHA-256 values in
    input/tian_tt_test/provenance.json and Tian's glucose/structures/provenance.json
  * cross-checks each energy stored in an XYZ file against the JSON records
  * barriers / reaction energies for BOTH reference states
    (energetics.py: relaxed_endpoint, connected_side_minimum)
  * keeps the relaxed-scan maximum separate, explicitly as NOT a TS
  * hydride-state labels and reactive-core distances for R, TS, P and the
    two side minima (from the coordinates)
  * conformer diagnostics: each side minimum vs the relaxed input endpoint
  * the saddle's status from the saved evidence: Tian's Sella / Hessian
    records are reported, but this pipeline has NOT re-run the Hessian, so the
    TS is labelled "NOT INDEPENDENTLY CERTIFIED"

Outputs (default audit/tian_tt_test/): tian_polar_summary.json

    python summarize_tian_polar.py [--config config_tian_tt_test.json] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ase.io import read  # noqa: E402

from config_io import load_config, resolve_path, write_json  # noqa: E402
from conformer_diagnostics import compare  # noqa: E402
from energetics import EV2KJ, diff, energetics, format_table, reference_state  # noqa: E402
from hydride import classify_hydride_state, core_distances  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TAG = "mace-polar_polar-1-s"
# Values transcribed from the supervisor's handoff table (a screenshot, not a
# file): used ONLY to report agreement/disagreement, never as inputs.
TIAN_TABLE_POLAR = {"barrier_kJ_mol": 126.4, "dE_kJ_mol": -19.2, "nu_imag_cm": -923.0,
                    "ts_Sn_O_A": [2.11, 2.13], "force_calls": 771, "wall_min": 9.0,
                    "_source": "supervisor handoff table (screenshot), transcribed by hand"}


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _energy_in_file(atoms):
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return None


def summarize(cfg: dict) -> dict:
    prov_path = resolve_path(cfg, cfg["provenance"]["provenance_file"])
    prov = json.loads(prov_path.read_text())
    src = {k: REPO / v for k, v in prov["tian_polar_results"].items() if not k.startswith("_")}
    gdir = src["relax_json"].parent

    # ---- file integrity -------------------------------------------------- #
    integrity = {"dataset_files": {}, "tian_provenance_files": {}}
    for name, rec in prov["files"].items():
        h = sha256_lf(prov_path.parent / name)
        integrity["dataset_files"][name] = {"sha256_lf": h, "matches": h == rec["sha256_lf"]}
    tprov = json.loads((gdir / "provenance.json").read_text())
    for name, rec in tprov["files"].items():
        p = gdir / name
        h = sha256_lf(p) if p.exists() else None
        integrity["tian_provenance_files"][name] = {"sha256_lf": h, "matches": h == rec["sha256"]}
    integrity["all_match"] = all(v["matches"] for d in integrity.values() if isinstance(d, dict)
                                 for v in d.values())

    relax = json.loads(src["relax_json"].read_text())
    tsj = json.loads(src["ts_json"].read_text())
    S = {"reactant": read(str(gdir / f"reactant_{TAG}.xyz")),
         "product": read(str(gdir / f"product_{TAG}.xyz")),
         "ts": read(str(src["ts_xyz"])),
         "reactant_side_min": read(str(src["reactant_side_xyz"])),
         "product_side_min": read(str(src["product_side_xyz"]))}

    # ---- energies: JSON vs XYZ cross-check ------------------------------- #
    E = {"reactant": relax["reactant"]["energy_eV"], "product": relax["product"]["energy_eV"],
         "ts": tsj["ts"]["energy_eV"],
         "reactant_side_min": tsj["connectivity"]["reactant_side"]["energy_eV"],
         "product_side_min": tsj["connectivity"]["product_side"]["energy_eV"]}
    xcheck = {}
    for k, at in S.items():
        ef = _energy_in_file(at)
        xcheck[k] = {"json_eV": E[k], "xyz_eV": ef,
                     "abs_diff_eV": None if ef is None else abs(ef - E[k])}
    xcheck["all_consistent"] = all(v["abs_diff_eV"] is not None and v["abs_diff_eV"] < 1e-6
                                   for v in xcheck.values() if isinstance(v, dict))

    # ---- geometry from coordinates --------------------------------------- #
    geom = {k: {"hydride_state": classify_hydride_state(a, cfg),
                "core_distances_A": core_distances(a, cfg)} for k, a in S.items()}

    # ---- saddle status from saved records (not recomputed) --------------- #
    vib, ts = tsj["vib"], tsj["ts"]
    sides_ok = (geom["reactant_side_min"]["hydride_state"]["state"] == "reactant-like"
                and geom["product_side_min"]["hydride_state"]["state"] == "product-like")
    evidence = {
        "sella_converged": ts["sella_converged"], "fmax_eV_A": ts["fmax_eV_A"],
        "fmax_criterion_eV_A": ts["fmax_criterion"],
        "n_significant_imaginary": vib["n_imaginary"], "nu_imag_cm": vib["nu_imag_cm"],
        "lowest_10_cm": vib["lowest_10_cm"],
        "reaction_mode_overlap": vib["reaction_mode_overlap"],
        "H_transfer_participation": next((p["fraction"] for p in vib["mode_atom_participation"]
                                          if p["index"] == cfg["atom_map"]["H_transfer"]), None),
        "eigenvector_saved": "mode_displacement" in vib,
        "side_minima_states": [geom["reactant_side_min"]["hydride_state"]["state"],
                               geom["product_side_min"]["hydride_state"]["state"]],
        "side_minima_connect_reactant_and_product_like": sides_ok,
        "source": str(src["ts_json"].relative_to(REPO)),
    }
    ts_status = ("NOT INDEPENDENTLY CERTIFIED: Tian's saved records show a converged Sella "
                 f"stationary point (fmax {ts['fmax_eV_A']:.1e} eV/Å), "
                 f"{vib['n_imaginary']} significant imaginary mode ({vib['nu_imag_cm']:.1f} cm^-1, "
                 f"overlap {vib['reaction_mode_overlap']:.2f} with the hydride coordinate) and "
                 "+/- mode side minima that are reactant-/product-like; the Hessian has not been "
                 "recomputed by this pipeline and the eigenvector was not saved")

    states = [
        reference_state("relaxed_endpoint", E["reactant"], E["product"],
                        {"reactant": f"glucose/structures/reactant_{TAG}.xyz",
                         "product": f"glucose/structures/product_{TAG}.xyz",
                         "energies": str(src["relax_json"].relative_to(REPO))}),
        reference_state("connected_side_minimum", E["reactant_side_min"], E["product_side_min"],
                        {"reactant": str(src["reactant_side_xyz"].relative_to(REPO)),
                         "product": str(src["product_side_xyz"].relative_to(REPO)),
                         "energies": str(src["ts_json"].relative_to(REPO)),
                         "method": "Tian: TS +/- 0.1 Å along the imaginary mode, BFGS 0.02 eV/Å "
                                   "(NOT an IRC)"}),
    ]
    mat = float((cfg.get("conformer") or {}).get("material_offset_kJ_mol", 4.184))
    en = energetics(E["ts"], states, ts_status, f"glucose/structures/ts_{TAG}.xyz", mat)

    # scan maximum: reported separately and never as a TS
    rows = list(csv.DictReader(open(src["scan_csv"])))
    k_max = max(range(len(rows)), key=lambda k: float(rows[k]["energy_eV"]))
    scan = {"status": "relaxed 1-D scan maximum along delta_H: NOT a transition state "
                      "(constrained geometry, fmax 0.03 eV/Å under constraint)",
            "point": k_max, "delta_A": float(rows[k_max]["delta_A"]),
            "E_scan_max_minus_E_relaxed_reactant": diff(float(rows[k_max]["energy_eV"]),
                                                        E["reactant"]),
            "E_scan_max_minus_E_saddle": diff(float(rows[k_max]["energy_eV"]), E["ts"])}

    conf = {"reactant_side": compare(S["reactant"], S["reactant_side_min"], cfg, E["reactant"],
                                     E["reactant_side_min"],
                                     "reactant-side minimum vs relaxed input reactant"),
            "product_side": compare(S["product"], S["product_side_min"], cfg, E["product"],
                                    E["product_side_min"],
                                    "product-side minimum vs relaxed input product")}

    # agreement with Tian's own JSON numbers and with the handoff table
    agree = {
        "vs_tian_json": {
            "barrier_forward_relaxed_endpoint_kJ_mol": [
                en["barrier_forward"]["relaxed_endpoint"]["kJ_mol"], tsj["barrier_forward_kJ_mol"]],
            "reaction_energy_relaxed_endpoint_kJ_mol": [
                en["reaction_energy"]["relaxed_endpoint"]["kJ_mol"], tsj["dE_kJ_mol"]],
            "note": "Tian converts with 96.485 kJ/mol per eV; this summary uses CODATA "
                    "96.48533212 (difference < 0.001 kJ/mol here)"},
        "vs_handoff_table": {
            "table": TIAN_TABLE_POLAR,
            "table_barrier_matches_reference_state": _which_state(en, "barrier_forward",
                                                                  TIAN_TABLE_POLAR["barrier_kJ_mol"]),
            "table_dE_matches_reference_state": _which_state(en, "reaction_energy",
                                                             TIAN_TABLE_POLAR["dE_kJ_mol"]),
            "ts_Sn_O1_Sn_O2_from_coordinates_A": [geom["ts"]["core_distances_A"]["Sn-O1"],
                                                  geom["ts"]["core_distances_A"]["Sn-O2"]],
            "force_calls_saved": tsj["force_calls"], "wall_s_saved": tsj["wall_s"],
            "wall_note": "wall_s covers scan/Sella/Hessian/side relaxations only; endpoint "
                         f"relaxations add {relax['reactant']['wall_s'] + relax['product']['wall_s']:.0f} s"},
    }
    return {
        "what": "Tian's MACE-POLAR-1-S results on the RECONSTRUCTED 42-atom vacuum cluster, "
                "recomputed from saved raw energies (no calculation run)",
        "method": tsj["model"],
        "charge": tsj["charge"], "spin_multiplicity": tsj["spin_multiplicity"],
        "charge_spin_status": cfg["provenance"]["charge_spin_status"],
        "environment": "vacuum; electronic potential energies; no ZPE/thermal terms",
        "workflow_note": "Tian's path is a 15-point relaxed scan + Sella (internal coordinates, "
                         "fmax 1e-3 eV/Å), NOT a CI-NEB",
        "file_integrity": integrity,
        "energy_cross_check": xcheck,
        "energies_eV": E,
        "energetics": en,
        "scan_maximum": scan,
        "saddle_evidence_saved": evidence,
        "geometry": geom,
        "conformer_diagnostics": conf,
        "agreement": agree,
        "conversion_eV_to_kJ_mol": EV2KJ,
    }


def _which_state(en, key, value, tol=0.06):
    hits = [n for n, v in en[key].items() if v is not None and abs(v["kJ_mol"] - value) <= tol]
    return hits or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarize Tian's saved MACE-POLAR results")
    ap.add_argument("--config", default=str(HERE / "config_tian_tt_test.json"))
    ap.add_argument("--out", default=str(HERE / "audit" / "tian_tt_test"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    s = summarize(cfg)
    out = Path(a.out)
    write_json(out / "tian_polar_summary.json", s)
    print(f"file integrity all match: {s['file_integrity']['all_match']}; "
          f"XYZ/JSON energies consistent: {s['energy_cross_check']['all_consistent']}")
    print(format_table(s["energetics"]))
    sc = s["scan_maximum"]
    print(f"  scan maximum (NOT a TS): {sc['E_scan_max_minus_E_relaxed_reactant']['kJ_mol']:.2f} "
          f"kJ/mol vs relaxed reactant")
    for k, c in s["conformer_diagnostics"].items():
        print(f"  {k}: {c['interpretation']}")
    print(f"  TS: {s['energetics']['ts']['status']}")
    print(f"  wrote {out / 'tian_polar_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
