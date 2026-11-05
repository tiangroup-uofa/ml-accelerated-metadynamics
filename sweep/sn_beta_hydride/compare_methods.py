#!/usr/bin/env python3
"""
Apples-to-apples table for two methods from SAVED results (no model calls).

    python compare_methods.py --omol outputs/tian_tt_test/mace-omol_extra_large

POLAR side = Tian's SAVED run (audit/tian_tt_test/tian_polar_summary.json + the
structures in glucose/structures/); this script only reads saved files. A separate cheap
re-run of POLAR through our pipeline (Hessian + +/-mode connectivity) reproduced those numbers;
see OMOL_BENCHMARK_REPORT.md. Every POLAR cell here is labelled "Tian's run".
Differences in protocol are listed explicitly, not hidden:
  * Tian: 15-point relaxed scan along delta_H + Sella (internal), fmax 1e-3
  * ours: 13-image CI-NEB (not converged is flagged) + Sella + Hessian + +/-mode relax
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

from config_io import atom_map, load_config, write_json
from conformer_diagnostics import compare
from coordination_check import check_coordination

HERE = Path(__file__).resolve().parent
EV = 96.48533212331
STRUCT = HERE.parent.parent / "glucose" / "structures"


def core_dists(a, m):
    d = lambda x, y: float(np.linalg.norm(a.positions[m[x]] - a.positions[m[y]]))
    return {"C2-H*": d("C2", "H_transfer"), "C1-H*": d("C1", "H_transfer"), "Sn-O1": d("Sn", "O1"),
            "Sn-O2": d("Sn", "O2"), "Sn-O_water": d("Sn", "O_water")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omol", required=True)
    ap.add_argument("--config", default="config_tian_tt_test.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    m = atom_map(cfg, 42)
    O = Path(a.omol)
    R_in, P_in = read(HERE / "input/tian_tt_test/reactant.xyz"), read(HERE / "input/tian_tt_test/product.xyz")
    pol = json.load(open(HERE / "audit/tian_tt_test/tian_polar_summary.json"))
    ps = pol["saddle_evidence_saved"]
    om = json.load(open(O / "protocol_summary.json"))
    vt = json.load(open(O / "vib/validate_ts.json"))

    S = {  # name -> (atoms, input reference, expected hydride state)
        "OMOL": {"R": read(O / "endpoints/reactant_opt.xyz"), "P": read(O / "endpoints/product_opt.xyz"),
                 "TS": read(O / "ts/ts_sella.xyz"), "R'": read(O / "vib/connectivity/conn_plus_relaxed.xyz"),
                 "P'": read(O / "vib/connectivity/conn_minus_relaxed.xyz")},
        "POLAR (Tian)": {"R": read(STRUCT / "reactant_mace-polar_polar-1-s.xyz"), "P": read(STRUCT / "product_mace-polar_polar-1-s.xyz"),
                         "TS": read(STRUCT / "ts_mace-polar_polar-1-s.xyz"), "R'": read(STRUCT / "ts_reactant_side_mace-polar_polar-1-s.xyz"),
                         "P'": read(STRUCT / "ts_product_side_mace-polar_polar-1-s.xyz")}}
    ref_of = {"R": (R_in, "reactant-like"), "R'": (R_in, "reactant-like"), "P": (P_in, "product-like"),
              "P'": (P_in, "product-like"), "TS": (R_in, "shared")}

    out = {"structures": {}, "conformer_offsets": {}}
    for meth, d in S.items():
        out["structures"][meth] = {}
        for k, at in d.items():
            ref, exp = ref_of[k]
            cc = check_coordination(at, ref, cfg, expected_hydride_state=exp)
            out["structures"][meth][k] = {"coordination": cc["verdict"], "reasons": cc["reasons"],
                                          "hydride": cc["hydride"]["state"], "water_intact": cc["water"]["intact"],
                                          "n_shell_heavy": cc["n_shell_now"], "dists_A": core_dists(at, m),
                                          "lengthened_A": cc["lengthened_gt_threshold_A"]}
    o, p = S["OMOL"], S["POLAR (Tian)"]
    pairs = {"OMOL R vs input R": (R_in, o["R"]), "OMOL P vs input P": (P_in, o["P"]),
             "POLAR R vs input R": (R_in, p["R"]), "POLAR P vs input P": (P_in, p["P"]),
             "OMOL R vs POLAR R": (p["R"], o["R"]), "OMOL R vs POLAR R'": (p["R'"], o["R"]),
             "OMOL R' vs POLAR R'": (p["R'"], o["R'"]), "OMOL R' vs POLAR R": (p["R"], o["R'"]),
             "OMOL R' vs OMOL R": (o["R"], o["R'"]), "POLAR R' vs POLAR R": (p["R"], p["R'"]),
             "OMOL P vs POLAR P": (p["P"], o["P"]), "OMOL P' vs POLAR P'": (p["P'"], o["P'"]),
             "OMOL P' vs OMOL P": (o["P"], o["P'"]), "POLAR P' vs POLAR P": (p["P"], p["P'"]),
             "OMOL TS vs POLAR TS": (p["TS"], o["TS"])}
    for lab, (ref, oth) in pairs.items():
        c = compare(ref, oth, cfg, label=lab)
        out["conformer_offsets"][lab] = {"rmsd_whole_A": c["rmsd_whole_A"], "rmsd_core_A": c["rmsd_core_A"],
                                         "n_changed_dihedrals": c["n_changed_dihedrals"],
                                         "largest_dihedral": (c["changed_dihedrals"][0] if c["changed_dihedrals"] else None)}

    en = om["energetics"]
    E = lambda k, ref: en[k][ref]["kJ_mol"]
    pe = pol["energetics"]
    pE = lambda k, ref: pe[k][ref]["kJ_mol"]
    e_o = {n: en["reference_states"][n] for n in ("relaxed_endpoint", "connected_side_minimum")}
    local_R = (e_o["relaxed_endpoint"]["reactant_eV"] - e_o["connected_side_minimum"]["reactant_eV"]) * EV
    local_P = (e_o["relaxed_endpoint"]["product_eV"] - e_o["connected_side_minimum"]["product_eV"]) * EV
    pr = pol["energies_eV"]
    out["energetics_kJ_mol"] = {
        "reference_states_are_labelled": True,
        "OMOL": {"barrier_fwd": {r: E("barrier_forward", r) for r in e_o}, "barrier_rev": {r: E("barrier_reverse", r) for r in e_o},
                 "dE": {r: E("reaction_energy", r) for r in e_o},
                 "R_relaxed_minus_R_prime": local_R, "P_relaxed_minus_P_prime": local_P},
        "POLAR (Tian's run)": {"barrier_fwd": {r: pE("barrier_forward", r) for r in e_o}, "barrier_rev": {r: pE("barrier_reverse", r) for r in e_o},
                               "dE": {r: pE("reaction_energy", r) for r in e_o},
                               "R_relaxed_minus_R_prime": (pr["reactant"] - pr["reactant_side_min"]) * EV,
                               "P_relaxed_minus_P_prime": (pr["product"] - pr["product_side_min"]) * EV},
        "OMOL_minus_POLAR": {"barrier_fwd": {r: E("barrier_forward", r) - pE("barrier_forward", r) for r in e_o}},
    }
    out["ts"] = {"OMOL": {"nu_imag_cm": vt["nu_imag_cm"], "n_imaginary": vt["n_imaginary"],
                          "overlap": vt["reaction_mode_overlap"], "H_participation": vt["mode_atom_participation"][0]["fraction"],
                          "certification": om["ts_certification"], "sella_converged": om["stages"]["refine_ts"]["converged"],
                          "neb_converged": om["stages"]["run_path"]["converged"],
                          "force_calls": om["total_calculator_calls"], "wall_h": om["wall_seconds"] / 3600},
                 "POLAR (Tian's run)": {"nu_imag_cm": ps["nu_imag_cm"], "overlap": ps["reaction_mode_overlap"],
                                        "H_participation": ps["H_transfer_participation"], "sella_fmax": ps["fmax_eV_A"],
                                        "certification": pe["ts"]["status"]}}
    path = Path(a.out or O.parent / "comparison_omol_vs_polar.json")
    write_json(path, out)
    print(json.dumps(out["energetics_kJ_mol"], indent=1, default=float))
    for meth, d in out["structures"].items():
        for k, v in d.items():
            print(f"{meth:13s} {k:3s} {v['coordination']:20s} hydride={v['hydride']:13s} water={v['water_intact']} "
                  + " ".join(f"{n}={x:.2f}" for n, x in v["dists_A"].items()) + (f"  weak:{v['lengthened_A']}" if v["lengthened_A"] else ""))
    for lab, v in out["conformer_offsets"].items():
        ld = v["largest_dihedral"]
        print(f"{lab:24s} whole={v['rmsd_whole_A']:.3f} core={v['rmsd_core_A']:.3f} nTors>={30}: {v['n_changed_dihedrals']}"
              + (f"  max {ld['label']} {ld['change_deg']:+.0f}" if ld else ""))
    print("wrote", path)


if __name__ == "__main__":
    main()
