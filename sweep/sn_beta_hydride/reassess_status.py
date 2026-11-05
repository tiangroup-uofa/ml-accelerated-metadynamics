#!/usr/bin/env python3
"""
Re-derive a saved run's saddle status under the CORRECTED certification rules, with NO model calls.

    python reassess_status.py outputs/tian_tt_test/mace-omol_extra_large [more run dirs ...]

For each run directory (a protocol or ladder output tree) it reads the saved Hessian record
(vib/validate_ts.json), the saddle's own Sella record (ts/ts_sella.xyz metadata + ts/refine_ts.json),
the saved relaxed +/- side-minimum structures, and (if present) path/path.json. It then calls the
PRODUCTION functions (`connectivity.side_coordination`, `assess_sides`, `side_minimum_reference`,
`ts_certification`, `run_protocol.ts_provenance`, `energetics.energetics`) so the answer cannot drift
from the pipeline. Barrier numbers are recomputed from the saved energies and must equal the saved ones.

This is a reassessment of saved raw outputs; it is not a recalculation of any chemistry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase.io import read

import connectivity as CN
from config_io import load_config, resolve_path, write_json
from energetics import energetics, reference_state
from run_protocol import ts_provenance

HERE = Path(__file__).resolve().parent


def result_statuses(label: str, provenance: dict, reproduces_reference: bool | None = None) -> dict:
    """Evidence-based status words, derived from the rules rather than typed by hand.

    saddle    CERTIFIED / NOT CERTIFIED   -- the connectivity.ts_certification verdict (saddle only)
    neb_path  NEB_CONVERGED / UNCONVERGED_DIAGNOSTIC_ONLY / NO_NEB_RUN   -- never merged into `saddle`
    barriers  always PRELIMINARY: electronic-energy differences to a saddle on ONE model's surface
              (inferred charge/spin, reconstructed structure); agreement with DFT is BLOCKED ON DFT DATA
    reproduced REPRODUCED when the saved numbers of an earlier run were recomputed to <= 0.01 kJ/mol
    """
    ps = provenance["path_status"]
    return {
        "saddle": "CERTIFIED" if label.startswith("CERTIFIED") else "NOT CERTIFIED",
        "neb_path": ("NEB_CONVERGED" if ps.startswith("NEB_CONVERGED") else
                     "UNCONVERGED_DIAGNOSTIC_ONLY" if ps.startswith("NEB_UNCONVERGED") else "NO_NEB_RUN"),
        "barriers": "PRELIMINARY",
        "accuracy_vs_dft": "BLOCKED ON DFT DATA",
        "reproduced": ("REPRODUCED" if reproduces_reference else None),
    }


def reassess(run_dir, cfg) -> dict:
    d = Path(run_dir)
    vt = json.loads((d / "vib" / "validate_ts.json").read_text())
    rts = json.loads((d / "ts" / "refine_ts.json").read_text())
    ts = read(str(d / "ts" / "ts_sella.xyz"))
    raw = ts.info.get("sella_converged")
    sella = None if raw is None else bool(raw)
    inputs = {r: read(str(resolve_path(cfg, cfg["structures"][r]))) for r in ("reactant", "product")}

    sides = {}
    for k, s in vt["connectivity"]["sides"].items():
        relaxed = read(s["files"]["relaxed"])
        coord = CN.side_coordination(relaxed, s["state"], inputs, cfg)
        sides[k] = {**s, "coordination": coord,
                    "structurally_valid": coord is None or coord["verdict"] != "STRUCTURALLY_INVALID"}
    conn = {**vt["connectivity"], "sides": sides, **CN.assess_sides(sides, coordination_checked=True)}

    opts = vt["settings"]
    stationary = vt["fmax_eV_A"] <= float(opts["stationary_fmax"])
    label = CN.ts_certification(
        stationary=stationary, fmax=vt["fmax_eV_A"], stationary_fmax=float(opts["stationary_fmax"]),
        n_imaginary=vt["n_imaginary"], overlap=vt["reaction_mode_overlap"],
        min_overlap=float(opts["min_reaction_overlap"]), geometry_valid=True, connectivity=conn,
        sella_converged=sella)

    path_json = d / "path" / "path.json"
    if path_json.exists():
        pj = json.loads(path_json.read_text())
        neb = {"converged": pj["converged"], "neb_fmax_final": pj["neb_fmax_final"], "steps": pj["steps"],
               "fmax_target": pj["settings"]["fmax"]}
    else:
        neb = None
    prov = ts_provenance(neb, {"converged": rts["converged"]})

    old = vt["energetics"]
    e_rs, e_ps, why = CN.side_minimum_reference(conn)
    ep = old["reference_states"]["relaxed_endpoint"]
    states = [reference_state("relaxed_endpoint", ep["reactant_eV"], ep["product_eV"]),
              reference_state("connected_side_minimum", e_rs, e_ps, None, why)]
    en = energetics(vt["energy_eV"], states, label, str(d / "ts" / "ts_sella.xyz"))
    drift = {}
    for q in ("barrier_forward", "barrier_reverse", "reaction_energy"):
        for ref in ("relaxed_endpoint", "connected_side_minimum"):
            new_v, old_v = en[q][ref], old[q][ref]
            if new_v is not None and old_v is not None:
                drift[f"{q}/{ref}"] = abs(new_v["kJ_mol"] - old_v["kJ_mol"])
    reproduced = None
    if "polar" in d.name:                      # Tian's saved POLAR run is the only earlier reference we hold
        sv = json.loads((HERE / "audit" / "tian_tt_test" / "tian_polar_summary.json").read_text())["energetics"]
        diffs = [abs(en[q][r]["kJ_mol"] - sv[q][r]["kJ_mol"])
                 for q in ("barrier_forward", "barrier_reverse", "reaction_energy")
                 for r in ("relaxed_endpoint", "connected_side_minimum")]
        reproduced = max(diffs) <= 0.01
    return {
        "statuses": result_statuses(label, prov, reproduced),
        "run": str(d), "reassessed_with": "corrected rules; saved raw outputs; NO model calls",
        "ts_certification_old_label": vt["ts_certification"], "ts_certification_new_label": label,
        "sella_converged": sella, "sella_nsteps": rts.get("iterations_nsteps"),
        "stationary_fmax_eV_A": vt["fmax_eV_A"], "n_imaginary": vt["n_imaginary"],
        "nu_imag_cm": vt["nu_imag_cm"], "hydride_overlap": vt["reaction_mode_overlap"],
        "connects": conn["connects"], "invalid_sides": conn["invalid_sides"],
        "side_coordination": {k: (s["coordination"] or {}).get("verdict") for k, s in sides.items()},
        "ts_provenance": prov,
        "energetics_kJ_mol": {q: {r: (None if v is None else round(v["kJ_mol"], 2)) for r, v in en[q].items()}
                              for q in ("barrier_forward", "barrier_reverse", "reaction_energy")},
        "max_drift_vs_saved_kJ_mol": max(drift.values()) if drift else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--config", default=str(HERE / "config_tian_tt_test.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = [reassess(r, cfg) for r in a.runs]
    for r in res:
        print(json.dumps(r, indent=1, default=float))
    if a.out:
        write_json(a.out, res)


if __name__ == "__main__":
    main()
