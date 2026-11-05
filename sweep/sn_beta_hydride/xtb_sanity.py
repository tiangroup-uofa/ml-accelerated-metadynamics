#!/usr/bin/env python3
"""
Cheapest meaningful GFN2-xTB structural sanity check (Phase 4). NOT a barrier calc.

Evaluates and relaxes the reconstructed reactant and product with the same
ChargeSpin plumbing as the protocol, and judges each relaxed structure against
its INPUT structure with `coordination_check` (relative rule, identical for all
methods). If either endpoint is STRUCTURALLY_INVALID, xTB is recorded as
`STRUCTURALLY INVALID FOR BARRIER COMPARISON` and no NEB/Sella is run.
If both are preserved the status is only `STRUCTURE PRESERVED - REVIEW BEFORE ANY PATH`.

    xtb python (conda-forge xtb-python + ase):
    python xtb_sanity.py --config config_tian_tt_test.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

import calculators as C
from config_io import load_config, resolve_path, write_json
from coordination_check import check_coordination

EXPECTED = {"reactant": "reactant-like", "product": "product-like"}


def run(cfg_path, fmax=0.02, steps=600, check_every=5, out_dir=None):
    cfg = load_config(cfg_path)
    handle = C.model_from_config(cfg, model="gfn2-xtb")
    out = Path(out_dir) if out_dir else resolve_path(cfg, cfg.get("output_dir", "outputs")) / "xtb_sanity"
    out.mkdir(parents=True, exist_ok=True)
    result = {"method": "gfn2-xtb", "calculator": handle.info(), "fmax_eV_A": fmax, "max_steps": steps,
              "purpose": "structural sanity only; no barrier is computed or reported", "roles": {}}
    t0 = time.time()
    for role in ("reactant", "product"):
        ref = read(resolve_path(cfg, cfg["structures"][role]))
        a = ref.copy()
        a.calc = handle.new_calc()
        e0, f0 = float(a.get_potential_energy()), float(np.abs(a.get_forces()).max())
        hist = []

        def watch(a=a, ref=ref, role=role, hist=hist):
            r = check_coordination(a, ref, cfg, expected_hydride_state=EXPECTED[role])
            hist.append({"step": len(hist) * check_every, "verdict": r["verdict"], "reasons": r["reasons"],
                         "lost": r["lost_from_shell"], "gained": r["gained_in_shell"]})

        opt = BFGS(a, logfile=str(out / f"{role}_bfgs.log"), trajectory=str(out / f"{role}_xtb.traj"))
        opt.attach(watch, interval=check_every)
        converged = bool(opt.run(fmax=fmax, steps=steps))
        final = check_coordination(a, ref, cfg, expected_hydride_state=EXPECTED[role])
        first_bad = next((h for h in hist if h["verdict"] == "STRUCTURALLY_INVALID"), None)
        write(out / f"{role}_xtb_relaxed.xyz", a)
        result["roles"][role] = {
            "E0_eV": e0, "fmax0_eV_A": f0, "E_relaxed_eV": float(a.get_potential_energy()),
            "fmax_final_eV_A": float(np.abs(a.get_forces()).max()), "converged": converged,
            "steps": int(opt.nsteps), "final": final,
            "first_invalid_during_relaxation": first_bad,
            "max_displacement_from_input_A": float(np.linalg.norm(a.positions - ref.positions, axis=1).max())}
        print(f"{role}: converged={converged} steps={opt.nsteps} verdict={final['verdict']} "
              f"{final['reasons']}", flush=True)
    bad = [r for r, v in result["roles"].items() if v["final"]["verdict"] == "STRUCTURALLY_INVALID"]
    transient = [r for r, v in result["roles"].items() if v["first_invalid_during_relaxation"] or not v["converged"]]
    result["status"] = ("STRUCTURALLY INVALID FOR BARRIER COMPARISON" if bad
                        else "STRUCTURE PRESERVED AT END BUT TRANSIENTLY INVALID OR UNCONVERGED - REVIEW" if transient
                        else "STRUCTURE PRESERVED - REVIEW BEFORE ANY PATH")
    result["transient_or_unconverged_roles"] = transient
    result["invalid_roles"] = bad
    result["total_calculator_calls"] = handle.total_calls
    result["wall_seconds"] = time.time() - t0
    write_json(out / "xtb_sanity.json", result)
    print("STATUS:", result["status"], flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_tian_tt_test.json")
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    run(a.config, a.fmax, a.steps, out_dir=a.out_dir)
