#!/usr/bin/env python3
"""
phaseC_reaction.py — C4 reaction-path benchmark
===============================================
Relaxed concerted scan (constrain forming C–C, relax the rest under the corrected
forces) for raw MACE and each fitted Delta (A baseline, B barrier-enriched,
C generic control), giving barrier / reaction energy / TS / full profile. All three
Deltas are conservative, so the CorrectedCalculator energy is used directly.
xTB target = 6.7 / −57.6 (relaxed scan). Output: phaseC/reaction_*.csv + summary.
Run: micromamba run -n macemd python phaseC_reaction.py
"""
from __future__ import annotations
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from ase.io import read
from ase.optimize import FIRE
from ase.constraints import FixBondLengths

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diels_alder" / "neb"))
sys.path.insert(0, str(HERE))
from corrections import CorrectedCalculator, Identity, Delta
from pairwise_delta import PairwiseDelta
DA = HERE.parent / "diels_alder"
FORMING = [[0, 5], [3, 4]]
EV2KCAL = 23.060548


def scan(base, correction):
    files = sorted(glob.glob(str(DA / "cscan" / "f_*.xyz")))
    coord, E = [], []
    for f in files:
        at = read(f)
        at.calc = CorrectedCalculator(base, correction)
        at.set_constraint(FixBondLengths(FORMING))
        FIRE(at, logfile=None).run(fmax=0.03, steps=300)
        at.set_constraint()
        p = at.get_positions()
        coord.append(0.5*(np.linalg.norm(p[0]-p[5]) + np.linalg.norm(p[3]-p[4])))
        E.append(float(at.get_potential_energy()))
    coord = np.array(coord); rel = (np.array(E) - E[0]) * EV2KCAL
    i = int(np.argmax(rel))
    return coord, rel, dict(barrier=float(rel[i]), ts_forming=float(coord[i]),
                            reaction_energy=float(rel[-1]))


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    import torch; torch.set_num_threads(4)
    from mace.calculators import mace_off
    base = mace_off(model="small", device="cpu", default_dtype="float64")
    d = json.loads((HERE / "phaseC" / "fitted_deltas.json").read_text())
    models = {"raw_MACE": Identity(),
              "A_baseline": Delta(PairwiseDelta.from_dict(d["A_baseline"])),
              "B_barrier": Delta(PairwiseDelta.from_dict(d["B_barrier"])),
              "C_generic": Delta(PairwiseDelta.from_dict(d["C_generic"]))}
    summary = {"xtb_target": {"barrier": 6.7, "reaction_energy": -57.6, "ts_forming": 2.32}}
    print(f"{'model':12s} {'barrier':>8s} {'react_E':>8s} {'TS(Å)':>6s}  (xTB target 6.7/−57.6)")
    for name, corr in models.items():
        coord, rel, s = scan(base, corr)
        summary[name] = s
        import csv
        with open(HERE / "phaseC" / f"reaction_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["coord", "rel_kcal"])
            for c, e in zip(coord, rel): w.writerow([f"{c:.3f}", f"{e:.3f}"])
        print(f"{name:12s} {s['barrier']:8.1f} {s['reaction_energy']:8.1f} {s['ts_forming']:6.2f}")
    (HERE / "phaseC" / "reaction_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> phaseC/reaction_summary.json")


if __name__ == "__main__":
    main()
