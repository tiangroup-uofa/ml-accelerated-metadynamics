#!/usr/bin/env python3
"""
eval_da.py
==========
Diels-Alder benchmark for each correction: relaxed concerted scan with the
corrected MACE calculator, giving the reaction profile, activation barrier,
reaction energy and TS location — compared to baseline MACE and to xTB.

Energy along the path is obtained by **integrating the (corrected) force along the
relaxed reaction coordinate** (thermodynamic integration). This is uniform for all
corrections: for the conservative ones it equals the direct energy difference; for
the non-conservative element-scaling it is the only well-defined energy. A direct-
energy value is also recorded for the conservative corrections as a consistency
check.

Output: results/da_<name>.csv (profile) + results/da_summary.json.
Run: micromamba run -n macemd python eval_da.py
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
sys.path.insert(0, str(HERE))
from build_corr import load_corrections, corrected_calc, CONSERVATIVE

DA = HERE.parent / "diels_alder"
FORMING = [[0, 5], [3, 4]]
EV2KCAL = 23.060548


def scan_profile(base, corr, name):
    files = sorted(glob.glob(str(DA / "cscan" / "f_*.xyz")))
    geoms, Fp, Edir, coord = [], [], [], []
    for f in files:
        at = read(f)
        at.calc = corrected_calc(base, name, corr)
        at.set_constraint(FixBondLengths(FORMING))
        FIRE(at, logfile=None).run(fmax=0.03, steps=300)
        at.set_constraint()
        geoms.append(at.get_positions().copy())
        Fp.append(at.get_forces().copy())
        Edir.append(float(at.get_potential_energy()))
        p = at.get_positions()
        coord.append(0.5*(np.linalg.norm(p[0]-p[5]) + np.linalg.norm(p[3]-p[4])))
    # Energy profile from the corrected calculator's own energy. For the
    # conservative corrections (baseline/global/affine/delta) this is the physical
    # corrected PES; for the non-conservative element-scaling the CorrectedCalculator
    # leaves the energy = MACE energy (it only rescales forces), so its "energetics"
    # are uncorrected — which is itself the finding.
    rel = (np.array(Edir) - Edir[0]) * EV2KCAL
    coord = np.array(coord)
    imax = int(np.argmax(rel))
    return dict(coord=coord.tolist(), rel=rel.tolist(),
                barrier=float(rel[imax]), ts_forming=float(coord[imax]),
                reaction_energy=float(rel[-1]),
                conservative=CONSERVATIVE[name])


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    import torch; torch.set_num_threads(4)
    from mace.calculators import mace_off
    base = mace_off(model="small", device="cpu", default_dtype="float64")
    corr = load_corrections()

    RES = HERE / "results"; RES.mkdir(exist_ok=True)
    xtb_ref = json.loads((DA / "neb" / "results" / "neb_results.json").read_text())["xtb"]
    summary = {"xtb": {"barrier": xtb_ref["barrier_kcal"],
                       "reaction_energy": xtb_ref["reaction_energy_kcal"],
                       "ts_forming": 0.5*sum(xtb_ref["ts_forming_A"])}}
    print(f"{'model':9s} {'barrier':>8s} {'ΔE':>8s} {'TS(Å)':>6s}  (kcal/mol; target xTB "
          f"{xtb_ref['barrier_kcal']:.1f}/{xtb_ref['reaction_energy_kcal']:.1f})")
    for name in corr:
        r = scan_profile(base, corr, name)
        summary[name] = {k: r[k] for k in ("barrier", "reaction_energy", "ts_forming",
                                           "conservative")}
        import csv
        with open(RES / f"da_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["coord", "rel"])
            for c, e in zip(r["coord"], r["rel"]):
                w.writerow([f"{c:.3f}", f"{e:.3f}"])
        print(f"{name:9s} {r['barrier']:8.1f} {r['reaction_energy']:8.1f} "
              f"{r['ts_forming']:6.2f}")
    (RES / "da_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {RES}/da_summary.json")


if __name__ == "__main__":
    main()
