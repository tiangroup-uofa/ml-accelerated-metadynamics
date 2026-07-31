#!/usr/bin/env python3
"""
mace_scan.py
============
MACE-OFF23 relaxed **concerted scan** — the MACE analogue of xTB's built-in
relaxed scan (`../cscan_profile.csv`). At each forming C–C distance we pin the two
forming bonds (FixBondLengths) and relax everything else with MACE, recording the
MACE energy. This gives a *both-relaxed* energy-difference profile
ΔE(s) = E_MACE(s) − E_xTB(s) for Task 1 (the raw single-point-on-xTB-geometry
version over-counts, because MACE is not relaxed there), and independently
cross-checks the MACE barrier from the NEB.

Reuses the xTB scan geometries as starting points (same reaction-coordinate grid).
Output: results/mace_cscan_profile.csv (forming_CC_A, rel_E_kcal_mol).
"""
from __future__ import annotations
import os, sys, warnings, glob
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ase.io import read
from ase.optimize import BFGS
from ase.constraints import FixBondLengths
from calculators import make_factory

HERE = os.path.dirname(os.path.abspath(__file__))
DA = os.path.dirname(HERE)
EV2KCAL = 23.060548
FORMING = [[0, 5], [3, 4]]


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    make = make_factory("mace", dtype="float64")
    files = sorted(glob.glob(os.path.join(DA, "cscan", "f_*.xyz")))
    r, E = [], []
    for f in files:
        at = read(f); at.calc = make()
        at.set_constraint(FixBondLengths(FORMING))   # pin both forming bonds at r
        BFGS(at, logfile=None).run(fmax=0.03, steps=300)
        at.set_constraint()
        p = at.get_positions()
        r.append(0.5*(np.linalg.norm(p[0]-p[5]) + np.linalg.norm(p[3]-p[4])))
        E.append(float(at.get_potential_energy()))
    E = np.array(E); rel = (E - E[0]) * EV2KCAL
    out = os.path.join(HERE, "results", "mace_cscan_profile.csv")
    with open(out, "w") as fh:
        fh.write("forming_CC_A,rel_E_kcal_mol\n")
        for rr, ee in zip(r, rel):
            fh.write(f"{rr:.3f},{ee:.3f}\n")
    print(f"MACE relaxed scan: barrier {rel.max():.1f} kcal/mol at "
          f"{r[int(np.argmax(rel))]:.2f} Å, ΔE {rel[-1]:.1f} -> {out}")


if __name__ == "__main__":
    main()
