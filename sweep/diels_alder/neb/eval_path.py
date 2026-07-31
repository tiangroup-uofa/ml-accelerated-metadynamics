#!/usr/bin/env python3
"""
eval_path.py
============
Single-point energy + forces for ONE potential on a set of reaction-path
geometries. Run once per calculator (separate processes) so MACE and xtb-python
never share a process — they both link OpenMP and co-loading can crash / corrupt.

Used by analyze_neb.py to build the along-the-path MACE-vs-xTB comparison
(energy-difference profile, force RMSE / cosine). Default geometries: the
concerted relaxed-scan path (`../cscan/f_*.xyz`) — a dense, evenly spaced,
common reaction coordinate for both methods.

    micromamba run -n macemd python eval_path.py --calc mace --out results/eval_mace.npz
    micromamba run -n macemd python eval_path.py --calc xtb  --out results/eval_xtb.npz
"""
from __future__ import annotations
import argparse, os, sys, warnings, glob
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
DA = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", choices=["mace", "xtb"], required=True)
    ap.add_argument("--geoms", default=os.path.join(DA, "cscan", "f_*.xyz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)

    if a.calc == "mace":
        import torch; torch.set_num_threads(a.threads)
        from mace.calculators import mace_off
        calc = mace_off(model="small", device="cpu", default_dtype="float64")
    else:
        from xtb.ase.calculator import XTB
        calc = XTB(method="GFN2-xTB")          # default GFN2 (accurate single point)

    files = sorted(glob.glob(a.geoms))
    E, F, B1, B2 = [], [], [], []
    for f in files:
        at = read(f); at.calc = calc
        E.append(float(at.get_potential_energy()))     # eV
        F.append(at.get_forces())                      # eV/Å
        p = at.get_positions()
        B1.append(float(np.linalg.norm(p[0]-p[5])))    # C1–C6
        B2.append(float(np.linalg.norm(p[3]-p[4])))    # C4–C5
    np.savez(a.out, energies=np.array(E), forces=np.array(F),
             b1=np.array(B1), b2=np.array(B2),
             coord=0.5*(np.array(B1)+np.array(B2)), files=np.array(files))
    print(f"[{a.calc}] {len(files)} geoms -> {a.out}  "
          f"(E range {min(E):.3f}..{max(E):.3f} eV)")


if __name__ == "__main__":
    main()
