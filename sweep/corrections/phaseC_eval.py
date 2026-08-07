#!/usr/bin/env python3
"""phaseC_eval.py — evaluate ONE potential on the Phase C geometry set (separate
processes; MACE and xtb-python both link OpenMP). Saves per-config energy+forces.
  micromamba run -n macemd python phaseC_eval.py --calc mace --out phaseC/mace.npz
  micromamba run -n macemd python phaseC_eval.py --calc xtb  --out phaseC/xtb.npz
"""
from __future__ import annotations
import argparse, os, glob
from pathlib import Path
import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", choices=["mace", "xtb"], required=True)
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
        calc = XTB(method="GFN2-xTB")
    files = sorted(glob.glob(str(HERE / "phaseC" / "geoms" / "*.xyz")))
    E, F = [], []
    for f in files:
        at = read(f); at.calc = calc
        E.append(float(at.get_potential_energy())); F.append(at.get_forces())
    np.savez(a.out, energies=np.array(E), forces=np.array(F, dtype=object),
             files=np.array(files))
    print(f"[{a.calc}] {len(files)} configs -> {a.out}")


if __name__ == "__main__":
    main()
