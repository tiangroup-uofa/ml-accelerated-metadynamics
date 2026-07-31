#!/usr/bin/env python3
"""
eval_ref.py
===========
Evaluate ONE potential (MACE-OFF23 or GFN2-xTB) on the dataset geometries,
saving per-config energy + forces. Run once per calculator in SEPARATE processes
(MACE and xtb-python both link OpenMP; co-loading can crash).

    micromamba run -n macemd python eval_ref.py --calc mace --out data/mace.npz
    micromamba run -n macemd python eval_ref.py --calc xtb  --out data/xtb.npz
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

    files = sorted(glob.glob(str(HERE / "data" / "geoms" / "*.xyz")))
    E, F, S, D = [], [], [], []
    for f in files:
        at = read(f); at.calc = calc
        E.append(float(at.get_potential_energy()))
        F.append(at.get_forces().astype(np.float64))
        S.append(np.array(at.get_chemical_symbols()))
        D.append(at.info.get("domain", os.path.basename(f).split("_")[0]))
    np.savez(a.out, energies=np.array(E),
             forces=np.array(F, dtype=object),
             symbols=np.array(S, dtype=object),
             domain=np.array(D), files=np.array(files))
    print(f"[{a.calc}] {len(files)} configs -> {a.out}")


if __name__ == "__main__":
    main()
