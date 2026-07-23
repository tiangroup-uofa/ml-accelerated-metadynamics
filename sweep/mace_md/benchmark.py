#!/usr/bin/env python3
"""
benchmark.py
============
Diagnose MACE-OFF23 throughput. The MD run in compare_md.py clocked ~2817 ms/step
— suspiciously slow (the static benchmark was ~75 ms/step). This times force
evaluations under different precision / thread settings to find a usable config
before scaling MACE MD/metadynamics up.

Times `nevals` force evaluations on a droplet, perturbing the geometry each step
(so the calculator actually recomputes, as in real MD).

Run one config per process (clean thread pool):
    micromamba run -n macemd python benchmark.py --dtype float32 --threads 8 --n 20
"""
from __future__ import annotations
import argparse, os, time, warnings
warnings.filterwarnings("ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--nevals", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import numpy as np, torch
    torch.set_num_threads(a.threads)
    from ase.io import read
    from mace.calculators import mace_off

    HERE = os.path.dirname(os.path.abspath(__file__))
    trj = f"{HERE}/../large/runs/n{a.n}/opt/xtbopt.xyz"
    atoms = read(trj)
    atoms.calc = mace_off(model="small", device="cpu", default_dtype=a.dtype)
    rng = np.random.default_rng(0)

    for _ in range(a.warmup):
        atoms.positions += rng.normal(0, 1e-3, atoms.positions.shape)
        atoms.get_forces()
    t0 = time.time()
    for _ in range(a.nevals):
        atoms.positions += rng.normal(0, 1e-3, atoms.positions.shape)
        atoms.get_forces()
    ms = (time.time() - t0) / a.nevals * 1000
    print(f"  n={a.n} ({len(atoms)} atoms)  {a.dtype}  {a.threads} threads  "
          f"-> {ms:.0f} ms/step   ({1000/ms:.1f} steps/s)")

if __name__ == "__main__":
    main()
