#!/usr/bin/env python3
"""
phase2_water.py — per-iteration water equilibrium sanity check.

Identical protocol to phaseC_water.py (same droplet, same MD settings, float32)
so the numbers are directly comparable to the verified Phase C references:
  raw MACE Rg 3.90 | Delta-A 3.82 | Delta-B 3.82 | xTB 3.77 (O-O peak 2.84/2.74/2.74/2.84)

Prints one JSON line (consumed by phase2_loop.py).
    /usr/bin/python3 phase2_water.py --delta phase2/iter00/delta.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diels_alder" / "neb"))

RUNS = HERE.parent / "large" / "runs"
BINS = np.linspace(2.2, 8.0, 60)
CTR = 0.5 * (BINS[1:] + BINS[:-1])


def oo_rg(atoms):
    sym = np.array(atoms.get_chemical_symbols())
    O = atoms.get_positions()[sym == "O"]
    rg = float(np.sqrt(((O - O.mean(0)) ** 2).sum(1).mean()))
    d = [np.linalg.norm(O[a] - O[b]) for a in range(len(O)) for b in range(a + 1, len(O))]
    return np.array(d), rg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True)
    a = ap.parse_args()

    import torch

    torch.set_num_threads(4)
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mace.calculators import mace_off

    from corrections import CorrectedCalculator, Delta
    from pairwise_delta import PairwiseDelta

    base = mace_off(model="small", device="cpu", default_dtype="float32")
    delta = PairwiseDelta.from_dict(json.loads(Path(a.delta).read_text()))
    calc = CorrectedCalculator(base, Delta(delta))

    at = read(str(RUNS / "n20" / "opt" / "xtbopt.xyz"))
    at.calc = calc
    MaxwellBoltzmannDistribution(at, temperature_K=300)
    dyn = Langevin(at, timestep=0.5 * units.fs, temperature_K=300,
                   friction=0.01 / units.fs)

    n, neq, every = 3000, 600, 20   # 1.5 ps total, 0.3 ps equilibration
    dd, rgs = [], []
    for i in range(n):
        dyn.run(1)
        if i >= neq and i % every == 0:
            d, rg = oo_rg(at)
            dd.append(d)
            rgs.append(rg)
    hist, _ = np.histogram(np.concatenate(dd), bins=BINS, density=True)
    rgs = np.array(rgs)
    pos = at.get_positions()
    exploded = bool(np.max(np.linalg.norm(pos - pos.mean(0), axis=1)) > 15.0)

    print(json.dumps({
        "rg_mean": float(rgs.mean()),
        "rg_std": float(rgs.std()),
        "oo_peak": float(CTR[hist.argmax()]),
        "stable": (not exploded) and bool(np.all(np.isfinite(pos))),
        "hist": hist.tolist(),
        "ctr": CTR.tolist(),
    }))


if __name__ == "__main__":
    main()
