#!/usr/bin/env python3
"""
build_geoms.py
==============
Assemble the geometry set for fitting/validating post-training corrections
(MACE-OFF23 -> GFN2-xTB). We want configurations with MEANINGFUL forces (not
relaxed minima, where one method's force is ~0), from both target domains:

  water : frames sampled from the droplet MD trajectories (thermal forces)
  da    : rattled Diels-Alder reaction-path geometries (reactive forces)

Writes data/geoms/*.xyz (domain + index in the filename); a later step evaluates
MACE and xTB on this fixed set (separate processes -> no OpenMP clash).
"""
from __future__ import annotations
import os, glob
from pathlib import Path
import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent
OUT = HERE / "data" / "geoms"
RNG = np.random.default_rng(0)
WATER_PER_SIZE = 12          # frames per droplet trajectory
DA_RATTLE = 3                # rattled copies per concerted-scan point
DA_AMP = 0.10                # Å RMS displacement


def sample_water():
    k = 0
    for n in (20, 30, 50):
        trj = SWEEP / "large" / "runs" / f"n{n}" / "low" / "xtb.trj"
        frames = read(str(trj), index=":", format="xyz")
        idx = np.linspace(len(frames)//4, len(frames)-1, WATER_PER_SIZE).round().astype(int)
        for j in idx:
            at = frames[int(j)]; at.info["domain"] = "water"
            write(str(OUT / f"water_n{n}_{k:03d}.xyz"), at); k += 1
    return k


def rattled_da():
    k = 0
    for f in sorted(glob.glob(str(SWEEP / "diels_alder" / "cscan" / "f_*.xyz"))):
        base = read(f)
        for _ in range(DA_RATTLE):
            at = base.copy()
            d = RNG.normal(0, 1, at.positions.shape)
            d *= DA_AMP / np.sqrt((d**2).sum(1).mean())
            at.positions += d; at.info["domain"] = "da"
            write(str(OUT / f"da_{k:03d}.xyz"), at); k += 1
    return k


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(OUT / "*.xyz")):
        os.remove(f)
    nw = sample_water(); nd = rattled_da()
    print(f"wrote {nw} water + {nd} DA = {nw+nd} configs -> {OUT}")


if __name__ == "__main__":
    main()
