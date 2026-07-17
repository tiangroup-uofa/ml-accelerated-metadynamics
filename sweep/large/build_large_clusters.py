#!/usr/bin/env python3
"""
build_large_clusters.py
=======================
Build starting geometries for LARGER gas-phase (vacuum) water clusters
(H2O)_n, n = 20, 30, 50 -- the step up from the earlier n = 1..8 work.

Unlike prepare_box.py (which writes a PERIODIC Turbomole coord), these are
free clusters in vacuum: NO cell, NO $periodic. That is deliberate -- the
question here (July-10 note, step 1/3) is whether a finite cluster left free
in vacuum under a gentle NVT thermostat holds together as a DROPLET or keeps
evaporating the way the tiny n=2..4 clusters did.

Recipe: place n waters on a compact cubic grid (random orientation per
molecule, fixed seed) with an oxygen-oxygen spacing a little beyond a real
H-bond, so the blob starts nearly condensed but with no atom clashes. xTB
then optimizes it (GFN2) before the metadynamics run, so the start only has
to be physically sensible, not perfect.

Output: geometries/water_<n>.xyz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

HERE = Path(__file__).resolve().parent
OUT = HERE / "geometries"

OH = 0.9584
HOH = np.radians(104.45 / 2.0)


def single_water(origin, rng) -> Atoms:
    """One water at origin with a random orientation (seeded rng)."""
    o = np.array([0.0, 0.0, 0.0])
    h1 = np.array([OH * np.sin(HOH), 0.0, OH * np.cos(HOH)])
    h2 = np.array([-OH * np.sin(HOH), 0.0, OH * np.cos(HOH)])
    pos = np.array([o, h1, h2])
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1 - 2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1 - 2*(x*x+y*y)],
    ])
    pos = pos @ R.T + np.asarray(origin)
    return Atoms("OH2", positions=pos)


def build_blob(n: int, spacing: float = 3.4, seed: int = 0) -> Atoms:
    """Compact cubic blob of n waters, O-O grid spacing ~`spacing` A.

    We fill the smallest cube that holds n grid points and keep the first n,
    then centre the blob on the origin. `spacing` a bit above the ~2.8 A
    H-bond length keeps the start clash-free while still nearly condensed.
    """
    rng = np.random.default_rng(seed)
    m = int(np.ceil(n ** (1.0 / 3.0)))
    origins = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if len(origins) < n:
                    origins.append(np.array([i, j, k], float) * spacing)
    origins = np.array(origins)
    origins -= origins.mean(axis=0)          # centre the droplet
    atoms = Atoms()
    for c in origins:
        atoms += single_water(c, rng)
    return atoms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[20, 30, 50])
    ap.add_argument("--spacing", type=float, default=3.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for n in args.sizes:
        atoms = build_blob(n, spacing=args.spacing, seed=args.seed)
        path = OUT / f"water_{n}.xyz"
        write(str(path), atoms,
              comment=f"(H2O){n} vacuum blob start, {len(atoms)} atoms, "
                      f"spacing={args.spacing} A seed={args.seed}")
        # report the tightest O-O in the start (clash check)
        pos = atoms.get_positions()
        o = pos[0::3]
        dmin = np.min([np.linalg.norm(o[a] - o[b])
                       for a in range(len(o)) for b in range(a + 1, len(o))])
        print(f"n={n}: {len(atoms)} atoms  min O-O start = {dmin:.2f} A  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
