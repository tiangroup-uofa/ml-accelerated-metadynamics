#!/usr/bin/env python3
"""
build_clusters.py
=================
Build approximate starting geometries for small gas-phase water clusters
(H2O)_n, n = 1..4, and write each as an XYZ file. These are only STARTING
points -- xTB optimizes them before the metadynamics run, so the exact
geometry does not need to be perfect, only physically sensible (roughly
H-bonded, no atom clashes).

Motifs used:
  n=1  single water (experimental-ish geometry)
  n=2  linear H-bonded dimer
  n=3  cyclic trimer (ring of 3 H-bonds)
  n=4  cyclic tetramer (ring of 4 H-bonds)

Output: geometries/water_1.xyz ... water_4.xyz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

HERE = Path(__file__).resolve().parent
OUT = HERE / "geometries"

# Internal geometry of a single water (Angstrom): O at origin, two H in xz.
OH = 0.9584
HOH_DEG = 104.45
half = np.radians(HOH_DEG / 2.0)


def single_water(origin=(0.0, 0.0, 0.0), rot_z_deg=0.0) -> Atoms:
    """One water molecule, optionally translated and rotated about z."""
    o = np.array([0.0, 0.0, 0.0])
    h1 = np.array([OH * np.sin(half), 0.0, OH * np.cos(half)])
    h2 = np.array([-OH * np.sin(half), 0.0, OH * np.cos(half)])
    pos = np.array([o, h1, h2])
    th = np.radians(rot_z_deg)
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th),  np.cos(th), 0.0],
                   [0.0,         0.0,        1.0]])
    pos = pos @ Rz.T + np.asarray(origin)
    return Atoms("OH2", positions=pos)


def ring_cluster(n: int, radius: float) -> Atoms:
    """Place n waters on a ring of given radius, each oxygen pointing
    roughly toward the next -- a crude cyclic H-bonded starting motif."""
    atoms = Atoms()
    for k in range(n):
        ang = 360.0 * k / n
        x = radius * np.cos(np.radians(ang))
        y = radius * np.sin(np.radians(ang))
        # rotate each monomer so its H's point tangentially (toward neighbor)
        w = single_water(origin=(x, y, 0.0), rot_z_deg=ang + 90.0)
        atoms += w
    return atoms


def build(n: int) -> Atoms:
    if n == 1:
        return single_water()
    if n == 2:
        # linear H-bonded dimer: donor water H --- O acceptor, ~2.9 A O-O
        a = single_water(origin=(0.0, 0.0, 0.0))
        b = single_water(origin=(2.9, 0.0, 0.0), rot_z_deg=180.0)
        return a + b
    if n == 3:
        return ring_cluster(3, radius=1.6)
    if n == 4:
        return ring_cluster(4, radius=1.9)
    raise ValueError(n)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n in (1, 2, 3, 4):
        atoms = build(n)
        path = OUT / f"water_{n}.xyz"
        write(str(path), atoms, comment=f"(H2O){n} starting geometry, {len(atoms)} atoms")
        print(f"n={n}: {len(atoms)} atoms -> {path.name}")


if __name__ == "__main__":
    main()
