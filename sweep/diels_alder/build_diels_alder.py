#!/usr/bin/env python3
"""
build_diels_alder.py
====================
Build reactant and product geometries for the classic Diels-Alder
cycloaddition (s-cis 1,3-butadiene + ethylene -> cyclohexene), following the
xtb reaction-path tutorial (https://xtb-docs.readthedocs.io/en/latest/path.html).

The RMSD-biased `xtb --path` driver requires the SAME atom ordering in the
start and end structures (it interpolates atom i -> atom i). We therefore build
BOTH with one fixed labelling:

    C1 C2 C3 C4  = the four butadiene carbons (C1,C4 terminal =CH2)
    C5 C6        = the two ethylene carbons
    new sigma bonds form  C1-C6  and  C4-C5
    product double bond is C2=C3

Reactant: planar s-cis butadiene in the z=0 plane with ethylene held ~2.2 A
above it, oriented for a suprafacial-suprafacial approach.
Product:  cyclohexene ring (C1-C2=C3-C4-C5-C6-C1) with the same atom order.

Both are only STARTING guesses; xtb --opt cleans them up (see run_path.py).
Outputs: reactant_raw.xyz, product_raw.xyz  (16 atoms each, C6H10).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

HERE = Path(__file__).resolve().parent
CC2 = 1.34    # C=C
CC1 = 1.47    # C-C (butadiene central / product single bonds)
CH = 1.08


def rot_z(v, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    x, y, z = v
    return np.array([x * c - y * s, x * s + y * c, z])


def add_sp2_H(C, carbon_neighbors, plane_normal=np.array([0, 0, 1.0]), n_h=1):
    """Place n_h H's completing trigonal-planar geometry around C.
    carbon_neighbors: list of neighbour-carbon positions."""
    dirs = [ (nb - C) / np.linalg.norm(nb - C) for nb in carbon_neighbors ]
    Hs = []
    if n_h == 1:
        # single H opposite the sum of the two C-neighbour directions
        d = -(dirs[0] + dirs[1]); d /= np.linalg.norm(d)
        Hs.append(C + CH * d)
    else:
        # CH2: rotate the single C-neighbour direction by +/-120 deg in-plane
        d = dirs[0]
        for ang in (+120.0, -120.0):
            Hs.append(C + CH * rot_z(d, ang))
    return Hs


def build_reactant() -> Atoms:
    # ---- s-cis butadiene in the z=0 plane -------------------------------
    C2 = np.array([0.0, 0.0, 0.0])
    C3 = np.array([CC1, 0.0, 0.0])
    C1 = C2 + CC2 * np.array([np.cos(np.radians(124)), np.sin(np.radians(124)), 0])
    C4 = C3 + CC2 * np.array([np.cos(np.radians(56)),  np.sin(np.radians(56)),  0])
    # ---- ethylene ~2.2 A above, C6 over C1 side, C5 over C4 side --------
    zc = 2.2
    mid = 0.5 * (C1 + C4); mid[2] = zc
    axis = (C4 - C1); axis = axis / np.linalg.norm(axis)
    C6 = mid - 0.5 * CC2 * axis
    C5 = mid + 0.5 * CC2 * axis

    order = [C1, C2, C3, C4, C5, C6]
    symbols = ["C"] * 6
    pos = [p.copy() for p in order]

    # hydrogens (butadiene plane normal = z; ethylene plane normal = z too)
    H = {}
    H["C1"] = add_sp2_H(C1, [C2], n_h=2)
    H["C2"] = add_sp2_H(C2, [C1, C3], n_h=1)
    H["C3"] = add_sp2_H(C3, [C2, C4], n_h=1)
    H["C4"] = add_sp2_H(C4, [C3], n_h=2)
    H["C5"] = add_sp2_H(C5, [C6], n_h=2)
    H["C6"] = add_sp2_H(C6, [C5], n_h=2)
    for key in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        for h in H[key]:
            symbols.append("H"); pos.append(h)
    return Atoms(symbols=symbols, positions=np.array(pos))


def build_product() -> Atoms:
    # cyclohexene ring, same atom order C1..C6; ring C1-C2=C3-C4-C5-C6-C1
    R = 1.45
    seq = [0, 1, 2, 3, 4, 5]          # ring position of C1..C6 = its index
    ring = {}
    for i, ci in enumerate(seq):
        ang = np.radians(60 * ci)
        ring[i] = np.array([R * np.cos(ang), R * np.sin(ang), 0.0])
    C1, C2, C3, C4, C5, C6 = (ring[i] for i in range(6))
    symbols = ["C"] * 6
    pos = [C1, C2, C3, C4, C5, C6]

    def ch2(C, nbrs):
        # sp3 CH2 in a ring: two H's out of the ring plane (+-z), bisecting
        d = -sum((nb - C) / np.linalg.norm(nb - C) for nb in nbrs)
        d = d / np.linalg.norm(d)
        out = np.array([0, 0, 1.0])
        h1 = C + CH * (0.6 * d + 0.8 * out)
        h2 = C + CH * (0.6 * d - 0.8 * out)
        return [h1, h2]

    Hs = []
    Hs += ch2(C1, [C2, C6])                       # sp3
    Hs += add_sp2_H(C2, [C1, C3], n_h=1)          # sp2 (C2=C3)
    Hs += add_sp2_H(C3, [C2, C4], n_h=1)          # sp2
    Hs += ch2(C4, [C3, C5])
    Hs += ch2(C5, [C4, C6])
    Hs += ch2(C6, [C5, C1])
    for h in Hs:
        symbols.append("H"); pos.append(h)
    return Atoms(symbols=symbols, positions=np.array(pos))


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    r = build_reactant()
    p = build_product()
    assert len(r) == len(p) == 16, (len(r), len(p))
    write(str(HERE / "reactant_raw.xyz"), r, comment="butadiene+ethylene reactant guess")
    write(str(HERE / "product_raw.xyz"), p, comment="cyclohexene product guess")
    print(f"reactant_raw.xyz / product_raw.xyz written ({len(r)} atoms each, C6H10)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
