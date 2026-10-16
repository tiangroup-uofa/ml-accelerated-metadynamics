#!/usr/bin/env python3
"""
geometry.py
===========
Small, dependency-light geometry helpers shared by the Sn-beta benchmark:
PBC-aware distances, covalent bond graph, and Kabsch alignment.

``kabsch_rmsd`` is numerically identical to ``sweep/corrections/phase2_lib.py``
(checked in tests/); it is re-implemented here, rather than imported, because
importing phase2_lib pulls in the Diels-Alder correction stack as a side effect.
"""
from __future__ import annotations

import numpy as np
from ase.data import covalent_radii


def uses_pbc(atoms) -> bool:
    return bool(np.any(atoms.pbc)) and abs(atoms.cell.volume) > 1e-8


def distance_matrix(atoms) -> np.ndarray:
    """All-pairs distances (Å), minimum-image when the structure is periodic."""
    if uses_pbc(atoms):
        return atoms.get_all_distances(mic=True)
    p = atoms.get_positions()
    d = p[:, None, :] - p[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d))


def distance(atoms, i: int, j: int) -> float:
    return float(atoms.get_distance(i, j, mic=uses_pbc(atoms)))


def covalent_cutoffs(numbers, scale: float = 1.2) -> np.ndarray:
    r = covalent_radii[np.asarray(numbers)]
    return scale * (r[:, None] + r[None, :])


def bond_set(atoms, scale: float = 1.2, D=None) -> set[tuple[int, int]]:
    """Covalent-radius bond graph: i<j with d_ij < scale*(r_i + r_j)."""
    D = distance_matrix(atoms) if D is None else D
    C = covalent_cutoffs(atoms.numbers, scale)
    n = len(atoms)
    iu = np.triu_indices(n, 1)
    mask = D[iu] < C[iu]
    return {(int(i), int(j)) for i, j in zip(iu[0][mask], iu[1][mask])}


def fragments(n: int, bonds) -> list[list[int]]:
    """Connected components of the bond graph (union-find)."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in bonds:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def kabsch_rotation(A, B) -> np.ndarray:
    """Rotation R minimising |(A - <A>) R - (B - <B>)|."""
    A = np.asarray(A, float) - np.asarray(A, float).mean(0)
    B = np.asarray(B, float) - np.asarray(B, float).mean(0)
    V, S, Wt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    return V @ D @ Wt


def kabsch_rmsd(A, B) -> float:
    """RMSD after optimal translation + rotation (same as phase2_lib)."""
    A = np.asarray(A, float) - np.asarray(A, float).mean(0)
    B = np.asarray(B, float) - np.asarray(B, float).mean(0)
    R = kabsch_rotation(A, B)
    return float(np.sqrt(np.mean(np.sum((A @ R - B) ** 2, axis=1))))


def kabsch_align(mobile, target):
    """Return a copy of ``mobile`` rigidly moved onto ``target`` (non-periodic)."""
    P, Q = mobile.get_positions(), target.get_positions()
    R = kabsch_rotation(P, Q)
    out = mobile.copy()
    out.positions = (P - P.mean(0)) @ R + Q.mean(0)
    return out


def per_atom_displacement(a, b, align: bool = True) -> np.ndarray:
    """|r_i(a) - r_i(b)| per atom, after Kabsch alignment of a onto b when
    the system is non-periodic (and ``align``); MIC displacement otherwise."""
    if uses_pbc(a) or uses_pbc(b):
        from ase.geometry import find_mic

        d, _ = find_mic(a.get_positions() - b.get_positions(), a.cell, a.pbc)
        return np.linalg.norm(d, axis=1)
    if align:
        a = kabsch_align(a, b)
    return np.linalg.norm(a.get_positions() - b.get_positions(), axis=1)


def max_force(forces) -> float:
    return float(np.max(np.linalg.norm(np.asarray(forces), axis=1)))
