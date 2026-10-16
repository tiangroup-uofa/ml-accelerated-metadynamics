#!/usr/bin/env python3
"""
hydride.py
==========
Mapped-atom descriptors of the C2 -> C1 hydride transfer, shared by structure
validation, the +/- imaginary-mode connectivity check and result summaries.

* ``classify_hydride_state`` labels a geometry from the *mapped* hydride
  coordinate only: H* bonded to C2 and not C1 -> ``reactant-like``; bonded to
  C1 and not C2 -> ``product-like``; otherwise ``shared`` when both C-H*
  distances are below ``shared_cutoff_A`` (1.6 Å: H* in flight between C2 and
  C1, as at a hydride-transfer saddle, e.g. 1.31 / 1.35 Å) and ``detached``
  when it is not. "Bonded" is the same covalent criterion as the validation
  layer (d < bond_scale x (r_C + r_H), 1.284 Å at the default 1.2).
* ``core_distances`` reports the reactive-core distances for whichever
  atom-map names are configured (Sn-O only when Sn is mapped).

No energy or model enters here: these are geometric labels, not evidence that
a structure is a minimum.
"""
from __future__ import annotations

from ase.data import covalent_radii, atomic_numbers

from config_io import atom_map
from geometry import distance

STATES = ("reactant-like", "product-like", "shared", "detached")
SHARED_CUTOFF_A = 1.6

# (label, atom-map name a, atom-map name b)
CORE_PAIRS = (
    ("C2-H*", "C2", "H_transfer"),
    ("C1-H*", "C1", "H_transfer"),
    ("C1-O1", "C1", "O1"),
    ("C2-O2", "C2", "O2"),
    ("C1-C2", "C1", "C2"),
    ("Sn-O1", "Sn", "O1"),
    ("Sn-O2", "Sn", "O2"),
    ("Sn-O_water", "Sn", "O_water"),
    ("C1-H_C1", "C1", "H_C1"),
)


def ch_bond_cutoff(bond_scale: float = 1.2) -> float:
    return bond_scale * (covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["H"]])


def classify_hydride_state(atoms, cfg: dict, bond_scale: float | None = None,
                           shared_cutoff_A: float = SHARED_CUTOFF_A) -> dict:
    """Label the hydride position from the mapped C1, C2 and H_transfer atoms."""
    if bond_scale is None:
        bond_scale = float((cfg.get("validation") or {}).get("bond_scale", 1.2))
    m = atom_map(cfg, len(atoms), require=["C1", "C2", "H_transfer"])
    d2 = distance(atoms, m["C2"], m["H_transfer"])
    d1 = distance(atoms, m["C1"], m["H_transfer"])
    cut = ch_bond_cutoff(bond_scale)
    on2, on1 = d2 < cut, d1 < cut
    if on2 and not on1:
        state = "reactant-like"
    elif on1 and not on2:
        state = "product-like"
    elif d1 < shared_cutoff_A and d2 < shared_cutoff_A:
        state = "shared"
    else:
        state = "detached"
    return {"state": state, "d_C2_H_A": d2, "d_C1_H_A": d1, "delta_H_A": d2 - d1,
            "bond_cutoff_A": cut, "shared_cutoff_A": shared_cutoff_A,
            "criterion": "covalent C-H* bond (d < bond_scale x (r_C + r_H)) on the mapped atoms"}


def core_distances(atoms, cfg: dict) -> dict:
    """Reactive-core distances (Å) for every pair whose atom-map names are set."""
    raw = cfg.get("atom_map") or {}
    names = [n for n, v in raw.items() if isinstance(v, int) and not isinstance(v, bool)]
    m = atom_map(cfg, len(atoms), require=names)
    return {lab: round(distance(atoms, m[a], m[b]), 4)
            for lab, a, b in CORE_PAIRS if a in m and b in m}


def core_indices(cfg: dict, natoms: int) -> list[int]:
    """Reactive-core atoms for RMSD diagnostics: config['conformer']['core'] if
    given (names or indices), else every configured atom-map entry."""
    from config_io import index_of

    spec = (cfg.get("conformer") or {}).get("core")
    if spec:
        return sorted({index_of(cfg, r, natoms) for r in spec})
    raw = cfg.get("atom_map") or {}
    names = [n for n, v in raw.items() if isinstance(v, int) and not isinstance(v, bool)]
    return sorted(atom_map(cfg, natoms, require=names).values())
