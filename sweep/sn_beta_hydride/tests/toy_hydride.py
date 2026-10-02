#!/usr/bin/env python3
"""
toy_hydride.py — TEST-ONLY synthetic system for software checks.

    *** NOT the Sn-beta structure. NOT chemistry. Never report its numbers. ***

A degenerate intermolecular hydride exchange, CH3O(-) + H2C=O -> H2C=O + CH3O(-)
(charge -1, singlet), built from idealized bond lengths/angles purely so the
pipeline has a small C1 <- H <- C2 system to run on. The product is the mirror
image of the reactant (x -> -x) re-indexed so that every index keeps its element
and role: C1 (formaldehyde carbon, acceptor) gains the hydride, C2 (methoxide
carbon, donor) loses it. No Sn: the Sn-specific checks are exercised separately.

Structures are generated in memory and only ever written to git-ignored test
output directories, with "TEST-ONLY" in the file name and comment line.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

TEST_ONLY_TAG = "TEST-ONLY synthetic toy (CH3O- + H2CO hydride exchange) - not Sn-beta, not chemistry"

# atom order: 0 C1, 1 O1, 2 H1a, 3 H1b, 4 C2, 5 O2, 6 H2a, 7 H2b, 8 H*
SYMBOLS = ["C", "O", "H", "H", "C", "O", "H", "H", "H"]
ROLES = {"C1": 0, "O1": 1, "H_C1": 2, "H_C1b": 3, "C2": 4, "O2": 5, "H_transfer": 8}
# reactant index -> product index carrying the mirrored position
PERM = {0: 4, 4: 0, 1: 5, 5: 1, 2: 6, 6: 2, 3: 7, 7: 3, 8: 8}


def reactant() -> Atoms:
    c1 = np.array([-1.55, 0.0, 0.0])     # C1/C2 mirror-symmetric so the fixed
    c2 = np.array([1.55, 0.0, 0.0])      # atoms coincide in both endpoints
    # sp2 formaldehyde in the yz plane, approached by H* along +x
    o1 = c1 + [0.0, 0.0, 1.22]
    h1a = c1 + [0.0, 0.95, -0.55]
    h1b = c1 + [0.0, -0.95, -0.55]
    # sp3 methoxide; H* points at C1 along -x
    t = np.array([[-1.0, 0.0, 0.0],
                  [1 / 3, 0.9428, 0.0],
                  [1 / 3, -0.4714, 0.8165],
                  [1 / 3, -0.4714, -0.8165]])
    hs = c2 + 1.12 * t[0]
    h2a = c2 + 1.10 * t[1]
    o2 = c2 + 1.33 * t[2]
    h2b = c2 + 1.10 * t[3]
    pos = [c1, o1, h1a, h1b, c2, o2, h2a, h2b, hs]
    a = Atoms(SYMBOLS, positions=pos)
    a.info["comment"] = TEST_ONLY_TAG
    return a


def product() -> Atoms:
    r = reactant()
    mirrored = r.get_positions() * np.array([-1.0, 1.0, 1.0])
    pos = np.zeros_like(mirrored)
    for i, j in PERM.items():
        pos[j] = mirrored[i]
    p = Atoms(SYMBOLS, positions=pos)
    p.info["comment"] = TEST_ONLY_TAG
    return p


def toy_config(outdir) -> dict:
    """Write the TEST-ONLY endpoints + config into ``outdir``; return the config dict."""
    from ase.io import write

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write(str(outdir / "TEST-ONLY_toy_reactant.xyz"), reactant())
    write(str(outdir / "TEST-ONLY_toy_product.xyz"), product())
    cfg = {
        "_comment": TEST_ONLY_TAG,
        "system": {"label": "TEST-ONLY toy hydride exchange (not Sn-beta)", "test_only": True,
                   "charge": -1, "spin_multiplicity": 1, "expected_n_atoms": 9,
                   "require_elements": ["C", "H", "O"]},
        "structures": {"reactant": "TEST-ONLY_toy_reactant.xyz",
                       "product": "TEST-ONLY_toy_product.xyz"},
        "atom_map": {"C1": 0, "C2": 4, "H_transfer": 8, "H_C1": 2},
        "expected_elements": {"C1": "C", "C2": "C", "H_transfer": "H", "H_C1": "H"},
        # pin the two carbons: the free intermolecular complex otherwise drifts apart
        # during optimization (as the DA vdW complex did); also exercises FixAtoms
        "fixed_atoms": ["C1", "C2"],
        "endpoint_bonds": {
            "reactant": {"bonded": [["C2", "H_transfer"]], "not_bonded": [["C1", "H_transfer"]]},
            "product": {"bonded": [["C1", "H_transfer"]], "not_bonded": [["C2", "H_transfer"]]}},
        "expected_bond_changes": {"break": [["C2", "H_transfer"]], "form": [["C1", "H_transfer"]]},
        "cvs": {
            "d_C1_H": {"type": "distance", "atoms": ["C1", "H_transfer"]},
            "d_C2_H": {"type": "distance", "atoms": ["C2", "H_transfer"]},
            "delta_H": {"type": "distance_difference", "plus": ["C2", "H_transfer"],
                        "minus": ["C1", "H_transfer"]},
            "CN_C1_H": {"type": "coordination", "center": ["C1"], "group": {"element": "H"},
                        "r0": 1.5, "n": 6, "m": 12},
            "CN_C2_H": {"type": "coordination", "center": ["C2"], "group": {"element": "H"},
                        "r0": 1.5, "n": 6, "m": 12}},
        "ts_mode_reference": "delta_H",
        "calculator": {"model": "mace-polar", "device": "cpu", "dtype": "float64",
                       "models": {"mace-omol": {"variant": "extra_large"},
                                  "mace-polar": {"variant": "polar-1-s"}}},
        "optimize": {"optimizer": "BFGS", "fmax": 0.05, "steps": 60},
        "path": {"n_images": 5, "k": 0.5, "climb": True, "interpolation": "idpp",
                 "optimizer": "FIRE", "fmax": 0.1, "steps": 40},
        "sella": {"order": 1, "internal": False, "fmax": 0.01, "steps": 60},
        "hessian": {"delta": 0.001, "imag_cutoff_cm": 50.0, "atoms": "free"},
        "output_dir": "outputs",
    }
    (outdir / "TEST-ONLY_toy_config.json").write_text(json.dumps(cfg, indent=2))
    return cfg


# TEST-ONLY analytic potential for Hessian checks
class Harmonic(Calculator):
    """E = 1/2 k (r - r0)^2 for every listed pair."""
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, pairs, k, r0, **kw):
        super().__init__(**kw)
        self.pairs, self.k, self.r0 = pairs, k, r0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        p = atoms.get_positions()
        e, f = 0.0, np.zeros_like(p)
        for i, j in self.pairs:
            d = p[i] - p[j]
            r = np.linalg.norm(d)
            e += 0.5 * self.k * (r - self.r0) ** 2
            g = self.k * (r - self.r0) * d / r
            f[i] -= g
            f[j] += g
        self.results = {"energy": e, "free_energy": e, "forces": f}
