#!/usr/bin/env python3
"""
workflow.py
===========
Helpers shared by the stage scripts (optimize_endpoints -> run_path -> refine_ts
-> validate_ts): output layout, fixed-atom constraints, energy/force records,
optimizer selection, CV evaluation and TEST-ONLY provenance tagging.

Output layout (per model, so OMOL and POLAR never overwrite each other):

    <output_dir>/<model>_<variant>/
        validation/   endpoints/   path/   ts/   vib/
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from config_io import fixed_atoms, output_root
from geometry import max_force

EV2KCAL = 23.060548
STAGES = ("validation", "endpoints", "path", "ts", "vib")
TEST_ONLY_BANNER = ("*** TEST-ONLY SYNTHETIC SYSTEM — software check, NOT the Sn-beta "
                    "structure; numbers below are not chemistry ***")


def is_test_only(cfg: dict) -> bool:
    return bool((cfg.get("system") or {}).get("test_only"))


def banner(cfg: dict) -> None:
    if is_test_only(cfg):
        print(TEST_ONLY_BANNER)


def stage_dir(cfg: dict, tag: str, stage: str, override=None) -> Path:
    d = Path(override) if override else output_root(cfg, tag) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def provenance(cfg: dict, handle=None, **extra) -> dict:
    out = {
        "TEST_ONLY": is_test_only(cfg),
        "system": (cfg.get("system") or {}).get("label"),
        "config": cfg.get("_config_path"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if is_test_only(cfg):
        out["warning"] = TEST_ONLY_BANNER
    if handle is not None:
        out["calculator"] = handle.info()
    out.update(extra)
    return out


def apply_fixed(atoms, cfg: dict):
    """Attach FixAtoms for config['fixed_atoms'] (no-op when empty)."""
    from ase.constraints import FixAtoms

    idx = fixed_atoms(cfg, len(atoms))
    atoms.set_constraint(FixAtoms(indices=idx) if idx else None)
    return idx


def free_indices(cfg: dict, natoms: int) -> list[int]:
    fx = set(fixed_atoms(cfg, natoms))
    return [i for i in range(natoms) if i not in fx]


def ef_record(atoms) -> dict:
    """Energy and force summary using the *constrained* forces the optimizer sees."""
    e = float(atoms.get_potential_energy())
    f = atoms.get_forces()
    k = int(np.argmax(np.linalg.norm(f, axis=1)))
    return {"energy_eV": e, "fmax_eV_A": max_force(f),
            "fnorm_eV_A": float(np.linalg.norm(f)), "fmax_atom": k}


OPTIMIZERS = ("BFGS", "LBFGS", "FIRE")


def make_optimizer(name: str, atoms, trajectory=None, logfile=None, **kw):
    from ase.optimize import BFGS, FIRE, LBFGS

    table = {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE}
    if name not in table:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {name!r}")
    return table[name](atoms, trajectory=str(trajectory) if trajectory else None,
                       logfile=str(logfile) if logfile else None, **kw)


def cv_values(cfg: dict, atoms) -> dict:
    from reaction_coordinate import ReactionCoordinates

    rc = ReactionCoordinates.from_config(cfg, len(atoms), atoms.get_chemical_symbols())
    return {k: float(v) for k, v in rc.compute(atoms).items()}


def strip_calc(atoms):
    """Copy without calculator/constraints for writing clean XYZ files."""
    a = atoms.copy()
    a.calc = None
    a.set_constraint()
    return a
