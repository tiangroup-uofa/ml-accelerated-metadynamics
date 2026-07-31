#!/usr/bin/env python3
"""
calculators.py
==============
Factories that build the base potentials (GFN2-xTB, MACE-OFF23) as ASE
calculators, optionally wrapped in a post-training `CorrectedCalculator`.

`make_factory` returns a **zero-argument factory** that yields a *fresh* calculator
per call — NEB wants an independent calculator object per image — while the
expensive MACE model is loaded **once** and shared underneath. This is the single
place the correction is injected, so the NEB/MD code never changes when a
correction is added later.
"""
from __future__ import annotations
from corrections import CorrectedCalculator, Identity, ForceCorrection


def make_factory(method: str, correction: ForceCorrection | None = None,
                 dtype: str = "float32", mace_model: str = "small"):
    """Return `() -> ASE calculator` for method in {"xtb","mace"}."""
    method = method.lower()
    corr = correction or Identity()

    if method == "mace":
        from mace.calculators import mace_off
        base = mace_off(model=mace_model, device="cpu", default_dtype=dtype)  # once
        return lambda: CorrectedCalculator(base, corr)

    if method == "xtb":
        from xtb.ase.calculator import XTB
        # xtb-python calculators are cheap to instantiate; fresh per image.
        # Electronic-temperature smearing + extra SCF cycles make the SCF robust
        # on the stretched intermediate geometries an interpolated NEB band visits
        # (plain GFN2 fails to converge on some of them).
        return lambda: CorrectedCalculator(
            XTB(method="GFN2-xTB", electronic_temperature=1000.0,
                max_iterations=500, accuracy=1.0), corr)

    raise ValueError(f"unknown method: {method}")


PRETTY = {"xtb": "GFN2-xTB", "mace": "MACE-OFF23"}
