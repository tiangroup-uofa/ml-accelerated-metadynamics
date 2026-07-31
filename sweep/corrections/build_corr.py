#!/usr/bin/env python3
"""
build_corr.py
=============
Reconstruct fitted corrections from fitted_corrections.json as CorrectedCalculator
factories. Shared by the Diels-Alder and water benchmarks so every evaluation uses
the *same* fitted models via the existing CorrectedCalculator API.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diels_alder" / "neb"))
sys.path.insert(0, str(HERE))
from corrections import (CorrectedCalculator, Identity, GlobalScale, Affine,
                         ElementScale, Delta)
from pairwise_delta import PairwiseDelta

CONSERVATIVE = {"baseline": True, "global": True, "affine": True,
                "element": False, "delta": True}


def load_corrections():
    d = json.loads((HERE / "fitted_corrections.json").read_text())
    return {
        "baseline": Identity(),
        "global": GlobalScale(alpha=d["global"]["alpha"]),
        "affine": Affine(alpha=d["affine"]["alpha"], e_shift=d["affine"]["e_shift"]),
        "element": ElementScale(scales=d["element"]["scales"]),
        "delta": Delta(PairwiseDelta.from_dict(d["delta"])),
    }


def corrected_calc(base, name, corrections):
    return CorrectedCalculator(base, corrections[name])
