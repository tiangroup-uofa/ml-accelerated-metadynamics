#!/usr/bin/env python3
"""
region_geoms.py — Phase B dataset
=================================
Rattled reaction-path geometries for the barrier-region force-error analysis.
Rattles each concerted-scan point (../cscan/f_*.xyz) N_RATTLE times by a fixed RMS
displacement, so the forces are meaningful (neither potential at its own minimum)
and the whole reaction coordinate (forming C–C 3.3 → 1.5 Å) is sampled. Denser
than the corrections dataset (8 vs 3 rattles) for per-region statistics.

Written once with a fixed seed so MACE and xTB (evaluated in separate processes)
see identical structures. Output: results/region_geoms/g_XXXX.xyz.
"""
from __future__ import annotations
import os, glob
from pathlib import Path
import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
DA = HERE.parent
N_RATTLE = 8
AMP = 0.10          # Å RMS displacement (perpendicular distortion, uniform across path)


def main():
    out = HERE / "results" / "region_geoms"
    out.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(out / "*.xyz")):
        os.remove(f)
    rng = np.random.default_rng(1)
    k = 0
    for f in sorted(glob.glob(str(DA / "cscan" / "f_*.xyz"))):
        base = read(f)
        for _ in range(N_RATTLE):
            at = base.copy()
            d = rng.normal(0, 1, at.positions.shape)
            d *= AMP / np.sqrt((d ** 2).sum(1).mean())
            at.positions += d
            write(str(out / f"g_{k:04d}.xyz"), at); k += 1
    print(f"wrote {k} rattled configs ({N_RATTLE}/scan-point, AMP={AMP} Å) -> {out}")


if __name__ == "__main__":
    main()
