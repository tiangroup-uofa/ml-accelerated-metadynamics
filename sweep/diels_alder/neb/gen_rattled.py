#!/usr/bin/env python3
"""
gen_rattled.py
==============
Write rattled reaction-path geometries for a *fair* force comparison (Task 2).

On a relaxed path one method's forces are ~zero (it sits at its own minimum), so
comparing forces there is confounded. Displacing each reaction-path structure by
a small random amount puts BOTH potentials off-equilibrium, giving meaningful,
comparable force vectors. The geometries are written once (fixed seed) so the two
calculators (evaluated in separate processes) see *identical* structures.

Output: results/force_geoms/g_XXXX.xyz  (n_rattle displacements per scan point).
"""
from __future__ import annotations
import os, glob
import numpy as np
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
DA = os.path.dirname(HERE)
AMP = 0.10          # Å RMS displacement
N_RATTLE = 3        # displacements per reaction-path point


def main():
    outdir = os.path.join(HERE, "results", "force_geoms")
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(os.path.join(outdir, "*.xyz")):
        os.remove(f)
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(DA, "cscan", "f_*.xyz")))
    k = 0
    for f in files:
        base = read(f)
        for _ in range(N_RATTLE):
            at = base.copy()
            d = rng.normal(0, 1, at.positions.shape)
            d *= AMP / np.sqrt((d**2).sum(1).mean())   # RMS displacement = AMP
            at.positions += d
            p = at.get_positions()
            at.info["coord"] = 0.5*(np.linalg.norm(p[0]-p[5]) + np.linalg.norm(p[3]-p[4]))
            write(os.path.join(outdir, f"g_{k:04d}.xyz"), at)
            k += 1
    print(f"wrote {k} rattled geometries (AMP={AMP} Å) -> {outdir}")


if __name__ == "__main__":
    main()
