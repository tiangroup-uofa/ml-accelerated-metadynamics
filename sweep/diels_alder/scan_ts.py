#!/usr/bin/env python3
"""
scan_ts.py
==========
Robust barrier estimate for the Diels-Alder cycloaddition via a relaxed
2-bond scan, after local saddle searches (ASE Dimer, Sella) kept sliding off
the poor RMSD-path TS guess to trivial stationary points.

We drive the TWO forming C-C bonds (C1-C6 and C4-C5, atoms 1&6, 4&5 in
1-indexed xtb numbering) TOGETHER from the reactant value (~3.35 A) down to the
product single-bond length (~1.54 A) using xtb's native `$constrain`/`$scan`,
optimizing every other coordinate at each step. The energy maximum along the
scan is a robust approximate barrier, and the corresponding frame is a good TS
guess (which Sella can then polish from a GOOD starting point).

Output: scan/xtbscan.log (energies), prints barrier + TS-frame geometry ->
scan_ts_guess.xyz.

Run: XTB_BIN=... python scan_ts.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
XTB_BIN = os.environ.get("XTB_BIN", "xtb")
HARTREE_KCAL = 627.509

SCAN_INP = """$constrain
   force constant=1.5
   distance: 1, 6, 3.35
   distance: 4, 5, 3.35
$scan
   1: 3.35, 1.54, 40
   2: 3.35, 1.54, 40
$end
$opt
   maxcycle=200
$end
"""


def main() -> int:
    d = HERE / "scan"; d.mkdir(exist_ok=True)
    shutil.copy(HERE / "start.xyz", d / "start.xyz")
    (d / "scan.inp").write_text(SCAN_INP)
    print("relaxed 2-bond scan (xtb native $scan, GFN2) ...", flush=True)
    with open(d / "scan_run.log", "w") as fh:
        subprocess.run([XTB_BIN, "start.xyz", "--opt", "--input", "scan.inp",
                        "--gfn", "2", "--chrg", "0", "--uhf", "0"],
                       cwd=str(d), stdout=fh, stderr=subprocess.STDOUT, text=True,
                       env={**os.environ, "OMP_NUM_THREADS": "4"})

    scanlog = d / "xtbscan.log"          # multi-frame xyz, comment = energy
    frames = read(str(scanlog), index=":")
    energies = []
    for fr in frames:
        # energy stored in the comment line "energy: <E>" or as 2nd token
        e = fr.info.get("energy")
        if e is None:
            # parse comment
            e = float(str(fr.info.get("comment", "0")).split()[-1])
        energies.append(float(e))
    energies = np.array(energies)
    e0 = energies[0]
    rel = (energies - e0) * HARTREE_KCAL
    imax = int(np.argmax(rel))
    barrier = rel[imax]
    dE = rel[-1]

    write(str(HERE / "scan_ts_guess.xyz"), frames[imax])
    p = frames[imax].get_positions()
    dd = lambda i, j: float(np.linalg.norm(p[i] - p[j]))
    print("\n=== Diels-Alder relaxed-scan barrier ===")
    print(f"  frames                 : {len(frames)}")
    print(f"  barrier (scan max)     : {barrier:.1f} kcal/mol at frame {imax}")
    print(f"  reaction energy (scan) : {dE:.1f} kcal/mol")
    print(f"  TS-guess forming C-C   : {dd(0,5):.2f}, {dd(3,4):.2f} A -> scan_ts_guess.xyz")
    print(f"  (compare tutorial barrier ~12.4, reaction ~ -25..-40)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
