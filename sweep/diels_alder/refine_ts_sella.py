#!/usr/bin/env python3
"""
refine_ts_sella.py
=================
Converge the RMSD-path TS guess to a genuine Diels-Alder transition state with
the Sella saddle optimizer (partitioned rational-function optimization in
internal coordinates) driving GFN2-xTB. Sella is the robust standard for
TS-with-ASE; the bare ASE Dimer would not converge here.

Then confirm exactly one imaginary frequency (xtb --hess) -- a real DA TS has a
large imaginary mode (~ -400..-600 cm^-1) that is the two forming C-C bonds
moving together -- and report the real barrier E(TS) - E(reactant complex),
replacing the inflated ~101 kcal/mol from the biased --path.

Run: micromamba run -n xtb python refine_ts_sella.py
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read, write
from sella import Sella
from xtb.ase.calculator import XTB

HERE = Path(__file__).resolve().parent
HARTREE_KCAL = 627.509
XTB_BIN = os.environ.get("XTB_BIN", "xtb")


def imaginary_freqs_from_vibspectrum(path: Path):
    """Read xtb's canonical `vibspectrum` file and return the imaginary
    (negative) wavenumbers. NOTE: do NOT scrape frequencies from the hess
    stdout by regex -- that also matches the 'imag. cutoff  -20.0 cm' THERMO
    PARAMETER line, which is not a real mode (a bug we hit earlier)."""
    negs = []
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) >= 3 and p[0].isdigit():
            nums = [float(x) for x in p if re.fullmatch(r"-?\d+\.\d+", x)]
            if nums and nums[0] < -1.0:
                negs.append(round(nums[0], 1))
    return sorted(set(negs))


def total_energy(t: str) -> float:
    return float(re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", t)[-1])


def xtb_bin(args, cwd):
    p = subprocess.run([XTB_BIN, *args], cwd=str(cwd), capture_output=True,
                       text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    return p.stdout + p.stderr


def forming_bonds(atoms):
    p = atoms.get_positions()
    d = lambda i, j: float(np.linalg.norm(p[i] - p[j]))
    return d(0, 5), d(3, 4)      # C1-C6, C4-C5


def main() -> int:
    # IMPORTANT: start from the CONCERTED-SCAN TS guess (symmetric, ~2.32 A),
    # NOT the raw --path guess. A local saddle search from the poor --path
    # guess slides off to a trivial intermolecular saddle (we verified this);
    # from the good symmetric guess Sella converges to the real DA TS.
    guess = HERE / "cscan_ts_guess.xyz"
    if not guess.exists():
        guess = HERE / "xtbpath_ts.xyz"
        print("WARNING: cscan_ts_guess.xyz missing; run scan_ts.py + concerted "
              "scan first for a reliable TS.")
    atoms = read(str(guess))
    atoms.calc = XTB(method="GFN2-xTB")

    print("Sella saddle search (order=1, GFN2-xTB) ...", flush=True)
    print(f"  guess forming C-C: {forming_bonds(atoms)}")
    opt = Sella(atoms, order=1, internal=True,
                trajectory=str(HERE / "ts_sella.traj"), logfile=None)
    conv = opt.run(fmax=0.03, steps=500)
    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    print(f"  converged={conv}  max force={fmax:.4f} eV/A")
    print(f"  TS forming C-C: {forming_bonds(atoms)}")
    write(str(HERE / "ts_opt.xyz"), atoms)

    d = HERE / "ts_refine"; d.mkdir(exist_ok=True)
    write(str(d / "ts.xyz"), atoms)
    hess = xtb_bin(["ts.xyz", "--hess", "--gfn", "2", "--chrg", "0", "--uhf", "0"], d)
    (d / "hess.log").write_text(hess)
    imag = imaginary_freqs_from_vibspectrum(d / "vibspectrum")
    ts_E = total_energy(hess)
    r_E = total_energy((HERE / "sp" / "sp_start.log").read_text())
    barrier = (ts_E - r_E) * HARTREE_KCAL

    is_ts = len(imag) == 1
    print("\n=== refined Diels-Alder transition state (Sella) ===")
    print(f"  imaginary frequencies : {imag if imag else 'NONE'}")
    print(f"  {'OK: exactly one imaginary mode -> genuine first-order saddle' if is_ts else 'WARNING: not exactly one imaginary frequency'}")
    print(f"  E(TS)                 : {ts_E:.6f} Eh")
    print(f"  E(reactant complex)   : {r_E:.6f} Eh")
    print(f"  REAL forward barrier  : {barrier:.1f} kcal/mol")
    print(f"  (raw --path max ~101 kcal/mol; tutorial ~12.4)")
    return 0 if is_ts else 2


if __name__ == "__main__":
    raise SystemExit(main())
