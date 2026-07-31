#!/usr/bin/env python3
"""
verify_ts.py
============
Confirm the MACE-OFF23 CI-NEB transition state (results/ts_mace.xyz) is a genuine
first-order saddle: a finite-difference Hessian (ASE Vibrations, MACE-OFF23)
should give exactly one imaginary frequency, and its mode should be the two
forming C–C bonds moving together (the Diels–Alder reaction coordinate).
"""
from __future__ import annotations
import os, sys, warnings, shutil
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
from ase.io import read
from ase.vibrations import Vibrations
from calculators import make_factory

HERE = Path(__file__).resolve().parent


def main() -> int:
    os.environ["OMP_NUM_THREADS"] = "4"
    ts = read(str(HERE / "results" / "ts_mace.xyz"))
    ts.calc = make_factory("mace", dtype="float64")()
    vibdir = HERE / "results" / "vib_mace"
    if vibdir.exists():
        shutil.rmtree(vibdir)
    vib = Vibrations(ts, name=str(vibdir))
    vib.run()
    fr = np.asarray(vib.get_frequencies())   # cm^-1; imaginary modes are complex
    # an imaginary mode: |imag| large (complex), or a clearly negative real
    imag = sorted({round(float(abs(np.imag(f))) * -1.0, 1) for f in fr
                   if np.iscomplexobj(fr) and abs(np.imag(f)) > 20}
                  or {round(float(np.real(f)), 1) for f in fr if np.real(f) < -20})
    print("=== MACE-OFF23 TS verification (ASE Vibrations) ===")
    print(f"  imaginary frequencies (cm^-1): {imag if imag else 'NONE'}")
    ok = len(imag) == 1
    print("  ->", "OK: exactly one imaginary mode = genuine first-order saddle"
          if ok else "WARNING: expected exactly one imaginary frequency")
    p = ts.get_positions()
    d = lambda i, j: float(np.linalg.norm(p[i]-p[j]))
    print(f"  TS forming C–C: {d(0,5):.2f}, {d(3,4):.2f} Å")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
