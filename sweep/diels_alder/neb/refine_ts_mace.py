#!/usr/bin/env python3
"""
refine_ts_mace.py
=================
The CI-NEB climbing image is only an approximate saddle (it can carry a second
imaginary mode). Refine it to a genuine MACE-OFF23 first-order transition state
with Sella (partitioned RFO), verify exactly one imaginary frequency, and report
the refined activation barrier relative to the MACE reactant complex.

This mirrors the xTB TS treatment (relaxed scan -> Sella + Hessian) so the two
methods' barriers are obtained the same way.
"""
from __future__ import annotations
import os, sys, warnings, shutil
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import json
import numpy as np
from ase.io import read, write
from ase.vibrations import Vibrations
from sella import Sella
from calculators import make_factory
from neb import optimize_endpoint, forming_bonds, FORMING

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
EV2KCAL = 23.060548


def imaginary(ts):
    d = RES / "vib_refined"
    if d.exists():
        shutil.rmtree(d)
    vib = Vibrations(ts, name=str(d))
    vib.run()
    fr = np.asarray(vib.get_frequencies())
    return sorted({round(-abs(float(np.imag(f))), 1) for f in fr
                   if np.iscomplexobj(fr) and abs(np.imag(f)) > 20}
                  or {round(float(np.real(f)), 1) for f in fr if np.real(f) < -20})


def main() -> int:
    os.environ["OMP_NUM_THREADS"] = "4"
    ts = read(str(RES / "ts_mace.xyz"))
    ts.calc = make_factory("mace", dtype="float64")()
    print("Sella saddle refinement (MACE-OFF23) ...", flush=True)
    print(f"  NEB-TS forming C–C: {forming_bonds(ts)}")
    Sella(ts, order=1, internal=True, logfile=None).run(fmax=0.03, steps=400)
    fmax = float(np.linalg.norm(ts.get_forces(), axis=1).max())
    print(f"  converged max force = {fmax:.4f} eV/Å; forming C–C: {forming_bonds(ts)}")
    write(str(RES / "ts_mace_refined.xyz"), ts)
    e_ts = ts.get_potential_energy()

    imag = imaginary(ts)

    # MACE reactant complex (forming bonds pinned, as in the NEB endpoint)
    reactant = read(str(HERE.parent / "start.xyz"))
    r_opt, e_r = optimize_endpoint(reactant, make_factory("mace", dtype="float64")(),
                                   fmax=0.03, fix_bonds=FORMING)
    barrier = (e_ts - e_r) * EV2KCAL

    is_ts = len(imag) == 1
    tf = forming_bonds(ts)
    print("\n=== MACE-OFF23 refined transition state ===")
    print(f"  imaginary frequencies (cm^-1): {imag}")
    print(f"  {'OK: one imaginary mode = genuine TS' if is_ts else 'WARNING: not exactly one imaginary mode'}")
    print(f"  TS forming C–C: {tf[0]:.2f}, {tf[1]:.2f} Å (mean {0.5*(tf[0]+tf[1]):.2f})")
    print(f"  refined barrier = {barrier:.1f} kcal/mol  (NEB estimate was ~37)")

    out = RES / "ts_mace_refined.json"
    out.write_text(json.dumps(dict(barrier_kcal=float(barrier),
                                   ts_forming_A=[tf[0], tf[1]],
                                   imaginary_cm=imag, is_first_order=is_ts),
                              indent=2))
    return 0 if is_ts else 2


if __name__ == "__main__":
    raise SystemExit(main())
