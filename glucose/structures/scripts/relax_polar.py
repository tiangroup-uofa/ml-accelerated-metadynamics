#!/usr/bin/env python3
"""
relax_polar.py
==============
Local relaxation of the hand-built Sn-beta hydride-transfer endpoints with a
MACE foundation model (default MACE-POLAR-1 polar-1-s), neutral singlet.

1. relax reactant_guess.xyz                         -> reactant_<tag>.xyz
2. product guess: move H_transfer from C2 onto C1 (sp3 site on the face the
   H comes from), keep every atom in the same order -> product_guess.xyz
3. relax the product guess                          -> product_<tag>.xyz

These are local minima near a hand-built guess, NOT a conformer search and
NOT the paper's geometries (the paper gives no coordinates).

    $XTB_ENV/bin/python relax_polar.py [--model mace-polar] [--variant polar-1-s]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, FIRE

# Keep archived structures untouched; all rerun files live in a separate workspace.
import os
import shutil

ARCHIVE = Path(__file__).resolve().parent.parent
HERE = Path(os.environ.get("GLUCOSE_WORKDIR", str(ARCHIVE / "runs" / "polar-1-s"))).resolve()
if HERE == ARCHIVE or ARCHIVE in HERE.parents and "runs" not in HERE.relative_to(ARCHIVE).parts:
    raise ValueError("Use a separate workspace, not the structure archive or scripts directory")
HERE.mkdir(parents=True, exist_ok=True)
for pattern in ("*.xyz", "*.json", "*.csv"):
    for source in ARCHIVE.glob(pattern):
        target = HERE / source.name
        if not target.exists():
            shutil.copy2(source, target)
from calculators import CalcSettings, build_model  # noqa: E402


def unit(v):
    return v / np.linalg.norm(v)


def product_guess(atoms, m):
    """Put H* on C1, opposite the plane of C1's three current neighbours."""
    a = atoms.copy()
    X = a.positions
    c1 = X[m["C1"]]
    nb = [X[m["H_C1"]], X[m["O1"]], X[m["C2"]]]
    n = unit(np.cross(nb[1] - nb[0], nb[2] - nb[0]))
    if np.dot(X[m["H_transfer"]] - c1, n) < 0:      # the face H* sits on
        n = -n
    X[m["H_transfer"]] = c1 + 1.10 * n
    a.positions = X
    return a


def report(atoms, m):
    X = atoms.positions
    d = lambda i, j: float(np.linalg.norm(X[m[i]] - X[m[j]]))  # noqa: E731
    sn = X[m["Sn"]]
    o_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"]
    shell = sorted(float(np.linalg.norm(X[i] - sn)) for i in o_idx)
    return {"C1-H*": d("C1", "H_transfer"), "C2-H*": d("C2", "H_transfer"),
            "C1-O1": d("C1", "O1"), "C2-O2": d("C2", "O2"),
            "Sn-O1": d("Sn", "O1"), "Sn-O2": d("Sn", "O2"), "Sn-Ow": d("Sn", "O_water"),
            "n_O_within_2.8": sum(r < 2.8 for r in shell),
            "Sn-O_shell": [round(r, 3) for r in shell[:7]]}


def relax(atoms, calc, stem, fmax, steps):
    atoms = atoms.copy()
    atoms.calc = calc
    t0 = time.time()
    # FIRE first (robust from a rough build), then BFGS to tighten
    FIRE(atoms, trajectory=str(HERE / f"{stem}.traj"), logfile=str(HERE / f"{stem}.log")
         ).run(fmax=0.2, steps=steps)
    opt = BFGS(atoms, trajectory=str(HERE / f"{stem}_bfgs.traj"),
               logfile=str(HERE / f"{stem}.log"))
    ok = opt.run(fmax=fmax, steps=steps)
    e = float(atoms.get_potential_energy())
    fm = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
    return atoms, {"energy_eV": e, "fmax_eV_A": fm, "converged": bool(ok),
                   "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mace-polar")
    ap.add_argument("--variant", default="polar-1-s")
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    meta = json.loads((HERE / "atom_map.json").read_text())
    m = meta["atom_map"]
    settings = CalcSettings(model=args.model, variant=args.variant, dtype="float64",
                            threads=args.threads)
    handle = build_model(settings, meta["charge"], meta["spin_multiplicity"])
    tag = settings.tag
    calc = handle.new_calc()

    react0 = read(HERE / "reactant_guess.xyz")
    handle.check_elements(react0)
    react, rinfo = relax(react0, calc, f"reactant_{tag}", args.fmax, args.steps)
    write(HERE / f"reactant_{tag}.xyz", react)
    print("reactant", rinfo, report(react, m), flush=True)

    prod0 = product_guess(react, m)
    write(HERE / "product_guess.xyz", prod0)
    prod, pinfo = relax(prod0, calc, f"product_{tag}", args.fmax, args.steps)
    write(HERE / f"product_{tag}.xyz", prod)
    print("product", pinfo, report(prod, m), flush=True)

    dE = (pinfo["energy_eV"] - rinfo["energy_eV"]) * 96.485
    out = {"model": handle.info(), "charge": meta["charge"],
           "spin_multiplicity": meta["spin_multiplicity"],
           "reactant": {**rinfo, **report(react, m)},
           "product": {**pinfo, **report(prod, m)},
           "dE_product_minus_reactant_kJ_mol": round(dE, 2),
           "note": "local minima from a hand-built guess; potential energy, not a free energy"}
    (HERE / f"relax_{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"dE(product - reactant) = {dE:.1f} kJ/mol  -> relax_{tag}.json")


if __name__ == "__main__":
    main()
