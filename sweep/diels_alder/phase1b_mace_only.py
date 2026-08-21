#!/usr/bin/env python3
"""
Phase 1B MACE part: Evaluate MACE-OFF23 on the three frozen xTB R/TS/P geometries.
Separate script to avoid xtb dependency; can run anywhere with torch+MACE.
"""
import json, os, sys
os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
import numpy as np
from ase.io import read

import torch
torch.set_num_threads(4)
from mace.calculators import mace_off

HERE = Path(__file__).resolve().parent
REACTANT = HERE / "start.xyz"
TS = HERE / "ts_opt.xyz"
PRODUCT = HERE / "end.xyz"

calc = mace_off(model="small", device="cpu", default_dtype="float64")
EV2KCAL = 23.060548

results = {}
for name, path in [("R", REACTANT), ("TS", TS), ("P", PRODUCT)]:
    at = read(str(path))
    at.calc = calc
    E = float(at.get_potential_energy())
    F = at.get_forces()
    fmax = float(np.max(np.linalg.norm(F, axis=1)))
    fnorm = float(np.linalg.norm(F))
    results[name] = {"E_eV": E, "fmax": fmax, "fnorm": fnorm}
    print(f"{name}: E={E:.6f} eV, fmax={fmax:.4f}, fnorm={fnorm:.4f}")

# Calculate energetics
barrier = (results["TS"]["E_eV"] - results["R"]["E_eV"]) * EV2KCAL
rxn = (results["P"]["E_eV"] - results["R"]["E_eV"]) * EV2KCAL
print(f"\nMAC E barrier: {barrier:.1f} kcal/mol")
print(f"MACE reaction: {rxn:.1f} kcal/mol")

# Write compact output
out = HERE / "phase1b_mace_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"wrote {out}")
