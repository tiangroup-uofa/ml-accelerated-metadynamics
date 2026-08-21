#!/usr/bin/env python3
"""
Phase 1C Hessian + barrier calculation for converged MACE TS.
"""
import os, json
os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
import numpy as np
from ase.io import read

import torch
torch.set_num_threads(4)
from mace.calculators import mace_off

HERE = Path(__file__).resolve().parent
TS_CONVERGED = HERE / "phase1c_results" / "mace_ts_converged.xyz"
REACTANT = HERE / "start.xyz"

if not TS_CONVERGED.exists():
    print(f"ERROR: {TS_CONVERGED} not found")
    exit(1)

calc = mace_off(model="small", device="cpu", default_dtype="float64")
EV2KCAL = 23.060548

# Read structures
at_ts = read(str(TS_CONVERGED))
at_r = read(str(REACTANT))

# Energies
at_ts.calc = calc
E_ts = float(at_ts.get_potential_energy())
at_r.calc = calc
E_r = float(at_r.get_potential_energy())

barrier = (E_ts - E_r) * EV2KCAL

print(f"\n=== MACE TS from Sella search ===")
print(f"E(reactant):      {E_r:.6f} eV")
print(f"E(TS converged):  {E_ts:.6f} eV")
print(f"Barrier:          {barrier:.1f} kcal/mol")

# Forming bond distances
d05 = float(np.linalg.norm(at_ts.positions[0] - at_ts.positions[5]))
d34 = float(np.linalg.norm(at_ts.positions[3] - at_ts.positions[4]))
print(f"Forming C–C:      {d05:.3f} / {d34:.3f} Å (mean {0.5*(d05+d34):.3f})")

# Simple Hessian eigenvalues
print("\n=== Hessian eigenvalues ===")
n_atoms = len(at_ts)
n_dof = 3 * n_atoms
H = np.zeros((n_dof, n_dof))
delta = 0.001

print(f"Computing {n_dof}×{n_dof} Hessian via finite differences (delta={delta})...")
for i in range(n_dof):
    if i % 6 == 0:
        print(f"  DOF {i}/{n_dof}...")
    # Forward
    at_fwd = at_ts.copy()
    at_fwd.positions.flat[i] += delta
    at_fwd.calc = calc
    F_fwd = at_fwd.get_forces().flatten()

    # Backward
    at_bwd = at_ts.copy()
    at_bwd.positions.flat[i] -= delta
    at_bwd.calc = calc
    F_bwd = at_bwd.get_forces().flatten()

    H[i, :] = (F_bwd - F_fwd) / (2 * delta)

# Eigenvalues
evals, _ = np.linalg.eigh(H)
evals_cm = evals * 5142.207  # eV/Å² to cm⁻¹

# Count imaginary modes
imag_modes = evals_cm[evals_cm < -1]
n_imag = len(imag_modes)

print(f"Total DOF: {n_dof} (3N for N={n_atoms} atoms)")
print(f"Imaginary modes (< -1 cm⁻¹): {n_imag}")
print(f"Lowest 10 eigenvalues (cm⁻¹): {sorted(evals_cm)[:10]}")
if n_imag > 0:
    print(f"Imaginary frequencies (cm⁻¹): {sorted(imag_modes)}")

# Verify it's a first-order saddle
if n_imag == 1:
    print("\n✓ FIRST-ORDER SADDLE: exactly one imaginary mode (genuine TS)")
elif n_imag == 0:
    print("\n✗ MINIMUM: no imaginary modes (not a saddle)")
else:
    print(f"\n✗ HIGHER-ORDER SADDLE: {n_imag} imaginary modes")

# Write results
result = {
    "converged_MACE_TS": {
        "E_eV": E_ts,
        "barrier_kcal_mol": float(barrier),
        "forming_CC_mean_Angstrom": float(0.5 * (d05 + d34)),
        "forming_CC_Angstrom": [float(d05), float(d34)]
    },
    "Hessian_analysis": {
        "imaginary_modes": int(n_imag),
        "lowest_10_cm_inv": sorted(evals_cm)[:10]
    },
    "comparison_to_NEB": {
        "NEB_barrier_kcal_mol": 36.0,
        "NEB_TS_forming_CC_mean_Angstrom": 2.03,
        "Sella_from_xTB_barrier_kcal_mol": float(barrier),
        "Sella_from_xTB_TS_forming_CC_mean_Angstrom": float(0.5 * (d05 + d34))
    },
    "comparison_to_xTB": {
        "xTB_barrier_kcal_mol": 6.7,
        "xTB_TS_forming_CC_Angstrom": 2.32,
        "MACE_TS_displacement_Angstrom": float(2.32 - 0.5 * (d05 + d34))
    }
}

out = HERE / "phase1c_hessian_results.json"
out.write_text(json.dumps(result, indent=2))
print(f"\nwrote {out}")
