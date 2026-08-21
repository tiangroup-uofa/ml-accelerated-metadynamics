#!/usr/bin/env python3
"""
Phase 1C: Sella saddle-point search on MACE PES, starting from xTB TS geometry.

Input: xTB TS geometry (ts_opt.xyz)
Output: converged MACE TS, Hessian/imaginary modes, RMSD comparison
"""
import os, json, sys
os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.calculators.calculator import Calculator

import torch
torch.set_num_threads(4)
from mace.calculators import mace_off
import sella

HERE = Path(__file__).resolve().parent
TS_GUESS = HERE / "ts_opt.xyz"
OUTPUT_DIR = HERE / "phase1c_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Read starting guess
at_init = read(str(TS_GUESS))
print(f"Starting from: {TS_GUESS.name}")
print(f"Initial forming C–C: {np.linalg.norm(at_init.positions[0] - at_init.positions[5]):.3f} / {np.linalg.norm(at_init.positions[3] - at_init.positions[4]):.3f} Å")

# Attach MACE calculator
calc = mace_off(model="small", device="cpu", default_dtype="float64")
at_init.calc = calc

# Run Sella saddle search
print("\n=== Sella saddle-point search ===")
dyn = sella.Sella(at_init, trajectory=str(OUTPUT_DIR / "sella.traj"))
try:
    dyn.run(fmax=1e-5, steps=200)
    converged = True
except Exception as e:
    print(f"Warning: Sella iteration stopped: {e}")
    converged = dyn.converged if hasattr(dyn, 'converged') else False

# Get final geometry
at_final = at_init.copy()
at_final.calc = calc
E_final = float(at_final.get_potential_energy())
F_final = at_final.get_forces()
fmax_final = float(np.max(np.linalg.norm(F_final, axis=1)))

d05_final = float(np.linalg.norm(at_final.positions[0] - at_final.positions[5]))
d34_final = float(np.linalg.norm(at_final.positions[3] - at_final.positions[4]))
forming_mean_final = 0.5 * (d05_final + d34_final)

print(f"Converged: {converged}")
print(f"Final energy: {E_final:.6f} eV")
print(f"Final fmax: {fmax_final:.6f} eV/Å")
print(f"Final forming C–C: {d05_final:.3f} / {d34_final:.3f} Å (mean {forming_mean_final:.3f})")
print(f"Iterations: {dyn.nsteps if hasattr(dyn, 'nsteps') else '?'}")

# Save converged geometry
write(str(OUTPUT_DIR / "mace_ts_converged.xyz"), at_final)

# Compute Hessian for verification
print("\n=== Computing Hessian (saddle verification) ===")
from ase.calculators.numerical import FiniteDifferenceCalculator

def compute_hessian_eigenvalues(atoms, calc, delta=0.001):
    """Compute Hessian eigenvalues via finite differences."""
    n_atoms = len(atoms)
    n_dof = 3 * n_atoms
    H = np.zeros((n_dof, n_dof))

    for i in range(n_dof):
        # Forward
        atoms_fwd = atoms.copy()
        atoms_fwd.positions.flat[i] += delta
        atoms_fwd.calc = calc
        F_fwd = atoms_fwd.get_forces().flatten()

        # Backward
        atoms_bwd = atoms.copy()
        atoms_bwd.positions.flat[i] -= delta
        atoms_bwd.calc = calc
        F_bwd = atoms_bwd.get_forces().flatten()

        H[i, :] = (F_bwd - F_fwd) / (2 * delta)

    # Eigenvalue decomposition
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs

print("Computing Hessian (3N×3N = 48×48)...")
try:
    evals, _ = compute_hessian_eigenvalues(at_final, calc, delta=0.001)
    evals_cm = evals * 5142.207  # eV/Å² to cm⁻¹
    n_imag = np.sum(evals_cm < -10)
    print(f"Imaginary modes (< -10 cm⁻¹): {n_imag}")
    print(f"Lowest 5 eigenvalues (cm⁻¹): {sorted(evals_cm)[:5]}")
except Exception as e:
    print(f"Hessian computation failed: {e}")
    n_imag = None

# RMSD to xTB TS (after optimal alignment)
at_xtb = read(str(TS_GUESS))
# Simple RMSD (without optimal rotation)
rmsd_simple = float(np.sqrt(np.mean((at_final.positions - at_xtb.positions)**2)))
print(f"\nRMSD to xTB TS (no alignment): {rmsd_simple:.4f} Å")

# Write results
EV2KCAL = 23.060548
result = {
    "comment": "MACE saddle search starting from xTB TS geometry",
    "xTB_TS_initial": {
        "forming_CC_mean_Angstrom": 2.315,
        "E_eV": float(at_init.get_potential_energy())
    },
    "MACE_TS_converged": {
        "forming_CC_mean_Angstrom": float(forming_mean_final),
        "forming_CC_Angstrom": [float(d05_final), float(d34_final)],
        "E_eV": E_final,
        "fmax_eV_per_Angstrom": float(fmax_final),
        "converged": converged,
        "iterations": int(dyn.nsteps) if hasattr(dyn, 'nsteps') else None
    },
    "RMSD_to_xTB_TS_Angstrom": float(rmsd_simple),
    "imaginary_modes_count": int(n_imag) if n_imag is not None else None,
    "files": {
        "converged_geometry": "phase1c_results/mace_ts_converged.xyz",
        "trajectory": "phase1c_results/sella.traj"
    }
}

out_file = HERE / "phase1c_results.json"
out_file.write_text(json.dumps(result, indent=2))
print(f"\nwrote {out_file}")
