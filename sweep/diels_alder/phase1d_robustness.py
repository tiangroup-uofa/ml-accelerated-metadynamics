#!/usr/bin/env python3
"""
Phase 1D: δx robustness test — small perturbations around MACE TS, Sella recovery.

Input: converged MACE TS from Phase 1C
Process:
  1. Compute Hessian to identify imaginary mode (reaction coordinate)
  2. Create ± perturbations along imaginary mode (±0.02 Å, ±0.05 Å)
  3. Run Sella from each perturbed geometry
  4. Verify convergence back to the same saddle
  5. Report geometry/barrier/imaginary-mode spread

Output: robustness analysis + table
"""
import os, json
os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

import torch
torch.set_num_threads(4)
from mace.calculators import mace_off
import sella

HERE = Path(__file__).resolve().parent
TS_REF = HERE / "phase1c_results" / "mace_ts_converged.xyz"
RESULTS_DIR = HERE / "phase1d_results"
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Phase 1D: δx Robustness Test")
print("=" * 70)

# Load reference MACE TS
at_ref = read(str(TS_REF))
calc = mace_off(model="small", device="cpu", default_dtype="float64")
at_ref.calc = calc
E_ref = float(at_ref.get_potential_energy())
F_ref = at_ref.get_forces()
fmax_ref = float(np.max(np.linalg.norm(F_ref, axis=1)))

print(f"\nReference MACE TS (from Phase 1C):")
print(f"  E = {E_ref:.6f} eV")
print(f"  fmax = {fmax_ref:.8f} eV/Å")

# Compute Hessian to identify imaginary mode
print("\n--- Computing Hessian to identify imaginary mode ---")
n_atoms = len(at_ref)
n_dof = 3 * n_atoms
H = np.zeros((n_dof, n_dof))
delta = 0.001

print(f"Finite differences (delta={delta})...")
for i in range(n_dof):
    if i % 6 == 0:
        print(f"  DOF {i}/{n_dof}...", flush=True)
    at_fwd = at_ref.copy()
    at_fwd.positions.flat[i] += delta
    at_fwd.calc = calc
    F_fwd = at_fwd.get_forces().flatten()

    at_bwd = at_ref.copy()
    at_bwd.positions.flat[i] -= delta
    at_bwd.calc = calc
    F_bwd = at_bwd.get_forces().flatten()

    H[i, :] = (F_bwd - F_fwd) / (2 * delta)

evals, evecs = np.linalg.eigh(H)
evals_cm = evals * 5142.207

# Find the imaginary mode (most negative)
idx_imag = np.argmin(evals)
eval_imag = evals[idx_imag]
eval_imag_cm = evals_cm[idx_imag]
evec_imag = evecs[:, idx_imag]

print(f"\nImaginary mode eigenvalue: {eval_imag:.6f} eV/Å²")
print(f"Imaginary mode frequency: {eval_imag_cm:.1f} cm⁻¹")
print(f"Norm of eigenvector: {np.linalg.norm(evec_imag):.6f}")

# Normalize eigenvector
evec_imag = evec_imag / np.linalg.norm(evec_imag)

# Create perturbations
perturbation_displacements = [0.02, 0.05]  # Å
perturbations = {}

print("\n--- Creating perturbations ---")
for disp in perturbation_displacements:
    for sign in [-1, +1]:
        label = f"{sign:+d}_{disp:.2f}A"

        # Displace along imaginary mode
        at_pert = at_ref.copy()
        displacement_vector = evec_imag.reshape((n_atoms, 3))
        at_pert.positions += sign * disp * displacement_vector

        at_pert.calc = calc
        E_pert_init = float(at_pert.get_potential_energy())
        F_pert_init = at_pert.get_forces()
        fmax_pert_init = float(np.max(np.linalg.norm(F_pert_init, axis=1)))

        perturbations[label] = {
            "displacement_Angstrom": sign * disp,
            "direction": "imaginary_mode",
            "initial_E_eV": E_pert_init,
            "initial_fmax_eV_per_Angstrom": fmax_pert_init,
            "atoms": at_pert
        }

        print(f"  {label:12s}: E_init={E_pert_init:.6f} eV, fmax_init={fmax_pert_init:.4f} eV/Å")

# Run Sella from each perturbation
print("\n--- Running Sella from each perturbation ---")
results = {}
EV2KCAL = 23.060548

for label, pert_data in perturbations.items():
    print(f"\n{label}:")
    at = pert_data["atoms"]
    at.calc = calc

    traj_file = RESULTS_DIR / f"sella_{label}.traj"
    dyn = sella.Sella(at, trajectory=str(traj_file))

    try:
        dyn.run(fmax=1e-5, steps=200)
        converged = True
    except Exception as e:
        print(f"  Warning: {e}")
        converged = False

    # Get final state
    E_final = float(at.get_potential_energy())
    F_final = at.get_forces()
    fmax_final = float(np.max(np.linalg.norm(F_final, axis=1)))

    # Forming C–C distances
    d05 = float(np.linalg.norm(at.positions[0] - at.positions[5]))
    d34 = float(np.linalg.norm(at.positions[3] - at.positions[4]))
    forming_mean = 0.5 * (d05 + d34)

    # RMSD to reference TS
    rmsd = float(np.sqrt(np.mean((at.positions - at_ref.positions)**2)))

    # Barrier
    at_reactant = read(str(HERE / "start.xyz"))
    at_reactant.calc = calc
    E_r = float(at_reactant.get_potential_energy())
    barrier = (E_final - E_r) * EV2KCAL

    results[label] = {
        "converged": converged,
        "iterations": int(dyn.nsteps) if hasattr(dyn, 'nsteps') else None,
        "final_E_eV": E_final,
        "final_fmax_eV_per_Angstrom": fmax_final,
        "final_forming_CC_mean_Angstrom": forming_mean,
        "final_forming_CC_Angstrom": [float(d05), float(d34)],
        "RMSD_to_ref_TS_Angstrom": rmsd,
        "barrier_kcal_mol": float(barrier),
        "trajectory_file": str(traj_file)
    }

    print(f"  Converged: {converged} ({dyn.nsteps if hasattr(dyn, 'nsteps') else '?'} iterations)")
    print(f"  Final fmax: {fmax_final:.8f} eV/Å")
    print(f"  Final forming C–C: {d05:.3f} / {d34:.3f} Å (mean {forming_mean:.3f})")
    print(f"  RMSD to ref TS: {rmsd:.4f} Å")
    print(f"  Barrier: {barrier:.1f} kcal/mol")

# Hessian verification for at least one recovered TS
print("\n--- Verifying recovered TS (checking imaginary mode count) ---")
label_verify = "+_0.05A"  # Verify the +0.05 Å perturbation
if label_verify in perturbations:
    at_verify = perturbations[label_verify]["atoms"]
    print(f"Computing Hessian for {label_verify} recovered TS...")

    H_verify = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        if i % 12 == 0:
            print(f"  DOF {i}/{n_dof}...", flush=True)
        at_fwd = at_verify.copy()
        at_fwd.positions.flat[i] += delta
        at_fwd.calc = calc
        F_fwd = at_fwd.get_forces().flatten()

        at_bwd = at_verify.copy()
        at_bwd.positions.flat[i] -= delta
        at_bwd.calc = calc
        F_bwd = at_bwd.get_forces().flatten()

        H_verify[i, :] = (F_bwd - F_fwd) / (2 * delta)

    evals_verify, _ = np.linalg.eigh(H_verify)
    evals_verify_cm = evals_verify * 5142.207
    n_imag_verify = np.sum(evals_verify_cm < -1)

    print(f"Imaginary modes: {n_imag_verify}")
    print(f"Lowest 3 eigenvalues (cm⁻¹): {sorted(evals_verify_cm)[:3]}")

    results["hessian_verification"] = {
        "structure": label_verify,
        "imaginary_modes": int(n_imag_verify),
        "lowest_cm_inv": sorted(evals_verify_cm)[:3]
    }

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY: δx Robustness")
print("=" * 70)

converged_count = sum(1 for r in results.values() if isinstance(r, dict) and r.get("converged"))
print(f"\nConverged: {converged_count}/{len(perturbations)}")

if converged_count > 0:
    barriers = [r["barrier_kcal_mol"] for r in results.values() if isinstance(r, dict) and r.get("converged")]
    rmsds = [r["RMSD_to_ref_TS_Angstrom"] for r in results.values() if isinstance(r, dict) and r.get("converged")]
    forming_means = [r["final_forming_CC_mean_Angstrom"] for r in results.values() if isinstance(r, dict) and r.get("converged")]

    print(f"\nBarrier range: {min(barriers):.1f} – {max(barriers):.1f} kcal/mol (spread {max(barriers) - min(barriers):.1f})")
    print(f"RMSD range: {min(rmsds):.4f} – {max(rmsds):.4f} Å (spread {max(rmsds) - min(rmsds):.4f})")
    print(f"Forming C–C mean range: {min(forming_means):.4f} – {max(forming_means):.4f} Å (spread {max(forming_means) - min(forming_means):.4f})")
    print(f"\nReference TS barrier: 35.8 kcal/mol")
    print(f"All recovered barriers within {max(abs(b - 35.8) for b in barriers):.1f} kcal/mol? {all(abs(b - 35.8) < 1.0 for b in barriers)}")

# Write results
out_file = RESULTS_DIR / "phase1d_results.json"
out_file.write_text(json.dumps(results, indent=2, default=str))
print(f"\nWrote {out_file}")
