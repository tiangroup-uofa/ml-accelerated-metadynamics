#!/usr/bin/env python3
"""
phase2_lib.py — shared helpers for the Phase 2 iterative correction loop.

Runs in the *system* python (ase + torch + mace + sella). xTB lives in a separate
micromamba env and is reached through `xtb_eval()` (subprocess), matching the
process-isolation pattern already used by phaseC_eval.py (MACE and xtb-python
both link OpenMP).

Nothing here changes the correction architecture: the pairwise Delta is imported
from the existing sweep/corrections/pairwise_delta.py with its default
hyperparameters (cutoff 3.6, n_rbf 8, rmin 0.7, ridge 1e-3). Only the reference
DATA grows across iterations.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
DA = HERE.parent / "diels_alder"
sys.path.insert(0, str(DA / "neb"))
sys.path.insert(0, str(HERE))

from corrections import CorrectedCalculator, Delta, Identity  # noqa: E402
from pairwise_delta import PairwiseDelta  # noqa: E402

EV2KCAL = 23.060548
FORMING = [(0, 5), (3, 4)]

# eV/Å²/amu  ->  cm⁻¹   (nu = 521.47 * sqrt(lambda))
LAMBDA_TO_CM = 521.47

MICROMAMBA = Path.home() / ".local/bin/micromamba"
MAMBA_ROOT = Path.home() / ".local/mm/root"
XTB_ENV = "xtbenv"

# reference geometries (native xTB, verified in Phase 1A)
REACTANT_XYZ = DA / "start.xyz"
XTB_TS_XYZ = DA / "ts_opt.xyz"
PRODUCT_XYZ = DA / "end.xyz"

# verified Phase 1 constants (targets / references, not recomputed)
XTB_BARRIER_KCAL = 6.7
XTB_RXN_KCAL = -57.6
XTB_TS_FORMING = 2.315


# --------------------------------------------------------------------------- #
#  calculators
# --------------------------------------------------------------------------- #
def mace_base(dtype="float64", threads=4):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import torch

    torch.set_num_threads(threads)
    from mace.calculators import mace_off

    return mace_off(model="small", device="cpu", default_dtype=dtype)


def corrected(base, delta: PairwiseDelta | None):
    """MACE (+ optional pairwise Delta) as one ASE calculator."""
    return CorrectedCalculator(base, Identity() if delta is None else Delta(delta))


# --------------------------------------------------------------------------- #
#  xTB via subprocess (separate env)
# --------------------------------------------------------------------------- #
def xtb_eval(xyz_paths, tag="xtb"):
    """GFN2-xTB energies + forces for a list of .xyz files. Returns (E, F).

    Uses the same robust settings as the A1 stability gate / phaseC.
    """
    xyz_paths = [str(p) for p in xyz_paths]
    worker = HERE / "phase2_xtb_worker.py"
    listing = HERE / f".xtb_in_{tag}.json"
    out = HERE / f".xtb_out_{tag}.json"
    listing.write_text(json.dumps(xyz_paths))
    env = dict(os.environ, MAMBA_ROOT_PREFIX=str(MAMBA_ROOT))
    r = subprocess.run(
        [str(MICROMAMBA), "run", "-n", XTB_ENV, "python", str(worker),
         str(listing), str(out)],
        capture_output=True, text=True, env=env, timeout=3600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"xTB worker failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    d = json.loads(out.read_text())
    listing.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    if d.get("errors"):
        raise RuntimeError(f"xTB errors: {d['errors']}")
    return np.array(d["energies"]), [np.array(f) for f in d["forces"]]


# --------------------------------------------------------------------------- #
#  geometry helpers
# --------------------------------------------------------------------------- #
def forming_distances(atoms):
    p = atoms.get_positions()
    return [float(np.linalg.norm(p[i] - p[j])) for i, j in FORMING]


def forming_mean(atoms):
    return float(np.mean(forming_distances(atoms)))


def kabsch_rmsd(A, B):
    """RMSD after optimal translation + rotation (Kabsch)."""
    A = np.asarray(A, float) - np.asarray(A, float).mean(0)
    B = np.asarray(B, float) - np.asarray(B, float).mean(0)
    V, S, Wt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    A_rot = A @ R
    return float(np.sqrt(np.mean(np.sum((A_rot - B) ** 2, axis=1))))


def reaction_coordinate_vector(atoms):
    """Unit Cartesian direction that symmetrically stretches both forming bonds."""
    p = atoms.get_positions()
    v = np.zeros_like(p)
    for i, j in FORMING:
        d = p[i] - p[j]
        u = d / np.linalg.norm(d)
        v[i] += u
        v[j] -= u
    return (v / np.linalg.norm(v)).ravel()


# --------------------------------------------------------------------------- #
#  Hessian / vibrational analysis (mass-weighted -> real cm^-1)
# --------------------------------------------------------------------------- #
def hessian(atoms, calc, delta=0.001):
    """Central-difference Cartesian Hessian (eV/Å²), symmetrized."""
    n = len(atoms)
    ndof = 3 * n
    H = np.zeros((ndof, ndof))
    for i in range(ndof):
        a = atoms.copy()
        a.calc = calc
        a.positions.flat[i] += delta
        Ff = a.get_forces().ravel()
        b = atoms.copy()
        b.calc = calc
        b.positions.flat[i] -= delta
        Fb = b.get_forces().ravel()
        H[i] = (Fb - Ff) / (2 * delta)
    return 0.5 * (H + H.T)


def vibrational_analysis(atoms, calc, delta=0.001, imag_cutoff_cm=50.0):
    """Mass-weighted normal-mode analysis.

    Returns dict with frequencies (cm⁻¹, negative = imaginary), the count of
    genuine imaginary modes (|nu| > imag_cutoff_cm), and the overlap of the
    softest imaginary mode with the Diels-Alder forming-bond coordinate.
    """
    H = hessian(atoms, calc, delta)
    m = atoms.get_masses()
    w = 1.0 / np.sqrt(np.repeat(m, 3))
    Hmw = H * np.outer(w, w)
    lam, vecs = np.linalg.eigh(Hmw)
    nu = np.sign(lam) * LAMBDA_TO_CM * np.sqrt(np.abs(lam))

    imag_idx = [i for i in range(len(nu)) if nu[i] < -imag_cutoff_cm]
    rc = reaction_coordinate_vector(atoms)
    overlap = None
    nu_imag = None
    if imag_idx:
        k = imag_idx[int(np.argmin([nu[i] for i in imag_idx]))]
        nu_imag = float(nu[k])
        # back-transform mass-weighted eigenvector to Cartesian displacement
        cart = vecs[:, k] * w
        cart /= np.linalg.norm(cart)
        overlap = float(abs(cart @ rc))
    return {
        "n_imaginary": len(imag_idx),
        "imag_freqs_cm": [float(nu[i]) for i in imag_idx],
        "nu_imag_cm": nu_imag,
        "reaction_mode_overlap": overlap,
        "lowest_10_cm": [float(x) for x in np.sort(nu)[:10]],
        "imag_cutoff_cm": imag_cutoff_cm,
    }


# --------------------------------------------------------------------------- #
#  saddle search
# --------------------------------------------------------------------------- #
def sella_saddle(atoms_init, calc, fmax=1e-4, steps=300, traj=None, log=None):
    """First-order saddle search (Sella). Returns (atoms, info)."""
    import sella

    at = atoms_init.copy()
    at.calc = calc
    dyn = sella.Sella(at, trajectory=str(traj) if traj else None,
                      logfile=str(log) if log else None)
    converged = False
    err = None
    try:
        converged = bool(dyn.run(fmax=fmax, steps=steps))
    except Exception as e:  # pragma: no cover - diagnostic path
        err = repr(e)
    F = at.get_forces()
    info = {
        "converged": converged,
        "error": err,
        "iterations": int(getattr(dyn, "nsteps", -1)),
        "final_fmax": float(np.max(np.linalg.norm(F, axis=1))),
        "final_fnorm": float(np.linalg.norm(F)),
        "energy_eV": float(at.get_potential_energy()),
        "forming_CC": forming_distances(at),
        "forming_CC_mean": forming_mean(at),
    }
    return at, info


# --------------------------------------------------------------------------- #
#  force-comparison metrics
# --------------------------------------------------------------------------- #
def force_metrics(F_pred, F_ref):
    a = np.asarray(F_pred).ravel()
    b = np.asarray(F_ref).ravel()
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return rmse, cos


def fit_delta(configs, dF, ridge=1e-3):
    """Fit the SAME pairwise Delta architecture (defaults fixed) on residuals."""
    return PairwiseDelta().fit(configs, dF, ridge=ridge)


def delta_hash(d: PairwiseDelta):
    import hashlib

    return hashlib.sha256(
        json.dumps([round(float(c), 12) for c in d.coeffs]).encode()
    ).hexdigest()[:16]
