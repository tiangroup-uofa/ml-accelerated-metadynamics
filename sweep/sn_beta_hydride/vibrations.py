#!/usr/bin/env python3
"""
vibrations.py
=============
Finite-difference Hessian + **mass-weighted** normal-mode analysis, generalized
from ``sweep/corrections/phase2_lib.py`` (the scientifically verified version)
so it carries no Diels-Alder atom indices:

* ``hessian``              identical central-difference scheme (delta = 1e-3 Å,
                           H_ij = (F_j(x_i - d) - F_j(x_i + d)) / 2d, symmetrized),
                           optionally restricted to a subset of atoms
                           (partial Hessian, e.g. excluding fixed atoms).
* ``vibrational_analysis`` eigen-decomposition of M^-1/2 H M^-1/2,
                           nu = sign(lambda) * 521.47 * sqrt|lambda|  [cm^-1]
                           (eV/Å²/amu -> cm^-1), imaginary modes reported as
                           negative wavenumbers; a mode is "significant" when
                           nu < -imag_cutoff_cm (default 50 cm^-1, as in phase2).
                           The overlap of the dominant imaginary mode with a
                           caller-supplied Cartesian reference direction (e.g.
                           the gradient of the hydride-transfer CV) is computed
                           exactly as in phase2_lib: |c_hat . v_hat| with c the
                           mass-weighted eigenvector back-transformed to
                           Cartesian displacements.

With ``indices=None`` and the Diels-Alder forming-bond vector as reference, the
results are numerically identical to ``phase2_lib.vibrational_analysis`` (see
tests/test_sn_beta_hydride.py::test_vibrations_match_phase2_lib).

NOT used: the raw-Hessian-eigenvalue x 5142 conversion in
``phase1c_mace_ts_search.py`` / ``phase1c_hessian.py`` — it omits mass weighting
and does not give vibrational frequencies.
"""
from __future__ import annotations

import numpy as np

LAMBDA_TO_CM = 521.47           # sqrt(eV / (Å^2 amu)) -> cm^-1 (same as phase2_lib)
HESS_DELTA = 0.001
IMAG_CUTOFF_CM = 50.0


def hessian(atoms, calc, delta: float = HESS_DELTA, indices=None, progress=None) -> np.ndarray:
    """Central-difference Cartesian Hessian (eV/Å²), symmetrized.

    ``indices``: atoms whose coordinates are displaced (rows/columns kept);
    default all atoms. Returns a (3k x 3k) matrix for k = len(indices)."""
    idx = np.arange(len(atoms)) if indices is None else np.asarray(sorted(indices), int)
    dof = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in idx]).astype(int)
    ndof = len(dof)
    H = np.zeros((ndof, ndof))
    for r, flat in enumerate(dof):
        a = atoms.copy()
        a.set_constraint()
        a.calc = calc
        a.positions.flat[flat] += delta
        Ff = a.get_forces().ravel()[dof]
        b = atoms.copy()
        b.set_constraint()
        b.calc = calc
        b.positions.flat[flat] -= delta
        Fb = b.get_forces().ravel()[dof]
        H[r] = (Fb - Ff) / (2 * delta)
        if progress:
            progress(r + 1, ndof)
    return 0.5 * (H + H.T)


def normal_modes(H, masses):
    """Mass-weighted eigen-decomposition. Returns (nu_cm, vecs_mw, w) with
    w = 1/sqrt(m) per DOF so that Cartesian displacement = vecs[:, k] * w."""
    w = 1.0 / np.sqrt(np.repeat(np.asarray(masses, float), 3))
    lam, vecs = np.linalg.eigh(H * np.outer(w, w))
    nu = np.sign(lam) * LAMBDA_TO_CM * np.sqrt(np.abs(lam))
    return nu, vecs, w


def analyze_hessian(H, atoms, indices=None, reference_vector=None,
                    imag_cutoff_cm: float = IMAG_CUTOFF_CM, top_atoms: int = 8) -> dict:
    """Frequencies + dominant-imaginary-mode diagnostics from a Hessian."""
    n = len(atoms)
    idx = np.arange(n) if indices is None else np.asarray(sorted(indices), int)
    m = atoms.get_masses()[idx]
    nu, vecs, w = normal_modes(H, m)
    imag_idx = [i for i in range(len(nu)) if nu[i] < -imag_cutoff_cm]
    small_imag = [float(x) for x in nu if -imag_cutoff_cm <= x < 0]

    out = {
        "n_atoms_in_hessian": int(len(idx)),
        "hessian_indices": idx.tolist() if indices is not None else "all",
        "frequencies_cm": [float(x) for x in nu],
        "n_imaginary": len(imag_idx),
        "imag_freqs_cm": [float(nu[i]) for i in imag_idx],
        "small_imaginary_below_cutoff_cm": small_imag,
        "lowest_10_cm": [float(x) for x in np.sort(nu)[:10]],
        "imag_cutoff_cm": imag_cutoff_cm,
        "nu_imag_cm": None,
        "reaction_mode_overlap": None,
        "mode_displacement": None,
        "mode_atom_participation": None,
    }
    if not imag_idx:
        return out
    k = imag_idx[int(np.argmin([nu[i] for i in imag_idx]))]
    out["nu_imag_cm"] = float(nu[k])
    cart = vecs[:, k] * w
    cart /= np.linalg.norm(cart)
    full = np.zeros((n, 3))
    full[idx] = cart.reshape(-1, 3)
    out["mode_displacement"] = full.tolist()     # unit-norm Cartesian, (n_atoms, 3)
    part = np.sum(full ** 2, axis=1)              # fraction of |c|^2 per atom
    order = np.argsort(part)[::-1][:top_atoms]
    syms = atoms.get_chemical_symbols()
    out["mode_atom_participation"] = [
        {"index": int(i), "element": syms[i], "fraction": float(part[i])} for i in order]
    if reference_vector is not None:
        rv = np.asarray(reference_vector, float).reshape(n, 3)[idx].ravel()
        nrm = np.linalg.norm(rv)
        if nrm > 0:
            out["reaction_mode_overlap"] = float(abs(cart @ (rv / nrm)))
    return out


def vibrational_analysis(atoms, calc, delta: float = HESS_DELTA,
                         imag_cutoff_cm: float = IMAG_CUTOFF_CM, indices=None,
                         reference_vector=None, return_hessian: bool = False,
                         progress=None):
    """Hessian + mass-weighted mode analysis in one call."""
    H = hessian(atoms, calc, delta, indices, progress)
    res = analyze_hessian(H, atoms, indices, reference_vector, imag_cutoff_cm)
    return (res, H) if return_hessian else res
