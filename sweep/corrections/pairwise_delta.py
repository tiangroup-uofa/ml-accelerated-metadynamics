#!/usr/bin/env python3
"""
pairwise_delta.py
=================
A conservative, lightweight **delta-learning** correction: the energy residual is
a sum of fitted **radial pair potentials**,

    ΔE(x) = Σ_{i<j} Σ_k c[type_ij, k] · φ_k(r_ij),

with φ_k Gaussian radial basis functions × a smooth cutoff. It is:
  * conservative — forces are the analytic gradient −∇ΔE (safe for MD/metadynamics);
  * element-pair specific — separate radial channels per pair (H–H, O–H, C–C, …),
    so one model spans both the water (H/O) and Diels–Alder (C/H) chemistries;
  * linear in the coefficients c — fit by ridge least-squares on the force
    residual F_ref − F_base (no neural-net training, no foundation-model retraining).

Callable `atoms -> (ΔE_eV, ΔF_eV/Å)`, i.e. a drop-in `model` for the existing
`Delta(model=...)` correction / `CorrectedCalculator`.
"""
from __future__ import annotations
import numpy as np

PAIR_TYPES = [("C", "C"), ("C", "H"), ("C", "O"),
              ("H", "H"), ("H", "O"), ("O", "O")]


def _key(a, b):
    return tuple(sorted((a, b)))


class PairwiseDelta:
    def __init__(self, cutoff=3.6, n_rbf=8, rmin=0.7, coeffs=None,
                 pair_types=PAIR_TYPES):
        self.cutoff, self.n_rbf, self.rmin = cutoff, n_rbf, rmin
        self.mu = np.linspace(rmin, cutoff, n_rbf)
        self.beta = 1.0 / ((self.mu[1] - self.mu[0]) ** 2 + 1e-9)
        self.pair_types = list(pair_types)
        self.pt = {t: i for i, t in enumerate(self.pair_types)}
        self.n_basis = len(self.pair_types) * n_rbf
        self.coeffs = None if coeffs is None else np.asarray(coeffs, float)

    # radial basis and cutoff
    def _phi(self, r):
        g = np.exp(-self.beta * (r - self.mu) ** 2)
        x = min(r / self.cutoff, 1.0)
        fc = 0.5 * (np.cos(np.pi * x) + 1.0)
        dfc = -0.5 * np.pi / self.cutoff * np.sin(np.pi * x)
        phi = g * fc
        dphi = (-2 * self.beta * (r - self.mu) * g) * fc + g * dfc
        return phi, dphi

    def _features(self, atoms):
        pos = atoms.get_positions(); sym = atoms.get_chemical_symbols()
        n = len(atoms)
        Ef = np.zeros(self.n_basis)
        Ff = np.zeros((n, 3, self.n_basis))
        for i in range(n):
            for j in range(i + 1, n):
                t = _key(sym[i], sym[j])
                idx = self.pt.get(t)
                if idx is None:
                    continue
                d = pos[i] - pos[j]; r = float(np.linalg.norm(d))
                if r >= self.cutoff or r < 1e-6:
                    continue
                s = idx * self.n_rbf
                phi, dphi = self._phi(r)
                Ef[s:s + self.n_rbf] += phi
                rhat = d / r
                Ff[i, :, s:s + self.n_rbf] += -np.outer(rhat, dphi)
                Ff[j, :, s:s + self.n_rbf] += np.outer(rhat, dphi)
        return Ef, Ff

    def fit(self, configs, residual_forces, ridge=1e-3):
        """configs: ASE atoms; residual_forces: list of (F_ref − F_base) arrays."""
        A, b = [], []
        for at, df in zip(configs, residual_forces):
            _, Ff = self._features(at)
            A.append(Ff.reshape(-1, self.n_basis)); b.append(np.asarray(df).reshape(-1))
        A = np.vstack(A); b = np.concatenate(b)
        self.coeffs = np.linalg.solve(A.T @ A + ridge * np.eye(self.n_basis), A.T @ b)
        return self

    def __call__(self, atoms):
        Ef, Ff = self._features(atoms)
        return float(Ef @ self.coeffs), Ff @ self.coeffs

    # serialization
    def to_dict(self):
        return dict(cutoff=self.cutoff, n_rbf=self.n_rbf, rmin=self.rmin,
                    pair_types=[list(t) for t in self.pair_types],
                    coeffs=self.coeffs.tolist())

    @classmethod
    def from_dict(cls, d):
        return cls(cutoff=d["cutoff"], n_rbf=d["n_rbf"], rmin=d["rmin"],
                   pair_types=[tuple(t) for t in d["pair_types"]],
                   coeffs=np.array(d["coeffs"]))
