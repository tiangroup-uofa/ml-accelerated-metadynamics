#!/usr/bin/env python3
"""
corrections.py
==============
Pluggable **post-training force corrections** for a base potential, exposed as a
drop-in ASE calculator. This is the modular hook for the publication direction
(Dr. Tian): the MACE-vs-xTB forces are nearly linearly correlated, so instead of
full fine-tuning a lightweight correction may suffice. The correction models live
behind one interface so they can be inserted into any force-evaluation pipeline
(NEB, MD, metadynamics) *without restructuring* — you just wrap the calculator.

    base = mace_off(...)                         # any ASE calculator
    calc = CorrectedCalculator(base, GlobalScale(alpha=1.05))
    atoms.calc = calc                            # use anywhere

Correction models implemented (all no-ops until fitted):
  Identity        F' = F
  GlobalScale     F' = α F
  Affine          F' = α F,  E' = α E + β        (energy affine; force scaled)
  ElementScale    F'_i = α_{Z_i} F_i             (per-element scaling)
  Delta           F' = F + g(atoms)              (residual / delta-learning; g pluggable)

Each has a `.fit(...)` classmethod to calibrate from paired (base, reference)
forces — not called by the NEB benchmark itself; it's here so the correction can
be trained later and slotted in.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ase.calculators.calculator import Calculator, all_changes


# --------------------------------------------------------------------------- #
#  Correction models
# --------------------------------------------------------------------------- #
class ForceCorrection:
    """Base class. `apply` maps a base (energy, forces) -> corrected pair."""
    def apply(self, atoms, energy: float, forces: np.ndarray):
        return energy, forces

    def __repr__(self):
        return f"{type(self).__name__}()"


class Identity(ForceCorrection):
    pass


@dataclass
class GlobalScale(ForceCorrection):
    alpha: float = 1.0

    def apply(self, atoms, energy, forces):
        return energy, self.alpha * forces

    @classmethod
    def fit(cls, base_forces, ref_forces):
        """Least-squares scalar α minimising ||α·F_base - F_ref||."""
        b = np.concatenate([f.ravel() for f in base_forces])
        r = np.concatenate([f.ravel() for f in ref_forces])
        return cls(alpha=float(b @ r / (b @ b)))


@dataclass
class Affine(ForceCorrection):
    alpha: float = 1.0          # force (and energy) slope
    e_shift: float = 0.0        # energy intercept (eV)

    def apply(self, atoms, energy, forces):
        return self.alpha * energy + self.e_shift, self.alpha * forces

    @classmethod
    def fit(cls, base_forces, ref_forces, base_energies=None, ref_energies=None):
        gs = GlobalScale.fit(base_forces, ref_forces)
        e_shift = 0.0
        if base_energies is not None and ref_energies is not None:
            e_shift = float(np.mean(np.asarray(ref_energies)
                                    - gs.alpha * np.asarray(base_energies)))
        return cls(alpha=gs.alpha, e_shift=e_shift)


@dataclass
class ElementScale(ForceCorrection):
    scales: dict = field(default_factory=dict)      # {'C': αC, 'H': αH, 'O': αO}

    def apply(self, atoms, energy, forces):
        a = np.array([self.scales.get(s, 1.0) for s in atoms.get_chemical_symbols()])
        return energy, forces * a[:, None]

    @classmethod
    def fit(cls, base_forces, ref_forces, symbols_list):
        """Per-element least-squares scaling."""
        acc = {}
        for fb, fr, syms in zip(base_forces, ref_forces, symbols_list):
            for i, s in enumerate(syms):
                bb, rr = fb[i], fr[i]
                d = acc.setdefault(s, [0.0, 0.0])
                d[0] += float(bb @ rr); d[1] += float(bb @ bb)
        return cls(scales={s: (num/den if den else 1.0) for s, (num, den) in acc.items()})


@dataclass
class Delta(ForceCorrection):
    """Residual (delta-learning) correction: adds g(atoms) -> (dE, dF) on top of
    the base. `model` is any callable atoms -> (dE_eV, dF_eV/Å). Stub by default."""
    model: object = None

    def apply(self, atoms, energy, forces):
        if self.model is None:
            return energy, forces
        de, df = self.model(atoms)
        return energy + de, forces + df


# --------------------------------------------------------------------------- #
#  Calculator wrapper — the single insertion point
# --------------------------------------------------------------------------- #
class CorrectedCalculator(Calculator):
    """ASE calculator that runs `base` then applies `correction`. Use anywhere a
    calculator is expected; sharing one instance across NEB images is safe
    (in-memory, recomputes on geometry change)."""
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, base, correction: ForceCorrection | None = None, **kw):
        super().__init__(**kw)
        self.base = base
        self.correction = correction or Identity()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = atoms.copy(); a.calc = self.base
        e = float(a.get_potential_energy()); f = a.get_forces()
        e, f = self.correction.apply(atoms, e, f)
        self.results["energy"] = e
        self.results["free_energy"] = e
        self.results["forces"] = f


# --------------------------------------------------------------------------- #
#  Convenience: fit any correction from two calculators over a set of structures
# --------------------------------------------------------------------------- #
def fit_correction(kind, base_calc, ref_calc, structures):
    """Evaluate both calculators on `structures` and fit a correction of type
    `kind` in {"global","affine","element"}. Returns the fitted ForceCorrection.
    (Provided for the later publication step; unused by the NEB benchmark.)"""
    bf, rf, be, re_, syms = [], [], [], [], []
    for at in structures:
        a = at.copy(); a.calc = base_calc
        bf.append(a.get_forces()); be.append(a.get_potential_energy())
        b = at.copy(); b.calc = ref_calc
        rf.append(b.get_forces()); re_.append(b.get_potential_energy())
        syms.append(at.get_chemical_symbols())
    if kind == "global":
        return GlobalScale.fit(bf, rf)
    if kind == "affine":
        return Affine.fit(bf, rf, be, re_)
    if kind == "element":
        return ElementScale.fit(bf, rf, syms)
    raise ValueError(kind)
