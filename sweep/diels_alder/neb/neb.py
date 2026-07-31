#!/usr/bin/env python3
"""
neb.py
======
Calculator-agnostic **NEB reaction-path workflow** (ASE). The same code runs with
GFN2-xTB or MACE-OFF23 so the Diels–Alder comparison is method-vs-method with
identical methodology (xTB has a built-in `--path`; MACE does not, hence NEB).

Pipeline:
  1. optimize both endpoints with the chosen potential (reactant complex, product)
  2. build an IDPP-interpolated band and run **climbing-image NEB**
  3. extract minimum energy path, TS (highest image), barrier, reaction energy,
     and the TS forming-bond geometry (the DA reaction coordinate).

Atom convention (from build_diels_alder.py): C1..C6 = indices 0..5; the two new
σ-bonds are C1–C6 (0,5) and C4–C5 (3,4); their length is the reaction coordinate.
"""
from __future__ import annotations
import numpy as np
from ase.optimize import BFGS, FIRE
from ase.mep import NEB
from ase.constraints import FixBondLengths

EV2KCAL = 23.060548
FORMING = [(0, 5), (3, 4)]             # the two new σ-bonds (C1–C6, C4–C5)


def forming_bonds(atoms):
    p = atoms.get_positions()
    d = lambda i, j: float(np.linalg.norm(p[i] - p[j]))
    return d(0, 5), d(3, 4)            # C1–C6, C4–C5


def mean_forming(atoms):
    a, b = forming_bonds(atoms)
    return 0.5 * (a + b)


def optimize_endpoint(atoms, calc, fmax=0.03, steps=800, fix_bonds=None):
    """Relax an endpoint. `fix_bonds` (list of (i,j)) pins those interatomic
    distances — used on the REACTANT so the weakly-bound butadiene+ethylene
    pre-reaction complex relaxes internally without the fragments drifting apart
    (free optimization dissociates the vdW complex and ruins the NEB path)."""
    a = atoms.copy(); a.calc = calc
    if fix_bonds:
        a.set_constraint(FixBondLengths([list(p) for p in fix_bonds]))
    BFGS(a, logfile=None).run(fmax=fmax, steps=steps)
    a.set_constraint()
    return a, float(a.get_potential_energy())


def _subsample(band, k):
    """Pick k geometries evenly from `band` (a list of Atoms)."""
    idx = np.linspace(0, len(band) - 1, k).round().astype(int)
    return [band[i].copy() for i in idx]


def run_neb(reactant, product, make_calc, n_images=11, fmax=0.05,
            climb=True, steps=800, k=0.1, seed_band=None):
    """Optimized `reactant`/`product` -> converged CI-NEB band + energies (eV).

    If `seed_band` (a list of Atoms spanning reactant→product, e.g. the concerted
    relaxed-scan geometries) is given, the band is initialised from it instead of
    IDPP interpolation. IDPP fails for this concerted bond-forming reaction — it
    produces clashy interior images whose energy dominates the "barrier"; a
    physically sensible seed fixes that."""
    if seed_band is not None:
        interior = _subsample(seed_band, n_images)[1:-1]
        images = [reactant.copy()] + interior + [product.copy()]
    else:
        images = [reactant.copy()] + [reactant.copy() for _ in range(n_images - 2)] \
                 + [product.copy()]
    for img in images:                 # every image needs a calc (energy readout)
        img.calc = make_calc()
    neb = NEB(images, climb=climb, k=k, method="improvedtangent",
              allow_shared_calculator=False)
    if seed_band is None:
        neb.interpolate(method="idpp")
    FIRE(neb, logfile=None).run(fmax=fmax, steps=steps)
    energies = np.array([img.get_potential_energy() for img in images])
    return images, energies


def analyze(images, energies):
    """MEP relative energies, TS, barrier, reaction energy, TS geometry.

    A `physical` mask drops spurious images an NEB can throw on a steep exothermic
    path — a fused geometry (forming C–C < 1.3 Å) or an absurd energy (|ΔE| > 400
    kcal/mol) — so they don't masquerade as the TS. Endpoints are always kept."""
    rel = (energies - energies[0]) * EV2KCAL
    coord = np.array([mean_forming(im) for im in images])
    physical = (coord > 1.3) & (np.abs(rel) < 400)
    physical[0] = physical[-1] = True
    masked = np.where(physical, rel, -np.inf)
    ts_i = int(np.argmax(masked))
    return dict(
        rel_kcal=rel,
        coord_A=coord,
        physical=physical,
        ts_index=ts_i,
        barrier_kcal=float(rel[ts_i]),
        reaction_energy_kcal=float(rel[-1]),
        ts_forming_A=forming_bonds(images[ts_i]),
        n_images=len(images),
        n_dropped=int((~physical).sum()),
    )


def full_workflow(reactant, product, make_calc, n_images=15,
                  fmax=0.05, opt_fmax=0.03, seed_band=None, k=1.0):
    """Optimize endpoints, run CI-NEB, analyze. Returns (images, result dict).
    `seed_band`: optional list of Atoms to initialise the band (recommended).
    `k`: NEB spring constant — must be stiff enough (≈1) that images don't slide
    off the barrier into the very exothermic product basin (they do at k≈0.1)."""
    # reactant: keep the pre-reaction complex compact (pin the forming bonds);
    # product (cyclohexene) is a genuine minimum, relax freely.
    r_opt, _ = optimize_endpoint(reactant, make_calc(), fmax=opt_fmax, fix_bonds=FORMING)
    p_opt, _ = optimize_endpoint(product, make_calc(), fmax=opt_fmax)
    images, energies = run_neb(r_opt, p_opt, make_calc, n_images=n_images,
                               fmax=fmax, seed_band=seed_band, k=k)
    res = analyze(images, energies)
    res["reactant_forming_A"] = forming_bonds(r_opt)
    res["product_forming_A"] = forming_bonds(p_opt)
    return images, res
