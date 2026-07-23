#!/usr/bin/env python3
"""
derive_bias_scaling.py
======================
Why the same `kpush` is "gentler" for a bigger cluster, and how to fix it
(Tian's per-molecule/atom `kpush` question).

xTB's RMSD metadynamics bias is a sum of Gaussians in Cartesian RMSD:

    V(x) = Σ_ref  kpush · exp(-alp · RMSD(x, x_ref)²)

with the **root-MEAN-square** deviation  RMSD² = (1/N) Σ_i |x_i - x_i,ref|²
(N = number of atoms). The bias force on atom i is therefore

    F_i = -∂V/∂x_i = kpush · alp · exp(-alp·RMSD²) · (2/N) · (x_i - x_i,ref)

so the **per-atom bias force carries an explicit 1/N** — the same nominal `kpush`
pushes each atom 1/N as hard in a larger system. To keep the per-atom bias effect
constant across sizes, `kpush` must scale **∝ N** (per-atom normalization); the
per-molecule version is the same with N = 3·(#waters).

This script demonstrates the law numerically on real water-cluster geometries
(random perturbations at a *fixed* RMSD, so the exp() factor is identical across
sizes and only the 1/N gradient term varies), for a fixed `kpush` vs an N-scaled
`kpush`. Output: figures/kpush_scaling_law.png.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(parents=True, exist_ok=True)

ALP = 1.2          # xTB default bias width
KPUSH = 0.02       # nominal
D0 = 0.3           # fixed RMSD (Å) between ref and perturbed structure
OH, HOH = 0.9584, np.radians(104.45 / 2)


def water(origin, rng):
    o = np.array([0., 0., 0.])
    h1 = np.array([OH*np.sin(HOH), 0, OH*np.cos(HOH)])
    h2 = np.array([-OH*np.sin(HOH), 0, OH*np.cos(HOH)])
    p = np.array([o, h1, h2])
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return p @ R.T + origin


def blob(nmol, rng, spacing=3.2):
    m = int(np.ceil(nmol ** (1/3)))
    pts = [np.array([i, j, k], float)*spacing
           for i in range(m) for j in range(m) for k in range(m)]
    return np.concatenate([water(pts[a], rng) for a in range(nmol)])


def per_atom_bias_force(ref, alp, kpush):
    """Mean per-atom bias-force magnitude for a perturbation of fixed RMSD D0."""
    rng = np.random.default_rng(0)
    N = len(ref)
    d = rng.normal(size=ref.shape)
    d *= D0 / np.sqrt((d**2).sum(1).mean())      # scale so RMSD == D0 exactly
    x = ref + d
    rmsd2 = (( (x-ref)**2 ).sum(1)).mean()
    Fi = kpush * alp * np.exp(-alp*rmsd2) * (2.0/N) * (x-ref)
    return np.linalg.norm(Fi, axis=1).mean()


def main() -> int:
    rng = np.random.default_rng(1)
    nmols = [1, 2, 4, 8, 16, 32]
    Ns, F_fixed, F_scaled = [], [], []
    Nref = 3  # atoms in the smallest (1 water)
    for nm in nmols:
        ref = blob(nm, rng); N = len(ref); Ns.append(N)
        F_fixed.append(per_atom_bias_force(ref, ALP, KPUSH))
        F_scaled.append(per_atom_bias_force(ref, ALP, KPUSH * N / Nref))
    Ns = np.array(Ns); F_fixed = np.array(F_fixed); F_scaled = np.array(F_scaled)

    print("N_atoms  per-atom |F| fixed kpush   N-scaled kpush")
    for N, a, b in zip(Ns, F_fixed, F_scaled):
        print(f"  {N:3d}      {a:.3e}            {b:.3e}")
    # confirm the 1/N law: F_fixed * N should be ~constant
    print(f"\nF_fixed × N (should be ~constant): "
          f"{np.round(F_fixed*Ns/(F_fixed[0]*Ns[0]),3)}")
    print(f"N-scaled |F| / first (should be ~1): "
          f"{np.round(F_scaled/F_scaled[0],3)}")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(Ns, F_fixed/F_fixed[0], "o-", color="#f43f5e",
              label="fixed kpush  (∝ 1/N — diluted)")
    ax.loglog(Ns, F_scaled/F_scaled[0], "s-", color="#2dd4bf",
              label="N-scaled kpush  (constant)")
    ax.loglog(Ns, Ns[0]/Ns, ":", color="#94a3b8", label="1/N reference")
    ax.set_xlabel("number of atoms N"); ax.set_ylabel("per-atom bias force (rel.)")
    ax.set_title("RMSD-bias per-atom force vs system size")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIG / "kpush_scaling_law.png", dpi=140)
    print(f"\nwrote {FIG/'kpush_scaling_law.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
