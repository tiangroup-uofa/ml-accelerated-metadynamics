#!/usr/bin/env python3
"""
analyze_large.py
================
Trajectory analysis for the larger vacuum water clusters (n=20/30/50),
tailored to the two July-10 questions:

  DROPLET vs EVAPORATION (step 1/3): does the free cluster stay a compact
  droplet, or do molecules keep peeling off to infinity like the tiny
  n=2..4 clusters did?  Reported with:
    - Rg            : radius of gyration of the O atoms (droplet size).
                      Stays ~constant for a stable droplet; grows without
                      bound if it evaporates / falls apart.
    - n_evaporated  : molecules whose O has NO other O within R_EVAP -- i.e.
                      monomers that have left the droplet surface.
    - frac_in_main  : fraction of waters in the largest O-O connected
                      cluster (1.0 = one intact droplet).

  BOND BREAKING / PROTON TRANSFER (step 2): as kpush rises, do covalent O-H
  bonds break or protons hop between molecules?  Reported with:
    - max_intra_OH  : longest H-to-nearest-O distance (dissociation if big).
    - n_transfer    : count of protons whose nearest O differs from the
                      START-frame nearest O = a proton that changed owner
                      (proton-transfer / autoionization signature). This is
                      the per-oxygen reassignment metric flagged as the
                      stronger reactive CV in the July-3 record.

Reuses the switching-function coordination idea from ../cv_analysis.py.
Vacuum runs => no minimum-image (mic=False); these clusters are free.

Usage:
    python analyze_large.py                         # all runs under runs/
    python analyze_large.py --run runs/n30/high
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "analysis"

DUMP_FS = 100.0
OH_BREAK = 1.30      # A: nearest O-H longer than this = a broken covalent bond
R_EVAP = 3.5         # A: no O within this of an O => that water has evaporated
R_CLUST = 3.5        # A: O-O within this => same connected droplet
R0_OO, SW_P, SW_Q = 3.2, 8, 14


def switch(r, r0=R0_OO, p=SW_P, q=SW_Q):
    x = r / r0
    x = np.where(np.isclose(x, 1.0), 1.0 + 1e-9, x)
    return (1.0 - x**p) / (1.0 - x**q)


def oh_owner(atoms, o_idx, h_idx):
    """Index (into o_idx) of the nearest O for each H."""
    owner = []
    for h in h_idx:
        ds = atoms.get_distances(h, o_idx, mic=False)
        owner.append(int(np.argmin(ds)))
    return np.array(owner)


def connected_fraction(oo):
    """Largest-connected-component fraction from an O-O distance matrix,
    edges where distance < R_CLUST (union-find)."""
    n = oo.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            if oo[a, b] < R_CLUST:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    sizes = {}
    for a in range(n):
        r = find(a)
        sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values()) / n


def frame_metrics(atoms, start_owner):
    sym = atoms.get_chemical_symbols()
    o_idx = [i for i, s in enumerate(sym) if s == "O"]
    h_idx = [i for i, s in enumerate(sym) if s == "H"]
    opos = atoms.get_positions()[o_idx]

    # radius of gyration of the oxygens
    com = opos.mean(axis=0)
    rg = float(np.sqrt(((opos - com) ** 2).sum(axis=1).mean()))

    # O-O distance matrix
    n = len(o_idx)
    oo = np.zeros((n, n))
    for a in range(n):
        ds = atoms.get_distances(o_idx[a], o_idx, mic=False)
        oo[a] = ds
    np.fill_diagonal(oo, np.inf)
    nn = oo.min(axis=1)                          # nearest-O per oxygen
    n_evap = int((nn > R_EVAP).sum())
    max_oo = float(oo[np.isfinite(oo)].max())
    frac_main = connected_fraction(oo)

    # coordination (mean H-bond-ish neighbours per O)
    with np.errstate(over="ignore"):
        mean_coord = float(switch(np.where(np.isfinite(oo), oo, 1e6)).sum(axis=1).mean())

    # O-H integrity + proton transfer (owner change vs start)
    owner = oh_owner(atoms, o_idx, h_idx)
    max_intra = 0.0
    for h in h_idx:
        max_intra = max(max_intra, float(atoms.get_distances(h, o_idx, mic=False).min()))
    n_transfer = int((owner != start_owner).sum()) if start_owner is not None else 0

    return dict(Rg=rg, max_OO_A=max_oo, n_evaporated=n_evap,
                frac_in_main=frac_main, mean_coordination=mean_coord,
                max_intra_OH_A=max_intra, n_transfer=n_transfer), owner


def analyze_run(run_dir: Path) -> pd.DataFrame | None:
    trj = run_dir / "xtb.trj"
    if not trj.exists():
        print(f"  (no xtb.trj in {run_dir}, skipping)")
        return None
    frames = read(str(trj), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]

    # proton ownership in the first frame = reference for transfer detection
    sym = frames[0].get_chemical_symbols()
    o0 = [i for i, s in enumerate(sym) if s == "O"]
    h0 = [i for i, s in enumerate(sym) if s == "H"]
    start_owner = oh_owner(frames[0], o0, h0)

    rows = []
    for fi, atoms in enumerate(frames):
        m, _ = frame_metrics(atoms, start_owner)
        m["frame"] = fi
        m["time_ps"] = fi * DUMP_FS / 1000.0
        rows.append(m)
    df = pd.DataFrame(rows)

    tag = f"{run_dir.parent.name}_{run_dir.name}"     # e.g. n30_high
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"cv_{tag}.csv"
    df.to_csv(out, index=False)

    last = df.iloc[-1]
    rg0, rg1 = df.Rg.iloc[0], last.Rg
    print(f"{tag}: {len(df)} frames")
    print(f"   Rg {rg0:.2f} -> {rg1:.2f} A ({'GREW' if rg1 > 1.2*rg0 else 'stable'})   "
          f"frac_in_main end={last.frac_in_main:.2f}  evaporated end={int(last.n_evaporated)}")
    print(f"   max intra O-H reached {df.max_intra_OH_A.max():.2f} A "
          f"({'DISSOCIATION' if df.max_intra_OH_A.max() >= OH_BREAK else 'no O-H broke'})   "
          f"max proton transfers {int(df.n_transfer.max())}")
    print(f"   -> {out}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None,
                    help="a single run dir (default: every runs/n*/<label>)")
    args = ap.parse_args()

    if args.run:
        analyze_run(args.run)
        return 0
    run_dirs = sorted(d for d in RUNS.glob("n*/*")
                      if d.is_dir() and d.name != "opt" and (d / "xtb.trj").exists())
    if not run_dirs:
        print("No finished runs found under runs/.")
        return 1
    for d in run_dirs:
        analyze_run(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
