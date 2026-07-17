#!/usr/bin/env python3
"""
cv_analysis.py
==============
Collective-variable analysis for N-water metadynamics that DISAMBIGUATES
the two ways a large O-O distance can arise:

    (a) DISSOCIATION      -- a covalent O-H bond inside a single water breaks
    (b) DRIFTING APART    -- intact waters separate (evaporation)

This directly answers the question: "max O-O distance is large -- is it
because a water dissociated, or because the molecules drifted apart?"

Per frame we compute three complementary CVs:

  1. max_intra_OH_A   -- the LONGEST intramolecular O-H bond (each H assigned
                         to its nearest O = its own molecule). Stays ~1.0 A
                         if all molecules are intact; grows only on
                         DISSOCIATION.
  2. max_OO_A         -- the LARGEST O-O distance = overall spread. Grows on
                         BOTH dissociation and drifting -- ambiguous alone.
  3. mean_coordination -- average number of hydrogen bonds per water, via a
                         smooth switching function on O...O contacts. This is
                         the principled structural CV; it decreases smoothly
                         as the H-bond network falls apart.

Classification per frame (the disambiguation):
    intact         : max_intra_OH < OH_BREAK and max_OO < DRIFT
    drifting_apart : max_intra_OH < OH_BREAK and max_OO >= DRIFT
    dissociated    : max_intra_OH >= OH_BREAK          (a bond actually broke)

Outputs (into analysis/):
    cv_<tag>.csv         per-frame CVs + classification
    (tag defaults to the run directory name)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

DUMP_FS = 100.0
OH_BREAK = 1.30       # A: an O-H longer than this = dissociation
DRIFT = 3.5           # A: O-O beyond this (with O-H intact) = drifted apart
# switching function for coordination number: n(r) = (1-(r/r0)^p)/(1-(r/r0)^q)
R0_OO = 3.2           # A: characteristic H-bonded O...O distance
SW_P, SW_Q = 8, 14


def switch(r, r0=R0_OO, p=SW_P, q=SW_Q):
    """Smooth 1->0 switching function, ~1 for r<<r0, ~0 for r>>r0."""
    x = (r / r0)
    # guard x==1 (removable singularity -> limit p/q)
    x = np.where(np.isclose(x, 1.0), 1.0 + 1e-9, x)
    return (1.0 - x**p) / (1.0 - x**q)


def frame_cvs(atoms, mic=False):
    """Per-frame CVs. If mic=True (periodic run), all distances use the
    minimum-image convention against atoms.cell -- REQUIRED for PBC
    trajectories, otherwise atoms wrapped across the box give spurious
    huge O-H / O-O distances. Note xTB's xtb.trj does NOT store the cell,
    so the caller must attach it (see analyze(..., cell=...))."""
    sym = atoms.get_chemical_symbols()
    o_idx = [i for i, s in enumerate(sym) if s == "O"]
    h_idx = [i for i, s in enumerate(sym) if s == "H"]

    # intramolecular O-H: each H to its nearest O (min-image if periodic)
    max_intra = 0.0
    for h in h_idx:
        ds = atoms.get_distances(h, o_idx, mic=mic)
        max_intra = max(max_intra, float(ds.min()))

    if len(o_idx) >= 2:
        max_oo = 0.0
        coord_per_o = np.zeros(len(o_idx))
        for a in range(len(o_idx)):
            ds = atoms.get_distances(o_idx[a], o_idx, mic=mic)
            for b, r in enumerate(ds):
                if b == a:
                    continue
                max_oo = max(max_oo, float(r))
                coord_per_o[a] += switch(r)
        mean_coord = float(coord_per_o.mean())
    else:
        max_oo = 0.0
        mean_coord = 0.0

    return max_intra, max_oo, mean_coord


def classify(max_intra, max_oo):
    if max_intra >= OH_BREAK:
        return "dissociated"
    if max_oo >= DRIFT:
        return "drifting_apart"
    return "intact"


def analyze(trj: Path, tag: str, out_dir: Path, cell=None):
    """cell: optional [a,b,c] (Angstrom) of a cubic periodic box. When given,
    it is attached to every frame and all distances use minimum image (for
    PBC runs, whose xtb.trj does not store the cell)."""
    frames = read(str(trj), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    mic = cell is not None
    rows = []
    for fi, atoms in enumerate(frames):
        if mic:
            atoms.set_cell(cell)
            atoms.set_pbc(True)
        mi, mo, mc = frame_cvs(atoms, mic=mic)
        rows.append({
            "frame": fi,
            "time_ps": fi * DUMP_FS / 1000.0,
            "max_intra_OH_A": mi,
            "max_OO_A": mo,
            "mean_coordination": mc,
            "state": classify(mi, mo),
        })
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cv_{tag}.csv"
    df.to_csv(out, index=False)

    counts = df["state"].value_counts().to_dict()
    print(f"{tag}: {len(df)} frames  states={counts}")
    print(f"   max intramolecular O-H reached: {df.max_intra_OH_A.max():.2f} A "
          f"({'DISSOCIATION seen' if df.max_intra_OH_A.max() >= OH_BREAK else 'no bond broke'})")
    print(f"   max O-O reached:                {df.max_OO_A.max():.2f} A")
    print(f"   wrote {out}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trj", required=True, type=Path,
                    help="path to an xtb.trj")
    ap.add_argument("--tag", default=None,
                    help="label for output file (default: parent dir name)")
    ap.add_argument("--out-dir", default=Path(__file__).resolve().parent / "analysis",
                    type=Path)
    ap.add_argument("--cell", type=float, default=None,
                    help="cubic box edge in Angstrom for a PERIODIC run; "
                         "enables minimum-image distances (xtb.trj has no cell)")
    args = ap.parse_args()
    tag = args.tag or args.trj.parent.name
    cell = [args.cell] * 3 if args.cell else None
    analyze(args.trj, tag, args.out_dir, cell=cell)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
