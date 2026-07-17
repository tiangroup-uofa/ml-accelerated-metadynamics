#!/usr/bin/env python3
"""
analyze_sweep.py
================
Analyze every run of the (size x kpush) metadynamics sweep and quantify
how well each water cluster stayed intact under the metadynamics bias.

For each run we read the dense trajectory xtb.trj (every 100 fs) and, per
frame, compute:

  * n_fragments      -- number of separate pieces the cluster broke into,
                        using a H-bond/contact graph between waters
                        (two waters are "connected" if any O-O distance
                        is below O_O_CUTOFF, i.e. still H-bonded/touching)
  * intra_OH_ok      -- whether every water still has its 2 O-H bonds
                        (checks no water itself fell apart into H + OH)
  * spread_A         -- max O-O distance in the frame (how far it spread)

From the per-frame series we summarize each run:

  * n_waters, kpush
  * frac_intact      -- fraction of frames that were a SINGLE fragment
                        with all O-H bonds intact
  * final_fragments  -- n_fragments in the last frame
  * max_fragments    -- worst (largest) fragmentation seen
  * t_first_break_ps -- first time the cluster stopped being one intact
                        piece (NaN if it never broke)
  * max_spread_A     -- largest O-O distance reached

Outputs:
  analysis/per_run_summary.csv     one row per run (the headline table)
  analysis/timeseries_<n>_<label>.csv   per-frame series for each run
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "analysis"

KPUSH = {"low": 0.008, "med": 0.02, "high": 0.05}
SIZES = [1, 2, 3, 4]

DUMP_FS = 100.0                 # matches metadyn.inp dump interval
O_O_CUTOFF = 3.5               # A: two waters count as H-bonded/contacting
OH_BOND_MAX = 1.3             # A: an intact O-H bond is shorter than this


def water_fragments(o_idx, pos) -> int:
    """Number of connected components among waters, linking two waters
    whose oxygens are within O_O_CUTOFF."""
    n = len(o_idx)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[o_idx[i]] - pos[o_idx[j]]) <= O_O_CUTOFF:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def assign_hydrogens(o_idx, h_idx, pos):
    """Assign each H to its nearest O; return list of O-H distances per O."""
    oh_per_o = {oi: [] for oi in o_idx}
    for h in h_idx:
        d = [np.linalg.norm(pos[h] - pos[oi]) for oi in o_idx]
        nearest = o_idx[int(np.argmin(d))]
        oh_per_o[nearest].append(min(d))
    return oh_per_o


def all_oh_intact(o_idx, h_idx, pos) -> bool:
    """True if every water still holds exactly 2 H within OH_BOND_MAX."""
    oh = assign_hydrogens(o_idx, h_idx, pos)
    for oi, dists in oh.items():
        close = [d for d in dists if d <= OH_BOND_MAX]
        if len(close) != 2:
            return False
    return True


def analyze_run(n: int, label: str):
    run_dir = RUNS / f"n{n}" / label
    trj = run_dir / "xtb.trj"
    if not trj.exists():
        print(f"  MISSING: {trj}")
        return None, None

    # xTB's xtb.trj is multi-frame XYZ; tell ASE explicitly (the .trj
    # extension is NOT ASE's own binary .traj format).
    frames = read(str(trj), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]

    sym = frames[0].get_chemical_symbols()
    o_idx = [i for i, s in enumerate(sym) if s == "O"]
    h_idx = [i for i, s in enumerate(sym) if s == "H"]

    rows = []
    for fi, atoms in enumerate(frames):
        pos = atoms.get_positions()
        nfrag = water_fragments(o_idx, pos)
        oh_ok = all_oh_intact(o_idx, h_idx, pos)
        # spread: max O-O distance (0 for a monomer)
        if len(o_idx) >= 2:
            spread = max(
                np.linalg.norm(pos[o_idx[i]] - pos[o_idx[j]])
                for i in range(len(o_idx)) for j in range(i + 1, len(o_idx))
            )
        else:
            # monomer: use max O-H distance as its "spread"/dissociation proxy
            spread = max(np.linalg.norm(pos[o_idx[0]] - pos[h]) for h in h_idx)
        rows.append({
            "frame": fi,
            "time_ps": fi * DUMP_FS / 1000.0,
            "n_fragments": nfrag,
            "oh_intact": oh_ok,
            "spread_A": spread,
        })

    ts = pd.DataFrame(rows)

    # intact frame = single fragment AND all O-H bonds present
    # (for the monomer, "single fragment" is trivially true, so O-H is what
    #  matters -- it dissociates by losing an O-H, exactly your n=1 result)
    intact_mask = (ts["n_fragments"] == 1) & (ts["oh_intact"])
    frac_intact = float(intact_mask.mean())

    broken = ~intact_mask
    if broken.any():
        t_first_break = float(ts.loc[broken, "time_ps"].iloc[0])
    else:
        t_first_break = np.nan

    summary = {
        "n_waters": n,
        "label": label,
        "kpush": KPUSH[label],
        "n_frames": len(ts),
        "frac_intact": frac_intact,
        "final_fragments": int(ts["n_fragments"].iloc[-1]),
        "max_fragments": int(ts["n_fragments"].max()),
        "final_oh_intact": bool(ts["oh_intact"].iloc[-1]),
        "t_first_break_ps": t_first_break,
        "max_spread_A": float(ts["spread_A"].max()),
    }
    return summary, ts


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for n in SIZES:
        for label in KPUSH:
            summary, ts = analyze_run(n, label)
            if summary is None:
                continue
            ts.to_csv(OUT / f"timeseries_n{n}_{label}.csv", index=False)
            summaries.append(summary)
            print(f"  n={n} {label:4s} kpush={summary['kpush']:.3f}  "
                  f"frac_intact={summary['frac_intact']:.2f}  "
                  f"final_frags={summary['final_fragments']}  "
                  f"t_break={summary['t_first_break_ps']}")

    df = pd.DataFrame(summaries)
    df.to_csv(OUT / "per_run_summary.csv", index=False)
    print(f"\nWrote {len(df)} rows -> {OUT / 'per_run_summary.csv'}")
    print("\n" + df.to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
