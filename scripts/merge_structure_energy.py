#!/usr/bin/env python3
"""
merge_structure_energy.py
=========================
Associate each ~1 ps structural snapshot (collective_variables.csv) with
the nearest-in-time potential-energy record (epot.csv).

The structure snapshots (scoord.*) and the energy table in output.log
are generally NOT written at exactly the same times or frequency, so we
match on the nearest timestamp using ``pandas.merge_asof`` with a
configurable tolerance instead of assuming index alignment.

Output
------
    trajectory_analysis.csv

Usage
-----
    python merge_structure_energy.py \
        --cvs ../analysis/collective_variables.csv \
        --epot ../analysis/epot.csv \
        --out ../analysis/trajectory_analysis.csv \
        --tolerance-ps 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cvs", default="../analysis/collective_variables.csv", type=Path)
    parser.add_argument("--epot", default="../analysis/epot.csv", type=Path)
    parser.add_argument("--out", default="../analysis/trajectory_analysis.csv", type=Path)
    parser.add_argument("--tolerance-ps", type=float, default=0.5,
                        help="Maximum |time difference| (ps) allowed when "
                             "matching a structure to an energy record.")
    args = parser.parse_args()

    for p in (args.cvs, args.epot):
        if not p.resolve().exists():
            print(f"ERROR: required input {p.resolve()} not found.", file=sys.stderr)
            return 1

    cvs = pd.read_csv(args.cvs)
    epot = pd.read_csv(args.epot)

    if "approx_time_ps" not in cvs.columns:
        print("ERROR: 'approx_time_ps' missing from CV file.", file=sys.stderr)
        return 1
    if "time_ps" not in epot.columns:
        print("ERROR: 'time_ps' missing from epot file.", file=sys.stderr)
        return 1

    # merge_asof requires both keys sorted ascending.
    cvs_sorted = cvs.sort_values("approx_time_ps").reset_index(drop=True)
    epot_sorted = epot.sort_values("time_ps").reset_index(drop=True)

    merged = pd.merge_asof(
        cvs_sorted,
        epot_sorted,
        left_on="approx_time_ps",
        right_on="time_ps",
        direction="nearest",
        tolerance=args.tolerance_ps,
    )

    # Rows where no energy record fell within tolerance: energy cols NaN
    unmatched = merged[merged["epot_hartree"].isna()]
    if not unmatched.empty:
        print(f"WARNING: {len(unmatched)} structure(s) had no energy record "
              f"within {args.tolerance_ps} ps:")
        for _, row in unmatched.iterrows():
            print(f"  frame {int(row['frame'])} "
                  f"({row['source_file']}) at "
                  f"{row['approx_time_ps']:.3f} ps")
    else:
        print("All structures matched to an energy record within tolerance.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Wrote {len(merged)} rows -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
