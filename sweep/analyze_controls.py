#!/usr/bin/env python3
"""
analyze_controls.py
===================
Analyze the controls + independent-initial-configuration replicas grid
(run_controls.py) with connectivity-based fragmentation metrics.

METHODOLOGICAL NOTES (why the metrics are defined this way):

* Fragmentation is judged by CONNECTIVITY, not by max O-O distance. For an
  8-water cluster the two most-distant oxygens are naturally > 3.5 A apart
  even when the whole thing is one connected blob, so "max O-O > 3.5 A" is
  NOT a fragmentation signal. We build an O-O contact graph (oxygens within
  OO_CONTACT of each other are in the same fragment) and report:
      n_fragments        number of connected water sub-clusters
      largest_fragment   size (in molecules) of the biggest sub-cluster
  max_OO_A is kept only as a DESCRIPTIVE observable (overall spread).

* "Coordination" here is OXYGEN-NEIGHBOUR coordination: the number of other
  oxygens within a smooth O-O switching function. It is NOT a hydrogen-bond
  count because no O-H...O angular criterion is applied. Named accordingly.

* Dissociation / proton transfer is detected with PER-OXYGEN O-H coordination:
  for each O, count H atoms within OH_BOND of THAT oxygen (not "nearest O").
  An intact water has 2; hydronium (H3O+) has 3; hydroxide (OH-) has 1. This
  reveals proton transfer / autoionization that a nearest-O assignment would
  hide. We report the fraction of frames in which every O has exactly 2 H
  (all-neutral-water), and flag frames with any O carrying 1 or 3 H.

* The 3 runs per cell are INDEPENDENT INITIAL-CONFIGURATION REPLICAS
  (different equilibrated starting boxes), aggregated as mean +/- std.

Outputs:
  analysis/controls_per_run.csv    one row per run
  analysis/controls_summary.csv    per-cell mean +/- std across replicas
  figures/controls_3panel.png      largest-fragment, all-water fraction proxy,
                                    and O-neighbour coordination vs time
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
OUT = HERE / "controls"
FIG = HERE / "figures"
ANA = HERE / "analysis"

SEEDS = [0, 1, 2]
BIASES = [0.0, 0.008]
BOUNDARIES = ["gas", "pbc"]
DUMP_FS = 100.0

OH_BOND = 1.30        # A: an H is "on" an O if within this distance
OO_CONTACT = 3.5      # A: two waters are in the same fragment if O-O < this
R0_OO = 3.2           # A: O-neighbour switching-function characteristic dist
ANG2BOHR = 1.8897259886

CELL_STYLE = {
    ("gas", 0.0):   dict(color="tab:green",  ls="--", label="gas, unbiased"),
    ("gas", 0.008): dict(color="tab:red",    ls="-",  label="gas, kpush=0.008"),
    ("pbc", 0.0):   dict(color="tab:blue",   ls="--", label="PBC, unbiased"),
    ("pbc", 0.008): dict(color="tab:purple", ls="-",  label="PBC, kpush=0.008"),
}


def switch(r, r0=R0_OO, p=8, q=14):
    x = r / r0
    x = np.where(np.isclose(x, 1.0), 1.0 + 1e-9, x)
    return (1.0 - x**p) / (1.0 - x**q)


def box_edge(rdir: Path):
    coord = rdir / "input.coord"
    if not coord.exists():
        return None
    for l in coord.read_text().splitlines():
        p = l.split()
        if len(p) == 6 and "." in p[0] and "periodic" not in l:
            return float(p[0]) / ANG2BOHR
    return None


def fragments(o_idx, atoms, mic):
    """Connected components of the O-O contact graph. Returns
    (n_fragments, largest_fragment_size_in_molecules)."""
    n = len(o_idx)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        ds = atoms.get_distances(o_idx[i], o_idx, mic=mic)
        for j in range(n):
            if i != j and ds[j] <= OO_CONTACT:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    sizes = {}
    for r in roots:
        sizes[r] = sizes.get(r, 0) + 1
    return len(set(roots)), max(sizes.values())


def per_oxygen_OH(o_idx, h_idx, atoms, mic):
    """For each O, count H within OH_BOND of THAT oxygen. Returns the list
    of counts (one per O). Intact water -> 2, H3O+ -> 3, OH- -> 1."""
    counts = []
    for o in o_idx:
        ds = atoms.get_distances(o, h_idx, mic=mic)
        counts.append(int((ds <= OH_BOND).sum()))
    return counts


def series(rdir: Path):
    trj = rdir / "xtb.trj"
    if not trj.exists():
        return None
    frames = read(str(trj), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    edge = box_edge(rdir)
    mic = edge is not None
    rows = []
    for fi, a in enumerate(frames):
        if mic:
            a.set_cell([edge] * 3)
            a.set_pbc(True)
        sym = a.get_chemical_symbols()
        o = [i for i, s in enumerate(sym) if s == "O"]
        h = [i for i, s in enumerate(sym) if s == "H"]

        nfrag, largest = fragments(o, a, mic)

        # per-oxygen O-H coordination
        ohc = per_oxygen_OH(o, h, a, mic)
        all_water = all(c == 2 for c in ohc)
        n_charged = sum(1 for c in ohc if c != 2)   # O with 1 or 3 H

        # descriptive: overall spread (NOT a fragmentation criterion)
        max_oo = 0.0
        cper = np.zeros(len(o))
        for ai in range(len(o)):
            ds = a.get_distances(o[ai], o, mic=mic)
            for bi, r in enumerate(ds):
                if bi == ai:
                    continue
                max_oo = max(max_oo, float(r))
                cper[ai] += switch(r)
        o_neighbour_coord = float(cper.mean())

        rows.append(dict(
            time_ps=fi * DUMP_FS / 1000.0,
            n_fragments=nfrag,
            largest_fragment=largest,
            all_water=all_water,
            n_charged_O=n_charged,
            max_OO_A=max_oo,                 # descriptive only
            o_neighbour_coord=o_neighbour_coord,
        ))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, n_mol: int) -> dict:
    return dict(
        frac_single_fragment=float((df.n_fragments == 1).mean()),
        mean_largest_fragment=float(df.largest_fragment.mean()),
        final_n_fragments=int(df.n_fragments.iloc[-1]),
        frac_all_water=float(df.all_water.mean()),
        frac_any_charged=float((df.n_charged_O > 0).mean()),
        mean_o_neighbour_coord=float(df.o_neighbour_coord.mean()),
        max_OO_A=float(df.max_OO_A.max()),          # descriptive
    )


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    ANA.mkdir(parents=True, exist_ok=True)

    per_run = []
    cell_series = {}
    for boundary in BOUNDARIES:
        for bias in BIASES:
            dfs = []
            for s in SEEDS:
                rdir = OUT / f"{boundary}_k{bias}_s{s}"
                df = series(rdir)
                if df is None:
                    continue
                dfs.append(df)
                row = dict(boundary=boundary, bias=bias, replica=s)
                row.update(summarize(df, n_mol=8))
                per_run.append(row)
            if dfs:
                cell_series[(boundary, bias)] = dfs

    pr = pd.DataFrame(per_run)
    if pr.empty:
        print("No runs found. Run run_controls.py first.")
        return 1

    agg = (pr.groupby(["boundary", "bias"])
             .agg(frac_single_fragment_mean=("frac_single_fragment", "mean"),
                  frac_single_fragment_std=("frac_single_fragment", "std"),
                  mean_largest_fragment_mean=("mean_largest_fragment", "mean"),
                  mean_largest_fragment_std=("mean_largest_fragment", "std"),
                  frac_all_water_mean=("frac_all_water", "mean"),
                  frac_any_charged_mean=("frac_any_charged", "mean"),
                  o_neighbour_coord_mean=("mean_o_neighbour_coord", "mean"),
                  max_OO_A_mean=("max_OO_A", "mean"),
                  n_replicas=("replica", "count"))
             .reset_index())
    pr.to_csv(ANA / "controls_per_run.csv", index=False)
    agg.to_csv(ANA / "controls_summary.csv", index=False)
    print("Per-cell mean across independent-configuration replicas:\n")
    print(agg.to_string(index=False))

    # ---- 3-panel figure ----
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    panels = [
        ("largest_fragment", "largest connected fragment (molecules)", False),
        ("n_fragments", "number of fragments", False),
        ("o_neighbour_coord", "O-neighbour coordination", False),
    ]
    for ax, (col, ylab, logy) in zip(axes, panels):
        for key, dfs in cell_series.items():
            n = min(len(d) for d in dfs)
            t = dfs[0].time_ps.values[:n]
            stack = np.vstack([d[col].values[:n] for d in dfs])
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)
            st = CELL_STYLE[key]
            ax.plot(t, mean, color=st["color"], ls=st["ls"], lw=1.7,
                    label=st["label"])
            ax.fill_between(t, mean - std, mean + std, color=st["color"],
                            alpha=0.15)
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, which="both")
    axes[0].axhline(8, ls=":", color="gray", lw=1)
    axes[0].text(0.2, 8.05, "fully connected (8 molecules)", fontsize=8,
                 color="gray")
    axes[-1].set_xlabel("time (ps)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title("8-water system: gas vs PBC, biased vs unbiased\n"
                      "(mean of 3 independent-configuration replicas, "
                      "+/- std band)")
    fig.tight_layout()
    fig.savefig(FIG / "controls_3panel.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {FIG / 'controls_3panel.png'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
