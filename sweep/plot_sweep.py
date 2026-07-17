#!/usr/bin/env python3
"""
plot_sweep.py
=============
Make the summary figures for the (size x kpush) metadynamics sweep.

Figures (into figures/):
  1. intact_heatmap.png       fraction of time each cluster stayed intact,
                              as a size x kpush grid. THE headline plot.
  2. fragments_vs_time.png    n_fragments over time, one panel per cluster
                              size, one line per kpush. Shows *when* and how
                              badly each run broke up.
  3. spread_vs_time.png       max O-O (or O-H for monomer) distance vs time,
                              same panel layout -- the "flying apart" signal.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANA = HERE / "analysis"
FIG = HERE / "figures"

KPUSH = {"low": 0.008, "med": 0.02, "high": 0.05}
SIZES = [1, 2, 3, 4]
COLORS = {"low": "tab:green", "med": "tab:orange", "high": "tab:red"}


def load_summary() -> pd.DataFrame:
    return pd.read_csv(ANA / "per_run_summary.csv")


def load_ts(n, label):
    p = ANA / f"timeseries_n{n}_{label}.csv"
    return pd.read_csv(p) if p.exists() else None


def plot_heatmap(df: pd.DataFrame) -> None:
    labels = list(KPUSH.keys())
    grid = np.full((len(SIZES), len(labels)), np.nan)
    for r, n in enumerate(SIZES):
        for c, lab in enumerate(labels):
            sub = df[(df.n_waters == n) & (df.label == lab)]
            if len(sub):
                grid[r, c] = sub["frac_intact"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                   origin="upper")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{lab}\nkpush={KPUSH[lab]}" for lab in labels])
    ax.set_yticks(range(len(SIZES)))
    ax.set_yticklabels([f"(H2O){n}" for n in SIZES])
    ax.set_xlabel("metadynamics bias strength")
    ax.set_ylabel("cluster size")
    ax.set_title("Fraction of trajectory the cluster stayed intact\n"
                 "(1 fragment, all O-H bonds present)")
    for r in range(len(SIZES)):
        for c in range(len(labels)):
            if not np.isnan(grid[r, c]):
                ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="fraction intact")
    fig.tight_layout()
    fig.savefig(FIG / "intact_heatmap.png", dpi=150)
    plt.close(fig)


def panel_plot(column, ylabel, title, fname, hline=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, n in zip(axes.flat, SIZES):
        for lab in KPUSH:
            ts = load_ts(n, lab)
            if ts is None:
                continue
            ax.plot(ts["time_ps"], ts[column], color=COLORS[lab],
                    label=f"{lab} (kpush={KPUSH[lab]})", lw=1.4)
        if hline is not None:
            ax.axhline(hline, ls="--", color="gray", lw=0.8)
        ax.set_title(f"(H2O){n}")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (ps)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    plot_heatmap(df)
    panel_plot("n_fragments", "number of fragments",
               "Cluster fragmentation vs time (by size and bias strength)",
               "fragments_vs_time.png", hline=1)
    panel_plot("spread_A", "max O-O distance (A)  [O-H for monomer]",
               "Cluster spread vs time (by size and bias strength)",
               "spread_vs_time.png")
    print(f"Wrote figures -> {FIG}")
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
