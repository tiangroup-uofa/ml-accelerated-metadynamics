#!/usr/bin/env python3
"""
plot_analysis.py
================
Produce the analysis figures from ``trajectory_analysis.csv``.

All energies plotted are RELATIVE potential energies (relative to the
minimum sampled Epot). These are NOT free energies: the trajectory is
biased by metadynamics, so Epot projected onto a structural coordinate
is a *sampled potential-energy landscape*, not an unbiased free-energy
surface. Recovering a free-energy surface would require reweighting the
bias, which is outside this workflow.

Figures (written to --fig-dir, default ../figures)
--------------------------------------------------
    rmsd_vs_time.png
    distances_vs_time.png
    distance_difference_vs_time.png
    dihedral_vs_time.png
    epot_vs_time.png
    epot_vs_rmsd.png
    epot_vs_cv.png
    structural_map_2d.png        (distance_1 vs distance_2, coloured by Epot)
    energy_landscape_2d.png      (binned MIN relative Epot over two CVs)

Usage
-----
    python plot_analysis.py \
        --data ../analysis/trajectory_analysis.csv \
        --fig-dir ../figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / HPC-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300
FIGSIZE = (7.0, 5.0)
ENERGY_COL = "epot_relative_kj_mol"
ENERGY_LABEL = "Relative $E_\\mathrm{pot}$ (kJ/mol)"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {path}")


def line_plot(df, x, y, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df[x], df[y], marker="o", ms=4, lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def scatter_energy(df, x, xlabel, title, path):
    sub = df.dropna(subset=[x, ENERGY_COL])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sc = ax.scatter(sub[x], sub[ENERGY_COL], c=sub["approx_time_ps"],
                    cmap="viridis", s=40, edgecolor="k", linewidth=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Approx. time (ps)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ENERGY_LABEL)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="../analysis/trajectory_analysis.csv", type=Path)
    parser.add_argument("--fig-dir", default="../figures", type=Path)
    parser.add_argument("--bins", type=int, default=15,
                        help="Number of bins per axis for the 2D landscape.")
    args = parser.parse_args()

    data_path = args.data.resolve()
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run merge_structure_energy.py first.",
              file=sys.stderr)
        return 1

    df = pd.read_csv(data_path)
    fig_dir = args.fig_dir.resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plotting {len(df)} frames -> {fig_dir}")

    t = "approx_time_ps"

    # 1. RMSD vs time
    line_plot(df, t, "rmsd_A", "Approx. time (ps)", "RMSD from frame 0 ($\\AA$)",
              "Cartesian RMSD vs time", fig_dir / "rmsd_vs_time.png")

    # 2. Bond distances vs time (both on one axis)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df[t], df["distance_1_A"], marker="o", ms=4, lw=1.2, label="distance 1")
    ax.plot(df[t], df["distance_2_A"], marker="s", ms=4, lw=1.2, label="distance 2")
    ax.set_xlabel("Approx. time (ps)")
    ax.set_ylabel("Distance ($\\AA$)")
    ax.set_title("Selected bond distances vs time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, fig_dir / "distances_vs_time.png")

    # 3. Distance-difference CV vs time
    line_plot(df, t, "distance_difference_A", "Approx. time (ps)",
              "$d_1 - d_2$ ($\\AA$)", "Distance-difference CV vs time",
              fig_dir / "distance_difference_vs_time.png")

    # 4. Dihedral vs time (only if present)
    if df["dihedral_deg"].notna().any():
        line_plot(df, t, "dihedral_deg", "Approx. time (ps)",
                  "Dihedral (deg)", "Dihedral angle vs time",
                  fig_dir / "dihedral_vs_time.png")
    else:
        print("  (skipping dihedral plot: no dihedral data)")

    # 4b. Fragmentation-diagnostic time series (only if columns exist)
    if "radius_of_gyration_A" in df.columns:
        line_plot(df, t, "radius_of_gyration_A", "Approx. time (ps)",
                  "Radius of gyration ($\\AA$)", "Radius of gyration vs time",
                  fig_dir / "radius_of_gyration_vs_time.png")
    if "max_pair_distance_A" in df.columns:
        line_plot(df, t, "max_pair_distance_A", "Approx. time (ps)",
                  "Max pairwise distance ($\\AA$)",
                  "Maximum pairwise atomic distance vs time",
                  fig_dir / "max_pair_distance_vs_time.png")
    if "estimated_components" in df.columns:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.step(df[t], df["estimated_components"], where="mid", marker="o", ms=4)
        ax.set_xlabel("Approx. time (ps)")
        ax.set_ylabel("Estimated connected components")
        ax.set_title("Estimated components vs time (heuristic)")
        ax.grid(True, alpha=0.3)
        _save(fig, fig_dir / "estimated_components_vs_time.png")

    # 5-7. Energy projections (only if energy present)
    if ENERGY_COL in df.columns and df[ENERGY_COL].notna().any():
        line_plot(df, t, ENERGY_COL, "Approx. time (ps)", ENERGY_LABEL,
                  "Relative potential energy vs time (biased trajectory)",
                  fig_dir / "epot_vs_time.png")
        scatter_energy(df, "rmsd_A", "RMSD from frame 0 ($\\AA$)",
                       "Sampled $E_\\mathrm{pot}$ vs RMSD (biased)",
                       fig_dir / "epot_vs_rmsd.png")
        scatter_energy(df, "distance_difference_A", "$d_1 - d_2$ ($\\AA$)",
                       "Sampled $E_\\mathrm{pot}$ vs distance-difference CV (biased)",
                       fig_dir / "epot_vs_cv.png")
    else:
        print("  (skipping energy plots: no energy column populated)")

    # 8. 2D structural map: d1 vs d2 coloured by energy
    if ENERGY_COL in df.columns and df[ENERGY_COL].notna().any():
        sub = df.dropna(subset=["distance_1_A", "distance_2_A", ENERGY_COL])
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sc = ax.scatter(sub["distance_1_A"], sub["distance_2_A"],
                        c=sub[ENERGY_COL], cmap="plasma", s=50,
                        edgecolor="k", linewidth=0.3)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(ENERGY_LABEL)
        ax.set_xlabel("distance 1 ($\\AA$)")
        ax.set_ylabel("distance 2 ($\\AA$)")
        ax.set_title("2D structural map (coloured by relative $E_\\mathrm{pot}$)")
        ax.grid(True, alpha=0.3)
        _save(fig, fig_dir / "structural_map_2d.png")

        # 9. OPTIONAL binned MIN-energy landscape over two CVs (d1, d2).
        #    The scatter map above is the PRIMARY 2D plot; this binned
        #    surface is only generated when sampling is dense enough that
        #    most bins are actually populated. With ~20 scoord.* frames a
        #    fixed 15x15 grid would be almost entirely empty and misleading.
        x = sub["distance_1_A"].to_numpy()
        y = sub["distance_2_A"].to_numpy()
        e = sub[ENERGY_COL].to_numpy()
        npts = len(sub)
        # Bins scale with sample size: ~sqrt(N) per axis, capped by user.
        bins = max(3, min(args.bins, int(np.sqrt(npts))))
        # Require several times more points than total bins so the grid is
        # not dominated by empty cells.
        if npts >= 4 * (bins * bins):
            stat, xedges, yedges = _binned_min(x, y, e, bins)
            fig, ax = plt.subplots(figsize=FIGSIZE)
            # masked array so empty bins stay blank (no interpolation).
            masked = np.ma.masked_invalid(stat.T)
            mesh = ax.pcolormesh(xedges, yedges, masked, cmap="plasma",
                                 shading="auto")
            cb = fig.colorbar(mesh, ax=ax)
            cb.set_label("Min. relative $E_\\mathrm{pot}$ in bin (kJ/mol)")
            ax.set_xlabel("distance 1 ($\\AA$)")
            ax.set_ylabel("distance 2 ($\\AA$)")
            ax.set_title("Binned minimum sampled $E_\\mathrm{pot}$ "
                         "— not a free-energy surface")
            _save(fig, fig_dir / "energy_landscape_2d.png")
        else:
            print(f"  (skipping 2D binned landscape: only {npts} points for "
                  f"{bins}x{bins} bins -- too sparse; use the scatter map "
                  "structural_map_2d.png instead)")
    else:
        print("  (skipping 2D maps: no energy column populated)")

    print("Done.")
    return 0


def _binned_min(x, y, e, bins):
    """Minimum energy per 2D bin (NaN where empty), no scipy dependency."""
    xedges = np.linspace(x.min(), x.max(), bins + 1)
    yedges = np.linspace(y.min(), y.max(), bins + 1)
    stat = np.full((bins, bins), np.nan)
    ix = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    for xi, yi, ei in zip(ix, iy, e):
        cur = stat[xi, yi]
        if np.isnan(cur) or ei < cur:
            stat[xi, yi] = ei
    return stat, xedges, yedges


if __name__ == "__main__":
    sys.exit(main())
