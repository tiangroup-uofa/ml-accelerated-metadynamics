#!/usr/bin/env python3
"""
plot_large.py
=============
Figures for the larger vacuum-cluster runs.

figures/droplet_stability.png  (step 1/3): Rg(t) and #evaporated(t) for the
    n=20/30/50 low-bias runs -- shows whether a bigger free cluster holds as
    a droplet instead of evaporating.

figures/bias_chemistry_n30.png (step 2): max intramolecular O-H(t) and number
    of transferred protons(t) for n=30 at low/med/high kpush -- shows whether
    raising the bias breaks O-H bonds or drives proton transfer.

Matplotlib only (no seaborn), matching the rest of the project.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ANA = HERE / "analysis"
FIG = HERE / "figures"


def load(tag):
    p = ANA / f"cv_{tag}.csv"
    return pd.read_csv(p) if p.exists() else None


def plot_droplet():
    sizes = [20, 30, 50]
    colors = {20: "tab:green", 30: "tab:orange", 50: "tab:red"}
    dfs = {n: load(f"n{n}_low") for n in sizes}
    if all(v is None for v in dfs.values()):
        print("  (no step-1 CSVs yet)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for n in sizes:
        df = dfs[n]
        if df is None:
            continue
        ax1.plot(df.time_ps, df.Rg, color=colors[n], label=f"n={n}")
        ax2.plot(df.time_ps, df.n_evaporated, color=colors[n], label=f"n={n}")
    ax1.set_xlabel("time (ps)"); ax1.set_ylabel("radius of gyration Rg (A)")
    ax1.set_title("Droplet size vs time (low bias)"); ax1.legend()
    ax2.set_xlabel("time (ps)"); ax2.set_ylabel("# evaporated molecules")
    ax2.set_title("Evaporation vs time (low bias)"); ax2.legend()
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "droplet_stability.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def plot_chemistry():
    labels = [("low", "tab:green"), ("med", "tab:orange"), ("high", "tab:red")]
    dfs = {lab: load(f"n30_{lab}") for lab, _ in labels}
    if all(v is None for v in dfs.values()):
        print("  (no step-2 CSVs yet)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab, c in labels:
        df = dfs[lab]
        if df is None:
            continue
        ax1.plot(df.time_ps, df.max_intra_OH_A, color=c, label=f"kpush {lab}")
        ax2.plot(df.time_ps, df.n_transfer, color=c, label=f"kpush {lab}")
    ax1.axhline(1.30, ls="--", color="k", lw=0.8, label="O-H break (1.3 A)")
    ax1.set_xlabel("time (ps)"); ax1.set_ylabel("max intramolecular O-H (A)")
    ax1.set_title("O-H integrity vs bias (n=30)"); ax1.legend()
    ax2.set_xlabel("time (ps)"); ax2.set_ylabel("# transferred protons")
    ax2.set_title("Proton transfer vs bias (n=30)"); ax2.legend()
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "bias_chemistry_n30.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def main():
    plot_droplet()
    plot_chemistry()


if __name__ == "__main__":
    main()
