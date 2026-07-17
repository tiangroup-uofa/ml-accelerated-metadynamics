#!/usr/bin/env python3
"""
plot_phase2.py
==============
Figures for the phase-2 work:

  1. temperature_vs_time.png  -- max O-O distance vs time for the trimer at
     several NVT temperatures (fixed kpush). Shows the temperature effect on
     evaporation, separate from the bias effect.

  2. pbc_vs_gas.png -- max O-O distance vs time comparing a gas-phase cluster
     (evaporates to 100+ A) against the PERIODIC box (stays condensed, capped
     by the cell). The headline PBC result: periodicity stops evaporation.
     Also shows mean coordination number for the PBC run (network intact vs
     degraded).
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

TEMPS = [200, 250, 298, 350]


def plot_temperature():
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("coolwarm")
    for i, T in enumerate(TEMPS):
        p = ANA / f"cv_T{T}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        ax.plot(df.time_ps, df.max_OO_A, lw=1.6,
                color=cmap(i / (len(TEMPS) - 1)), label=f"{T} K")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("max O-O distance (A)")
    ax.set_yscale("log")
    ax.set_title("(H2O)3 at fixed bias (kpush=0.02): temperature effect\n"
                 "higher T -> faster / farther evaporation")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="NVT temperature")
    fig.tight_layout()
    fig.savefig(FIG / "temperature_vs_time.png", dpi=150)
    plt.close(fig)
    print("  wrote temperature_vs_time.png")


def plot_pbc_vs_gas():
    # gas reference: the trimer at low bias (drifts to ~100 A) from main sweep
    gas = ANA / "cv_n3_low.csv"
    pbc = ANA / "cv_n8_PBC_v2.csv"
    if not (gas.exists() and pbc.exists()):
        print("  (need cv_n3_low.csv and cv_n8_PBC_v2.csv)")
        return
    dgas = pd.read_csv(gas)
    dpbc = pd.read_csv(pbc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(dgas.time_ps, dgas.max_OO_A, color="tab:red", lw=1.6,
             label="gas-phase (H2O)3 -- evaporates")
    ax1.plot(dpbc.time_ps, dpbc.max_OO_A, color="tab:blue", lw=1.6,
             label="periodic box (H2O)8 -- condensed")
    ax1.axhline(9.2, ls="--", color="tab:blue", alpha=0.5,
                label="PBC box edge (9.2 A)")
    ax1.set_yscale("log")
    ax1.set_xlabel("time (ps)")
    ax1.set_ylabel("max O-O distance (A)")
    ax1.set_title("Evaporation: gas vs periodic")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8)

    ax2.plot(dpbc.time_ps, dpbc.mean_coordination, color="tab:purple", lw=1.6)
    ax2.set_xlabel("time (ps)")
    ax2.set_ylabel("mean coordination (H-bonds per water)")
    ax2.set_title("PBC run: H-bond network (coordination CV)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "pbc_vs_gas.png", dpi=150)
    plt.close(fig)
    print("  wrote pbc_vs_gas.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plot_temperature()
    plot_pbc_vs_gas()


if __name__ == "__main__":
    main()
