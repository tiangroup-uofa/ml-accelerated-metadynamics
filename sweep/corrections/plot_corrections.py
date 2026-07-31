#!/usr/bin/env python3
"""
plot_corrections.py
===================
Publication-quality figures + summary table for the post-training correction
benchmark. Reads fitted_corrections.json (force metrics), results/da_summary.json
+ da_*.csv (Diels-Alder), results/water_{xtb,mace}.json (droplet structure).

Figures (figures/):
  corr_force_metrics.png   val force RMSE per correction, per domain (+ reduction)
  corr_da_energetics.png   reaction profiles + barrier/ΔE bars vs xTB target
  corr_water_structure.png O–O distribution overlay + Rg, vs xTB reference
And a markdown summary table -> summary_table.md.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
ORDER = ["baseline", "global", "affine", "element", "delta"]
COL = {"baseline": "#6b7280", "global": "#e07a5f", "affine": "#f2cc8f",
       "element": "#81b29a", "delta": "#2e6f95"}
XTB_C = "#c1121f"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200,
                     "figure.dpi": 200, "font.family": "DejaVu Sans"})


def force_fig(fit):
    m = fit["metrics"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(ORDER)); w = 0.38
    vw = [m[n]["val_water"]["rmse"] for n in ORDER]
    vd = [m[n]["val_da"]["rmse"] for n in ORDER]
    ax.bar(x - w/2, vw, w, color="#3d5a80", label="water (val)")
    ax.bar(x + w/2, vd, w, color="#ee6c4d", label="Diels–Alder (val)")
    base = m["baseline"]["val"]["rmse"]
    for i, n in enumerate(ORDER):
        red = 100*(1 - m[n]["val"]["rmse"]/base)
        ax.annotate(f"−{red:.0f}%" if n != "baseline" else "ref",
                    (i, max(vw[i], vd[i])), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    ax.set_ylabel("validation force RMSE (eV/Å)")
    ax.set_title("Force error vs xTB, before/after correction (held-out)")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIG / "corr_force_metrics.png", bbox_inches="tight"); plt.close(fig)


def da_fig(da):
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    for n in ORDER:
        d = pd.read_csv(RES / f"da_{n}.csv")
        aL.plot(d.coord, d.rel, "-o", ms=3, lw=1.6, color=COL[n],
                label=n + ("" if da[n]["conservative"] else " (non-cons.)"))
    aL.axhline(da["xtb"]["barrier"], color=XTB_C, ls="--", lw=1,
               label=f"xTB barrier {da['xtb']['barrier']:.1f}")
    aL.invert_xaxis(); aL.set_xlabel("forming C–C distance (Å)")
    aL.set_ylabel("relative energy (kcal/mol)")
    aL.set_title("Corrected MACE reaction profiles"); aL.legend(frameon=False, fontsize=8)

    x = np.arange(len(ORDER)); w = 0.36
    bar = [da[n]["barrier"] for n in ORDER]; dE = [da[n]["reaction_energy"] for n in ORDER]
    aR.bar(x - w/2, bar, w, color="#2e6f95", label="barrier")
    aR.bar(x + w/2, dE, w, color="#e07a5f", label="reaction E")
    aR.axhline(da["xtb"]["barrier"], color=XTB_C, ls="--", lw=1)
    aR.axhline(da["xtb"]["reaction_energy"], color=XTB_C, ls=":", lw=1)
    aR.annotate(f"xTB barrier {da['xtb']['barrier']:.1f}", (len(ORDER)-1, da['xtb']['barrier']),
                fontsize=8, color=XTB_C, ha="right", va="bottom")
    aR.set_xticks(x); aR.set_xticklabels(ORDER, rotation=30, ha="right")
    aR.set_ylabel("kcal/mol"); aR.set_title("Energetics vs xTB target")
    aR.legend(frameon=False, fontsize=9); aR.axhline(0, color="#333", lw=.7)
    fig.suptitle("Diels–Alder: does the correction move MACE toward xTB?", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "corr_da_energetics.png", bbox_inches="tight")
    plt.close(fig)


def water_fig(wx, wm):
    ctr = np.array(wm["ctr"])
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    aL.plot(ctr, wx["xtb"]["hist"], color=XTB_C, lw=2.2, label="xTB (reference)")
    for n in ORDER:
        aL.plot(ctr, wm[n]["hist"], lw=1.5, color=COL[n], alpha=.9, label=n)
    aL.set_xlabel("O–O distance (Å)"); aL.set_ylabel("pair-distance density")
    aL.set_title("Droplet O–O distribution"); aL.legend(frameon=False, fontsize=8)

    x = np.arange(len(ORDER))
    rg = [wm[n]["rg_mean"] for n in ORDER]; rgs = [wm[n]["rg_std"] for n in ORDER]
    aR.bar(x, rg, 0.6, yerr=rgs, color=[COL[n] for n in ORDER], capsize=3)
    aR.axhline(wx["xtb"]["rg_mean"], color=XTB_C, ls="--", lw=1.2,
               label=f"xTB Rg {wx['xtb']['rg_mean']:.2f}")
    aR.set_xticks(x); aR.set_xticklabels(ORDER, rotation=30, ha="right")
    aR.set_ylabel("radius of gyration (Å)"); aR.set_ylim(min(rg)-0.15, max(rg)+0.15)
    aR.set_title("Droplet size"); aR.legend(frameon=False, fontsize=9)
    fig.suptitle("Water droplet: corrected-MACE structure vs xTB reference", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "corr_water_structure.png", bbox_inches="tight")
    plt.close(fig)


def table(fit, da, wx, wm):
    m = fit["metrics"]
    lines = ["| model | val force RMSE (eV/Å) | RMSE ↓ | DA barrier | DA ΔE | DA TS (Å) | droplet Rg (Å) | O–O peak (Å) | conservative |",
             "|---|---|---|---|---|---|---|---|---|"]
    base = m["baseline"]["val"]["rmse"]
    for n in ORDER:
        red = "ref" if n == "baseline" else f"{100*(1-m[n]['val']['rmse']/base):.0f}%"
        lines.append(f"| {n} | {m[n]['val']['rmse']:.3f} | {red} | "
                     f"{da[n]['barrier']:.1f} | {da[n]['reaction_energy']:.1f} | "
                     f"{da[n]['ts_forming']:.2f} | {wm[n]['rg_mean']:.2f} | "
                     f"{wm[n]['peak']:.2f} | {'yes' if da[n]['conservative'] else 'NO'} |")
    lines.append(f"| **xTB (target)** | 0 | — | **{da['xtb']['barrier']:.1f}** | "
                 f"**{da['xtb']['reaction_energy']:.1f}** | {da['xtb']['ts_forming']:.2f} | "
                 f"**{wx['xtb']['rg_mean']:.2f}** | **{wx['xtb']['peak']:.2f}** | — |")
    (HERE / "summary_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    FIG.mkdir(exist_ok=True)
    fit = json.loads((HERE / "fitted_corrections.json").read_text())
    da = json.loads((RES / "da_summary.json").read_text())
    wx = json.loads((RES / "water_xtb.json").read_text())
    wm = json.loads((RES / "water_mace.json").read_text())
    force_fig(fit); da_fig(da); water_fig(wx, wm); table(fit, da, wx, wm)
    print(f"\nwrote figures/ + summary_table.md")


if __name__ == "__main__":
    main()
