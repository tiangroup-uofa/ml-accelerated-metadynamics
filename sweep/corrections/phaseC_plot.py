#!/usr/bin/env python3
"""phaseC_plot.py — Phase C deliverable figures + summary table.
  1 region-wise force metrics (baseline A vs barrier-enriched B, + raw/C)
  2 reaction energetics (raw / A / B / C / xTB)
  3 equilibrium water sanity (raw / A / B / xTB)
Reads phaseC/{force_test,reaction_summary,water_mace,water_xtb}.json + reaction_*.csv.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PC = HERE / "phaseC"; FIG = HERE / "figures"
REG = ["reactant_basin", "pre_TS_rising", "TS_region", "post_TS", "product_basin"]
RLAB = ["reactant", "pre-TS", "TS", "post-TS", "product"]
COL = {"raw_MACE": "#6b7280", "A_baseline": "#e07a5f", "B_barrier": "#2e6f95",
       "C_generic": "#81b29a"}
XTB_C = "#c1121f"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def fig_forces():
    ft = json.loads((PC / "force_test.json").read_text())
    x = np.arange(len(REG)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, mdl in enumerate(["raw_MACE", "A_baseline", "B_barrier", "C_generic"]):
        vals = [ft[mdl].get(r, {}).get("rmse", np.nan) for r in REG]
        ax.bar(x + (i-1.5)*w, vals, w, color=COL[mdl], label=mdl)
    ax.axvspan(1.5, 3.5, color="0.93", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(RLAB)
    ax.set_ylabel("held-out force RMSE (eV/Å)")
    ax.set_title("Region-wise held-out force error (grey = barrier region)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "phaseC_force_regions.png", bbox_inches="tight")
    plt.close(fig)


def fig_reaction():
    s = json.loads((PC / "reaction_summary.json").read_text())
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    for mdl in ["raw_MACE", "A_baseline", "B_barrier", "C_generic"]:
        d = pd.read_csv(PC / f"reaction_{mdl}.csv")
        aL.plot(d.coord, d.rel_kcal, "-o", ms=3, lw=1.7, color=COL[mdl],
                label=f"{mdl} ({s[mdl]['barrier']:.1f})")
    aL.axhline(s["xtb_target"]["barrier"], color=XTB_C, ls="--", lw=1,
               label=f"xTB target ({s['xtb_target']['barrier']:.1f})")
    aL.invert_xaxis(); aL.set_xlabel("forming C–C distance (Å)")
    aL.set_ylabel("relative energy (kcal/mol)")
    aL.set_title("Corrected-MACE reaction profiles"); aL.legend(frameon=False, fontsize=8.5)
    order = ["raw_MACE", "A_baseline", "B_barrier", "C_generic"]
    bar = [s[m]["barrier"] for m in order]
    aR.bar(range(4), bar, 0.6, color=[COL[m] for m in order])
    aR.axhline(s["xtb_target"]["barrier"], color=XTB_C, ls="--", lw=1.2,
               label=f"xTB {s['xtb_target']['barrier']:.1f}")
    for i, v in enumerate(bar):
        aR.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)
    aR.set_xticks(range(4)); aR.set_xticklabels(["raw", "A\nbase", "B\nbarrier", "C\ngeneric"])
    aR.set_ylabel("activation barrier (kcal/mol)"); aR.legend(frameon=False, fontsize=9)
    aR.set_title("Barrier → xTB target")
    fig.suptitle("Phase C: barrier-enriched Delta gives the barrier closest to xTB", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "phaseC_reaction.png", bbox_inches="tight")
    plt.close(fig)


def fig_water():
    wm = json.loads((PC / "water_mace.json").read_text())
    wx = json.loads((PC / "water_xtb.json").read_text())
    ctr = np.array(wm["ctr"])
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    aL.plot(ctr, wx["xtb"]["hist"], color=XTB_C, lw=2.2, label="xTB (ref)")
    for mdl in ["raw_MACE", "A_baseline", "B_barrier"]:
        aL.plot(ctr, wm[mdl]["hist"], lw=1.5, color=COL[mdl], label=mdl)
    aL.set_xlabel("O–O distance (Å)"); aL.set_ylabel("pair-distance density")
    aL.set_title("Droplet O–O distribution"); aL.legend(frameon=False, fontsize=9)
    order = ["raw_MACE", "A_baseline", "B_barrier"]
    rg = [wm[m]["rg_mean"] for m in order]; rs = [wm[m]["rg_std"] for m in order]
    aR.bar(range(3), rg, 0.6, yerr=rs, capsize=3, color=[COL[m] for m in order])
    aR.axhline(wx["xtb"]["rg_mean"], color=XTB_C, ls="--", lw=1.2,
               label=f"xTB Rg {wx['xtb']['rg_mean']:.2f}")
    aR.set_xticks(range(3)); aR.set_xticklabels(["raw", "A base", "B barrier"])
    aR.set_ylabel("radius of gyration (Å)"); aR.set_ylim(min(rg)-0.2, max(rg)+0.2)
    aR.legend(frameon=False, fontsize=9); aR.set_title("Droplet size")
    fig.suptitle("Phase C: water equilibrium sanity — does barrier enrichment damage water?", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "phaseC_water.png", bbox_inches="tight")
    plt.close(fig)


def table():
    ft = json.loads((PC / "force_test.json").read_text())
    s = json.loads((PC / "reaction_summary.json").read_text())
    wm = json.loads((PC / "water_mace.json").read_text())
    wx = json.loads((PC / "water_xtb.json").read_text())

    def tspt(m):
        num = sum(ft[m][k]["rmse"]*ft[m][k]["n"] for k in ("TS_region", "post_TS"))
        den = sum(ft[m][k]["n"] for k in ("TS_region", "post_TS")); return num/den
    lines = ["| model | overall RMSE | TS+postTS RMSE | barrier | reaction E | TS Å | droplet Rg |",
             "|---|---|---|---|---|---|---|"]
    for m, lab in [("raw_MACE", "raw MACE"), ("A_baseline", "Delta-A baseline"),
                   ("B_barrier", "Delta-B barrier-enriched"), ("C_generic", "Delta-C generic")]:
        rg = wm.get(m, {}).get("rg_mean", "—")
        rg = f"{rg:.2f}" if isinstance(rg, float) else rg
        lines.append(f"| {lab} | {ft[m]['overall']['rmse']:.3f} | {tspt(m):.3f} | "
                     f"{s[m]['barrier']:.1f} | {s[m]['reaction_energy']:.1f} | "
                     f"{s[m]['ts_forming']:.2f} | {rg} |")
    lines.append(f"| **xTB (target)** | 0 | 0 | **{s['xtb_target']['barrier']:.1f}** | "
                 f"**{s['xtb_target']['reaction_energy']:.1f}** | {s['xtb_target']['ts_forming']:.2f} | "
                 f"**{wx['xtb']['rg_mean']:.2f}** |")
    (PC / "summary_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    FIG.mkdir(exist_ok=True)
    fig_forces(); fig_reaction(); fig_water(); table()
    print("\nwrote phaseC figures + summary_table.md")


if __name__ == "__main__":
    main()
