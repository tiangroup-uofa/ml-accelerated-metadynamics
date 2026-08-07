#!/usr/bin/env python3
"""
region_analysis.py — Phase B
============================
Partition the Diels-Alder reaction coordinate (mean forming C–C distance) into
five physically-motivated regions and quantify the MACE-OFF23 vs GFN2-xTB
discrepancy in each: force RMSE, force cosine similarity, and energy discrepancy.

Tests Dr. Tian's hypothesis that the largest errors are concentrated in the
highly-distorted barrier (TS) region.

Regions (forming C–C, Å) — thresholds documented explicitly:
  reactant_basin :  r > 2.9      (weakly-bound pre-reaction complex)
  pre_TS_rising  :  2.4 < r ≤ 2.9 (bonds beginning to form, energy rising)
  TS_region      :  1.95 < r ≤ 2.4 (brackets BOTH saddles: MACE 2.0, xTB 2.32 Å)
  post_TS        :  1.7 < r ≤ 1.95 (past the barrier, steep exothermic descent)
  product_basin  :  r ≤ 1.7       (nearly-formed cyclohexene ring)

Inputs: results/region_{mace,xtb}.npz (from eval_path.py on region_geoms).
Energy discrepancy: d = E_MACE − E_xTB has a large constant offset (different
absolute references, fixed C6H10 composition); we report it RELATIVE TO THE
REACTANT-BASIN MEAN, so it shows how the gap *grows* along the coordinate (0 in
the reactant basin by construction).

Output: results/region_stats.csv/.json + figures/regionwise_error.png.
Pure analysis (numpy/matplotlib); system python3.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; FIG = HERE.parent / "figures"
EV2KCAL = 23.060548
REGIONS = [("reactant_basin", 2.9, 99.0), ("pre_TS_rising", 2.4, 2.9),
           ("TS_region", 1.95, 2.4), ("post_TS", 1.7, 1.95),
           ("product_basin", 0.0, 1.7)]
LABELS = ["reactant\nbasin", "pre-TS\nrising", "TS\nregion", "post-TS", "product\nbasin"]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def main():
    m = np.load(RES / "region_mace.npz"); x = np.load(RES / "region_xtb.npz")
    coord = m["coord"]; Fm, Fx = m["forces"], x["forces"]
    # per-config metrics
    rmse = np.sqrt(((Fm - Fx) ** 2).reshape(len(coord), -1).mean(1))
    cos = np.array([fm.ravel() @ fx.ravel() /
                    (np.linalg.norm(fm) * np.linalg.norm(fx) + 1e-12)
                    for fm, fx in zip(Fm, Fx)])
    dE = (m["energies"] - x["energies"]) * EV2KCAL         # has constant offset

    # region assignment
    reg = np.full(len(coord), "", dtype=object)
    for name, lo, hi in REGIONS:
        reg[(coord > lo) & (coord <= hi)] = name
    react_mean_dE = dE[reg == "reactant_basin"].mean()      # reference for ΔE

    rows = []
    for name, _, _ in REGIONS:
        sel = reg == name
        n = int(sel.sum())
        rows.append(dict(
            region=name, n=n,
            forming_CC_range=[float(coord[sel].min()), float(coord[sel].max())] if n else None,
            force_rmse_mean=float(rmse[sel].mean()) if n else None,
            force_rmse_median=float(np.median(rmse[sel])) if n else None,
            force_rmse_std=float(rmse[sel].std()) if n else None,
            cosine_mean=float(cos[sel].mean()) if n else None,
            cosine_median=float(np.median(cos[sel])) if n else None,
            dE_rel_reactant_mean=float(dE[sel].mean() - react_mean_dE) if n else None,
            dE_rel_reactant_std=float(dE[sel].std()) if n else None,
        ))
    (RES / "region_stats.json").write_text(json.dumps(rows, indent=2))
    import csv
    with open(RES / "region_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"{'region':16s} {'N':>4s} {'r-range':>12s} {'RMSE(mean±std)':>18s} "
          f"{'RMSE med':>9s} {'cos mean':>9s} {'ΔE-rel(kcal)':>13s}")
    for r in rows:
        rng = f"{r['forming_CC_range'][0]:.2f}-{r['forming_CC_range'][1]:.2f}" if r["n"] else "-"
        print(f"{r['region']:16s} {r['n']:>4d} {rng:>12s} "
              f"{r['force_rmse_mean']:>10.3f}±{r['force_rmse_std']:.3f}   "
              f"{r['force_rmse_median']:>9.3f} {r['cosine_mean']:>9.3f} "
              f"{r['dE_rel_reactant_mean']:>+13.1f}")

    # ---- figure: 3 panels sharing the region axis ----
    xpos = np.arange(len(REGIONS))
    rm = [r["force_rmse_mean"] for r in rows]; rs = [r["force_rmse_std"] for r in rows]
    cm = [r["cosine_mean"] for r in rows]
    de = [r["dE_rel_reactant_mean"] for r in rows]
    nn = [r["n"] for r in rows]
    grad = ["#8ecae6", "#4c9fc0", "#ee6c4d", "#c1503a", "#7d8597"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4.3))
    a1.bar(xpos, rm, yerr=rs, capsize=3, color=grad)
    a1.set_ylabel("force RMSE (eV/Å)"); a1.set_title("Force error by region")
    for i, (v, n) in enumerate(zip(rm, nn)):
        a1.annotate(f"n={n}", (i, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    a2.bar(xpos, cm, color=grad); a2.set_ylim(min(cm) - 0.03, 1.001)
    a2.axhline(1.0, color="#9aa5b1", lw=.7, ls=":")
    a2.set_ylabel("force cosine similarity"); a2.set_title("Force direction agreement")
    a3.bar(xpos, de, color=grad)
    a3.axhline(0, color="#333", lw=.7)
    a3.set_ylabel("ΔE = E_MACE − E_xTB\n(rel. reactant basin, kcal/mol)")
    a3.set_title("Energy discrepancy by region")
    for ax in (a1, a2, a3):
        ax.set_xticks(xpos); ax.set_xticklabels(LABELS, fontsize=9)
    fig.suptitle("Diels–Alder: MACE↔xTB discrepancy is concentrated in the distorted TS region",
                 y=1.03, fontsize=13)
    fig.tight_layout(); FIG.mkdir(exist_ok=True)
    out = FIG / "regionwise_error.png"
    fig.savefig(out, bbox_inches="tight"); print(f"\nwrote {out}, region_stats.csv/json")


if __name__ == "__main__":
    main()
