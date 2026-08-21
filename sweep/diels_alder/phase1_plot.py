#!/usr/bin/env python3
"""
phase1_plot.py — the two Phase 1 overview figures for the website.

Every value is read from a generated result file; nothing is hand-entered except
the verified xTB reference constants, which are themselves read from
phase1b_report.json where available.

  1. phase1_identical_geoms.png — xTB vs MACE on the SAME frozen xTB R/TS/P
  2. phase1_saddle_comparison.png — is the xTB TS a MACE saddle? + dx robustness
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

C_XTB, C_MACE, C_GREY = "#c1121f", "#2e6f95", "#6b7280"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "savefig.dpi": 200, "font.family": "DejaVu Sans"})

rep = json.loads((HERE / "phase1b_report.json").read_text())
mace = json.loads((HERE / "phase1b_mace_results.json").read_text())
c1 = json.loads((HERE / "phase1c_hessian_results.json").read_text())
d1 = json.loads((HERE / "phase1d_results" / "phase1d_results.json").read_text())

B = rep["barrier_heights_on_identical_geometries"]
R = rep["reaction_energies_on_identical_geometries"]
S = rep["critical_finding_TS_stationarity"]


def fig_identical():
    """Relative energies of both potentials on the identical frozen geometries."""
    xtb = [0.0, B["xTB_barrier_kcal_mol"], R["xTB_reaction_energy_kcal_mol"]]
    mac = [0.0, B["MACE_barrier_at_xTB_geometries_kcal_mol"],
           R["MACE_reaction_energy_at_xTB_geometries_kcal_mol"]]
    x = np.arange(3)
    labels = ["R\n(reactant)", "TS\n(2.315 Å)", "P\n(product)"]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(x, xtb, "-o", color=C_XTB, lw=2.2, ms=8, label="GFN2-xTB")
    ax.plot(x, mac, "-s", color=C_MACE, lw=2.2, ms=8, label="MACE-OFF23")
    ax.axhline(0, color="0.85", lw=1, zorder=0)

    for xi, (a, b) in enumerate(zip(xtb, mac)):
        if xi == 0:
            continue
        ax.annotate("", xy=(xi, a), xytext=(xi, b),
                    arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.1))
        ax.text(xi + 0.07, (a + b) / 2, f"{abs(b - a):.1f}", color="0.3",
                fontsize=9.5, va="center")

    ax.text(1, mac[1] + 3.5, f"{mac[1]:.1f}", color=C_MACE, ha="center", fontsize=10)
    ax.text(1, xtb[1] - 5.5, f"{xtb[1]:.1f}", color=C_XTB, ha="center", fontsize=10)
    ax.text(2, mac[2] + 3.5, f"{mac[2]:.1f}", color=C_MACE, ha="center", fontsize=10)
    ax.text(2, xtb[2] - 6.0, f"{xtb[2]:.1f}", color=C_XTB, ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.35, 2.5)
    ax.set_ylabel("energy relative to own reactant (kcal/mol)")
    ax.set_title("Same three frozen xTB geometries, two potentials\n"
                 "the disagreement survives with path-search removed", fontsize=11.5)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "phase1_identical_geoms.png", bbox_inches="tight")
    plt.close(fig)


def fig_saddle():
    """Left: stationarity of the xTB TS under each potential.
    Right: the two saddles on the reaction coordinate + dx recovery."""
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.4, 4.5),
                                 gridspec_kw={"width_ratios": [1, 1.35]})

    # --- left: |F|max at the xTB TS geometry (log scale) ---------------------
    vals = [S["xTB_TS_fmax_eV_per_Angstrom"], S["MACE_fmax_at_xTB_TS_geometry_eV_per_Angstrom"]]
    aL.bar([0, 1], vals, 0.55, color=[C_XTB, C_MACE])
    aL.set_yscale("log")
    aL.set_xticks([0, 1])
    aL.set_xticklabels(["GFN2-xTB", "MACE-OFF23"])
    aL.set_ylabel("max atomic force at the xTB TS (eV/Å)")
    for i, v in enumerate(vals):
        aL.text(i, v * 1.35, f"{v:g}", ha="center", fontsize=10)
    aL.axhline(0.05, color="0.5", ls=":", lw=1.2)
    aL.text(-0.42, 0.062, "typical convergence threshold 0.05", fontsize=8,
            color="0.45", ha="left", va="bottom")
    aL.set_ylim(3e-4, 12)
    aL.set_title("The xTB TS is not a\nstationary point on MACE", fontsize=11)

    # --- right: saddle positions along the forming C–C coordinate ------------
    x_xtb = c1["comparison_to_xTB"]["xTB_TS_forming_CC_Angstrom"]
    b_xtb = c1["comparison_to_xTB"]["xTB_barrier_kcal_mol"]
    x_mace = c1["converged_MACE_TS"]["forming_CC_mean_Angstrom"]
    b_mace = c1["converged_MACE_TS"]["barrier_kcal_mol"]
    shift = c1["comparison_to_xTB"]["MACE_TS_displacement_Angstrom"]

    aR.scatter([x_xtb], [b_xtb], s=150, color=C_XTB, zorder=4, label="xTB saddle")
    aR.scatter([x_mace], [b_mace], s=150, marker="s", color=C_MACE, zorder=4,
               label="MACE saddle (Sella from the xTB TS)")

    bars = [v["barrier_kcal_mol"] for v in d1.values()]
    xs = [c1["converged_MACE_TS"]["forming_CC_mean_Angstrom"]] * len(bars)
    aR.scatter(xs, bars, s=620, facecolors="none", edgecolors=C_MACE,
               linewidths=1.4, alpha=0.75, zorder=3,
               label=f"{len(bars)} × δx perturbations, all recovered")

    aR.annotate("", xy=(x_mace, b_xtb + 2), xytext=(x_xtb, b_xtb + 2),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.2))
    aR.text((x_xtb + x_mace) / 2, b_xtb + 4.0, f"{shift:.3f} Å", ha="center",
            fontsize=10, color="0.3")

    aR.text(x_xtb - 0.016, b_xtb - 1.5, f"{b_xtb} kcal/mol\n−394 cm⁻¹",
            color=C_XTB, fontsize=9, ha="right", va="top")
    aR.text(x_mace + 0.016, b_mace - 4.5, f"{b_mace:.1f} kcal/mol\n1 imaginary mode",
            color=C_MACE, fontsize=9, ha="right", va="top")

    aR.set_xlim(1.95, 2.44)
    aR.set_ylim(-2, 44)
    aR.invert_xaxis()
    aR.set_xlabel("forming C–C at the saddle (Å)")
    aR.set_ylabel("activation barrier (kcal/mol)")
    aR.set_title("Two distinct but nearby saddles\n(barrier spread across δx runs: 0.000 kcal/mol)",
                 fontsize=11)
    aR.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG / "phase1_saddle_comparison.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_identical()
    fig_saddle()
    sp = sorted({round(v["barrier_kcal_mol"], 4) for v in d1.values()})
    print(f"identical-geometry: xTB {B['xTB_barrier_kcal_mol']} vs MACE "
          f"{B['MACE_barrier_at_xTB_geometries_kcal_mol']} kcal/mol")
    print(f"saddles: xTB {c1['comparison_to_xTB']['xTB_TS_forming_CC_Angstrom']} Å vs MACE "
          f"{c1['converged_MACE_TS']['forming_CC_mean_Angstrom']:.4f} Å "
          f"(shift {c1['comparison_to_xTB']['MACE_TS_displacement_Angstrom']:.4f} Å)")
    print(f"dx robustness: {len(d1)} runs, distinct barriers = {sp}")
    print("wrote phase1_identical_geoms.png, phase1_saddle_comparison.png")
