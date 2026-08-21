#!/usr/bin/env python3
"""phase2_plot.py — Phase 2 convergence figures (reads phase2/summary.json only)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE / "phase2"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

XTB_BARRIER, XTB_FORMING, XTB_RG = 6.7, 2.315, 3.77
RAW_MACE_BARRIER, RAW_MACE_FORMING = 35.8, 2.004   # Phase 1C verified

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "savefig.dpi": 200, "font.family": "DejaVu Sans"})
C_SADDLE, C_SCAN, C_TGT, C_BAD = "#2e6f95", "#e07a5f", "#c1121f", "#9aa0a6"


def load():
    rows = json.loads((P2 / "summary.json").read_text())
    d = {
        "it": np.array([r["iteration"] for r in rows]),
        "saddle": np.array([r["ts"]["barrier_kcal_frozen_R"] for r in rows]),
        "scan": np.array([r["reaction"]["barrier_kcal"] for r in rows]),
        "forming": np.array([r["ts"]["forming_CC_mean"] for r in rows]),
        "rmsd": np.array([r["ts"]["rmsd_to_xtb_ts"] for r in rows]),
        "valid": np.array([r["ts"]["valid_da_saddle"] for r in rows]),
        "xtbf": np.array([r["xtb_at_ts"]["fmax"] for r in rows]),
        "frmse": np.array([r["xtb_at_ts"]["force_rmse_corr_vs_xtb"] for r in rows]),
        "ho_ts": np.array([r["heldout"]["TS_plus_postTS_rmse"] for r in rows]),
        "ho_all": np.array([r["heldout"]["overall"]["rmse"] for r in rows]),
        "rg": np.array([r["water"]["rg_mean"] for r in rows]),
        "rgsd": np.array([r["water"]["rg_std"] for r in rows]),
    }
    return d


def mark_invalid(ax, d, y):
    bad = ~d["valid"]
    if bad.any():
        ax.scatter(d["it"][bad], y[bad], s=150, facecolors="none",
                   edgecolors=C_BAD, linewidths=2, zorder=5,
                   label="no certified 1st-order saddle")


def fig1(d):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(d["it"], d["saddle"], "-o", color=C_SADDLE, lw=2, label="Sella saddle barrier")
    ax.plot(d["it"], d["scan"], "-s", color=C_SCAN, lw=2, label="constrained-scan barrier")
    mark_invalid(ax, d, d["saddle"])
    ax.axhline(XTB_BARRIER, color=C_TGT, ls="--", lw=1.5, label=f"xTB target {XTB_BARRIER}")
    ax.annotate("iteration 0 is the best\nsaddle barrier of the run",
                xy=(0.04, d["saddle"][0]), xytext=(1.6, 6.9), fontsize=9,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1))
    ax.set_xlabel("iteration"); ax.set_ylabel("activation barrier (kcal/mol)")
    ax.set_title("Barrier does not converge to the xTB target\n"
                 f"(raw uncorrected MACE = {RAW_MACE_BARRIER} kcal/mol, off-scale)", fontsize=11)
    ax.set_xticks(d["it"]); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "phase2_barrier.png", bbox_inches="tight")
    plt.close(fig)


def fig2(d):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
    a.plot(d["it"], d["forming"], "-o", color=C_SADDLE, lw=2)
    mark_invalid(a, d, d["forming"])
    a.axhline(XTB_FORMING, color=C_TGT, ls="--", lw=1.5, label=f"xTB TS {XTB_FORMING} Å")
    a.axhline(RAW_MACE_FORMING, color="0.5", ls=":", lw=1.2,
              label=f"raw MACE TS {RAW_MACE_FORMING} Å")
    a.set_xlabel("iteration"); a.set_ylabel("forming C–C at TS (Å)")
    a.set_title("TS position oscillates (±0.5 Å)"); a.set_xticks(d["it"])
    a.legend(frameon=False, fontsize=8.5)

    b.plot(d["it"], d["rmsd"], "-o", color=C_SADDLE, lw=2)
    mark_invalid(b, d, d["rmsd"])
    b.set_xlabel("iteration"); b.set_ylabel("RMSD to xTB TS (Å)")
    b.set_title("RMSD to xTB TS\n(open circles = scan-max fallback, not a saddle)",
                fontsize=10)
    b.set_xticks(d["it"]); b.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "phase2_geometry.png", bbox_inches="tight")
    plt.close(fig)


def fig3(d):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(d["it"], d["xtbf"], "-o", color=C_SADDLE, lw=2,
            label="xTB |F|max at the corrected TS")
    ax.plot(d["it"], d["frmse"], "-s", color=C_SCAN, lw=2,
            label="force RMSE (corrected vs xTB) at the TS")
    mark_invalid(ax, d, d["xtbf"])
    ax.set_xlabel("iteration"); ax.set_ylabel("force (eV/Å)")
    ax.set_title("xTB residual force at the corrected TS does not decrease\n"
                 "(cosine omitted: |F_corrected| ≈ 1e-4 at a converged saddle, so it is 0/0 noise)",
                 fontsize=10)
    ax.set_xticks(d["it"]); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "phase2_forces.png", bbox_inches="tight")
    plt.close(fig)


def fig4(d):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
    a.plot(d["it"], d["ho_ts"], "-o", color=C_SADDLE, lw=2, label="TS+post-TS")
    a.plot(d["it"], d["ho_all"], "-s", color=C_SCAN, lw=2, label="overall")
    a.set_xlabel("iteration"); a.set_ylabel("held-out force RMSE (eV/Å)")
    a.set_title("Held-out reactive set (never trained on)\nslowly degrades", fontsize=10)
    a.set_xticks(d["it"]); a.legend(frameon=False, fontsize=9)

    b.errorbar(d["it"], d["rg"], yerr=d["rgsd"], fmt="-o", color=C_SADDLE,
               capsize=3, lw=2)
    b.axhline(XTB_RG, color=C_TGT, ls="--", lw=1.5, label=f"xTB Rg {XTB_RG} Å")
    b.set_xlabel("iteration"); b.set_ylabel("droplet radius of gyration (Å)")
    b.set_title("Water stays stable\n(scatter within MD noise)", fontsize=10)
    b.set_xticks(d["it"]); b.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "phase2_generalization.png", bbox_inches="tight")
    plt.close(fig)


def table(d):
    lines = ["| iter | new xTB calls (cum.) | saddle barrier | scan barrier | forming C–C (Å) "
             "| certified saddle | xTB \\|F\\|max at TS | held-out TS+post | water Rg |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i in range(len(d["it"])):
        lines.append(
            f"| {d['it'][i]} | {7*(i+1)} | {d['saddle'][i]:.2f} | {d['scan'][i]:.2f} | "
            f"{d['forming'][i]:.3f} | {'yes' if d['valid'][i] else 'NO'} | "
            f"{d['xtbf'][i]:.3f} | {d['ho_ts'][i]:.4f} | {d['rg'][i]:.3f} |")
    lines.append(f"| **xTB target** | — | **{XTB_BARRIER}** | **{XTB_BARRIER}** | "
                 f"**{XTB_FORMING}** | yes (−394 cm⁻¹) | 0 | — | **{XTB_RG}** |")
    (P2 / "summary_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    d = load()
    fig1(d); fig2(d); fig3(d); fig4(d); table(d)
    print("\nwrote phase2 figures + summary_table.md")
