#!/usr/bin/env python3
"""
a2_plot.py — standardized-methodology comparison figure (Phase A2/A3).

xTB-ASE-NEB does NOT converge to a physical band for this reaction (documented in
a2_summary.json), so its scrambled band is NOT plotted as a reaction profile.
Instead the figure makes the apples-to-apples point rigorously:

  left  : MACE ASE-NEB profile vs MACE relaxed scan (they agree -> the reaction-
          path *methodology* is not the source of the discrepancy) vs the xTB
          relaxed scan (the appropriate xTB reaction-path method).
  right : MACE barrier across every methodology (scan / Sella-TS / ASE-NEB) — all
          ~36 — vs xTB scan 6.7; xTB-ASE-NEB flagged as FAILED (unphysical band).

Output: ../figures/a2_standardized_neb.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; DA = HERE.parent; FIG = DA / "figures"
XTB_C, MACE_C, PREV_C = "#c1121f", "#2e6f95", "#9aa5b1"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def main():
    s = json.loads((RES / "a2_summary.json").read_text())
    mace_neb = pd.read_csv(RES / "a2_mace.csv"); mace_neb = mace_neb[mace_neb.physical == 1]
    mace_scan = pd.read_csv(RES / "mace_cscan_profile.csv")
    xtb_scan = pd.read_csv(DA / "cscan_profile.csv")
    sella = json.loads((RES / "ts_mace_refined.json").read_text())

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(12, 4.7),
                                 gridspec_kw={"width_ratios": [1.55, 1.15]})
    # --- left: profiles ---
    aL.plot(mace_neb.forming_CC, mace_neb.rel_kcal, "-o", color=MACE_C, ms=4, lw=1.9,
            label=f"MACE-OFF23  ASE-NEB ({s['mace']['barrier_kcal']:.1f})")
    aL.plot(mace_scan.forming_CC_A, mace_scan.rel_E_kcal_mol, "--", color=MACE_C,
            lw=1.5, alpha=.7, label="MACE-OFF23  relaxed scan (36.0)")
    aL.plot(xtb_scan.forming_CC_A, xtb_scan.rel_E_kcal_mol, "-s", color=XTB_C, ms=3,
            lw=1.7, label="GFN2-xTB  relaxed scan (6.7)")
    aL.axhline(0, color="#cbd5e1", lw=.7, ls=":"); aL.invert_xaxis()
    aL.set_xlabel("forming C–C distance (Å)"); aL.set_ylabel("relative energy (kcal/mol)")
    aL.set_title("MACE: scan ≈ NEB  (methodology not the cause)")
    aL.legend(frameon=False, fontsize=8.5, loc="upper left")
    aL.text(0.98, 0.05, "xTB ASE-NEB band did not converge\n(unphysical — not shown)",
            transform=aL.transAxes, ha="right", va="bottom", fontsize=8, color=XTB_C,
            style="italic")

    # --- right: MACE barrier across methods vs xTB ---
    bars = [("MACE\nscan", 36.0, MACE_C), ("MACE\nSella-TS", sella["barrier_kcal"], MACE_C),
            ("MACE\nASE-NEB", s["mace"]["barrier_kcal"], MACE_C),
            ("xTB\nscan", 6.7, XTB_C), ("xTB\nASE-NEB", s["xtb"]["barrier_kcal"], XTB_C)]
    x = np.arange(len(bars))
    for i, (lab, v, c) in enumerate(bars):
        failed = lab.startswith("xTB\nASE")
        aR.bar(i, v, 0.62, color=c, hatch="///" if failed else None,
               alpha=0.45 if failed else 0.9, edgecolor=c)
        aR.annotate(f"{v:.0f}" + ("\nFAILED" if failed else ""), (i, min(v, 60)),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, color=c if failed else "black")
    aR.set_ylim(0, 65); aR.set_xticks(x); aR.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    aR.set_ylabel("activation barrier (kcal/mol)")
    aR.set_title("Barrier is ~36 for MACE by every method")
    aR.annotate("", xy=(3, 6.7), xytext=(2, 36),
                arrowprops=dict(arrowstyle="<->", color="0.4", lw=1))
    aR.text(2.5, 22, "genuine\nPES gap\n~30", ha="center", fontsize=8.5, color="0.3")
    fig.suptitle("Diels–Alder: same-methodology check — the MACE↔xTB barrier gap is a real PES difference",
                 y=1.03, fontsize=12.5)
    fig.tight_layout(); FIG.mkdir(exist_ok=True)
    out = FIG / "a2_standardized_neb.png"
    fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


if __name__ == "__main__":
    main()
