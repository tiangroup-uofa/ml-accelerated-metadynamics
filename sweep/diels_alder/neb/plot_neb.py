#!/usr/bin/env python3
"""
plot_neb.py
===========
Publication-quality comparison of the Diels–Alder reaction profile:
MACE-OFF23 (CI-NEB) vs GFN2-xTB (built-in path search). Reads results/.

Left  : minimum energy path — relative energy vs forming C–C distance, both
        methods, transition states marked, barriers annotated.
Right : barrier and reaction energy side by side.
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
FIG = HERE.parent / "figures"
XTB_C, MACE_C = "#c1121f", "#2e6f95"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200, "font.family": "DejaVu Sans",
})


def main() -> int:
    res = json.loads((RES / "neb_results.json").read_text())
    x = res["xtb"]; mc = res["mace"]

    # xTB curve from the built-in path (scan); MACE curve from the NEB band (physical images)
    xc, xe = np.array(x["coord_A"]), np.array(x["rel_kcal"])
    md = pd.read_csv(RES / "mep_mace.csv")
    md = md[md.physical == 1]
    mcx, mce = md.forming_CC_A.to_numpy(), md.rel_E_kcal.to_numpy()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.75, 1]})

    for (cx, ce, c, lab, bar, tsf, off) in [
        (xc, xe, XTB_C, "GFN2-xTB (built-in path)", x["barrier_kcal"],
         0.5*sum(x["ts_forming_A"]), (14, 6)),
        (mcx, mce, MACE_C, "MACE-OFF23 (CI-NEB)", mc["barrier_kcal"],
         0.5*sum(mc["ts_forming_A"]), (18, -22)),
    ]:
        order = np.argsort(-cx)                       # reactant (large d) -> product
        axL.plot(cx[order], ce[order], "-o", color=c, ms=4, lw=1.8, label=lab)
        ti = int(np.argmax(ce))
        axL.scatter([cx[ti]], [ce[ti]], s=200, marker="*", color=c,
                    edgecolor="white", linewidth=.8, zorder=6)
        axL.annotate(f"{bar:.1f} kcal/mol\n@ {tsf:.2f} Å", (cx[ti], ce[ti]),
                     textcoords="offset points", xytext=off, fontsize=9, color=c)
    axL.axhline(0, color="#9aa5b1", lw=.7, ls=":")
    axL.invert_xaxis()
    axL.set_ylim(top=46)
    axL.set_xlabel("forming C–C distance (Å)")
    axL.set_ylabel("relative energy (kcal/mol)")
    axL.set_title("Diels–Alder minimum energy path", pad=12)
    axL.legend(frameon=False, loc="lower left", fontsize=10)

    labels = ["barrier", "reaction\nenergy"]
    xpos = np.arange(len(labels)); w = 0.36
    for i, (src, c, lab) in enumerate([(x, XTB_C, "GFN2-xTB"), (mc, MACE_C, "MACE-OFF23")]):
        vals = [src["barrier_kcal"], src["reaction_energy_kcal"]]
        bars = axR.bar(xpos + (i - 0.5) * w, vals, w, color=c, label=lab, alpha=.9)
        for b, v in zip(bars, vals):
            axR.annotate(f"{v:.1f}", (b.get_x()+b.get_width()/2, v), ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=9,
                         xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
    axR.axhline(0, color="#333", lw=.8)
    axR.set_xticks(xpos); axR.set_xticklabels(labels)
    axR.set_ylabel("kcal/mol"); axR.set_title("Energetics")
    axR.legend(frameon=False, fontsize=9)

    fig.suptitle("Diels–Alder: MACE-OFF23 vs GFN2-xTB (no fine-tuning)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "neb_da_mace_vs_xtb.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
