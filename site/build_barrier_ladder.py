#!/usr/bin/env python3
"""
build_barrier_ladder.py — Diels–Alder activation barriers by model, estimator and
reactant reference, read directly from the repository's result files.

    python3 site/build_barrier_ladder.py      # -> site/assets/barrier_ladder.png

Nothing is typed in by hand: every plotted value is read from the files listed in
SOURCES below, and the script prints the exact values it plotted. The result files
are only read, never written.

The figure deliberately does NOT treat all barriers as one quantity. Each point
carries two labels:
  * estimator  — constrained relaxed-scan maximum / CI-NEB highest image /
                 Sella saddle (certified or not)
  * reference  — which reactant energy the barrier is measured from:
                 scan start, model reactant relaxed with forming bonds pinned,
                 frozen xTB reactant geometry, or reactant relaxed on that model.
Requires matplotlib + numpy (the same stack as the study plotting scripts).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent / "sweep"
OUT = HERE / "assets" / "barrier_ladder.png"

SOURCES = {
    "neb": SWEEP / "diels_alder/neb/results/neb_results.json",
    "refined": SWEEP / "diels_alder/neb/results/ts_mace_refined.json",
    "phase1c": SWEEP / "diels_alder/phase1c_hessian_results.json",
    "corr": SWEEP / "corrections/results/da_summary.json",
    "phaseC": SWEEP / "corrections/phaseC/reaction_summary.json",
    "phase2": SWEEP / "corrections/phase2/summary.json",
}

# repository plotting conventions (sweep/corrections/phase2_plot.py)
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "savefig.dpi": 200, "font.family": "DejaVu Sans"})
C_SADDLE, C_SCAN, C_TGT, C_BAD = "#2e6f95", "#e07a5f", "#c1121f", "#9aa0a6"
C_NEB = "#5b7083"

# estimator -> marker; reference -> short tag printed beside each point
MARK = {"scan": "s", "neb": "^", "saddle": "o"}
REF = {"scan": "vs scan start", "pinned": "vs pinned-relaxed R",
       "frozen": "vs frozen xTB R", "relaxed": "vs model-relaxed R"}


def load():
    j = {k: json.loads(p.read_text()) for k, p in SOURCES.items()}
    it0 = j["phase2"][0]
    assert abs(it0["reaction"]["barrier_kcal"] - j["phaseC"]["B_barrier"]["barrier"]) < 1e-6, \
        "phase2 iteration 0 is expected to be the Phase C barrier-enriched model"
    xtb = j["neb"]["xtb"]["barrier_kcal"]
    groups = [
        ("Raw MACE-OFF23", [
            ("constrained scan max", "scan", "scan", j["corr"]["baseline"]["barrier"], True),
            ("CI-NEB highest image*", "neb", "pinned", j["neb"]["mace"]["barrier_kcal"], True),
            ("Sella saddle from NEB image", "saddle", "pinned", j["refined"]["barrier_kcal"],
             bool(j["refined"]["is_first_order"])),
            ("Sella saddle from xTB TS", "saddle", "frozen",
             j["phase1c"]["converged_MACE_TS"]["barrier_kcal_mol"], True),
        ]),
        ("Corrected (pairwise Δ)\nconstrained-scan maxima", [
            ("first conservative Δ", "scan", "scan", j["corr"]["delta"]["barrier"], True),
            ("Phase C baseline Δ", "scan", "scan", j["phaseC"]["A_baseline"]["barrier"], True),
            ("generic-data control", "scan", "scan", j["phaseC"]["C_generic"]["barrier"], True),
            ("barrier-enriched Δ", "scan", "scan", j["phaseC"]["B_barrier"]["barrier"], True),
        ]),
        ("Barrier-enriched Δ\ncertified saddle", [
            ("certified saddle", "saddle", "frozen", it0["ts"]["barrier_kcal_frozen_R"],
             bool(it0["ts"]["valid_da_saddle"])),
            ("certified saddle", "saddle", "relaxed", it0["ts"]["barrier_kcal_relaxed_R"],
             bool(it0["ts"]["valid_da_saddle"])),
        ]),
    ]
    iters = {
        "it": [r["iteration"] for r in j["phase2"]],
        "frozen": [r["ts"]["barrier_kcal_frozen_R"] for r in j["phase2"]],
        "relaxed": [r["ts"]["barrier_kcal_relaxed_R"] for r in j["phase2"]],
        "scan": [r["reaction"]["barrier_kcal"] for r in j["phase2"]],
        "valid": [bool(r["ts"]["valid_da_saddle"]) for r in j["phase2"]],
        "n_imag": [r["ts"]["vib"]["n_imaginary"] for r in j["phase2"]],
    }
    return xtb, groups, iters


def colour(est, certified):
    if est == "scan":
        return C_SCAN
    if est == "neb":
        return C_NEB
    return C_SADDLE if certified else C_BAD


def main():
    xtb, groups, iters = load()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 5.6),
                               gridspec_kw={"width_ratios": [1.55, 1]})

    # ---------------- panel A: estimators side by side ------------------- #
    x, ticks, ticklabels, printed = 0.0, [], [], []
    for gname, pts in groups:
        start = x
        for (label, est, ref, val, cert) in pts:
            c = colour(est, cert)
            a.plot(x, val, MARK[est], ms=9, color=c, mec=c,
                   mfc=c if (est != "saddle" or ref != "relaxed") else "white", mew=2)
            a.annotate(f"{val:.2f}", (x, val), xytext=(0, 9), textcoords="offset points",
                       ha="center", fontsize=8.6, color="0.15")
            ticks.append(x)
            ticklabels.append(f"{label}\n({REF[ref]})")
            printed.append((gname.replace("\n", " "), label, est, REF[ref], val, cert))
            x += 1
        a.axvspan(start - 0.45, x - 0.55, color="0.965", zorder=0)
        a.text((start + x - 1) / 2, 41.5, gname, ha="center", va="bottom", fontsize=9.5,
               fontweight="bold", color="0.25")
        x += 0.6
    a.axhline(xtb, color=C_TGT, ls="--", lw=1.4)
    a.text(-0.55, xtb + 0.8, f"xTB reference {xtb:.1f} (verified saddle, −394 cm⁻¹)",
           ha="left", va="bottom", color=C_TGT, fontsize=8.6)
    a.set_xticks(ticks)
    a.set_xticklabels(ticklabels, rotation=58, ha="right", fontsize=8.1)
    a.set_ylim(0, 47)
    a.set_xlim(-0.7, x - 0.6)
    a.set_ylabel("activation barrier (kcal/mol)")
    a.set_title("Diels–Alder barrier by model, estimator and reactant reference",
                fontsize=11, loc="left")

    # ---------------- panel B: iterative correction ---------------------- #
    it = np.array(iters["it"])
    valid = np.array(iters["valid"])
    b.plot(it, iters["scan"], "-s", color=C_SCAN, lw=1.6, ms=6,
           label="constrained-scan max (vs scan start)")
    b.plot(it, iters["relaxed"], ":", color=C_SADDLE, lw=1.2)
    b.plot(it, iters["frozen"], "-", color=C_SADDLE, lw=1.6)
    for k in range(len(it)):
        if valid[k]:
            b.plot(it[k], iters["frozen"][k], "o", ms=8, color=C_SADDLE)
            b.plot(it[k], iters["relaxed"][k], "o", ms=8, mfc="white", mec=C_SADDLE, mew=2)
        else:
            b.plot(it[k], iters["frozen"][k], "X", ms=9, color=C_BAD)
            b.plot(it[k], iters["relaxed"][k], "X", ms=9, color=C_BAD, alpha=0.6)
            b.annotate(f"uncertified\n({iters['n_imag'][k]} imag. modes)",
                       (it[k], iters["frozen"][k]), xytext=(0, -26), textcoords="offset points",
                       ha="center", fontsize=7.8, color="0.4")
    b.plot([], [], "o", ms=8, color=C_SADDLE, label="certified saddle (vs frozen xTB R)")
    b.plot([], [], "o", ms=8, mfc="white", mec=C_SADDLE, mew=2,
           label="certified saddle (vs model-relaxed R)")
    b.plot([], [], "X", ms=9, color=C_BAD, label="UNCERTIFIED stationary point")
    b.axhline(xtb, color=C_TGT, ls="--", lw=1.4, label=f"xTB reference {xtb:.1f}")
    b.set_xticks(it)
    b.set_xlabel("iteration of the correction loop")
    b.set_ylabel("activation barrier (kcal/mol)")
    b.set_ylim(4, 16.5)
    b.set_title("Iterating the correction does not converge", fontsize=11, loc="left")
    b.legend(frameon=False, fontsize=8.1, loc="upper left")

    # shared estimator legend for panel A
    handles = [plt.Line2D([], [], ls="", marker=MARK["scan"], color=C_SCAN, ms=8,
                          label="constrained relaxed-scan maximum"),
               plt.Line2D([], [], ls="", marker=MARK["neb"], color=C_NEB, ms=8,
                          label="CI-NEB highest image (*band convergence\nnot recorded; diagnostic)"),
               plt.Line2D([], [], ls="", marker="o", color=C_SADDLE, ms=8,
                          label="Sella saddle, one imaginary mode"),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec=C_SADDLE, mew=2, ms=8,
                          label="same saddle, model-relaxed reactant")]
    a.legend(handles=handles, frameon=False, fontsize=8.1, loc="upper right",
             bbox_to_anchor=(1.0, 0.86))

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")

    print(f"wrote {OUT}")
    print(f"xTB reference: {xtb}")
    for g, label, est, ref, val, cert in printed:
        status = cert if est == "saddle" else "n/a (not a saddle estimator)"
        print(f"  [{g}] {label:28s} {est:6s} {ref:22s} {val:8.3f}  first-order saddle: {status}")
    for k in range(len(it)):
        print(f"  iter {it[k]}: frozen-R {iters['frozen'][k]:.3f}  relaxed-R "
              f"{iters['relaxed'][k]:.3f}  scan {iters['scan'][k]:.3f}  certified={valid[k]} "
              f"(n_imag={iters['n_imag'][k]})")


if __name__ == "__main__":
    main()
