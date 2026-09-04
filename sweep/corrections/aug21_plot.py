#!/usr/bin/env python3
"""
aug21_plot.py — figures + summary table for the Aug 21 handoff-cost benchmark.

Reads aug21_handoff/results.json only; every number is taken from the generated
record. Runs in system python (matplotlib).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "aug21_handoff"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

C_OK, C_FAIL, C_BASE, C_CTRL = "#2e6f95", "#c1121f", "#1f2933", "#5b8c5a"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "savefig.dpi": 200, "font.family": "DejaVu Sans"})

d = json.loads((OUT / "results.json").read_text())
recs = {r["case"]: r for r in d["records"]}
base = recs["baseline_scan_sella"]
ctrl = recs["control_idpp_interp"]
hand = [recs[f"handoff_iter{n:02d}"] for n in range(6)]

base_total = base["total_marginal_xtb_calls"]
ctrl_total = ctrl["total_marginal_xtb_calls"]


def fig1():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(6)
    tot = [h["total_marginal_xtb_calls"] for h in hand]
    ok = [h["reached_reference_saddle"] for h in hand]

    ax.axhline(base_total, color=C_BASE, lw=2, ls="-",
               label=f"xTB-only baseline: scan+Sella = {base_total}")
    ax.axhline(ctrl_total, color=C_CTRL, lw=1.6, ls="--",
               label=f"IDPP interpolation control = {ctrl_total}")

    for xi, (t, good) in enumerate(zip(tot, ok)):
        ax.plot([xi, xi], [0, t], color="0.85", lw=1, zorder=0)
        ax.scatter([xi], [t], s=150, zorder=3,
                   color=C_OK if good else "white",
                   edgecolors=C_OK if good else C_FAIL, linewidths=2,
                   marker="o" if good else "X")
        ax.annotate(f"{t}", (xi, t), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9.5,
                    color=C_OK if good else C_FAIL)

    ax.scatter([], [], s=110, color=C_OK, label="reached the xTB TS")
    ax.scatter([], [], s=110, marker="X", color="white", edgecolors=C_FAIL,
               linewidths=2, label="failed (not converged / wrong saddle)")

    ax.set_xticks(x)
    ax.set_xlabel("handoff iteration $n$  (prior adaptive cost $7n$)")
    ax.set_ylabel("total marginal xTB evaluations\n$7n$ + handoff calls")
    ax.set_ylim(0, max(max(tot), base_total) * 1.18)
    ax.set_title("Cost to reach the verified xTB transition state\n"
                 "Phase C correction is a separate 106-reference sunk cost",
                 fontsize=11.5)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "aug21_total_cost.png", bbox_inches="tight")
    plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(6)
    h_only = [h["handoff_xtb_calculate_calls"] for h in hand]
    ok = [h["reached_reference_saddle"] for h in hand]

    ax.bar(x, h_only, 0.55,
           color=[C_OK if g else "#f0d3d1" for g in ok],
           edgecolor=[C_OK if g else C_FAIL for g in ok], linewidth=1.4)
    for xi, (v, g) in enumerate(zip(h_only, ok)):
        ax.annotate(f"{v}" + ("" if g else "\nfailed"), (xi, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=C_OK if g else C_FAIL)

    ax.axhline(base["handoff_xtb_calculate_calls"], color=C_BASE, lw=1.8,
               label=f"baseline Sella only = {base['handoff_xtb_calculate_calls']} "
                     f"(after a {base['baseline_scan_xtb_calls']}-call scan)")
    ax.axhline(ctrl["handoff_xtb_calculate_calls"], color=C_CTRL, lw=1.5, ls="--",
               label=f"IDPP control = {ctrl['handoff_xtb_calculate_calls']}")
    ax.set_xticks(x)
    ax.set_xlabel("handoff iteration $n$")
    ax.set_ylabel("xTB evaluations after handoff")
    ax.set_title("Starting-point quality alone\n(excludes the cost of producing that start)",
                 fontsize=11.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "aug21_handoff_only.png", bbox_inches="tight")
    plt.close(fig)


def table():
    hdr = ("| case | start type | prior refs $7n$ | handoff calls | **total** | "
           "reached TS | strict Hessian | fmax | forming Å | barrier |")
    sep = "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [hdr, sep]
    order = [base] + hand + [ctrl]
    for r in order:
        n = r.get("handoff_iteration")
        label = r["case"].replace("handoff_", "").replace("_", " ")
        prior = r["prior_phase2_unique_refs"]
        prior_s = f"{prior}" if n is not None else f"{r['baseline_scan_xtb_calls']} (scan)"
        reach = "yes" if r["reached_reference_saddle"] else "**no**"
        strict = "clean" if r["certified_ts"] else f"{r['n_imaginary']} imag"
        lines.append(
            f"| {label} | {r['start_type']} | {prior_s} | "
            f"{r['handoff_xtb_calculate_calls']} | **{r['total_marginal_xtb_calls']}** | "
            f"{reach} | {strict} | {r['final_fmax']:.1e} | "
            f"{r['forming_CC_mean']:.4f} | {r['barrier_kcal_mol']:.2f} |")
    md = "\n".join(lines) + "\n"
    (OUT / "summary_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    fig1(); fig2(); table()
    succ = [(h["handoff_iteration"], h["total_marginal_xtb_calls"])
            for h in hand if h["reached_reference_saddle"]]
    best = min(succ, key=lambda t: t[1])
    print(f"baseline total          : {base_total}")
    print(f"IDPP control total      : {ctrl_total}")
    print(f"successful handoffs     : {succ}")
    print(f"cheapest handoff        : iter{best[0]} at {best[1]} "
          f"({base_total/best[1]:.1f}x cheaper than baseline)")
    print("wrote aug21_total_cost.png, aug21_handoff_only.png, summary_table.md")
