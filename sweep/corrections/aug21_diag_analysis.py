#!/usr/bin/env python3
"""
aug21_diag_analysis.py — Part C: relate starting-state descriptors to outcome.

Exploratory only: 7 primary starting points, 4 successes. No classifier, no
feature selection, no significance claims. Rank ordering, twin-pair comparison,
and clearly-labelled small-n Spearman coefficients.

Reads aug21_diagnostics/{descriptors,reproducibility}.json. System python.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DIAG = HERE / "aug21_diagnostics"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

C_OK, C_FAIL = "#2e6f95", "#c1121f"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "savefig.dpi": 200, "font.family": "DejaVu Sans"})

rows = json.loads((DIAG / "descriptors.json").read_text())
D = {r["case"]: r for r in rows}
order = ["IDPP", "iter00", "iter01", "iter02", "iter03", "iter04", "iter05"]

DESC = [
    ("xtb_fmax", "xTB |F|max at start (eV/Å)"),
    ("xtb_fnorm", "xTB |F| norm (eV/Å)"),
    ("xtb_force_RC_fraction", "xTB force fraction along RC"),
    ("xtb_E_rel_to_TS_kcal", "E − E(TS) (kcal/mol)"),
    ("xtb_lowest_mode_cm", "xTB lowest mode (cm⁻¹)"),
    ("xtb_n_imaginary", "xTB imaginary modes"),
    ("xtb_reaction_mode_overlap", "xTB reaction-mode overlap"),
    ("forming_CC_mean", "forming C–C mean (Å)"),
    ("forming_asymmetry", "forming-bond asymmetry (Å)"),
    ("kabsch_rmsd_to_xtb_ts", "Kabsch RMSD to TS (Å)"),
    ("signature_delta_to_xtb_ts", "perm-invariant Δ to TS"),
    ("disp_proj_fraction", "displacement ∥ reaction mode"),
    ("mace_xtb_force_rmse", "MACE↔xTB force RMSE"),
    ("mace_corr_rc_overlap", "corrected-MACE RC overlap"),
]

TWINS = [("iter01", "iter04"), ("iter05", "iter03")]


def table():
    lines = ["| case | success | handoff calls | total | " +
             " | ".join(lbl for _, lbl in DESC) + " |",
             "|---|:---:|:---:|:---:|" + "---|" * len(DESC)]
    for c in order:
        r = D[c]
        cells = []
        for k, _ in DESC:
            v = r.get(k)
            cells.append("—" if v is None else
                         (f"{v:.4g}" if isinstance(v, float) else str(v)))
        lines.append(f"| {c} | {'yes' if r['success'] else '**no**'} | "
                     f"{r['handoff_calls']} | {r['total_marginal_cost']} | "
                     + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    (DIAG / "descriptor_table.md").write_text(md)
    return md


def twin_report():
    out = ["\n=== TWIN PAIRS: near-identical starts, opposite outcomes ==="]
    twin_json = {}
    for a, b in TWINS:
        ra, rb = D[a], D[b]
        diffs = {}
        for k, _ in DESC:
            va, vb = ra.get(k), rb.get(k)
            if va is None or vb is None or not isinstance(va, (int, float)):
                continue
            diffs[k] = abs(va - vb) / (max(abs(va), abs(vb)) + 1e-12)
        med = float(np.median(list(diffs.values())))
        out.append(f"  {a} ({ra['handoff_calls']} calls, "
                   f"{'OK' if ra['success'] else 'FAIL'})  vs  "
                   f"{b} ({rb['handoff_calls']} calls, "
                   f"{'OK' if rb['success'] else 'FAIL'})")
        out.append(f"    median relative descriptor difference: {med*100:.1f}%")
        out.append(f"    cost ratio: {max(ra['handoff_calls'],rb['handoff_calls'])/min(ra['handoff_calls'],rb['handoff_calls']):.1f}x")
        twin_json[f"{a}_vs_{b}"] = {
            "median_relative_descriptor_diff": med,
            "per_descriptor_relative_diff": diffs,
            "outcomes": {a: ra["success"], b: rb["success"]},
            "calls": {a: ra["handoff_calls"], b: rb["handoff_calls"]},
        }
    (DIAG / "twin_pairs.json").write_text(json.dumps(twin_json, indent=2))
    return "\n".join(out)


def correlations():
    succ = [c for c in order if D[c]["success"]]
    out = [f"\n=== Spearman vs handoff calls, SUCCESSFUL cases only (n={len(succ)}) ===",
           f"  cases: {succ}",
           "  n=4 — reported because requested; NOT evidence of anything."]
    res = {}
    y = [D[c]["handoff_calls"] for c in succ]
    for k, lbl in DESC:
        x = [D[c].get(k) for c in succ]
        if any(v is None for v in x):
            continue
        rho, p = spearmanr(x, y)
        res[k] = {"rho": float(rho), "p": float(p), "n": len(succ)}
        out.append(f"    {lbl:<34} rho={rho:+.2f}  p={p:.2f}")
    (DIAG / "spearman_successes.json").write_text(json.dumps(res, indent=2))
    return "\n".join(out)


def figure():
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # --- left: the classic candidate rule (residual force) fails ----------- #
    for c in order:
        r = D[c]
        aL.scatter(r["xtb_fmax"], r["handoff_calls"], s=140, zorder=3,
                   color=C_OK if r["success"] else "white",
                   edgecolors=C_OK if r["success"] else C_FAIL, linewidths=2,
                   marker="o" if r["success"] else "X")
        aL.annotate(c, (r["xtb_fmax"], r["handoff_calls"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8.5,
                    color=C_OK if r["success"] else C_FAIL)
    aL.set_yscale("log")
    aL.set_xlabel("xTB |F|max at the handoff geometry (eV/Å)")
    aL.set_ylabel("xTB evaluations after handoff")
    aL.set_title("The intuitive rule fails\nlow residual force does not predict success",
                 fontsize=11)
    aL.scatter([], [], s=110, color=C_OK, label="reached xTB TS")
    aL.scatter([], [], s=110, marker="X", color="white", edgecolors=C_FAIL,
               linewidths=2, label="failed")
    aL.legend(frameon=False, fontsize=9, loc="upper right")

    # --- right: twin pairs ------------------------------------------------- #
    keys = ["xtb_fmax", "xtb_fnorm", "xtb_lowest_mode_cm",
            "xtb_reaction_mode_overlap", "forming_CC_mean",
            "kabsch_rmsd_to_xtb_ts", "disp_proj_fraction"]
    xs = np.arange(len(keys))
    for (a, b), mk in zip(TWINS, ["o", "s"]):
        for case, ls in [(a, "-"), (b, "--")]:
            vals = []
            for k in keys:
                pair = [abs(D[a][k]), abs(D[b][k])]
                m = max(pair) + 1e-12
                vals.append(abs(D[case][k]) / m)
            aR.plot(xs, vals, ls, marker=mk, ms=6, lw=1.6,
                    color=C_OK if D[case]["success"] else C_FAIL,
                    label=f"{case} ({D[case]['handoff_calls']} calls)")
    aR.set_xticks(xs)
    aR.set_xticklabels(["|F|max", "|F|", "low mode", "RC ovlp", "form C–C",
                        "RMSD", "disp∥"], rotation=30, ha="right", fontsize=8.5)
    aR.set_ylabel("descriptor, normalised within pair")
    aR.set_ylim(0, 1.15)
    aR.set_title("Two near-twin pairs (RMSD 0.007 / 0.022 Å)\n"
                 "indistinguishable starts, opposite outcomes", fontsize=11)
    aR.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG / "aug21_descriptor_diagnosis.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print(table())
    print(twin_report())
    print(correlations())
    figure()
    rep = json.loads((DIAG / "reproducibility.json").read_text())["summary"]
    print("\n=== reproducibility summary ===")
    for k, v in rep.items():
        print(f"  {k}: calls={v['calls']} median={v['calls_median']:.0f} "
              f"range={v['calls_range']} deterministic={v['deterministic']}")
    print("\nwrote descriptor_table.md, twin_pairs.json, spearman_successes.json, "
          "figures/aug21_descriptor_diagnosis.png")
