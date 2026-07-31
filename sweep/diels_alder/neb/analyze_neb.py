#!/usr/bin/env python3
"""
analyze_neb.py
==============
Extract the scientific value from the finished MACE-OFF23 vs GFN2-xTB Diels–Alder
benchmark. Produces three publication-quality figures + a summary JSON.

  1. energy-difference profile   ΔE(s) = E_MACE(s) − E_xTB(s) along the reaction
     coordinate, using the BOTH-relaxed concerted scans (xTB built-in scan +
     `mace_scan.py`), and its shape (constant / linear / localized).
  2. force agreement             per-configuration force RMSE and cosine
     similarity on *rattled* reaction-path geometries (gen_rattled.py) so neither
     potential sits at its own minimum (a fair comparison).
  3. bond evolution              forming bonds b1,b2 (MACE NEB band) + asynchrony.

Pure analysis (numpy/matplotlib/ase); runs in system python3.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import read
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DA = HERE.parent
FIG = DA / "figures"
XTB_C, MACE_C, D_C = "#c1121f", "#2e6f95", "#6a4c93"
TS_BAND = (1.9, 2.4)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200, "font.family": "DejaVu Sans",
})


def task1_energy_diff():
    x = pd.read_csv(DA / "cscan_profile.csv")            # xTB relaxed scan
    m = pd.read_csv(RES / "mace_cscan_profile.csv")      # MACE relaxed scan (both-relaxed)
    n = min(len(x), len(m))
    c = x["forming_CC_A"].to_numpy()[:n]
    relX = x["rel_E_kcal_mol"].to_numpy()[:n]
    relM = m["rel_E_kcal_mol"].to_numpy()[:n]
    D = relM - relX
    ts_mask = (c >= TS_BAND[0]) & (c <= TS_BAND[1])
    imax = int(np.argmax(np.abs(D)))
    # linearity of D vs bond-formation progress (reactant=0 .. product=1)
    prog = (c[0] - c) / (c[0] - c[-1])
    A = np.vstack([prog, np.ones_like(prog)]).T
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)
    r2 = 1 - np.sum((D - A @ coef)**2) / np.sum((D - D.mean())**2)

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.axvspan(*TS_BAND, color="0.92", zorder=0, label="TS region")
    ax.plot(c, relX, "-", color=XTB_C, lw=1.3, alpha=.55, label="rel. E  GFN2-xTB")
    ax.plot(c, relM, "-", color=MACE_C, lw=1.3, alpha=.55, label="rel. E  MACE-OFF23")
    ax.plot(c, D, "-o", color=D_C, lw=2.3, ms=4, label="ΔE = MACE − xTB")
    ax.axhline(0, color="#9aa5b1", lw=.7, ls=":")
    ax.scatter([c[imax]], [D[imax]], s=160, marker="*", color=D_C,
               edgecolor="white", zorder=6)
    ax.annotate(f"peak ΔE {D[imax]:+.1f} kcal/mol @ {c[imax]:.2f} Å",
                (c[imax], D[imax]), textcoords="offset points", xytext=(8, -14),
                fontsize=9, color=D_C)
    ax.invert_xaxis(); ax.set_xlabel("forming C–C distance (Å)   [reactant → product]")
    ax.set_ylabel("energy (kcal/mol)")
    ax.set_title("Energy-difference profile along the reaction path (both relaxed)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "neb_energy_diff.png", bbox_inches="tight")
    plt.close(fig)
    return dict(D_reactant=float(D[0]), D_TS=float(D[ts_mask].max()),
                D_product=float(D[-1]), D_peak=float(D[imax]),
                coord_at_peak=float(c[imax]), r2_linear_in_progress=float(r2))


def task2_forces():
    m = np.load(RES / "force_mace.npz"); x = np.load(RES / "force_xtb.npz")
    c = m["coord"]; Fm, Fx = m["forces"], x["forces"]
    rmse = np.sqrt(((Fm - Fx)**2).reshape(len(c), -1).mean(1))
    cos = np.array([fm.ravel() @ fx.ravel() /
                    (np.linalg.norm(fm) * np.linalg.norm(fx) + 1e-12)
                    for fm, fx in zip(Fm, Fx)])
    o = np.argsort(-c)
    c, rmse, cos = c[o], rmse[o], cos[o]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)
    for ax in (a1, a2):
        ax.axvspan(*TS_BAND, color="0.92", zorder=0)
    a1.scatter(c, rmse, s=26, color="#e07a5f", edgecolor="white", lw=.4)
    a1.axhline(rmse.mean(), color="#e07a5f", ls="--", lw=1,
               label=f"mean {rmse.mean():.2f} eV/Å")
    a1.set_ylabel("force RMSE (eV/Å)"); a1.legend(frameon=False, fontsize=9)
    a1.set_title("MACE-OFF23 vs GFN2-xTB forces on rattled reaction-path geometries")
    a2.scatter(c, cos, s=26, color="#3d5a80", edgecolor="white", lw=.4)
    a2.axhline(cos.mean(), color="#3d5a80", ls="--", lw=1,
               label=f"mean {cos.mean():.3f}")
    a2.axhline(1.0, color="#9aa5b1", lw=.7, ls=":")
    a2.set_ylim(min(0.9, cos.min()-0.02), 1.003)
    a2.set_ylabel("force cosine similarity"); a2.legend(frameon=False, fontsize=9)
    a2.set_xlabel("forming C–C distance (Å)"); a2.invert_xaxis()
    fig.tight_layout(); fig.savefig(FIG / "neb_force_agreement.png", bbox_inches="tight")
    plt.close(fig)
    return dict(rmse_mean=float(rmse.mean()), rmse_max=float(rmse.max()),
                cos_mean=float(cos.mean()), cos_min=float(cos.min()),
                n_geoms=int(len(c)))


def task3_bonds():
    frames = read(str(RES / "neb_mace.xyz"), index=":")
    mep = pd.read_csv(RES / "mep_mace.csv")
    b1, b2, co = [], [], []
    for i, at in enumerate(frames):
        if not mep.physical.iloc[i]:
            continue
        p = at.get_positions()
        b1.append(np.linalg.norm(p[0]-p[5])); b2.append(np.linalg.norm(p[3]-p[4]))
        co.append(0.5*(b1[-1]+b2[-1]))
    o = np.argsort(-np.array(co))
    b1, b2, co = np.array(b1)[o], np.array(b2)[o], np.array(co)[o]
    x = pd.read_csv(DA / "cscan_profile.csv")
    cx = x["forming_CC_A"].to_numpy()
    ts = json.loads((RES / "ts_mace_refined.json").read_text())["ts_forming_A"]

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.axvspan(*TS_BAND, color="0.92", zorder=0, label="TS region")
    ax.plot(co, b1, "-o", color=MACE_C, ms=4, lw=1.7, label="MACE forming bond 1 (C1–C6)")
    ax.plot(co, b2, "--s", color=MACE_C, ms=4, lw=1.7, mfc="white",
            label="MACE forming bond 2 (C4–C5)")
    ax.plot(cx, cx, "-", color=XTB_C, lw=1.5, alpha=.8, label="xTB (synchronous, b1=b2)")
    ax.scatter([0.5*sum(ts)]*2, ts, s=110, marker="*", color=MACE_C,
               edgecolor="white", zorder=6)
    ax.annotate(f"MACE TS  {ts[0]:.2f}/{ts[1]:.2f} Å\nΔ={abs(ts[0]-ts[1]):.2f} Å (asynchronous)",
                (0.5*sum(ts), max(ts)), textcoords="offset points", xytext=(8, 8),
                fontsize=9, color=MACE_C)
    ax.set_xlabel("reaction coordinate — mean forming C–C (Å)")
    ax.set_ylabel("individual forming C–C distance (Å)")
    ax.set_title("Forming-bond evolution & synchronicity")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(FIG / "neb_bond_evolution.png", bbox_inches="tight")
    plt.close(fig)
    return dict(mace_ts_bonds=ts, mace_ts_asynchrony=float(abs(ts[0]-ts[1])),
                mace_path_max_asynchrony=float(np.abs(b1-b2).max()),
                xtb_synchronous=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    s1, s2, s3 = task1_energy_diff(), task2_forces(), task3_bonds()
    (RES / "analysis_summary.json").write_text(
        json.dumps({"energy_diff": s1, "forces": s2, "bonds": s3}, indent=2))
    print("Task 1 — ΔE profile (both relaxed):")
    print(f"  reactant {s1['D_reactant']:+.1f}, TS {s1['D_TS']:+.1f}, "
          f"product {s1['D_product']:+.1f}; peak {s1['D_peak']:+.1f} @ "
          f"{s1['coord_at_peak']:.2f} Å; linear-in-progress R²={s1['r2_linear_in_progress']:.2f}")
    print("Task 2 — forces (rattled path geoms):")
    print(f"  RMSE {s2['rmse_mean']:.2f} (max {s2['rmse_max']:.2f}) eV/Å; "
          f"cosine {s2['cos_mean']:.3f} (min {s2['cos_min']:.3f}), n={s2['n_geoms']}")
    print("Task 3 — bonds:")
    print(f"  MACE TS asynchrony {s3['mace_ts_asynchrony']:.2f} Å; xTB synchronous")
    print(f"  figures -> {FIG}/neb_energy_diff.png, neb_force_agreement.png, neb_bond_evolution.png")


if __name__ == "__main__":
    main()
