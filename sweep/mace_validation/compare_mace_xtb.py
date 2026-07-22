#!/usr/bin/env python3
"""
compare_mace_xtb.py
===================
First step of the MACE-OFF23 direction: compare the **MACE-OFF23** foundation
MLIP against **GFN2-xTB** on the water-droplet trajectories we already generated
(sweep/large/runs/n{20,30,50}/low). The goal is to see *how far apart the two
potential-energy surfaces are, and where* — before building any enhanced
sampling on top of MACE.

Important framing: MACE-OFF23 is trained on DFT (ωB97M-D3BJ) and GFN2-xTB is
semiempirical, so this is **not** "validate MACE against xTB as ground truth" —
it measures the discrepancy between the two surrogates (MACE is expected to be
the more accurate one). We compare the physically meaningful quantities:

  * FORCES — same units (eV/Å); parity, MAE/RMSE, Pearson r on all components.
  * RELATIVE ENERGIES — absolute energies use different atomic references, but
    for a fixed-composition trajectory those references cancel, so ΔE (energy
    minus its per-trajectory mean) is directly comparable.

Per-atom force cost is also recorded (MACE vs xTB timing) since "does the
surrogate actually save time at this size?" is a live question.

Outputs (analysis/, figures/):
  cv? no — mace_xtb_forces_<tag>.csv, mace_xtb_energy_<tag>.csv,
  a per-run metrics line, and parity/energy figures.

Run: micromamba run -n xtb python compare_mace_xtb.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "large" / "runs"
OUT = HERE / "analysis"
FIG = HERE / "figures"
EV_PER_A = "eV/Å"
SIZES = [20, 30, 50]


def get_calcs():
    from mace.calculators import mace_off
    from xtb.ase.calculator import XTB
    mace = mace_off(model="small", device="cpu", default_dtype="float64")
    return mace, XTB


def evaluate(trj: Path, mace, XTB, stride: int = 1):
    frames = read(str(trj), index=f"::{stride}", format="xyz")
    if not isinstance(frames, list):
        frames = [frames]
    Em, Ex, Fm, Fx = [], [], [], []
    tm = tx = 0.0
    for at in frames:
        a = at.copy(); a.calc = mace
        t = time.time(); Em.append(a.get_potential_energy()); Fm.append(a.get_forces()); tm += time.time() - t
        b = at.copy(); b.calc = XTB(method="GFN2-xTB")
        t = time.time(); Ex.append(b.get_potential_energy()); Fx.append(b.get_forces()); tx += time.time() - t
    Em, Ex = np.array(Em), np.array(Ex)
    Fm, Fx = np.concatenate(Fm).ravel(), np.concatenate(Fx).ravel()
    natoms = len(frames[0])
    return dict(Em=Em, Ex=Ex, Fm=Fm, Fx=Fx, n=natoms, nframes=len(frames),
                t_mace=tm / len(frames), t_xtb=tx / len(frames))


def metrics(d):
    # forces (eV/A)
    dF = d["Fm"] - d["Fx"]
    f_mae = np.abs(dF).mean(); f_rmse = np.sqrt((dF ** 2).mean())
    f_r = np.corrcoef(d["Fm"], d["Fx"])[0, 1]
    # relative energies (subtract mean) in eV and meV/atom
    em = d["Em"] - d["Em"].mean(); ex = d["Ex"] - d["Ex"].mean()
    e_rmse = np.sqrt(((em - ex) ** 2).mean())
    e_r = np.corrcoef(em, ex)[0, 1] if len(em) > 1 else float("nan")
    return dict(f_mae=f_mae, f_rmse=f_rmse, f_r=f_r,
                e_rmse_eV=e_rmse, e_rmse_meV_atom=1000 * e_rmse / d["n"], e_r=e_r)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {20: "tab:green", 30: "tab:orange", 50: "tab:red"}

    # force parity
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = 0
    for n, d in results.items():
        ax.scatter(d["Fx"], d["Fm"], s=3, alpha=0.25, color=colors[n], label=f"n={n}")
        lim = max(lim, np.abs(d["Fx"]).max(), np.abs(d["Fm"]).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set_xlabel(f"GFN2-xTB force component ({EV_PER_A})")
    ax.set_ylabel(f"MACE-OFF23 force component ({EV_PER_A})")
    ax.set_title("Force parity: MACE-OFF23 vs GFN2-xTB"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "force_parity.png", dpi=140); plt.close(fig)

    # relative energy along trajectory
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.6), squeeze=False)
    for ax, (n, d) in zip(axes[0], results.items()):
        t = np.arange(d["nframes"])
        ax.plot(t, d["Em"] - d["Em"].mean(), color="tab:blue", label="MACE-OFF23")
        ax.plot(t, d["Ex"] - d["Ex"].mean(), color="tab:red", label="GFN2-xTB")
        ax.set_title(f"n={n} relative energy"); ax.set_xlabel("frame")
        ax.set_ylabel("E - mean (eV)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "relative_energy.png", dpi=140); plt.close(fig)
    print(f"  wrote {FIG/'force_parity.png'}, {FIG/'relative_energy.png'}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading MACE-OFF23 (small) + xTB ...", flush=True)
    mace, XTB = get_calcs()
    results, rows = {}, []
    for n in SIZES:
        trj = RUNS / f"n{n}" / "low" / "xtb.trj"
        if not trj.exists():
            print(f"  (missing {trj}, skip)"); continue
        d = evaluate(trj, mace, XTB)
        m = metrics(d)
        results[n] = d
        rows.append({"n": n, "atoms": d["n"], "frames": d["nframes"], **m,
                     "t_mace_s": d["t_mace"], "t_xtb_s": d["t_xtb"],
                     "mace_slowdown": d["t_mace"] / d["t_xtb"]})
        print(f"n={n}: forces MAE={m['f_mae']:.3f} RMSE={m['f_rmse']:.3f} eV/A r={m['f_r']:.3f} | "
              f"relE RMSE={m['e_rmse_meV_atom']:.2f} meV/atom r={m['e_r']:.3f} | "
              f"MACE {d['t_mace']*1000:.0f} ms/frame ({d['t_mace']/d['t_xtb']:.1f}x xTB)")
    pd.DataFrame(rows).to_csv(OUT / "mace_xtb_metrics.csv", index=False)
    print(f"  wrote {OUT/'mace_xtb_metrics.csv'}")
    if results:
        plot(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
