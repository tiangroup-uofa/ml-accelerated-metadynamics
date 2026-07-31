#!/usr/bin/env python3
"""
converge_neb.py
===============
Task 4 — is the MACE-OFF23 NEB barrier numerically converged w.r.t. the number of
NEB images? Re-runs the *same* workflow (concerted-scan seed, k=0.5, forming-bond-
pinned reactant) at several image counts and reports the barrier spread.

MACE-only (no xtb-python in the process → no OpenMP clash). Output:
results/barrier_convergence.csv + figures/neb_barrier_convergence.png.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
from ase.io import read
from calculators import make_factory
from neb import full_workflow

HERE = Path(__file__).resolve().parent
DA = HERE.parent
FIG = DA / "figures"
IMAGE_COUNTS = [9, 11, 13, 17, 21]


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    import torch; torch.set_num_threads(4)
    reactant = read(str(DA / "start.xyz")); product = read(str(DA / "end.xyz"))
    seed = [read(str(p)) for p in sorted((DA / "cscan").glob("f_*.xyz"))]

    rows = []
    for n in IMAGE_COUNTS:
        make = make_factory("mace", dtype="float64")
        _, res = full_workflow(reactant, product, make, n_images=n,
                               fmax=0.05, seed_band=seed, k=0.5)
        tf = 0.5 * sum(res["ts_forming_A"])
        rows.append((n, res["barrier_kcal"], res["reaction_energy_kcal"], tf))
        print(f"  images={n:2d}: barrier {res['barrier_kcal']:6.1f} kcal/mol, "
              f"ΔE {res['reaction_energy_kcal']:6.1f}, TS {tf:.2f} Å")

    n, bar, dE, ts = map(np.array, zip(*rows))
    spread = bar.max() - bar.min()
    out = HERE / "results" / "barrier_convergence.csv"
    with open(out, "w") as fh:
        fh.write("n_images,barrier_kcal,reaction_energy_kcal,ts_forming_A\n")
        for row in rows:
            fh.write(",".join(f"{v:.3f}" for v in row) + "\n")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 200})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(n, bar, "-o", color="#2e6f95", ms=6, lw=1.8)
    ax.axhspan(bar.mean()-1, bar.mean()+1, color="#2e6f95", alpha=.10)
    ax.axhline(bar.mean(), color="#2e6f95", ls="--", lw=.9,
               label=f"mean {bar.mean():.1f} ± {bar.std():.1f} kcal/mol")
    for xi, yi in zip(n, bar):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("number of NEB images"); ax.set_ylabel("activation barrier (kcal/mol)")
    ax.set_title(f"MACE-OFF23 NEB barrier convergence (spread {spread:.1f} kcal/mol)")
    ax.set_xticks(n); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "neb_barrier_convergence.png", bbox_inches="tight")
    print(f"\n  barrier spread {spread:.1f} kcal/mol over {IMAGE_COUNTS} images "
          f"-> {'CONVERGED' if spread < 2.0 else 'check'}")
    print(f"  wrote {out}, {FIG/'neb_barrier_convergence.png'}")


if __name__ == "__main__":
    main()
