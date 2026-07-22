#!/usr/bin/env python3
"""
plot_reaction_profile.py
========================
Reaction-energy profile for the concerted Diels-Alder scan: single-point the
constrained-scan geometries (cscan/f_*.xyz) and plot GFN2-xTB energy vs the
forming C-C bond distance, marking the transition state.

Run: micromamba run -n xtb python plot_reaction_profile.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
from xtb.ase.calculator import XTB

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
HARTREE = 27.211386245988  # eV per Hartree -> we work in eV then convert
KCAL = 23.060548           # kcal/mol per eV


def main() -> int:
    files = sorted((HERE / "cscan").glob("f_*.xyz"))
    if not files:
        print("no cscan/f_*.xyz — run the concerted scan first"); return 1
    calc = XTB(method="GFN2-xTB")
    r, E = [], []
    for f in files:
        a = read(str(f)); a.calc = calc
        E.append(a.get_potential_energy())               # eV
        p = a.get_positions()
        r.append(0.5 * (np.linalg.norm(p[0] - p[5]) + np.linalg.norm(p[3] - p[4])))
    r = np.array(r); E = np.array(E)
    rel = (E - E[0]) * KCAL                               # kcal/mol rel. reactant
    imax = int(np.argmax(rel))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(r, rel, "-o", color="#2dd4bf", ms=4, lw=1.8)
    ax.scatter([r[imax]], [rel[imax]], s=90, facecolor="#f43f5e",
               edgecolor="white", zorder=5, label=f"TS  {rel[imax]:.1f} kcal/mol")
    ax.axhline(0, color="#94a3b8", lw=0.6, ls=":")
    ax.annotate(f"TS: {r[imax]:.2f} Å,  {rel[imax]:.1f} kcal/mol",
                (r[imax], rel[imax]), textcoords="offset points", xytext=(10, -4),
                fontsize=9)
    ax.set_xlabel("forming C–C distance (Å)  [both bonds, concerted]")
    ax.set_ylabel("relative energy (kcal/mol)")
    ax.set_title("Diels–Alder concerted reaction profile (GFN2-xTB)")
    ax.invert_xaxis()                                     # reaction proceeds → shorter bond
    ax.legend(frameon=False)
    fig.tight_layout(); out = FIG / "reaction_profile.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"barrier {rel[imax]:.1f} kcal/mol at {r[imax]:.2f} A -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
