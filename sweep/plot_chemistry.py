#!/usr/bin/env python3
"""
plot_chemistry.py
=================
Chemistry-specific visualizations for the water-cluster metadynamics
sweep, complementing the intactness summary in plot_sweep.py.

Reads the dense trajectory xtb.trj for each run (energy is stored in each
frame's comment line by xTB, parsed by ASE as atoms.info['energy']).

Figures (into figures/):

  1. hbond_distance_vs_time_n<N>.png   (for N = 2, 3, 4)
        The intermolecular HYDROGEN-BOND distance over time -- specifically
        the shortest O(molecule A) ... H(molecule B) contact, i.e. the bond
        that actually holds the cluster together. One line per bias level.
        When this shoots up, the cluster is coming apart. This is the
        chemically meaningful "key-atom" distance you asked for.

  2. energy_vs_cv_n<N>.png   (for N = 2, 3, 4)
        Potential energy (relative, kcal/mol) vs a collective variable =
        the maximum O-O distance in the cluster (the natural "separation"
        reaction coordinate for evaporation). This is the SAMPLED
        potential-energy landscape along the CV -- NOT a free-energy
        surface (the run is biased; see the project README caveat).

  3. energy_vs_cv_all.png
        All cluster sizes/biases overlaid on one energy-vs-CV plot for a
        compact overview.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIG = HERE / "figures"

KPUSH = {"low": 0.008, "med": 0.02, "high": 0.05}
COLORS = {"low": "tab:green", "med": "tab:orange", "high": "tab:red"}
CLUSTER_SIZES = [2, 3, 4]          # H-bond / multi-molecule plots only
DUMP_FS = 100.0
HARTREE_TO_KCAL = 627.509


def load_frames(n, label):
    trj = RUNS / f"n{n}" / label / "xtb.trj"
    if not trj.exists():
        return None
    frames = read(str(trj), index=":", format="extxyz")
    return frames if isinstance(frames, list) else [frames]


def read_trj_energies(n, label):
    """Parse the per-frame energy from xTB's xtb.trj comment lines.

    xTB writes comment lines like ' energy: -10.1413 gnorm: ... xtb: ...'
    which are NOT extxyz key=value pairs, so ASE cannot parse them. We
    grab the token immediately after 'energy:' ourselves. One value per
    frame, in file order.
    """
    trj = RUNS / f"n{n}" / label / "xtb.trj"
    energies = []
    with trj.open() as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        natoms = int(lines[i].split()[0])
        comment = lines[i + 1]
        toks = comment.split()
        e = np.nan
        if "energy:" in toks:
            try:
                e = float(toks[toks.index("energy:") + 1])
            except (ValueError, IndexError):
                e = np.nan
        energies.append(e)
        i += natoms + 2
    return np.array(energies, float)


def min_intermolecular_OH(atoms, o_idx, h_idx):
    """Shortest O(one molecule) ... H(a DIFFERENT molecule) distance.

    Each H is owned by its nearest O (its own molecule). We then find the
    smallest distance from any O to an H that belongs to another molecule
    -- that is the cluster's tightest intermolecular H-bond.
    """
    pos = atoms.get_positions()
    # owner O for each H
    owner = {}
    for h in h_idx:
        d = [np.linalg.norm(pos[h] - pos[o]) for o in o_idx]
        owner[h] = o_idx[int(np.argmin(d))]
    best = np.inf
    for o in o_idx:
        for h in h_idx:
            if owner[h] == o:
                continue                     # same molecule -> covalent, skip
            best = min(best, np.linalg.norm(pos[o] - pos[h]))
    return best


def max_OO(atoms, o_idx):
    pos = atoms.get_positions()
    return max(np.linalg.norm(pos[o_idx[i]] - pos[o_idx[j]])
               for i in range(len(o_idx)) for j in range(i + 1, len(o_idx)))


def series(n, label):
    frames = load_frames(n, label)
    if frames is None:
        return None
    sym = frames[0].get_chemical_symbols()
    o_idx = [i for i, s in enumerate(sym) if s == "O"]
    h_idx = [i for i, s in enumerate(sym) if s == "H"]
    energy = read_trj_energies(n, label)
    t, hbond, oo = [], [], []
    for fi, atoms in enumerate(frames):
        t.append(fi * DUMP_FS / 1000.0)
        hbond.append(min_intermolecular_OH(atoms, o_idx, h_idx))
        oo.append(max_OO(atoms, o_idx))
    return dict(t=np.array(t), hbond=np.array(hbond),
                oo=np.array(oo), energy=energy)


def plot_hbond(n, data_by_label):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, d in data_by_label.items():
        ax.plot(d["t"], d["hbond"], color=COLORS[label], lw=1.5,
                label=f"{label} (kpush={KPUSH[label]})")
    ax.axhspan(1.5, 2.5, color="gray", alpha=0.12,
               label="typical H-bond range (~1.5-2.5 A)")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("shortest intermolecular O...H distance (A)")
    ax.set_title(f"(H2O){n}: hydrogen-bond distance vs time\n"
                 "(the intermolecular bond that holds the cluster together)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"hbond_distance_vs_time_n{n}.png", dpi=150)
    plt.close(fig)


def plot_energy_cv(n, data_by_label):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, d in data_by_label.items():
        e = d["energy"]
        if np.all(np.isnan(e)):
            continue
        erel = (e - np.nanmin(e)) * HARTREE_TO_KCAL
        ax.scatter(d["oo"], erel, s=10, color=COLORS[label], alpha=0.6,
                   label=f"{label} (kpush={KPUSH[label]})")
    ax.set_xlabel("collective variable: max O-O distance (A)")
    ax.set_ylabel("relative potential energy (kcal/mol)")
    ax.set_title(f"(H2O){n}: sampled potential energy vs separation CV\n"
                 "(BIASED run -- not a free-energy surface)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"energy_vs_cv_n{n}.png", dpi=150)
    plt.close(fig)


def plot_energy_cv_all(all_data):
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {2: "o", 3: "s", 4: "^"}
    for n in CLUSTER_SIZES:
        for label, d in all_data[n].items():
            e = d["energy"]
            if np.all(np.isnan(e)):
                continue
            erel = (e - np.nanmin(e)) * HARTREE_TO_KCAL
            ax.scatter(d["oo"], erel, s=8, alpha=0.4, marker=markers[n],
                       color=COLORS[label],
                       label=f"(H2O){n} {label}")
    ax.set_xlabel("collective variable: max O-O distance (A)")
    ax.set_ylabel("relative potential energy (kcal/mol)")
    ax.set_title("Sampled potential energy vs separation CV -- all clusters\n"
                 "(color = bias strength, marker = cluster size; BIASED)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "energy_vs_cv_all.png", dpi=150)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    all_data = {}
    for n in CLUSTER_SIZES:
        data_by_label = {}
        for label in KPUSH:
            d = series(n, label)
            if d is not None:
                data_by_label[label] = d
        all_data[n] = data_by_label
        if data_by_label:
            plot_hbond(n, data_by_label)
            plot_energy_cv(n, data_by_label)
            print(f"  n={n}: wrote hbond + energy-CV plots")
    plot_energy_cv_all(all_data)
    print("  wrote energy_vs_cv_all.png")
    print(f"\nFigures in {FIG}")
    for p in sorted(FIG.glob("*.png")):
        print("   ", p.name)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
