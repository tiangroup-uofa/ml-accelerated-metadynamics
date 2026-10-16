#!/usr/bin/env python3
"""
visualize_structures.py
=======================
Orthogonal x/y/z projections of several structures of the same system in ONE
common Sn-centred frame, for qualitative comparison against the literature
drawing. A sanity picture only — never a pass/fail criterion.

Frame (from the mapped atoms of each structure):
  origin = Sn;  x = direction from Sn to the C1/C2 midpoint;
  z = normal of the O1-Sn-O2 plane (orthogonalized to x);  y = z cross x.
Drawn bonds use the validation bond criterion (1.15 x covalent sum here);
Sn-O within 2.6 Å are drawn, dashed beyond 2.2 Å. H* is outlined in green,
with its C-H* bonds (< 1.6 Å) in green.

    python visualize_structures.py --config config_tian_tt_test.json \
        --structure "Reactant=input/tian_tt_test/reactant.xyz" \
        --structure "TS=../../glucose/structures/ts_mace-polar_polar-1-s.xyz" \
        --out audit/tian_tt_test/structure_views.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from config_io import atom_map, load_config  # noqa: E402
from hydride import core_distances  # noqa: E402

COLORS = {"H": "#d9d9d9", "C": "#404040", "O": "#e03030", "Si": "#d8b060", "Sn": "#5a7fa8"}
LABELS = {"C1": "C1", "C2": "C2", "H_transfer": "H*", "Sn": "Sn", "O1": "O1", "O2": "O2",
          "O_water": "Ow"}
VIEWS = (((0, 1), "x-y (down z)"), ((0, 2), "x-z (down y)"), ((1, 2), "y-z (down x)"))


def sn_frame(atoms, m) -> np.ndarray:
    """Coordinates in the common Sn-centred frame (n x 3)."""
    X = atoms.get_positions() - atoms.positions[m["Sn"]]
    ex = (X[m["C1"]] + X[m["C2"]]) / 2
    ex /= np.linalg.norm(ex)
    n = np.cross(X[m["O1"]], X[m["O2"]])
    ez = n - ex * (n @ ex)
    ez /= np.linalg.norm(ez)
    return X @ np.array([ex, np.cross(ez, ex), ez]).T


def draw(structures: list[tuple[str, object]], cfg: dict, out: Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.data import covalent_radii

    names = ["C1", "C2", "H_transfer", "Sn", "O1", "O2", "O_water"]
    fig, axes = plt.subplots(len(structures), 3, figsize=(15, 5 * len(structures)),
                             squeeze=False)
    for r, (label, a) in enumerate(structures):
        m = atom_map(cfg, len(a), require=[n for n in names if n in (cfg.get("atom_map") or {})])
        lab = {i: LABELS[n] for n, i in m.items()}
        h = m["H_transfer"]
        sn = m["Sn"]
        X = sn_frame(a, m)
        sym = a.get_chemical_symbols()
        D = a.get_all_distances()
        for c, ((i, j), vname) in enumerate(VIEWS):
            ax = axes[r, c]
            for p in range(len(a)):
                for q in range(p + 1, len(a)):
                    d = D[p, q]
                    if sn in (p, q):
                        if "O" in (sym[p], sym[q]) and d < 2.6:
                            ax.plot(X[[p, q], i], X[[p, q], j], color="#888", lw=1.5,
                                    ls="--" if d > 2.2 else "-", zorder=1)
                        continue
                    if h in (p, q) and "C" in (sym[p], sym[q]) and d < 1.6:
                        ax.plot(X[[p, q], i], X[[p, q], j], color="#1f9e5a", lw=3, zorder=1)
                    elif d < 1.15 * (covalent_radii[a.numbers[p]] + covalent_radii[a.numbers[q]]):
                        ax.plot(X[[p, q], i], X[[p, q], j], color="#999", lw=1.5, zorder=1)
            depth = 3 - i - j
            for p in np.argsort(X[:, depth]):
                ax.scatter(X[p, i], X[p, j], s=(60 if sym[p] == "H" else 160) * (2 if p == h else 1),
                           c=COLORS.get(sym[p], "#b0b"), zorder=2,
                           edgecolors="#1f9e5a" if p == h else "k",
                           linewidths=2 if p == h else 0.5)
                if p in lab:
                    ax.annotate(lab[p], X[p, [i, j]], xytext=(5, 5), textcoords="offset points",
                                fontsize=9, zorder=3)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.set_title(f"{label}\n{vname}", fontsize=10)
        cd = core_distances(a, cfg)
        txt = "  ".join(f"{k} {cd[k]:.3f}" for k in ("C2-H*", "C1-H*") if k in cd) + "\n" + \
              "  ".join(f"{k} {cd[k]:.3f}" for k in ("Sn-O1", "Sn-O2", "Sn-O_water") if k in cd)
        axes[r, 0].text(0.01, 0.01, txt + " (Å)", transform=axes[r, 0].transAxes, fontsize=9,
                        family="monospace", bbox=dict(fc="white", ec="#ccc"))
    fig.suptitle(title or "Common Sn-centred frame (Å); qualitative sanity check only", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    from ase.io import read

    ap = argparse.ArgumentParser(description="Orthogonal views in a common Sn-centred frame")
    ap.add_argument("--config", required=True)
    ap.add_argument("--structure", action="append", required=True, help="LABEL=path.xyz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    items = []
    for s in a.structure:
        label, path = s.split("=", 1)
        items.append((label, read(path)))
    print(f"wrote {draw(items, cfg, Path(a.out), a.title)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
