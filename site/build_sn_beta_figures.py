#!/usr/bin/env python3
"""
build_sn_beta_figures.py — figures for the Current work (Sn-β) page, generated
from repository data only.

    python3 site/build_sn_beta_figures.py
    # -> site/assets/sn_beta_reaction.png            hero: reactant -> TS -> product
    #    site/assets/sn_beta_reaction_stacked.png    same, stacked (phones)
    #    site/assets/sn_beta_energy_profile.png      both reference conventions
    #    site/assets/sn_beta_structure_views.png     x/y/z validation views

Sources (read only, never written):
  * structures — sweep/sn_beta_hydride/input/tian_tt_test/{reactant,product}.xyz
    (byte-identical to Tian's MACE-POLAR-1-S-relaxed endpoints, tt-test b97ef46)
    and glucose/structures/ts_mace-polar_polar-1-s.xyz (his saved Sella saddle);
    atom map from sweep/sn_beta_hydride/config_tian_tt_test.json
  * energies — sweep/sn_beta_hydride/audit/tian_tt_test/tian_polar_summary.json,
    itself generated from Tian's saved total energies by summarize_tian_polar.py.
    The script recomputes every plotted difference from the raw energies in that
    file and checks it against the file's labelled energetics.

These are RECONSTRUCTED structures (hand-built model, not literature/CPMD
coordinates) and the saddle is NOT independently re-certified; the figures say
so. No value is typed in by hand. Requires numpy, matplotlib and ASE.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.data import covalent_radii  # noqa: E402
from ase.io import read  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SN = REPO / "sweep" / "sn_beta_hydride"
ASSETS = HERE / "assets"
SOURCES = {
    "reactant": SN / "input" / "tian_tt_test" / "reactant.xyz",
    "ts": REPO / "glucose" / "structures" / "ts_mace-polar_polar-1-s.xyz",
    "product": SN / "input" / "tian_tt_test" / "product.xyz",
    "config": SN / "config_tian_tt_test.json",
    "summary": SN / "audit" / "tian_tt_test" / "tian_polar_summary.json",
}
EV2KJ = 96.48533212           # CODATA 2018, as in sweep/sn_beta_hydride/energetics.py

# site palette (site/styles.css)
INK, INK_SOFT, RULE = "#1f2933", "#52606d", "#e4e7eb"
ACCENT, OK = "#2e6f95", "#3f7d4e"
C_SIDE = "#7a5c99"            # second reference convention (neutral; not xTB red)
ELEM = {"H": "#e6e8eb", "C": "#3d4852", "O": "#d1495b", "Si": "#d4b483", "Sn": ACCENT}
RADIUS = {"H": 0.20, "C": 0.30, "O": 0.30, "Si": 0.36, "Sn": 0.46}   # drawing radii (Å)
HSTAR = "#e09f3e"             # the transferring hydride
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "savefig.dpi": 200, "font.family": "DejaVu Sans"})


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def load():
    cfg = json.loads(SOURCES["config"].read_text(encoding="utf-8"))
    amap = cfg["atom_map"]
    S = {k: read(str(SOURCES[k])) for k in ("reactant", "ts", "product")}
    for k, a in S.items():
        assert a.get_chemical_symbols() == S["reactant"].get_chemical_symbols(), k
    summ = json.loads(SOURCES["summary"].read_text(encoding="utf-8"))
    return amap, S, summ


def kabsch(P, Q):
    """Rotation R and translation so that (P - Pc) @ R + Qc ~ Q."""
    Pc, Qc = P.mean(0), Q.mean(0)
    V, _, Wt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    d = np.sign(np.linalg.det(V @ Wt))
    return V @ np.diag([1, 1, d]) @ Wt, Pc, Qc


def common_frame(S, m):
    """Align R and P onto the TS on the Sn/O1/O2/C1/C2 core; Sn at the origin."""
    core = [m[k] for k in ("Sn", "O1", "O2", "C1", "C2")]
    T = S["ts"].get_positions()
    out = {}
    for k, a in S.items():
        X = a.get_positions()
        if k != "ts":
            R, Pc, Qc = kabsch(X[core], T[core])
            X = (X - Pc) @ R + Qc
        out[k] = X - T[m["Sn"]]
    return out


def view_rotation(X, m):
    """Deterministic view search on a Fibonacci sphere: screen-x along C2 -> C1
    (so H* moves left -> right through R, TS, P), H* above and Sn below the C1-C2
    midpoint at the TS, maximizing the smallest projected separation of the
    labelled core atoms over all three structures."""
    ts = X["ts"]
    labelled = [m[k] for k in ("Sn", "O1", "O2", "C1", "C2", "H_transfer", "O_water", "H_C1")]
    e1 = ts[m["C1"]] - ts[m["C2"]]
    n = 6000
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    best = None
    for d in np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1):
        sx = e1 - (e1 @ d) * d
        if np.linalg.norm(sx) < 0.9 * np.linalg.norm(e1):
            continue                                  # keep C1-C2 nearly unforeshortened
        sx /= np.linalg.norm(sx)
        sy = np.cross(d, sx)
        R = np.array([sx, sy, d]).T
        p = ts @ R
        mid = 0.5 * (p[m["C1"]] + p[m["C2"]])
        if p[m["H_transfer"], 1] - mid[1] < 0.6 or p[m["Sn"], 1] > mid[1] - 0.6:
            continue
        score = min(min(np.linalg.norm((X[k] @ R)[a, :2] - (X[k] @ R)[b, :2])
                        for ai, a in enumerate(labelled) for b in labelled[ai + 1:])
                    for k in X)
        if best is None or score > best[0]:
            best = (score, R)
    return best[1]


# --------------------------------------------------------------------------- #
#  ball-and-stick drawing (flat, depth-sorted; no lighting effects)
# --------------------------------------------------------------------------- #
def draw_structure(ax, P, syms, m, core_ids, mode, label_core=True):
    """P: projected coordinates (n x 3, column 2 = depth toward the viewer)."""
    h = m["H_transfer"]
    n = len(syms)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)   # rotation preserves distances
    is_core = np.zeros(n, bool)
    is_core[core_ids] = True
    sn = m["Sn"]
    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            if h in (i, j) and {syms[i], syms[j]} == {"C", "H"} and {i, j} & {m["C1"], m["C2"]}:
                continue                                         # H*-C drawn separately
            if sn in (i, j):
                if "O" not in (syms[i], syms[j]) or d > 2.6:
                    continue
                ls = (0, (3, 2)) if d > 2.2 else "-"
            elif d < 1.15 * (covalent_radii_of(syms[i]) + covalent_radii_of(syms[j])):
                ls = "-"
            else:
                continue
            core_bond = is_core[i] and is_core[j]
            ax.plot(P[[i, j], 0], P[[i, j], 1], ls=ls, lw=2.4 if core_bond else 1.6,
                    color="#6b7781" if core_bond else "#c3c9cf", zorder=1 + 0.01 * min(P[i, 2], P[j, 2]),
                    solid_capstyle="round")
    # H* to C1 / C2
    for c in (m["C1"], m["C2"]):
        d = D[h, c]
        if d < 1.6:
            ax.plot(P[[h, c], 0], P[[h, c], 1], color=HSTAR, lw=3.0,
                    ls="-" if d < 1.23 else (0, (2.5, 1.6)), zorder=5, solid_capstyle="round")
    for i in np.argsort(P[:, 2]):
        el = syms[i]
        star = i == h
        alpha = 1.0 if (is_core[i] or star) else 0.42
        circ = Circle(P[i, :2], RADIUS.get(el, 0.3) * (1.25 if star else 1.0),
                      facecolor=HSTAR if star else ELEM.get(el, "#999"),
                      edgecolor=INK if (is_core[i] or star) else "#9aa5b1",
                      lw=1.1 if (is_core[i] or star) else 0.6, alpha=alpha,
                      zorder=6 + 0.01 * P[i, 2] if (is_core[i] or star) else 2 + 0.01 * P[i, 2])
        ax.add_patch(circ)
    ax.set_aspect("equal")
    ax.axis("off")


def covalent_radii_of(sym):
    from ase.data import atomic_numbers

    return covalent_radii[atomic_numbers[sym]]


LABEL_OFFSET = {"Sn": (-0.05, -0.62), "O1": (0.42, -0.42), "O2": (-0.42, -0.42),
                "C1": (0.38, -0.32), "C2": (-0.38, -0.32), "O_water": (0.0, -0.55)}


def label_core(ax, P, m, fs=12):
    names = {"Sn": "Sn", "O1": "O1", "O2": "O2", "C1": "C1", "C2": "C2", "O_water": "O$_w$"}
    for k, txt in names.items():
        dx, dy = LABEL_OFFSET[k]
        ax.text(P[m[k], 0] + dx, P[m[k], 1] + dy, txt, ha="center", va="center", fontsize=fs,
                color=INK, fontweight="bold" if k in ("Sn", "C1", "C2") else "normal", zorder=20,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    h = m["H_transfer"]
    ax.text(P[h, 0] - 0.12, P[h, 1] + 0.42, "H*", ha="center", va="bottom", fontsize=fs + 1,
            color="#9c6514", fontweight="bold", zorder=20,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


def annotate_ch(ax, P, m):
    """Non-bonded C...H* contacts (>= 1.6 Å) drawn as a thin dotted guide; the
    distances themselves are printed under each panel, not on the drawing."""
    h = m["H_transfer"]
    for c in (m["C2"], m["C1"]):
        if np.linalg.norm(P[h] - P[c]) >= 1.6:
            ax.plot(P[[h, c], 0], P[[h, c], 1], color=HSTAR, lw=1.0, ls=":", zorder=4)


# --------------------------------------------------------------------------- #
#  1. hero: reactant -> TS -> product
# --------------------------------------------------------------------------- #
TITLES = {"reactant": ("Reactant", "H* on C2 (donor)"),
          "ts": ("Transition state", "H* shared between C2 and C1"),
          "product": ("Product", "H* on C1 (acceptor)")}


def hero(amap, S, out, stacked=False):
    X = common_frame(S, amap)
    R = view_rotation(X, amap)
    syms = S["ts"].get_chemical_symbols()
    core = sorted({amap[k] for k in ("Sn", "O1", "O2", "C1", "C2", "O_water", "H_C1")}
                  | {i for i, s in enumerate(syms) if False})
    P = {k: X[k] @ R for k in X}
    allxy = np.vstack([p[:, :2] for p in P.values()])
    lo, hi = allxy.min(0) - 0.7, allxy.max(0) + 0.7
    order = ("reactant", "ts", "product")
    span = hi - lo
    if stacked:
        fig, axes = plt.subplots(3, 1, figsize=(4.4, 3 * 4.4 * span[1] / span[0] * 0.92 + 1.2))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25 * span[1] / span[0] + 1.3))
    for ax, k in zip(axes, order):
        draw_structure(ax, P[k], syms, amap, core, k)
        label_core(ax, P[k], amap)
        annotate_ch(ax, P[k], amap)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        t, sub = TITLES[k]
        ax.set_title(f"{t}\n", fontsize=16.5, fontfamily="serif", fontweight="semibold",
                     color=INK, pad=2)
        ax.text(0.5, 1.0, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=12,
                color=INK_SOFT)
        D = lambda a, b: np.linalg.norm(S[k].positions[amap[a]] - S[k].positions[amap[b]])  # noqa: E731
        ax.text(0.5, -0.01, f"C2–H* {D('C2', 'H_transfer'):.2f} Å   ·   "
                f"C1–H* {D('C1', 'H_transfer'):.2f} Å", transform=ax.transAxes, ha="center",
                va="top", fontsize=11.5, color="#7a4f10", fontweight="bold")
        ax.text(0.5, -0.085, f"Sn–O1 {D('Sn', 'O1'):.2f} Å   ·   Sn–O2 {D('Sn', 'O2'):.2f} Å",
                transform=ax.transAxes, ha="center", va="top", fontsize=11.2, color=INK_SOFT)
    if stacked:
        fig.subplots_adjust(hspace=0.62, top=0.96, bottom=0.045, left=0.02, right=0.98)
        for a, b in ((axes[0], axes[1]), (axes[1], axes[2])):
            h = a.get_position().height
            y0 = a.get_position().y0 - 0.13 * h          # below the two caption lines
            y1 = b.get_position().y1 + 0.17 * h          # above the next title
            fig.patches.append(FancyArrowPatch((0.5, y0), (0.5, y1), transform=fig.transFigure,
                                               arrowstyle="-|>", mutation_scale=16, lw=1.6,
                                               color="#9aa5b1"))
    else:
        fig.subplots_adjust(wspace=0.22, top=0.87, bottom=0.12, left=0.01, right=0.99)
        for a, b in ((axes[0], axes[1]), (axes[1], axes[2])):
            x0 = a.get_position().x1 - 0.004
            x1 = b.get_position().x0 + 0.004
            y = 0.5 * (a.get_position().y0 + a.get_position().y1)
            fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                               arrowstyle="-|>", mutation_scale=18, lw=1.8,
                                               color="#9aa5b1"))
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return P


# --------------------------------------------------------------------------- #
#  2. energy profile with both reference conventions
# --------------------------------------------------------------------------- #
def energy_profile(summ, out):
    E = summ["energies_eV"]
    r = lambda k: (E[k] - E["reactant"]) * EV2KJ           # noqa: E731  relative to relaxed R
    lv = {"R": 0.0, "R'": r("reactant_side_min"), "TS": r("ts"), "P": r("product"),
          "P'": r("product_side_min")}
    bf_ep = lv["TS"] - lv["R"]
    bf_cs = lv["TS"] - lv["R'"]
    de_ep = lv["P"] - lv["R"]
    de_cs = lv["P'"] - lv["R'"]
    off = lv["R'"] - lv["R"]
    en = summ["energetics"]
    for got, key, st in ((bf_ep, "barrier_forward", "relaxed_endpoint"),
                         (bf_cs, "barrier_forward", "connected_side_minimum"),
                         (de_ep, "reaction_energy", "relaxed_endpoint"),
                         (de_cs, "reaction_energy", "connected_side_minimum")):
        assert abs(got - en[key][st]["kJ_mol"]) < 1e-6, (key, st, got)
    cd = summ["conformer_diagnostics"]["reactant_side"]
    assert abs(off - cd["energy_offset_kJ_mol"]) < 1e-6

    # relaxed-endpoint path outside, connected-side path inside: the lines never cross
    x = {"R": 0.0, "R'": 1.15, "TS": 3.0, "P'": 4.85, "P": 6.45}
    w = 0.62
    fig, ax = plt.subplots(figsize=(10.2, 6.2))

    def level(k, color, lw=3.2):
        ax.plot([x[k] - w / 2, x[k] + w / 2], [lv[k]] * 2, color=color, lw=lw,
                solid_capstyle="butt", zorder=4)

    def link(a, b, color, ls):
        ax.plot([x[a] + w / 2, x[b] - w / 2] if x[a] < x[b] else [x[a] - w / 2, x[b] + w / 2],
                [lv[a], lv[b]], color=color, lw=1.3, ls=ls, zorder=2)

    def vbar(xx, y0, y1, color, txt, side="right", fs=12.5, bold=True, frac=0.5):
        ax.add_patch(FancyArrowPatch((xx, y0), (xx, y1), arrowstyle="<|-|>", mutation_scale=10,
                                     lw=1.3, color=color, zorder=6, shrinkA=0, shrinkB=0))
        ax.text(xx + (0.09 if side == "right" else -0.09), y0 + frac * (y1 - y0), txt,
                ha="left" if side == "right" else "right", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    # paths
    link("R", "TS", ACCENT, "-")
    link("TS", "P", ACCENT, "-")
    link("R'", "TS", C_SIDE, (0, (4, 2.5)))
    link("TS", "P'", C_SIDE, (0, (4, 2.5)))
    level("R", ACCENT)
    level("P", ACCENT)
    level("R'", C_SIDE)
    level("P'", C_SIDE)
    level("TS", INK, lw=3.6)

    # guides at the TS level and at the two reactant levels
    ax.plot([x["R"], x["TS"] - w / 2], [lv["TS"]] * 2, color="#c3c9cf", lw=0.9, ls=":", zorder=1)
    ax.plot([x["R"] + w / 2, x["P"]], [lv["R"]] * 2, color=ACCENT, lw=0.8, ls=":", zorder=1,
            alpha=0.6)
    ax.plot([x["R'"] + w / 2, x["P'"]], [lv["R'"]] * 2, color=C_SIDE, lw=0.8, ls=":", zorder=1,
            alpha=0.6)

    # barriers
    vbar(x["R"], lv["R"], lv["TS"], ACCENT, f"{bf_ep:.2f}", "right")
    vbar(x["R'"], lv["R'"], lv["TS"], C_SIDE, f"{bf_cs:.2f}", "right", frac=0.36)
    # reaction energies
    vbar(x["P"], lv["R"], lv["P"], ACCENT, f"ΔE {de_ep:+.2f}", "right", fs=11.5)
    vbar(x["P'"], lv["R'"], lv["P'"], C_SIDE, f"ΔE {de_cs:+.2f}", "right", fs=11.5)
    # conformer offset between the two reactant references
    xo = 0.5 * (x["R'"] + x["R"])
    ax.add_patch(FancyArrowPatch((xo, lv["R"]), (xo, lv["R'"]), arrowstyle="<|-|>",
                                 mutation_scale=8, lw=1.1, color=INK_SOFT, zorder=6,
                                 shrinkA=0, shrinkB=0))
    ax.annotate(f"{off:.2f} kJ/mol\nreactant conformer offset\n(reactive-core RMSD "
                f"{cd['rmsd_core_A']:.2f} Å)", (xo, 0.5 * (lv["R"] + lv["R'"])),
                xytext=(xo, -50), ha="center", va="top", fontsize=10.2, color=INK_SOFT,
                arrowprops=dict(arrowstyle="-", color="#9aa5b1", lw=0.8))

    # level labels
    lab = {"R": ("R", "relaxed input\nreactant"), "R'": ("R′", "reactant-side\nconnected minimum"),
           "TS": ("TS", "Sella saddle\n(not re-certified)"),
           "P": ("P", "relaxed input\nproduct"), "P'": ("P′", "product-side\nconnected minimum")}
    for k, (short, long_) in lab.items():
        col = INK if k == "TS" else (ACCENT if k in ("R", "P") else C_SIDE)
        if k == "TS":
            ax.text(x[k], lv[k] + 4, f"{short}  ", ha="center", va="bottom", fontsize=13,
                    fontweight="bold", color=col)
            ax.text(x[k], lv[k] + 13.5, long_.replace("\n", " "), ha="center", va="bottom",
                    fontsize=10.2, color=INK_SOFT)
        else:
            above = False
            y = lv[k] + (4 if above else -4)
            ax.text(x[k], y, short, ha="center", va="bottom" if above else "top", fontsize=13,
                    fontweight="bold", color=col)
            ax.text(x[k], y + (8 if above else -10.5), long_, ha="center",
                    va="bottom" if above else "top", fontsize=9.8, color=INK_SOFT,
                    linespacing=1.15)

    ax.set_xlim(-0.75, 7.25)
    ax.set_ylim(-80, 150)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("E − E(relaxed input reactant)  (kJ/mol)", fontsize=12)
    ax.axhline(0, color=RULE, lw=0.8, zorder=0)
    from matplotlib.lines import Line2D

    ax.legend(handles=[Line2D([], [], color=ACCENT, lw=2.6, label="relaxed_endpoint  (R → TS → P)"),
                       Line2D([], [], color=C_SIDE, lw=2.6, ls=(0, (4, 2.5)),
                              label="connected_side_minimum  (R′ → TS → P′)")],
              loc="upper right", bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=11,
              title="reference convention", title_fontsize=11, alignment="left")
    ax.set_title("Same saddle, two reactant references", loc="left", fontsize=14.5,
                 fontfamily="serif", fontweight="semibold", color=INK, pad=10)
    fig.text(0.01, 0.005, "MACE-POLAR-1-S, vacuum electronic energies (no ZPE / thermal terms); "
             "reconstructed cluster.\nNeither reactant structure is shown to be the global "
             "minimum. Barrier and ΔE values in kJ/mol.",
             fontsize=9.6, color=INK_SOFT, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return lv, {"barrier_relaxed_endpoint": bf_ep, "barrier_connected_side_minimum": bf_cs,
                "dE_relaxed_endpoint": de_ep, "dE_connected_side_minimum": de_cs,
                "reactant_conformer_offset": off}


# --------------------------------------------------------------------------- #
#  3. orthogonal x/y/z validation views (same frame as audit/structure_views.png)
# --------------------------------------------------------------------------- #
def sn_frame(atoms, m):
    """As sweep/sn_beta_hydride/visualize_structures.sn_frame: origin Sn; x to the
    C1/C2 midpoint; z normal to the O1-Sn-O2 plane."""
    X = atoms.get_positions() - atoms.positions[m["Sn"]]
    ex = (X[m["C1"]] + X[m["C2"]]) / 2
    ex /= np.linalg.norm(ex)
    n = np.cross(X[m["O1"]], X[m["O2"]])
    ez = n - ex * (n @ ex)
    ez /= np.linalg.norm(ez)
    return X @ np.array([ex, np.cross(ez, ex), ez]).T


def structure_views(amap, S, out):
    views = (((0, 1), "view down z\n(onto the O1–Sn–O2 plane)"), ((0, 2), "view down y"),
             ((1, 2), "view down x\n(from Sn toward C1/C2)"))
    order = ("reactant", "ts", "product")
    F = {k: sn_frame(S[k], amap) for k in order}
    syms = S["ts"].get_chemical_symbols()
    core = sorted({amap[k] for k in ("Sn", "O1", "O2", "C1", "C2", "O_water", "H_C1")})
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 11.0))
    for c, ((i, j), vname) in enumerate(views):
        allv = np.vstack([F[k][:, [i, j]] for k in order])
        lo, hi = allv.min(0) - 0.6, allv.max(0) + 0.6
        for r, k in enumerate(order):
            ax = axes[r, c]
            k3 = 3 - i - j
            P = np.column_stack([F[k][:, i], F[k][:, j], F[k][:, k3] * (1 if c != 1 else -1)])
            draw_structure(ax, P, syms, amap, core, k)
            h = amap["H_transfer"]
            ax.text(P[h, 0], P[h, 1] + 0.42, "H*", ha="center", va="bottom", fontsize=11,
                    color="#9c6514", fontweight="bold", zorder=20)
            ax.text(P[amap["Sn"], 0], P[amap["Sn"], 1] - 0.62, "Sn", ha="center", va="top",
                    fontsize=11, color=ACCENT, fontweight="bold", zorder=20)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
    row_lab = {"reactant": "Reactant", "ts": "TS", "product": "Product"}
    for r, k in enumerate(order):
        D = lambda a, b: np.linalg.norm(S[k].positions[amap[a]] - S[k].positions[amap[b]])  # noqa: E731
        axes[r, 0].text(-0.04, 0.5, f"{row_lab[k]}\n\nC2–H* {D('C2', 'H_transfer'):.2f}\n"
                        f"C1–H* {D('C1', 'H_transfer'):.2f}\nSn–O1 {D('Sn', 'O1'):.2f}\n"
                        f"Sn–O2 {D('Sn', 'O2'):.2f} Å", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=12, color=INK, linespacing=1.35)
    # 2 Å scale bar
    ax = axes[2, 0]
    x0, y0 = ax.get_xlim()[0] + 0.3, ax.get_ylim()[0] + 0.35
    ax.plot([x0, x0 + 2.0], [y0, y0], color=INK_SOFT, lw=2)
    ax.text(x0 + 1.0, y0 + 0.15, "2 Å", ha="center", va="bottom", fontsize=11, color=INK_SOFT)
    fig.subplots_adjust(left=0.17, right=0.995, top=0.94, bottom=0.01, wspace=0.04, hspace=0.06)
    for c, (_, vname) in enumerate(views):          # headers on one line, whatever the panel height
        p = axes[0, c].get_position()
        fig.text(0.5 * (p.x0 + p.x1), 0.985, vname, ha="center", va="top", fontsize=12.5,
                 color=INK_SOFT)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main():
    amap, S, summ = load()
    ASSETS.mkdir(exist_ok=True)
    hero(amap, S, ASSETS / "sn_beta_reaction.png")
    hero(amap, S, ASSETS / "sn_beta_reaction_stacked.png", stacked=True)
    lv, vals = energy_profile(summ, ASSETS / "sn_beta_energy_profile.png")
    structure_views(amap, S, ASSETS / "sn_beta_structure_views.png")
    print("sources:")
    for k, p in SOURCES.items():
        print(f"  {k:9s} {p.relative_to(REPO).as_posix()}")
    print("levels (kJ/mol vs relaxed input reactant):",
          {k: round(v, 2) for k, v in lv.items()})
    print("plotted:", {k: round(v, 2) for k, v in vals.items()})
    for k in ("reactant", "ts", "product"):
        a = S[k]
        d = lambda x, y: a.get_distance(amap[x], amap[y])  # noqa: E731
        print(f"  {k:8s} C2-H* {d('C2', 'H_transfer'):.3f}  C1-H* {d('C1', 'H_transfer'):.3f}  "
              f"Sn-O1 {d('Sn', 'O1'):.3f}  Sn-O2 {d('Sn', 'O2'):.3f}")
    for f in ("sn_beta_reaction.png", "sn_beta_reaction_stacked.png",
              "sn_beta_energy_profile.png", "sn_beta_structure_views.png"):
        print(f"wrote site/assets/{f}")


if __name__ == "__main__":
    main()
