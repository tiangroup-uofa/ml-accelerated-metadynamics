#!/usr/bin/env python3
"""
cv_scan.py
==========
Collective-variable scoping for water metadynamics (Tian's "hard one": which CVs
to bias). We compute a battery of candidate CVs on trajectories spanning the
distinct physical states we care about, then ask which CVs are (a) informative
(separate the states), (b) non-redundant, and (c) capture the processes of
interest — as the basis for choosing what to bias in MACE-driven metadynamics.

Candidate CVs (per frame):
  Rg            radius of gyration of O atoms          — droplet size / condensation
  asphericity   from the gyration tensor               — droplet shape (0 = sphere)
  max_OO        largest O–O distance                   — spread / an atom leaving
  coord_OO      mean O–O switching-function coord.      — network density
  n_hbond_pm    H-bonds per molecule (O···H<2.5Å,      — H-bond network (angular!)
                O–H···O > 150°)
  q_tet         tetrahedral order (Errington–Deb.)     — local liquid structure
  max_OH        max intramolecular O–H                 — dissociation (chemistry)

States sampled: stable droplet, fragmenting droplet, big droplet, drifting small
cluster, monomer dissociation.

Outputs (analysis/, figures/): cv_table.csv, cv_correlation.png, cv_pca.png,
and a printed informativeness / redundancy summary + recommendation.
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent
OUT = HERE / "analysis"; FIG = HERE / "figures"
R0_OO, SW_P, SW_Q = 3.2, 8, 14

TRAJS = {
    "droplet-stable (n30)":       "large/runs/n30/low/xtb.trj",
    "droplet-fragmenting (n30)":  "kpush_scaling/n30_k0.08/xtb.trj",
    "droplet-big (n50)":          "large/runs/n50/low/xtb.trj",
    "cluster-drift (n3)":         "runs/n3/low/xtb.trj",
    "monomer-dissoc (n1)":        "runs/n1/high/xtb.trj",
}
CVS = ["Rg", "asphericity", "max_OO", "coord_OO", "n_hbond_pm", "q_tet", "max_OH"]


def switch(r):
    x = r / R0_OO; x = np.where(np.isclose(x, 1.0), 1 + 1e-9, x)
    return (1 - x**SW_P) / (1 - x**SW_Q)


def frame_cvs(atoms):
    sym = np.array(atoms.get_chemical_symbols())
    oi = np.where(sym == "O")[0]; hi = np.where(sym == "H")[0]
    P = atoms.get_positions(); O = P[oi]; H = P[hi]
    nO = len(O)

    com = O.mean(0); dO = O - com
    Rg = float(np.sqrt((dO**2).sum(1).mean()))
    if nO >= 2:
        S = (dO.T @ dO) / nO
        ev = np.sort(np.linalg.eigvalsh(S))        # ascending
        asph = float((ev[2] - 0.5*(ev[0]+ev[1])) / ev.sum()) if ev.sum() > 0 else np.nan
    else:
        asph = np.nan                              # monomer has no shape

    OO = np.linalg.norm(O[:, None] - O[None, :], axis=2)
    np.fill_diagonal(OO, np.inf)
    if nO >= 2:
        max_OO = float(OO[np.isfinite(OO)].max())
        coord = float(switch(np.where(np.isfinite(OO), OO, 1e6)).sum(1).mean())
    else:
        max_OO, coord = 0.0, 0.0          # monomer: no O–O pairs

    OH = np.linalg.norm(H[:, None] - O[None, :], axis=2)   # (nH, nO)
    owner = OH.argmin(1)
    max_OH = float(OH.min(1).max())

    # H-bonds: donor O = owner(H); acceptor = other O with H···O<2.5 and angle>150
    nhb = 0
    for h in range(len(H)):
        dO = owner[h]
        for a in range(nO):
            if a == dO:
                continue
            if OH[h, a] < 2.5:
                v1 = O[dO] - H[h]; v2 = O[a] - H[h]
                cang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
                if np.degrees(np.arccos(np.clip(cang, -1, 1))) > 150:
                    nhb += 1
    n_hbond_pm = nhb / nO

    # tetrahedral order (needs >=4 O neighbours)
    qs = []
    for a in range(nO):
        order = np.argsort(OO[a])
        if len(order) < 4:            # cluster too small for tetrahedral order
            break
        nn = order[:4]
        if not np.isfinite(OO[a, nn[3]]):
            continue
        vs = O[nn] - O[a]; vs /= np.linalg.norm(vs, axis=1, keepdims=True)
        s = 0.0
        for j in range(3):
            for k in range(j+1, 4):
                s += (np.dot(vs[j], vs[k]) + 1/3)**2
        qs.append(1 - 3/8*s)
    q_tet = float(np.mean(qs)) if qs else np.nan

    return dict(Rg=Rg, asphericity=asph, max_OO=max_OO, coord_OO=coord,
                n_hbond_pm=n_hbond_pm, q_tet=q_tet, max_OH=max_OH)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, rel in TRAJS.items():
        trj = SWEEP / rel
        if not trj.exists():
            print(f"  (missing {rel})"); continue
        frames = read(str(trj), index=":", format="xyz")
        if not isinstance(frames, list):
            frames = [frames]
        for fi, at in enumerate(frames):
            r = frame_cvs(at); r.update(state=label, frame=fi)
            rows.append(r)
        print(f"  {label}: {len(frames)} frames")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cv_table.csv", index=False)

    # ---- informativeness: how well each CV separates the states ----
    # ratio of between-state variance to within-state variance (F-like)
    print("\n=== CV informativeness (between/within-state variance ratio) ===")
    info = {}
    for cv in CVS:
        sub = df[[cv, "state"]].dropna()
        g = sub.groupby("state")[cv]
        grand = sub[cv].mean()
        between = sum(len(v)*(v.mean()-grand)**2 for _, v in g) / max(len(sub), 1)
        within = g.transform(lambda s: (s-s.mean())**2).mean()
        info[cv] = float(between/within) if within and within > 0 else np.inf
        info[cv + "_ndef"] = f"({sub.state.nunique()}/{df.state.nunique()} states)"
    for cv in sorted(CVS, key=lambda c: -info[c]):
        print(f"  {cv:12s} separation = {info[cv]:6.2f}  {info[cv+'_ndef']}")

    # ---- redundancy: correlation matrix ----
    C = df[CVS].corr()
    print("\n=== redundant pairs (|r| > 0.9) ===")
    red = [(a, b, C.loc[a, b]) for i, a in enumerate(CVS) for b in CVS[i+1:]
           if abs(C.loc[a, b]) > 0.9]
    for a, b, r in red:
        print(f"  {a} ~ {b}  (r={r:+.2f})")
    if not red:
        print("  none")

    # ---- PCA (numpy) ----
    X = df[CVS].fillna(df[CVS].mean()).values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    var = (s**2) / (s**2).sum()
    pcs = U[:, :2] * s[:2]
    print(f"\n=== PCA: variance explained PC1={var[0]:.0%} PC2={var[1]:.0%} "
          f"(PC1+PC2={var[0]+var[1]:.0%}) ===")
    print("PC1 loadings:", dict(zip(CVS, np.round(Vt[0], 2))))
    print("PC2 loadings:", dict(zip(CVS, np.round(Vt[1], 2))))

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(C.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(CVS))); ax.set_xticklabels(CVS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(CVS))); ax.set_yticklabels(CVS, fontsize=9)
    for i in range(len(CVS)):
        for j in range(len(CVS)):
            ax.text(j, i, f"{C.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(C.values[i,j]) < 0.6 else "white")
    ax.set_title("CV correlation"); fig.colorbar(im, shrink=.8)
    fig.tight_layout(); fig.savefig(FIG / "cv_correlation.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5))
    for label in TRAJS:
        m = df.state.values == label
        ax.scatter(pcs[m, 0], pcs[m, 1], s=16, alpha=.7, label=label)
    ax.set_xlabel(f"PC1 ({var[0]:.0%})"); ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.set_title("States in CV space (PCA)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "cv_pca.png", dpi=140); plt.close(fig)
    print(f"\nwrote {OUT/'cv_table.csv'}, {FIG/'cv_correlation.png'}, {FIG/'cv_pca.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
