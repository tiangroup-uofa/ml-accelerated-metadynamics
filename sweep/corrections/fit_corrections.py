#!/usr/bin/env python3
"""
fit_corrections.py
==================
Fit the four post-training corrections (MACE-OFF23 -> GFN2-xTB) on paired force
data and report train/validation force metrics.

  global   : F' = α F                       (1 param)
  affine   : F' = α F,  E' = α E + β         (2 params)
  element  : F'_i = α_{Z_i} F_i             (per-element; non-conservative)
  delta    : F' = F + (−∇ Σ pair potentials) (conservative pairwise radial)

Reports force RMSE and cosine similarity, before vs after each correction, on the
train and held-out validation splits, overall and per domain (water / DA). Saves
fitted parameters to fitted_corrections.json for the downstream benchmarks.

Pure fit/metrics (numpy + ase) — runs in system python3.
"""
from __future__ import annotations
import json, sys, os, glob
from pathlib import Path
import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diels_alder" / "neb"))
sys.path.insert(0, str(HERE))
from corrections import GlobalScale, Affine, ElementScale, Delta, Identity
from pairwise_delta import PairwiseDelta


def load():
    m = np.load(HERE / "data" / "mace.npz", allow_pickle=True)
    x = np.load(HERE / "data" / "xtb.npz", allow_pickle=True)
    files = sorted(glob.glob(str(HERE / "data" / "geoms" / "*.xyz")))
    atoms = [read(f) for f in files]
    return (list(m["forces"]), list(x["forces"]), list(m["energies"]),
            list(x["energies"]), [list(s) for s in m["symbols"]],
            list(m["domain"]), atoms)


def split(domain, frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for d in set(domain):
        idx = [i for i, dd in enumerate(domain) if dd == d]
        rng.shuffle(idx); k = int(round(frac * len(idx)))
        tr += idx[:k]; va += idx[k:]
    return sorted(tr), sorted(va)


def corrected_forces(name, corr, atoms, Fm, Em):
    """Apply a fitted correction to each config's MACE forces."""
    out = []
    for at, f, e in zip(atoms, Fm, Em):
        _, fc = corr.apply(at, float(e), np.asarray(f))
        out.append(fc)
    return out


def metrics(Fpred, Fref):
    rmse, cos = [], []
    for fp, fr in zip(Fpred, Fref):
        fp, fr = np.asarray(fp).ravel(), np.asarray(fr).ravel()
        rmse.append(np.sqrt(np.mean((fp - fr) ** 2)))
        cos.append(fp @ fr / (np.linalg.norm(fp) * np.linalg.norm(fr) + 1e-12))
    return float(np.mean(rmse)), float(np.mean(cos))


def sub(lst, idx):
    return [lst[i] for i in idx]


def main():
    Fm, Fx, Em, Ex, S, dom, atoms = load()
    tr, va = split(dom)
    print(f"dataset: {len(Fm)} configs ({dom.count('water')} water, "
          f"{dom.count('da')} DA); train {len(tr)}, val {len(va)}")

    # residual forces (target for delta) on train
    dF_tr = [np.asarray(Fx[i]) - np.asarray(Fm[i]) for i in tr]

    corrs = {
        "baseline": Identity(),
        "global": GlobalScale.fit(sub(Fm, tr), sub(Fx, tr)),
        "affine": Affine.fit(sub(Fm, tr), sub(Fx, tr), sub(Em, tr), sub(Ex, tr)),
        "element": ElementScale.fit(sub(Fm, tr), sub(Fx, tr), sub(S, tr)),
        "delta": Delta(PairwiseDelta().fit(sub(atoms, tr), dF_tr)),
    }

    rows = {}
    for name, c in corrs.items():
        Fc = corrected_forces(name, c, atoms, Fm, Em)
        r = {}
        for split_name, idx in (("train", tr), ("val", va)):
            r[split_name] = dict(zip(("rmse", "cos"),
                                     metrics(sub(Fc, idx), sub(Fx, idx))))
            for d in ("water", "da"):
                di = [i for i in idx if dom[i] == d]
                r[f"{split_name}_{d}"] = dict(zip(("rmse", "cos"),
                                                  metrics(sub(Fc, di), sub(Fx, di))))
        rows[name] = r

    # print table
    base = rows["baseline"]["val"]["rmse"]
    print("\n=== force metrics (corrected MACE vs xTB) ===")
    print(f"{'model':9s} | {'train RMSE':>10s} {'val RMSE':>9s} {'val cos':>8s} "
          f"{'val RMSE↓':>9s} | {'val water':>9s} {'val DA':>7s}  (RMSE eV/Å)")
    for name, r in rows.items():
        red = 100 * (1 - r["val"]["rmse"] / base) if name != "baseline" else 0.0
        print(f"{name:9s} | {r['train']['rmse']:10.3f} {r['val']['rmse']:9.3f} "
              f"{r['val']['cos']:8.3f} {red:8.0f}% | "
              f"{r['val_water']['rmse']:9.3f} {r['val_da']['rmse']:7.3f}")

    # save fitted params
    out = {
        "global": {"alpha": corrs["global"].alpha},
        "affine": {"alpha": corrs["affine"].alpha, "e_shift": corrs["affine"].e_shift},
        "element": {"scales": corrs["element"].scales},
        "delta": corrs["delta"].model.to_dict(),
        "metrics": rows,
        "split": {"train": tr, "val": va},
    }
    (HERE / "fitted_corrections.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {HERE/'fitted_corrections.json'}")


if __name__ == "__main__":
    main()
