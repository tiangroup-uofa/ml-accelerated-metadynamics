#!/usr/bin/env python3
"""
phaseC_fit.py — fit the 3 Delta models and report region-wise held-out force metrics
====================================================================================
Models (identical PairwiseDelta architecture/hyperparameters; only training DATA
differs):
  A  baseline          : role in {baseline, water}
  B  barrier-enriched  : A + role barrier_extra   (targeted, +32 configs)
  C  generic control   : A + role generic_extra   (generic,  +32 configs)

Deltas are fit on the force residual (F_xtb − F_mace) of their TRAINING configs
(no test parents). Evaluated on the held-out reactive TEST set (role=test):
overall + per-region force RMSE and cosine of corrected-MACE (MACE+Delta) vs xTB.
Reports the % TS+post-TS improvement of B and C over the raw MACE baseline.
Pure numpy/ase — system python3. Saves phaseC/fitted_deltas.json + metrics.
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diels_alder" / "neb"))
sys.path.insert(0, str(HERE))
from pairwise_delta import PairwiseDelta
REGION_ORDER = ["reactant_basin", "pre_TS_rising", "TS_region", "post_TS", "product_basin"]


def metrics(Fp, Fx):
    rmse, cos = [], []
    for fp, fx in zip(Fp, Fx):
        fp, fx = np.asarray(fp).ravel(), np.asarray(fx).ravel()
        rmse.append(np.sqrt(np.mean((fp - fx) ** 2)))
        cos.append(fp @ fx / (np.linalg.norm(fp) * np.linalg.norm(fx) + 1e-12))
    return np.array(rmse), np.array(cos)


def main():
    man = pd.read_csv(HERE / "phaseC" / "manifest.csv")
    m = np.load(HERE / "phaseC" / "mace.npz", allow_pickle=True)
    x = np.load(HERE / "phaseC" / "xtb.npz", allow_pickle=True)
    Fm, Fx = list(m["forces"]), list(x["forces"])
    files = sorted(glob.glob(str(HERE / "phaseC" / "geoms" / "*.xyz")))
    atoms = [read(f) for f in files]
    assert len(atoms) == len(man) == len(Fm)
    role = man["role"].values; reg = man["region"].values

    train = {
        "A_baseline": np.isin(role, ["baseline", "water"]),
        "B_barrier":  np.isin(role, ["baseline", "water", "barrier_extra"]),
        "C_generic":  np.isin(role, ["baseline", "water", "generic_extra"]),
    }
    deltas = {}
    for name, mask in train.items():
        idx = np.where(mask)[0]
        dF = [np.asarray(Fx[i]) - np.asarray(Fm[i]) for i in idx]
        deltas[name] = PairwiseDelta().fit([atoms[i] for i in idx], dF)
        print(f"fit {name}: {len(idx)} train configs "
              f"({(role[idx]=='barrier_extra').sum()} barrier, "
              f"{(role[idx]=='generic_extra').sum()} generic, "
              f"{(role[idx]=='water').sum()} water)")

    # held-out reactive test
    test = np.where(role == "test")[0]
    Ft_x = [Fx[i] for i in test]
    results = {}
    for label, delta in [("raw_MACE", None)] + [(k, v) for k, v in deltas.items()]:
        if delta is None:
            Fc = [Fm[i] for i in test]
        else:
            Fc = [np.asarray(Fm[i]) + delta(atoms[i])[1] for i in test]
        rmse, cos = metrics(Fc, Ft_x)
        rr = {"overall": {"rmse": float(rmse.mean()), "cos": float(cos.mean()),
                          "n": int(len(test))}}
        for rg in REGION_ORDER:
            sel = reg[test] == rg
            if sel.any():
                rr[rg] = {"rmse": float(rmse[sel].mean()),
                          "cos": float(cos[sel].mean()), "n": int(sel.sum())}
        results[label] = rr
    (HERE / "phaseC" / "force_test.json").write_text(json.dumps(results, indent=2))
    (HERE / "phaseC" / "fitted_deltas.json").write_text(
        json.dumps({k: v.to_dict() for k, v in deltas.items()}, indent=2))

    # ---- report ----
    print("\n=== held-out reactive-test force RMSE (eV/Å) by region ===")
    hdr = ["model", "overall"] + REGION_ORDER
    print("  " + " ".join(f"{h[:12]:>13s}" for h in hdr))
    for label, rr in results.items():
        cells = [label] + [f"{rr[k]['rmse']:.3f}" if k in rr else "-"
                           for k in ["overall"] + REGION_ORDER]
        print("  " + " ".join(f"{c:>13s}" for c in cells))
    print("\n=== cosine similarity by region ===")
    for label, rr in results.items():
        cells = [label] + [f"{rr[k]['cos']:.3f}" if k in rr else "-"
                           for k in ["overall"] + REGION_ORDER]
        print("  " + " ".join(f"{c:>13s}" for c in cells))

    def ts_region_rmse(rr):
        num = sum(rr[k]["rmse"] * rr[k]["n"] for k in ("TS_region", "post_TS") if k in rr)
        den = sum(rr[k]["n"] for k in ("TS_region", "post_TS") if k in rr)
        return num / den
    base = ts_region_rmse(results["raw_MACE"])
    print(f"\n=== TS+post-TS region force RMSE (the key metric) ===")
    for label, rr in results.items():
        t = ts_region_rmse(rr)
        imp = 100 * (1 - t / base)
        print(f"  {label:14s} {t:.3f} eV/Å   ({imp:+.0f}% vs raw MACE)")
    print(f"\nsaved -> phaseC/force_test.json, phaseC/fitted_deltas.json")


if __name__ == "__main__":
    main()
