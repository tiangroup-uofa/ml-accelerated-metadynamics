#!/usr/bin/env python3
"""
phase2_loop.py — Phase 2 controlled iterative correction loop.

    corrected MACE -> Sella TS -> query xTB at/near that TS -> add reference data
    -> refit the SAME pairwise Delta -> re-search TS -> repeat

Iteration 0 is the verified Phase C "B_barrier" Delta (baseline + water +
barrier_extra = 106 configs), loaded from phaseC/fitted_deltas.json and NOT
refit differently. Every later iteration refits the identical architecture
(cutoff 3.6, n_rbf 8, rmin 0.7, ridge 1e-3) on a strictly growing dataset.

Controls held fixed throughout:
  * held-out reactive test set (48 configs, no-leakage parent split) never trained on
  * water droplet sanity re-measured every iteration
  * correction architecture / hyperparameters frozen

Run:  /usr/bin/python3 phase2_loop.py [--max-iter 5]
Resumable: completed iterations are skipped if their metrics.json exists.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read, write
from ase.optimize import FIRE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase2_lib as L  # noqa: E402
from pairwise_delta import PairwiseDelta  # noqa: E402

PC = HERE / "phaseC"
OUT = HERE / "phase2"
OUT.mkdir(exist_ok=True)

REGION_ORDER = ["reactant_basin", "pre_TS_rising", "TS_region", "post_TS", "product_basin"]
BASE_ROLES = ["baseline", "water", "barrier_extra"]  # == Phase C model B

# query packet design (see PHASE2 findings for justification)
MODE_DISPLACEMENTS = [-0.10, -0.05, 0.05, 0.10]   # Å along the imaginary mode
N_RANDOM = 2
RANDOM_AMP = 0.05                                  # Å
RNG_SEED = 20250807


# --------------------------------------------------------------------------- #
def load_phaseC():
    """Base training pool + held-out test set, with cached MACE/xTB forces."""
    man = pd.read_csv(PC / "manifest.csv")
    m = np.load(PC / "mace.npz", allow_pickle=True)
    x = np.load(PC / "xtb.npz", allow_pickle=True)
    Fm, Fx = list(m["forces"]), list(x["forces"])
    files = sorted(glob.glob(str(PC / "geoms" / "*.xyz")))
    atoms = [read(f) for f in files]
    assert len(atoms) == len(man) == len(Fm) == len(Fx)

    tr = np.where(man.role.isin(BASE_ROLES).values)[0]
    te = np.where((man.role == "test").values)[0]
    train = [(atoms[i], np.asarray(Fm[i]), np.asarray(Fx[i])) for i in tr]
    test = {
        "atoms": [atoms[i] for i in te],
        "F_mace": [np.asarray(Fm[i]) for i in te],
        "F_xtb": [np.asarray(Fx[i]) for i in te],
        "region": man.region.values[te],
    }
    return train, test


def heldout_metrics(delta, test):
    """Force RMSE / cosine of corrected MACE vs xTB on the untouched test split."""
    rmse, cos = [], []
    for at, fm, fx in zip(test["atoms"], test["F_mace"], test["F_xtb"]):
        fc = fm if delta is None else fm + delta(at)[1]
        r, c = L.force_metrics(fc, fx)
        rmse.append(r)
        cos.append(c)
    rmse, cos = np.array(rmse), np.array(cos)
    reg = test["region"]
    out = {"overall": {"rmse": float(rmse.mean()), "cos": float(cos.mean()),
                       "n": int(len(rmse))}}
    for rg in REGION_ORDER:
        sel = reg == rg
        if sel.any():
            out[rg] = {"rmse": float(rmse[sel].mean()), "cos": float(cos[sel].mean()),
                       "n": int(sel.sum())}
    num = sum(out[k]["rmse"] * out[k]["n"] for k in ("TS_region", "post_TS") if k in out)
    den = sum(out[k]["n"] for k in ("TS_region", "post_TS") if k in out)
    out["TS_plus_postTS_rmse"] = float(num / den)
    return out


def make_packet(ts_atoms, calc, vib, rng, outdir):
    """Small local reference packet around the current TS."""
    outdir.mkdir(parents=True, exist_ok=True)
    entries = []

    # 0: the exact TS
    entries.append(("ts", 0.0, ts_atoms.copy()))

    # ± along the imaginary (reaction) mode, in Cartesian unit-norm displacement
    H = L.hessian(ts_atoms, calc)
    m = ts_atoms.get_masses()
    w = 1.0 / np.sqrt(np.repeat(m, 3))
    lam, vecs = np.linalg.eigh(H * np.outer(w, w))
    k = int(np.argmin(lam))
    mode = vecs[:, k] * w
    mode /= np.linalg.norm(mode)
    mode = mode.reshape(-1, 3)
    for d in MODE_DISPLACEMENTS:
        a = ts_atoms.copy()
        a.positions += d * mode
        entries.append((f"mode{d:+.2f}", d, a))

    # small isotropic perturbations (orthogonal information)
    for i in range(N_RANDOM):
        a = ts_atoms.copy()
        pert = rng.normal(scale=RANDOM_AMP / np.sqrt(3), size=a.positions.shape)
        a.positions += pert
        entries.append((f"rand{i}", float(np.linalg.norm(pert)), a))

    paths = []
    meta = []
    for i, (label, amt, a) in enumerate(entries):
        p = outdir / f"q_{i:02d}_{label}.xyz"
        write(str(p), a)
        paths.append(p)
        meta.append({"index": i, "label": label, "displacement": amt,
                     "file": p.name, "forming_CC": L.forming_distances(a),
                     "forming_CC_mean": L.forming_mean(a)})
    return paths, meta, [e[2] for e in entries]


def run_subscript(script, delta_path, extra=()):
    """Run a side-analysis (water / reaction scan) in its own process."""
    cmd = ["/usr/bin/python3", str(HERE / script), "--delta", str(delta_path), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        return {"error": r.stderr[-1500:]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": f"parse: {e}", "stdout": r.stdout[-800:]}


# A converged saddle only counts as a Diels-Alder TS if it is a first-order
# saddle whose imaginary mode actually points along the forming-bond coordinate.
# Random overlap in 48 DOF is ~1/sqrt(48)=0.14; the verified xTB TS scores 0.801,
# the spurious reactant-basin saddle found by naive seeding scored 1.2e-06.
TS_MIN_OVERLAP = 0.30
TS_FORMING_RANGE = (1.5, 3.0)


def validate_ts(info, vib):
    reasons = []
    if not info["converged"]:
        reasons.append("not_converged")
    if vib["n_imaginary"] != 1:
        reasons.append(f"n_imaginary={vib['n_imaginary']}")
    ov = vib["reaction_mode_overlap"]
    if ov is None or ov < TS_MIN_OVERLAP:
        reasons.append(f"reaction_mode_overlap={ov}")
    lo, hi = TS_FORMING_RANGE
    if not (lo <= info["forming_CC_mean"] <= hi):
        reasons.append(f"forming_CC={info['forming_CC_mean']:.3f}_out_of_range")
    return len(reasons) == 0, reasons


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--skip-water", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    train, test = load_phaseC()
    print(f"[setup] base training pool {len(train)} configs; held-out test {len(test['atoms'])}")

    base = L.mace_base("float64")
    reactant = read(str(L.REACTANT_XYZ))
    xtb_ts = read(str(L.XTB_TS_XYZ))  # reference geometry for RMSD only

    # Iteration 0 delta = verified Phase C B_barrier (loaded, not refit)
    d0 = PairwiseDelta.from_dict(json.loads((PC / "fitted_deltas.json").read_text())["B_barrier"])
    print(f"[setup] iteration-0 delta hash {L.delta_hash(d0)} (Phase C B_barrier)")

    delta = d0
    rows = []

    for k in range(args.max_iter + 1):
        it = OUT / f"iter{k:02d}"
        it.mkdir(exist_ok=True)
        mfile = it / "metrics.json"
        if mfile.exists():
            print(f"[iter {k}] already done -> skip")
            rec = json.loads(mfile.read_text())
            rows.append(rec)
            # restore loop state exactly as it was at the end of this iteration
            qr = json.loads((it / "query_refs.json").read_text())
            for fname, fm, fx in zip(qr["files"], qr["F_mace"], qr["F_xtb"]):
                train.append((read(str(it / "query" / fname)),
                              np.array(fm), np.array(fx)))
            delta = L.fit_delta([t[0] for t in train], [t[2] - t[1] for t in train])
            continue

        print(f"\n{'='*70}\n[iter {k}] delta hash {L.delta_hash(delta)} | train {len(train)}\n{'='*70}")
        (it / "delta.json").write_text(json.dumps(delta.to_dict(), indent=2))

        calc = L.corrected(base, delta)

        # ---- Step 0: relaxed concerted scan (also supplies the Sella seed) -- #
        # Project convention (CLAUDE.md): seed the saddle optimizer from a
        # relaxed-scan guess. Seeding from the xTB TS instead sends Sella to a
        # spurious reactant-basin saddle -- see phase2_naive_seed/.
        reaction = run_subscript("phase2_reaction.py", it / "delta.json",
                                 extra=["--save-max", str(it / "scan_max.xyz")])
        print(f"  scan: barrier={reaction.get('barrier_kcal', float('nan')):.2f} "
              f"at {reaction.get('ts_forming', float('nan')):.3f} Å")
        seed = read(str(it / "scan_max.xyz"))

        # ---- Step 1: find the current corrected-MACE TS -------------------- #
        ts, info = L.sella_saddle(seed, calc, fmax=1e-4, steps=300,
                                  traj=it / "sella.traj", log=it / "sella.log")
        write(str(it / "ts.xyz"), ts)
        vib = L.vibrational_analysis(ts, calc)
        ts_valid, ts_reasons = validate_ts(info, vib)
        print(f"  TS: converged={info['converged']} iters={info['iterations']} "
              f"fmax={info['final_fmax']:.2e} forming={info['forming_CC_mean']:.4f} Å")
        print(f"      imaginary modes={vib['n_imaginary']} nu={vib['nu_imag_cm']} "
              f"overlap_with_RC={vib['reaction_mode_overlap']}")
        print(f"      VALID DA saddle: {ts_valid} {ts_reasons if not ts_valid else ''}")
        if not ts_valid:
            # never query xTB around a structure that is not a barrier-region TS;
            # fall back to the scan maximum, which is a legitimate barrier-region
            # geometry even when no certified saddle is found.
            print("      -> falling back to the scan maximum as the query centre")
            ts = seed.copy()
            ts.calc = calc
            write(str(it / "ts.xyz"), ts)
            info = {**info, "fallback_to_scan_max": True,
                    "energy_eV": float(ts.get_potential_energy()),
                    "forming_CC": L.forming_distances(ts),
                    "forming_CC_mean": L.forming_mean(ts)}
            vib = L.vibrational_analysis(ts, calc)

        # barrier vs the frozen xTB reactant (Phase 1C convention)
        r_frozen = reactant.copy()
        r_frozen.calc = calc
        E_r_frozen = float(r_frozen.get_potential_energy())
        barrier = (info["energy_eV"] - E_r_frozen) * L.EV2KCAL

        # barrier vs the reactant relaxed on this corrected PES (rigor check)
        r_rel = reactant.copy()
        r_rel.calc = calc
        FIRE(r_rel, logfile=None).run(fmax=0.02, steps=300)
        E_r_rel = float(r_rel.get_potential_energy())
        barrier_rel = (info["energy_eV"] - E_r_rel) * L.EV2KCAL
        print(f"      barrier={barrier:.2f} kcal/mol (frozen R) | {barrier_rel:.2f} (relaxed R)"
              f" | xTB target {L.XTB_BARRIER_KCAL}")

        # ---- Step 2: query xTB at this exact TS ---------------------------- #
        E_xtb_ts, F_xtb_ts = L.xtb_eval([it / "ts.xyz"], tag=f"ts{k}")
        F_corr_ts = ts.get_forces()
        rmse_ts, cos_ts = L.force_metrics(F_corr_ts, F_xtb_ts[0])
        xtb_fmax_at_ts = float(np.max(np.linalg.norm(F_xtb_ts[0], axis=1)))
        rmsd_xtbts = L.kabsch_rmsd(ts.get_positions(), xtb_ts.get_positions())
        print(f"  xTB@TS: fmax={xtb_fmax_at_ts:.4f} eV/Å | force RMSE(corr,xTB)={rmse_ts:.4f}"
              f" cos={cos_ts:.4f} | RMSD to xTB TS={rmsd_xtbts:.4f} Å")

        # ---- controls ------------------------------------------------------ #
        ho = heldout_metrics(delta, test)
        print(f"  held-out: overall {ho['overall']['rmse']:.4f} | "
              f"TS+postTS {ho['TS_plus_postTS_rmse']:.4f}")

        water = {}
        if not args.skip_water:
            water = run_subscript("phase2_water.py", it / "delta.json")
            print(f"  water: Rg={water.get('rg_mean')} peak={water.get('oo_peak')} "
                  f"stable={water.get('stable')}")

        # ---- Step 3: build + label the next query packet -------------------- #
        paths, meta, qatoms = make_packet(ts, calc, vib, rng, it / "query")
        F_mace_q = []
        for a in qatoms:
            b = a.copy()
            b.calc = base
            F_mace_q.append(b.get_forces())
        E_xtb_q, F_xtb_q = L.xtb_eval(paths, tag=f"q{k}")
        (it / "query_refs.json").write_text(json.dumps({
            "files": [p.name for p in paths],
            "meta": meta,
            "E_xtb": [float(e) for e in E_xtb_q],
            "F_mace": [f.tolist() for f in F_mace_q],
            "F_xtb": [f.tolist() for f in F_xtb_q],
        }))
        print(f"  query packet: {len(paths)} new xTB calculations")

        rec = {
            "iteration": k,
            "delta_hash": L.delta_hash(delta),
            "n_train": len(train),
            "ts": {**info, "vib": vib,
                   "valid_da_saddle": bool(ts_valid),
                   "invalid_reasons": ts_reasons,
                   "barrier_kcal_frozen_R": float(barrier),
                   "barrier_kcal_relaxed_R": float(barrier_rel),
                   "rmsd_to_xtb_ts": float(rmsd_xtbts)},
            "xtb_at_ts": {"E_eV": float(E_xtb_ts[0]),
                          "fmax": xtb_fmax_at_ts,
                          "force_rmse_corr_vs_xtb": float(rmse_ts),
                          "force_cos_corr_vs_xtb": float(cos_ts)},
            "heldout": ho,
            "water": water,
            "reaction": reaction,
            "n_new_xtb_calls": len(paths),
        }
        mfile.write_text(json.dumps(rec, indent=2))
        rows.append(rec)

        # grow the cumulative dataset and refit for the next iteration
        for a, fm, fx in zip(qatoms, F_mace_q, F_xtb_q):
            train.append((a, np.asarray(fm), np.asarray(fx)))

        if k < args.max_iter:
            delta = L.fit_delta([t[0] for t in train],
                                [t[2] - t[1] for t in train])

        # ---- experiment-level stopping check -------------------------------- #
        if k >= 1:
            db = abs(rows[-1]["ts"]["barrier_kcal_frozen_R"]
                     - rows[-2]["ts"]["barrier_kcal_frozen_R"])
            dg = abs(rows[-1]["ts"]["forming_CC_mean"] - rows[-2]["ts"]["forming_CC_mean"])
            print(f"  [stop-check] Δbarrier={db:.3f} kcal/mol  Δforming={dg:.4f} Å")
            if db < 0.5 and dg < 0.01:
                print(f"  ** stopping: barrier and TS geometry both converged at iter {k}")
                break

    df = pd.json_normalize(rows)
    df.to_csv(OUT / "metrics.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT/'metrics.csv'} and {OUT/'summary.json'} ({len(rows)} iterations)")


if __name__ == "__main__":
    main()
