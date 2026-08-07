#!/usr/bin/env python3
"""
a2_standardized_neb.py  —  Phase A2: apples-to-apples ASE-NEB (xTB vs MACE)
===========================================================================
Run GFN2-xTB (via xtb-python, make_factory("xtb")) and MACE-OFF23 through the
*identical* ASE-NEB algorithm and settings, so the barrier comparison is
method-vs-method with no methodological difference:

  seed        = concerted-scan geometries (../cscan/f_*.xyz)
  n_images    = 13
  spring k    = 0.5
  optimizer   = FIRE, fmax = 0.05 eV/Å, climbing image on, method='improvedtangent'
  endpoints   = reactant relaxed with forming bonds pinned; product free
  TS / barrier / reaction energy / forming-C–C = same analyze() logic

xTB uses the robust SCF settings baked into make_factory (etemp=1000, max_iter=500,
accuracy=1.0) — A1 showed these give path energetics identical to plain GFN2.

Per method we record: barrier, reaction energy, TS forming C–C, converged image
count, FIRE optimizer status + final NEB max-force, and band diagnostics
(monotonic ordering, collapsed/fused images, scrambled/flew-apart images).

Output: results/a2_<method>.{xyz,csv}, results/a2_summary.json.
Run: micromamba run -n macemd python a2_standardized_neb.py
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE
from ase.mep import NEB
try:
    from ase.mep.neb import NEBOptimizer      # ODE-based robust NEB optimizer
except Exception:
    NEBOptimizer = None
from calculators import make_factory, PRETTY
from neb import optimize_endpoint, analyze, forming_bonds, mean_forming, FORMING, _subsample

HERE = Path(__file__).resolve().parent
DA = HERE.parent
RES = HERE / "results"
# standardized settings (from the converged MACE run)
N_IMAGES, K, FMAX, STEPS = 13, 0.5, 0.05, 800


def band_diagnostics(images, res):
    coord = np.array([mean_forming(im) for im in images])
    b12 = np.array([forming_bonds(im) for im in images])
    # ordering: forming C–C should decrease monotonically reactant -> product
    inversions = int(np.sum(np.diff(coord) > 0.10))          # +0.1 Å tolerance
    monotonic = inversions == 0
    # collapsed: an interior image fused (either forming bond < 1.3 Å)
    collapsed = [i for i in range(1, len(images)-1) if (b12[i] < 1.3).any()]
    # scrambled: image spread beyond the reactant, or absurd energy
    scrambled = [i for i in range(len(images))
                 if coord[i] > coord[0] + 0.5 or abs(res["rel_kcal"][i]) > 400]
    return dict(coord=coord.round(3).tolist(),
                monotonic_ordering=bool(monotonic), inversions=inversions,
                collapsed_images=collapsed, scrambled_images=scrambled,
                band_physical=bool(monotonic and not collapsed and not scrambled))


def run_method(method, optimizer="fire"):
    make = make_factory(method, dtype="float64" if method == "mace" else "float32")
    reactant = read(str(DA / "start.xyz")); product = read(str(DA / "end.xyz"))
    seed = [read(str(p)) for p in sorted((DA / "cscan").glob("f_*.xyz"))]
    # endpoints (identical treatment)
    r_opt, _ = optimize_endpoint(reactant, make(), fmax=0.03, fix_bonds=FORMING)
    p_opt, _ = optimize_endpoint(product, make(), fmax=0.03)
    # seeded 13-image band
    interior = _subsample(seed, N_IMAGES)[1:-1]
    images = [r_opt.copy()] + interior + [p_opt.copy()]
    for im in images:
        im.calc = make()
    neb = NEB(images, climb=True, k=K, method="improvedtangent",
              allow_shared_calculator=False)
    if optimizer == "neb" and NEBOptimizer is not None:
        opt = NEBOptimizer(neb, logfile=None)          # ODE-based, robust for NEB
    else:
        opt = FIRE(neb, logfile=None)
    converged = bool(opt.run(fmax=FMAX, steps=STEPS))
    nsteps = opt.get_number_of_steps()
    # final NEB max force (on the moving images)
    fmax_final = float(np.sqrt((neb.get_forces() ** 2).sum(axis=1).max()))
    energies = np.array([im.get_potential_energy() for im in images])
    res = analyze(images, energies)
    diag = band_diagnostics(images, res)
    e0 = energies[0]
    for im in images:
        im.info["rel_kcal"] = round(float((im.get_potential_energy()-e0)*23.060548), 2)
    write(str(RES / f"a2_{method}.xyz"), images)
    import csv
    with open(RES / f"a2_{method}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "forming_CC", "rel_kcal", "physical"])
        for i, (c, r, p) in enumerate(zip(res["coord_A"], res["rel_kcal"], res["physical"])):
            w.writerow([i, f"{c:.3f}", f"{r:.3f}", int(p)])
    tf = res["ts_forming_A"]
    return dict(method=method, barrier_kcal=res["barrier_kcal"],
                reaction_energy_kcal=res["reaction_energy_kcal"],
                ts_forming_A=[tf[0], tf[1]], ts_forming_mean=0.5*(tf[0]+tf[1]),
                ts_index=res["ts_index"], n_images=res["n_images"],
                n_dropped=res["n_dropped"], fire_converged=converged,
                fire_steps=nsteps, neb_fmax_final=fmax_final,
                settings=dict(n_images=N_IMAGES, k=K, fmax=FMAX, seed="concerted-scan",
                              optimizer="FIRE", climb=True),
                **diag)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--optimizer", default="fire",
                                                    choices=["fire", "neb"])
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "4"
    try:
        import torch; torch.set_num_threads(4)
    except Exception:
        pass
    RES.mkdir(exist_ok=True)
    summary = {"_optimizer": args.optimizer}
    for method in ("xtb", "mace"):
        print(f"\n### {PRETTY[method]} — standardized ASE-NEB "
              f"({N_IMAGES} images, k={K}, opt={args.optimizer}, fmax={FMAX}) ###", flush=True)
        r = run_method(method, optimizer=args.optimizer)
        summary[method] = r
        print(f"  barrier         = {r['barrier_kcal']:.1f} kcal/mol")
        print(f"  reaction energy = {r['reaction_energy_kcal']:.1f} kcal/mol")
        print(f"  TS forming C–C  = {r['ts_forming_A'][0]:.2f}, {r['ts_forming_A'][1]:.2f} Å")
        print(f"  FIRE converged={r['fire_converged']} in {r['fire_steps']} steps, "
              f"final NEB fmax={r['neb_fmax_final']:.3f} eV/Å")
        print(f"  band: monotonic={r['monotonic_ordering']} (inv={r['inversions']}), "
              f"collapsed={r['collapsed_images']}, scrambled={r['scrambled_images']}, "
              f"physical={r['band_physical']}")

    # previous xTB reference (built-in path / relaxed scan) for context
    prev = json.loads((RES / "neb_results.json").read_text())["xtb"]
    summary["xtb_prev_scan"] = {"barrier_kcal": prev["barrier_kcal"],
                                "reaction_energy_kcal": prev["reaction_energy_kcal"],
                                "ts_forming_mean": 0.5*sum(prev["ts_forming_A"]),
                                "source": prev.get("source", "xtb built-in path / scan")}
    (RES / "a2_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    x, m = summary["xtb"], summary["mace"]
    print("\n=== A2: same-algorithm NEB comparison ===")
    print(f"  {'quantity':22s} {'xTB-NEB':>9s} {'MACE-NEB':>9s} {'xTB scan(prev)':>15s}")
    print(f"  {'barrier (kcal/mol)':22s} {x['barrier_kcal']:>9.1f} {m['barrier_kcal']:>9.1f} "
          f"{summary['xtb_prev_scan']['barrier_kcal']:>15.1f}")
    print(f"  {'reaction E (kcal/mol)':22s} {x['reaction_energy_kcal']:>9.1f} "
          f"{m['reaction_energy_kcal']:>9.1f} {summary['xtb_prev_scan']['reaction_energy_kcal']:>15.1f}")
    print(f"  {'TS forming C–C (Å)':22s} {x['ts_forming_mean']:>9.2f} {m['ts_forming_mean']:>9.2f} "
          f"{summary['xtb_prev_scan']['ts_forming_mean']:>15.2f}")
    gap_same = m['barrier_kcal'] - x['barrier_kcal']
    gap_mixed = m['barrier_kcal'] - summary['xtb_prev_scan']['barrier_kcal']
    print(f"\n  MACE−xTB barrier gap, SAME algorithm : {gap_same:+.1f} kcal/mol")
    print(f"  MACE−xTB barrier gap, MIXED (prev)   : {gap_mixed:+.1f} kcal/mol")
    print(f"\nsaved -> {RES}/a2_summary.json")


if __name__ == "__main__":
    main()
