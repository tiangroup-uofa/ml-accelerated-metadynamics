#!/usr/bin/env python3
"""
run_neb.py
==========
Generate the Diels–Alder reaction profile for **MACE-OFF23** via CI-NEB (ASE) and
compare it to the **GFN2-xTB** profile from xTB's built-in path search (the
relaxed concerted scan, `../cscan_profile.csv`). This mirrors the intended
comparison: xTB has a built-in reaction-path search; MACE does not, so we build
the path with NEB.

The band is seeded with the concerted-scan geometries (a clean physical
reactant→TS→product path) and relaxed with stiff springs — plain IDPP
interpolation produces clashy images and the band slides off the very exothermic
barrier at weak spring constants. Endpoints: the reactant is relaxed with the two
forming C–C bonds pinned (the vdW complex would otherwise dissociate); the product
(cyclohexene) is a genuine minimum.

xTB-via-NEB is available (`--methods xtb`) but unreliable here — its SCF is fragile
on the stretched intermediate geometries and the band scrambles; we therefore
compare against xTB's own path search, not an xTB NEB.

Outputs (results/): neb_mace.xyz, ts_mace.xyz, mep_mace.csv, neb_results.json,
and a printed comparison + which scientific outcome we're in.

Run:  micromamba run -n macemd python run_neb.py --images 13 --k 0.5
"""
from __future__ import annotations
import argparse, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.io import read, write
from calculators import make_factory, PRETTY
from neb import full_workflow

HERE = Path(__file__).resolve().parent
DA = HERE.parent
RES = HERE / "results"
HARTREE_KCAL = 627.509


def xtb_reference():
    """xTB reaction profile from the built-in path search (relaxed concerted scan)."""
    df = pd.read_csv(DA / "cscan_profile.csv")
    coord = df["forming_CC_A"].to_numpy()
    rel = df["rel_E_kcal_mol"].to_numpy()
    i = int(np.argmax(rel))
    dE = float(rel[-1])
    sps, spe = DA / "sp" / "sp_start.log", DA / "sp" / "sp_end.log"
    if sps.exists() and spe.exists():                 # true optimized reaction energy
        E = lambda p: float(re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", p.read_text())[-1])
        dE = (E(spe) - E(sps)) * HARTREE_KCAL
    return dict(coord_A=coord.tolist(), rel_kcal=rel.tolist(),
                barrier_kcal=float(rel[i]), reaction_energy_kcal=dE,
                ts_forming_A=[float(coord[i]), float(coord[i])],
                ts_index=i, source="xtb built-in path search (relaxed scan)")


def save_mace(images, res):
    RES.mkdir(parents=True, exist_ok=True)
    e0 = images[0].get_potential_energy()
    for im in images:
        im.info["rel_kcal"] = (im.get_potential_energy() - e0) * 23.060548
    write(str(RES / "neb_mace.xyz"), images)
    write(str(RES / "ts_mace.xyz"), images[res["ts_index"]])
    import csv
    with open(RES / "mep_mace.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "forming_CC_A", "rel_E_kcal", "physical"])
        for i, (c, e, ok) in enumerate(zip(res["coord_A"], res["rel_kcal"], res["physical"])):
            w.writerow([i, f"{c:.3f}", f"{e:.3f}", int(ok)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["mace"])
    ap.add_argument("--images", type=int, default=13)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    try:
        import torch; torch.set_num_threads(args.threads)
    except Exception:
        pass

    reactant = read(str(DA / "start.xyz"))
    product = read(str(DA / "end.xyz"))
    seed = sorted((DA / "cscan").glob("f_*.xyz"))
    seed_band = [read(str(p)) for p in seed] if seed else None
    print(f"seed band: {len(seed_band) if seed_band else 0} concerted-scan geometries")

    results = {"xtb": xtb_reference()}
    print(f"\nxTB reference (built-in path): barrier {results['xtb']['barrier_kcal']:.1f} "
          f"kcal/mol, ΔE {results['xtb']['reaction_energy_kcal']:.1f}, "
          f"TS {results['xtb']['ts_forming_A'][0]:.2f} Å")

    for m in args.methods:
        print(f"\n### {PRETTY.get(m, m)} — endpoints + CI-NEB ({args.images} images, k={args.k}) ###",
              flush=True)
        try:
            make_calc = make_factory(m, dtype="float64" if m == "mace" else "float32")
            images, res = full_workflow(reactant, product, make_calc,
                                        n_images=args.images, fmax=args.fmax,
                                        seed_band=seed_band, k=args.k)
        except Exception as e:
            print(f"  !! {m} NEB failed: {type(e).__name__}: {e}"); continue
        if m == "mace":
            save_mace(images, res)
        results[m] = {k: v for k, v in res.items()
                      if k not in ("rel_kcal", "coord_A", "physical")}
        results[m]["ts_forming_A"] = list(res["ts_forming_A"])
        tf = res["ts_forming_A"]
        print(f"  barrier         = {res['barrier_kcal']:.1f} kcal/mol"
              f"  ({res['n_dropped']} spurious image(s) dropped)")
        print(f"  reaction energy = {res['reaction_energy_kcal']:.1f} kcal/mol")
        print(f"  TS forming C–C  = {tf[0]:.2f}, {tf[1]:.2f} Å (mean {0.5*(tf[0]+tf[1]):.2f})")

    (RES / "neb_results.json").write_text(json.dumps(results, indent=2, default=float))

    if "mace" in results:
        x, mc = results["xtb"], results["mace"]
        dbar = mc["barrier_kcal"] - x["barrier_kcal"]
        ddE = mc["reaction_energy_kcal"] - x["reaction_energy_kcal"]
        tsx = 0.5 * sum(x["ts_forming_A"]); tsm = 0.5 * sum(mc["ts_forming_A"])
        print("\n=== MACE-OFF23 (NEB) vs GFN2-xTB (built-in path) ===")
        print(f"  {'quantity':22s} {'xTB':>9s} {'MACE':>9s} {'Δ(MACE−xTB)':>13s}")
        print(f"  {'barrier (kcal/mol)':22s} {x['barrier_kcal']:>9.1f} {mc['barrier_kcal']:>9.1f} {dbar:>+13.1f}")
        print(f"  {'reaction E (kcal/mol)':22s} {x['reaction_energy_kcal']:>9.1f} {mc['reaction_energy_kcal']:>9.1f} {ddE:>+13.1f}")
        print(f"  {'TS forming C–C (Å)':22s} {tsx:>9.2f} {tsm:>9.2f} {tsm-tsx:>+13.2f}")
        ts_match = abs(tsm - tsx) < 0.35
        e_match = abs(dbar) < 3.0 and abs(ddE) < 5.0
        print("\n  Scientific outcome:")
        if e_match and ts_match:
            print("  -> (1) MACE reproduces barrier AND reaction energy closely.")
        elif ts_match:
            print("  -> (2) TS location matches (~2.0–2.3 Å) but absolute energetics differ.")
        else:
            print("  -> neither clean outcome — inspect the band (mep_mace.csv).")
    print(f"\nwrote {RES}/  (run plot_neb.py for the figure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
