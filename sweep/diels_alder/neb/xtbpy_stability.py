#!/usr/bin/env python3
"""
xtbpy_stability.py  —  Phase A1 stability gate for xtb-python
=============================================================
Decide whether the `xtb-python` (`xtb.ase.calculator.XTB`) binding is reliable
enough to drive the full 13-image standardized ASE-NEB, BEFORE committing to it.
Runs xtb ONLY (no MACE in the process) so this isolates xtb-python behaviour and
matches how the xTB-NEB process will actually run.

Battery (each wrapped so an SCF failure / crash is caught, not fatal):
  1. single-point energy            — reactant, TS, distorted intermediates
  2. forces                          — same geometries
  3. BFGS geometry optimization      — reactant complex (forming bonds pinned)
  4. repeated evaluations            — 20× FRESH calculators on the same geometry
                                       (the NEB usage pattern) → determinism check
  5. small NEB smoke test            — 5-image seeded CI-NEB via the real driver

Two SCF configurations are compared to document whether smearing is *needed*:
  - plain    : GFN2 defaults (electronic_temperature≈300 K)
  - robust   : electronic_temperature=1000 K, max_iterations=500, accuracy=1.0
               (the settings already in calculators.make_factory)

PASS/FAIL criteria (all must hold for "trustworthy for the full NEB"):
  C1  single-point + forces succeed on ALL geometries (no SCF failure / crash)
  C2  determinism: 20 fresh-instance energies have std < 1e-6 eV (xtb is
      deterministic; nonzero spread = a red flag)
  C3  BFGS optimization converges (fmax < 0.05 eV/Å) without crash
  C4  NEB smoke test completes without crash; energies finite; band physical
  C5  no OpenMP/runtime crash across the whole battery
Verdict: xtb-python is trustworthy iff C1..C5 hold under the *robust* config.
"""
from __future__ import annotations
import os, sys, json, time, traceback, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
from ase.io import read
from ase.optimize import BFGS
from ase.constraints import FixBondLengths

HERE = Path(__file__).resolve().parent
DA = HERE.parent
FORMING = [[0, 5], [3, 4]]
DETERMINISM_TOL = 1e-6      # eV
OPT_FMAX = 0.05


def xtb_calc(config):
    from xtb.ase.calculator import XTB
    if config == "plain":
        return XTB(method="GFN2-xTB")
    return XTB(method="GFN2-xTB", electronic_temperature=1000.0,
               max_iterations=500, accuracy=1.0)


def geoms():
    """reactant, TS, and distorted barrier-region intermediates (+ rattled)."""
    g = {"reactant": read(str(DA / "start.xyz")),
         "product": read(str(DA / "end.xyz")),
         "TS_guess": read(str(DA / "cscan_ts_guess.xyz"))}
    for k in (5, 8, 10, 12):            # cscan points across the barrier region
        g[f"scan_f{k:02d}"] = read(str(DA / "cscan" / f"f_{k:02d}.xyz"))
    rng = np.random.default_rng(0)      # heavily distorted worst-case geometries
    for k in (8, 10):
        at = read(str(DA / "cscan" / f"f_{k:02d}.xyz"))
        at.positions += rng.normal(0, 0.15, at.positions.shape)
        g[f"rattled_f{k:02d}"] = at
    return g


def forming(at):
    p = at.get_positions()
    return (float(np.linalg.norm(p[0]-p[5])), float(np.linalg.norm(p[3]-p[4])))


def try_call(fn):
    try:
        return {"ok": True, "value": fn(), "error": None}
    except Exception as e:
        return {"ok": False, "value": None,
                "error": f"{type(e).__name__}: {str(e).splitlines()[-1][:200]}"}


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    G = geoms()
    report = {"config": {}, "criteria": {}}

    for config in ("plain", "robust"):
        c = {"single_point": {}, "forces": {}, "timing_ms": {}}
        # 1+2 single point energy & forces on every geometry
        for name, at in G.items():
            def sp(at=at):
                a = at.copy(); a.calc = xtb_calc(config)
                t = time.time(); e = float(a.get_potential_energy())
                f = a.get_forces(); dt = (time.time()-t)*1000
                assert np.isfinite(e) and np.isfinite(f).all()
                return e, float(np.abs(f).max()), dt
            r = try_call(sp)
            c["single_point"][name] = {"ok": r["ok"], "error": r["error"],
                                       "energy_eV": r["value"][0] if r["ok"] else None,
                                       "fmax": r["value"][1] if r["ok"] else None}
            if r["ok"]:
                c["timing_ms"][name] = round(r["value"][2], 1)
        report["config"][config] = c

    # 3 BFGS optimization (robust config)
    def do_opt():
        a = G["reactant"].copy(); a.calc = xtb_calc("robust")
        a.set_constraint(FixBondLengths(FORMING))
        conv = BFGS(a, logfile=None).run(fmax=OPT_FMAX, steps=300)
        fmax = float(np.linalg.norm(a.get_forces(), axis=1).max())
        return {"converged": bool(conv), "final_fmax": fmax}
    report["optimization"] = try_call(do_opt)

    # 4 determinism: 20 FRESH calculators on the same distorted geometry
    def determinism(name):
        es, fs = [], []
        for _ in range(20):
            a = G[name].copy(); a.calc = xtb_calc("robust")   # fresh instance each time
            es.append(float(a.get_potential_energy()))
            fs.append(a.get_forces())
        es = np.array(es)
        f_spread = float(np.max([np.abs(fs[i]-fs[0]).max() for i in range(20)]))
        return {"n": 20, "energy_mean_eV": float(es.mean()),
                "energy_std_eV": float(es.std()),
                "energy_ptp_eV": float(np.ptp(es)),
                "force_max_dev_eV_A": f_spread}
    report["determinism"] = {name: try_call(lambda name=name: determinism(name))
                             for name in ("TS_guess", "rattled_f10")}

    # 5 NEB smoke test: 5-image seeded CI-NEB via the real driver (robust xtb)
    def smoke():
        from calculators import make_factory
        from neb import run_neb, optimize_endpoint, FORMING as F2
        make = make_factory("xtb")     # robust settings, Identity correction
        seed = [read(str(p)) for p in sorted((DA / "cscan").glob("f_*.xyz"))]
        r_opt, _ = optimize_endpoint(G["reactant"], make(), fmax=0.05, fix_bonds=F2)
        p_opt, _ = optimize_endpoint(G["product"], make(), fmax=0.05)
        images, energies = run_neb(r_opt, p_opt, make, n_images=5, seed_band=seed,
                                   k=0.5, fmax=0.1, steps=40)   # capped: smoke only
        fbonds = [forming(im) for im in images]
        return {"n_images": len(images),
                "energies_finite": bool(np.isfinite(energies).all()),
                "rel_kcal": [round(float((e-energies[0])*23.060548), 1) for e in energies],
                "forming_CC": [[round(a, 2), round(b, 2)] for a, b in fbonds],
                "band_physical": bool(all(1.2 < a < 6 and 1.2 < b < 6 for a, b in fbonds))}
    report["neb_smoke"] = try_call(smoke)

    # ---- evaluate criteria (under robust config) ----
    rob = report["config"]["robust"]["single_point"]
    C1 = all(v["ok"] for v in rob.values())
    det = report["determinism"]
    C2 = all(d["ok"] and d["value"]["energy_std_eV"] < DETERMINISM_TOL for d in det.values())
    C3 = report["optimization"]["ok"] and report["optimization"]["value"]["converged"]
    sm = report["neb_smoke"]
    C4 = sm["ok"] and sm["value"]["energies_finite"] and sm["value"]["band_physical"]
    C5 = report["optimization"]["ok"] and sm["ok"] and C1   # no crash anywhere
    plain = report["config"]["plain"]["single_point"]
    report["criteria"] = {"C1_singlepoint_forces": C1, "C2_determinism": C2,
                          "C3_optimization": C3, "C4_neb_smoke": C4, "C5_no_crash": C5,
                          "plain_scf_success": all(v["ok"] for v in plain.values())}
    report["verdict_trustworthy"] = bool(C1 and C2 and C3 and C4 and C5)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "xtbpy_stability.json").write_text(json.dumps(report, indent=2))

    # ---- printed summary ----
    print("=" * 68)
    print("A1  xtb-python STABILITY GATE")
    print("=" * 68)
    for config in ("plain", "robust"):
        sp = report["config"][config]["single_point"]
        ok = sum(v["ok"] for v in sp.values())
        print(f"\n[{config}] single-point/forces: {ok}/{len(sp)} geometries OK")
        for name, v in sp.items():
            tag = "ok " if v["ok"] else "FAIL"
            extra = (f"E={v['energy_eV']:.4f} eV fmax={v['fmax']:.3f}" if v["ok"]
                     else v["error"])
            print(f"   {tag} {name:14s} {extra}")
    print("\noptimization:", report["optimization"].get("value") or report["optimization"]["error"])
    for name, d in det.items():
        if d["ok"]:
            v = d["value"]
            print(f"determinism {name:12s}: 20 fresh evals  E_std={v['energy_std_eV']:.2e} eV  "
                  f"ptp={v['energy_ptp_eV']:.2e}  Fdev={v['force_max_dev_eV_A']:.2e}")
        else:
            print(f"determinism {name:12s}: FAIL {d['error']}")
    print("neb smoke:", report["neb_smoke"].get("value") or report["neb_smoke"]["error"])
    print("\n--- CRITERIA (robust config) ---")
    for k, v in report["criteria"].items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nVERDICT: xtb-python is {'TRUSTWORTHY' if report['verdict_trustworthy'] else 'NOT trustworthy'} "
          f"for the full 13-image NEB")
    print(f"saved -> {HERE/'results'/'xtbpy_stability.json'}")


if __name__ == "__main__":
    main()
