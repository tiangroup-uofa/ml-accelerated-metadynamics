#!/usr/bin/env python3
"""
run_smoke_test.py
=================
SOFTWARE smoke test for the Sn-beta hydride benchmark while the authoritative
Sn-beta coordinates are unavailable.

    *** Everything this script computes is TEST-ONLY. It uses a synthetic toy
    *** (tests/toy_hydride.py: CH3O- + H2CO hydride exchange) and tiny fixtures.
    *** No number it prints is a Sn-beta energy, barrier or frequency.

Checks
  1. imports of every module
  2. shipped config.json loads and is (correctly) unconfigured
  3. validation + CV machinery on the TEST-ONLY toy, plus deliberate failures
  4. mass-weighted Hessian on an analytic harmonic diatomic
  5. per model (MACE-OMOL, MACE-POLAR): construction, element coverage (Sn),
     energy/force call on the toy, response to total charge
  6. --pipeline: optimize -> NEB -> Sella -> Hessian on the toy with short
     limits, to prove the stage scripts run end to end (not to converge them)

Outputs go to smoke_output/ (git-ignored): smoke_summary.json and, with
--pipeline, the per-stage JSON/XYZ files of the toy run.

    /usr/bin/python3 run_smoke_test.py                       # checks 1-5
    /usr/bin/python3 run_smoke_test.py --pipeline            # + end-to-end toy run
    /usr/bin/python3 run_smoke_test.py --models mace-omol    # one model only
Exit code 0 when every executed check passed (unavailable models are SKIPPED
unless --require-models).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "tests")]

BANNER = "=" * 78 + "\n  TEST-ONLY SMOKE TEST — synthetic structures, NOT Sn-beta, NOT chemistry\n" + "=" * 78


class Smoke:
    def __init__(self):
        self.results = []

    def check(self, name, fn, skip_exc=()):
        t = time.time()
        try:
            detail = fn()
            status = "PASS"
        except skip_exc as e:
            detail, status = f"{type(e).__name__}: {e}", "SKIPPED"
        except Exception as e:
            detail = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
            status = "FAIL"
        dt = time.time() - t
        self.results.append({"check": name, "status": status, "seconds": round(dt, 2),
                             "detail": detail})
        short = detail if isinstance(detail, str) else json.dumps(detail, default=str)
        print(f"[{status:7s}] {name} ({dt:.1f}s)\n          {short[:400]}", flush=True)
        return status == "PASS", detail


class ModelUnavailable(RuntimeError):
    pass


def check_imports():
    import importlib

    mods = ["config_io", "geometry", "calculators", "validate_structure", "reaction_coordinate",
            "vibrations", "workflow", "optimize_endpoints", "run_path", "refine_ts",
            "validate_ts", "toy_hydride"]
    for m in mods:
        importlib.import_module(m)
    import ase
    import sella  # noqa: F401
    from importlib.metadata import version

    return {"modules": mods, "ase": ase.__version__, "sella": version("sella"),
            "mace-torch": version("mace-torch")}


def check_shipped_config():
    from config_io import ConfigError, atom_map, load_config

    cfg = load_config(HERE / "config.json")
    try:
        atom_map(cfg)
    except ConfigError as e:
        return f"loads; atom_map unconfigured as expected -> stages will refuse to run ({str(e).splitlines()[1].strip()} ...)"
    raise AssertionError("shipped config has a configured atom_map — was a structure added?")


def check_validation_and_cvs(tmp):
    import copy

    from reaction_coordinate import ReactionCoordinates
    from toy_hydride import product, reactant, toy_config
    from validate_structure import validate_pair, validate_single

    cfg = toy_config(tmp)
    rep = validate_pair(reactant(), product(), cfg)
    assert rep["status"] == "PASS", rep
    rc = ReactionCoordinates.from_config(cfg, 9, reactant().get_chemical_symbols())
    cr, cp = rc.compute(reactant()), rc.compute(product())
    assert cr["delta_H"] < 0 < cp["delta_H"]
    # deliberate failures must be caught
    bad = reactant()
    bad.positions[7] = bad.positions[6]
    assert validate_single(bad, cfg, "dup")["status"] == "FAIL"
    swapped = validate_pair(product(), reactant(), cfg)
    assert swapped["status"] == "FAIL"
    c2 = copy.deepcopy(cfg)
    c2["atom_map"]["H_transfer"] = None
    assert validate_single(reactant(), c2, "unmapped")["status"] == "FAIL"
    return {"toy_pair": "PASS", "toy_reactant_cvs": cr, "toy_product_cvs": cp,
            "deliberate_failures_caught": ["duplicate atom", "swapped endpoints", "unset mapping"]}


def check_hessian():
    import numpy as np
    from ase import Atoms
    from toy_hydride import Harmonic
    from vibrations import LAMBDA_TO_CM, vibrational_analysis

    a = Atoms("CO", positions=[[0, 0, 0], [1.1, 0, 0]])
    res = vibrational_analysis(a, Harmonic([(0, 1)], 30.0, 1.1))
    m = a.get_masses()
    exp = LAMBDA_TO_CM * np.sqrt(30.0 / (m[0] * m[1] / m.sum()))
    got = max(res["frequencies_cm"])
    assert abs(got - exp) / exp < 1e-5, (got, exp)
    return f"harmonic C-O stretch {got:.2f} cm^-1 vs analytic {exp:.2f} cm^-1"


def build_model(model, cfg):
    import calculators as C

    try:
        if model == "mace-polar":
            C.check_polar_dependency()
        return C.model_from_config(cfg, model=model)
    except RuntimeError as e:
        raise ModelUnavailable(str(e)) from e


def check_model(model, cfg, handles):
    import numpy as np
    from ase import Atoms

    from toy_hydride import reactant

    t = time.time()
    h = build_model(model, cfg)
    handles[model] = h
    load_s = time.time() - t
    info = h.info()
    d = 1.95 / np.sqrt(3)
    o = np.array([[d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]])
    h.check_elements(Atoms("SnO4H4", positions=np.vstack([[0, 0, 0], o, o * 1.5])))
    a = reactant()
    a.calc = h.new_calc()
    t = time.time()
    e = a.get_potential_energy()
    f = a.get_forces()
    ef_s = time.time() - t
    assert np.isfinite(e) and np.all(np.isfinite(f)) and f.shape == (9, 3)
    from calculators import ChargeSpinCalculator

    b = reactant()
    b.calc = ChargeSpinCalculator(h.base, 0, 2)  # same geometry, neutral doublet
    e_neutral = b.get_potential_energy()
    return {"model_info": info, "load_seconds": round(load_s, 1),
            "ef_call_seconds": round(ef_s, 3), "Sn_supported": True,
            "toy_energy_eV_TEST_ONLY": e,
            "charge_response_eV_TEST_ONLY": e - e_neutral,
            "responds_to_charge": bool(abs(e - e_neutral) > 1e-3)}


def run_pipeline(model, handles, tmp):
    from config_io import load_config
    from toy_hydride import toy_config

    import optimize_endpoints
    import refine_ts
    import run_path
    import validate_ts

    toy_config(tmp)
    cfg = load_config(tmp / "TEST-ONLY_toy_config.json")
    cfg["calculator"]["model"] = model
    h = handles.get(model) or build_model(model, cfg)
    stages = {}
    s1 = optimize_endpoints.run(cfg, handle=h)
    stages["optimize_endpoints"] = {"ran": True, "reactant_converged": s1["endpoints"]["reactant"]["converged"],
                                    "product_converged": s1["endpoints"]["product"]["converged"]}
    s2 = run_path.run(cfg, handle=h)
    stages["run_path"] = {"ran": True, "converged": s2["converged"], "n_images": len(s2["images"]),
                          "ts_candidate_image": s2["ts_candidate"]["image"]}
    s3 = refine_ts.run(cfg, handle=h)
    stages["refine_ts"] = {"ran": True, "converged": s3["converged"],
                           "nsteps": s3["iterations_nsteps"], "calls": s3["calculator_calls"]}
    s4 = validate_ts.run(cfg, handle=h)
    stages["validate_ts"] = {"ran": True, "n_imaginary": s4["n_imaginary"],
                             "verdict_TEST_ONLY": s4["verdict"]}
    root = Path(cfg["_config_dir"]) / "outputs" / h.settings.tag
    for f in ["endpoints/endpoints.json", "path/path.json", "path/path.csv", "path/band.xyz",
              "path/neb.traj", "path/ts_candidate.xyz", "ts/refine_ts.json", "ts/ts_sella.xyz",
              "ts/sella.traj", "vib/validate_ts.json", "vib/hessian.npy", "vib/frequencies.csv"]:
        assert (root / f).exists(), f"missing output {f}"
    stages["outputs"] = str(root)
    return stages


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TEST-ONLY software smoke test")
    ap.add_argument("--models", nargs="*", default=["mace-omol", "mace-polar"])
    ap.add_argument("--pipeline", action="store_true", help="also run all stages on the toy")
    ap.add_argument("--pipeline-model", default=None,
                    help="model for --pipeline (default: first available, POLAR-S preferred)")
    ap.add_argument("--require-models", action="store_true",
                    help="treat an unavailable model as a failure")
    ap.add_argument("--out", default=str(HERE / "smoke_output"))
    a = ap.parse_args(argv)

    print(BANNER)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sm = Smoke()
    sm.check("imports", check_imports)
    sm.check("shipped config.json", check_shipped_config)
    sm.check("validation + CVs on TEST-ONLY toy", lambda: check_validation_and_cvs(out / "toy"))
    sm.check("mass-weighted Hessian (analytic diatomic)", check_hessian)

    from config_io import load_config

    handles = {}
    skip = () if a.require_models else (ModelUnavailable,)
    for model in a.models:
        cfg = load_config(out / "toy" / "TEST-ONLY_toy_config.json")
        sm.check(f"model {model}: build + E/F + Sn coverage + charge response",
                 lambda m=model, c=cfg: check_model(m, c, handles), skip)

    if a.pipeline:
        pm = a.pipeline_model or ("mace-polar" if "mace-polar" in handles else
                                  next(iter(handles), None))
        if pm is None:
            sm.results.append({"check": "pipeline", "status": "SKIPPED",
                               "detail": "no model available", "seconds": 0})
            print("[SKIPPED] pipeline: no model available")
        else:
            sm.check(f"end-to-end pipeline on TEST-ONLY toy ({pm})",
                     lambda: run_pipeline(pm, handles, out / "pipeline"))

    statuses = [r["status"] for r in sm.results]
    summary = {"TEST_ONLY": True,
               "warning": "synthetic structures; not Sn-beta; numbers are not chemistry",
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "python": sys.version.split()[0], "checks": sm.results,
               "n_pass": statuses.count("PASS"), "n_fail": statuses.count("FAIL"),
               "n_skipped": statuses.count("SKIPPED")}
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("=" * 78)
    print(f"  {summary['n_pass']} passed, {summary['n_fail']} failed, "
          f"{summary['n_skipped']} skipped  ->  {out / 'smoke_summary.json'}")
    print("  (TEST-ONLY: software verification, not Sn-beta results)")
    return 1 if summary["n_fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
