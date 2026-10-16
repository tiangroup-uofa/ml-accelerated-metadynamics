#!/usr/bin/env python3
"""
run_protocol.py
===============
The standardized Sn-beta hydride-transfer protocol for ONE method, so that
MACE-OMOL, MACE-POLAR and GFN2-xTB are run identically from the same config:

  same provenance-labelled R/P input
  -> validate (pair; refuses before any model load)
  -> method-specific endpoint relaxation        (optimize_endpoints.py)
  -> 13-image CI-NEB                            (run_path.py)
  -> Sella refinement of the highest image      (refine_ts.py)
  -> mass-weighted Hessian, exactly-one-imaginary-mode check, overlap with the
     hydride coordinate, +/- mode connectivity relaxation, and barriers against
     BOTH labelled reference states             (validate_ts.py)

Every setting comes from the config (``optimize``, ``path``, ``sella``,
``hessian``, ``connectivity``, ``conformer``); only the calculator changes
between methods. The highest NEB image is ONLY Sella's starting guess and is
never reported as a TS; an unconverged band is recorded as such.

    python run_protocol.py --config config_tian_tt_test.json --model mace-polar --dry-run
    python run_protocol.py --config config_tian_tt_test.json --model mace-polar
    python run_protocol.py --config config_tian_tt_test.json --model gfn2-xtb

Writes <output_dir>/<model>_<variant>/protocol_summary.json.
Exit: 0 certified TS; 1 refused (invalid input / missing dependency);
2 ran but something did not converge or the TS is not certified.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calculators as C  # noqa: E402
from config_io import load_config, output_root, resolve_path, section, write_json  # noqa: E402
from validate_structure import (ValidationError, print_report, require_valid,  # noqa: E402
                                validate_pair)
from workflow import banner, provenance, structure_class  # noqa: E402

STAGES = ("optimize_endpoints", "run_path", "refine_ts", "validate_ts")


def plan(cfg, settings, natoms: int | None) -> dict:
    """What will run, with rough force-call counts (no model needed)."""
    import optimize_endpoints as OE
    import refine_ts as RT
    import run_path as RP
    import validate_ts as VT
    from connectivity import options as conn_options

    o = {**OE.DEFAULTS, **section(cfg, "optimize")}
    p = {**RP.DEFAULTS, **section(cfg, "path")}
    s = {**RT.DEFAULTS, **section(cfg, "sella")}
    h = {**VT.DEFAULTS, **section(cfg, "hessian")}
    c = conn_options(cfg)
    n = natoms or 0
    moving = int(p["n_images"]) - 2
    return {
        "method": settings.model, "variant": settings.variant, "device": settings.device,
        "dtype": settings.dtype, "options": settings.options,
        "charge_spin_handling": C.MODELS[settings.model].charge_spin_mode,
        "settings": {"optimize": o, "path": p, "sella": s, "hessian": h, "connectivity": c},
        "force_call_bounds": {
            "endpoints_max": 2 * (int(o["steps"]) + 1),
            "neb_max": moving * (int(p["steps"]) + 1),
            "sella_max": "~" + str(int(s["steps"]) + 1) + " x 1-4 (Sella nsteps undercounts calls)",
            "hessian": 6 * n,
            "connectivity_max": 2 * (int(c["steps"]) + 1),
        },
        "outputs": str(output_root(cfg, settings.tag)),
    }


def run(cfg, model=None, variant=None, device=None, dtype=None, handle=None,
        stop_on_unconverged_neb=False) -> dict:
    import optimize_endpoints as OE
    import refine_ts as RT
    import run_path as RP
    import validate_ts as VT

    banner(cfg)
    settings = handle.settings if handle else C.parse_calc_settings(cfg, model, variant,
                                                                    device, dtype)
    st = cfg.get("structures") or {}
    rpath, ppath = resolve_path(cfg, st["reactant"]), resolve_path(cfg, st["product"])
    rep = validate_pair(rpath, ppath, cfg)
    require_valid(rep, "protocol input pair")            # before any model load
    print_report(rep)

    if handle is None:
        dep = C.MODELS[settings.model].dependency_check
        if dep is not None:
            dep()                                       # clear RuntimeError if missing
        handle = C.model_from_config(cfg, model, variant, device, dtype)
    t0 = time.time()
    rec = {"stages": {}}
    s1 = OE.run(cfg, handle=handle)
    rec["stages"]["optimize_endpoints"] = {
        "ok": s1["ok"], "reactant_converged": s1["endpoints"]["reactant"]["converged"],
        "product_converged": s1["endpoints"]["product"]["converged"],
        "reaction_energy_relaxed_endpoint": s1["reaction_energy_model"]}
    status = "ok"
    if not s1["ok"]:
        status = "stopped: endpoint relaxation did not converge or failed validation"
    else:
        s2 = RP.run(cfg, handle=handle)
        rec["stages"]["run_path"] = {
            "converged": s2["converged"], "steps": s2["steps"],
            "neb_fmax_final": s2["neb_fmax_final"], "warnings": s2["warnings"],
            "highest_image": s2["ts_candidate"],
            "note": "highest NEB image = Sella starting guess only; NOT a TS"}
        if not s2["converged"] and stop_on_unconverged_neb:
            status = "stopped: NEB band not converged (--stop-on-unconverged-neb)"
        else:
            s3 = RT.run(cfg, handle=handle)
            rec["stages"]["refine_ts"] = {
                "converged": s3["converged"], "sella_reported_converged":
                    s3["sella_reported_converged"], "fmax_criterion_met": s3["fmax_criterion_met"],
                "nsteps": s3["iterations_nsteps"], "calls": s3["calculator_calls"],
                "final_fmax_eV_A": s3["final"]["fmax_eV_A"], "cvs_final": s3["cvs_final"]}
            s4 = VT.run(cfg, handle=handle)
            rec["stages"]["validate_ts"] = {
                "n_imaginary": s4["n_imaginary"], "imag_freqs_cm": s4["imag_freqs_cm"],
                "reaction_mode_overlap": s4["reaction_mode_overlap"],
                "mode_verdict": s4["verdict"], "ts_certification": s4["ts_certification"],
                "connectivity_verdict": (s4["connectivity"] or {}).get("verdict"),
                "connectivity_skipped": s4["connectivity_skipped"]}
            rec["energetics"] = s4["energetics"]
            rec["ts_certification"] = s4["ts_certification"]
            if not s3["converged"]:
                status = "Sella did not converge (TS not certified)"
    rec["status"] = status
    summary = {**provenance(cfg, handle, stage="run_protocol"),
               "structure_class": structure_class(cfg),
               "inputs": {"reactant": str(rpath), "product": str(ppath)},
               "plan": plan(cfg, handle.settings, None),
               "total_calculator_calls": handle.total_calls,
               "wall_seconds": round(time.time() - t0, 1), **rec}
    out = output_root(cfg, handle.settings.tag) / "protocol_summary.json"
    write_json(out, summary)
    print(f"== protocol {handle.settings.tag}: {status}; "
          f"TS: {rec.get('ts_certification', 'n/a')}")
    print(f"   wrote {out}")
    return summary


def main(argv=None) -> int:
    from ase.io import read

    ap = argparse.ArgumentParser(description="Standardized Sn-beta protocol for one method")
    C.add_calc_args(ap)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate input, check the method's dependency, print the plan; no model load")
    ap.add_argument("--stop-on-unconverged-neb", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    settings = C.parse_calc_settings(cfg, a.model, a.variant, a.device, a.dtype)
    if a.dry_run:
        import json

        banner(cfg)
        st = cfg.get("structures") or {}
        rpath = resolve_path(cfg, st["reactant"])
        rep = validate_pair(rpath, resolve_path(cfg, st["product"]), cfg)
        print(f"input validation: {rep['status']} ({rep['n_errors']} errors, "
              f"{rep['n_warnings']} warnings)")
        dep = C.MODELS[settings.model].dependency_check
        try:
            if dep is not None:
                dep()
            dep_status = "available"
        except RuntimeError as e:
            dep_status = f"MISSING: {str(e).splitlines()[0]}"
        print(f"dependency for {settings.model}: {dep_status}")
        print(json.dumps(plan(cfg, settings, len(read(str(rpath)))), indent=2, default=str))
        return 1 if rep["status"] == "FAIL" else 0
    try:
        s = run(cfg, a.model, a.variant, a.device, a.dtype,
                stop_on_unconverged_neb=a.stop_on_unconverged_neb)
    except (ValidationError, RuntimeError, FileNotFoundError) as e:
        print(f"ABORTED: {e}")
        return 1
    return 0 if str(s.get("ts_certification", "")).startswith("CERTIFIED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
