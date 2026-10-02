#!/usr/bin/env python3
"""
optimize_endpoints.py
=====================
Stage 1: relax the authoritative reactant and product with the selected MACE
foundation model.

  1. strict validation of the pair (validate_structure.py) — abort on errors
  2. build the calculator; abort if the model lacks any element present
  3. per endpoint: initial E/F -> optimizer (fixed atoms honoured) -> final E/F
  4. post-optimization checks: CV drift, covalent bonds formed/broken *during*
     the optimization (e.g. the hydride migrating, glucose detaching from Sn),
     and re-validation of the optimized pair
  5. write <role>_opt.xyz, <role>_opt.traj, <role>_opt.log, endpoints.json

Exit code: 0 ok, 1 validation failed, 2 optimization finished but not converged
or the optimized structures fail validation.

    python optimize_endpoints.py --config config.json [--model mace-polar]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ase.io import read, write  # noqa: E402

from calculators import add_calc_args, model_from_config  # noqa: E402
from config_io import load_config, resolve_path, section, write_json  # noqa: E402
from geometry import bond_set  # noqa: E402
from validate_structure import (print_report, require_valid, validate_atoms,  # noqa: E402
                                validate_pair, vcfg)
from workflow import (EV2KCAL, apply_fixed, banner, cv_values, ef_record,  # noqa: E402
                      make_optimizer, provenance, stage_dir, strip_calc)

DEFAULTS = {"optimizer": "BFGS", "fmax": 0.03, "steps": 800}


def optimize_one(atoms, role, cfg, handle, out: Path, opts: dict) -> dict:
    a = atoms.copy()
    fixed = apply_fixed(a, cfg)
    a.calc = handle.new_calc()
    calls0 = handle.total_calls
    init = ef_record(a)
    cv0 = cv_values(cfg, a)
    bonds0 = bond_set(a, vcfg(cfg)["bond_scale"])
    print(f"  {role}: E0={init['energy_eV']:.6f} eV  fmax0={init['fmax_eV_A']:.4f} eV/Å", flush=True)
    opt = make_optimizer(opts["optimizer"], a, out / f"{role}_opt.traj", out / f"{role}_opt.log")
    converged = bool(opt.run(fmax=opts["fmax"], steps=opts["steps"]))
    final = ef_record(a)
    cv1 = cv_values(cfg, a)
    bonds1 = bond_set(a, vcfg(cfg)["bond_scale"])
    syms = a.get_chemical_symbols()
    lab = lambda p: [f"{syms[p[0]]}{p[0]}", f"{syms[p[1]]}{p[1]}"]
    broken = [lab(p) for p in sorted(bonds0 - bonds1)]
    formed = [lab(p) for p in sorted(bonds1 - bonds0)]
    clean = strip_calc(a)
    clean.info.update({"energy_eV": final["energy_eV"], "role": role,
                       "model": handle.settings.tag})
    write(str(out / f"{role}_opt.xyz"), clean)
    post = validate_atoms(clean, cfg, f"{role}_opt", role).to_dict()
    print(f"  {role}: E ={final['energy_eV']:.6f} eV  fmax ={final['fmax_eV_A']:.4f} eV/Å  "
          f"converged={converged} steps={opt.get_number_of_steps()}  "
          f"post-validation={post['status']}", flush=True)
    if broken or formed:
        print(f"  !! {role}: covalent bonds changed during optimization: "
              f"broken {broken} formed {formed}")
    return {
        "converged": converged,
        "steps": int(opt.get_number_of_steps()),
        "calculator_calls": handle.total_calls - calls0,
        "fixed_atoms": fixed,
        "initial": init, "final": final,
        "delta_E_eV": final["energy_eV"] - init["energy_eV"],
        "cvs_initial": cv0, "cvs_final": cv1,
        "bonds_broken_during_opt": broken, "bonds_formed_during_opt": formed,
        "post_validation": post,
        "files": {"xyz": str(out / f"{role}_opt.xyz"), "traj": str(out / f"{role}_opt.traj"),
                  "log": str(out / f"{role}_opt.log")},
    }


def run(cfg, model=None, variant=None, device=None, dtype=None, reactant=None,
        product=None, out=None, handle=None) -> dict:
    banner(cfg)
    st = cfg.get("structures") or {}
    rpath = Path(reactant) if reactant else resolve_path(cfg, st.get("reactant", "input/reactant.xyz"))
    ppath = Path(product) if product else resolve_path(cfg, st.get("product", "input/product.xyz"))

    report = validate_pair(rpath, ppath, cfg)
    require_valid(report, "reactant/product pair")          # raises before any model load
    print_report(report)

    handle = handle or model_from_config(cfg, model, variant, device, dtype)
    tag = handle.settings.tag
    write_json(stage_dir(cfg, tag, "validation") / "endpoints_input_validation.json", report)
    R, P = read(str(rpath)), read(str(ppath))
    handle.check_elements(R)
    out = stage_dir(cfg, tag, "endpoints", out)
    opts = {**DEFAULTS, **section(cfg, "optimize")}

    print(f"== endpoint optimization with {tag} ({opts['optimizer']}, fmax={opts['fmax']}) ==")
    res = {"reactant": optimize_one(R, "reactant", cfg, handle, out, opts),
           "product": optimize_one(P, "product", cfg, handle, out, opts)}
    post_pair = validate_pair(read(str(out / "reactant_opt.xyz")),
                              read(str(out / "product_opt.xyz")), cfg)
    dE = res["product"]["final"]["energy_eV"] - res["reactant"]["final"]["energy_eV"]
    summary = {
        **provenance(cfg, handle, stage="optimize_endpoints"),
        "inputs": {"reactant": str(rpath), "product": str(ppath)},
        "settings": opts,
        "endpoints": res,
        "post_optimization_pair_validation": post_pair,
        "reaction_energy_model": {
            "note": "E(product_opt) - E(reactant_opt) with this model; not a free energy",
            "eV": dE, "kcal_mol": dE * EV2KCAL},
        "ok": bool(res["reactant"]["converged"] and res["product"]["converged"]
                   and post_pair["status"] != "FAIL"),
    }
    write_json(out / "endpoints.json", summary)
    print(f"  ΔE(product - reactant, optimized, {tag}) = {dE:.4f} eV "
          f"({dE * EV2KCAL:.2f} kcal/mol)")
    if post_pair["status"] == "FAIL":
        print("  !! optimized endpoints FAIL validation:")
        print_report(post_pair)
    print(f"  wrote {out}/endpoints.json")
    return summary


def main(argv=None) -> int:
    from validate_structure import ValidationError

    ap = argparse.ArgumentParser(description="Optimize reactant/product endpoints")
    add_calc_args(ap)
    ap.add_argument("--reactant", default=None)
    ap.add_argument("--product", default=None)
    ap.add_argument("--out", default=None, help="override output directory")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    try:
        s = run(cfg, a.model, a.variant, a.device, a.dtype, a.reactant, a.product, a.out)
    except ValidationError as e:
        print(f"ABORTED: {e}")
        return 1
    return 0 if s["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
