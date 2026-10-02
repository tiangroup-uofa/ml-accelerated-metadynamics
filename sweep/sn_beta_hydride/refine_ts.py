#!/usr/bin/env python3
"""
refine_ts.py
============
Stage 3: first-order saddle search with **Sella** from a selected path image,
following the repository's established methodology
(``sweep/corrections/phase2_lib.py::sella_saddle`` and ``aug21_handoff.py``):
``Sella(order=1)``, Cartesian coordinates by default (``internal=false``),
tight ``fmax = 1e-4`` eV/Å, float64 model, seeded from a good guess (the NEB
maximum), never from a crude interpolation midpoint. Fixed atoms are honoured.

Cost is reported both as Sella's ``nsteps`` and as genuine calculator calls —
the Aug-21 benchmark found ``nsteps`` undercounts evaluations by 1-4x.

Outputs (ts/): ts_sella.xyz, sella.traj, sella.log, refine_ts.json

    python refine_ts.py --config config.json                   # path/ts_candidate.xyz
    python refine_ts.py --config config.json --band outputs/.../path/band.xyz --image 7
    python refine_ts.py --config config.json --start my_guess.xyz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from ase.io import read, write  # noqa: E402

from calculators import add_calc_args, model_from_config, parse_calc_settings  # noqa: E402
from config_io import load_config, output_root, section, write_json  # noqa: E402
from geometry import per_atom_displacement  # noqa: E402
from validate_structure import (ValidationError, print_report, require_valid,  # noqa: E402
                                validate_single)
from workflow import (apply_fixed, banner, cv_values, ef_record, provenance,  # noqa: E402
                      stage_dir, strip_calc)

DEFAULTS = {"order": 1, "internal": False, "fmax": 1e-4, "steps": 500}


def pick_start(cfg, tag, start=None, band=None, image=None):
    if start:
        return read(str(start)), str(start)
    if band is not None or image is not None:
        band = Path(band) if band else output_root(cfg, tag) / "path" / "band.xyz"
        frames = read(str(band), index=":")
        if image is None or not (0 <= image < len(frames)):
            raise ValueError(f"--image must be in [0, {len(frames) - 1}] for {band}")
        return frames[image], f"{band}@{image}"
    p = output_root(cfg, tag) / "path" / "ts_candidate.xyz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run run_path.py or pass --start/--band")
    return read(str(p)), str(p)


def run(cfg, model=None, variant=None, device=None, dtype=None, start=None, band=None,
        image=None, out=None, handle=None) -> dict:
    import sella

    banner(cfg)
    opts = {**DEFAULTS, **section(cfg, "sella")}
    tag = (handle.settings if handle else
           parse_calc_settings(cfg, model, variant, device, dtype)).tag
    at0, src = pick_start(cfg, tag, start, band, image)
    rep = validate_single(at0, cfg, f"TS guess ({src})")
    require_valid(rep, "TS starting geometry")
    print_report(rep)

    handle = handle or model_from_config(cfg, model, variant, device, dtype)
    handle.check_elements(at0)
    out = stage_dir(cfg, tag, "ts", out)

    at = strip_calc(at0)
    fixed = apply_fixed(at, cfg)
    at.calc = handle.new_calc()
    init = ef_record(at)
    cv0 = cv_values(cfg, at)
    calls0 = handle.total_calls
    print(f"== Sella saddle search ({tag}): order={opts['order']} internal={opts['internal']} "
          f"fmax={opts['fmax']} steps={opts['steps']} ==", flush=True)
    print(f"  start: E={init['energy_eV']:.6f} eV fmax={init['fmax_eV_A']:.4f} "
          + " ".join(f"{k}={v:.3f}" for k, v in cv0.items()), flush=True)
    dyn = sella.Sella(at, order=int(opts["order"]), internal=bool(opts["internal"]),
                      trajectory=str(out / "sella.traj"), logfile=str(out / "sella.log"))
    sella_conv, err = False, None
    try:
        sella_conv = bool(dyn.run(fmax=float(opts["fmax"]), steps=int(opts["steps"])))
    except Exception as e:  # keep the record; report rather than hide
        err = f"{type(e).__name__}: {e}"
        print(f"  !! Sella raised {err}")
    final = ef_record(at)
    # Sella tests its own *projected* forces; check the plain free-atom criterion
    # independently and only call it converged when both agree.
    fmax_ok = final["fmax_eV_A"] <= float(opts["fmax"])
    converged = sella_conv and fmax_ok
    if sella_conv != fmax_ok:
        print(f"  !! Sella reported converged={sella_conv} but free-atom fmax "
              f"{final['fmax_eV_A']:.3e} {'<=' if fmax_ok else '>'} target {opts['fmax']}")
    cv1 = cv_values(cfg, at)
    disp = per_atom_displacement(strip_calc(at), strip_calc(at0), align=not fixed)
    clean = strip_calc(at)
    clean.info.update({"energy_eV": final["energy_eV"], "sella_converged": converged,
                       "model": tag})
    write(str(out / "ts_sella.xyz"), clean)
    post = validate_single(clean, cfg, "ts_sella")

    summary = {
        **provenance(cfg, handle, stage="refine_ts"),
        "start": src, "settings": opts, "fixed_atoms": fixed,
        "converged": converged,
        "sella_reported_converged": sella_conv,
        "fmax_criterion_met": fmax_ok,
        "convergence_note": "converged = Sella's projected-force test AND free-atom fmax <= target",
        "error": err,
        "iterations_nsteps": int(getattr(dyn, "nsteps", -1)),
        "calculator_calls": handle.total_calls - calls0,
        "initial": init, "final": final,
        "energy_change_eV": final["energy_eV"] - init["energy_eV"],
        "cvs_initial": cv0, "cvs_final": cv1,
        "rmsd_from_start_A": float(np.sqrt(np.mean(disp ** 2))),
        "max_displacement_from_start_A": float(disp.max()),
        "post_validation": post,
        "status": ("Sella converged to a stationary point; order NOT yet verified — run "
                   "validate_ts.py (Hessian)" if converged else "Sella did NOT converge"),
        "files": {"xyz": str(out / "ts_sella.xyz"), "traj": str(out / "sella.traj"),
                  "log": str(out / "sella.log")},
    }
    write_json(out / "refine_ts.json", summary)
    print(f"  converged={converged} nsteps={summary['iterations_nsteps']} "
          f"calls={summary['calculator_calls']} E={final['energy_eV']:.6f} eV "
          f"fmax={final['fmax_eV_A']:.2e} eV/Å")
    print("  final CVs: " + " ".join(f"{k}={v:.4f}" for k, v in cv1.items()))
    if post["status"] == "FAIL":
        print_report(post)
    print(f"  wrote {out}/refine_ts.json, ts_sella.xyz, sella.traj")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sella first-order saddle refinement")
    add_calc_args(ap)
    ap.add_argument("--start", default=None, help="explicit starting geometry")
    ap.add_argument("--band", default=None, help="band file (default: path/band.xyz)")
    ap.add_argument("--image", type=int, default=None, help="image index within --band")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    try:
        s = run(cfg, a.model, a.variant, a.device, a.dtype, a.start, a.band, a.image, a.out)
    except (ValidationError, FileNotFoundError, ValueError) as e:
        print(f"ABORTED: {e}")
        return 1
    return 0 if s["converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
