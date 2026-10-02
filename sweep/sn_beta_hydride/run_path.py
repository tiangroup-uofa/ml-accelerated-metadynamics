#!/usr/bin/env python3
"""
run_path.py
===========
Stage 2: endpoint-to-path with a climbing-image NEB (ASE), generalized from the
Diels-Alder NEB (``sweep/diels_alder/neb/neb.py``) with the same proven defaults:
``method="improvedtangent"``, ``allow_shared_calculator=False`` (one cheap
wrapper per image around one loaded model), FIRE, climbing image, k = 0.5,
13 images. Nothing here knows about specific atoms: CVs come from config.

  1. load the *optimized* endpoints from stage 1 (or explicit files with
     --allow-unoptimized) and validate the pair — abort on errors
  2. (non-periodic, no fixed atoms) Kabsch-align product onto reactant so the
     interpolation carries no rigid rotation; atom order is never changed
  3. initial band: IDPP (default) / linear interpolation, or a seed band
     (any multi-frame file spanning reactant -> product, subsampled), as the
     DA study found seeding essential when IDPP gives clashing images
  4. NEB optimization; every step written to neb.traj
  5. per image: energy, relative energy, max force, CVs, short-contact flag

The highest-energy *interior* image is written as ``ts_candidate.xyz``. It is
only a starting guess for refine_ts.py — a CI-NEB maximum is NOT a verified
transition state (the DA climbing image carried a second imaginary mode).

Outputs (path/): images/image_XX.xyz, band.xyz, neb.traj, neb.log, path.csv,
path.json, ts_candidate.xyz

    python run_path.py --config config.json [--model mace-polar] [--images 13]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from ase.io import read, write  # noqa: E402

from calculators import add_calc_args, model_from_config, parse_calc_settings  # noqa: E402
from config_io import (fixed_atoms, load_config, output_root, resolve_path,  # noqa: E402
                       section, write_json)
from geometry import kabsch_align, kabsch_rmsd, max_force, uses_pbc  # noqa: E402
from validate_structure import (Report, ValidationError, _check_distances,  # noqa: E402
                                print_report, require_valid, validate_pair, vcfg)
from workflow import (EV2KCAL, apply_fixed, banner, cv_values, provenance,  # noqa: E402
                      stage_dir, strip_calc)

DEFAULTS = {"n_images": 13, "k": 0.5, "climb": True, "method": "improvedtangent",
            "interpolation": "idpp", "optimizer": "FIRE", "fmax": 0.05, "steps": 800,
            "align_product": True}


def _subsample(band, k):
    """k geometries evenly from a list of Atoms (same as neb.py::_subsample)."""
    idx = np.linspace(0, len(band) - 1, k).round().astype(int)
    return [band[i].copy() for i in idx]


def build_band(R, P, opts, seed_band=None):
    n = int(opts["n_images"])
    if n < 3:
        raise ValueError("n_images must be >= 3 (two endpoints + at least one image)")
    if seed_band is not None:
        if any(len(s) != len(R) or s.get_chemical_symbols() != R.get_chemical_symbols()
               for s in seed_band):
            raise ValueError("seed band frames do not match the endpoints' atoms/order")
        images = [R.copy()] + _subsample(seed_band, n)[1:-1] + [P.copy()]
        return images, "seed_band"
    images = [R.copy()] + [R.copy() for _ in range(n - 2)] + [P.copy()]
    return images, opts["interpolation"]


def run(cfg, model=None, variant=None, device=None, dtype=None, reactant=None,
        product=None, seed_band=None, out=None, handle=None, n_images=None,
        allow_unoptimized=False) -> dict:
    from ase.mep import NEB

    banner(cfg)
    opts = {**DEFAULTS, **section(cfg, "path")}
    if n_images:
        opts["n_images"] = int(n_images)

    # model tag only (no weights loaded) — validation must come before the load
    tag = (handle.settings if handle else
           parse_calc_settings(cfg, model, variant, device, dtype)).tag
    ep = output_root(cfg, tag) / "endpoints"
    if reactant is None or product is None:
        if (ep / "reactant_opt.xyz").exists() and (ep / "product_opt.xyz").exists():
            reactant = reactant or ep / "reactant_opt.xyz"
            product = product or ep / "product_opt.xyz"
        elif allow_unoptimized:
            st = cfg.get("structures") or {}
            reactant = reactant or resolve_path(cfg, st["reactant"])
            product = product or resolve_path(cfg, st["product"])
            print("WARNING: using UNOPTIMIZED endpoints (--allow-unoptimized)")
        else:
            raise FileNotFoundError(f"optimized endpoints not found in {ep}; run "
                                    "optimize_endpoints.py first (or pass --allow-unoptimized)")

    report = validate_pair(Path(reactant), Path(product), cfg)
    require_valid(report, "path endpoints")
    print_report(report)
    R, P = read(str(reactant)), read(str(product))
    handle = handle or model_from_config(cfg, model, variant, device, dtype)
    handle.check_elements(R)
    out = stage_dir(cfg, tag, "path", out)

    fixed = fixed_atoms(cfg, len(R))
    aligned_rmsd = None
    if opts["align_product"] and not fixed and not uses_pbc(R):
        P = kabsch_align(P, R)
        aligned_rmsd = kabsch_rmsd(P.positions, R.positions)

    seed = read(str(seed_band), index=":") if seed_band else None
    images, init_method = build_band(R, P, opts, seed)
    for im in images:
        apply_fixed(im, cfg)
        im.calc = handle.new_calc()
    neb = NEB(images, climb=bool(opts["climb"]), k=float(opts["k"]), method=opts["method"],
              allow_shared_calculator=False)
    if seed is None:
        neb.interpolate(method=opts["interpolation"], mic=uses_pbc(R))

    # initial band sanity: interpolation through clashing geometries is the
    # classic failure (DA study); fail before burning hours on it.
    clash = []
    for i, im in enumerate(images[1:-1], 1):
        rep = Report(f"image{i}")
        _check_distances(strip_calc(im), cfg, vcfg(cfg), rep)
        if any("duplicate" in e or "impossibly short" in e for e in rep.errors):
            clash.append({"image": i, "errors": rep.errors[:5]})
    if clash:
        write_json(out / "initial_band_clashes.json", clash)
        for im_i, im in enumerate(images):
            write(str(out / f"initial_image_{im_i:02d}.xyz"), strip_calc(im))
        raise ValidationError(f"initial band has clashing atoms in images "
                              f"{[c['image'] for c in clash]} ({init_method}); see "
                              f"{out}/initial_band_clashes.json — use a seed band")
    write(str(out / "initial_band.xyz"), [strip_calc(im) for im in images])

    if opts["optimizer"] == "NEBOptimizer":
        from ase.mep.neb import NEBOptimizer

        opt = NEBOptimizer(neb, logfile=str(out / "neb.log"), trajectory=str(out / "neb.traj"))
    elif opts["optimizer"] == "FIRE":
        from ase.optimize import FIRE

        opt = FIRE(neb, logfile=str(out / "neb.log"), trajectory=str(out / "neb.traj"))
    else:
        raise ValueError("path.optimizer must be FIRE or NEBOptimizer")
    calls0 = handle.total_calls
    print(f"== NEB with {tag}: {opts['n_images']} images, k={opts['k']}, climb={opts['climb']}, "
          f"init={init_method}, fmax={opts['fmax']} ==", flush=True)
    converged = bool(opt.run(fmax=float(opts["fmax"]), steps=int(opts["steps"])))
    nsteps = int(opt.get_number_of_steps())
    neb_fmax = float(np.sqrt((neb.get_forces() ** 2).sum(axis=1).max()))

    E = np.array([im.get_potential_energy() for im in images])
    rel = E - E[0]
    rows, geom_err = [], {}
    imgdir = out / "images"
    imgdir.mkdir(exist_ok=True)
    for i, im in enumerate(images):
        cvs = cv_values(cfg, im)
        rep = Report(f"image{i}")
        _check_distances(strip_calc(im), cfg, vcfg(cfg), rep)
        geom_err[i] = rep.errors
        rows.append({"image": i, "energy_eV": float(E[i]), "rel_eV": float(rel[i]),
                     "rel_kcal_mol": float(rel[i] * EV2KCAL),
                     "fmax_true_eV_A": max_force(im.get_forces()),
                     "n_geometry_errors": len(rep.errors), **cvs})
        c = strip_calc(im)
        c.info.update({"image": i, "energy_eV": float(E[i]), "rel_eV": float(rel[i])})
        write(str(imgdir / f"image_{i:02d}.xyz"), c)
    write(str(out / "band.xyz"), [read(str(imgdir / f"image_{i:02d}.xyz"))
                                  for i in range(len(images))])

    interior = list(range(1, len(images) - 1))
    ts_i = interior[int(np.argmax(E[1:-1]))]
    for r in rows:
        r["ts_candidate"] = int(r["image"] == ts_i)
    with open(out / "path.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tsc = strip_calc(images[ts_i])
    tsc.info.update({"image": ts_i, "energy_eV": float(E[ts_i]),
                     "note": "highest NEB image; NOT a verified TS"})
    write(str(out / "ts_candidate.xyz"), tsc)

    warnings = []
    if E[ts_i] <= max(E[0], E[-1]):
        warnings.append("no interior maximum above both endpoints: the band shows no barrier")
    if not converged:
        warnings.append(f"NEB not converged in {nsteps} steps (fmax {neb_fmax:.3f} > {opts['fmax']})")
    for i, errs in geom_err.items():
        if errs:
            warnings.append(f"image {i} fails geometry checks after NEB: {'; '.join(errs[:3])}")

    summary = {
        **provenance(cfg, handle, stage="run_path"),
        "endpoints": {"reactant": str(reactant), "product": str(product)},
        "settings": {**opts, "initialization": init_method,
                     "seed_band": str(seed_band) if seed_band else None},
        "product_aligned_to_reactant": aligned_rmsd is not None,
        "aligned_endpoint_rmsd_A": aligned_rmsd,
        "fixed_atoms": fixed,
        "converged": converged, "steps": nsteps, "neb_fmax_final": neb_fmax,
        "calculator_calls": handle.total_calls - calls0,
        "images": rows,
        "ts_candidate": {"image": ts_i, "rel_eV": float(rel[ts_i]),
                         "rel_kcal_mol": float(rel[ts_i] * EV2KCAL),
                         "file": str(out / "ts_candidate.xyz"),
                         "status": "highest-energy NEB image; NOT a verified transition "
                                   "state — refine with refine_ts.py, verify with validate_ts.py"},
        "reaction_energy_band": {"eV": float(rel[-1]), "kcal_mol": float(rel[-1] * EV2KCAL)},
        "warnings": warnings,
    }
    write_json(out / "path.json", summary)
    print(f"  converged={converged} steps={nsteps} NEB fmax={neb_fmax:.4f}")
    for r in rows:
        star = "  <- highest image (candidate only)" if r["ts_candidate"] else ""
        print(f"  img {r['image']:2d}  ΔE={r['rel_kcal_mol']:9.2f} kcal/mol  "
              + "  ".join(f"{k}={r[k]:.3f}" for k in cvs) + star)
    for wmsg in warnings:
        print(f"  WARNING: {wmsg}")
    print(f"  wrote {out}/path.json, path.csv, band.xyz, neb.traj, ts_candidate.xyz")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CI-NEB reaction path between endpoints")
    add_calc_args(ap)
    ap.add_argument("--reactant", default=None)
    ap.add_argument("--product", default=None)
    ap.add_argument("--seed-band", default=None, help="multi-frame file to seed the band")
    ap.add_argument("--images", type=int, default=None)
    ap.add_argument("--allow-unoptimized", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    try:
        s = run(cfg, a.model, a.variant, a.device, a.dtype, a.reactant, a.product,
                a.seed_band, a.out, n_images=a.images, allow_unoptimized=a.allow_unoptimized)
    except (ValidationError, FileNotFoundError) as e:
        print(f"ABORTED: {e}")
        return 1
    return 0 if s["converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
