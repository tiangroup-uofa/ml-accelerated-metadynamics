#!/usr/bin/env python3
"""
cross_evaluate.py — same-geometry cross-evaluation of MACE-OMOL and MACE-POLAR.

    cd sweep/sn_beta_hydride
    python cross_evaluate.py            # writes audit/omol_benchmark/cross_evaluation.json
    python cross_evaluate.py --check    # recompute the derived tables from the stored energies; no model calls

Question answered: does the OMOL-vs-POLAR barrier disagreement persist when BOTH models are
evaluated at IDENTICAL geometries? For each of the five saved structures

    R   relaxed reactant       P   relaxed product       TS   Sella saddle
    R'  TS-connected reactant-side minimum               P'   TS-connected product-side minimum

of each method (committed in audit/omol_benchmark/), both models compute a SINGLE-POINT energy
at both methods' structures. 20 single points in total, about 10 s on CPU (float64, charge and
spin set by the project's ChargeSpinCalculator from config_tian_tt_test.json, as in every other run).

Labels (every energy and difference carries one; none is a relaxed barrier):
    omol_at_omol_geometries   omol_at_polar_geometries
    polar_at_polar_geometries polar_at_omol_geometries
The quantities are SINGLE-POINT energy differences at fixed geometries: TS-R, TS-R', P-R.
They are NOT relaxed barriers and are not comparable to the relaxed_endpoint /
connected_side_minimum reference states of energetics.py, which relax each method on its own surface.

Provenance recorded in the output: SHA-256 of every structure file read, the calculator info,
and a check that each model reproduces its own stored protocol energies at its own geometries
(to 1e-6 eV), so the cross-evaluation provably uses the same structures and settings as the runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from energetics import EV2KJ  # noqa: E402

AUDIT = HERE / "audit" / "omol_benchmark"
OUT_DEFAULT = AUDIT / "cross_evaluation.json"
RUN_DIRS = {"omol": AUDIT / "omol_1024_full_protocol", "polar": AUDIT / "polar_ladder_ours"}
MODEL_KEY = {"omol": "mace-omol", "polar": "mace-polar"}
LABELS = ("R", "TS", "P", "R'", "P'")
DIFFS = {"TS-R": ("TS", "R"), "TS-R'": ("TS", "R'"), "P-R": ("P", "R")}
NOT_RELAXED = ("single-point energy differences at fixed, saved geometries; NOT relaxed barriers and not "
               "comparable to the relaxed_endpoint / connected_side_minimum reference states")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def structure_files(run: Path) -> dict:
    """label -> xyz path for one method's committed audit structures. The reactant/product side of the
    +/- mode relaxation is taken from the saved connectivity record, not assumed."""
    vt = json.loads((run / "vib" / "validate_ts.json").read_text())
    conn = vt["connectivity"]
    side = lambda s: run / "vib" / "connectivity" / ("conn_plus_relaxed.xyz" if s == "+" else "conn_minus_relaxed.xyz")
    return {"R": run / "endpoints" / "reactant_opt.xyz", "P": run / "endpoints" / "product_opt.xyz",
            "TS": run / "ts" / "ts_sella.xyz", "R'": side(conn["reactant_side"]), "P'": side(conn["product_side"])}


def own_stored_energies(run: Path) -> dict:
    """The energies the method's own protocol stored for these structures (eV)."""
    vt = json.loads((run / "vib" / "validate_ts.json").read_text())
    rs = vt["energetics"]["reference_states"]
    return {"R": rs["relaxed_endpoint"]["reactant_eV"], "P": rs["relaxed_endpoint"]["product_eV"],
            "TS": vt["energetics"]["ts"]["energy_eV"], "R'": rs["connected_side_minimum"]["reactant_eV"],
            "P'": rs["connected_side_minimum"]["product_eV"]}


def key(model: str, geom: str) -> str:
    return f"{model}_at_{geom}_geometries"


def compute_tables(E: dict) -> dict:
    """Pure arithmetic on single-point energies. E[key(model, geom)][label] in eV. No model calls."""
    kj = lambda a, b: (a - b) * EV2KJ
    diffs = {k: {n: kj(v[a], v[b]) for n, (a, b) in DIFFS.items()} for k, v in E.items()}
    penalty = {}
    for m, other in (("omol", "polar"), ("polar", "omol")):
        pen = {l: kj(E[key(m, other)][l], E[key(m, m)][l]) for l in LABELS}
        penalty[key(m, other)] = {"per_structure_kJ_mol": pen, "min_kJ_mol": min(pen.values()),
                                  "max_kJ_mol": max(pen.values()),
                                  "meaning": f"E_{m}(other method's structure) - E_{m}(own structure)"}
    gap = {}
    for g in ("polar", "omol"):
        gap[f"on_{g}_geometries"] = {n: diffs[key("polar", g)][n] - diffs[key("omol", g)][n] for n in ("TS-R", "TS-R'")}
    allgaps = [v for d in gap.values() for v in d.values()]
    return {"single_point_differences_kJ_mol": diffs, "geometry_penalty": penalty,
            "omol_below_polar_kJ_mol": {**gap, "min": min(allgaps), "max": max(allgaps),
                                        "meaning": "POLAR - OMOL for the same single-point difference on the same geometry set"}}


def evaluate(cfg_path: str) -> tuple[dict, dict]:
    import numpy as np
    from ase.io import read

    import calculators as C
    from config_io import load_config

    cfg = load_config(cfg_path)
    files = {m: structure_files(RUN_DIRS[m]) for m in RUN_DIRS}
    E, info = {}, {}
    for model in ("omol", "polar"):
        handle = C.model_from_config(cfg, model=MODEL_KEY[model])
        info[model] = handle.info()
        for geom in ("omol", "polar"):
            e = {}
            for lab, f in files[geom].items():
                at = read(str(f))
                at.calc = handle.new_calc()
                e[lab] = float(at.get_potential_energy())
            E[key(model, geom)] = e
        own = own_stored_energies(RUN_DIRS[model])
        bad = {l: E[key(model, model)][l] - own[l] for l in LABELS if abs(E[key(model, model)][l] - own[l]) > 1e-6}
        if bad:
            raise RuntimeError(f"{model} does not reproduce its own stored energies at its own geometries: {bad}")
    return E, info


def build(E: dict, info: dict | None) -> dict:
    files = {m: structure_files(RUN_DIRS[m]) for m in RUN_DIRS}
    return {
        "what": "OMOL-vs-POLAR same-geometry cross-evaluation (single points at saved structures)",
        "not_relaxed": NOT_RELAXED,
        "labels": [key(m, g) for m in ("omol", "polar") for g in ("omol", "polar")],
        "energies_eV": E,
        **compute_tables(E),
        "structures": {m: {l: {"path": f.relative_to(HERE).as_posix(), "sha256": sha256(f)} for l, f in fs.items()}
                       for m, fs in files.items()},
        "own_geometry_check": "each model reproduces its stored protocol energies at its own five structures to 1e-6 eV",
        "calculators": info,
        "charge_spin": "from config_tian_tt_test.json via ChargeSpinCalculator (0 / singlet, inferred)",
        "command": "python cross_evaluate.py",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config_tian_tt_test.json"))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--check", action="store_true", help="recompute derived tables from stored energies; no model calls")
    a = ap.parse_args()
    if a.check:
        stored = json.loads(Path(a.out).read_text())
        again = compute_tables(stored["energies_eV"])
        for k in again:
            assert again[k] == stored[k], f"stored {k} does not match recomputation"
        print("OK: derived tables match the stored energies")
        return
    E, info = evaluate(a.config)
    out = build(E, info)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    t = out["geometry_penalty"]
    print(f"wrote {a.out}")
    for k, v in t.items():
        print(f"  {k}: penalty {v['min_kJ_mol']:.1f}..{v['max_kJ_mol']:.1f} kJ/mol")
    g = out["omol_below_polar_kJ_mol"]
    print(f"  OMOL below POLAR by {g['min']:.1f}..{g['max']:.1f} kJ/mol on identical geometries")


if __name__ == "__main__":
    main()
