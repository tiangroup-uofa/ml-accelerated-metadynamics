#!/usr/bin/env python3
"""
Compare MACE-OMOL / MACE-POLAR against a directory of Gaussian reference jobs, and
(optionally) run the sparse DFT-budget experiment.

    python run_dft_comparison.py --dir /path/to/gaussian_dir --models mace-omol mace-polar
    python run_dft_comparison.py --dir ... --budget          # only after the full set was analysed

The directory needs manifest.json (see dft_reference/dft_compare.py). Nothing is
read from a default location: with no Gaussian data this script refuses to run.
A manifest with "synthetic": true is refused unless --allow-synthetic is given,
and every output is then stamped SYNTHETIC, so test fixtures cannot be mistaken
for results.

Charge/multiplicity come from the config and are checked against both the manifest
and every Gaussian job (a mismatch aborts). The MACE side is evaluated at the DFT
geometries (`dft_geometry_single_point` reference state).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calculators as C                                    # noqa: E402
from ase import Atoms                                      # noqa: E402
from config_io import atom_map, charge_and_multiplicity, load_config, write_json  # noqa: E402
from dft_reference import dft_compare as D                 # noqa: E402
from dft_reference import gaussian_parser as G            # noqa: E402
from config_io import resolve_path                         # noqa: E402
from ase.io import read                                    # noqa: E402
from dft_reference import sparse_budget as S               # noqa: E402


def subsets_from_cfg(cfg, atoms):
    m = atom_map(cfg, len(atoms))
    core = [m[k] for k in ("C1", "C2", "H_transfer", "O1", "O2") if k in m]
    sn = m.get("Sn")
    out = {"core": core}
    if sn is not None:
        d = np.linalg.norm(atoms.positions - atoms.positions[sn], axis=1)
        out["sn_centered"] = [int(i) for i in np.where(d <= 3.0)[0]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--config", default="config_tian_tt_test.json")
    ap.add_argument("--models", nargs="+", default=["mace-omol"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    q, mult = charge_and_multiplicity(cfg)
    records, man, jobs = D.load_reference_set(a.dir, a.manifest, expect_charge=q, expect_multiplicity=mult)
    synthetic = bool(man.get("synthetic")) or "synthetic" in str(man.get("basis_note", "")).lower()
    if synthetic and not a.allow_synthetic:
        sys.exit("manifest is marked synthetic; refusing to produce results (use --allow-synthetic for tests only)")
    stamp = "SYNTHETIC-NOT-A-RESULT" if synthetic else "gaussian-reference"
    out = Path(a.out or Path(a.dir) / "mace_comparison"); out.mkdir(parents=True, exist_ok=True)
    for j in jobs:
        print(G.describe_job(j))
    numbers = records[0].numbers
    proj = read(resolve_path(cfg, cfg["structures"]["reactant"]))
    if not np.array_equal(proj.numbers, numbers):
        sys.exit("Gaussian atom ordering/elements differ from the project reactant structure; atom map would be wrong")
    ref_atoms = Atoms(numbers=numbers, positions=records[0].positions)
    subs = subsets_from_cfg(cfg, ref_atoms)

    print(f"cost gate: {len(records)} records x {len(a.models)} model(s) = {len(records) * len(a.models)} "
          f"single-point force calls (~0.5 s each for MACE-OMOL/POLAR float64 CPU on 42 atoms => "
          f"~{len(records) * len(a.models) * 0.5 / 60:.1f} min)", flush=True)
    tables, summaries, evaluated_by_model = {}, {}, {}
    for model in a.models:
        handle = C.model_from_config(cfg, model=model)
        calc = handle.new_calc()

        def evaluate(nums, pos, calc=calc):
            at = Atoms(numbers=nums, positions=pos); at.calc = calc
            return at.get_potential_energy(), at.get_forces()

        t0 = time.time()
        ev = D.evaluate_records(records, evaluate)
        rows, summ = D.analyse(records, ev, subsets=subs)
        summ.update({"model": model, "stamp": stamp, "charge": q, "multiplicity": mult,
                     "wall_s": time.time() - t0, "force_calls": handle.total_calls,
                     "basis_note": man.get("basis_note")})
        tables[model], summaries[model], evaluated_by_model[model] = rows, summ, ev
        write_json(out / f"{model}_summary.json", summ)
        print(model, json.dumps({k: summ[k] for k in ("alignment", "energy", "force_cos_mean")}, default=float))
    D.plot_path_diagnostics(tables, out / "path_diagnostics.png", title=f"DFT vs MACE [{stamp}]")

    if a.budget:
        for model in a.models:
            fr = D.finals(records)
            ev = evaluated_by_model[model]
            center = S.mace_max_energy_location(fr, {r.rid: ev[r.rid][0] for r in records})
            n_int = len(fr) - 2
            subsets = [S.endpoints_only(records), S.uniform(records, min(4, len(fr)))]
            subsets += [S.ts_enriched(records, k, center) for k in range(1, min(n_int, 10) + 1)]
            res = S.run_budget_experiment(records, ev, subsets, kinds=("none", "global", "affine", "element"))
            for r in res:
                r["stamp"] = stamp
            write_json(out / f"{model}_budget.json", res)
            S.plot_budget(res, str(out / f"{model}_budget"))
            print(model, "budget experiment:", len(res), "rows")


if __name__ == "__main__":
    main()
