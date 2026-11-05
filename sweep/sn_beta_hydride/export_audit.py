#!/usr/bin/env python3
"""
Copy the SMALL, curated results of the OMOL/POLAR/xTB benchmark from the (git-ignored) outputs/
tree into the tracked audit/omol_benchmark/ directory, like audit/tian_tt_test/.

    python export_audit.py

Copies only: protocol/stage JSONs, frequencies, the saddle + endpoint + side-minimum xyz files, the
comparison/reassessment/xTB JSONs. NEVER trajectories (*.traj), Hessians, normal-mode arrays or logs.
Absolute paths inside the JSON/xyz text are rewritten (repo root -> "<repo>", home -> "~") so no
username enters version control. Numbers are not touched.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "outputs" / "tian_tt_test"
DEST = HERE / "audit" / "omol_benchmark"
RUNS = {
    "omol_1024_full_protocol": OUT / "mace-omol_extra_large",
    "omol_4M_ladder": HERE / "outputs" / "tian_tt_test_omol4m" / "mace-omol_extra_large",
    "polar_ladder_ours": OUT / "mace-polar_polar-1-s",
}
RUN_FILES = ["protocol_summary.json", "endpoints/endpoints.json", "path/path.json", "ts/refine_ts.json",
             "ts/ts_sella.xyz", "vib/validate_ts.json", "vib/frequencies.csv", "endpoints/reactant_opt.xyz",
             "endpoints/product_opt.xyz", "vib/connectivity/conn_plus_relaxed.xyz",
             "vib/connectivity/conn_minus_relaxed.xyz"]
TOP_FILES = [OUT / "comparison_omol_vs_polar.json", OUT / "reassessment_corrected_rules.json",
             OUT / "xtb_sanity" / "xtb_sanity.json", OUT / "xtb_sanity" / "reactant_xtb_relaxed.xyz",
             OUT / "xtb_sanity" / "product_xtb_relaxed.xyz",
             OUT / "phase3_seed_from_polar_ts" / "refine_ts.json"]
FORBIDDEN = (".traj", ".npy", ".npz", ".log")


def clean(text: str) -> str:
    return text.replace(str(REPO), "<repo>").replace(str(Path.home()), "~")


def copy(src: Path, dst: Path):
    assert src.suffix not in FORBIDDEN, src
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(clean(src.read_text()))


def main():
    n = 0
    for tag, root in RUNS.items():
        for rel in RUN_FILES:
            if (root / rel).exists():
                copy(root / rel, DEST / tag / rel); n += 1
    for f in TOP_FILES:
        if f.exists():
            copy(f, DEST / (f.parent.name + "__" + f.name if f.parent.name not in ("tian_tt_test",) else f.name)); n += 1
    ck = Path.home() / ".cache" / "mace"
    manifest = {
        "what": "curated small results of the OMOL/POLAR/xTB benchmark on the reconstructed Sn-beta cluster",
        "not_included": "trajectories, Hessians, normal modes, logs (regenerable; see OMOL_BENCHMARK_REPORT.md)",
        "paths": "absolute paths rewritten to <repo> and ~",
        "checkpoint_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (ck / "MACE-omol-0-extra-large-1024.model", ck / "mace-omol-0-extra-large-4M.model",
                                        ck / "MACEPOLAR1Smodel") if p.exists()},
        "files": n,
    }
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"exported {n} files to {DEST}")


if __name__ == "__main__":
    main()
