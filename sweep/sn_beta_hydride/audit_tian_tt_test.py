#!/usr/bin/env python3
"""
audit_tian_tt_test.py
=====================
Regenerates every committed audit artifact for the RECONSTRUCTED tian_tt_test
input set, without running any calculator (seconds, numpy/ASE/matplotlib only):

  audit/tian_tt_test/input_validation.json   reactant/product pair validation
  audit/tian_tt_test/structure_validation.json  single-structure validation of
                                             Tian's POLAR TS and side minima
  audit/tian_tt_test/tian_polar_summary.json both-reference-state energetics
                                             from saved raw energies
  audit/tian_tt_test/structure_views.png     x/y/z views of R, TS, P (common
                                             Sn-centred frame; qualitative)

    python audit_tian_tt_test.py [--config config_tian_tt_test.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ase.io import read  # noqa: E402

import summarize_tian_polar as STP  # noqa: E402
from config_io import load_config, resolve_path, write_json  # noqa: E402
from validate_structure import print_report, validate_pair, validate_single  # noqa: E402
from visualize_structures import draw  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GLU = REPO / "glucose" / "structures"


def relativize(obj):
    """Committed artifacts carry repo-relative POSIX paths, not machine paths."""
    if isinstance(obj, dict):
        return {k: relativize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relativize(v) for v in obj]
    if isinstance(obj, str):
        for root in {str(REPO), str(REPO).replace("\\", "/")}:
            if obj.startswith(root):
                return obj[len(root):].lstrip("\\/").replace("\\", "/")
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit artifacts for the tian_tt_test input set")
    ap.add_argument("--config", default=str(HERE / "config_tian_tt_test.json"))
    ap.add_argument("--out", default=str(HERE / "audit" / "tian_tt_test"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = Path(a.out)
    st = cfg["structures"]
    rpath, ppath = resolve_path(cfg, st["reactant"]), resolve_path(cfg, st["product"])

    pair = validate_pair(rpath, ppath, cfg)
    print_report(pair)
    write_json(out / "input_validation.json", relativize(pair))

    singles = {}
    for label, f, role in (("tian_polar_ts", f"ts_{STP.TAG}.xyz", None),
                           ("tian_polar_reactant_side_min", f"ts_reactant_side_{STP.TAG}.xyz",
                            "reactant"),
                           ("tian_polar_product_side_min", f"ts_product_side_{STP.TAG}.xyz",
                            "product")):
        singles[label] = validate_single(GLU / f, cfg, label, role)
        print(f"[{singles[label]['status']}] {label}: hydride "
              f"{singles[label]['info'].get('hydride_state', {}).get('state')}")
    write_json(out / "structure_validation.json", relativize(singles))

    summary = STP.summarize(cfg)
    write_json(out / "tian_polar_summary.json", relativize(summary))

    fig = draw([("Reactant (Tian, POLAR-relaxed input)", read(str(rpath))),
                ("TS (Tian, POLAR Sella saddle; not independently certified)",
                 read(str(GLU / f"ts_{STP.TAG}.xyz"))),
                ("Product (Tian, POLAR-relaxed input)", read(str(ppath)))],
               cfg, out / "structure_views.png",
               "tian_tt_test RECONSTRUCTED cluster, MACE-POLAR-1-S geometries; common "
               "Sn-centred frame (Å); qualitative only. Green = H* and its C bonds; "
               "dashed = Sn-O > 2.2 Å")
    print(f"pair: {pair['status']} ({pair['n_errors']} errors, {pair['n_warnings']} warnings); "
          f"summary + {fig.name} -> {out}")
    return 1 if pair["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
