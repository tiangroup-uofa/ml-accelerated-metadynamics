"""
cross_evaluate.py: labelling, arithmetic and provenance. No model calls, no downloads.

The committed audit/omol_benchmark/cross_evaluation.json is the stored result of the 20 single-point
cross-evaluations; these tests recompute everything derived from its stored energies, check that the
structure hashes still match the committed structure files, and pin the report's prose to the data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PKG)]

import cross_evaluate as X  # noqa: E402

AUD = PKG / "audit" / "omol_benchmark"
STORED = json.loads((AUD / "cross_evaluation.json").read_text())
REPORT = (PKG / "OMOL_BENCHMARK_REPORT.md").read_text(encoding="utf-8")
EV = X.EV2KJ


def synthetic():
    base = {"R": -10.0, "TS": -9.0, "P": -10.3, "R'": -10.15, "P'": -10.28}
    E = {}
    for m, off in (("omol", 0.0), ("polar", 0.5)):
        for g, shift in (("omol", 0.0), ("polar", 0.1)):
            E[X.key(m, g)] = {k: v + off + shift for k, v in base.items()}
    return E


def test_labels_are_explicit_single_point_labels_not_barriers():
    assert X.key("omol", "polar") == "omol_at_polar_geometries" and X.key("polar", "omol") == "polar_at_omol_geometries"
    assert set(STORED["labels"]) == {"omol_at_omol_geometries", "omol_at_polar_geometries",
                                     "polar_at_polar_geometries", "polar_at_omol_geometries"}
    assert "NOT relaxed barriers" in STORED["not_relaxed"]
    keys = {k.lower() for k in json.dumps(STORED).replace('"', " ").replace(":", " ").split()}
    assert "barrier" not in set(STORED) and "barrier" not in set(STORED["single_point_differences_kJ_mol"])


def test_arithmetic_on_synthetic_energies():
    t = X.compute_tables(synthetic())
    d = t["single_point_differences_kJ_mol"]["omol_at_omol_geometries"]
    assert d["TS-R"] == pytest.approx(1.0 * EV) and d["P-R"] == pytest.approx(-0.3 * EV)
    # a uniform 0.1 eV geometry shift is a penalty of 0.1 eV for every structure, for both models
    for k in ("omol_at_polar_geometries", "polar_at_omol_geometries"):
        pen = t["geometry_penalty"][k]
        assert pen["min_kJ_mol"] == pytest.approx(pen["max_kJ_mol"])
    assert t["geometry_penalty"]["omol_at_polar_geometries"]["min_kJ_mol"] == pytest.approx(0.1 * EV)
    assert t["geometry_penalty"]["polar_at_omol_geometries"]["min_kJ_mol"] == pytest.approx(-0.1 * EV)
    # identical single-point differences => zero gap
    assert t["omol_below_polar_kJ_mol"]["max"] == pytest.approx(0.0, abs=1e-9)


def test_stored_derived_tables_match_a_recomputation_from_stored_energies():
    again = X.compute_tables(STORED["energies_eV"])
    for k in again:
        assert again[k] == STORED[k]


def test_structure_provenance_matches_the_committed_files():
    for model, files in STORED["structures"].items():
        assert set(files) == set(X.LABELS)
        for lab, rec in files.items():
            f = PKG / rec["path"]
            assert f.exists() and X.sha256(f) == rec["sha256"], (model, lab)
    # the run directories the script reads are the committed audit copies, not outputs/
    assert all("audit/omol_benchmark" in r["path"] for fs in STORED["structures"].values() for r in fs.values())


def test_own_geometry_energies_equal_the_stored_protocol_energies():
    for model in ("omol", "polar"):
        own = X.own_stored_energies(X.RUN_DIRS[model])
        got = STORED["energies_eV"][X.key(model, model)]
        assert all(abs(got[l] - own[l]) < 1e-6 for l in X.LABELS)


def test_report_statements_follow_the_stored_data():
    pen = STORED["geometry_penalty"]
    lo, hi = pen["omol_at_polar_geometries"]["min_kJ_mol"], pen["omol_at_polar_geometries"]["max_kJ_mol"]
    lo2, hi2 = pen["polar_at_omol_geometries"]["min_kJ_mol"], pen["polar_at_omol_geometries"]["max_kJ_mol"]
    assert (round(lo), round(hi), round(lo2), round(hi2)) == (6, 20, 11, 23)
    assert "16–20" not in REPORT
    assert f"+{round(lo)}–{round(hi)}" in REPORT and f"+{round(lo2)}–{round(hi2)}" in REPORT
    assert f"{round(min(lo, lo2))}–{round(max(hi, hi2))} kJ/mol" in REPORT          # overall 6–23
    # 4M barrier shift from the UNROUNDED stored values
    om = json.loads((AUD / "comparison_omol_vs_polar.json").read_text())["energetics_kJ_mol"]["OMOL"]["barrier_fwd"]
    m4 = json.loads((AUD / "omol_4M_ladder" / "vib" / "validate_ts.json").read_text())["energetics"]["barrier_forward"]
    shift = m4["relaxed_endpoint"]["kJ_mol"] - om["relaxed_endpoint"]
    assert 0.855 < shift < 0.865
    assert f"+{shift:.2f}" in REPORT and "+0.85" not in REPORT
