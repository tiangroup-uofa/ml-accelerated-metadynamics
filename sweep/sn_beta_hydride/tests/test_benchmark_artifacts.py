"""
Guards on the OMOL benchmark artifacts and model-size/training-subset configuration.
Artifact tests read outputs/ if it exists locally (outputs are not committed) and
SKIP otherwise; they never run a model.

    cd sweep/sn_beta_hydride && python -m pytest tests/test_benchmark_artifacts.py -q
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PKG)]

import calculators as C                       # noqa: E402
from config_io import load_config             # noqa: E402

OUT = PKG / "outputs" / "tian_tt_test"
REF_STATES = {"relaxed_endpoint", "connected_side_minimum"}


def _summ(name):
    p = OUT / name / "protocol_summary.json"
    if not p.exists():
        pytest.skip(f"{p} not present")
    return json.loads(p.read_text())


@pytest.mark.parametrize("run", ["mace-omol_extra_large"])
def test_every_energetic_quantity_names_its_reference_state(run):
    e = _summ(run)["energetics"]
    for q in ("barrier_forward", "barrier_reverse", "reaction_energy"):
        assert set(e[q]) <= REF_STATES and e[q], q
    assert not {"barrier", "barrier_kJ_mol", "barrier_eV"} & set(e)           # no bare barrier anywhere at top level
    assert set(e["reference_states"]) == REF_STATES
    assert "not free energies" in e["units_note"]


def test_neb_maximum_is_not_a_barrier_or_certified():
    p = OUT / "mace-omol_extra_large" / "path" / "path.json"
    if not p.exists():
        pytest.skip("path.json not present")
    d = json.loads(p.read_text())
    tc = d["ts_candidate"]
    assert tc["certified"] is False and "NOT a verified transition state" in tc["status"]
    assert "barrier" not in {k.lower() for k in d} | {k.lower() for k in tc}
    assert tc["E_highest_image_minus_E_band_image0"]["reference_state"]       # even the candidate gap names its reference
    assert d["converged"] is False or d["neb_fmax_final"] <= d["settings"]["fmax"]   # unconverged NEB must say so
    if d["converged"] is False:
        assert any("not converged" in w for w in d["warnings"])


def test_certified_label_requires_a_converged_sella_in_the_record():
    """Known pipeline gap (run_protocol.py:120-136): CERTIFIED can be emitted while Sella is
    unconverged. The saved OMOL run must not be in that state."""
    s = _summ("mace-omol_extra_large")
    if str(s["ts_certification"]).startswith("CERTIFIED"):
        assert s["stages"]["refine_ts"]["converged"] is True


def test_omol_4m_config_is_isolated_and_differs_only_where_declared():
    a, b = load_config(PKG / "config_tian_tt_test.json"), load_config(PKG / "config_omol4m.json")
    assert a["output_dir"] != b["output_dir"]                                  # cannot overwrite the 1024 run
    ma, mb = a["calculator"]["models"]["mace-omol"], b["calculator"]["models"]["mace-omol"]
    assert "model_path" not in ma and mb["model_path"].endswith("mace-omol-0-extra-large-4M.model")
    strip = lambda c: {k: v for k, v in c.items() if k not in ("output_dir", "calculator", "_comment", "_config_path")}
    assert strip(a) == strip(b)                                                # same structures/charge/spin/protocol
    assert a["system"] == b["system"]
    # the calculator block may differ ONLY in the OMOL checkpoint path and its note (dtype/device/threads/other models identical)
    ca, cb = copy.deepcopy(a["calculator"]), copy.deepcopy(b["calculator"])
    cb["models"]["mace-omol"].pop("model_path"); cb["models"]["mace-omol"].pop("_note", None)
    ca["models"]["mace-omol"].pop("_note", None)
    assert ca == cb


def test_model_path_settings_are_parsed_and_unknown_variant_still_rejected(tmp_path):
    cfg = load_config(PKG / "config_omol4m.json")
    fake = tmp_path / "x.model"; fake.write_text("x")
    cfg["calculator"]["models"]["mace-omol"]["model_path"] = str(fake)
    s = C.parse_calc_settings(cfg, model="mace-omol")
    assert s.model_path == str(fake)
    cfg2 = load_config(PKG / "config_tian_tt_test.json")
    cfg2["calculator"]["models"]["mace-omol"]["variant"] = "small"            # no such OMOL size installed
    with pytest.raises(Exception):
        C.parse_calc_settings(cfg2, model="mace-omol")
