"""
Regression tests for the corrected TS-certification / side-minimum / NEB-status methodology.

Policy under test (Council checkpoint #2, snapshot 024086aa):
  Q1  CERTIFIED requires a CONVERGED Sella search on record (missing record => not certified),
      plus the existing stationarity / one imaginary mode / hydride overlap / valid +/- minima.
  Q2  A +/- side minimum that is STRUCTURALLY_INVALID relative to the INPUT chemistry (Sn shell,
      bound water, heavy-atom bonds, hydride state) is not a connected minimum: `connects` is
      False and it never becomes a `connected_side_minimum` reference.
  Q3  The label is about the SADDLE. An unconverged NEB is reported separately as DIAGNOSTIC ONLY;
      it never becomes a converged path, a TS or a barrier, and a converged saddle reached from it
      stays certifiable.
  Q4  Both reference states stay explicit; no bare `barrier` key.

Real input structures (the 42-atom cluster) drive the side-minimum tests; only the relaxation step
is stubbed. Nothing here runs a foundation model.

    cd sweep/sn_beta_hydride && python -m pytest tests/test_certification_rules.py -q
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read, write

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path[:0] = [str(PKG), str(HERE)]

import calculators as C                                          # noqa: E402
import connectivity as CN                                        # noqa: E402
import reassess_status as RS                                     # noqa: E402
import run_protocol                                              # noqa: E402
import validate_ts as VT                                         # noqa: E402
from config_io import atom_map, load_config                      # noqa: E402
from energetics import energetics, reference_state               # noqa: E402
from test_protocol_infrastructure import ToyHandle, toy          # noqa: E402,F401  (fixture + helper)
from toy_hydride import DoubleWellHydride, ts_geometry          # noqa: E402

CFG = load_config(PKG / "config_tian_tt_test.json")
M = atom_map(CFG, 42)
R0 = read(PKG / "input/tian_tt_test/reactant.xyz")
P0 = read(PKG / "input/tian_tt_test/product.xyz")
INPUTS = {"reactant": R0, "product": P0}

GOOD_CONN = {"connects": True, "both_converged": True, "verdict": "CONNECTS", "coordination_checked": True}
KW = dict(stationary=True, fmax=1e-4, stationary_fmax=0.01, n_imaginary=1, overlap=0.8,
          min_overlap=0.3, geometry_valid=True, connectivity=GOOD_CONN, sella_converged=True)


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def _bare_barrier_keys(obj):
    return {k for k in _keys(obj) if str(k).lower() in ("barrier", "barrier_ev", "barrier_kj_mol", "barrier_kcal_mol")}


# ------------------------------------------------------------- Q1: Sella convergence
def test_unconverged_sella_cannot_emit_certified():
    ok = CN.ts_certification(**KW)
    assert ok.startswith("CERTIFIED") and "(not an IRC)" in ok
    bad = CN.ts_certification(**{**KW, "sella_converged": False, "fmax": 1e-9})   # even a perfect fmax
    assert bad.startswith("NOT CERTIFIED") and "did NOT converge" in bad
    unknown = CN.ts_certification(**{**KW, "sella_converged": None})
    assert unknown.startswith("NOT CERTIFIED") and "no Sella-convergence record" in unknown


def test_sella_evidence_is_required_not_defaulted():
    kw = {k: v for k, v in KW.items() if k != "sella_converged"}
    with pytest.raises(TypeError, match="sella_converged"):
        CN.ts_certification(**kw)


class UnbiasedToyHandle(ToyHandle):
    """ts_geometry() is the exact saddle of the UNBIASED toy surface (ToyHandle's -0.05 tilt moves it)."""

    def new_calc(self, correction=None):
        c = DoubleWellHydride(bias=0.0)
        self._calcs.append(c)
        return c


def test_validate_ts_reads_sella_flag_from_the_saddle_and_gates_on_it(toy, tmp_path):
    """End to end through validate_ts on the analytic toy: the xyz metadata written by refine_ts decides."""
    labels = {}
    for name, flag in (("conv", True), ("unconv", False), ("none", None)):
        a = ts_geometry()
        if flag is not None:
            a.info["sella_converged"] = flag
        f = tmp_path / f"ts_{name}.xyz"
        write(str(f), a)
        s = VT.run(copy.deepcopy(toy), ts=f, out=tmp_path / f"vib_{name}", handle=UnbiasedToyHandle(toy))
        labels[name] = s["ts_certification"]
        assert s["sella_converged"] is flag
        assert not _bare_barrier_keys(s)
        assert set(s["energetics"]["barrier_forward"]) == {"relaxed_endpoint", "connected_side_minimum"}
    assert labels["conv"].startswith("CERTIFIED")
    assert "Sn coordination NOT checked" in labels["conv"]                                      # toy has no Sn map
    assert labels["unconv"].startswith("NOT CERTIFIED") and "did NOT converge" in labels["unconv"]
    assert labels["none"].startswith("NOT CERTIFIED") and "no Sella-convergence record" in labels["none"]


def test_other_certification_criteria_are_unchanged():
    for change, msg in ((dict(stationary=False, fmax=0.05), "not a stationary point"),
                        (dict(n_imaginary=2), "2 significant"),
                        (dict(overlap=0.1), "not aligned"),
                        (dict(geometry_valid=False), "geometry"),
                        (dict(connectivity=None), "connectivity not checked"),
                        (dict(connectivity={**GOOD_CONN, "connects": False, "verdict": "DOES NOT CONNECT"}), "DOES NOT CONNECT")):
        lab = CN.ts_certification(**{**KW, **change})
        assert lab.startswith("NOT CERTIFIED") and msg in lab, (change, lab)


# ------------------------------------------------- Q2: side-minimum structural validity
def _dissociate_water(atoms):
    a = atoms.copy()
    sn, ow = M["Sn"], M["O_water"]
    v = a.positions[ow] - a.positions[sn]
    water = [ow] + [i for i in range(len(a)) if a.numbers[i] == 1 and np.linalg.norm(a.positions[i] - a.positions[ow]) < 1.2]
    a.positions[water] += 1.5 * v / np.linalg.norm(v)
    return a


def _cleave_si_o(atoms):
    a = atoms.copy()
    sym = a.get_chemical_symbols()
    si = next(i for i, x in enumerate(sym) if x == "Si")
    o_nb = min((i for i, x in enumerate(sym) if x == "O"), key=lambda i: np.linalg.norm(a.positions[i] - a.positions[si]))
    v = a.positions[si] - a.positions[o_nb]
    grp = [si] + [i for i in range(len(a)) if a.numbers[i] == 1 and np.linalg.norm(a.positions[i] - a.positions[si]) < 1.7]
    a.positions[grp] += 1.5 * v / np.linalg.norm(v)
    return a


def _run_conn(monkeypatch, plus, minus, inputs=INPUTS):
    seq = [plus, minus]

    def fake_relax(atoms, calc, cfg, opts, stem):
        a = seq.pop(0).copy()
        return a, {"converged": True, "steps": 1, "energy_eV": -10.0 if len(seq) == 1 else -10.1,
                   "fmax_eV_A": 0.01, "seconds": 0.0}

    monkeypatch.setattr(CN, "_relax", fake_relax)
    mode = np.random.default_rng(1).normal(size=(42, 3))
    conn = CN.connectivity_check(R0.copy(), mode, lambda: None, CFG, None, None, None, inputs=inputs)
    conn.pop("_atoms")
    return conn


def test_valid_sides_connect_and_supply_a_reference(monkeypatch):
    conn = _run_conn(monkeypatch, R0, P0)
    assert conn["connects"] and conn["coordination_checked"] and conn["invalid_sides"] == []
    assert conn["coordination_reference"] == "input structure of the same role"
    assert all(s["coordination"]["verdict"] == "PRESERVED" for s in conn["sides"].values())
    er, ep, why = CN.side_minimum_reference(conn)
    assert why is None and er is not None and ep is not None


def test_side_minimum_with_sn_water_dissociation_is_invalid(monkeypatch):
    conn = _run_conn(monkeypatch, _dissociate_water(R0), P0)
    assert conn["connects_by_hydride_state"] is True            # the hydride state alone would have said "connects"
    assert conn["connects"] is False and conn["invalid_sides"] == ["+"]
    assert conn["sides"]["+"]["structurally_valid"] is False
    assert conn["verdict"].startswith("DOES NOT CONNECT VALID MINIMA")
    er, ep, why = CN.side_minimum_reference(conn)
    assert er is None and ep is None and "structurally invalid" in why
    lab = CN.ts_certification(**{**KW, "connectivity": conn})
    assert lab.startswith("NOT CERTIFIED") and "DOES NOT CONNECT VALID MINIMA" in lab
    en = energetics(-9.0, [reference_state("relaxed_endpoint", -10.6, -10.9),
                           reference_state("connected_side_minimum", er, ep, None, why)], lab)
    assert en["barrier_forward"]["connected_side_minimum"] is None            # invalid side is not a reference
    assert en["reference_states"]["connected_side_minimum"]["unavailable_reason"] == why
    assert en["barrier_forward"]["relaxed_endpoint"] is not None


def test_product_side_with_broken_framework_bond_is_invalid(monkeypatch):
    conn = _run_conn(monkeypatch, R0, _cleave_si_o(P0))
    assert conn["connects"] is False and conn["invalid_sides"] == ["-"]
    assert conn["sides"]["-"]["coordination"]["heavy_bonds_broken"]
    assert CN.side_minimum_reference(conn)[0] is None


def test_hydride_state_mismatch_invalidates_connected_reference(monkeypatch):
    """Both sides fall to the reactant basin: valid chemistry, but not a reactant/product pair."""
    conn = _run_conn(monkeypatch, R0, R0)
    assert conn["connects"] is False and conn["connects_by_hydride_state"] is False
    assert "DOES NOT CONNECT reactant-like and product-like" in conn["verdict"]
    er, ep, why = CN.side_minimum_reference(conn)
    assert er is None and ep is None and "does not connect" in why


def test_coordination_not_requested_is_recorded_not_assumed(monkeypatch):
    conn = _run_conn(monkeypatch, R0, P0, inputs=None)
    assert conn["coordination_checked"] is False and conn["coordination_reference"] is None
    assert all(s["coordination"] is None for s in conn["sides"].values())
    assert "side-minimum Sn coordination NOT checked" in CN.ts_certification(**{**KW, "connectivity": conn})
    assert CN.coordination_configured(CFG) is True


def test_toy_without_sn_site_is_not_configured(toy):
    assert CN.coordination_configured(toy) is False


# ------------------------------------------------------ Q3: NEB status vs saddle status
def test_unconverged_neb_is_diagnostic_only_and_never_a_path_ts_or_barrier():
    p = run_protocol.ts_provenance({"converged": False, "neb_fmax_final": 0.594, "steps": 1500, "fmax_target": 0.05},
                                   {"converged": True})
    assert p["neb_converged"] is False and p["mep_claimed"] is False
    assert p["path_status"].startswith("NEB_UNCONVERGED") and "DIAGNOSTIC ONLY" in p["path_status"]
    assert "no converged reaction path" in p["path_status"]
    assert p["highest_image_is_ts"] is False and p["highest_image_energy_is_barrier"] is False
    assert p["neb_fmax_final"] == 0.594 and p["neb_fmax_target"] == 0.05 and p["neb_steps"] == 1500
    assert not _bare_barrier_keys(p)


def test_converged_sella_saddle_stays_certifiable_when_the_neb_did_not_converge():
    """Council Q3 policy: the saddle label is independent of how the seed was found."""
    prov = run_protocol.ts_provenance({"converged": False, "neb_fmax_final": 0.594, "steps": 1500, "fmax_target": 0.05},
                                      {"converged": True})
    label = CN.ts_certification(**KW)
    assert label.startswith("CERTIFIED")
    assert "NEB" not in label                                   # the label never carries / hides NEB status
    assert prov["path_status"].startswith("NEB_UNCONVERGED") and prov["mep_claimed"] is False
    assert prov["sella_converged"] is True


def test_converged_neb_and_no_neb_provenance():
    ok = run_protocol.ts_provenance({"converged": True, "neb_fmax_final": 0.04, "steps": 300, "fmax_target": 0.05},
                                    {"converged": True})
    assert ok["path_status"].startswith("NEB_CONVERGED") and "not an IRC" in ok["path_status"] and ok["mep_claimed"] is True
    assert ok["highest_image_is_ts"] is False                   # even a converged NEB's top image is a guess
    none = run_protocol.ts_provenance(None, {"converged": True})
    assert none["path_status"] == "NO_NEB_RUN" and none["mep_claimed"] is False and none["neb_converged"] is None


# --------------------------------------------------------------- Q4: barrier labelling
def test_barrier_outputs_always_name_their_reference_state():
    en = energetics(-9.0, [reference_state("relaxed_endpoint", -10.6, -10.9),
                           reference_state("connected_side_minimum", -10.8, -10.85)], "CERTIFIED: x")
    for q in ("barrier_forward", "barrier_reverse", "reaction_energy"):
        assert set(en[q]) == {"relaxed_endpoint", "connected_side_minimum"}
    assert not _bare_barrier_keys(en)
    assert set(en["reference_states"]) == {"relaxed_endpoint", "connected_side_minimum"}


# ------------------------------------------------------ portability of the 4M config
def test_model_path_expands_home_and_env_vars(monkeypatch):
    cfg = load_config(PKG / "config_omol4m.json")
    s = C.parse_calc_settings(cfg, model="mace-omol")
    assert s.model_path and "~" not in s.model_path and s.model_path.endswith("mace-omol-0-extra-large-4M.model")
    assert "/Users/" not in (PKG / "config_omol4m.json").read_text()      # no username in the tracked config
    monkeypatch.setenv("MACE_TEST_DIR", "/tmp/zz")
    cfg["calculator"]["models"]["mace-omol"]["model_path"] = "$MACE_TEST_DIR/a.model"
    assert C.parse_calc_settings(cfg, model="mace-omol").model_path == "/tmp/zz/a.model"


# ------------------------------------------------------------- result status words
def test_result_statuses_are_derived_from_the_evidence():
    unconv = run_protocol.ts_provenance({"converged": False, "neb_fmax_final": 0.59, "steps": 1500, "fmax_target": 0.05},
                                        {"converged": True})
    st = RS.result_statuses(CN.ts_certification(**KW), unconv, reproduces_reference=True)
    assert st == {"saddle": "CERTIFIED", "neb_path": "UNCONVERGED_DIAGNOSTIC_ONLY", "barriers": "PRELIMINARY",
                  "accuracy_vs_dft": "BLOCKED ON DFT DATA", "reproduced": "REPRODUCED"}
    bad = RS.result_statuses(CN.ts_certification(**{**KW, "sella_converged": False}), unconv)
    assert bad["saddle"] == "NOT CERTIFIED" and bad["reproduced"] is None
    conv = RS.result_statuses("CERTIFIED: x", run_protocol.ts_provenance({"converged": True, "neb_fmax_final": .04, "steps": 9, "fmax_target": .05},
                                                                           {"converged": True}))
    assert conv["neb_path"] == "NEB_CONVERGED" and conv["barriers"] == "PRELIMINARY"      # barriers never auto-upgrade
    assert RS.result_statuses("CERTIFIED: x", run_protocol.ts_provenance(None, {"converged": True}))["neb_path"] == "NO_NEB_RUN"
