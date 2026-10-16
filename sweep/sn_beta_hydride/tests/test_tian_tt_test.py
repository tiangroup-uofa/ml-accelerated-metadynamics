"""
Tests on Tian's RECONSTRUCTED tt-test input set and his saved MACE-POLAR
records (glucose/structures/). Read-only; nothing is computed with a model.

They pin: file provenance (hashes), the atom map / charge / spin agreement
between our config and Tian's files, structural validation of the 42-atom
cluster, and the both-reference-state numbers derived from raw energies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from ase.io import read

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[1]
GLU = REPO / "glucose" / "structures"
sys.path[:0] = [str(PKG), str(HERE)]

from config_io import load_config, resolve_path  # noqa: E402
from validate_structure import validate_pair, validate_single  # noqa: E402

pytestmark = pytest.mark.skipif(not GLU.exists(), reason="glucose/structures (tt-test) not present")


@pytest.fixture(scope="module")
def cfg():
    return load_config(PKG / "config_tian_tt_test.json")


@pytest.fixture(scope="module")
def summary(cfg):
    import summarize_tian_polar as STP

    return STP.summarize(cfg)


def test_dataset_files_match_recorded_hashes(summary):
    integ = summary["file_integrity"]
    assert integ["all_match"], integ
    assert set(integ["dataset_files"]) == {"reactant.xyz", "product.xyz", "atom_map_tian.json"}
    assert len(integ["tian_provenance_files"]) == 12


def test_dataset_is_byte_identical_to_tian_relaxed_endpoints():
    from summarize_tian_polar import sha256_lf

    d = PKG / "input" / "tian_tt_test"
    assert sha256_lf(d / "reactant.xyz") == sha256_lf(GLU / "reactant_mace-polar_polar-1-s.xyz")
    assert sha256_lf(d / "product.xyz") == sha256_lf(GLU / "product_mace-polar_polar-1-s.xyz")


def test_atom_map_charge_spin_agree_with_tian(cfg):
    tian = json.loads((GLU / "atom_map.json").read_text())
    assert cfg["atom_map"] == tian["atom_map"]
    assert (cfg["system"]["charge"], cfg["system"]["spin_multiplicity"]) == \
        (tian["charge"], tian["spin_multiplicity"])
    assert cfg["system"]["expected_n_atoms"] == tian["n_atoms"] == 42


def test_provenance_labels_reconstructed_and_inferred(cfg):
    p = cfg["provenance"]
    assert p["structure_class"] == "reconstructed" and p["not_literature_coordinates"] is True
    assert p["charge_spin_status"] == "inferred"
    prov = json.loads(resolve_path(cfg, p["provenance_file"]).read_text())
    assert prov["authoritative"] is False and prov["charge_spin_status"] == "inferred"
    assert prov["source"]["commit"].startswith("b97ef46")


def test_shipped_authoritative_config_stays_unconfigured():
    c = load_config(PKG / "config.json")
    assert c["provenance"]["structure_class"] == "authoritative"
    assert all(v is None for v in c["atom_map"].values())
    assert c["system"]["charge"] is None


def test_endpoint_pair_validation(cfg):
    st = cfg["structures"]
    rep = validate_pair(resolve_path(cfg, st["reactant"]), resolve_path(cfg, st["product"]), cfg)
    assert rep["n_errors"] == 0, rep
    R, P, pair = rep["reactant"]["info"], rep["product"]["info"], rep["pair"]
    assert R["formula"] == P["formula"] == "C6H22O10Si3Sn"
    assert R["hydride_state"]["state"] == "reactant-like"
    assert P["hydride_state"]["state"] == "product-like"
    for info in (R, P):
        assert len(info["cap_groups"]) == 3
        assert all(g["neighbour_counts"] == {"H": 3, "O": 1} for g in info["cap_groups"])
        assert info["Sn_environment"][0]["n_O"] == 6
        assert info["min_distance_A"] > 0.9
    assert ["C2(C2)", "H13(H_transfer)"] in pair["info"]["bonds_broken"]
    assert pair["info"]["bonds_formed"] == [["C1(C1)", "H13(H_transfer)"]]
    assert any("RECONSTRUCTED" in w for w in pair["warnings"])
    assert any("INFERRED" in w for w in pair["warnings"])


def test_tian_ts_and_side_minima_geometry(cfg):
    ts = validate_single(GLU / "ts_mace-polar_polar-1-s.xyz", cfg, "ts")
    assert ts["n_errors"] == 0 and ts["info"]["hydride_state"]["state"] == "shared"
    cd = ts["info"]["core_distances_A"]
    assert cd["C2-H*"] == pytest.approx(1.3144, abs=1e-3)
    assert cd["C1-H*"] == pytest.approx(1.3480, abs=1e-3)
    assert (round(cd["Sn-O1"], 2), round(cd["Sn-O2"], 2)) == (2.11, 2.13)
    rs = validate_single(GLU / "ts_reactant_side_mace-polar_polar-1-s.xyz", cfg, "rs", "reactant")
    ps = validate_single(GLU / "ts_product_side_mace-polar_polar-1-s.xyz", cfg, "ps", "product")
    assert rs["n_errors"] == ps["n_errors"] == 0


def test_polar_both_reference_states_from_raw_energies(summary):
    """Derived from saved total energies; compared with Tian's own JSON fields."""
    en = summary["energetics"]
    tsj = json.loads((GLU / "ts_mace-polar_polar-1-s.json").read_text())
    bf, rx = en["barrier_forward"], en["reaction_energy"]
    assert bf["relaxed_endpoint"]["kJ_mol"] == pytest.approx(tsj["barrier_forward_kJ_mol"], abs=2e-3)
    assert rx["relaxed_endpoint"]["kJ_mol"] == pytest.approx(tsj["dE_kJ_mol"], abs=2e-3)
    assert round(bf["relaxed_endpoint"]["kJ_mol"], 2) == 112.97
    assert round(bf["connected_side_minimum"]["kJ_mol"], 2) == 126.41
    assert round(rx["relaxed_endpoint"]["kJ_mol"], 2) == -30.02
    assert round(rx["connected_side_minimum"]["kJ_mol"], 2) == -19.17
    assert en["reference_state_dependent"] is True
    assert summary["energy_cross_check"]["all_consistent"]


def test_handoff_table_uses_side_minimum_reference(summary):
    hv = summary["agreement"]["vs_handoff_table"]
    assert hv["table_barrier_matches_reference_state"] == ["connected_side_minimum"]
    assert hv["table_dE_matches_reference_state"] == ["connected_side_minimum"]


def test_scan_maximum_is_not_the_ts(summary):
    sc = summary["scan_maximum"]
    assert "NOT a transition state" in sc["status"]
    assert sc["E_scan_max_minus_E_relaxed_reactant"]["kJ_mol"] == pytest.approx(118.46, abs=0.01)
    en = summary["energetics"]
    assert en["ts"]["energy_eV"] == summary["energies_eV"]["ts"]
    assert en["barrier_forward"]["relaxed_endpoint"]["kJ_mol"] < \
        sc["E_scan_max_minus_E_relaxed_reactant"]["kJ_mol"]
    assert sc["E_scan_max_minus_E_saddle"]["kJ_mol"] > 0


def test_polar_ts_not_independently_certified(summary):
    en = summary["energetics"]
    assert en["ts"]["certified"] is False
    assert en["ts"]["status"].startswith("NOT INDEPENDENTLY CERTIFIED")
    ev = summary["saddle_evidence_saved"]
    assert ev["n_significant_imaginary"] == 1 and ev["eigenvector_saved"] is False
    assert ev["side_minima_connect_reactant_and_product_like"] is True


def test_conformer_offsets(summary):
    rs = summary["conformer_diagnostics"]["reactant_side"]
    ps = summary["conformer_diagnostics"]["product_side"]
    assert rs["energy_offset_kJ_mol"] == pytest.approx(-13.44, abs=0.01)
    assert rs["material"] is True and rs["core_similar"] is True
    assert rs["rmsd_whole_A"] == pytest.approx(1.245, abs=2e-3)
    assert rs["n_changed_dihedrals"] >= 1
    assert ps["energy_offset_kJ_mol"] == pytest.approx(-2.58, abs=0.01)
    assert ps["material"] is False
    assert "not evidence of the global minimum" in rs["note"]


def test_relaxed_xyz_headers_are_inherited_from_guess():
    """Documented provenance quirk: product.xyz's comment line says REACTANT.
    The file is kept byte-identical; this test fails if someone edits it."""
    head = (PKG / "input" / "tian_tt_test" / "product.xyz").read_text().splitlines()[1]
    assert "REACTANT," in head and "energy=-56520.96505096508" in head
    assert read(str(PKG / "input" / "tian_tt_test" / "product.xyz")).get_potential_energy() == \
        pytest.approx(-56520.96505096508)
