"""
Unit tests for the Sn-beta hydride-transfer benchmark infrastructure.

Fast (no foundation-model weights are loaded). Structures used here are either
the TEST-ONLY toy (tests/toy_hydride.py), tiny hand-built fixtures, or existing
repository files (Diels-Alder TS) used purely as numerical test inputs.

    cd sweep/sn_beta_hydride && /usr/bin/python3 -m pytest tests -q
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO_SWEEP = PKG.parent
sys.path[:0] = [str(PKG), str(HERE)]

import calculators as C  # noqa: E402
from config_io import ConfigError, atom_map, charge_and_multiplicity, load_config  # noqa: E402
from geometry import kabsch_rmsd  # noqa: E402
from reaction_coordinate import ReactionCoordinates, basin_diagnostics, switching  # noqa: E402
from toy_hydride import Harmonic, product, reactant, toy_config  # noqa: E402
from validate_structure import (ValidationError, require_valid, validate_pair,  # noqa: E402
                                validate_single)
from vibrations import LAMBDA_TO_CM, analyze_hessian, hessian, vibrational_analysis  # noqa: E402


@pytest.fixture()
def toy(tmp_path):
    return toy_config(tmp_path)


def errors(rep):
    if "label" in rep:
        return rep["errors"]
    return [e for k in ("reactant", "product", "pair") for e in rep[k]["errors"]]


def warnings_(rep):
    return [w for k in ("reactant", "product", "pair") for w in rep[k]["warnings"]]


# --------------------------------------------------------------------------- #
#  the shipped scientific config
# --------------------------------------------------------------------------- #
def test_shipped_config_is_unconfigured_and_fails_loudly():
    cfg = load_config(PKG / "config.json")
    with pytest.raises(ConfigError, match="not configured"):
        atom_map(cfg)
    with pytest.raises(ConfigError):
        charge_and_multiplicity(cfg)
    rep = validate_pair(PKG / "input" / "reactant.xyz", PKG / "input" / "product.xyz", cfg)
    assert rep["status"] == "FAIL"
    assert not cfg["system"]["test_only"]


def test_toy_pair_passes(toy):
    rep = validate_pair(reactant(), product(), toy)
    assert rep["status"] == "PASS", errors(rep) + warnings_(rep)
    assert rep["pair"]["info"]["bonds_broken"] == [["C4(C2)", "H8(H_transfer)"]]
    assert rep["pair"]["info"]["bonds_formed"] == [["C0(C1)", "H8(H_transfer)"]]


# --------------------------------------------------------------------------- #
#  invalid atom mappings
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,msg", [
    (None, "not configured"),
    (99, "out of range"),
    (-1, "negative"),
    (True, "not an integer"),
    ("4", "not an integer"),
])
def test_bad_atom_map_values(toy, value, msg):
    cfg = copy.deepcopy(toy)
    cfg["atom_map"]["H_transfer"] = value
    with pytest.raises(ConfigError, match=msg):
        atom_map(cfg, 9)
    rep = validate_single(reactant(), cfg, "r", "reactant")
    assert rep["status"] == "FAIL"


def test_duplicate_mapping(toy):
    cfg = copy.deepcopy(toy)
    cfg["atom_map"]["H_C1"] = 8
    with pytest.raises(ConfigError, match="both point to index 8"):
        atom_map(cfg, 9)


def test_reactive_atom_wrong_element(toy):
    cfg = copy.deepcopy(toy)
    cfg["atom_map"]["H_transfer"] = 5          # an oxygen
    rep = validate_single(reactant(), cfg, "r", "reactant")
    assert any("expected H" in e for e in errors(rep))


def test_wrong_hydrogen_index_caught_by_endpoint_bonds(toy):
    cfg = copy.deepcopy(toy)
    cfg["atom_map"]["H_transfer"] = 6          # an H on C2 that does NOT transfer
    rep = validate_pair(reactant(), product(), cfg)
    assert rep["status"] == "FAIL"
    assert any("expected C1-H_transfer bonded" in e for e in errors(rep))


def test_swapped_endpoints_fail(toy):
    rep = validate_pair(product(), reactant(), toy)
    assert rep["status"] == "FAIL"
    assert any("swapped endpoints" in e for e in errors(rep))


# --------------------------------------------------------------------------- #
#  mismatched endpoints
# --------------------------------------------------------------------------- #
def test_mismatched_atom_count(toy):
    p = product()
    del p[7]
    rep = validate_pair(reactant(), p, toy)
    assert any("reactant has 9 atoms, product has 8" in e for e in errors(rep))


def test_mismatched_element_order(toy):
    p = product()
    syms = p.get_chemical_symbols()
    syms[1], syms[2] = syms[2], syms[1]       # O <-> H swap in ordering only
    p.set_chemical_symbols(syms)
    rep = validate_pair(reactant(), p, toy)
    assert any("element ordering differs" in e for e in errors(rep))


def test_permuted_spectator_atoms_warn(toy):
    p = product()
    pos = p.get_positions()
    pos[[2, 6]] = pos[[6, 2]]                  # same elements, different atoms
    p.set_positions(pos)
    rep = validate_pair(reactant(), p, toy)
    assert any("possible atom permutation" in w or "unexpected bond" in w
               for w in warnings_(rep))


# --------------------------------------------------------------------------- #
#  geometry sanity
# --------------------------------------------------------------------------- #
def test_overlapping_atoms(toy):
    r = reactant()
    r.positions[7] = r.positions[6] + [0.05, 0, 0]
    rep = validate_single(r, toy, "r")
    assert any("duplicate/overlapping" in e for e in errors(rep))


def test_impossible_short_contact(toy):
    r = reactant()
    r.positions[8] = r.positions[0] + [0.35, 0, 0]
    rep = validate_single(r, toy, "r")
    assert any("impossibly short" in e for e in errors(rep))


def test_nonfinite_coordinates(toy):
    r = reactant()
    r.positions[3, 1] = np.nan
    rep = validate_single(r, toy, "r")
    assert any("non-finite" in e for e in errors(rep))


def test_isolated_atom(toy):
    r = reactant()
    r += Atoms("H", positions=[[20.0, 0, 0]])
    cfg = copy.deepcopy(toy)
    cfg["system"]["expected_n_atoms"] = None
    rep = validate_single(r, cfg, "r")
    assert any("isolated atom" in e for e in errors(rep))


def test_file_level_errors(toy, tmp_path):
    rep = validate_single(tmp_path / "missing.xyz", toy, "x")
    assert any("does not exist" in e for e in errors(rep))
    bad = tmp_path / "bad.xyz"
    bad.write_text("not\nan xyz file at all\n???")
    rep = validate_single(bad, toy, "x")
    assert rep["status"] == "FAIL"
    multi = tmp_path / "multi.xyz"
    write(str(multi), [reactant(), product()])
    rep = validate_single(multi, toy, "x")
    assert any("2 frames" in e for e in errors(rep))


def test_charge_multiplicity_parity(toy):
    cfg = copy.deepcopy(toy)
    cfg["system"]["spin_multiplicity"] = 2     # 9-atom anion has an even electron count
    rep = validate_single(reactant(), cfg, "r")
    assert any("parity" in e for e in errors(rep))
    cfg["system"]["charge"] = None
    rep = validate_single(reactant(), cfg, "r")
    assert any("charge/multiplicity not configured" in e for e in errors(rep))


def test_required_sn_and_sn_environment(toy):
    cfg = copy.deepcopy(toy)
    cfg["system"]["require_elements"] = ["Sn"]
    rep = validate_single(reactant(), cfg, "r")
    assert any("required element Sn is absent" in e for e in errors(rep))

    # tetrahedral Sn(OH)4 fixture: Sn shell is a diagnostic
    d = 1.95 / np.sqrt(3)
    o = np.array([[d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]])
    h = o * (1 + 0.97 / 1.95)
    sn = Atoms("SnO4H4", positions=np.vstack([[0, 0, 0], o, h]))
    cfg2 = {"system": {"charge": 0, "spin_multiplicity": 1}, "atom_map": {"Sn": 0},
            "expected_elements": {"Sn": "Sn"}}
    rep = validate_single(sn, cfg2, "sn")
    env = rep["info"]["Sn_environment"][0]
    assert env["n_O"] == 4 and env["coordination"] == 4
    assert abs(env["nearest_O_A"] - 1.95) < 1e-3


def test_require_valid_raises(toy):
    r = reactant()
    r.positions[7] = r.positions[6]
    with pytest.raises(ValidationError):
        require_valid(validate_single(r, toy, "r"), "test")


def test_stage_refuses_invalid_input_before_loading_model(toy, monkeypatch, tmp_path):
    import optimize_endpoints

    def boom(*a, **k):
        raise AssertionError("model must not be loaded for invalid input")

    monkeypatch.setattr(optimize_endpoints, "model_from_config", boom)
    cfg = copy.deepcopy(toy)
    cfg["_config_dir"] = str(tmp_path)
    cfg["atom_map"]["H_transfer"] = None
    with pytest.raises(ValidationError):
        optimize_endpoints.run(cfg)


# --------------------------------------------------------------------------- #
#  CVs
# --------------------------------------------------------------------------- #
def test_switching_function():
    assert switching(np.array([0.0]), 1.5)[0] == 1.0
    assert abs(switching(np.array([1.5]), 1.5, 6, 12)[0] - 0.5) < 1e-12   # n/m
    x = 1.1 / 1.5
    assert abs(switching(np.array([1.1]), 1.5)[0] - 1 / (1 + x ** 6)) < 1e-12
    assert switching(np.array([10.0]), 1.5)[0] < 1e-4
    # continuous through the removable singularity
    a, b = switching(np.array([1.5 - 1e-6, 1.5 + 1e-6]), 1.5)
    assert abs(a - b) < 1e-5


def test_cv_values_and_symmetry(toy):
    rc = ReactionCoordinates.from_config(toy, 9, reactant().get_chemical_symbols())
    r, p = rc.compute(reactant()), rc.compute(product())
    assert abs(r["d_C2_H"] - 1.12) < 1e-9 and abs(r["d_C1_H"] - 1.98) < 1e-9
    assert abs(r["delta_H"] - (1.12 - 1.98)) < 1e-9
    assert abs(r["delta_H"] + p["delta_H"]) < 1e-9          # mirror image
    assert abs(r["CN_C1_H"] - p["CN_C2_H"]) < 1e-9
    rows = rc.compute_many([reactant(), product()])
    assert len(rows) == 2 and rows[1]["delta_H"] > 0


def test_cv_gradient_matches_analytic(toy):
    a = reactant()
    rc = ReactionCoordinates.from_config(toy, 9, a.get_chemical_symbols(), ["delta_H"])
    g = rc.gradient(a, "delta_H")
    p = a.get_positions()
    u2 = (p[8] - p[4]) / np.linalg.norm(p[8] - p[4])
    u1 = (p[8] - p[0]) / np.linalg.norm(p[8] - p[0])
    assert np.allclose(g[8], u2 - u1, atol=1e-6)
    assert np.allclose(g[4], -u2, atol=1e-6) and np.allclose(g[0], u1, atol=1e-6)


def test_cv_config_errors(toy):
    cfg = copy.deepcopy(toy)
    cfg["cvs"]["bad"] = {"type": "angle"}
    with pytest.raises(ConfigError, match="unknown type"):
        ReactionCoordinates.from_config(cfg, 9, reactant().get_chemical_symbols())
    cfg = copy.deepcopy(toy)
    cfg["cvs"]["CN_C1_H"]["r0"] = None
    with pytest.raises(ConfigError, match="r0"):
        ReactionCoordinates.from_config(cfg, 9, reactant().get_chemical_symbols())
    rc = ReactionCoordinates.from_config(toy, 9, reactant().get_chemical_symbols())
    with pytest.raises(ValueError, match="configured for 9 atoms"):
        rc.compute(reactant()[:5])


def test_basin_diagnostics_is_diagnostic():
    cfg = {"reference_basins": {"tolerance": 0.3, "basins": {
        "reactant": {"CV1": 0.9, "CV2": 0.9}, "product": {"CV1": 1.8, "CV2": 0.1}}}}
    d = basin_diagnostics({"CV1": 1.7, "CV2": 0.2}, cfg)
    assert d["nearest"] == "product"
    assert d["basins"]["product"]["within_tolerance"] is True
    assert "diagnostic" in d["note"]


# --------------------------------------------------------------------------- #
#  calculator selection / config parsing (no weights loaded)
# --------------------------------------------------------------------------- #
def test_calc_settings_parsing(toy):
    s = C.parse_calc_settings(toy)
    assert (s.model, s.variant, s.dtype, s.tag) == ("mace-polar", "polar-1-s", "float64",
                                                    "mace-polar_polar-1-s")
    s = C.parse_calc_settings(toy, model="mace-omol", dtype="float32")
    assert (s.model, s.variant, s.dtype) == ("mace-omol", "extra_large", "float32")
    with pytest.raises(ConfigError, match="unknown calculator model"):
        C.parse_calc_settings(toy, model="mace-off23")
    with pytest.raises(ConfigError, match="unknown variant"):
        C.parse_calc_settings(toy, model="mace-polar", variant="polar-9")
    with pytest.raises(ConfigError, match="dtype"):
        C.parse_calc_settings(toy, dtype="float16")
    cfg = copy.deepcopy(toy)
    cfg["calculator"]["models"]["mace-polar"]["options"] = {"device": "cuda"}
    with pytest.raises(ConfigError, match="dedicated field"):
        C.parse_calc_settings(cfg)


class FakeBase(Calculator):
    """Stand-in for a MACECalculator: energy encodes the charge/spin it saw."""
    implemented_properties = ["energy", "free_energy", "forces"]

    class _Z:
        zs = [1, 6, 8]

    z_table = _Z()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.seen = (atoms.info.get("charge"), atoms.info.get("spin"))
        e = 10.0 * atoms.info["charge"] + atoms.info["spin"]
        self.results = {"energy": e, "free_energy": e, "forces": np.zeros((len(atoms), 3))}


def test_polar_dependency_check_in_fresh_process():
    """check_polar_dependency must not crash on import order in a clean interpreter
    (regression: importing graph_longrange/e3nn before mace fails on torch >= 2.6)."""
    import subprocess

    code = ("import sys; sys.path.insert(0, %r)\n"
            "import calculators as C\n"
            "try:\n    C.check_polar_dependency(); print('OK')\n"
            "except RuntimeError as e:\n    print('MISSING', str(e).splitlines()[0])\n" % str(PKG))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    out = r.stdout.strip().splitlines()
    assert r.returncode == 0, r.stderr[-2000:]
    assert out and out[-1].split()[0] in ("OK", "MISSING"), r.stdout + r.stderr[-2000:]


def test_registry_dispatch_and_charge_spin_injection(toy, monkeypatch):
    built = {}

    def fake_builder(variant, device, dtype, options):
        built.update(variant=variant, device=device, dtype=dtype)
        return FakeBase()

    spec = C.MODELS["mace-polar"]
    monkeypatch.setitem(C.MODELS, "mace-polar",
                        C.ModelSpec(spec.key, spec.pretty, fake_builder, spec.variants,
                                    spec.default_variant))
    h = C.model_from_config(toy)
    assert built == {"variant": "polar-1-s", "device": "cpu", "dtype": "float64"}
    a = reactant()
    a.calc = h.new_calc()
    assert a.get_potential_energy() == pytest.approx(10 * -1 + 1)
    assert h.base.seen == (-1, 1)
    assert h.total_calls == 1
    b = reactant()
    b.info["charge"] = 0                       # conflicting with configured -1
    b.calc = h.new_calc()
    with pytest.raises(ValueError, match="conflicts"):
        b.get_potential_energy()
    with pytest.raises(ValueError, match="does not support"):
        h.check_elements(Atoms("SnH4", positions=np.random.rand(5, 3) * 3))


# --------------------------------------------------------------------------- #
#  Hessian / mass-weighted vibrational analysis
# --------------------------------------------------------------------------- #
def test_harmonic_diatomic_frequency():
    k, r0 = 30.0, 1.1                          # eV/Å², Å
    a = Atoms("CO", positions=[[0, 0, 0], [r0, 0, 0]])
    calc = Harmonic([(0, 1)], k, r0)
    res = vibrational_analysis(a, calc)
    m = a.get_masses()
    mu = m[0] * m[1] / (m[0] + m[1])
    expected = LAMBDA_TO_CM * np.sqrt(k / mu)
    assert max(res["frequencies_cm"]) == pytest.approx(expected, rel=1e-5)
    assert res["n_imaginary"] == 0


def test_inverted_spring_gives_one_imaginary_mode_with_full_overlap():
    a = Atoms("CO", positions=[[0, 0, 0], [1.1, 0, 0]])
    calc = Harmonic([(0, 1)], -30.0, 1.1)       # a 1-D "barrier" along the bond
    ref = np.zeros((2, 3))
    ref[0, 0], ref[1, 0] = -1, 1                # bond stretch direction
    res = vibrational_analysis(a, calc, reference_vector=ref)
    assert res["n_imaginary"] == 1
    assert res["nu_imag_cm"] < -50
    # the stretch keeps the centre of mass fixed: Cartesian displacements are
    # (-m2, +m1)/norm, so the overlap with the symmetric (-1, +1)/sqrt2 stretch
    # is (m1 + m2) / sqrt(2 (m1^2 + m2^2)), not 1, for unequal masses
    m1, m2 = a.get_masses()
    expected = (m1 + m2) / np.sqrt(2 * (m1 ** 2 + m2 ** 2))
    assert res["reaction_mode_overlap"] == pytest.approx(expected, abs=1e-6)
    disp = np.array(res["mode_displacement"])
    assert abs(disp[0, 0] / disp[1, 0]) == pytest.approx(m2 / m1, rel=1e-6)


def test_partial_hessian_shape():
    a = reactant()
    calc = Harmonic([(4, 8), (0, 1)], 30.0, 1.1)
    H = hessian(a, calc, indices=[0, 4, 8])
    assert H.shape == (9, 9) and np.allclose(H, H.T)
    res = analyze_hessian(H, a, indices=[0, 4, 8])
    assert res["n_atoms_in_hessian"] == 3


def test_vibrations_match_phase2_lib():
    """Generalized helper == verified phase2_lib on identical input (EMT on the DA TS)."""
    pytest.importorskip("sella")
    sys.path.insert(0, str(REPO_SWEEP / "corrections"))
    try:
        import phase2_lib as L
    except Exception as e:  # pragma: no cover
        pytest.skip(f"phase2_lib not importable here: {e}")
    from ase.calculators.emt import EMT
    from ase.io import read

    ts = read(str(REPO_SWEEP / "diels_alder" / "ts_opt.xyz"))
    ref_p2 = L.vibrational_analysis(ts, EMT())
    ref_vec = L.reaction_coordinate_vector(ts).reshape(-1, 3)
    mine = vibrational_analysis(ts, EMT(), reference_vector=ref_vec)
    assert mine["n_imaginary"] == ref_p2["n_imaginary"]
    assert np.allclose(mine["lowest_10_cm"], ref_p2["lowest_10_cm"], atol=1e-8)
    assert np.allclose(mine["imag_freqs_cm"], ref_p2["imag_freqs_cm"], atol=1e-8)
    if ref_p2["reaction_mode_overlap"] is not None:
        assert mine["reaction_mode_overlap"] == pytest.approx(ref_p2["reaction_mode_overlap"],
                                                              abs=1e-10)
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(12, 3)), rng.normal(size=(12, 3))
    assert kabsch_rmsd(A, B) == pytest.approx(L.kabsch_rmsd(A, B), abs=1e-12)


def test_ts_verdict_requires_stationarity_and_one_aligned_mode():
    import validate_ts as V

    opts = dict(V.DEFAULTS)
    one = {"n_imaginary": 1, "reaction_mode_overlap": 0.9,
           "mode_atom_participation": [{"index": 8, "element": "H", "fraction": 0.7}]}
    assert V.verdict(one, opts, 8, fmax=1e-4).startswith("FIRST-ORDER SADDLE CANDIDATE")
    assert V.verdict(one, opts, 8, fmax=0.05).startswith("NOT A VERIFIED STATIONARY POINT")
    misaligned = dict(one, reaction_mode_overlap=0.1)
    assert "NOT the configured hydride" in V.verdict(misaligned, opts, 8, fmax=1e-4)
    assert V.verdict(dict(one, n_imaginary=0), opts, 8).startswith("NOT A SADDLE")
    assert V.verdict(dict(one, n_imaginary=2), opts, 8).startswith("HIGHER-ORDER")


# --------------------------------------------------------------------------- #
#  literature-derived settings (Oct 2026 audit; see README "Literature evidence")
# --------------------------------------------------------------------------- #
def test_shipped_cvs_use_both_hydrogens_and_stay_unconfirmed():
    cfg = load_config(PKG / "config.json")
    for name, center in (("CV1", ["C1"]), ("CV2", ["C2"])):
        cv = cfg["cvs"][name]
        assert cv["type"] == "coordination" and cv["center"] == center
        assert sorted(cv["group"]) == ["H_C1", "H_transfer"]      # H1 and H2 of Fig. 1c
        assert cv["parameters_confirmed"] is False
    assert cfg["system"]["charge"] is None and cfg["system"]["spin_multiplicity"] is None
    assert "Si" not in cfg["system"]["require_elements"]       # Si: warning only
    assert "Si" in cfg["system"]["expect_elements"]
    assert {"Sn", "C", "H", "O"} <= set(cfg["system"]["require_elements"])


def test_cv2_counts_both_hydrogens(toy):
    cfg = copy.deepcopy(toy)
    cfg["cvs"] = {"CV2": {"type": "coordination", "center": ["C2"],
                          "group": ["H_C1", "H_transfer"], "r0": 1.5, "n": 6, "m": 12}}
    a = reactant()
    v = ReactionCoordinates.from_config(cfg, 9).compute(a)["CV2"]
    d = lambda i, j: np.linalg.norm(a.positions[i] - a.positions[j])
    expected = switching(np.array([d(4, 2), d(4, 8)]), 1.5).sum()
    assert v == pytest.approx(expected, abs=1e-12)


def test_basins_not_interpreted_with_unconfirmed_parameters():
    cfg = {"cvs": {"CV1": {"parameters_confirmed": False}, "CV2": {}},
           "reference_basins": {"tolerance": 0.3, "basins": {
               "reactant": {"CV1": 0.9, "CV2": 0.9}, "product": {"CV1": 1.8, "CV2": 0.1}}}}
    d = basin_diagnostics({"CV1": 1.8, "CV2": 0.1}, cfg)
    assert d["comparable"] is False and "nearest" not in d
    assert d["basins"]["product"]["within_tolerance"] is None
    assert d["basins"]["product"]["distance"] == pytest.approx(0.0)


def test_sn_site_expectations_are_warnings_only():
    d = 1.95 / np.sqrt(3)
    o = np.array([[d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]])
    h = o * (1 + 0.97 / 1.95)
    # Sn(OH)4 plus a water 4 Å away that is (wrongly) mapped as the Sn-bound water
    w = np.array([[4.0, 0, 0], [4.6, 0.75, 0], [4.6, -0.75, 0]])
    at = Atoms("SnO4H4OH2", positions=np.vstack([[0, 0, 0], o, h, w]))
    cfg = {"system": {"charge": 0, "spin_multiplicity": 1}, "atom_map": {"Sn": 0, "O_water": 9},
           "expected_elements": {"Sn": "Sn", "O_water": "O"},
           "sn_site": {"expected_n_O": 6, "expected_neighbours": ["O_water"]}}
    rep = validate_single(at, cfg, "sn")
    assert rep["status"] == "WARN", rep["errors"]
    assert any("has 4 O" in w and "expected 6" in w and "diagnostic only" in w for w in rep["warnings"])
    assert any("expected neighbour O_water" in w for w in rep["warnings"])


def test_missing_expected_element_warns_but_missing_required_element_fails(toy):
    """Si is expected (SiH3-capped cluster in the literature drawing) but not
    required: its absence must warn, never fail validation."""
    cfg = copy.deepcopy(toy)
    cfg["system"]["expect_elements"] = ["Si"]           # the toy has no Si
    rep = validate_pair(reactant(), product(), cfg)
    assert rep["status"] == "WARN", errors(rep)
    assert rep["n_errors"] == 0
    assert any("expected element Si is absent" in w for w in warnings_(rep))
    cfg["system"]["require_elements"] = ["Si"]          # contrast: required -> hard error
    rep = validate_pair(reactant(), product(), cfg)
    assert rep["status"] == "FAIL"
    assert any("required element Si is absent" in e for e in errors(rep))
