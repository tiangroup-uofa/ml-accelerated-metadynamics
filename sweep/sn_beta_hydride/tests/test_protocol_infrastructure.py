"""
Tests for the reference-state, connectivity, conformer, provenance and xTB
infrastructure added after the tt-test audit. Fast; no foundation-model weights,
no xtb binary. Physics checks use the TEST-ONLY analytic DoubleWellHydride toy
(tests/toy_hydride.py), whose saddle and minima are known exactly.

    cd sweep/sn_beta_hydride && python -m pytest tests -q
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import write

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path[:0] = [str(PKG), str(HERE)]

import calculators as C  # noqa: E402
from config_io import ConfigError, load_config  # noqa: E402
from conformer_diagnostics import compare, representative_dihedrals  # noqa: E402
from connectivity import (NOT_IRC, connectivity_check, displaced_pair,  # noqa: E402
                          side_minimum_reference, ts_certification)
from energetics import EV2KJ, energetics, reference_state  # noqa: E402
from hydride import classify_hydride_state, core_distances  # noqa: E402
from toy_hydride import (DoubleWellHydride, product_minimum, reactant,  # noqa: E402
                         toy_config, ts_geometry)
from validate_structure import ValidationError, validate_pair  # noqa: E402
from vibrations import vibrational_analysis  # noqa: E402


@pytest.fixture()
def toy(tmp_path):
    cfg = toy_config(tmp_path)
    cfg["fixed_atoms"] = []
    cfg["_config_dir"] = str(tmp_path)
    return cfg


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


# --------------------------------------------------------------------------- #
#  reference-state energetics
# --------------------------------------------------------------------------- #
def test_reference_state_barrier_arithmetic():
    states = [reference_state("relaxed_endpoint", -10.0, -10.3),
              reference_state("connected_side_minimum", -10.14, -10.33)]
    en = energetics(-8.9, states, "CERTIFIED: test")
    bf, rx = en["barrier_forward"], en["reaction_energy"]
    assert bf["relaxed_endpoint"]["eV"] == pytest.approx(1.1)
    assert bf["connected_side_minimum"]["eV"] == pytest.approx(1.24)
    assert bf["relaxed_endpoint"]["kJ_mol"] == pytest.approx(1.1 * EV2KJ)
    assert en["barrier_reverse"]["relaxed_endpoint"]["eV"] == pytest.approx(1.4)
    assert rx["relaxed_endpoint"]["eV"] == pytest.approx(-0.3)
    assert rx["connected_side_minimum"]["eV"] == pytest.approx(-0.19)
    assert en["reference_state_spread_kJ_mol"]["barrier_forward"] == pytest.approx(0.14 * EV2KJ)
    assert en["reference_state_dependent"] is True
    assert en["ts"]["certified"] is True


def test_no_unlabeled_barrier_anywhere():
    en = energetics(-8.9, [reference_state("relaxed_endpoint", -10.0, -10.3),
                           reference_state("connected_side_minimum")], "NOT CERTIFIED: x")
    assert "barrier" not in set(_keys(en))                  # never a bare barrier key
    assert set(en["barrier_forward"]) == {"relaxed_endpoint", "connected_side_minimum"}
    assert en["barrier_forward"]["connected_side_minimum"] is None
    assert en["reference_states"]["connected_side_minimum"]["unavailable_reason"]
    assert "UNCERTIFIED" in en["quantity_label"] and en["ts"]["certified"] is False


def test_reference_state_names_are_closed():
    with pytest.raises(ValueError, match="unknown reference state"):
        reference_state("scan_maximum", -1.0, -1.0)
    with pytest.raises(ValueError, match="duplicate"):
        energetics(0.0, [reference_state("relaxed_endpoint", -1, -1)] * 2, "x")


# --------------------------------------------------------------------------- #
#  hydride-state classification and core distances
# --------------------------------------------------------------------------- #
def test_hydride_state_classification(toy):
    assert classify_hydride_state(reactant(), toy)["state"] == "reactant-like"
    assert classify_hydride_state(product_minimum(), toy)["state"] == "product-like"
    ts = classify_hydride_state(ts_geometry(), toy)
    assert ts["state"] == "shared" and abs(ts["delta_H_A"]) < 1e-12
    far = reactant()
    far.positions[8] += [0, 4.0, 0]
    assert classify_hydride_state(far, toy)["state"] == "detached"
    cd = core_distances(reactant(), toy)
    assert cd["C2-H*"] == pytest.approx(1.12, abs=1e-4) and "Sn-O1" not in cd


def test_endpoint_validation_uses_hydride_state(toy):
    rep = validate_pair(product_minimum(), reactant(), toy)          # roles swapped
    errs = [e for k in ("reactant", "product") for e in rep[k]["errors"]]
    assert any("hydride state is product-like" in e and "expected reactant-like" in e for e in errs)
    assert rep["reactant"]["info"]["hydride_state"]["state"] == "product-like"


# --------------------------------------------------------------------------- #
#  +/- mode displacement and connectivity
# --------------------------------------------------------------------------- #
def test_displaced_pair_logic():
    ts = ts_geometry()
    mode = np.zeros((9, 3))
    mode[8] = [2.0, 0, 0]
    mode[0] = [-0.5, 0, 0]
    pos0 = ts.get_positions().copy()
    plus, minus, step = displaced_pair(ts, mode, 0.1)
    assert np.allclose(ts.get_positions(), pos0)                       # TS preserved
    assert np.linalg.norm(step, axis=1).max() == pytest.approx(0.1)    # lead atom 0.1 Å
    assert np.allclose(plus.positions - pos0, -(minus.positions - pos0))
    assert np.allclose(step[0], [-0.025, 0, 0])                       # proportional
    with pytest.raises(ValueError):
        displaced_pair(ts, np.zeros((9, 3)), 0.1)


def _ts_mode(toy):
    from reaction_coordinate import ReactionCoordinates

    ts = ts_geometry()
    rc = ReactionCoordinates.from_config(toy, 9, ts.get_chemical_symbols(), ["delta_H"])
    return vibrational_analysis(ts.copy(), DoubleWellHydride(),
                                reference_vector=rc.gradient(ts, "delta_H"))


def test_toy_saddle_has_one_imaginary_hydride_mode(toy):
    res = _ts_mode(toy)
    assert res["n_imaginary"] == 1 and res["nu_imag_cm"] < -1000
    assert res["reaction_mode_overlap"] > 0.8
    assert res["mode_atom_participation"][0]["index"] == 8


def test_connectivity_identifies_reactant_and_product_sides(toy, tmp_path):
    res = _ts_mode(toy)
    ts = ts_geometry()
    pos0 = ts.get_positions().copy()
    eps = {"reactant": {"atoms": reactant(), "energy_eV": 0.0},
           "product": {"atoms": product_minimum(), "energy_eV": 0.0}}
    conn = connectivity_check(ts, res["mode_displacement"], DoubleWellHydride, toy, eps,
                              tmp_path / "conn", ts_energy_eV=1.0)
    conn.pop("_atoms")
    assert np.allclose(ts.get_positions(), pos0)
    assert conn["connects"] and conn["both_converged"]
    assert {s["state"] for s in conn["sides"].values()} == {"reactant-like", "product-like"}
    assert conn["sides"][conn["reactant_side"]]["matched_endpoint"] == "reactant"
    for s in conn["sides"].values():
        m = s["matched_endpoint"]
        assert s["vs_endpoints"][m]["rmsd_whole_A"] < 0.01
        assert s["conformer_vs_matched_endpoint"]["material"] is False
        assert s["energy_below_ts_kJ_mol"] == pytest.approx(EV2KJ, rel=1e-3)
        assert "C2-H*" in s["core_distances_A"]
        assert Path(s["files"]["relaxed"]).exists() and Path(s["files"]["start"]).exists()
    assert conn["method_note"] == NOT_IRC and "NOT an intrinsic reaction coordinate" in NOT_IRC
    e_r, e_p, why = side_minimum_reference(conn)
    # relaxed to fmax 0.02 eV/Å (Tian's setting): residual energy ~ f^2/k ~ 1e-5 eV
    assert why is None and e_r == pytest.approx(0.0, abs=1e-4) and e_p == pytest.approx(0.0, abs=1e-4)


def test_connectivity_reports_failure_when_both_sides_fall_to_reactant(toy, tmp_path):
    """A 'mode' that does not move H* (all on spectators) relaxes back to the
    reactant on both sides -> must NOT be reported as connecting."""
    mode = np.zeros((9, 3))
    mode[6] = [0, 0, 1.0]
    start = reactant()
    eps = {"reactant": {"atoms": reactant(), "energy_eV": 0.0}}
    conn = connectivity_check(start, mode, DoubleWellHydride, toy, eps, None)
    conn.pop("_atoms")
    assert conn["connects"] is False and "DOES NOT CONNECT" in conn["verdict"]
    assert side_minimum_reference(conn)[2] is not None


def test_ts_certification_requires_all_evidence():
    good = {"connects": True, "both_converged": True, "verdict": "CONNECTS"}
    kw = dict(stationary=True, fmax=1e-4, stationary_fmax=0.01, n_imaginary=1, overlap=0.8,
              min_overlap=0.3, geometry_valid=True, connectivity=good)
    assert ts_certification(**kw).startswith("CERTIFIED")
    assert "(not an IRC)" in ts_certification(**kw)
    for change, msg in ((dict(stationary=False, fmax=0.05), "not a stationary point"),
                        (dict(n_imaginary=0), "0 significant"),
                        (dict(n_imaginary=2), "2 significant"),
                        (dict(overlap=0.1), "not aligned"),
                        (dict(overlap=None), "not aligned"),
                        (dict(geometry_valid=False), "geometry"),
                        (dict(connectivity=None), "connectivity not checked"),
                        (dict(connectivity={**good, "connects": False,
                                            "verdict": "DOES NOT CONNECT"}), "DOES NOT CONNECT"),
                        (dict(connectivity={**good, "both_converged": False}), "did not converge")):
        lab = ts_certification(**{**kw, **change})
        assert lab.startswith("NOT CERTIFIED") and msg in lab, (change, lab)


# --------------------------------------------------------------------------- #
#  conformer diagnostics
# --------------------------------------------------------------------------- #
def test_conformer_offset_reporting(toy):
    ref = reactant()
    other = reactant()
    # rotate the O2/H2a/H2b substituents of C2 by 60 deg about the C1-C2 axis:
    # a "conformer" change away from the H* core atoms only
    ax = other.positions[4] - other.positions[0]
    ax /= np.linalg.norm(ax)
    th = np.radians(60)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    for i in (5, 6, 7):
        other.positions[i] = other.positions[4] + R @ (other.positions[i] - other.positions[4])
    cfg = copy.deepcopy(toy)
    cfg["conformer"] = {"core": ["C1", "C2", "H_transfer"]}
    big = compare(ref, other, cfg, -10.0, -10.0 - 13.44 / EV2KJ, "test")
    assert big["material"] is True and big["energy_offset_kJ_mol"] == pytest.approx(-13.44)
    assert big["core_similar"] is True and big["rmsd_core_A"] < 1e-6
    assert big["rmsd_whole_A"] > 0.1
    assert "MATERIAL conformational" in big["interpretation"]
    small = compare(ref, other, cfg, -10.0, -10.0 - 2.58 / EV2KJ)
    assert small["material"] is False and "below the materiality" in small["interpretation"]
    assert "not evidence of the global minimum" in small["note"]
    assert compare(ref, other, cfg)["material"] is None              # energies unknown


def test_changed_dihedral_detected_on_butane_rotation():
    """trans-butane -> rotate one end 120 deg about the central C-C bond: the
    central torsion must be reported with a ~120 deg change; core untouched."""
    from ase.build import molecule

    ref = molecule("trans-butane")
    syms = ref.get_chemical_symbols()
    from geometry import bond_set
    cs = [i for i, s in enumerate(syms) if s == "C"]
    bonds = bond_set(ref)
    deg = {c: sum(1 for b in bonds if c in b and all(syms[x] == "C" for x in b)) for c in cs}
    c2, c3 = [c for c in cs if deg[c] == 2]
    # atoms on the c3 side of the c2-c3 bond
    side, frontier = {c3}, [c3]
    while frontier:
        a = frontier.pop()
        for b in bonds:
            if a in b:
                o = b[0] if b[1] == a else b[1]
                if o not in side and o != c2:
                    side.add(o)
                    frontier.append(o)
    other = ref.copy()
    other.rotate(120, ref.positions[c3] - ref.positions[c2], center=ref.positions[c2])
    other.positions[sorted(set(range(len(ref))) - side)] = ref.positions[sorted(set(range(len(ref))) - side)]
    cfg = {"atom_map": {"C1": c2, "C2": c3}, "conformer": {"core": ["C1", "C2"]}}
    res = compare(ref, other, cfg, 0.0, 0.0)
    assert res["n_changed_dihedrals"] >= 1
    top = res["changed_dihedrals"][0]
    assert set(top["atoms"][1:3]) == {c2, c3}
    assert abs(abs(top["change_deg"]) - 120) < 1.0
    assert res["material"] is False and res["rmsd_whole_A"] > 0.3


def test_representative_dihedrals_skip_metal_bonds():
    from ase import Atoms

    a = Atoms("SnOCC", positions=[[0, 0, 0], [2.0, 0, 0], [2.9, 1.0, 0], [4.3, 1.0, 0.3]])
    for q in representative_dihedrals(a):
        assert 0 not in q[1:3]


# --------------------------------------------------------------------------- #
#  provenance policy
# --------------------------------------------------------------------------- #
def _real_cfg(**changes):
    cfg = load_config(PKG / "config_tian_tt_test.json")
    for k, v in changes.items():
        cfg["provenance"][k] = v
    return cfg


def test_provenance_block_required(toy):
    cfg = copy.deepcopy(toy)
    cfg["system"]["test_only"] = False
    rep = validate_pair(reactant(), product_minimum(), cfg)
    assert any("no 'provenance' block" in e for e in rep["pair"]["errors"])


@pytest.mark.parametrize("field,value,msg", [
    ("structure_class", "guess", "structure_class must be"),
    ("constructed_by", None, "need provenance.constructed_by"),
    ("charge_spin_status", None, "need provenance.charge_spin_status"),
    ("charge_spin_status", "assumed", "charge_spin_status must be"),
    ("not_literature_coordinates", False, "not_literature_coordinates"),
])
def test_reconstructed_provenance_fields(field, value, msg):
    cfg = _real_cfg(**{field: value})
    from config_io import resolve_path

    st = cfg["structures"]
    rep = validate_pair(resolve_path(cfg, st["reactant"]), resolve_path(cfg, st["product"]), cfg)
    assert rep["status"] == "FAIL"
    assert any(msg in e for e in rep["pair"]["errors"]), rep["pair"]["errors"]


def test_run_path_refuses_unoptimized_reconstructed_input(tmp_path, monkeypatch):
    import run_path

    cfg = load_config(PKG / "config_tian_tt_test.json")
    cfg["output_dir"] = str(tmp_path / "out")                 # no stage-1 endpoints there

    def boom(*a, **k):
        raise AssertionError("model must not be loaded")
    monkeypatch.setattr(run_path, "model_from_config", boom)
    with pytest.raises(ValidationError, match="must be relaxed"):
        run_path.run(cfg, allow_unoptimized=True)


# --------------------------------------------------------------------------- #
#  GFN2-xTB calculator path (no xtb needed)
# --------------------------------------------------------------------------- #
class FakeXTB:
    """Stand-in for xtb.ase.calculator.XTB: reads charge/uhf the way xtb-python
    does (sums of initial charges / magnetic moments)."""

    implemented_properties = ["energy", "forces"]

    def __init__(self):
        self.seen = None

    def reset(self):
        pass

    def get_potential_energy(self, atoms):
        self.seen = (float(atoms.get_initial_charges().sum()),
                     int(round(atoms.get_initial_magnetic_moments().sum())))
        return 10.0 * self.seen[0] + self.seen[1]

    def get_forces(self, atoms):
        return np.zeros((len(atoms), 3))

    def get_property(self, name, atoms, allow_calculation=True):
        return self.get_potential_energy(atoms) if name in ("energy", "free_energy") \
            else self.get_forces(atoms)

    def calculation_required(self, atoms, props):
        return True

    def check_state(self, atoms):
        return []


def test_xtb_registry_and_charge_spin_arrays(toy, monkeypatch):
    spec = C.MODELS["gfn2-xtb"]
    assert spec.charge_spin_mode == "initial_arrays" and 50 in spec.supported_z
    built = {}

    def fake_builder(variant, device, dtype, options):
        built.update(variant=variant, device=device, options=options)
        return FakeXTB()
    monkeypatch.setitem(C.MODELS, "gfn2-xtb", C.ModelSpec(
        "gfn2-xtb", spec.pretty, fake_builder, spec.variants, spec.default_variant,
        charge_spin_mode="initial_arrays", family="xtb", supported_z=spec.supported_z))
    cfg = copy.deepcopy(toy)
    cfg["system"]["spin_multiplicity"] = 1
    cfg["calculator"]["models"]["gfn2-xtb"] = {"variant": "gfn2",
                                              "options": {"electronic_temperature": 300.0}}
    h = C.model_from_config(cfg, model="gfn2-xtb")
    assert built["options"] == {"electronic_temperature": 300.0}
    a = reactant()
    a.calc = h.new_calc()
    assert a.get_potential_energy() == pytest.approx(10 * -1 + 0)
    assert h.base.seen == (-1.0, 0)                      # charge -1, 2S = mult - 1 = 0
    info = h.info()
    assert info["method"] == "GFN2-xTB" and info["charge_spin_mode"] == "initial_arrays"
    h.check_elements(reactant())                          # C, H, O within Z = 1-86
    b = reactant()
    b.set_initial_charges([1.0] + [0.0] * 8)              # sums to +1, conflicts with -1
    b.calc = h.new_calc()
    with pytest.raises(ValueError, match="initial charges"):
        b.get_potential_energy()
    # a doublet passes 2S = 1 to xtb
    d = C.ChargeSpinCalculator(FakeXTB(), 0, 2, mode="initial_arrays")
    c = reactant()
    c.calc = d
    c.get_potential_energy()
    assert d.base.seen == (0.0, 1)


def test_xtb_settings_guards(toy):
    with pytest.raises(ConfigError, match="float64"):
        C.parse_calc_settings(toy, model="gfn2-xtb", dtype="float32")
    with pytest.raises(ConfigError, match="CPU only"):
        C._build_xtb("gfn2", "cuda", "float64", {})
    s = C.parse_calc_settings(toy, model="gfn2-xtb")
    assert (s.variant, s.tag) == ("gfn2", "gfn2-xtb_gfn2")


def test_dependency_checks_give_clear_errors():
    for key in ("mace-omol", "mace-polar", "gfn2-xtb"):
        dep = C.MODELS[key].dependency_check
        assert dep is not None
        try:
            dep()
        except RuntimeError as e:
            assert "install" in str(e).lower()            # actionable message, not ImportError


# --------------------------------------------------------------------------- #
#  end-to-end protocol on the analytic toy (Sella needed)
# --------------------------------------------------------------------------- #
class ToyHandle:
    """Duck-typed ModelHandle around DoubleWellHydride (TEST-ONLY)."""

    def __init__(self, cfg):
        self.settings = C.parse_calc_settings(cfg)
        self._calcs = []

    def new_calc(self, correction=None):
        c = DoubleWellHydride(bias=-0.05)
        self._calcs.append(c)
        return c

    @property
    def total_calls(self):
        return sum(c.n_calls for c in self._calcs)

    def check_elements(self, atoms):
        pass

    def info(self):
        return {"model": "TEST-ONLY DoubleWellHydride"}


def test_protocol_end_to_end_on_toy(toy, tmp_path):
    """validate -> relax -> CI-NEB -> Sella -> Hessian -> +/- mode connectivity ->
    both reference states, on an analytic surface with a known saddle. Also
    checks that the NEB maximum is never labelled a certified TS."""
    pytest.importorskip("sella")
    import run_protocol

    cfg = copy.deepcopy(toy)
    write(str(tmp_path / "R.xyz"), reactant())
    write(str(tmp_path / "P.xyz"), product_minimum())
    cfg["structures"] = {"reactant": "R.xyz", "product": "P.xyz"}
    cfg["optimize"] = {"optimizer": "BFGS", "fmax": 0.01, "steps": 200}
    cfg["path"] = {"n_images": 5, "k": 0.5, "climb": True, "interpolation": "linear",
                   "optimizer": "FIRE", "fmax": 0.05, "steps": 300}
    cfg["sella"] = {"order": 1, "internal": False, "fmax": 1e-3, "steps": 200}
    cfg["connectivity"] = {"mode_displacement_A": 0.1, "fmax": 0.01, "steps": 500}
    s = run_protocol.run(cfg, handle=ToyHandle(cfg))
    neb = s["stages"]["run_path"]
    assert neb["highest_image"]["certified"] is False
    assert "NOT a verified transition state" in neb["highest_image"]["status"]
    assert "NOT a barrier" in neb["highest_image"]["E_highest_image_minus_E_band_image0"]["note"]
    assert s["stages"]["validate_ts"]["n_imaginary"] == 1
    assert s["ts_certification"].startswith("CERTIFIED"), s["ts_certification"]
    en = s["energetics"]
    assert set(en["barrier_forward"]) == {"relaxed_endpoint", "connected_side_minimum"}
    # bias -0.05 eV/Å * delta: barrier ~1 eV, product lower by ~2 * 0.05 * 0.86 eV
    bf = en["barrier_forward"]["relaxed_endpoint"]["eV"]
    assert 0.9 < bf < 1.1
    rx = en["reaction_energy"]["relaxed_endpoint"]["eV"]
    assert rx == pytest.approx(en["reaction_energy"]["connected_side_minimum"]["eV"], abs=1e-3)
    assert -0.1 < rx < -0.07
    assert "barrier" not in set(_keys(en))
    root = Path(cfg["_config_dir"]) / "outputs" / ToyHandle(cfg).settings.tag
    for f in ("vib/normal_modes.npz", "vib/imag_mode_vector.npy", "vib/validate_ts.json",
              "vib/connectivity/conn_plus_relaxed.xyz", "vib/connectivity/conn_minus_relaxed.xyz",
              "protocol_summary.json"):
        assert (root / f).exists(), f
    modes = np.load(root / "vib" / "normal_modes.npz")
    assert modes["frequencies_cm"].min() < -1000
    vec = np.load(root / "vib" / "imag_mode_vector.npy")
    assert vec.shape == (9, 3) and np.linalg.norm(vec) == pytest.approx(1.0)
    vt = json.loads((root / "vib" / "validate_ts.json").read_text())
    assert vt["ts_certification"] == s["ts_certification"]
