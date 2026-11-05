"""
xTB (and any-method) Sn-coordination failure detection, judged against the INPUT
structure of the same role. Uses the real reconstructed input xyz files with
geometric perturbations; no calculator is run.

    cd sweep/sn_beta_hydride && python -m pytest tests/test_coordination_check.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

PKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PKG)]

from config_io import atom_map, load_config           # noqa: E402
import coordination_check as CC                        # noqa: E402

CFG = load_config(PKG / "config_tian_tt_test.json")
M = atom_map(CFG, 42)
R0 = read(PKG / "input/tian_tt_test/reactant.xyz")
P0 = read(PKG / "input/tian_tt_test/product.xyz")


def test_reference_is_preserved_against_itself_including_stretched_product_sn_o2():
    assert CC.check_coordination(R0, R0, CFG, expected_hydride_state="reactant-like")["verdict"] == "PRESERVED"
    r = CC.check_coordination(P0, P0, CFG, expected_hydride_state="product-like")
    assert r["verdict"] == "PRESERVED"
    # the input product already has Sn-O2 beyond the 2.46 A bond criterion; the rule must be relative
    assert r["key_sn_o"]["O2"]["now_A"] > 2.46 and r["key_sn_o"]["O2"]["in_shell"]


def test_water_dissociation_is_structurally_invalid():
    a = R0.copy()
    sn, ow = M["Sn"], M["O_water"]
    v = a.positions[ow] - a.positions[sn]
    water = [ow] + [i for i in range(len(a)) if a.numbers[i] == 1 and np.linalg.norm(a.positions[i] - a.positions[ow]) < 1.2]
    a.positions[water] += 1.5 * v / np.linalg.norm(v)            # whole water leaves along Sn->O
    r = CC.check_coordination(a, R0, CFG, expected_hydride_state="reactant-like")
    assert r["verdict"] == "STRUCTURALLY_INVALID"
    assert r["lost_from_shell"] == [ow] and r["water"]["intact"] and not r["key_sn_o"]["O_water"]["in_shell"]


def test_water_losing_a_proton_is_invalid():
    a = R0.copy()
    ow = M["O_water"]
    h = next(i for i in range(len(a)) if a.numbers[i] == 1 and np.linalg.norm(a.positions[i] - a.positions[ow]) < 1.2)
    a.positions[h] += np.array([3.0, 0.0, 0.0])
    r = CC.check_coordination(a, R0, CFG)
    assert r["verdict"] == "STRUCTURALLY_INVALID" and not r["water"]["intact"]


def test_weakened_but_still_bound_is_a_warning_not_invalid():
    a = R0.copy()
    sn, o2 = M["Sn"], M["O2"]
    v = a.positions[o2] - a.positions[sn]
    a.positions[o2] += 0.35 * v / np.linalg.norm(v)              # 2.04 -> 2.39 A, still in the shell
    r = CC.check_coordination(a, R0, CFG)
    assert r["verdict"] == "WEAKENED" and o2 in r["lengthened_gt_threshold_A"]


def test_hydride_state_flip_is_invalid_for_expected_role():
    a = R0.copy()
    h, c1 = M["H_transfer"], M["C1"]
    a.positions[h] = a.positions[c1] + 1.10 * (P0.positions[h] - P0.positions[c1]) / np.linalg.norm(P0.positions[h] - P0.positions[c1])
    r = CC.check_coordination(a, R0, CFG, expected_hydride_state="reactant-like")
    assert r["hydride"]["state"] == "product-like" and r["verdict"] == "STRUCTURALLY_INVALID"
    assert any("hydride state" in x for x in r["reasons"])


def test_new_atom_entering_shell_is_invalid():
    a = R0.copy()
    sn = M["Sn"]
    far = next(i for i in range(len(a)) if a.numbers[i] > 1 and np.linalg.norm(a.positions[i] - a.positions[sn]) > 4.0)
    a.positions[far] = a.positions[sn] + np.array([0.0, 0.0, 2.5])
    r = CC.check_coordination(a, R0, CFG)
    assert r["verdict"] == "STRUCTURALLY_INVALID" and far in r["gained_in_shell"]


def test_water_rotation_about_sn_o_axis_is_not_a_coordination_change():
    """Regression: the first draft counted H atoms in the Sn shell and flagged an intact,
    merely rotated xTB product water as 'dissociated'."""
    a = R0.copy()
    sn, ow = M["Sn"], M["O_water"]
    axis = a.positions[ow] - a.positions[sn]; axis /= np.linalg.norm(axis)
    hs = [i for i in range(len(a)) if a.numbers[i] == 1 and np.linalg.norm(a.positions[i] - a.positions[ow]) < 1.2]
    from ase.build.tools import rotate  # noqa: F401  (ensures ase is the real package)
    for ang in np.linspace(0.0, np.pi, 13):
        b = a.copy()
        for h in hs:
            v = b.positions[h] - b.positions[ow]
            c, s_ = np.cos(ang), np.sin(ang)
            b.positions[h] = b.positions[ow] + v * c + np.cross(axis, v) * s_ + axis * (axis @ v) * (1 - c)
        r = CC.check_coordination(b, R0, CFG, expected_hydride_state="reactant-like")
        assert r["verdict"] == "PRESERVED" and r["water"]["intact"], (ang, r["reasons"])


def test_atom_order_mismatch_raises_and_thresholds_are_recorded():
    with pytest.raises(ValueError):
        CC.check_coordination(R0[::-1], R0, CFG)
    r = CC.check_coordination(R0, R0, CFG, shell_A=3.0, weaken_A=0.5)
    assert r["thresholds"]["shell_A"] == 3.0 and r["thresholds"]["weaken_A"] == 0.5


def test_si_o_cap_cleavage_is_invalid_even_though_si_is_outside_the_sn_shell():
    """Audit finding: the shell-only rule could not see a framework Si-O cleavage."""
    a = R0.copy()
    sym = a.get_chemical_symbols()
    si = next(i for i, x in enumerate(sym) if x == "Si")
    o_nb = min((i for i, x in enumerate(sym) if x == "O"), key=lambda i: np.linalg.norm(a.positions[i] - a.positions[si]))
    v = a.positions[si] - a.positions[o_nb]
    group = [si] + [i for i in range(len(a)) if a.numbers[i] == 1 and np.linalg.norm(a.positions[i] - a.positions[si]) < 1.7]
    a.positions[group] += 1.5 * v / np.linalg.norm(v)
    r = CC.check_coordination(a, R0, CFG, expected_hydride_state="reactant-like")
    assert r["verdict"] == "STRUCTURALLY_INVALID" and r["heavy_bonds_broken"], r["reasons"]


def test_hydride_landing_on_sn_is_invalid():
    a = R0.copy()
    sn, h = M["Sn"], M["H_transfer"]
    a.positions[h] = a.positions[sn] + np.array([0.0, 0.0, 1.8])
    r = CC.check_coordination(a, R0, CFG)
    assert r["verdict"] == "STRUCTURALLY_INVALID" and h in r["hydrogen_near_sn"]


def test_reference_methods_geometries_have_no_spurious_heavy_bond_flags():
    assert CC.check_coordination(R0, R0, CFG)["heavy_bonds_broken"] == []
    assert CC.check_coordination(P0, P0, CFG)["heavy_bonds_formed"] == []
