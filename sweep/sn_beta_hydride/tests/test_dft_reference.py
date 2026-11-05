"""
Tests for the Gaussian ingestion / DFT-vs-MACE analysis / sparse-budget framework.

ALL FIXTURES ARE SYNTHETIC: Gaussian-format text written by `write_log` below
from my knowledge of the g09 layout, and made-up potentials. They test the code
paths and the failure modes. They are NOT Gaussian data and prove nothing about
the real Sn-beta reference set, which has not been delivered.

    cd sweep/sn_beta_hydride && python -m pytest tests/test_dft_reference.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path[:0] = [str(PKG), str(HERE)]

from dft_reference import dft_compare as D          # noqa: E402
from dft_reference import gaussian_parser as G      # noqa: E402
from dft_reference import sparse_budget as S        # noqa: E402

NUMS = np.array([6, 6, 1, 8, 50, 8])


# ------------------------------------------------------------ synthetic logs
def _orient_block(title, nums, xyz):
    s = f"                          {title}\n"
    s += " ---------------------------------------------------------------------\n"
    s += " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
    s += " Number     Number       Type             X           Y           Z\n"
    s += " ---------------------------------------------------------------------\n"
    for i, (z, p) in enumerate(zip(nums, xyz), 1):
        s += f"{i:7d}{z:11d}{0:12d}{p[0]:16.6f}{p[1]:12.6f}{p[2]:12.6f}\n"
    s += " ---------------------------------------------------------------------\n"
    return s


def write_log(path, steps, *, route="# b3lyp/def2tzvp force nosymm", charge=0, mult=1,
              terminate=True, opt_done=None, freqs=None, standard_only=False,
              truncate_in_forces=False, energy_line=True, error=False):
    """steps: list of (xyz[Å], E[Eh], F[Eh/Bohr])"""
    t = " Entering Gaussian System, Link 0=g09\n Gaussian 09:  ES64L-G09RevD.01 24-Apr-2013\n"
    t += f" ----------------------------------------------------------------------\n {route}\n"
    t += " ----------------------------------------------------------------------\n"
    t += f" Charge = {charge:2d} Multiplicity = {mult}\n NAtoms=     {len(NUMS)}\n"
    for xyz, e, f in steps:
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)       # standard frame != input frame
        if standard_only:
            t += _orient_block("Standard orientation:", NUMS, xyz @ rot.T)
        else:
            t += _orient_block("Input orientation:", NUMS, xyz)
            t += _orient_block("Standard orientation:", NUMS, xyz @ rot.T)
        if energy_line:
            t += f" SCF Done:  E(RB3LYP) =  {e:.9f}     A.U. after   10 cycles\n"
        t += " ***** Axes restored to original set *****\n"
        t += " -------------------------------------------------------------------\n"
        t += " Center     Atomic                   Forces (Hartrees/Bohr)\n"
        t += " Number     Number              X              Y              Z\n"
        t += " -------------------------------------------------------------------\n"
        rows = list(enumerate(f, 1))
        if truncate_in_forces:
            rows = rows[:3]
        for i, row in rows:
            t += f"{i:7d}{NUMS[i - 1]:9d}{row[0]:15.9f}{row[1]:15.9f}{row[2]:15.9f}\n"
        if truncate_in_forces:
            Path(path).write_text(t); return
        t += " -------------------------------------------------------------------\n"
    if opt_done:
        t += "         -- Stationary point found.\n"
    if freqs:
        t += " Frequencies --  " + "  ".join(f"{x:9.4f}" for x in freqs) + "\n"
    if error:
        t += " Error termination via Lnk1e in /g09/l502.exe\n"
    elif terminate:
        t += " Normal termination of Gaussian 09 at Mon Oct 19 12:00:00 2026.\n"
    Path(path).write_text(t)


def _xyz(rng, scale=1.0):
    return rng.normal(size=(len(NUMS), 3)) * scale


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ------------------------------------------------------------------- parser
def test_units_and_roundtrip(tmp_path, rng):
    xyz, F = _xyz(rng), rng.normal(size=(6, 3)) * 0.01
    write_log(tmp_path / "a.log", [(xyz, -123.456789, F)])
    j = G.parse_gaussian_log(tmp_path / "a.log")
    assert j.version == "G09" and j.charge == 0 and j.multiplicity == 1 and j.natoms == 6
    s = j.final
    assert np.allclose(s.positions, xyz, atol=1e-5)                    # INPUT frame, not the rotated standard frame
    assert s.orientation == "input"
    assert np.allclose(s.forces_eV_A, F * 51.4220865, rtol=1e-6)
    assert s.energy_eV == pytest.approx(-123.456789 * 27.211386246, rel=1e-12)
    assert G.HARTREE_BOHR_TO_EV_ANG == pytest.approx(51.4220675, rel=1e-5)


def test_force_sign_is_force_not_gradient(tmp_path):
    """Log forces are -dE/dx. Harmonic E = k/2 sum x^2 -> F = -k x; parse must keep that sign."""
    x = np.array([[0.5, 0, 0]] * 6, float)
    k_eh_per_bohr2 = 0.3
    xb = x / G.BOHR_TO_ANG
    F = -k_eh_per_bohr2 * xb                                           # Eh/Bohr
    write_log(tmp_path / "h.log", [(x, 0.5 * k_eh_per_bohr2 * (xb ** 2).sum(), F)])
    s = G.parse_gaussian_log(tmp_path / "h.log").final
    num_grad = np.zeros_like(x)
    h = 1e-4
    for i in range(6):
        for c in range(3):
            xp, xm = x.copy(), x.copy(); xp[i, c] += h; xm[i, c] -= h
            ep = 0.5 * k_eh_per_bohr2 * ((xp / G.BOHR_TO_ANG) ** 2).sum() * G.HARTREE_TO_EV
            em = 0.5 * k_eh_per_bohr2 * ((xm / G.BOHR_TO_ANG) ** 2).sum() * G.HARTREE_TO_EV
            num_grad[i, c] = (ep - em) / (2 * h)
    assert np.allclose(s.forces_eV_A, -num_grad, rtol=1e-4)
    assert np.allclose(G.gradient_to_force(num_grad), s.forces_eV_A, rtol=1e-4)   # fchk-style conversion


@pytest.mark.parametrize("kw,msg", [
    ({"terminate": False}, "Normal termination"),
    ({"error": True}, "error"),
    ({"truncate_in_forces": True}, "truncated|not terminated"),
    ({"energy_line": False}, "SCF Done|no complete"),
    ({"standard_only": True, "route": "# b3lyp/def2tzvp force"}, "Standard orientation"),
    ({"route": "# mp2/def2tzvp force nosymm"}, "only SCF"),
])
def test_incomplete_or_unsupported_jobs_fail(tmp_path, rng, kw, msg):
    write_log(tmp_path / "bad.log", [(_xyz(rng), -1.0, rng.normal(size=(6, 3)) * .01)], **kw)
    with pytest.raises(G.GaussianParseError, match="(?i)" + msg):
        G.parse_gaussian_log(tmp_path / "bad.log")


def test_standard_orientation_allowed_with_nosymm(tmp_path, rng):
    write_log(tmp_path / "s.log", [(_xyz(rng), -1.0, rng.normal(size=(6, 3)) * .01)], standard_only=True)
    assert G.parse_gaussian_log(tmp_path / "s.log").final.orientation == "standard"


def test_unconverged_optimisation_fails(tmp_path, rng):
    steps = [(_xyz(rng), -1.0 - i * 1e-3, rng.normal(size=(6, 3)) * .01) for i in range(3)]
    write_log(tmp_path / "o.log", steps, route="# b3lyp/def2tzvp opt nosymm", opt_done=False)
    with pytest.raises(G.GaussianParseError, match="Stationary point"):
        G.parse_gaussian_log(tmp_path / "o.log")
    write_log(tmp_path / "o2.log", steps, route="# b3lyp/def2tzvp opt nosymm", opt_done=True)
    assert len(G.parse_gaussian_log(tmp_path / "o2.log").steps) == 3


def test_metadata_and_warnings(tmp_path, rng):
    write_log(tmp_path / "m.log", [(_xyz(rng), -1.0, rng.normal(size=(6, 3)) * .01)],
              route="# b3lyp/genecp empiricaldispersion=gd3 scrf=(smd,solvent=water) force nosymm",
              charge=-1, mult=2, freqs=[-923.1, 100.0, 200.0])
    j = G.parse_gaussian_log(tmp_path / "m.log")
    assert (j.charge, j.multiplicity) == (-1, 2) and j.method.lower() == "b3lyp" and j.basis_is_gen
    assert j.n_imaginary == 1
    w = " ".join(j.warnings).lower()
    assert "dispersion" in w and "ecp" in w and "solvent" in w


def test_repo_xtb_g98_artifacts_are_rejected():
    """sweep/diels_alder/*/g98.out are xtb's Gaussian-98-style *frequency export*:
    no SCF energies, forces or termination line. They must not parse as DFT data."""
    p = PKG.parent / "diels_alder" / "ts_verify" / "g98.out"
    if not p.exists():
        pytest.skip("artifact not present")
    with pytest.raises(G.GaussianParseError):
        G.parse_gaussian_log(p)


# ------------------------------------------------------- manifest / loading
def _toy_path(tmp_path, rng, n=7, with_ts_freq=True):
    """R (point 0) -> TS (middle) -> P: smooth synthetic profile, 2 extra 'opt steps' per point."""
    base = _xyz(rng)
    d = rng.normal(size=(6, 3))
    ents = []
    for k in range(n):
        t = k / (n - 1)
        xyz = base + t * d
        e0 = -100.0 + 0.5 * np.sin(np.pi * t) - 0.2 * t               # Eh, synthetic barrier
        steps = [(xyz + 0.01 * (2 - i) * rng.normal(size=(6, 3)), e0 + 1e-4 * (2 - i), rng.normal(size=(6, 3)) * .02)
                 for i in range(3)]
        role = "reactant" if k == 0 else "product" if k == n - 1 else "ts" if k == n // 2 else "path"
        write_log(tmp_path / f"p{k}.log", steps, route="# b3lyp/def2tzvp opt nosymm", opt_done=True,
                  freqs=[-900.0, 50.0, 80.0] if (role == "ts" and with_ts_freq) else None)
        ents.append({"file": f"p{k}.log", "role": role, "pathway": "A", "point": k})
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"system": {"charge": 0, "multiplicity": 1}, "basis_note": "SYNTHETIC", "structures": ents}))
    return tmp_path


def test_load_reference_set_groups_and_final(tmp_path, rng):
    d = _toy_path(tmp_path, rng)
    recs, man, jobs = D.load_reference_set(d, expect_charge=0, expect_multiplicity=1)
    assert len(recs) == 7 * 3 and len(D.finals(recs)) == 7
    assert {r.group for r in recs} == {f"A:{k}" for k in range(7)}
    s = D.path_coordinate(D.finals(recs))
    assert s["A:0"] == 0 and s["A:6"] == pytest.approx(1.0)


def test_charge_spin_mismatch_is_loud(tmp_path, rng):
    d = _toy_path(tmp_path, rng)
    with pytest.raises(D.ReferenceSetError, match="charge"):
        D.load_reference_set(d, expect_charge=-1)
    with pytest.raises(D.ReferenceSetError, match="multiplicity"):
        D.load_reference_set(d, expect_multiplicity=3)
    m = json.loads((d / "manifest.json").read_text()); m["system"]["charge"] = 1
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(D.ReferenceSetError, match="Gaussian charge/mult"):
        D.load_reference_set(d)


def test_missing_manifest_and_bad_role(tmp_path, rng):
    with pytest.raises(D.ReferenceSetError, match="manifest not found"):
        D.load_reference_set(tmp_path)
    d = _toy_path(tmp_path, rng)
    m = json.loads((d / "manifest.json").read_text()); m["structures"][1]["role"] = "saddle?"
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(D.ReferenceSetError, match="unknown role"):
        D.load_reference_set(d)


# -------------------------------------------------------------- metrics
def test_cosine_similarity():
    a = np.arange(12.).reshape(4, 3) + 1
    assert D.cosine(a, a) == pytest.approx(1.0)
    assert D.cosine(a, -a) == pytest.approx(-1.0)
    assert D.cosine(a, 3.7 * a) == pytest.approx(1.0)
    assert D.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert np.isnan(D.cosine(np.zeros(3), np.ones(3)))


def test_force_metrics_subsets_and_ratio():
    f = np.random.default_rng(1).normal(size=(6, 3))
    m = D.force_metrics(f, 0.8 * f, {"core": [0, 1], "sn": [4]})
    assert m["cos"] == pytest.approx(1.0) and m["norm_ratio"] == pytest.approx(0.8)
    assert m["core_cos"] == pytest.approx(1.0) and m["sn_rmse"] > 0
    assert m["per_atom_err"].shape == (6,)
    with pytest.raises(D.ReferenceSetError):
        D.force_metrics(f, f[:5])


def test_energy_alignment_removes_constant_offset_only():
    d = np.array([0., 0.5, 0.2]); m = d + 1234.5                       # MACE vs DFT differ by an absolute constant
    dr, mr, mode = D.align_energies(d, m, 0)
    assert mode == "reference_point" and np.allclose(dr, mr)
    dr2, mr2, _ = D.align_energies(d, m * 1.1 + 5, mode="mean_offset")
    assert D.energy_metrics(dr2, mr2)["rmse_kJ_mol"] > 0               # a real slope error is NOT hidden
    with pytest.raises(D.ReferenceSetError):
        D.align_energies(d, m, None, "reference_point")


def _fake_eval(scale=1.0, shift=0.0):
    def f(numbers, positions):
        return 0.0, np.zeros_like(positions)
    return f


def _evaluated(recs, e_scale=1.0, f_scale=1.0, e_shift=0.0, noise=0.0, seed=3):
    r = np.random.default_rng(seed)
    return {x.rid: (e_scale * x.e_dft_eV + e_shift + noise * r.normal(), f_scale * x.f_dft + noise * r.normal(size=x.f_dft.shape))
            for x in recs}


def test_analysis_table_and_barrier_labels(tmp_path, rng):
    recs, *_ = D.load_reference_set(_toy_path(tmp_path, rng))
    ev = _evaluated(recs, e_scale=1.05, f_scale=0.9)
    rows, summ = D.analyse(recs, ev, subsets={"core": [0, 1]})
    assert summ["alignment"] == "reference_point" and len(rows) == 21
    b = summ["barrier_comparison"]
    assert b["reference_state"] == "dft_geometry_single_point"          # no unlabeled barrier
    assert b["dft_ts_status"] == "DFT_TS_ONE_IMAGINARY_MODE_MODE_IDENTITY_UNCHECKED"
    assert b["barrier_forward"]["error_kJ_mol"] == pytest.approx(
        b["barrier_forward"]["mace_kJ_mol"] - b["barrier_forward"]["dft_kJ_mol"])
    assert all(abs(r["cos"] - 1) < 1e-9 and abs(r["norm_ratio"] - 0.9) < 1e-9 for r in rows)
    png = D.plot_path_diagnostics({"synthetic": rows}, tmp_path / "p.png")
    assert Path(png).stat().st_size > 1000


def test_ts_without_dft_frequency_is_unverified(tmp_path, rng):
    recs, *_ = D.load_reference_set(_toy_path(tmp_path, rng, with_ts_freq=False))
    _, summ = D.analyse(recs, _evaluated(recs))
    assert summ["barrier_comparison"]["dft_ts_status"] == "DFT_TS_UNVERIFIED"


def test_path_maximum_is_never_promoted_to_ts(tmp_path, rng):
    d = _toy_path(tmp_path, rng)
    m = json.loads((d / "manifest.json").read_text())
    for e in m["structures"]:
        if e["role"] == "ts":
            e["role"] = "path"                                            # no TS declared -> path max exists but must not be used
    (d / "manifest.json").write_text(json.dumps(m))
    recs, *_ = D.load_reference_set(d)
    _, summ = D.analyse(recs, _evaluated(recs))
    b = summ["barrier_comparison"]
    assert "unavailable" in b and "ts" in b["unavailable"]
    assert b["reference_state"] == "dft_geometry_single_point"            # even the 'unavailable' record is labelled
    assert "barrier" not in summ                                         # no bare top-level barrier key


def test_atom_order_mismatch_to_model_is_refused(tmp_path, rng):
    recs, *_ = D.load_reference_set(_toy_path(tmp_path, rng))
    with pytest.raises(D.ReferenceSetError, match="ordering"):
        D.evaluate_records(recs, lambda n, p: (0.0, np.zeros_like(p)), expect_numbers=NUMS[::-1])


# --------------------------------------------------------- sparse budget
@pytest.fixture
def pathset(tmp_path, rng):
    recs, *_ = D.load_reference_set(_toy_path(tmp_path, rng, n=9))
    return recs


def test_subset_bookkeeping(pathset):
    e = S.endpoints_only(pathset)
    assert e.n_labels == 2 and set(e.train_groups) == {"A:0", "A:8"} and e.label_s == (0.0, 1.0)
    u = S.uniform(pathset, 5)
    assert u.n_labels == 5 and "A:0" in u.train_groups and "A:8" in u.train_groups
    assert len(set(u.train_groups)) == 5
    ts = S.ts_enriched(pathset, 3, ts_center_s=0.5)
    assert ts.n_labels == 5 and set(ts.train_groups[:2]) == {"A:0", "A:8"}
    s_of = D.path_coordinate(D.finals(pathset))
    interior = [g for g in ts.train_groups if g not in ("A:0", "A:8")]
    others = [g for g in s_of if g not in ts.train_groups and g not in ("A:0", "A:8")]
    assert max(abs(s_of[g] - .5) for g in interior) <= min(abs(s_of[g] - .5) for g in others) + 1e-12
    all_steps = S.endpoints_only(pathset, labels="all")
    assert all_steps.n_labels == 6                                       # 3 optimisation steps per group counted as 3 DFT calcs
    with pytest.raises(D.ReferenceSetError):
        S.ts_enriched(pathset, 50, 0.5)


def test_no_train_eval_leakage(pathset):
    sub = S.endpoints_only(pathset, labels="all")
    ev = _evaluated(pathset, e_scale=1.1, f_scale=1.1)
    res = S.evaluate_subset("affine", sub, pathset, ev)
    assert res["n_heldout_records"] == 7 * 3
    held_groups = {r.group for r in pathset if r.group not in sub.train_groups}
    assert held_groups.isdisjoint(sub.train_groups)
    group_of = {r.rid: r.group for r in pathset}
    other_step_same_group = next(r.rid for r in pathset if r.group == "A:0" and r.rid not in sub.train_rids[:1])
    with pytest.raises(S.LeakageError, match="straddle"):
        S.assert_no_leakage(sub.train_rids[:1], [other_step_same_group], group_of)
    with pytest.raises(S.LeakageError, match="both train and eval"):
        S.assert_no_leakage(sub.train_rids, sub.train_rids[:1], group_of)


def test_all_groups_labelled_has_no_heldout(pathset):
    allg = S._mk("all", "x", pathset, sorted({r.group for r in pathset}))
    with pytest.raises(D.ReferenceSetError, match="held-out"):
        S.evaluate_subset("none", allg, pathset, _evaluated(pathset))


def test_correction_recovers_linear_error_and_baseline_does_not(pathset):
    ev = _evaluated(pathset, e_scale=1.25, f_scale=1.25)                 # MACE = 1.25 x DFT (synthetic)
    sub = S.endpoints_only(pathset)
    base = S.evaluate_subset("none", sub, pathset, ev)
    aff = S.evaluate_subset("affine", sub, pathset, ev)
    assert abs(base["barrier_error_kJ_mol"]) > 1.0
    assert abs(aff["barrier_error_kJ_mol"]) < 1e-6 < abs(base["barrier_error_kJ_mol"])
    assert aff["heldout_force_cos"] == pytest.approx(1.0) and aff["heldout_force_rmse_eV_A"] < 1e-9
    assert aff["ts_in_training"] is False and aff["reactant_in_training"] is True


def test_every_budget_result_is_reference_state_labelled(pathset, tmp_path):
    ev = _evaluated(pathset, e_scale=1.1, f_scale=1.1, noise=0.01)
    subs = [S.endpoints_only(pathset), S.uniform(pathset, 4), S.ts_enriched(pathset, 2, 0.5)]
    res = S.run_budget_experiment(pathset, ev, subs, kinds=("none", "global", "affine", "element"))
    assert len(res) == 3 * 4
    for r in res:
        assert r["reference_state"] == "dft_geometry_single_point"
        assert {"n_labels", "label_groups", "label_s", "barrier_error_kJ_mol", "ts_in_training"} <= set(r)
    for p in S.plot_budget(res, str(tmp_path / "budget")):
        assert Path(p).stat().st_size > 1000


def test_ts_center_is_label_free_and_not_a_barrier(pathset):
    ev = _evaluated(pathset)
    s = S.mace_max_energy_location(D.finals(pathset), {r.rid: ev[r.rid][0] for r in pathset})
    assert 0.0 <= s <= 1.0
    import inspect
    src = inspect.getsource(S.mace_max_energy_location)
    assert "e_dft" not in src and "f_dft" not in src                    # selection never touches DFT labels


# ---- audit-driven additions -------------------------------------------------
def test_budget_flags_when_barrier_is_not_heldout(pathset):
    ev = _evaluated(pathset, e_scale=1.1, f_scale=1.1)
    ts_group = next(r.group for r in D.finals(pathset) if r.role == "ts")
    center = D.path_coordinate(D.finals(pathset))[ts_group]
    held = S.evaluate_subset("affine", S.endpoints_only(pathset), pathset, ev)
    enriched = S.evaluate_subset("affine", S.ts_enriched(pathset, 1, center), pathset, ev)
    assert held["barrier_error_is_heldout_wrt_ts"] is True and enriched["barrier_error_is_heldout_wrt_ts"] is False
    assert enriched["ts_in_training"] is True and held["product_in_training"] is True and held["reactant_in_training"] is True
    assert held["reaction_energy_error_kJ_mol"] is not None
    assert S.evaluate_subset("element", S.endpoints_only(pathset), pathset, ev)["energy_is_corrected"] is False


def test_affine_ties_energy_slope_to_force_slope_so_mixed_errors_remain():
    """Honest limit: Affine takes alpha from FORCES and applies it to energies, so a MACE whose energies are
    off by 1.25 but forces by 0.8 is NOT fixed. The synthetic test above (equal scales) cannot show this."""
    import numpy as np
    rng = np.random.default_rng(0)
    base = rng.normal(size=(6, 3))
    recs = []
    for k in range(7):
        t = k / 6
        recs.append(D.Record(f"r{k}", f"A:{k}", "A", k, "reactant" if k == 0 else "product" if k == 6 else "ts" if k == 3 else "path",
                             0, True, NUMS, base + t * np.ones((6, 3)), -100.0 + 0.5 * np.sin(np.pi * t), rng.normal(size=(6, 3))))
    ev = {r.rid: (1.25 * r.e_dft_eV, 0.8 * r.f_dft) for r in recs}
    res = S.evaluate_subset("affine", S.endpoints_only(recs), recs, ev)
    assert abs(res["barrier_error_kJ_mol"]) > 1.0


def test_parser_rejects_scan_jobs_and_unacknowledged_settings(tmp_path, rng):
    steps = [(_xyz(rng), -1.0, rng.normal(size=(6, 3)) * .01)]
    write_log(tmp_path / "scan.log", steps, route="# b3lyp/def2tzvp opt=modredundant nosymm", opt_done=True)
    with pytest.raises(G.GaussianParseError, match="(?i)scan/IRC|modredundant"):
        G.parse_gaussian_log(tmp_path / "scan.log")
    d = _toy_path(tmp_path / "ds", rng) if (tmp_path / "ds").mkdir() is None else None
    m = json.loads((d / "manifest.json").read_text())
    txt = (d / "p0.log").read_text().replace("# b3lyp/def2tzvp opt nosymm", "# b3lyp/def2tzvp opt nosymm scrf=(smd,solvent=water)")
    (d / "p0.log").write_text(txt)
    with pytest.raises(D.ReferenceSetError, match="unacknowledged"):
        D.load_reference_set(d)
    m["acknowledged_warnings"] = ["solvent"]
    (d / "manifest.json").write_text(json.dumps(m))
    D.load_reference_set(d)                                              # accepted once the decision is recorded


def test_parser_fails_when_a_step_is_lost(tmp_path, rng):
    """Two different geometry blocks with no energy/forces between them must raise, not drop the first."""
    good = tmp_path / "g.log"
    write_log(good, [(_xyz(rng), -1.0, rng.normal(size=(6, 3)) * .01)])
    txt = good.read_text()
    blk = txt[txt.index("                          Input orientation:"):txt.index(" SCF Done")]
    other = blk.replace("0.", "1.", 3)
    (tmp_path / "lost.log").write_text(txt.replace(blk, blk + other))
    with pytest.raises(G.GaussianParseError, match="(?i)lost|no energy"):
        G.parse_gaussian_log(tmp_path / "lost.log")
