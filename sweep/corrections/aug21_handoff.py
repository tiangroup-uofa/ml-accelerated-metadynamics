#!/usr/bin/env python3
"""
aug21_handoff.py — Aug 21 xTB/Sella handoff-cost benchmark.

Question: given the existing Phase C corrected-MACE model, does *continuing*
Phase 2 iterative sampling reduce the additional xTB evaluations needed to reach
the verified xTB transition state, and which handoff iteration is optimal?

Runs entirely inside the `xtbenv` micromamba environment (ase + xtb-python +
sella 2.5.0, the same Sella version Phase 2 used):

    MAMBA_ROOT_PREFIX=~/.local/mm/root \
      ~/.local/bin/micromamba run -n xtbenv python aug21_handoff.py --smoke
    ... same command without --smoke for the full benchmark

COST ACCOUNTING (established by the repository audit)
  * The authoritative metric is the number of `Calculator.calculate()`
    invocations, counted inside the calculator. xtb-python returns energy AND
    forces from one calculate(), so one call == one SCF. Sella's `nsteps`
    undercounts by ~30% (unlogged finite-difference eigenvector steps) and is
    recorded only as a cross-check, never as the cost.
  * Candidate iterN/ts.xyz was produced BEFORE iteration N's own query packet,
    so its prior adaptive cost is 7*N, not 7*(N+1). Verified against the
    recorded n_train (106 + 7N) in phase2/iterNN/metrics.json.
  * The 106 Phase C training configs are a separate sunk/pretraining column and
    are never folded into the adaptive curve.
  * Hessian certification (2*3N = 96 calls) and the reactant reference energy
    are VERIFICATION, not part of reaching the TS. They are counted in their own
    column and excluded from the primary metric. Every case, baseline included,
    pays the same verification cost, so it is a constant offset.

Nothing here reads or writes anything under phase2/ — results go to
aug21_handoff/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")

from ase.constraints import FixBondLengths
from ase.io import read, write
from ase.optimize import FIRE
from xtb.ase.calculator import XTB

import sella

HERE = Path(__file__).resolve().parent
DA = HERE.parent / "diels_alder"
PHASE2 = HERE / "phase2"
OUT = HERE / "aug21_handoff"
CASES = OUT / "cases"

# ---- constants shared with phase2_lib.py (kept numerically identical) ------- #
EV2KCAL = 23.060548
FORMING = [(0, 5), (3, 4)]
LAMBDA_TO_CM = 521.47
HESS_DELTA = 0.001
IMAG_CUTOFF_CM = 50.0
TS_MIN_OVERLAP = 0.30
TS_FORMING_RANGE = (1.5, 3.0)

# ---- fixed configuration, identical for EVERY case ------------------------- #
XTB_KW = dict(method="GFN2-xTB", electronic_temperature=1000.0,
              max_iterations=500, accuracy=1.0)
SELLA_FMAX = 1e-4          # matches phase2_lib.sella_saddle
SELLA_STEPS = 500
SELLA_ORDER = 1
SELLA_INTERNAL = False     # Cartesian, matching Phase 2
SCAN_FMAX = 0.03           # matches phase2_reaction.py / phaseC_reaction.py
SCAN_STEPS = 300

PHASEC_SUNK_REFS = 106     # baseline 44 + barrier_extra 32 + water 30
PER_ITER_UNIQUE_REFS = 7


# --------------------------------------------------------------------------- #
#  counting calculator — the authoritative cost meter
# --------------------------------------------------------------------------- #
class CountingXTB(XTB):
    """GFN2-xTB that counts genuine calculate() invocations, by phase."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_total = 0
        self.phase = "unassigned"
        self.by_phase: dict[str, int] = {}

    def calculate(self, atoms=None, properties=None, system_changes=None):
        self.n_total += 1
        self.by_phase[self.phase] = self.by_phase.get(self.phase, 0) + 1
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)

    def set_phase(self, name):
        self.phase = name
        return self.by_phase.get(name, 0)

    def phase_delta(self, name, before):
        return self.by_phase.get(name, 0) - before


def new_calc():
    return CountingXTB(**XTB_KW)


# --------------------------------------------------------------------------- #
#  geometry / vibrational helpers (identical logic to phase2_lib.py)
# --------------------------------------------------------------------------- #
def forming_distances(atoms):
    p = atoms.get_positions()
    return [float(np.linalg.norm(p[i] - p[j])) for i, j in FORMING]


def forming_mean(atoms):
    return float(np.mean(forming_distances(atoms)))


def kabsch_rmsd(A, B):
    A = np.asarray(A, float) - np.asarray(A, float).mean(0)
    B = np.asarray(B, float) - np.asarray(B, float).mean(0)
    V, S, Wt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    return float(np.sqrt(np.mean(np.sum((A @ R - B) ** 2, axis=1))))


def distance_signature(atoms):
    """Permutation-invariant structural fingerprint: sorted row-sums of the
    sorted interatomic distance matrix. Kabsch RMSD alone is misleading here --
    a relabelled/mirrored copy of the same saddle scores ~0.9 A."""
    D = np.sort(atoms.get_all_distances(), axis=1)
    return np.sort(D.sum(1))


def signature_distance(a, b):
    return float(np.linalg.norm(distance_signature(a) - distance_signature(b)))


def reaction_coordinate_vector(atoms):
    p = atoms.get_positions()
    v = np.zeros_like(p)
    for i, j in FORMING:
        d = p[i] - p[j]
        u = d / np.linalg.norm(d)
        v[i] += u
        v[j] -= u
    return (v / np.linalg.norm(v)).ravel()


def hessian(atoms, calc, delta=HESS_DELTA):
    n = len(atoms)
    ndof = 3 * n
    H = np.zeros((ndof, ndof))
    for i in range(ndof):
        a = atoms.copy(); a.calc = calc
        a.positions.flat[i] += delta
        Ff = a.get_forces().ravel()
        b = atoms.copy(); b.calc = calc
        b.positions.flat[i] -= delta
        Fb = b.get_forces().ravel()
        H[i] = (Fb - Ff) / (2 * delta)
    return 0.5 * (H + H.T)


def vibrational_analysis(atoms, calc):
    H = hessian(atoms, calc)
    m = atoms.get_masses()
    w = 1.0 / np.sqrt(np.repeat(m, 3))
    lam, vecs = np.linalg.eigh(H * np.outer(w, w))
    nu = np.sign(lam) * LAMBDA_TO_CM * np.sqrt(np.abs(lam))
    imag_idx = [i for i in range(len(nu)) if nu[i] < -IMAG_CUTOFF_CM]
    rc = reaction_coordinate_vector(atoms)
    overlap = nu_imag = None
    if imag_idx:
        k = imag_idx[int(np.argmin([nu[i] for i in imag_idx]))]
        nu_imag = float(nu[k])
        cart = vecs[:, k] * w
        cart /= np.linalg.norm(cart)
        overlap = float(abs(cart @ rc))
    return {
        "n_imaginary": len(imag_idx),
        "imag_freqs_cm": [float(nu[i]) for i in imag_idx],
        "nu_imag_cm": nu_imag,
        "reaction_mode_overlap": overlap,
        "lowest_10_cm": [float(x) for x in np.sort(nu)[:10]],
    }


REF_NU_CM = -394.0          # verified xTB TS imaginary frequency
NU_TOL_CM = 15.0            # tolerance on the principal mode
SIG_TOL = 0.05              # permutation-invariant signature tolerance


def reached_reference_saddle(converged, vib, forming_cc_mean, sig_d):
    """Did this run land on the VERIFIED xTB transition state?

    Kept separate from the strict Phase 2 criterion because a finite-difference
    Hessian on xTB's soft reactant-side vdW modes intermittently produces extra
    modes just past the -50 cm-1 cutoff on structures that are otherwise
    identical (same forming distance, same barrier, same -393.4 cm-1 principal
    mode). This checks identity of the saddle; `certify` checks Hessian
    cleanliness. Both are reported.
    """
    reasons = []
    if not converged:
        reasons.append("not_converged")
    nu = vib["nu_imag_cm"]
    if nu is None or abs(nu - REF_NU_CM) > NU_TOL_CM:
        reasons.append(f"principal_mode={nu}")
    ov = vib["reaction_mode_overlap"]
    if ov is None or ov < TS_MIN_OVERLAP:
        reasons.append(f"reaction_mode_overlap={ov}")
    lo, hi = TS_FORMING_RANGE
    if not (lo <= forming_cc_mean <= hi):
        reasons.append(f"forming_CC={forming_cc_mean:.3f}_out_of_range")
    if sig_d > SIG_TOL:
        reasons.append(f"structure_signature_delta={sig_d:.4f}")
    return len(reasons) == 0, reasons


def certify(converged, vib, forming_cc_mean):
    """Same criteria as phase2_loop.validate_ts (strict Hessian cleanliness)."""
    reasons = []
    if not converged:
        reasons.append("not_converged")
    if vib["n_imaginary"] != 1:
        reasons.append(f"n_imaginary={vib['n_imaginary']}")
    ov = vib["reaction_mode_overlap"]
    if ov is None or ov < TS_MIN_OVERLAP:
        reasons.append(f"reaction_mode_overlap={ov}")
    lo, hi = TS_FORMING_RANGE
    if not (lo <= forming_cc_mean <= hi):
        reasons.append(f"forming_CC={forming_cc_mean:.3f}_out_of_range")
    return len(reasons) == 0, reasons


# --------------------------------------------------------------------------- #
#  one handoff / baseline Sella case
# --------------------------------------------------------------------------- #
def run_sella_case(name, start_atoms, ref_ts, E_reactant, calc,
                   scan_calls=0, meta=None):
    """Sella-on-xTB from `start_atoms`. Returns a fully populated record."""
    case_dir = CASES / name
    case_dir.mkdir(parents=True, exist_ok=True)
    # ASE opens optimizer logfiles in append mode; stale logs from an earlier
    # run would inflate the nsteps cross-check. Start each case clean.
    for stale in ("sella.log", "sella.traj"):
        (case_dir / stale).unlink(missing_ok=True)
    write(str(case_dir / "start.xyz"), start_atoms)

    at = start_atoms.copy()
    at.calc = calc

    before = calc.set_phase(f"{name}:sella")
    t0 = time.time()
    err = None
    try:
        converged = bool(sella.Sella(
            at,
            order=SELLA_ORDER,
            internal=SELLA_INTERNAL,
            trajectory=str(case_dir / "sella.traj"),
            logfile=str(case_dir / "sella.log"),
        ).run(fmax=SELLA_FMAX, steps=SELLA_STEPS))
    except Exception as e:                       # pragma: no cover
        converged = False
        err = repr(e)
    dyn_steps = sum(1 for ln in (case_dir / "sella.log").read_text().splitlines()
                    if ln.startswith("Sella")) - 1
    sella_calls = calc.phase_delta(f"{name}:sella", before)
    wall = time.time() - t0

    frames = len(read(str(case_dir / "sella.traj"), index=":"))
    F = at.get_forces()
    fmax_final = float(np.max(np.linalg.norm(F, axis=1)))
    E_ts = float(at.get_potential_energy())
    write(str(case_dir / "final.xyz"), at)

    # ---- verification (NOT charged to the handoff cost) -------------------- #
    vbefore = calc.set_phase(f"{name}:verify")
    vib = vibrational_analysis(at, calc)
    verify_calls = calc.phase_delta(f"{name}:verify", vbefore)
    calc.set_phase("idle")

    fmean = forming_mean(at)
    ok, reasons = certify(converged, vib, fmean)
    sig_d = signature_distance(at, ref_ts)
    reached, reach_reasons = reached_reference_saddle(converged, vib, fmean, sig_d)

    # Rejected Sella steps evaluate the calculator without writing a frame, so
    # frames is a LOWER BOUND on calls; the counter is authoritative.
    consistent = (frames <= sella_calls)

    rec = {
        "case": name,
        **(meta or {}),
        "start_forming_CC": forming_distances(start_atoms),
        "start_forming_CC_mean": forming_mean(start_atoms),
        "start_rmsd_to_xtb_ts": kabsch_rmsd(start_atoms.get_positions(),
                                            ref_ts.get_positions()),
        "handoff_xtb_calculate_calls": int(sella_calls),
        "baseline_scan_xtb_calls": int(scan_calls),
        "verification_xtb_calls": int(verify_calls),
        "sella_nsteps": int(dyn_steps),
        "trajectory_frames": int(frames),
        "counter_matches_frames": bool(consistent),
        "converged": bool(converged),
        "sella_error": err,
        "final_fmax": fmax_final,
        "energy_eV": E_ts,
        "barrier_kcal_mol": (E_ts - E_reactant) * EV2KCAL,
        "forming_CC": forming_distances(at),
        "forming_CC_mean": fmean,
        "rmsd_to_xtb_ts": kabsch_rmsd(at.get_positions(), ref_ts.get_positions()),
        "n_imaginary": vib["n_imaginary"],
        "imaginary_frequency": vib["nu_imag_cm"],
        "imag_freqs_cm": vib["imag_freqs_cm"],
        "reaction_mode_overlap": vib["reaction_mode_overlap"],
        "certified_ts": bool(ok),
        "certification_failures": reasons,
        "reached_reference_saddle": bool(reached),
        "reached_saddle_failures": reach_reasons,
        "structure_signature_delta": sig_d,
        "wall_seconds": round(wall, 1),
    }
    (case_dir / "record.json").write_text(json.dumps(rec, indent=2))

    flag = ("REACHED xTB TS" if reached else f"FAILED {reach_reasons}")
    flag += " | strict-Hessian:" + ("clean" if ok else f"{reasons}")
    print(f"  [{name}] xTB calls={sella_calls} (nsteps={dyn_steps}, frames={frames}"
          f"{'' if consistent else ' MISMATCH!'}) conv={converged} "
          f"fmax={fmax_final:.2e} forming={fmean:.4f} "
          f"RMSD={rec['rmsd_to_xtb_ts']:.4f} -> {flag}")
    return rec


def run_baseline_scan(calc):
    """Concerted relaxed scan on xTB, run as a GENUINE sequential scan.

    The cscan/f_*.xyz frames are already xTB-relaxed at exactly these
    constraints, so relaxing them again costs ~1 call each and would hand the
    baseline a free scan. Instead we walk the constraint down from the xTB
    reactant (start.xyz), each step seeded by the PREVIOUS relaxed geometry --
    which is what a relaxed scan actually is, and what produced those frames in
    the first place. Same 19 constraint values, same FixBondLengths+FIRE
    protocol as phase2_reaction.py; the cost is now real.
    """
    targets = [forming_mean(read(f))
               for f in sorted(glob.glob(str(DA / "cscan" / "f_*.xyz")))]
    before = calc.set_phase("baseline:scan")
    cur = read(str(DA / "start.xyz"))
    coord, E, relaxed = [], [], []
    for t in targets:
        at = cur.copy()
        # drive both forming bonds to the target separation, then relax the rest
        p = at.get_positions()
        for i, j in FORMING:
            d = p[i] - p[j]
            L = np.linalg.norm(d)
            shift = 0.5 * (t - L) * d / L
            p[i] += shift
            p[j] -= shift
        at.set_positions(p)
        at.calc = calc
        at.set_constraint(FixBondLengths([list(q) for q in FORMING]))
        FIRE(at, logfile=None).run(fmax=SCAN_FMAX, steps=SCAN_STEPS)
        at.set_constraint()
        coord.append(forming_mean(at))
        E.append(float(at.get_potential_energy()))
        relaxed.append(at)
        cur = at                      # sequential propagation
    n = calc.phase_delta("baseline:scan", before)
    calc.set_phase("idle")
    rel = (np.array(E) - E[0]) * EV2KCAL
    i = int(np.argmax(rel))
    print(f"  [baseline scan] {len(targets)} sequential constrained relaxations, "
          f"{n} xTB calls ({n/len(targets):.1f}/frame), "
          f"max {rel[i]:.2f} kcal/mol at {coord[i]:.3f} A")
    return relaxed[i], n, {"coord": list(map(float, coord)),
                           "rel_kcal": list(map(float, rel)),
                           "scan_max_index": i, "n_xtb_calls": int(n)}


def idpp_control_geometry():
    """Zero-xTB-cost interpolation candidate: IDPP band start->end, image whose
    forming C-C is nearest the known xTB TS value. Secondary control only —
    the audit showed interpolation is pathological for this reaction."""
    from ase.mep import NEB
    R, P = read(str(DA / "start.xyz")), read(str(DA / "end.xyz"))
    n = 11
    imgs = [R.copy() for _ in range(n - 1)] + [P.copy()]
    NEB(imgs).interpolate(method="idpp")
    target = 2.315
    k = int(np.argmin([abs(forming_mean(im) - target) for im in imgs]))
    return imgs[k], k, n


# --------------------------------------------------------------------------- #
def smoke_test():
    print("=== SMOKE TEST ===")
    at = read(str(DA / "ts_opt.xyz"))
    c = new_calc()
    at.calc = c

    c.set_phase("smoke:one_geom")
    E = at.get_potential_energy()
    F = at.get_forces()
    n1 = c.n_total
    print(f"  1. live xTB works: E={E:.6f} eV, fmax={np.abs(F).max():.4f} eV/Å")
    print(f"  2. E + F on ONE geometry => calculate() calls = {n1} "
          f"({'PASS' if n1 == 1 else 'FAIL — expected 1'})")

    at2 = at.copy(); at2.positions[0] += 0.01
    at2.calc = c
    at2.get_potential_energy()
    print(f"  3. new geometry increments counter: {c.n_total} "
          f"({'PASS' if c.n_total == 2 else 'FAIL'})")

    # displace off the saddle and use the production fmax so Sella really steps
    small = read(str(DA / "ts_opt.xyz"))
    rng = np.random.default_rng(0)
    small.positions += rng.normal(scale=0.03, size=small.positions.shape)
    cs = new_calc(); small.calc = cs
    cs.set_phase("smoke:sella")
    tmp = OUT / "_smoke"
    tmp.mkdir(parents=True, exist_ok=True)
    nsteps_req = 5
    dyn = sella.Sella(small, order=SELLA_ORDER, internal=SELLA_INTERNAL,
                      trajectory=str(tmp / "s.traj"), logfile=str(tmp / "s.log"))
    dyn.run(fmax=SELLA_FMAX, steps=nsteps_req)
    frames = len(read(str(tmp / "s.traj"), index=":"))
    logged = sum(1 for ln in (tmp / "s.log").read_text().splitlines()
                 if ln.startswith("Sella")) - 1
    print(f"  4. Sella 2.5.0 runs in xtbenv: {logged} logged steps -> "
          f"calls={cs.n_total}, frames={frames} "
          f"({'PASS' if cs.n_total == frames else 'FAIL — counter != frames'})")
    print(f"  5. calls({cs.n_total}) > logged nsteps({logged}): "
          f"{'PASS' if cs.n_total > logged else 'FAIL'}  "
          f"[confirms nsteps undercounts the true cost]")
    ok = (n1 == 1 and cs.n_total == frames and cs.n_total > logged)
    print(f"\n  SMOKE TEST: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="smoke test only")
    ap.add_argument("--only", default=None, help="comma list of case names")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    CASES.mkdir(exist_ok=True)

    if a.smoke:
        return 0 if smoke_test() else 1

    ref_ts = read(str(DA / "ts_opt.xyz"))
    calc = new_calc()

    # shared reference energy (verification cost, charged to no case)
    r = read(str(DA / "start.xyz")); r.calc = calc
    calc.set_phase("reference")
    E_reactant = float(r.get_potential_energy())
    calc.set_phase("idle")
    print(f"[ref] E(xTB reactant, start.xyz) = {E_reactant:.6f} eV\n")

    want = set(a.only.split(",")) if a.only else None
    records = []

    # ---- 1. xTB-only baseline: scan -> scan max -> Sella ------------------- #
    if want is None or "baseline_scan_sella" in want:
        print("[baseline] concerted relaxed scan on xTB (established procedure)")
        scan_max, scan_calls, profile = run_baseline_scan(calc)
        write(str(OUT / "baseline_scan_max.xyz"), scan_max)
        (OUT / "baseline_scan_profile.json").write_text(json.dumps(profile, indent=2))
        records.append(run_sella_case(
            "baseline_scan_sella", scan_max, ref_ts, E_reactant, calc,
            scan_calls=scan_calls,
            meta={"handoff_iteration": None, "start_type": "xtb_scan_maximum",
                  "starting_geometry": "aug21_handoff/baseline_scan_max.xyz",
                  "role": "primary_baseline",
                  "prior_phase2_unique_refs": 0,
                  "phaseC_sunk_refs": 0}))

    # ---- 2. handoff cases from Phase 2 candidates -------------------------- #
    start_types = {0: "corrected_MACE_sella_saddle", 1: "corrected_MACE_sella_saddle",
                   2: "corrected_MACE_sella_saddle", 3: "scan_max_fallback",
                   4: "corrected_MACE_sella_saddle", 5: "scan_max_fallback"}
    for n in range(6):
        name = f"handoff_iter{n:02d}"
        if want is not None and name not in want:
            continue
        src = PHASE2 / f"iter{n:02d}" / "ts.xyz"
        records.append(run_sella_case(
            name, read(str(src)), ref_ts, E_reactant, calc,
            meta={"handoff_iteration": n, "start_type": start_types[n],
                  "starting_geometry": str(src.relative_to(HERE.parent.parent)),
                  "role": "handoff",
                  "prior_phase2_unique_refs": PER_ITER_UNIQUE_REFS * n,
                  "phaseC_sunk_refs": PHASEC_SUNK_REFS}))

    # ---- 3. literal interpolation control (Tian's sketch, zero xTB cost) --- #
    if want is None or "control_idpp_interp" in want:
        geo, k, nimg = idpp_control_geometry()
        write(str(OUT / "control_idpp_start.xyz"), geo)
        print(f"[control] IDPP image {k}/{nimg-1}, forming={forming_mean(geo):.3f} Å "
              f"(zero xTB cost to generate)")
        records.append(run_sella_case(
            "control_idpp_interp", geo, ref_ts, E_reactant, calc,
            meta={"handoff_iteration": None, "start_type": "idpp_interpolation",
                  "starting_geometry": "aug21_handoff/control_idpp_start.xyz",
                  "role": "secondary_control",
                  "idpp_image_index": k, "idpp_n_images": nimg,
                  "prior_phase2_unique_refs": 0,
                  "phaseC_sunk_refs": 0}))

    # ---- totals ------------------------------------------------------------ #
    for rec in records:
        rec["total_marginal_xtb_calls"] = (
            rec["prior_phase2_unique_refs"]
            + rec["baseline_scan_xtb_calls"]
            + rec["handoff_xtb_calculate_calls"]
        )

    (OUT / "results.json").write_text(json.dumps({
        "config": {
            "xtb": XTB_KW, "sella_fmax": SELLA_FMAX, "sella_steps": SELLA_STEPS,
            "sella_order": SELLA_ORDER, "sella_internal": SELLA_INTERNAL,
            "scan_fmax": SCAN_FMAX, "hessian_delta": HESS_DELTA,
            "phaseC_sunk_refs": PHASEC_SUNK_REFS,
            "per_iter_unique_refs": PER_ITER_UNIQUE_REFS,
        },
        "totals": {"xtb_calculate_calls_all_phases": calc.n_total,
                   "by_phase": calc.by_phase},
        "records": records,
    }, indent=2))
    print(f"\nwrote {OUT/'results.json'}  |  total xTB calls this session: {calc.n_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
