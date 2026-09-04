#!/usr/bin/env python3
"""
aug21_diagnostics.py — Part A/B diagnostics for the Aug 21 handoff benchmark.

A. Reproducibility: replicate three representative cases 3x each under identical
   settings (iter01 cheap-success, iter02 non-converged, iter03 wrong-basin).
B. Starting-state descriptors for all seven primary starting geometries.

Reuses aug21_handoff.py wholesale (same CountingXTB, same Sella config, same
vibrational analysis) so nothing about the measurement changes. Reads Phase 2
metrics for descriptors that already exist; only computes what is genuinely
missing. Writes to aug21_diagnostics/ — never touches phase2/ or aug21_handoff/.

    MAMBA_ROOT_PREFIX=~/.local/mm/root \
      ~/.local/bin/micromamba run -n xtbenv python aug21_diagnostics.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aug21_handoff as A  # noqa: E402

OUT = HERE / "aug21_diagnostics"
REPL = OUT / "replicates"
DA = A.DA
PHASE2 = A.PHASE2
AUG = A.OUT

REPLICATE_CASES = ["handoff_iter01", "handoff_iter02", "handoff_iter03"]
N_REPLICATES = 3


# --------------------------------------------------------------------------- #
#  Part A — reproducibility
# --------------------------------------------------------------------------- #
def part_a(ref_ts, E_reactant):
    print("=== PART A: reproducibility (3 replicates per case) ===")
    REPL.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in REPLICATE_CASES:
        n = int(case[-2:])
        start = read(str(PHASE2 / f"iter{n:02d}" / "ts.xyz"))
        for rep in range(N_REPLICATES):
            # fresh calculator each replicate: no cached state carries over
            calc = A.new_calc()
            name = f"{case}_rep{rep}"
            A.CASES = REPL                      # redirect case output
            rec = A.run_sella_case(
                name, start, ref_ts, E_reactant, calc,
                meta={"handoff_iteration": n, "replicate": rep,
                      "start_type": "replicate",
                      "starting_geometry": f"phase2/iter{n:02d}/ts.xyz",
                      "prior_phase2_unique_refs": A.PER_ITER_UNIQUE_REFS * n,
                      "phaseC_sunk_refs": A.PHASEC_SUNK_REFS})
            rows.append(rec)
    A.CASES = AUG / "cases"

    # summarise variability
    summary = {}
    for case in REPLICATE_CASES:
        rs = [r for r in rows if r["case"].startswith(case)]
        calls = [r["handoff_xtb_calculate_calls"] for r in rs]
        summary[case] = {
            "n_replicates": len(rs),
            "calls": calls,
            "calls_median": float(np.median(calls)),
            "calls_min": int(min(calls)), "calls_max": int(max(calls)),
            "calls_range": int(max(calls) - min(calls)),
            "deterministic": len(set(calls)) == 1,
            "reached_reference_saddle": [r["reached_reference_saddle"] for r in rs],
            "converged": [r["converged"] for r in rs],
            "forming_CC_mean": [round(r["forming_CC_mean"], 4) for r in rs],
            "barrier_kcal_mol": [round(r["barrier_kcal_mol"], 3) for r in rs],
            "signature_delta": [round(r["structure_signature_delta"], 4) for r in rs],
            "n_imaginary": [r["n_imaginary"] for r in rs],
            "nsteps": [r["sella_nsteps"] for r in rs],
        }
        s = summary[case]
        print(f"  {case}: calls={calls} median={s['calls_median']:.0f} "
              f"range={s['calls_range']} deterministic={s['deterministic']} "
              f"reached={s['reached_reference_saddle']}")
    return rows, summary


# --------------------------------------------------------------------------- #
#  Part B — starting-state descriptors
# --------------------------------------------------------------------------- #
def xtb_curvature(atoms, calc):
    """xTB Hessian descriptors at a geometry. Costs 2*3N = 96 calls."""
    H = A.hessian(atoms, calc)
    m = atoms.get_masses()
    w = 1.0 / np.sqrt(np.repeat(m, 3))
    lam, vecs = np.linalg.eigh(H * np.outer(w, w))
    nu = np.sign(lam) * A.LAMBDA_TO_CM * np.sqrt(np.abs(lam))
    rc = A.reaction_coordinate_vector(atoms)
    imag = [i for i in range(len(nu)) if nu[i] < -A.IMAG_CUTOFF_CM]
    nu_imag = overlap = None
    if imag:
        k = imag[int(np.argmin([nu[i] for i in imag]))]
        nu_imag = float(nu[k])
        cart = vecs[:, k] * w
        cart /= np.linalg.norm(cart)
        overlap = float(abs(cart @ rc))
    # softest mode regardless of sign, and its alignment with the RC
    cart0 = vecs[:, 0] * w
    cart0 /= np.linalg.norm(cart0)
    return {
        "xtb_lowest_eigenvalue": float(lam[0]),
        "xtb_lowest_mode_cm": float(nu[0]),
        "xtb_lowest_mode_rc_overlap": float(abs(cart0 @ rc)),
        "xtb_n_imaginary": len(imag),
        "xtb_n_below_50cm": int(np.sum(nu < 50.0)),
        "xtb_nu_imag_cm": nu_imag,
        "xtb_reaction_mode_overlap": overlap,
        "xtb_lowest_10_cm": [float(x) for x in np.sort(nu)[:10]],
    }


def part_b(ref_ts, E_reactant, calc):
    print("\n=== PART B: starting-state descriptors ===")
    aug = json.loads((AUG / "results.json").read_text())
    aug_recs = {r["case"]: r for r in aug["records"]}

    starts = [("IDPP", AUG / "control_idpp_start.xyz", "control_idpp_interp", None)]
    for n in range(6):
        starts.append((f"iter{n:02d}", PHASE2 / f"iter{n:02d}" / "ts.xyz",
                       f"handoff_iter{n:02d}", n))

    # xTB reaction mode at the FINAL xTB TS (once) — for displacement projection
    print("  computing xTB Hessian at the reference TS (96 calls, once)...")
    calc.set_phase("ref_ts_hessian")
    H = A.hessian(ref_ts, calc)
    m = ref_ts.get_masses()
    w = 1.0 / np.sqrt(np.repeat(m, 3))
    lam, vecs = np.linalg.eigh(H * np.outer(w, w))
    ref_mode = vecs[:, 0] * w
    ref_mode /= np.linalg.norm(ref_mode)
    print(f"    reference imaginary mode = "
          f"{np.sign(lam[0])*A.LAMBDA_TO_CM*np.sqrt(abs(lam[0])):.1f} cm-1")

    rows = []
    for label, path, aug_case, n in starts:
        at = read(str(path))
        d = {"case": label, "handoff_iteration": n,
             "starting_geometry": str(path.relative_to(HERE.parent.parent))}

        # ---- geometry (free) --------------------------------------------- #
        f = A.forming_distances(at)
        d.update({
            "forming_CC_1": f[0], "forming_CC_2": f[1],
            "forming_CC_mean": float(np.mean(f)),
            "forming_asymmetry": float(abs(f[0] - f[1])),
            "kabsch_rmsd_to_xtb_ts": A.kabsch_rmsd(at.get_positions(),
                                                   ref_ts.get_positions()),
            "signature_delta_to_xtb_ts": A.signature_distance(at, ref_ts),
        })

        # ---- xTB single point (1 call; already stored for iter00-05) ------ #
        calc.set_phase(f"{label}:sp")
        at2 = at.copy(); at2.calc = calc
        E = float(at2.get_potential_energy())
        F = at2.get_forces()
        rc = A.reaction_coordinate_vector(at).reshape(-1, 3)
        f_along = float(np.sum(F * rc))
        f_norm = float(np.linalg.norm(F))
        d.update({
            "xtb_E_eV": E,
            "xtb_E_rel_to_TS_kcal": (E - E_ts_ref) * A.EV2KCAL,
            "xtb_E_rel_to_reactant_kcal": (E - E_reactant) * A.EV2KCAL,
            "xtb_fmax": float(np.max(np.linalg.norm(F, axis=1))),
            "xtb_fnorm": f_norm,
            "xtb_force_along_RC": f_along,
            "xtb_force_orthogonal_to_RC": float(np.sqrt(max(f_norm**2 - f_along**2, 0))),
            "xtb_force_RC_fraction": float(abs(f_along) / (f_norm + 1e-12)),
        })

        # ---- displacement onto the reference reaction mode (free) --------- #
        disp = (at.get_positions() - ref_ts.get_positions()).ravel()
        d["disp_norm_to_xtb_ts"] = float(np.linalg.norm(disp))
        d["disp_proj_on_ref_mode"] = float(abs(disp @ ref_mode))
        d["disp_proj_fraction"] = float(abs(disp @ ref_mode)
                                        / (np.linalg.norm(disp) + 1e-12))

        # ---- recovered corrected-MACE descriptors (free, from Phase 2) ---- #
        if n is not None:
            pm = json.loads((PHASE2 / f"iter{n:02d}" / "metrics.json").read_text())
            d.update({
                "mace_corr_E_eV": pm["ts"]["energy_eV"],
                "mace_corr_fmax": pm["ts"]["final_fmax"],
                "mace_corr_barrier_kcal": pm["ts"]["barrier_kcal_frozen_R"],
                "mace_xtb_force_rmse": pm["xtb_at_ts"]["force_rmse_corr_vs_xtb"],
                "mace_xtb_force_cos": pm["xtb_at_ts"]["force_cos_corr_vs_xtb"],
                "mace_corr_n_imaginary": pm["ts"]["vib"]["n_imaginary"],
                "mace_corr_nu_imag_cm": pm["ts"]["vib"]["nu_imag_cm"],
                "mace_corr_rc_overlap": pm["ts"]["vib"]["reaction_mode_overlap"],
                "phase2_valid_da_saddle": pm["ts"]["valid_da_saddle"],
            })
        else:
            d.update({k: None for k in
                      ["mace_corr_E_eV", "mace_corr_fmax", "mace_corr_barrier_kcal",
                       "mace_xtb_force_rmse", "mace_xtb_force_cos",
                       "mace_corr_n_imaginary", "mace_corr_nu_imag_cm",
                       "mace_corr_rc_overlap", "phase2_valid_da_saddle"]})

        # ---- xTB curvature at the start (96 calls) ------------------------ #
        calc.set_phase(f"{label}:hessian")
        d.update(xtb_curvature(at, calc))

        # ---- outcome (from the Aug 21 benchmark) -------------------------- #
        ar = aug_recs[aug_case]
        d.update({
            "success": ar["reached_reference_saddle"],
            "handoff_calls": ar["handoff_xtb_calculate_calls"],
            "total_marginal_cost": ar["total_marginal_xtb_calls"],
            "converged": ar["converged"],
            "final_forming_CC_mean": ar["forming_CC_mean"],
            "final_barrier_kcal": ar["barrier_kcal_mol"],
        })
        rows.append(d)
        print(f"  {label:<7} fmax={d['xtb_fmax']:.4f} |F|={d['xtb_fnorm']:.3f} "
              f"E-TS={d['xtb_E_rel_to_TS_kcal']:+7.2f} "
              f"lowest={d['xtb_lowest_mode_cm']:8.1f}cm-1 "
              f"n_imag={d['xtb_n_imaginary']} "
              f"RC_ov={d['xtb_reaction_mode_overlap']} "
              f"-> {'OK' if d['success'] else 'FAIL'} ({d['handoff_calls']})")
    return rows


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(exist_ok=True)
    global E_ts_ref

    ref_ts = read(str(DA / "ts_opt.xyz"))
    calc = A.new_calc()
    calc.set_phase("reference")
    r = read(str(DA / "start.xyz")); r.calc = calc
    E_reactant = float(r.get_potential_energy())
    t = ref_ts.copy(); t.calc = calc
    E_ts_ref = float(t.get_potential_energy())
    print(f"[ref] E(reactant)={E_reactant:.6f} eV  E(xTB TS)={E_ts_ref:.6f} eV\n")

    rep_rows, rep_summary = part_a(ref_ts, E_reactant)
    desc_rows = part_b(ref_ts, E_reactant, calc)

    (OUT / "reproducibility.json").write_text(json.dumps(
        {"replicates": rep_rows, "summary": rep_summary}, indent=2))
    (OUT / "descriptors.json").write_text(json.dumps(desc_rows, indent=2))
    print(f"\nwrote {OUT/'reproducibility.json'} and {OUT/'descriptors.json'}")
    print(f"total xTB calls this diagnostic session: {calc.n_total}")


if __name__ == "__main__":
    main()
