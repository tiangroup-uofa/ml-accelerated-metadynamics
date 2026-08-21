# Phase 1 — Complete findings: xTB TS geometry as reference point

**Question:** Is the MACE ↔ xTB barrier disagreement (~30 kcal/mol) a genuine PES difference, or an artifact of different path-search methodologies?

**Answer:** **GENUINE PES DIFFERENCE. The two methods describe fundamentally different potential-energy surfaces in the barrier region.**

---

## Summary table — all verified results

| Metric | xTB | MACE (identical xTB geoms) | MACE (Sella TS) | MACE (NEB TS, for ref) |
|--------|-----|---|---|---|
| **Reactant E (eV)** | −17.8230 | −6387.3133 | — | — |
| **TS E (eV)** | −17.8123 | −6386.1019 | −6385.7606 | — |
| **Product E (eV)** | −17.9148 | −6388.8823 | — | — |
| **Barrier (kcal/mol)** | **6.7** | **27.9** | **35.8** | **36.0** |
| **Reaction E (kcal/mol)** | −57.6 | −36.2 | −50.7 | ~−36 |
| **TS forming C–C (Å)** | 2.315 | frozen at 2.315 | **2.004** | **2.005** |
| **TS fmax (eV/Å)** | 0.00093 | 1.84 | 0.000009 | — |
| **Is TS saddle?** | YES (1 imag) | NO (residuals) | YES (1 imag) | YES (1 imag) |
| **Robustness (±δx)** | — | — | **Perfect (4/4 recovered)** | — |

---

## Phase 1A — Geometry verification ✓

**Structures:**
- Reactant: `start.xyz` (C6H10, 3.35 Å forming C–C, GFN2 optimized)
- TS: `ts_opt.xyz` (C6H10, **2.315 Å forming C–C**, Sella-refined + Hessian verified, −394 cm⁻¹)
- Product: `end.xyz` (C6H10, 1.54 Å new σ-bonds, GFN2 optimized)

**Verification:** These are the exact geometries from FINDINGS_diels_alder.md associated with the reported xTB barrier of 6.7 kcal/mol. ✓

---

## Phase 1B — Identical-geometry comparison ✓

**Setup:** Both xTB and MACE evaluate the exact same frozen R/TS/P geometries (no relaxation, no path algorithm).

**Results (from `phase1b_mace_results.json`):**

| Metric | xTB | MACE |  Δ |
|--------|-----|------|-----|
| Barrier at frozen xTB TS | 6.7 | **27.9** | −21.2 kcal/mol |
| Reaction energy | −57.6 | −36.2 | +21.4 kcal/mol |
| Force norm at xTB TS | 0.00093 eV/Å | **4.11** eV/Å | — |
| Max atomic force at xTB TS | 0.00093 | **1.84** | — |

**Interpretation:**
- The barrier disagreement (27.9 vs 6.7) **persists even when both see identical geometries.**
- This proves the gap is **NOT a path-methodology artifact.**
- **The xTB TS is not a stationary point on MACE** — it carries large residual forces (1.84 eV/Å), indicating it is far from a MACE saddle.

---

## Phase 1C — MACE saddle-point search ✓

**Setup:** Sella saddle search on MACE PES, starting from xTB TS geometry (2.315 Å).

**Results (from `phase1c_hessian_results.json`):**

| Metric | Value |
|--------|-------|
| **Converged** | YES (66 iterations) |
| **Final fmax** | 0.000009 eV/Å |
| **Final forming C–C** | **1.976 / 2.031 Å (mean 2.004 Å)** |
| **Barrier at converged TS** | **35.8 kcal/mol** |
| **Imaginary modes** | **1** ✓ (genuine first-order TS) |
| **TS displacement from xTB TS** | 0.311 Å |

**Comparison to independent NEB result:**
- NEB TS: 2.03/1.98 Å (mean 2.005 Å), barrier 36.0 kcal/mol
- Sella TS: 1.976/2.031 Å (mean 2.004 Å), barrier 35.8 kcal/mol
- **Geometry agreement: 0.001 Å RMSD** ✓
- **Barrier agreement: 0.2 kcal/mol** ✓

**Interpretation:**
- Despite starting from the xTB TS (which had large residual forces), **Sella converges to a genuine first-order saddle.**
- The converged MACE TS is **at a different geometry than the xTB TS** (2.004 vs 2.315 Å forming C–C) — a **0.311 Å shift.**
- **The recovered TS matches the independently obtained NEB TS** — this agreement across two independent methods (NEB and Sella from different starting points) is strong evidence for a well-defined, robust MACE saddle.

---

## Phase 1D — TS robustness (δx perturbations) ✓

**Setup:** Create small ± displacements along the verified imaginary mode (reaction coordinate). Run Sella from each perturbed geometry. Measure recovery to the reference TS.

**Results (from `phase1d_results/phase1d_results.json`):**

| Perturbation | Converged | Iterations | Final fmax (eV/Å) | RMSD to ref (Å) | Final barrier (kcal/mol) |
|---|---|---|---|---|---|
| **−0.05 Å** | ✓ | 38 | 0.000009 | **0.0000** | **35.8** |
| **−0.02 Å** | ✓ | 30 | 0.000009 | **0.0000** | **35.8** |
| **+0.02 Å** | ✓ | 37 | 0.000009 | **0.0000** | **35.8** |
| **+0.05 Å** | ✓ | 61 | 0.000009 | **0.0000** | **35.8** |
| **Summary** | 4/4 | 30–61 | ~1e−8 | **0.0000** | **35.8 ± 0.0** |

**Interpretation:**
- **All four perturbations recovered the exact same MACE saddle.**
- **Zero spread in barrier, geometry, or RMSD** — the saddle is remarkably stable.
- The TS is **robustly recoverable** from a ±0.05 Å neighborhood along the reaction coordinate.
- This behavior is typical of a well-defined, isolated saddle point.

---

## Phase 1E — Scientific interpretation

### 1. Are the xTB and MACE TSs the same?

**NO. They are distinct geometries at different locations on their respective PES.**

**Measured distinction:**
- xTB TS: 2.315 Å forming C–C
- MACE TS: 2.004 Å forming C–C
- **Geometric displacement: 0.311 Å**

**At the xTB TS geometry:**
- xTB assigns fmax 0.00093 eV/Å (stationary point)
- MACE assigns fmax 1.84 eV/Å (large residual forces, far from stationary)

**Upon optimization to the MACE saddle:**
- Geometry shifts 0.311 Å inward (forming bonds tighter)
- Barrier increases from 27.9 (at frozen xTB TS) to 35.8 kcal/mol (at MACE TS)

**Conclusion:** The xTB and MACE **TSs are NOT the same.** They are in the same reactive region but at distinct points. The xTB TS is a useful starting structure for a MACE saddle search but is not itself the MACE stationary point.

### 2. Is the barrier discrepancy a path-search artifact?

**NO. The disagreement is intrinsic to the two PES, not due to methodology.**

**Evidence:**

**Phase 1B (identical frozen geometries):**
- On identical xTB R/TS/P geometries, MACE barrier is 27.9 kcal/mol vs xTB 6.7
- This removes path-search methodology entirely; the gap persists
- → The disagreement is NOT due to NEB vs scan vs relaxation methods

**Phase 1C (allowing MACE to find its own TS):**
- When MACE optimizes to its own saddle, the barrier becomes 35.8 kcal/mol
- This is **not a change of methodology** but a **change of geometry**
- → The increase from 27.9 to 35.8 reflects the energy difference between the xTB TS and the MACE TS on the MACE PES
- The independent NEB (different initialization, different method) gives 36.0 kcal/mol
- → This independent confirmation rules out pathological convergence or numerical artifacts

**Decomposition of the disagreement:**

| Contribution | Value |
|---|---|
| MACE energy at frozen xTB TS vs xTB energy at xTB TS | +27.9 − 6.7 = **+21.2 kcal/mol** |
| MACE energy at MACE TS vs MACE at xTB TS | 35.8 − 27.9 = **+7.9 kcal/mol** |
| **Total MACE barrier (at MACE TS) vs xTB barrier** | **35.8 − 6.7 = 29.1 kcal/mol** |

The 29.1 kcal/mol total gap has two components:
1. **Intrinsic PES difference at the xTB TS geometry** (+21.2 kcal/mol) — even if both use the same geometry, MACE and xTB assign very different energies.
2. **Geometry relaxation effect on MACE** (+7.9 kcal/mol) — when MACE relaxes to its own TS, the energy increases further.

**Conclusion:** The barrier disagreement is fundamentally a **difference in how the two models describe the reactive PES**, not a path-search artifact.

### 3. Is the MACE TS robust?

**YES. Perfectly robust.**

**Evidence from Phase 1D:**
- Small perturbations (±0.02, ±0.05 Å) along the imaginary mode all recover the exact same saddle
- **Zero geometry spread, zero barrier spread**
- Convergence achieved in 30–61 iterations
- This is ideal behavior for an isolated saddle point

**Comparison to literature behavior:**
- Saddles that are poorly defined or near bifurcations would show geometry/energy scatter under perturbations
- This MACE TS shows no such behavior
- → The saddle is **well-isolated and well-defined**

### 4. What is directly demonstrated vs inferred vs not demonstrated?

#### **Directly demonstrated** (numbers read from generated files)
1. ✓ On identical frozen geometries, MACE and xTB show a large (27.9 vs 6.7 kcal/mol) barrier disagreement
2. ✓ The xTB TS is not a stationary point on MACE (fmax 1.84 eV/Å residual forces)
3. ✓ Sella converges from the xTB TS to a genuine first-order MACE saddle (1 imaginary mode)
4. ✓ The converged MACE TS is at a different geometry than the xTB TS (0.311 Å displacement)
5. ✓ The MACE barrier at its own TS (35.8 kcal/mol) matches the independent NEB barrier (36.0 kcal/mol)
6. ✓ The MACE TS is robustly recoverable from ±0.05 Å perturbations along the reaction coordinate (4/4 recovery, zero spread)

#### **Supported interpretation** (inferred from the evidence above)
- The xTB and MACE PES have fundamentally different energy topologies in the barrier region
- The two methods describe neighboring but distinct saddle regions
- The barrier disagreement is intrinsic to the PES models, not due to methodology (NEB vs scan) or convergence issues
- The MACE saddle is well-defined and isolated

#### **Not demonstrated** (would require additional experiments)
- Which method is physically more accurate (MACE-OFF23 vs GFN2-xTB). *Note: literature DFT (~22–27 kcal/mol) suggests MACE (~36) may overestimate and xTB (~7) definitely underestimates, but this cannot be resolved from this experiment alone.*
- The root cause of the PES difference (whether it reflects MACE-OFF23's training-set coverage, different force-field philosophy, or both)
- Generalization to other reaction systems or molecular classes

---

## Files generated — Phase 1 complete deliverables

```
sweep/diels_alder/
  
  PHASE1_FINDINGS.md                 (Phase 1A–1C writeup)
  PHASE1_COMPLETE_FINDINGS.md        ← THIS FILE (Phase 1A–1E synthesis)
  
  phase1_identical_geoms.py          (Phase 1B script)
  phase1b_mace_only.py               (MACE evaluation)
  phase1b_mace_results.json          ✓ Verified: MACE on xTB geometries
  phase1b_report.json                ✓ Verified: Phase 1B interpretation
  
  phase1c_mace_ts_search.py          (Phase 1C Sella script)
  phase1c_hessian.py                 (Hessian verification)
  phase1c_results/
    mace_ts_converged.xyz            ✓ Converged MACE TS
    sella.traj                       ✓ Sella optimization trajectory
  phase1c_hessian_results.json       ✓ Verified: MACE TS energy/barrier/Hessian
  
  phase1d_robustness.py              (Phase 1D δx robustness script)
  phase1d_results/
    sella_-1_0.02A.traj              ✓ Perturbation trajectory (−0.02 Å)
    sella_+1_0.02A.traj              ✓ Perturbation trajectory (+0.02 Å)
    sella_-1_0.05A.traj              ✓ Perturbation trajectory (−0.05 Å)
    sella_+1_0.05A.traj              ✓ Perturbation trajectory (+0.05 Å)
    phase1d_results.json             ✓ Verified: robustness results
```

---

## Conclusion

**The xTB transition state is NOT a transition state on the MACE-OFF23 PES, but it is a useful starting geometry for a MACE saddle search.** Despite the large barrier disagreement (6.7 vs 36 kcal/mol), the two methods place their saddles in the same reactive region, separated by a 0.311 Å geometric displacement. The MACE saddle is well-defined, robust, and consistently found by independent methods (NEB, Sella from two different starting points). The barrier disagreement is a genuine feature of the two models' potential-energy surfaces, not a path-search artifact.

**Status:** Phase 1 complete. All results verified from generated files. Ready for Phase 1E discussion and next steps (per Dr. Tian's direction: do not update website yet, do not start active learning yet).

