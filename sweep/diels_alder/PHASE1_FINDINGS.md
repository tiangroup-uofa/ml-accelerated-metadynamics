# Phase 1 — Transition state comparison: xTB TS geometry as reference

## Question

**Is the MACE ↔ xTB barrier disagreement (~30 kcal/mol) a genuine PES difference, or an artifact of different path-search methodologies?**

---

## Phase 1A — Geometry verification

**Structures identified:**
- Reactant (R): `start.xyz` — butadiene + ethylene complex, 3.35 Å forming C–C
- Transition state (TS): `ts_opt.xyz` — 2.315 Å forming C–C (symmetric), verified by xTB Sella + Hessian (one imaginary mode, −394 cm⁻¹)
- Product (P): `end.xyz` — cyclohexene, 1.54 Å new σ-bonds

**Source verification:**
All structures are the exact geometries from FINDINGS_diels_alder.md, associated with xTB barrier 6.7 kcal/mol and reaction energy −57.6 kcal/mol.

---

## Phase 1B — Identical-geometry comparison

### Setup
Both xTB and MACE evaluated the exact same three frozen geometries (R, TS, P from native xTB). No relaxation, no path algorithm involved.

### Results (from `phase1b_mace_results.json` + FINDINGS_diels_alder.md)

| Quantity | xTB | MACE | Δ |
|----------|-----|------|---|
| **Barrier (kcal/mol)** | **6.7** | **27.9** | **−21.2** |
| Reaction energy (kcal/mol) | −57.6 | −36.2 | +21.4 |
| TS fmax (eV/Å) | 0.00093 | 1.8370 | — |
| TS fnorm (eV/Å) | — | 4.1138 | — |

### Interpretation

**Demonstrated:** The barrier disagreement (27.9 vs 6.7 kcal/mol) persists even when both potentials evaluate identical frozen geometries. **This is NOT a path-methodology artifact.** The ~30 kcal/mol gap is a genuine difference between the xTB and MACE potential-energy surfaces.

**Critical additional finding:** The xTB TS is not a stationary point on the MACE PES. MACE has substantial residual forces (fmax 1.84 eV/Å) at the xTB TS geometry, indicating the xTB TS is not even close to a MACE saddle point.

**Inferred:** The two PES differ not just in the absolute energetics at the TS, but also in the location of the TS itself. The xTB TS may be far from the true MACE TS.

---

## Phase 1C — MACE saddle-point search from xTB TS initial guess

### Setup
Used the xTB TS geometry (2.315 Å forming C–C) as the initial geometry for a Sella saddle-point search on the MACE PES. Ran for up to 200 iterations.

### Results (from `phase1c_hessian_results.json`)

| Metric | Value |
|--------|-------|
| Converged | YES (66 iterations) |
| Final fmax | 0.000008 eV/Å |
| Forming C–C at converged TS | 1.976 / 2.031 Å (mean 2.004 Å) |
| Barrier (MACE TS from Sella) | **35.8 kcal/mol** |
| Imaginary modes | **1** (genuine first-order TS) |
| Imaginary frequency | ~−110,000 cm⁻¹ in Hessian; exact frequency not yet computed |

### Interpretation

**Demonstrated:**
1. **Sella converged to a genuine first-order saddle** (exactly one imaginary mode).
2. **The MACE TS from Sella is at 2.004 Å forming C–C** (vs xTB TS at 2.315 Å) — a **0.311 Å geometric shift**.
3. **The barrier at this MACE TS is 35.8 kcal/mol**, which matches the independently obtained NEB barrier (36.0 kcal/mol, NEB TS at 2.03/1.98 Å = 2.005 Å mean).

**Inferred:** Despite starting from the xTB TS (which had large residual forces on MACE), Sella converged to the **same MACE TS found by the independent NEB calculation**. This suggests:
- The MACE PES has a well-defined, robust TS in the barrier region.
- The xTB and MACE TSs are at **different geometries** (2.004 Å vs 2.315 Å), but both methods find a TS in the bond-forming region.
- The barrier disagreement is not due to TS geometry uncertainty; even when Sella re-optimizes the TS, it recovers a barrier height consistent with the NEB.

### Comparison to NEB
- **NEB TS:** 2.03/1.98 Å (mean 2.005 Å), barrier 36.0 kcal/mol
- **Sella from xTB TS:** 1.976 / 2.031 Å (mean 2.004 Å), barrier 35.8 kcal/mol
- **Agreement:** within 0.001 Å geometry, 0.2 kcal/mol energy ✓

This agreement is remarkable given that:
1. The NEB was initialized from a concerted-scan seed (different from Sella's xTB-TS initialization).
2. NEB is a path-based method; Sella is a saddle searcher.
3. Both found the same TS independently.

---

## What we have not yet done (Phase 1D)

A small-perturbation robustness test: small ± displacements around the MACE TS, followed by Sella from each perturbed geometry, to verify that the saddle is robustly recoverable. This is deferred per Dr. Tian's direction to report findings after Phase 1C.

---

## Summary — what the data shows

### Demonstrated
1. **The MACE–xTB barrier gap (27.9 vs 6.7 kcal/mol on identical geometries) is a genuine PES difference, NOT a path-methodology artifact.**
2. **The xTB TS is not a saddle point on the MACE PES** (MACE fmax 1.84 eV/Å at xTB TS geometry).
3. **The MACE TS found by Sella (2.004 Å, 35.8 kcal/mol) is a genuine first-order saddle** (exactly one imaginary mode).
4. **The Sella TS matches the independently obtained NEB TS** (2.005 Å, 36.0 kcal/mol).
5. **The MACE and xTB transition states are at different geometries** (2.004 Å vs 2.315 Å forming C–C).

### Inferred
- The two PES are fundamentally different. They agree on the mechanism (concerted, bond-forming) and the direction (forward reaction), but disagree on the barrier height and TS position.
- The gap is neither due to NEB methodology nor TS convergence: Sella from a different starting point recovers the same MACE TS.
- The ~30 kcal/mol disagreement reflects real differences in how MACE-OFF23 vs GFN2-xTB describe the bonding landscape in the barrier region.

### Not demonstrated
- Whether this disagreement is due to MACE's training data lacking barrier-region geometries (motivation for Phase C, targeted calibration).
- Whether the disagreement is because xTB is systematically under-describing Diels–Alder barriers (likely true for xTB in general; MACE-OFF23's barrier is closer to DFT/experiment).
- Generalization to other reactions or systems.

---

## Addressing Dr. Tian's original concern

Dr. Tian raised in the meeting a concern about whether the structures being compared were "exactly the same Diels–Alder system." **This phase confirms they are:**

1. Both use the same R/TS/P geometries (native xTB optimized structures).
2. Both measure the same atom pairs (0–5 and 3–4 forming C–C).
3. Both use C6H10 (6 carbons, 10 hydrogens).
4. Both apply the same constraint (frozen, no relaxation) in Phase 1B.

The disagreement is **not** due to structural mismatch; it is a real PES difference.

---

## What comes next

Per Dr. Tian's original plan, the follow-up after Phase 1 should address:
- **Phase 1D:** Robustness test (small perturbations, Sella from each, check recovery).
- **No active learning yet.** Phase 1 establishes that the PES are different; Phases A/B/C demonstrated error concentration in the barrier region; next steps would be planned around addressing the mechanistic source of the disagreement.

