# Phase 2 — controlled iterative correction loop (Aug 7 follow-up)

**Question.** Can a lightweight conservative correction be iteratively updated with a
small number of targeted xTB evaluations so that the corrected-MACE transition state
and barrier progressively move toward the xTB target, without full MACE fine-tuning?

**Answer: no — not with this correction architecture and this acquisition rule.**
The loop does not converge. **Iteration 0 was the best model of the entire run**, and
42 additional xTB reference calculations made the barrier steadily *worse*. This is a
negative result and is reported as such; it was not forced toward convergence.

Every number below is read back from a generated file under
`sweep/corrections/phase2/` (`summary.json`, `metrics.csv`, `iter*/metrics.json`).

---

## 0. Environment and reproducibility notes (read this first)

Three things were established before any Phase 2 measurement, because Phase 2 —
unlike Phase 1B — requires **live xTB evaluations at new geometries** that cannot be
looked up:

1. **xTB was absent from this machine and was reinstalled** (micromamba env `xtbenv`,
   conda-forge `xtb-python`, kept at `~/.local` so it survives scratchpad wipes). It
   was then validated against the frozen Phase 1 baseline and reproduces it:
   barrier **6.71** kcal/mol (baseline 6.7), reaction energy **−57.64** (baseline
   −57.6), reactant energy **−484.987332 eV** (matches `xtbpy_stability.json`'s
   −484.9873315). MACE/Sella run in the system python; xTB runs in a separate
   process, the same OpenMP isolation pattern `phaseC_eval.py` already uses.

2. **Iteration 0 is bit-exactly reproducible.** Refitting the pairwise Delta from the
   Phase C data (roles `baseline`+`water`+`barrier_extra`, 106 configs) reproduces the
   stored `B_barrier` coefficients with `max|refit − stored| = 0.0`.

3. **Hessian units fix.** Phase 1's Hessian was mass-*unweighted*, so its
   eigenvalue→cm⁻¹ conversion produced a non-physical number (−109,924 cm⁻¹). The
   *count* of imaginary modes was still correct (Sylvester's law of inertia: mass
   weighting is a congruence transform and preserves the sign structure), and no
   published Phase 1 claim quoted that number — the findings said "exact frequency not
   yet computed" and the website says only "one imaginary mode". **No Phase 1
   conclusion changes.** Phase 2 uses a properly mass-weighted Hessian, validated
   against a known answer: at the xTB TS it returns **−394.1 cm⁻¹**, matching xTB's
   own `vibspectrum` value of −394, and cleanly separates the six translation/rotation
   modes (next-lowest −13.3 cm⁻¹).

---

## 1. Protocol, and a protocol failure worth recording

### 1.1 The naive seeding strategy fails (archived as a negative control)

The first run seeded each Sella saddle search from the **xTB TS** (2.315 Å) — the
strategy that worked for *raw* MACE in Phase 1C. On the *corrected* PES it fails:

| | iter 0 | iter 1 |
|---|---|---|
| converged saddle | yes (1 imaginary mode, −150.2 cm⁻¹) | yes (1 mode, −167.0 cm⁻¹) |
| forming C–C | **3.203 Å** | **3.251 Å** |
| overlap of imaginary mode with the forming-bond coordinate | **1.2e−06** | **0.014** |
| barrier vs frozen reactant | **−3.84 kcal/mol** | **−4.18** |
| xTB force cosine at that geometry | 0.048 | **−0.334** |
| constrained-scan barrier | 9.87 | **12.91** |

These are *genuine first-order saddles* — but not Diels–Alder ones. At 3.2 Å the
fragments are essentially separated, the imaginary mode is **orthogonal** to bond
formation (overlap ~1e−6; random overlap in 48 DOF is ~1/√48 ≈ 0.14), and the
"barrier" is *below* the reactant. The loop then queried xTB around that non-TS
geometry and the refit degraded the reaction barrier (9.87 → 12.91). Run archived at
`phase2_naive_seed/`.

**Fix, following the project's own convention** (CLAUDE.md: *"use an external saddle
optimizer (Sella) … seeded from a good guess (a relaxed scan), not from the raw
`--path` guess"*): every iteration seeds Sella from **that iteration's own
constrained-scan maximum**, and every converged saddle must pass a validity gate.

### 1.2 One iteration (final protocol)

1. Relaxed concerted scan on the current corrected PES → scan barrier + scan-max geometry.
2. Sella first-order saddle search seeded from that scan maximum.
3. **TS validity gate**: converged ∧ exactly 1 imaginary mode ∧ forming C–C ∈ [1.5, 3.0] Å
   ∧ |overlap of imaginary mode with the forming-bond coordinate| ≥ 0.30. If the gate
   fails, no xTB query is centred on that structure — the loop falls back to the scan
   maximum (a legitimate barrier-region geometry) and the failure is recorded.
4. Query GFN2-xTB at that exact geometry.
5. Build a small local packet: the TS itself, ±0.05 and ±0.10 Å along the imaginary
   mode, and 2 isotropic 0.05 Å perturbations = **7 new xTB calculations/iteration**.
6. Refit the **same** pairwise Delta (cutoff 3.6, n_rbf 8, rmin 0.7, ridge 1e-3, 48
   coefficients, 6 element-pair channels) on the **cumulative** dataset.
7. Re-measure held-out reactive forces and water; repeat.

Controls held fixed: the 48-config no-leakage held-out reactive split is **never**
trained on; the correction architecture and all hyperparameters are frozen; only the
reference dataset grows (106 → 141 configs).

---

## 2. Results

| iter | cum. new xTB calls | saddle barrier | scan barrier | forming C–C (Å) | certified saddle | xTB \|F\|max at TS (eV/Å) | held-out TS+post RMSE | water Rg (Å) |
|---|---|---|---|---|---|---|---|---|
| 0 | 7 | **6.33** | 9.87 | 2.428 | yes (−358.3 cm⁻¹) | 0.770 | 0.6879 | 3.738 |
| 1 | 14 | 9.72 | 10.51 | 1.996 | yes (−758.1) | 2.805 | 0.6880 | 3.690 |
| 2 | 21 | 7.11 | 12.50 | 2.415 | yes (−333.4) | 0.686 | 0.6973 | 3.841 |
| 3 | 28 | 8.62 | 12.75 | 2.519 | **NO** (2 imaginary) | 0.461 | 0.6966 | 3.796 |
| 4 | 35 | 9.46 | 13.54 | 2.046 | yes (−461.3) | 2.340 | 0.6987 | 3.803 |
| 5 | 42 | 9.25 | 13.91 | 2.519 | **NO** (2 imaginary) | 0.428 | 0.7113 | 3.834 |
| **xTB target** | — | **6.7** | **6.7** | **2.315** | yes (−394 cm⁻¹) | 0 | — | **3.77** |

Figures: `figures/phase2_barrier.png`, `phase2_geometry.png`, `phase2_forces.png`,
`phase2_generalization.png`.

### 2.1 Iteration 0 is much better than Phase C suggested — because of the estimator

Phase C reported the `B_barrier` delta at **9.9 kcal/mol** — that is a *constrained-scan*
barrier. A proper saddle search on the same model gives **6.33 kcal/mol at 2.428 Å**,
with one imaginary mode at −358.3 cm⁻¹ and RMSD 0.173 Å to the xTB TS. The scan path
does not pass through the saddle, so it overestimates by ~3.5 kcal/mol.

Measured correctly, **the Phase C correction was already within 0.37 kcal/mol of the
xTB target before the loop started** (6.33 vs 6.7), with an imaginary frequency
comparable to xTB's (−358 vs −394 cm⁻¹). Both estimators are tracked separately
throughout; neither replaces the other.

### 2.2 The loop diverges

- **Saddle barrier**: 6.33 → 9.72 → 7.11 → 8.62 → 9.46 → 9.25. Oscillates; error vs
  target grows from **0.37 to 2.55 kcal/mol (~7×)**.
- **Scan barrier**: 9.87 → 10.51 → 12.50 → 12.75 → 13.54 → **13.91**. Degrades
  **monotonically** away from 6.7.
- **TS position**: 2.428 → 1.996 → 2.415 → 2.519 → 2.046 → 2.519 Å. Swings by ±0.5 Å
  around the 2.315 Å target without settling.
- **2 of 6 iterations produced no certified first-order saddle** (two imaginary modes).

### 2.3 The failure is not numerical instability

Delta coefficient norm across iterations: 6.051, 5.953, 5.988, 5.966, 5.947, 6.111 —
essentially constant (max |c| 3.20 → 2.76). The least-squares fit is well conditioned;
the correction is **reallocating a fixed, limited capacity**, not blowing up.

### 2.4 A metric that must not be over-read

`force cosine (corrected vs xTB) at the TS` is reported in the raw JSON but is
**meaningless at a converged saddle**: |F_corrected|max there is ~1e−4 eV/Å by
construction, so the cosine is a 0/0 noise ratio (values ranged 0.296, 0.003, 0.108,
−0.663, 0.011, −0.654 with no physical trend). The interpretable quantity is
**xTB's own residual force at the corrected TS**, which oscillates 0.43–2.81 eV/Å and
shows **no systematic decrease**. Figure 3 plots that and omits the cosine.

### 2.5 Controls

- **Held-out reactive set (never trained on)**: TS+post-TS RMSE 0.6879 → **0.7113**
  (+3.4%); overall 0.4556 → 0.4704 (+3.3%). Generalization *slowly degrades*. The
  local improvement was not purchased at the price of catastrophic overfitting, but it
  was not purchased at all.
- **Water**: Rg spans 3.690–3.841 Å around the xTB reference 3.77, with per-iteration
  MD standard deviations of 0.037–0.086 Å — the scatter is **within MD noise**; O–O peak
  is 2.74 Å in 3 of 6 iterations and 2.84–2.94 Å in the rest (bin width 0.098 Å). Droplet remained stable and
  un-condensed in every iteration. **Equilibrium behaviour was not damaged.**

---

## 3. Answers to the Phase 2 questions

1. **Does iterative targeted correction monotonically reduce the barrier error?**
   **No — the opposite.** The saddle-barrier error grows from 0.37 to 2.55 kcal/mol and
   the scan barrier degrades monotonically (9.87 → 13.91).
2. **Does the corrected TS geometry move toward the xTB TS?** **No.** Forming C–C
   oscillates ±0.5 Å about the target without settling. RMSD-to-xTB-TS does trend down
   (0.173 → 0.067 Å) but the two lowest values are the iterations that failed the
   saddle gate and fell back to the scan maximum, so that trend is confounded and is
   not evidence of convergence.
3. **Does xTB's residual force at the corrected TS decrease?** **No.** It oscillates
   over 0.43–2.81 eV/Å with no systematic trend.
4. **How many new xTB calculations per meaningful improvement?** **No meaningful
   improvement was obtained from any of the 42 new calculations.** Cost per improvement
   is undefined (division by zero), which is the cleanest way to state the result.
5. **Does performance saturate?** It does not saturate — it **oscillates and drifts**.
   The experiment-level stopping rule (Δbarrier < 0.5 ∧ Δforming < 0.01 Å) was never
   satisfied; the run ended at the preset 5-iteration cap.
6. **Does held-out barrier-region force error improve, or are we just fitting the
   queried neighbourhood?** It **degrades slightly** (0.6879 → 0.7113). The added data
   is not generalizing.
7. **Does water/equilibrium behaviour remain stable?** **Yes** — within MD noise
   throughout, no condensation or instability.
8. **Is the pairwise Delta expressive enough to converge to the xTB TS/PES?** The
   evidence is **consistent with a representational ceiling**, and this is the single
   most important open question the experiment raises (see §4).
9. **Does this justify building an automated active-sampling mechanism?** **Not yet,
   and not on this architecture.** See §5.

---

## 4. Why it fails — interpretation (NOT demonstrated)

These are mechanistic hypotheses consistent with the measurements, not results:

- **Radial-only capacity.** The correction is a sum of radial pair potentials — 48
  coefficients over 6 element-pair channels, with no angular or many-body terms. The
  location and curvature of a Diels–Alder saddle depend on angular reorganization of
  the forming six-membered ring. A radial correction can shift energies along the
  forming C–C distance but cannot independently control the saddle's angular
  curvature, so new local constraints move the saddle rather than pin it. The flat
  coefficient norm (§2.3) supports "fixed capacity being reallocated" over "fit
  diverging".
- **The acquisition point is not an xTB stationary point.** xTB carries 0.43–2.81 eV/Å
  of residual force at every corrected TS. Force-matching there tells the correction to
  reproduce a *large* xTB gradient at a geometry the corrected model believes is
  stationary — a directly conflicting demand that displaces the saddle.
- **No trust region and no acquisition criterion.** The query location is a function of
  the model and the model is a function of the queries. Nothing in this loop makes that
  map contractive, which is consistent with the observed oscillation.
- **Tightly clustered packets.** Seven configurations within ~0.1 Å of one another add
  nearly collinear rows to the least-squares problem: locally informative, globally weak.

---

## 5. Scientific framing

### Directly demonstrated
- On this system, with this fixed conservative pairwise-Delta architecture and this
  TS-centred acquisition rule, **42 targeted xTB reference calculations over 5
  iterations did not improve the corrected-MACE barrier or TS geometry**; the
  scan barrier degraded monotonically and the saddle barrier oscillated away from target.
- **Iteration 0 was the best model of the run** (saddle barrier 6.33 vs target 6.7).
- Measured with a proper saddle search rather than a constrained scan, the Phase C
  barrier-enriched Delta already sits within 0.37 kcal/mol of the xTB barrier, at
  2.428 Å with a −358.3 cm⁻¹ reaction mode and 0.173 Å RMSD to the xTB TS.
- The held-out reactive split degraded slightly (+3.4% TS+post-TS RMSE); water
  equilibrium structure stayed within MD noise.
- Seeding a saddle search on the corrected PES from the xTB TS finds a **spurious
  reactant-basin saddle** (3.20 Å, imaginary mode overlap 1.2e−06, barrier −3.84
  kcal/mol) — TS searches on corrected surfaces need a validity gate.

### Supported interpretation
- Iterative TS-centred querying **does not appear data-efficient here**; the binding
  constraint looks like correction *expressiveness*, not reference-data quantity.
- A correction can reproduce a target barrier *height* while still not sharing the
  target's stationary point — geometry agreement and energy agreement are separate
  achievements (consistent with the Phase 1 conclusion).

### Not demonstrated / explicitly not claimed
- **No active learning** was implemented; there is no acquisition metric, no
  uncertainty estimate, and no autonomous sampling. This was a fixed, hand-specified
  query rule.
- **No claim about DFT accuracy or physical superiority of either method.** xTB remains
  a proof-of-concept reference target, not ground truth — for this reaction it
  *under*-estimates the barrier relative to DFT/experiment.
- **No claim that the pairwise Delta is definitively incapable** of matching the xTB
  saddle — only that it did not, under this acquisition rule and budget. A capacity
  ablation (angular terms, larger cutoff/basis) would be needed to establish a ceiling.
- **No transfer claim** to other reactions, other systems, or foundation-model retraining.

---

## 6. What this implies for Phase 3

The failure mode is informative but **cuts against simply automating this loop**. Before
an acquisition/reward metric is worth building, the more diagnostic experiment is a
**capacity ablation**: hold the acquisition rule fixed and vary the correction's
expressiveness (angular/three-body terms, larger cutoff, more radial channels). If a
richer correction converges under the same 42-query budget, the ceiling is
architectural and an acquisition metric becomes worthwhile. If it does not, no
acquisition function will rescue the loop. A trust-region constraint on the per-iteration
coefficient update, and querying at points where xTB is *near* stationary rather than at
the corrected saddle, are the two other changes the data points to.

---

## 7. Files

```
sweep/corrections/
  phase2_lib.py                phase2_loop.py           phase2_plot.py
  phase2_xtb_worker.py         phase2_water.py          phase2_reaction.py
  phase2/
    iter00..iter05/            delta.json, ts.xyz, scan_max.xyz, sella.traj,
                               sella.log, query/q_*.xyz, query_refs.json, metrics.json
    summary.json  metrics.csv  summary_table.md
  phase2_naive_seed/           archived failed-seeding run (negative control)
  figures/phase2_{barrier,geometry,forces,generalization}.png
```

Fully reproducible from Iteration 0: `/usr/bin/python3 phase2_loop.py --max-iter 5`
(resumable; completed iterations are skipped).
