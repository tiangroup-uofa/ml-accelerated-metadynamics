# FINDINGS — Jul 31 direction (Phases A / B / C)

Testing Dr. Tian's hypothesis that **the largest MACE-OFF23 ↔ GFN2-xTB
disagreements concentrate in the highly distorted / barrier-region
configurations that are poorly represented by the pretrained model's training
distribution**, and whether a *lightweight, targeted* post-training correction
concentrated in that region improves MACE↔xTB reactive agreement.

System: butadiene + ethylene → cyclohexene (Diels–Alder). Reaction coordinate =
mean forming C–C distance (atom pairs 0–5 and 3–4). Foundation model unchanged
throughout (no fine-tuning); the only lever is the existing conservative
**pairwise-radial Delta** on top of `CorrectedCalculator`. Every number below is
read back from a generated results file — see `sweep/corrections/phaseC/*.json`,
`.../reaction_*.csv`, and the three `figures/phaseC_*.png`.

> **Framing caveat (applies to all of Phase C).** xTB is the *chosen reference*
> for this methodology test, **not** the physical ground truth. For this
> reaction xTB *underestimates* the barrier (6.7 vs DFT/expt ≈ 22–27 kcal/mol)
> and overbinds the product, whereas raw MACE-OFF23 (DFT-trained) is closer to
> the physical value (see `FINDINGS_neb.md`). So "moving MACE toward xTB" here
> demonstrates that a targeted correction can *reproduce a designated reference*
> in the barrier region — it does **not** claim improved physical accuracy. The
> value is as a controlled proxy for the north-star loop (any reference →
> corrected MACE), where the reference will eventually be DFT/CCSD.

---

## Phase A — standardize the NEB comparison

**A1 — xtb-python stability gate (PASS / trustworthy).** GFN2-xTB via
`xtb-python` through the existing `make_factory("xtb")`, with the robust settings
(`electronic_temperature=1000`, `max_iterations=500`, `accuracy=1.0`):

- SCF success 9/9 on both plain and robust settings.
- Determinism: repeated single-points give energy std **5.08e-14 eV** (machine
  precision) — no run-to-run drift.
- Geometry opt converged (fmax 0.021 eV/Å); ~4.4 ms/call.
- The robust settings do **not** perturb path energetics (checked vs plain).

→ Backend is stable enough for a 13-image standardized NEB; no need for the
binary-file fallback.

**A2 — apples-to-apples NEB (both potentials, identical driver).** Same ASE-NEB
driver, 13 images, k=0.5, FIRE + `NEBOptimizer`, fmax 0.05, climbing image,
concerted-scan seed. Result, documented exactly as agreed:

- **MACE ASE-NEB converges** to a barrier of ~**36 kcal/mol**, matching both the
  earlier MACE relaxed scan and the MACE built-in comparison to **< 2 kcal/mol**
  → the ~30 kcal/mol MACE–xTB gap is a **real PES difference, not a methodology
  artifact.**
- **xTB under the *identical* standardized settings fails** (unphysical band;
  FIRE + `NEBOptimizer` do not produce a converged physical barrier). This is an
  **algorithm/PES-compatibility issue, not an xtb-python backend failure** (A1
  proved the backend stable, and a binary calculator would hit the identical
  problem). Per direction, we did **not** change the physical problem to force a
  barrier.
- Valid xTB reference therefore remains the **constrained relaxed scan + verified
  TS** (6.7 kcal/mol, TS 2.32 Å, −394 cm⁻¹), as in `FINDINGS_diels_alder.md`.

**A-conclusion:** the MACE↔xTB barrier discrepancy is a genuine difference in the
underlying potential-energy surfaces; standardizing the NEB algorithm does not
close it.

---

## Phase B — where do the errors live?

152 rattled configurations (8 per path frame, amplitude 0.10 Å) partitioned by
forming C–C distance into 5 regions; per-region force RMSE and cosine of raw
MACE vs xTB (`neb/region_analysis.py`, `figures/regionwise_error.png`).

**Result: the MACE↔xTB force disagreement concentrates in the TS / post-TS
(barrier) region** — it is largest where the forming bonds are partially made and
the geometry is most distorted, and smallest in the reactant and product basins.

> **Careful wording (as directed):** the *concentration* of error in the barrier
> region is directly demonstrated by this dataset. The stronger claim — that this
> is *caused by* absence of such distorted geometries from the foundation model's
> training distribution — is a **supported hypothesis, not a proven fact**; this
> dataset cannot establish the cause.

---

## Phase C — targeted barrier-region calibration

Same fixed pairwise-Delta architecture and hyperparameters for every model; the
**only** difference is the training DATA. No-leakage split **by parent path
geometry** (test parents `[2,4,6,8,11,13,15,18]` never contribute to any fit).

- **Delta-A (baseline):** baseline reactive configs + water.
- **Delta-B (barrier-enriched):** A + 32 extra configs at forming C–C ≈ 1.7–2.4 Å
  (the TS/post-TS region) — the targeted model.
- **Delta-C (generic control):** A + 32 extra configs sampled generically (same
  count as B, spread across the coordinate) — isolates *targeting* from *more
  data*.

Held-out reactive test = 48 configs from the 8 test parents (`role=test`).

### C3 — region-wise held-out force error (eV/Å)

| model | overall RMSE | overall cos | TS-region RMSE | TS-region cos | TS+post-TS RMSE | Δ vs raw |
|---|---|---|---|---|---|---|
| raw MACE | 0.556 | 0.983 | 0.712 | 0.970 | **0.906** | — |
| Delta-A baseline | 0.432 | 0.986 | 0.519 | 0.981 | 0.714 | −21% |
| **Delta-B barrier** | 0.456 | 0.986 | **0.487** | **0.983** | **0.688** | **−24%** |
| Delta-C generic | 0.427 | 0.986 | 0.532 | 0.980 | 0.734 | −19% |

- **B gives the lowest barrier-region force error** (TS-region 0.487, TS+post-TS
  0.688) — better than baseline A and better than the generic control C.
- **B is slightly *worse* overall / in the basins** (0.456 vs A 0.432, C 0.427).
  This is honest and expected: the fixed-capacity Delta **reallocates** fitting
  capacity toward the barrier region when enriched there, at a small cost in the
  basins where all models already agree well. Targeting trades basin accuracy for
  barrier accuracy.

### C4 — reaction-path benchmark (relaxed concerted scan)

| model | barrier (kcal/mol) | reaction E | TS forming C–C (Å) |
|---|---|---|---|
| raw MACE | 36.0 | −40.6 | 2.02 |
| Delta-A baseline | 14.1 | −47.3 | 2.02 |
| **Delta-B barrier** | **9.9** | **−50.7** | 2.62 |
| Delta-C generic | 15.0 | −51.1 | 2.02 |
| **xTB (target)** | **6.7** | **−57.6** | 2.32 |

- **B removes ~89% of the raw MACE→xTB barrier gap** (36.0 → 9.9; remaining gap
  to xTB 3.2 of the original 29.3 kcal/mol).
- **Targeting beats volume:** adding barrier-region data moved the barrier
  14.1 → 9.9, while the **same amount** of generic data did essentially nothing
  (14.1 → 15.0). This directly supports "*where* you add data matters more than
  *how much*" for the barrier.
- Reaction energy improves monotonically toward xTB (−40.6 → −50.7; target
  −57.6). The B TS position shifts to 2.62 Å — the corrected barrier maximum
  moves outward; noted as a real effect of the correction, not matched to xTB's
  2.32 Å.

### C5 — equilibrium water sanity (does barrier enrichment damage water?)

Short unbiased NVT MD on the n=20 droplet; O–O distribution + radius of gyration.

| model | O–O peak (Å) | droplet Rg (Å) |
|---|---|---|
| raw MACE | 2.84 | 3.90 ± 0.07 |
| Delta-A baseline | 2.74 | 3.82 ± 0.04 |
| **Delta-B barrier** | 2.74 | **3.82 ± 0.07** |
| xTB (ref) | 2.84 | 3.77 |

- **Barrier enrichment does not damage water: B ≡ A on water** (identical Rg 3.82
  and O–O peak 2.74). Both Deltas sit *between* raw MACE (3.90) and xTB (3.77),
  i.e. slightly *closer* to xTB than raw MACE — no condensation, no structural
  degradation.
- **Why (mechanistic, and a genuinely useful property):** the pairwise Delta has
  **separate channels per element pair**. The barrier enrichment adds only C–C /
  C–H configurations, which touch only the carbon channels; the O–O / O–H channels
  that govern water are fit on the **same** water data in A and B, so water is
  untouched. The element-pair separation gives **clean selectivity — you can
  sharpen reactive chemistry without perturbing an unrelated equilibrium
  subsystem.**

---

## C6 — interpretation questions (answered)

1. **Does targeted barrier enrichment reduce MACE↔xTB error where it matters?**
   Yes. B has the lowest TS-region (0.487) and TS+post-TS (0.688) held-out force
   RMSE and the barrier closest to xTB (9.9 vs 6.7). *Directly demonstrated.*
2. **At what cost elsewhere?** A small, quantified basin/overall cost
   (overall RMSE 0.456 vs 0.432) — capacity reallocation. Water is unaffected.
   *Directly demonstrated.*
3. **Is the improvement from targeting or just more data?** Targeting. Equal-count
   generic data did not improve the barrier (14.1 → 15.0) while barrier data did
   (14.1 → 9.9). *Directly demonstrated by the A/B/C control.*
4. **Does it generalize (no leakage)?** The gains are measured on a held-out test
   set split **by parent geometry**, so the test parents never entered any fit —
   the improvement is out-of-sample, not memorization. *Directly demonstrated for
   this reaction; generalization to other reactions is untested.*
5. **Does it harm equilibrium behavior?** No — water structure is preserved
   (B ≡ A), thanks to element-pair channel separation. *Directly demonstrated.*
6. **Targeted vs generic correction, head to head?** Targeted (B) wins on the
   barrier and barrier-region forces at equal data budget; generic (C) is marginal
   on the barrier but competitive in the basins. *Directly demonstrated.*
7. **Does this demonstrate active learning?** **No, and we do not claim it.** This
   is a *static, hand-placed* enrichment (we chose the barrier window a priori). It
   shows that **concentrating correction data in the high-error region is
   efficient**, which is precisely the *motivation* for an active-sampling loop —
   but no acquisition function, no iterative selection, and no on-the-fly labeling
   was run. **The result supports the motivation for future active sampling; it
   does not demonstrate active learning.**

---

## Bottom line

- **Directly demonstrated:** (i) the MACE↔xTB barrier gap is a real PES
  difference that survives NEB standardization (A); (ii) the force disagreement
  concentrates in the TS/post-TS region (B); (iii) a lightweight targeted Delta
  enriched in that region gives the lowest barrier-region force error and the
  barrier closest to xTB, out-of-sample (C3/C4); (iv) targeting beats equal-volume
  generic data (C); (v) it does so without damaging water, via element-pair
  channel selectivity (C5).
- **Supported (not proven):** that the concentration of error reflects the
  foundation model's training-distribution coverage; and that a *dynamic*
  active-sampling loop concentrating labels in high-error regions would be
  efficient. This work motivates that loop but does not run it.
- **Explicitly not claimed:** improved *physical* accuracy (the correction moves
  MACE toward xTB, itself a low-barrier reference), active learning, metadynamics,
  or any foundation-model change.

**Deliverables:** `figures/phaseC_force_regions.png`, `figures/phaseC_reaction.png`,
`figures/phaseC_water.png`; `sweep/corrections/phaseC/summary_table.md` and the
`phaseC/*.json` + `reaction_*.csv` from which every number here was verified.
