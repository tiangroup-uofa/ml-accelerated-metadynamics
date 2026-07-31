# Post-training corrections for MACE-OFF23 — how much error can a lightweight model remove?

## What was tested

Following Dr. Tian's direction: since MACE-OFF23 and GFN2-xTB forces are highly
correlated, can a **lightweight post-training correction** (no retraining of the
foundation model) remove the systematic MACE↔xTB discrepancy? We fit four
corrections on **paired MACE/xTB forces** and evaluate them end-to-end.

- **Base** = MACE-OFF23; **reference / target** = GFN2-xTB (a cheap stand-in for
  the eventual DFT reference — the *machinery* is what is being validated).
- **Dataset:** 93 configurations with meaningful forces — water-droplet MD frames
  (n=20/30/50) + rattled Diels–Alder reaction-path geometries — split 80/20 train/val.
- **Corrections** (all via the existing `CorrectedCalculator` API):

  | model | form | conservative | # params |
  |---|---|:---:|---|
  | global | F′ = αF, E′ = αE | yes | 1 |
  | affine | F′ = αF, E′ = αE + β | yes | 2 |
  | element | F′ᵢ = α₍Zᵢ₎ Fᵢ | **no** | per-element |
  | delta | ΔE = Σ fitted radial pair potentials, F′ = F − ∇ΔE | yes | 48 (linear fit) |

## Results

| model | val force RMSE (eV/Å) | RMSE ↓ | DA barrier | DA ΔE | droplet Rg (Å) | conservative |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline (MACE) | 0.411 | ref | 36.0 | −40.6 | 3.94 | yes |
| global | 0.377 | 8% | 32.3 | −36.5 | 3.92 | yes |
| affine | 0.377 | 8% | 32.3 | −36.5 | 4.00 | yes |
| element | 0.373 | 9% | 36.0 | −40.6 | 4.02 | **no** |
| **delta** | **0.347** | **16%** | **11.7** | **−57.1** | 3.71 | yes |
| **xTB (target)** | 0 | — | **6.7** | **−57.6** | **3.89** | — |

### 1. Forces are already directionally aligned — corrections fix *magnitude*
Baseline force cosine is already 0.985; the corrections barely change direction
and instead shrink the magnitude error. Scalar corrections remove ~8–9% of the
force RMSE; the **delta removes 16%** (best), because it captures the *localized*
residual (per element-pair, radial) rather than a single scale.

### 2. Diels–Alder energetics — the delta is transformative, the rest are not
A modest force-RMSE improvement translates into a **large energetics correction**,
because the delta corrects the bonding-region forces that integrate into the
reaction energetics:

- **delta:** reaction energy **−40.6 → −57.1** (target −57.6 — essentially exact);
  barrier **36.0 → 11.7** (target 6.7 — ~68% of the gap removed).
- **global / affine:** only apply the scalar α (barrier 36 → 32) — a global scale
  cannot fix a reactive, localized discrepancy.
- **element:** force-only, so it leaves the **energetics uncorrected** (barrier 36).

### 3. Water droplet — MACE was already right; corrections don't help (and can hurt)
Baseline MACE already reproduces the xTB droplet (Rg 3.94 vs 3.89, identical O–O
peak). No correction improves structural agreement; global shifts the O–O peak,
and the **delta over-compacts the droplet (Rg 3.71)** — the correction, dominated
by the reactive data, mildly degrades the already-accurate equilibrium structure.

### 4. Conservativeness matters
Element-specific *force* scaling is **not conservative** (no energy whose gradient
gives per-element-scaled forces) — it corrects nothing energetically and would
inject energy in MD/metadynamics. Global, affine, and delta are conservative.

## Why it matters

- **How much systematic error can be removed without retraining?** For the
  *reactive* discrepancy that dominates: a conservative delta removes **~all of the
  reaction-energy error and ~2/3 of the barrier error**, from a linear fit on cheap
  reference forces. Scalar corrections remove almost none of it.
- The result validates the **Δ-ML route to the project goal**: a lightweight,
  conservative correction can pull the fast MACE surrogate toward a reference
  method's energetics — the machinery needed for "MACE surrogate + correction →
  reference-level trajectory", with DFT swapped in for xTB later.

## Recommendation

**Use the conservative pairwise **delta** correction** for future active-learning
and metadynamics workflows:

- it is the only correction that meaningfully removes the systematic *energetics*
  error (the quantity that matters for reactions and free energies);
- it is **conservative** — safe for MD/metadynamics (verified: stable droplet MD);
- it is lightweight (linear least-squares, no foundation-model retraining) and
  drops into the existing `CorrectedCalculator` with no workflow changes.

**Do not use** element-specific force scaling (non-conservative, energetically
inert). Global/affine scaling is safe but too weak to be worth it.

**Caveats / next steps.** (i) The delta was fit on data spanning both chemistries;
its strong DA result is in-domain — deploy it fit on reference data representative
of the target system. (ii) It slightly over-compacts equilibrium water, so validate
on equilibrium properties and consider domain-weighted or regularized fits.
(iii) A ~5 kcal/mol barrier residual remains — a richer delta (more basis / angular
terms) or eventual fine-tuning would close it; the point here is that a *linear,
conservative* correction already removes the bulk cheaply. (iv) In production the
reference is DFT, not xTB — the same pipeline applies.
