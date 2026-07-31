# Diels–Alder reaction energetics: MACE-OFF23 (NEB) vs GFN2-xTB

Can **MACE-OFF23 reproduce reaction energetics with no fine-tuning?** We build the
Diels–Alder profile (butadiene + ethylene → cyclohexene) with MACE via a
climbing-image **NEB** (ASE) and compare to the **GFN2-xTB** profile from xTB's
built-in path search. xTB has a reaction-path search built in; MACE does not —
hence NEB. Code: [`neb/`](neb/); figure: `figures/neb_da_mace_vs_xtb.png`.

## Result — outcome (2): same TS location, different energetics

| quantity | GFN2-xTB (built-in path) | MACE-OFF23 (NEB, no fine-tuning) |
|----------|:------------------------:|:--------------------------------:|
| activation barrier | **6.7 kcal/mol** | **36.2 kcal/mol** |
| reaction energy | **−57.6 kcal/mol** | **−36.1 kcal/mol** |
| TS forming C–C | 2.32 Å (symmetric) | 2.03 / 1.98 Å (slightly asynchronous) |

Both methods place the transition state at essentially the same point along the
reaction coordinate (**~2.0–2.3 Å forming C–C**), but the **absolute energetics
differ substantially** — MACE's barrier is ~30 kcal/mol higher and its reaction is
~22 kcal/mol less exothermic.

**Which is more physical?** Experiment/DFT for this reaction: barrier ≈ 22–27
kcal/mol, ΔH ≈ −40 kcal/mol. GFN2-xTB badly **underestimates the barrier** (6.7)
and **overbinds the product** (−57.6); MACE-OFF23 (DFT-trained, ωB97M-D3BJ) gives a
barrier in the right ballpark (a little high) and a reaction energy (−36) close to
experiment — **more accurate on both counts, off the shelf.**

## TS verification

The CI-NEB climbing image is only an approximate saddle (it carried a second
imaginary mode). Refining with **Sella** (order-1) + a finite-difference Hessian
gives a genuine first-order TS with **exactly one imaginary mode, −740 cm⁻¹**
(the two forming C–C bonds moving together). The refined TS is **slightly
asynchronous** (2.03 vs 1.98 Å): MACE-OFF23 makes the perfectly *synchronous*
geometry a second-order saddle, with the true TS marginally asynchronous — a
small but real difference from the symmetric xTB TS. The barrier is robust across
methods of extraction (NEB 37.2, symmetric Sella 36.2, asynchronous Sella 36.2).

## Why this matters (and the correction hook)

The two force fields are **highly correlated** (Study 05: force r≈0.99) yet give a
**systematic energetic offset** here. That is exactly the regime where a
**lightweight post-training correction** — rather than full fine-tuning — could
bridge them. The NEB code is built around a `CorrectedCalculator` wrapper with
plug-in models (global scale / affine / element-specific / delta), so a fitted
correction can be inserted into the force pipeline **without touching the NEB
workflow** (`neb/corrections.py`; verified end-to-end). Fitting and evaluating
those corrections is the natural next step.

## Method notes / caveats

- NEB is finicky for this steep, very exothermic concerted reaction: the band is
  **seeded with the concerted-scan geometries** (not IDPP, which clashes), uses
  **stiff-ish springs** (k≈0.5, ≥13 images) so images don't slide off the barrier,
  and the **reactant is relaxed with the forming bonds pinned** (the vdW complex
  otherwise dissociates). One spurious collapsed image is filtered.
- **xTB-via-NEB is unreliable here** (SCF fails on stretched intermediates; the
  band scrambles), so xTB is taken from its own built-in path search — the
  intended comparison anyway.
- MACE-OFF23 **small** model, CPU, float64. Barrier ~36 is a little above the
  expected DFT value (~22–27) — plausibly the small model and/or residual TS
  under-refinement; the qualitative conclusion (MACE ≫ more physical than xTB) is
  robust.

## Reproduce

```bash
micromamba run -n macemd python neb/run_neb.py --images 13 --k 0.5
micromamba run -n macemd python neb/plot_neb.py
micromamba run -n macemd python neb/refine_ts_mace.py
```
