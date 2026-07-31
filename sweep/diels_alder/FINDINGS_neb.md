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

## Extended analysis — where and why the two potentials differ

Four analyses on the finished benchmark (scripts in `neb/`; figures in `figures/`).

### 1. Energy-difference profile — the discrepancy is a *reactive-region* effect
ΔE(s) = E_MACE(s) − E_xTB(s) along the reaction coordinate, using **both-relaxed**
concerted scans (xTB built-in scan + `mace_scan.py`, barrier 36.0 kcal/mol —
consistent with the NEB). It is **~0 for the separated reactants**, rises as the
bonds form, **peaks in the transition-state region (~1.9–2.0 Å)**, and settles to a
**persistent ~+23 kcal/mol offset in the product** (`figures/neb_energy_diff.png`).
The point-wise peak (~+50) is amplified by the two barriers sitting at slightly
different C–C distances; the clean anchors are the **barrier-height difference
(+29)** and the **reaction-energy difference (+23)**. Key point: the discrepancy is
**not a constant offset** — it is zero at the reactant and grows with bond
formation.

### 2. Forces — directionally near-identical, but a magnitude gap localized at the TS
On rattled reaction-path geometries (both potentials off-equilibrium, `gen_rattled.py`),
the MACE and xTB force vectors have **cosine similarity 0.982** (mean; min 0.90) —
they point almost the same way, confirming the supervisor's observation. But the
**force RMSE is 0.52 eV/Å**, and — like the energy difference — **both metrics are
localized in the TS region**: RMSE spikes to ~1.4 eV/Å and cosine dips to ~0.90
around 2.0 Å, versus ~0.3 eV/Å and ~0.99 in the reactant/product basins
(`figures/neb_force_agreement.png`). So the disagreement is a **magnitude effect
concentrated where bonds are forming**.

### 3. Bond evolution — concerted and synchronous for both methods
Both forming bonds contract together (b1 ≈ b2) along the whole path; MACE's refined
TS is only marginally asynchronous (2.03 / 1.98 Å, Δ = 0.05 Å) and xTB's is
symmetric (`figures/neb_bond_evolution.png`). **Both potentials agree the reaction
is a concerted, synchronous cycloaddition** — they differ in energetics, not
mechanism.

### 4. Barrier convergence — the MACE barrier is numerically converged
NEB image counts {9, 11, 13, 17, 21}: the barrier is **35.9 ± 1.0 kcal/mol for
≥11 images** (spread 2.1); 9 images is too few to resolve the TS (barrier collapses
to 18 kcal/mol, TS at 2.5 Å). Cross-validated by the relaxed scan (36.0) and the
Sella-refined TS (36.2) (`figures/neb_barrier_convergence.png`). The reported
barrier is stable.

### Literature context

| | activation barrier | reaction energy |
|---|:---:|:---:|
| Experiment (butadiene + ethylene) | ≈ 27.5 kcal/mol (Ea) | ≈ −40 kcal/mol (ΔH) |
| GFN2-xTB | 6.7 (≈ 21 too low) | −57.6 (≈ 18 too exothermic) |
| MACE-OFF23 (no fine-tuning) | 36.0 (≈ 9 too high) | −36 to −41 (≈ right) |

MACE is **far closer to experiment than xTB overall** — it essentially nails the
reaction energy and is in the right kinetic regime — but it **overestimates the
barrier by ~9 kcal/mol** (vs. its ωB97M-D3(BJ) reference level), i.e. off-the-shelf
MACE-OFF23-small is good on thermodynamics and only fair on this barrier. xTB is
qualitatively wrong on both.

### Why it matters → the post-training-correction framework

The analyses converge on one picture: **MACE and xTB agree on geometry, mechanism,
and force direction, and disagree on energy/force *magnitude*, specifically in the
bond-forming / transition-state region.** This is exactly the regime a *lightweight*
correction can target — and it tells us *which* correction:

- a **global energy shift is ruled out** (ΔE is zero at the reactant, not constant);
- the near-perfect force **direction** agreement with a **magnitude** gap points to a
  **force-scaling** correction (global / element-specific);
- because the gap is **localized in the reactive region**, an **environment- or
  coordination-dependent** correction (or a residual/delta model) is the natural
  form — a single scalar will not capture it.

The NEB pipeline already wraps every potential in a `CorrectedCalculator` with
exactly these plug-in models (global / affine / element / delta), so fitting and
inserting one is the immediate next step — with this benchmark as the target.

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
