# NEB reaction-path module

Generate the Diels–Alder reaction profile with **MACE-OFF23** via a Nudged Elastic
Band (ASE) and compare it to the **GFN2-xTB** profile from xTB's built-in path
search. xTB has a reaction-path search built in; MACE does not — hence NEB.

The force-evaluation pipeline is **modular**: any potential is wrapped in a
`CorrectedCalculator`, so a post-training force correction (global / affine /
element-specific / delta) can be inserted later *without touching the NEB code*.

## Files

| file | role |
|------|------|
| `corrections.py` | `CorrectedCalculator` + correction models (Identity, GlobalScale, Affine, ElementScale, Delta) with `.fit()`; `fit_correction()` helper. The single insertion point for post-training corrections. |
| `calculators.py` | factories building GFN2-xTB / MACE-OFF23 as ASE calculators, correction injected here; MACE model loaded once and shared across images. |
| `neb.py` | calculator-agnostic CI-NEB workflow: endpoint optimization, seeded band, climbing-image relaxation, MEP/TS/barrier/ΔE analysis (with a physical-image guard). |
| `run_neb.py` | driver: MACE NEB vs the xTB scan reference; writes `results/`. |
| `plot_neb.py` | publication-quality comparison figure → `../figures/neb_da_mace_vs_xtb.png`. |
| `verify_ts.py` | frequency check of the CI-NEB TS (ASE Vibrations). |
| `refine_ts_mace.py` | refine the CI-NEB TS to a genuine first-order saddle (Sella) + barrier. |

## Run

```bash
export XTB_BIN=$(which xtb)      # macemd env has xtb + xtb-python + mace + sella
micromamba run -n macemd python run_neb.py --images 13 --k 0.5
micromamba run -n macemd python plot_neb.py
micromamba run -n macemd python refine_ts_mace.py     # clean TS + barrier
```

## Method notes (why the settings)

- **Seed the band with the concerted-scan geometries** (`../cscan/f_*.xyz`), not
  IDPP — IDPP produces clashy interior images for this concerted bond-forming
  reaction and the reported "barrier" becomes a clash artifact.
- **Stiff-ish springs** (`k≈0.5`) and **≥13 images** — at weak springs the band
  slides off the barrier into the very exothermic (−36 to −58 kcal/mol) product
  basin, leaving the TS unresolved.
- **Reactant endpoint relaxed with the two forming C–C bonds pinned** — the
  butadiene+ethylene vdW complex dissociates under free optimization, which ruins
  the path.
- **xTB via NEB is unreliable here** (SCF fails / band scrambles on stretched
  intermediates), so we compare MACE-NEB against xTB's own built-in path search.
- **CI-NEB TS is approximate** (can have a 2nd imaginary mode); `refine_ts_mace.py`
  gives the genuine first-order saddle.

Results + interpretation: [`../FINDINGS_neb.md`](../FINDINGS_neb.md).
