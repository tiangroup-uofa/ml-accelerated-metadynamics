# Sn-beta glucose hydride transfer — MACE-OMOL / MACE-POLAR benchmark

**Status: infrastructure only.** Every stage is implemented and tested, but **no
Sn-beta calculation has been run**, because the authoritative reactant and
product coordinates are not available yet. Nothing in this directory is a
Sn-beta result.

## Scientific question

Can the general-purpose MACE foundation models **MACE-OMOL** and **MACE-POLAR**
reproduce the C2 → C1 intramolecular hydride-transfer step of glucose bound to a
partially hydrolyzed Sn-beta site? Specifically:

- Do they keep both endpoint basins stable?
- Do they find a first-order saddle whose imaginary mode is the hydride moving
  from C2 to C1?
- What barrier and reaction energy do they give, compared with the reference
  CPMD work?

Source: S. H. Mushrif, J. J. Varghese, C. B. Krishnamurthy, *Solvation dynamics
and energetics of intramolecular hydride transfer reactions in biomass
conversion*, PCCP **17**, 4961 (2015), DOI
[10.1039/C4CP05063K](https://doi.org/10.1039/C4CP05063K).

## Why OMOL and POLAR (and not MACE-OFF23)

MACE-OFF23 was trained on organic chemistry (H, C, N, O, F, P, S, Cl, Br, I) and
has **no tin**. The models tested here cover Z = 1–83, and each takes the
system's total charge and spin as input. I checked both directly in
mace-torch 0.3.16 (`mace.calculators`):

| key | model | API | notes |
|---|---|---|---|
| `mace-omol` | MACE-OMOL-0 `extra_large` (422 MB) | `mace_omol(model, device, default_dtype)` | ScaleShiftMACE, head `omol`; total charge and spin enter through categorical embeddings |
| `mace-polar` | MACE-POLAR-1 `polar-1-s` / `-m` / `-l` | `mace_polar(model, device, default_dtype)` | PolarMACE with long-range electrostatics; **needs an extra package**, see Setup |

Both models read `atoms.info["charge"]` and `atoms.info["spin"]`, where `spin`
is the **multiplicity** 2S+1. If either key is missing they **silently assume
0 and 1**. To prevent that, `calculators.ChargeSpinCalculator` sets both values
from `config.json` on every call, and refuses any atoms object that already
carries a conflicting value. Both models give different energies for different
charges and multiplicities (checked in the smoke test).

The Diels–Alder MACE-OFF23 workflows are unchanged.

## Setup

Use the Python that already has the stack: `/usr/bin/python3` (3.9) with ase
3.26, torch 2.7.1, mace-torch 0.3.16, sella 2.5.0 and pytest. The `macemd` env
named in older READMEs no longer exists.

- **MACE-OMOL:** works as is. The weights are cached in `~/.cache/mace/`.
- **MACE-POLAR:** also needs `graph_longrange`. With mace-torch 0.3.16 only
  **tag v0.4.0** works. PyPI 0.4.4 and GitHub HEAD both fail at the first energy
  call (`unexpected keyword argument 'force_pbc_evaluator'`). Its dependencies
  (torch, e3nn 0.4.4, numpy, ase) are already installed:

  ```bash
  /usr/bin/python3 -m pip install --user --no-deps \
      "git+https://github.com/WillBaldwin0/graph_electrostatics@v0.4.0"
  ```

  Until it is installed, POLAR shows as SKIPPED in the smoke test, and selecting
  it stops with a message containing this command.

## Workflow

```
validate → optimize endpoints → NEB path → Sella saddle → Hessian / TS validation
```

Run every command from this directory. Each stage accepts `--config`,
`--model {mace-omol,mace-polar}`, `--variant`, `--device` and `--dtype`.
Outputs for each model go to `outputs/<model>_<variant>/`, so OMOL and POLAR
runs never overwrite each other.

| # | command | refuses to run when | main outputs |
|---|---|---|---|
| 0 | `python validate_structure.py --config config.json --json outputs/input_validation.json` | — (exits 1 on FAIL) | readable report + JSON |
| 1 | `python optimize_endpoints.py --config config.json --model mace-omol` | the endpoint pair fails validation (checked **before** the model is loaded), or an element is not supported by the model | `endpoints/{reactant,product}_opt.{xyz,traj,log}`, `endpoints.json`, `validation/endpoints_input_validation.json` |
| 2 | `python run_path.py --config config.json --model mace-omol [--seed-band f.xyz] [--images 13]` | optimized endpoints are missing or invalid, or the initial band has clashing atoms | `path/images/image_XX.xyz`, `band.xyz`, `neb.traj`, `neb.log`, `path.csv`, `path.json`, `ts_candidate.xyz` |
| 3 | `python refine_ts.py --config config.json --model mace-omol [--band … --image N \| --start f.xyz]` | the starting geometry fails validation | `ts/ts_sella.xyz`, `sella.traj`, `sella.log`, `refine_ts.json` |
| 4 | `python validate_ts.py --config config.json --model mace-omol [--ts f.xyz]` | the TS geometry fails validation | `vib/validate_ts.json`, `frequencies.csv`, `hessian.npy`, `imag_mode.xyz` |

Exit codes:

- **0** — success
- **1** — refused, because the input or config is invalid
- **2** — ran, but did not converge or did not verify

`reaction_coordinate.py frames.xyz --config config.json --csv cvs.csv` computes
CVs for any XYZ file or trajectory (for example `path/band.xyz` or
`ts/sella.traj`), for plotting.

### Method choices (and where they come from)

- **NEB** (`run_path.py`): the Diels–Alder NEB defaults —
  `improvedtangent`, climbing image, k = 0.5, 13 images, FIRE, one
  calculator per image around one shared model. The product is
  Kabsch-aligned onto the reactant (non-periodic, no fixed atoms) without
  reordering any atoms. A **seed band** (`--seed-band`) replaces IDPP when
  interpolation goes through clashing geometries, as happened in the DA study.
  The highest image is saved as `ts_candidate.xyz` and is labelled
  **not a verified TS**.
- **Sella** (`refine_ts.py`): repository settings from `phase2_lib.sella_saddle`
  and `aug21_handoff.py` — `order=1`, Cartesian, `fmax = 1e-4` eV/Å, float64.
  Sella checks convergence on its own *projected* forces, which can disagree
  with the plain per-atom force on the free atoms (this happened in the smoke
  test). The record therefore keeps `sella_reported_converged` and
  `fmax_criterion_met` separate. `converged` is true only when both are.
  Cost is reported both as `nsteps` and as genuine calculator calls (the Aug-21
  benchmark found `nsteps` undercounts calls by 1–4×).
- **Hessian** (`validate_ts.py`, `vibrations.py`): the **mass-weighted**
  analysis from `sweep/corrections/phase2_lib.py`, generalized to any atom
  subset and any reference direction. It uses central differences with
  δ = 1e-3 Å, a symmetrized Hessian, ν = 521.47·√λ cm⁻¹, and counts a mode as
  significant imaginary when ν < −50 cm⁻¹. A test confirms it matches
  `phase2_lib` numerically.
  - The raw-eigenvalue × 5142 conversion in `phase1c_*.py` is **not** used.
  - The output reports every frequency, the dominant imaginary mode, its
    Cartesian displacement, how much each atom takes part in it, and its
    overlap with the gradient of `delta_H`.
  - The verdict is "candidate" only when the structure is stationary (free-atom
    fmax ≤ `stationary_fmax`) **and** has exactly one significant imaginary mode
    aligned with the hydride transfer. Anything else is reported as it is.
  - With fixed atoms, the Hessian covers the free atoms only. The output marks
    this as an approximation.

## Reaction coordinates (`config.json` → `cvs`)

All CVs are defined by atom-map names. No atom index is hard-coded in the code.

| CV | definition | expected sign / range |
|---|---|---|
| `d_C1_H`, `d_C2_H` | distance from C1 or C2 to H\* (minimum image if periodic) | Å |
| `delta_H` | d(C2–H\*) − d(C1–H\*) | < 0 reactant, ≈ 0 near a symmetric TS, > 0 product |
| `CV1`, `CV2` | coordination number CN = Σ<sub>i∈center</sub> Σ<sub>j∈group</sub> s(r<sub>ij</sub>), with s = (1−x<sup>n</sup>)/(1−x<sup>m</sup>) and x = (r−d0)/r0 | **UNCONFIRMED** template |

The CPMD basins quoted for this work, reactant (CV1, CV2) ≈ (0.9, 0.9) and
product ≈ (1.8, 0.1), are stored under `reference_basins`. They are used **only
as a diagnostic**: the code reports the distance to each basin and never passes
or fails a structure on it.

The shipped `CV1` = CN(C1; H_C1, H\*) and `CV2` = CN(C2; H\*) (r0 = 1.5 Å, n = 6,
m = 12) are my inference from those basin values. I could not access the paper's
CV definitions. **Replace them with the original definitions before comparing
against the basins.**

## Validation layer (`validate_structure.py`)

**Hard errors** — the run is refused:

- a file that is missing, unreadable, or contains more than one frame
- non-finite coordinates
- a periodic structure with a degenerate cell
- duplicate atoms (< 0.1 Å) or impossible contacts (< 0.5 Å)
- isolated atoms
- a required element (Sn) missing, or the atom count ≠ `expected_n_atoms`
- an atom map that is unset, out of range, duplicated, or points to the wrong
  element
- charge or multiplicity unset, or an electron count inconsistent with the
  multiplicity
- endpoint bond expectations violated: in the reactant H\* must be bonded to C2
  and not C1, and the reverse in the product. This catches swapped files and
  wrong H indices.
- reactant and product with different atom counts or element order

**Warnings** — the run continues:

- short contacts
- bonds that change between the endpoints other than C2–H\* breaking and
  C1–H\* forming
- large movements of non-reactive atoms between the endpoints (possible atom
  permutation)
- fixed atoms that are not identical in both endpoints
- Sn without an O neighbour
- CVs closer to the other endpoint's reference basin

**Diagnostics only:**

- formula and element counts
- cell and PBC
- fragments
- the Sn coordination shell: neighbours, distances, CN, number of O
- reactive-atom distances
- CVs

## Smoke-test structures vs scientific structures

`run_smoke_test.py` and `tests/` use a **TEST-ONLY** toy:
CH₃O⁻ + H₂C=O degenerate hydride exchange (9 atoms, charge −1, C1/C2 pinned).

- It is built in code by `tests/toy_hydride.py`.
- It is written only to the git-ignored `smoke_output/`, as `TEST-ONLY_*` files.
- Every stage prints a TEST-ONLY banner for it, and every JSON it produces
  carries `"TEST_ONLY": true`.

It exists to show that the code runs. **Its energies, barriers and frequencies
are not chemistry and must never be reported.** The scientific config
(`config.json`) has `test_only: false`, and its atom map is deliberately
unconfigured.

```bash
/usr/bin/python3 -m pytest tests -q            # 36 unit tests, no model weights needed
/usr/bin/python3 run_smoke_test.py             # imports, config, validation, CVs, Hessian, both models
/usr/bin/python3 run_smoke_test.py --pipeline  # + all four stages end to end on the toy
```

## Blocked on authoritative coordinates

The following need the real structures:

- `input/reactant.xyz`, `input/product.xyz` and their provenance
- the `config.json` fields `system.charge`, `spin_multiplicity`,
  `expected_n_atoms`, `atom_map` and `fixed_atoms`
- the confirmed CV definitions
- whether the original model was periodic or a cluster. OMOL was trained on
  molecular (non-periodic) data, so a periodic zeolite would be outside its
  training domain. That has to be reported with any result.
- the real choices about cost: the Hessian scope and the number of NEB images.
  OMOL extra-large took about 0.2 s per call on a 9-atom CPU test, and the
  Hessian alone needs 6N force calls.

## Files

| file | role |
|---|---|
| `config.json` | scientific configuration (atom map, charge/spin, CVs, stage settings) — **unconfigured** |
| `config_io.py` | config loading, path resolution, atom-name → index resolution |
| `calculators.py` | MACE-OMOL / MACE-POLAR registry, charge/spin wrapper, POLAR dependency check |
| `validate_structure.py` | strict pre-calculation validation (CLI + library) |
| `reaction_coordinate.py` | config-driven CVs (+ gradients, basin diagnostics, CSV) |
| `geometry.py` | distances with periodic boundaries (minimum image), bond graph, Kabsch (same as `phase2_lib`) |
| `vibrations.py` | Hessian + mass-weighted mode analysis (generalized `phase2_lib`) |
| `workflow.py` | shared stage helpers (outputs, fixed atoms, E/F records, TEST-ONLY tagging) |
| `optimize_endpoints.py`, `run_path.py`, `refine_ts.py`, `validate_ts.py` | the four stages |
| `run_smoke_test.py` | TEST-ONLY software smoke test |
| `tests/` | pytest suite + TEST-ONLY toy builder |
| `input/README.md` | what to put in `input/` and how to configure it |
