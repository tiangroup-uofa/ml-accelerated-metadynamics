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

## Literature evidence (audit of Oct 2026)

**What could be read.** The PCCP paper is closed access: RSC refused scripted
access, and Unpaywall/OpenAlex list no open copy. **Neither the paper text nor
its Supporting Information has been read.** Two sources were used:

- **[P]** the PubMed abstract (PMID 25591500)
- **[T]** M. A. Y. Ali, MSc thesis, University of Alberta (2025). It cites the
  paper as ref. [54]. Its Fig. 4.1 is captioned as adapted from the paper's
  Fig. 1. Its Ch. 4–7 describe Ali's own **gas-phase** CPMD-metadynamics dataset.

Labels: **stated [P]/[T]** = written in that source; **as drawn** = our reading
of a figure; **inferred** = our interpretation; **unknown** = not found. Nothing
below has been checked against the paper's full text.

**Calculations** — kept separate, and each attribute is cited on its own:

| calculation | catalyst model | solvent | method | source |
|---|---|---|---|---|
| paper, main study | **unknown from [P]**. [T] Fig. 4.1a/b (adapted from the paper) draws a SiH₃-capped Sn cluster with solvent markers. | explicit water or methanol, quantum-mechanical — [P]. [T] says only "solvent molecules". | CPMD-metadynamics, finite T — [P] | [P] + [T] figure |
| paper, no-solvent run | unknown | none ("in the absence of any solvent") — [P] | CPMD-metadynamics — [P] | [P] |
| paper, implicit-solvent comparison | unknown | implicit continuum — [P] | DFT — [P] | [P] |
| **Ali's dataset** (20,000 timesteps) | Sn defect site, three framework O + –OH (§4.2), drawn as Sn(–O–SiH₃)₃ (Fig. 4.1) | **gas phase** (Ch. 4, 5, 7 titles) | CPMD-metadynamics | [T] |

Two things are **unknown**:

- Whether Ali's gas-phase dataset *is* the paper's no-solvent run, or a separate
  run.
- Whether the paper's solvated runs use the same cluster and cell.

The setup of Ali's dataset must not be generalized to the paper's solvated
calculations.

**System of the hydride step:**

- **Catalyst** — stated [T] §4.2: a partially hydrolyzed Sn defect site, Sn
  tetrahedrally bonded to three framework O plus an –OH. It is drawn as
  Sn(–O–SiH₃)₃(OH) (Fig. 4.1; caption: "Sn-beta cluster model").
- **Glucose binding** — stated [T] §4.2: glucose binds Sn through the C1
  carbonyl and the C2 hydroxyl, forming an octahedral complex. The Fig. 4.1a
  caption says Sn is octahedrally coordinated by 6 O. That statement is for
  this *pre-deprotonation* complex.
- **Reactant of the hydride step** — stated [T] §4.2: C2–OH is deprotonated
  and the proton goes to Sn–OH. The water formed "remains coupled to Sn", and
  C2–O–Sn is covalent. Glucose is drawn open-chain (Fig. 4.1).
- **Sn coordination in the hydride-step reactant and product** — **as drawn**
  (Fig. 4.1b/c): 3 O–SiH₃ + O1 + O2 + H₂O, i.e. 6 O. This count is our reading
  of the drawing, not a statement in the text.
- **After the transfer** — **as drawn** (Fig. 4.1c): C1 carries H₁ and H₂, C1–O
  is drawn bonded to Sn, and C2=O is drawn coordinated (dashed) to Sn.
- **Cell** — stated [T] §4.2, for Ali's dataset: 18 × 14.5 × 16 Å, with the
  reactant and catalyst "positioned isolated". Whether the paper's solvated
  cells match is unknown.
- **Periodic zeolite framework** — none appears in either source read. That
  the paper used no periodic framework is **inferred**, not verified.

**Electronic state — unknown.** Neither source states charge or multiplicity. A
neutral closed-shell cluster is plausible (inferred) but is **not** set in the
config.

**Reference quantities:**

- **Paper, [P] abstract only:**
  - methanol's activation free-energy barrier is 50 kJ/mol higher than water's
  - the step is exergonic in water and endergonic in methanol
  - implicit-solvent DFT barriers are "almost identical"

  The paper's numerical barriers and reaction free energies were **not
  available**.
- **Values from Ali's gas-phase FES — not verified as published PCCP values.**
  [T] Fig. 7.3 is captioned as the ground-truth FES for the gas-phase reaction,
  "adapted from Mushrif et al.". As reproduced in the thesis it is a MATLAB plot
  ("Free energy from metadynamics"). Its data tips:

  | data tip | CV1 | CV2 | level (kJ/mol) |
  |---|---|---|---|
  | reactant | 1.042 | 0.901 | −99.52 |
  | point between basins | 1.425 | 0.642 | −17.91 |
  | product | 1.677 | 0.25 | −110.61 |

  From these data tips:
  - barrier ΔF‡ ≈ **81.6 kJ/mol** (the thesis text says ≈ 82)
  - reaction ΔF ≈ **−11.1 kJ/mol**. The thesis text says −10.6, using −110.1;
    that does not match the −110.61 data tip.

  These are differences on a metadynamics free-energy surface for the
  **gas-phase** run only. Absolute levels are offsets of the reconstructed
  bias (inferred). The middle data tip is not shown to be the true saddle
  point. Convergence is unknown. These values are not electronic energies and
  not the paper's solvated results.

**Metadynamics settings** — stated [T], for Ali's dataset only:

- 20,000 timesteps (§4.2)
- the `colvar_mtd`/`parvar_mtd` excerpts shown (Figs. 7.1–7.2): hills every 100
  steps, width 0.1, height 0.001 Ha. The thesis reads the constant height as
  suggesting non-well-tempered metadynamics.

**Unknown:**

- functional, pseudopotentials, cutoff, timestep, temperature, thermostat
- the number of solvent molecules, total atom counts
- the CV parameters d⁰, p, q

**Unresolved:** the **~44,000-step** figure. It appears in neither source; its
origin needs to be clarified with Tian.

**Coordinates:** none in either source read.

**What this suggests here (our recommendation, not a literature result).**
Tian's immediate experiment is an endpoint-to-endpoint path with OMOL/POLAR,
initially without the full solvent environment. For that, a cluster in vacuum
is the natural setup, and it lies within OMOL's molecular (non-periodic)
training domain; PBC are only needed if the received files are periodic.
NEB/Sella give potential-energy barriers, which are **not** comparable
one-to-one with a metadynamics free-energy surface.

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
| `CV1` | CN(C1; H_C1, H\*) = coordination number of **C1 with H₁ and H₂** | atoms confirmed (secondary), parameters **placeholders** |
| `CV2` | CN(C2; H_C1, H\*) = coordination number of **C2 with the same H₁ and H₂** | atoms confirmed (secondary), parameters **placeholders** |

Here CN = Σ<sub>j</sub> (1−x<sup>n</sup>)/(1−x<sup>m</sup>) with x = r/r0. This is
the CPMD rational coordination function of [T] eq. 4.1 (exponents p and p+q),
written with r0 = d⁰, n = p and m = p + q.

- **Atoms:** taken from [T] Fig. 4.1c, adapted from the paper's Fig. 1c.
  An earlier version used only H\* for CV2, which was wrong.
- **Parameters:** the values r0 = 1.5 Å, n = 6, m = 12 are **placeholders**,
  because d⁰, p and q are unknown. Both CVs carry
  `"parameters_confirmed": false`.
- **Effect on the reference basins:** `basin_diagnostics` still lists the
  distances to the stored basins — reactant (0.9, 0.9) and product (1.8, 0.1)
  as quoted in [T] §4.2 — but marks them **not comparable**. It withholds both
  "within tolerance" and "nearest basin" until the original parameters are
  entered and the flag is set to `true`.

## Validation layer (`validate_structure.py`)

**Hard errors** — the run is refused:

- a file that is missing, unreadable, or contains more than one frame
- non-finite coordinates
- a periodic structure with a degenerate cell
- duplicate atoms (< 0.1 Å) or impossible contacts (< 0.5 Å)
- isolated atoms
- a required element (Sn, Si, C, H, O) missing, or the atom count ≠
  `expected_n_atoms`
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
- `sn_site` expectations: the mapped Sn should have 6 O within `sn_shell_A`,
  including O1, O2 and O_water. These come from our reading of thesis Fig. 4.1
  (see above). They are **warnings only**, never acceptance criteria.
- CVs closer to the other endpoint's reference basin (only once the CV
  parameters are confirmed)

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
/usr/bin/python3 -m pytest tests -q            # 40 unit tests, no model weights needed
/usr/bin/python3 run_smoke_test.py             # imports, config, validation, CVs, Hessian, both models
/usr/bin/python3 run_smoke_test.py --pipeline  # + all four stages end to end on the toy
```

## Blocked on authoritative coordinates

The following need the real structures:

- `input/reactant.xyz`, `input/product.xyz` and their provenance
- the `config.json` fields `system.charge`, `spin_multiplicity`,
  `expected_n_atoms`, `atom_map` and `fixed_atoms`
- the CV parameters d⁰, p, q (the atoms are known; see above), ideally from the
  original CPMD input
- the solvent content of the received files. Tian's immediate experiment is
  initially without the full solvent environment; if the files contain
  solvent, decide with Tian whether to strip it.
- which cap atoms to freeze, if any (`fixed_atoms`)
- the paper's own reaction-energy and barrier values, which need the full text
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
