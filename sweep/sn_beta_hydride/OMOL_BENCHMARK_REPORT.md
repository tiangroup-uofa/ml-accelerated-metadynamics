# MACE-OMOL benchmark on the reconstructed Sn-beta cluster — audit/results report

Date: 2026-11-04. Status: **for review; nothing committed or pushed; website not updated.**
Tests: 142 pass (78 original, unchanged except one input line, see §7). Corrected-rules reassessment: `reassess_status.py`.
System: reconstructed 42-atom vacuum cluster (Tian), charge 0 / singlet **inferred**.
All energies are electronic potential energies (no ZPE/thermal), not free energies.
Environment: mace-torch 0.3.16, torch 2.7.1, ase 3.26.0, float64, CPU. Git base `fbbaa12`.

## 0. Council verdict, final rules, final status (read this first)

**Council (checkpoint #2, snapshot `024086aa`, 17 files, after `git add -N` made the new files visible).**
Security: nothing above low severity (two low items fixed: a home path in a tracked config, checkpoint hash recorded).
Performance: nothing blocking (three tidy-ups applied). Both explicitly declined the methodology questions as out of scope.
**The methodology rulings come from the architecture reviewer alone.** Its key claims were verified against the code before acting.
Its rulings: **Q1** yes, CERTIFIED must require a converged Sella search (missing record ⇒ not certified);
**Q2** yes, each ± side minimum must pass the coordination rule, and an invalid side is neither "connected" nor a
reference; **Q3** keep certification about the saddle and carry NEB status as a separate provenance field;
**Q4** both reference states stay explicit. Council decided no new calculation was necessary.

**Final certification rule — `CERTIFIED` means exactly this (all required):**
1. a *converged Sella search on record* for that structure (`sella_converged` in `ts_sella.xyz`, written by `refine_ts`;
   its test is Sella's projected force AND free-atom fmax ≤ the Sella target). No record ⇒ `NOT CERTIFIED`.
2. Hessian-stage free-atom fmax ≤ 0.01 eV/Å;
3. exactly one significant imaginary mode (below −50 cm⁻¹);
4. that mode has overlap ≥ 0.30 with the hydride-transfer coordinate;
5. a valid geometry;
6. ±mode relaxation (converged) reaches a reactant-like and a product-like minimum, **and neither is structurally invalid**.

**Side-minimum validity.** Each relaxed ± side is judged by `coordination_check` against the **input structure of the same role**
(single convention everywhere): a heavy Sn-shell atom leaving (e.g. the bound water), a heavy atom entering, a broken/formed
heavy-atom bond, the water losing integrity, a hydrogen on Sn, or a hydride-state mismatch ⇒ `STRUCTURALLY_INVALID` ⇒ the pair does
not connect and gives no `connected_side_minimum` reference (that reference is reported unavailable with the reason).
Sn–O lengthening is only a warning (`WEAKENED`). Not a criterion: TS coordination is recorded as information only.

**NEB policy.** The label is about the saddle, never about the path. NEB status is reported separately as
`ts_provenance` (`NEB_CONVERGED` / `NEB_UNCONVERGED: DIAGNOSTIC ONLY` / `NO_NEB_RUN`). An unconverged NEB is never a converged
path, its highest image is never a TS or a barrier, and a converged Sella saddle reached from it remains certifiable.
**Barrier reporting.** Always both `relaxed_endpoint` and `connected_side_minimum`; no bare barrier key anywhere (tested).

**Final status under the corrected rules (reassessed offline from saved raw outputs, no model calls; every barrier recomputed to 0.0 kJ/mol drift):**

| run | saddle | NEB path | barriers | accuracy vs DFT | notes |
|---|---|---|---|---|---|
| MACE-OMOL-1024 (full protocol) | **CERTIFIED** | **UNCONVERGED, diagnostic only** (fmax 0.594 vs 0.05) | PRELIMINARY | BLOCKED ON DFT DATA | Sella 26 steps; −1299 cm⁻¹; overlap 0.72; both sides PRESERVED |
| MACE-OMOL-4M (ladder) | **CERTIFIED** | NO_NEB_RUN | PRELIMINARY | BLOCKED ON DFT DATA | TS seeded from the 1024 saddle; −1039 cm⁻¹ |
| MACE-POLAR (our re-run of Tian's) | **CERTIFIED** + Tian's numbers **REPRODUCED** | NO_NEB_RUN (his path was a relaxed scan) | PRELIMINARY | BLOCKED ON DFT DATA | its Sella record is 0 steps (started at his already-stationary saddle): independent evidence is our Hessian + ±mode connectivity |
| GFN2-xTB | — | — | — | — | **STRUCTURALLY INVALID** FOR BARRIER COMPARISON (reactant loses the Sn-bound water) |

"CERTIFIED" here is scoped: *a verified first-order saddle on that model's potential-energy surface for this reconstructed,
inferred-charge/spin cluster; ±mode connectivity, not an IRC.* It says nothing about accuracy, and the barriers are
electronic-energy differences to that saddle. The corrected rules changed **no** status: the two gaps were latent in these runs
(Sella had converged; no side minimum had lost its coordination). OMOL is **not** claimed more accurate than POLAR, and the paper's
~82 kJ/mol is not used as ground truth.

## Provenance caveat that governs every comparison

`input/tian_tt_test/{reactant,product}.xyz` are **bit-identical** (max |Δx| = 0.0) to Tian's
MACE-POLAR-relaxed endpoints. The xyz header ("hand-built guess (unrelaxed)") is stale;
`provenance.json` states the truth. So OMOL starts inside POLAR's basins. This is not two independent
relaxations from a neutral start.

## 1. Primary OMOL result (extra-large-1024, `run_protocol.py`, unchanged protocol)

| quantity | `relaxed_endpoint` | `connected_side_minimum` |
|---|---|---|
| forward barrier | 72.33 kJ/mol | 90.77 kJ/mol |
| reverse barrier | 99.54 | 95.40 |
| ΔE(P−R) | −27.22 | −4.62 |

Reference-state spread (forward) 18.4 kJ/mol: the TS-connected reactant-side minimum R′ lies 18.4 kJ/mol **below**
OMOL's relaxed R (torsion C1–C2–C4–O5 −87°); P′ lies 4.1 kJ/mol **above** relaxed P.

Saddle evidence (OMOL's own Hessian, recomputed, not saved records): Sella converged (26 steps, fmax 7.1e-4 eV/Å);
exactly one imaginary mode **−1299.0 cm⁻¹**; overlap with hydride coordinate 0.72; H\* participation 0.93;
±0.1 Å mode displacement + BFGS relaxes to reactant-like and product-like minima (not an IRC).
Independent re-evaluation of R, P, TS, R′, P′ with a fresh calculator reproduced the stored energies to ≤6e-11 eV.

**Caveats that travel beside the (saddle-only) CERTIFIED label, in `ts_provenance`:** the 13-image CI-NEB did **not converge** (1500 steps,
fmax 0.594 vs 0.05) and is diagnostic only; Sella was seeded from its highest image, so the NEB is a starting guess and **no converged reaction path or MEP is claimed**. The saddle stands on its own evidence (Sella, Hessian, mode, ±minima), and independent seeds (Tian's POLAR saddle) reach the same saddle.
Charge/spin and structure are inferred/reconstructed. ±mode connectivity is not an IRC. The highest NEB image
(86.6 kJ/mol above image 0) is **not** a barrier and differs from the Sella saddle by 14.3 kJ/mol.
Cost: 17,442 force calls, 3.19 h wall (NEB 16,513 calls).

## 2. OMOL vs POLAR (POLAR column = Tian's saved run; independently re-checked in §2b)

| metric | POLAR (Tian's run) | OMOL-1024 (ours) |
|---|---|---|
| forward barrier, relaxed_endpoint | 112.97 | 72.33 |
| forward barrier, connected_side_minimum | 126.41 | 90.77 |
| ΔE, relaxed_endpoint | −30.02 | −27.22 |
| ΔE, connected_side_minimum | −19.17 | −4.62 |
| TS imaginary frequency | −923.1 cm⁻¹ | −1299.0 cm⁻¹ |
| hydride-mode overlap / H\* participation | 0.75 / 0.85 | 0.72 / 0.93 |
| stationary fmax | 6.9e-4 (Sella, saved) | 7.1e-4 (Sella) ; 6e-4 re-evaluated |
| TS C2–H\* / C1–H\* | 1.31 / 1.35 Å | 1.38 / 1.39 Å |
| TS Sn–O1 / Sn–O2 / Sn–O_water | 2.11 / 2.13 / 2.32 | 2.09 / 2.13 / 2.26 |
| Sn coordination, water integrity (R,P,TS,R′,P′) | all PRESERVED | all PRESERVED |
| R-side conformer offset (R−R′) | 13.4 kJ/mol | 18.4 kJ/mol |
| TS geometry difference | whole RMSD 0.40 Å, reactive-core 0.095 Å | |
| path method | 15-pt relaxed scan + Sella | 13-image CI-NEB (unconverged) + Sella |
| saddle status (corrected rules, §0) | CERTIFIED (our Hessian + connectivity; Tian's numbers REPRODUCED) | CERTIFIED (NEB unconverged, diagnostic only) |
| force calls / wall | not recorded | 17,442 / 3.19 h |

Interpretation (supported by the data):
* Same chemistry: one hydride-transfer saddle, near-identical Sn–O set, same imaginary-mode character; Sella seeded from
  **Tian's POLAR saddle** converges to the identical OMOL saddle (ΔE 0.0 kJ/mol, RMSD 0.001 Å).
* OMOL lowers the forward barrier by **40.6 / 35.6 kJ/mol** (relaxed_endpoint / connected). This is a PES difference, not a
  relaxation artefact: OMOL single points **at POLAR's geometries** give TS−R 74.7 and TS−R′ 93.7 kJ/mol
  (labelled `omol_at_polar_geometries`, not relaxed). POLAR's structures sit 6–20 kJ/mol above OMOL's own minima (per structure: R 16.3, TS 18.7, P 20.0, R′ 15.8, P′ 6.2).
* "Does OMOL worsen endpoint energetics?" **Cannot be answered** — there is no DFT reference yet, and the paper's ~82 kJ/mol is
  a free energy from a different system and is not used as truth. The two models differ most on the connected ΔE
  (−4.6 vs −19.2).

### 2b. POLAR re-run through OUR pipeline (cheap ladder, no NEB) and cross-evaluation

`check_polar_dependency()` resolves (graph_longrange 0.4.0), and a POLAR single point at Tian's saved TS reproduces his energy to
3e-8 eV (torch 2.7.1/ase 3.26 here vs his 2.12/3.28 — no effect). Our endpoint stage needed **0 steps** (input fmax 0.0195/0.0190 ≤ 0.02),
confirming the inputs are POLAR-relaxed; Sella from Tian's saddle needed 0 steps (fmax 6.9e-4 already). The Hessian and ±mode
connectivity were **recomputed**: one imaginary mode −923.08 cm⁻¹, overlap 0.75, H\* 0.85, side minima reactant-/product-like,
barriers **112.97 / 126.41**, ΔE **−30.02 / −19.17 kJ/mol** — identical to Tian's saved numbers. POLAR's saddle is therefore CERTIFIED under the corrected rules (§0), with the caveat that its Sella record is trivial
(0 steps), so the independent evidence is our Hessian and ±mode connectivity (3.5 min total; Hessian 191 s). Note this did not re-derive POLAR's path: Tian's route was a relaxed scan,
ours for OMOL a (non-converged) NEB; the two TS searches were never run the same way.

Pairing check (the like-for-like test): OMOL's relaxed R stays in the input conformer (core RMSD 0.06 Å, no torsion changed ≥ 30°) and
OMOL R′ matches POLAR R′ (core 0.09 Å, no torsion ≥ 30°). So **72.3 ↔ 113.0 (relaxed_endpoint)** and **90.8 ↔ 126.4
(connected_side_minimum)** are like-for-like comparisons.

Cross-evaluation (energies at the *other* method's saved geometries; `…_at_*_geometries` states, not relaxed barriers). Reproducible with `python cross_evaluate.py`, which writes `audit/omol_benchmark/cross_evaluation.json` (20 single points on the committed structures; each model first reproduces its own stored protocol energies to 1e-6 eV):

| | at POLAR geometries | at own geometries | at OMOL geometries |
|---|---|---|---|
| OMOL TS−R / TS−R′ | 74.7 / 93.7 | 72.3 / 90.8 | — |
| POLAR TS−R / TS−R′ | — | 113.0 / 126.4 | 123.5 / 132.5 |

Geometry mismatch costs 6–23 kJ/mol (POLAR's structures in OMOL: +6–20; OMOL's structures in POLAR: +11–23), but the
~40–50 kJ/mol barrier gap persists at identical geometries: it is a **PES difference between the models**, not a relaxation artefact.

## 3. Model-size / training-subset sensitivity (Phase 2)

The installed mace-torch 0.3.16 and the upstream `mace_omol_0` release provide only **two** OMOL checkpoints, both extra-large
(422 MB `…-1024` on 100M data; 414 MB `…-4M` on a 4M subset). **There is no smaller OMOL model.** What was run is therefore a
**training-data** test, not a size test. Cheap ladder (no second NEB): endpoint relaxations → Sella seeded from the 1024 saddle →
Hessian → ±mode connectivity (config `config_omol4m.json`, own output dir).

| | 1024 | 4M | shift |
|---|---|---|---|
| forward barrier, relaxed_endpoint | 72.33 | 73.18 | +0.86 |
| forward barrier, connected_side_minimum | 90.77 | 90.16 | −0.61 |
| ΔE, relaxed_endpoint | −27.22 | −21.20 | **+6.02** |
| ΔE, connected_side_minimum | −4.62 | −6.27 | −1.65 |
| imaginary frequency | −1299.0 | −1039.2 cm⁻¹ | 260 cm⁻¹ |
| overlap / H\* participation | 0.72 / 0.93 | 0.72 / 0.87 | |

Analysis thresholds (not universal): barrier shift > 4 kJ/mol noticeable, > 10 clearly material. Barrier shifts are **below**
noticeable; the relaxed-endpoint ΔE shift (6.0) is noticeable and its **cause was not isolated**. A first draft attributed it to a
tighter Sn–O2 contact in the 4M product; that was **wrong** — the 4M product's Sn–O2 is 2.37 Å vs 2.35 Å for 1024, and **both** OMOL
models register the same `Sn23–O3` bond (O3 = the mapped O2) as formed during product relaxation, measured against the input
product's 2.47 Å. Common to both OMOL checkpoints (not to POLAR, whose input value is 2.47 Å): the product Sn–O2 contracts
2.47 → ~2.35 Å. The 4M TS is **seeded** from the 1024 saddle, so it is not an independent path search. All 4M structures keep the Sn site (PRESERVED); same-role RMSD to 1024 ≤ 0.50 Å whole, ≤ 0.10 Å core.

## 4. Pathway / conformer sensitivity (Phase 3, cheap stage only)

Answered from data in hand; **no alternative-pathway CI-NEB was run** (needs cost gate + Council).
* Reference-state dependence on the reactant conformer is **≥ 13 kJ/mol in both models** (POLAR 13.4, OMOL 18.4) and
  comes from one torsion (C1–C2–C4–O5, −70° to −90°) with the reactive core unchanged (0.02–0.09 Å). So the "~10+ kJ/mol" case
  is **real for the reported barrier** unless the reference conformer is fixed by convention.
* The TS itself is robust to the starting guess (POLAR-TS seed and NEB-image seed → same OMOL saddle).
* **Not established:** that R′ is the lowest reactant, or that no lower-barrier route exists from another conformer.
  No sampled structure is called a global minimum.

## 5. xTB structural sanity (Phase 4)

GFN2-xTB (xtb 6.7.1 / xtb-python in the pre-existing `xtbenv`), inferred charge 0/singlet, BFGS fmax 0.02:
**reactant: bound water leaves Sn (Sn–O_water 2.31 → 3.12 Å; O1 also weakens 2.34 → 2.71 Å); first invalid at step ~135 of 429.**
Product: preserved. Status: **STRUCTURALLY INVALID FOR BARRIER COMPARISON**; no NEB/Sella run. 646 force calls, ~47 s.
The first-draft rule counted H atoms in the Sn shell and falsely flagged the product; it was corrected to heavy atoms
(a water rotation regression test is in the suite), and the final verdict is unchanged.

## 6. What changed in the repository

**Existing modules edited (methodology, Council-approved):**

| file | change |
|---|---|
| `connectivity.py` | `ts_certification` now requires `sella_converged` (keyword, no default); new `assess_sides` (single decision rule), `side_coordination`, `coordination_configured`; `connectivity_check(..., inputs=)` judges each ± side with `coordination_check`; `side_minimum_reference` reports an invalid side as unavailable with the reason |
| `validate_ts.py` | reads `sella_converged` from the saddle's own metadata; passes the input structures to the connectivity check; records TS coordination as information only; records `sella_converged`, `ts_source` |
| `run_protocol.py` | new `ts_provenance()` (NEB status separate from the saddle label); printed and written to `protocol_summary.json` |
| `calculators.py` | `model_path` expands `~` and `$VAR` (no username in tracked configs) |
| `tests/test_protocol_infrastructure.py` | one test input gained `sella_converged=True` (new required evidence; **no assertion changed**) plus added provenance assertions in the end-to-end test |

**New files:** `dft_reference/{gaussian_parser,dft_compare,sparse_budget}.py` (Gaussian ingestion, DFT-vs-MACE metrics,
sparse-label framework — **synthetic fixtures only; never run on a real Gaussian log**), `coordination_check.py`,
`xtb_sanity.py`, `reassess_status.py`, `export_audit.py`, `compare_methods.py`, `run_dft_comparison.py` (never run end-to-end),
`config_omol4m.json`, `config_metad_dryrun.json`, `METADYNAMICS_READINESS.md`, this report,
`tests/{test_dft_reference,test_coordination_check,test_benchmark_artifacts,test_certification_rules}.py`,
and `audit/omol_benchmark/` (curated small results; no trajectories/Hessians; paths sanitized).
`.gitignore` (root + `sweep/sn_beta_hydride/`): runtime outputs, logs, trajectories (scoped to this study), caches.

The `g98.out` files in `sweep/diels_alder/` are xtb's frequency-export imitation (no energies, forces or termination line);
the parser rejects them (tested). They are **not** Gaussian reference data.

## 7. Known issues

**Fixed in this change:** (1) `CERTIFIED` could be emitted with an unconverged Sella; (2) side minima were judged by hydride state only;
(3) no NEB status travelled with a TS result; (4) tracked config hardcoded a home path.

**Still open (not touched):**
1. The output tag ignores `model_path`, dtype, device (`config_io.output_root`); the 4M run needed its own `output_dir`.
2. `PairwiseDelta` (the earlier "winner" correction) has no Sn/Si pair types and fits no energy offset: it would silently do nothing at the Sn site.
3. `CorrectedCalculator` drops charge/spin; use `ChargeSpinCalculator(correction=...)`.
4. Stale text elsewhere: input xyz headers ("unrelaxed"); `test_tian_tt_test.py:165` pins that header; `provenance.json` cites a non-existent
   `tests/test_tian_dataset.py`; README says "electronic state unknown"; **`site/current-work.qmd:99`** ("six O around Sn" for both endpoints; the input
   product has five within 2.46 Å) and **:273** ("MACE-OMOL wrapper validated / Sn support confirmed"). Not edited: the website is untouched.
5. The coordination rule uses a 2.80 Å Sn shell and a 1.2× covalent scale, both analysis thresholds recorded in every result.
6. Checkpoint integrity is recorded (SHA-256 in `audit/omol_benchmark/MANIFEST.json`), not enforced at load time.

## 8. Evidence table

| Claim | Evidence | Status |
|---|---|---|
| Input endpoints are Tian's POLAR-relaxed structures | max \|Δx\| = 0.0 vs saved POLAR xyz; 0 relaxation steps under POLAR; provenance.json | VERIFIED |
| Tian's POLAR numbers (112.97/126.41; −30.02/−19.17; −923 cm⁻¹) | recomputed through our Hessian + connectivity; single point matches to 3e-8 eV; 0.0 kJ/mol drift | REPRODUCED |
| MACE-OMOL-1024 saddle is a first-order saddle connecting valid R-/P-like minima | corrected rules, §0; Sella 26 steps, −1299 cm⁻¹, overlap 0.72, both sides PRESERVED | CERTIFIED (saddle only) |
| MACE-POLAR saddle (our re-run) | same rules; Sella record trivial (0 steps); our Hessian + connectivity | CERTIFIED (saddle only) |
| MACE-OMOL-4M saddle | same rules; seeded from the 1024 saddle | CERTIFIED (saddle only) |
| OMOL 13-image CI-NEB is a reaction path | not converged: fmax 0.594 vs 0.05 after 1500 steps | NOT CLAIMED — unconverged, diagnostic only |
| OMOL and POLAR describe the same hydride-transfer saddle | same imaginary-mode character; TS RMSD 0.40/0.095 Å; Sella from POLAR's saddle ⇒ identical OMOL saddle | VERIFIED |
| OMOL barriers 72.3 (relaxed_endpoint) / 90.8 (connected) kJ/mol; ΔE −27.2 / −4.6 | one run; energies re-evaluated to 6e-11 eV | PRELIMINARY |
| OMOL barrier is 36–41 kJ/mol below POLAR | like-for-like pairing + cross-evaluation at identical geometries | VERIFIED as a model difference; which is closer to DFT: BLOCKED ON DFT DATA |
| OMOL barrier insensitive to training subset (1024 vs 4M) | shifts +0.86/−0.61 kJ/mol; ΔE shift 6.0 unexplained | PRELIMINARY |
| OMOL model-size dependence | no smaller OMOL checkpoint exists | NOT RUN (not possible) |
| Reference conformer shifts the reported barrier by ≥ 13 kJ/mol | R−R′ = 13.4 (POLAR), 18.4 (OMOL), one torsion | VERIFIED (reference-state dependence); alternative pathways NOT RUN |
| xTB cannot be used for this comparison | reactant loses the bound water (Sn–O_water 3.12 Å); final rule re-applied | STRUCTURALLY INVALID |
| Gaussian parser/analysis correct on real Gaussian09 output | synthetic fixtures only | BLOCKED ON DFT DATA |
| Sparse-DFT-budget result (labels vs barrier error) | framework + synthetic tests | BLOCKED ON DFT DATA |
| MACE metadynamics reproduces the two-well surface | checklist + dry-run config only | NOT RUN |
| Website / Current Work update | not edited | NOT RUN |

## 9. Reproducibility

Interpreters: MACE stack `/Library/Developer/CommandLineTools/usr/bin/python3` (3.9; packages in `~/Library/Python/3.9`);
xTB `~/.local/mm/root/envs/xtbenv/bin/python` (pre-existing env: xtb 6.7.1, xtb-python 22.1, ase 3.29.0; nothing installed into it).
New on disk: `~/.cache/mace/mace-omol-0-extra-large-4M.model` (414,102,366 bytes; SHA-256
`1876b02448337595d9417f656f720e700140a1ac6d6d75308dab4aa85cf6d79a`; public `ACEsuit/mace-foundations` release `mace_omol_0`).
Other checkpoint hashes are in `audit/omol_benchmark/MANIFEST.json`.

```bash
cd sweep/sn_beta_hydride
PY=/Library/Developer/CommandLineTools/usr/bin/python3
$PY run_protocol.py --config config_tian_tt_test.json --model mace-omol          # 3.19 h, 17,442 calls (upper bound: xTB/pytest ran alongside)
bash logs_local/omol4m_ladder.sh      # 4M: endpoints -> Sella --start <1024 TS> -> validate_ts   (~12 min wall)
bash logs_local/polar_ladder.sh       # POLAR through our pipeline, no NEB (3.5 min)
~/.local/mm/root/envs/xtbenv/bin/python xtb_sanity.py --config config_tian_tt_test.json   # 646 calls, ~47 s
$PY compare_methods.py --omol outputs/tian_tt_test/mace-omol_extra_large
$PY reassess_status.py outputs/tian_tt_test/mace-omol_extra_large outputs/tian_tt_test_omol4m/mace-omol_extra_large \
    outputs/tian_tt_test/mace-polar_polar-1-s --out outputs/tian_tt_test/reassessment_corrected_rules.json   # no model calls
$PY export_audit.py                    # curated copy -> audit/omol_benchmark/
$PY -m pytest tests -q                 # 142 passed
```
(`logs_local/*.sh` are git-ignored helper scripts; their contents are the stage commands listed in §3 and §2b.) Raw outputs live in the
ignored `outputs/` tree; the curated tracked copy is `audit/omol_benchmark/` (its saved JSON paths are rewritten to `<repo>`, so
`reassess_status.py` must be run on `outputs/`, not on the audit copy). The §2b cross-evaluation is reproduced by
`python cross_evaluate.py` (reads only the committed `audit/omol_benchmark/` structures; ~10 s on CPU) and stored in
`audit/omol_benchmark/cross_evaluation.json`.

## 10. Next actions

**Can do now:** review and decide the commit split (below); correct the stale site/README/provenance text of §7.4 once you approve what goes
public; optionally enforce the checkpoint hash at load time and include `model_path` in the output tag.

**Waiting on Tian's Gaussian09 data:** ingest and eyeball the first real log (`describe_job`); a real `# opt freq` TS log is a compound
job and the job-count/termination check may reject exactly that file type — verify, don't assume; run `run_dft_comparison.py` (never executed
end-to-end); decide the Sn ECP/dispersion/solvent policy (the loader refuses until the manifest acknowledges each); then the sparse-label
experiment, which needs a real TS-region definition and a treatment of the `PairwiseDelta` Sn/Si gap.

**Optional / expensive (cost-gate report + Council first):** alternative-conformer CI-NEBs (R′/P′ and other nearby pairs, ~3 h each at 0.55 s/call);
a full independent 4M NEB (~3 h); a converged NEB (needs a better initial band — would be the only way to claim a reaction path); any metadynamics
(`METADYNAMICS_READINESS.md`).
