# Blind metadynamics with MACE (OMOL / POLAR): readiness — NOT LAUNCHED

Nothing in this file has been run. It records what exists, what is missing, and a
validation checklist to pass **before** any production run. Companion config:
`config_metad_dryrun.json` (unknowns left `null`).

## What exists vs. what is missing

| need | status | where |
|---|---|---|
| MACE-OMOL / MACE-POLAR calculators with charge/spin on every call | **exists** | `calculators.py` `ChargeSpinCalculator` (`correction=` hook too) |
| CV definitions (distance, Δ-distance, rational coordination) by atom-map name | **exists** | `reaction_coordinate.py`, `config_tian_tt_test.json` `cvs` |
| CV gradients | **exists, numerical** (central difference, tested) | `reaction_coordinate.py:144` |
| Langevin MD driver pattern | **exists for unbiased water** | `sweep/mace_md/compare_md.py` |
| Sn-site/water integrity monitor | **new** (this session) | `coordination_check.py` |
| Bias potential, hills, FES reconstruction | **missing** (no PLUMED / ASE bias in this path; `plumed` not installed) | — |
| Checkpoint / restart | **missing** | — |
| Confirmed CV parameters (r0, p, q), timestep, temperature, thermostat, well-tempering | **unknown** (thesis lists them as unknown) | README "Unknown" |
| Throughput | measured: 0.55 s per OMOL force call (float64, CPU, 42 atoms) | this session |

## Required decisions (owner: Tian / group)

1. **CV parameters.** `r0=1.5, n=6, m=12` are placeholders. Without the originals the
   thesis basins (0.9, 0.9) and (1.8, 0.1) cannot be interpreted. **Do not tune CVs to
   force agreement with the paper.**
2. Timestep, temperature, thermostat, friction.
3. Hill height/width/stride (thesis: 0.001 Ha, 0.1, every 100 steps — CPMD dataset only)
   and well-tempered or not.
4. Run length (20,000 steps stated for the CPMD set; "~44,000" unresolved).
5. Engine: conda-forge `plumed` + ASE (preferred; gives `sum_hills`/reweighting) or an in-repo bias wrapper.

## How the Figure-3-like surface would be reconstructed

F(CV1, CV2) = −V_bias(CV1, CV2) (+ constant) from the summed hills, **only after**
convergence checks (hill-height decay if well-tempered, repeated recrossings, block
averages of the barrier). Report: the two basin positions, the barrier between
them (`ΔF‡`) and ΔF. This is a **free energy on a biased-sampled CV space**; it is not
comparable to the NEB/Sella electronic-energy barriers of `energetics.py` and must
carry its own reference-state label. The only gas-phase reference number in the
repo (ΔF‡ ≈ 81.6 kJ/mol from thesis data tips) is **not verified** as a published
value and is not ground truth.

## Validation checklist before any production run (all must pass; none done)

- [ ] **Charge/spin**: assert `atoms.info` charge/spin equal the config on the first and
      every 1000th step; run one force call at a deliberately wrong charge to see it differ.
- [ ] **Basin stability, unbiased**: ≥ 2 ps NVT from the model-relaxed R and P; no
      `STRUCTURALLY_INVALID` from `coordination_check`; hydride stays in its state.
- [ ] **Energy conservation**: short NVE; drift per atom per ps below a threshold chosen
      before looking (sets the maximum timestep).
- [ ] **Bias force**: −∇V_bias from the code equals a finite-difference of V_bias to 1e-6;
      CV gradient test on the 42-atom system, not only the toy.
- [ ] **Hill bookkeeping**: deposit rate, positions, heights reproduce a second run with the same seed.
- [ ] **Restart**: kill at step k, restart, and compare to the uninterrupted run.
- [ ] **Water/Sn loss under bias**: run a short biased trial; if the Sn shell loses a heavy
      atom, the model is not usable for this run (record, do not hide).
- [ ] **Both models** run with identical settings and seeds.
- [ ] **Cost gate** (CLAUDE.md protocol): command, steps × 0.55 s, outputs, purpose, and
      Council review before any multi-hour run.

## Not claimed

No metadynamics has been run, no surface reconstructed, and no agreement or
disagreement with the paper is implied by anything here.
