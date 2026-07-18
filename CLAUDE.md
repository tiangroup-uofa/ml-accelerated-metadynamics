# CLAUDE.md

Shared project context for Claude Code sessions working in this repo (used by
multiple collaborators — keep it accurate and concise; it loads every session).

## What this project is

Reproducible **xTB 6.7.1** RMSD-metadynamics workflows for water systems, plus
analysis/prep/visualization tooling. Long-term goal: build validated reference
trajectories to **assess GNN surrogate potentials (MACE-OFF) for accelerated
enhanced sampling**. Read `README.md` first, then the `FINDINGS_*.md` files.

## Environment / how to run xtb

- The program is **not** in the repo. `tools/` (a Windows-only build) is
  git-ignored. Install xtb 6.7.1 yourself and expose it:
  - conda/mamba (macOS arm64, Linux): `conda install -c conda-forge xtb`
  - HPC: `module load xtb/6.7.1`
  - then `export XTB_BIN=$(which xtb)`
- Newer drivers (`sweep/large/`, `sweep/diels_alder/`) read `$XTB_BIN` (falling
  back to `xtb` on PATH). Some original `sweep/` scripts still hardcode the old
  Windows path — fix to `$XTB_BIN` if you rerun them.
- Python: `pip install ase numpy pandas matplotlib`; for the Diels-Alder TS
  search also `scipy`, `xtb-python` (conda-forge), `sella`. Details in `SETUP.md`.

## Repo map

- `scripts/` + `run/` — single-molecule end-to-end demo (run → CVs → energy → plots).
- `sweep/` — bias×size sweep n=1–4 (`FINDINGS.md`); PBC/temperature/CV design
  (`FINDINGS_phase2.md`, `JULY3_PROGRESS_RECORD.md`).
- `sweep/large/` — vacuum GFN2 droplets n=20/30/50 (`FINDINGS_large.md`).
- `sweep/diels_alder/` — butadiene+ethylene→cyclohexene reaction path + TS
  (`FINDINGS_diels_alder.md`).

Each study directory has build/run/analyze/plot scripts and its own write-up.

## Key facts & gotchas (learned the hard way)

- **The bias is Cartesian RMSD to visited structures — chemically blind.** It
  breaks the weakest available bond and maximizes structural novelty, not
  meaningful chemistry. `Epot` vs a CV is a *biased sampled* landscape, **not**
  a free-energy surface. All CVs are post-hoc descriptors.
- **kpush is effectively size-dependent.** Global RMSD is spread over all DOF, so
  the same nominal kpush is much gentler for a big droplet than a small cluster.
  A size-normalized (per-molecule/atom) kpush is the open modelling question.
- **GFN2 cannot run under PBC** in this build (`Multipoles not available with
  PBC`) — periodic runs use GFN1, so gas (GFN2) vs periodic (GFN1) are not the
  same level of theory. Free vacuum droplets keep GFN2.
- **xtb has NO TS optimizer** (`--optts` is not a flag). Use an external saddle
  optimizer (Sella) with xtb-python as the ASE calculator, seeded from a good
  guess (a relaxed scan), not from the raw `--path` guess.
- **Don't regex frequencies from `xtb --hess` stdout** — it matches the
  `imag. cutoff -20.0 cm` thermostat *parameter*. Read the `vibspectrum` file.
- Prefer `xtb.trj` (dense) over `scoord.*` (sparse ~1 ps) for fast events.

## Status & next directions (agreed with the group)

Done: n=1–8 sweeps + PBC/T/CV; larger vacuum droplets (evaporation suppressed by
size; kpush=0.05 doesn't break n=30 — a size-dilution artifact, not robustness);
Diels-Alder verified TS (2.32 Å, −394 cm⁻¹, barrier 6.7 kcal/mol).

Priorities:
1. **MACE-OFF23 surrogate (main direction).** Validate the pretrained MLIP
   against these xTB trajectories (forces/energies), then use it where xTB's
   O(N³) scaling hurts — larger droplets. The **hard open question is which
   collective variables to use** for the enhanced sampling.
2. **kpush scaling rule** — test a per-molecule/atom-normalized kpush; the
   scaling behaviour across system size is the interesting part.

## Working conventions

- Don't commit `tools/`, verbose `output.log`, xtb restart artifacts, or caches
  (see `.gitignore`). Trajectories (`xtb.trj`) are small enough to keep.
- Commit messages: **no Co-Authored-By trailer** in this repo.
