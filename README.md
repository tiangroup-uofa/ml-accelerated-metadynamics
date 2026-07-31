# xTB Metadynamics — Water Clusters & a Reaction Example

Reproducible **xTB 6.7.1** metadynamics workflows for water systems, plus the
analysis, preparation and visualization tooling around them. No PLUMED — this
uses xTB's built-in **RMSD-based** metadynamics with ASE / NumPy / pandas /
Matplotlib for post-processing. The longer-term aim is to build validated
reference trajectories for **assessing GNN surrogate potentials for accelerated
enhanced sampling**.

> **What the bias is.** xTB biases along **Cartesian RMSD** to previously
> visited structures — not along any bond, angle or dihedral. Every distance /
> angle / CV computed in analysis is a **post-hoc descriptor**, not the biased
> coordinate. `Epot` plots are a **sampled potential-energy landscape from a
> biased run, not a free-energy surface** (that would need reweighting the bias).

## Setup

The xTB program and Python deps are **not** bundled — see **[SETUP.md](SETUP.md)**.
In short: install xtb 6.7.1 (`conda install -c conda-forge xtb` or `module load
xtb/6.7.1`), `export XTB_BIN=$(which xtb)`, and
`pip install ase numpy pandas matplotlib` (plus `scipy xtb-python sella` for the
Diels-Alder TS search). The newer drivers read `$XTB_BIN`; the git-ignored
`tools/` dir previously held a Windows-only build.

## What's here

| Area | Directory | Question | Write-up |
|---|---|---|---|
| **Single-molecule demo** | [`scripts/`](scripts/) + [`run/`](run/) | End-to-end metadynamics + CV/energy analysis on one system | this file, §"Single-molecule workflow" |
| **Bias × size sweep** | [`sweep/`](sweep/) | How do `kpush` and cluster size (n=1–4) affect stability? | [`sweep/FINDINGS.md`](sweep/FINDINGS.md) |
| **PBC, temperature, CVs** | [`sweep/`](sweep/) | Fixing evaporation with PBC; T dependence; dissociation vs drift | [`sweep/FINDINGS_phase2.md`](sweep/FINDINGS_phase2.md), [`sweep/JULY3_PROGRESS_RECORD.md`](sweep/JULY3_PROGRESS_RECORD.md) |
| **Large vacuum droplets** | [`sweep/large/`](sweep/large/) | Do bigger free clusters (n=20/30/50) form droplets or evaporate? Does raising `kpush` break them? | [`sweep/large/FINDINGS_large.md`](sweep/large/FINDINGS_large.md) |
| **Diels-Alder reaction** | [`sweep/diels_alder/`](sweep/diels_alder/) | Can we drive a real reaction, and can MACE-OFF23 reproduce its energetics (NEB)? | [`FINDINGS_diels_alder.md`](sweep/diels_alder/FINDINGS_diels_alder.md), [`FINDINGS_neb.md`](sweep/diels_alder/FINDINGS_neb.md) |
| **MACE-OFF23 surrogate** | [`sweep/mace_validation/`](sweep/mace_validation/) + [`sweep/mace_md/`](sweep/mace_md/) | How does the MACE-OFF23 MLIP compare to GFN2-xTB (forces, energies, MD structure, speed)? | [`FINDINGS_mace.md`](sweep/mace_validation/FINDINGS_mace.md), [`FINDINGS_mace_md.md`](sweep/mace_md/FINDINGS_mace_md.md) |
| **kpush size-scaling** | [`sweep/kpush_scaling/`](sweep/kpush_scaling/) | Why is the same bias gentler on a bigger cluster, and how to normalize it? | [`sweep/kpush_scaling/FINDINGS_kpush_scaling.md`](sweep/kpush_scaling/FINDINGS_kpush_scaling.md) |
| **CV design** | [`sweep/cv_design/`](sweep/cv_design/) | Which collective variables to bias in MACE metadynamics? | [`sweep/cv_design/FINDINGS_cv_design.md`](sweep/cv_design/FINDINGS_cv_design.md) |
| **Post-training corrections** | [`sweep/corrections/`](sweep/corrections/) | Can a lightweight correction remove MACE↔xTB error without retraining? | [`sweep/corrections/FINDINGS_corrections.md`](sweep/corrections/FINDINGS_corrections.md) |

## Key results at a glance

- **The bias breaks the weakest bond.** A lone water only dissociates at high
  `kpush` (must break a covalent O–H); hydrogen-bonded clusters come apart even
  at the lowest bias — so small clusters **evaporate** rather than rearrange.
- **A large O–O distance ≠ dissociation.** Check the intramolecular O–H:
  clusters **drift apart intact**; only the monomer truly **dissociates**.
- **PBC removes gas-phase evaporation** by confinement (but forces GFN1 — GFN2
  is unsupported under PBC in this build).
- **Size suppresses evaporation.** Free vacuum droplets n=20/30/50 stay intact
  at low bias, and raising `kpush` to 0.05 at n=30 doesn't break them — because
  global-RMSD bias is **diluted over system size** (→ a `kpush` normalized per
  degree of freedom is the natural next step).
- **Diels-Alder works and is verified.** Butadiene + ethylene → cyclohexene via
  `xtb --path`; concerted TS confirmed (forming bonds 2.32 Å, single imaginary
  mode −394 cm⁻¹), barrier **6.7 kcal/mol** from a relaxed scan and Sella+Hessian.
  The `--path` "barrier" itself is a biased-path artifact — enhanced sampling
  *finds* reactivity, but a real barrier needs a proper TS treatment.
- **MACE-OFF23 tracks xTB and scales better.** On the droplets, MACE-OFF23 vs
  GFN2-xTB forces correlate at r≈0.99 and relative energies agree to ~2–3
  meV/atom; MACE's cost relative to xTB falls 2.5×→1.6×→1.0× over n=20→30→50, so
  the ~linear MLIP overtakes O(N³) xTB by ~150 atoms — the surrogate's payoff is
  the large-droplet regime.
- **On the Diels-Alder, MACE gets the TS geometry but different (more physical)
  energetics.** Both put the TS at ~2.0–2.3 Å, but MACE's barrier (36 kcal/mol)
  and reaction energy (−36) are far from xTB's (6.7, −57.6) and closer to
  DFT/experiment — off the shelf. The systematic offset motivates a lightweight
  force correction (modular `CorrectedCalculator` in [`sweep/diels_alder/neb/`](sweep/diels_alder/neb/)).

---

## Single-molecule workflow ([`scripts/`](scripts/) + [`run/`](run/))

The original end-to-end demo: run metadynamics on one system, then turn its
outputs into CVs, an energy trace, and plots.

```bash
export XTB_BIN=$(which xtb)

# 1. optimize, then run metadynamics (from run/)
xtb molecule.xyz --opt tight --gfn 2 --chrg 0 --uhf 0 > optimization.log
python ../scripts/prepare_input.py xtbopt.xyz coord      # -> Turbomole coord (Bohr)
xtb coord --md --input metadyn.inp --gfn 2 --chrg 0 --uhf 0 > output.log

# 2. analysis (from scripts/)
./run_analysis.sh          # convert_scoord -> calculate_cvs -> extract_epot
                           # -> merge_structure_energy -> plot_analysis
```

**Two output cadences.** `run/xtb.trj` is the dense trajectory (dumped every
`dump`=100 fs); `run/scoord.*` are sparse ~1 ps snapshots for quick inspection
(**not** the RMSD-bias reference set — `save` controls that, not the snapshot
rate). Use `xtb.trj` for fast events; a bond can break *between* snapshots.

**Choosing CVs (edit the CONFIG block in [`scripts/calculate_cvs.py`](scripts/calculate_cvs.py)).**
ASE indices are **zero-based**. Defaults are placeholders — pick indices that
track the process you expect:

| Process | Suggested CV |
|---|---|
| Bond breaking / formation | the breaking / forming bond distance |
| Proton transfer | `d(H, donor) − d(H, acceptor)` |
| Concerted break + form | `d(breaking) − d(forming)` |
| Conformational change | a dihedral angle |

Outputs land in [`analysis/`](analysis/) (CVs, `epot.csv`, merged CSV) and
[`figures/`](figures/) (`rmsd_vs_time`, `distances_vs_time`, `epot_vs_*`,
`structural_map_2d`, `radius_of_gyration_vs_time`, …).

---

## Caveats that apply throughout

- **Post-hoc descriptors, biased energies.** CVs are computed after the run;
  `Epot` is from a biased run — not thermodynamic free energy.
- **Connectivity/coordination metrics are heuristic** (covalent-radii cutoffs,
  O–O switching functions — *not* angular H-bond counts). Inspect flagged frames
  visually; corroborate any "event" with the intramolecular O–H distance.
- **Parameters need tuning.** `kpush`, `alp`, `save`, `temp`, `time` are
  conservative testing values, not universal production settings. Larger `kpush`
  increases the push away from visited structures and the risk of unphysical
  dissociation — and its effective strength depends on system size.
- **Small, short, often single trajectories** — trends, not statistics.
- **GFN1 ≠ GFN2.** Periodic runs use GFN1 (GFN2 unsupported under PBC here), so
  gas-phase and periodic results are not at an identical level of theory.
