# Gas-phase xTB Metadynamics — Workflow and Trajectory Analysis

A reproducible workflow for running a **gas-phase metadynamics**
simulation with **xTB 6.7.1** and analysing the ~1 ps coordinate
snapshots it writes to `scoord.*`. No PLUMED is used — this relies only
on xTB's built-in metadynamics plus ASE / NumPy / pandas / Matplotlib
for post-processing.

> **Trajectory vs snapshots.** Two different outputs are produced at two
> different cadences:
> * `run/xtb.trj` — the **dense full trajectory**, dumped every `dump` fs
>   (set to 100 fs in `metadyn.inp`, i.e. ~0.1 ps per frame). Use this for
>   detailed transition analysis.
> * `run/scoord.*` — **periodic coordinate snapshots** written roughly
>   every 1 ps. These are sparse and are meant for quick inspection; they
>   are *not* the set of structures currently retained for the RMSD bias,
>   and the `save` keyword does **not** control how often they are written.

> ⚠️ **Important conceptual distinction.** The plots of `Epot` versus a
> collective variable produced here are a **sampled potential-energy
> landscape from a biased trajectory**, *not* an unbiased free-energy
> surface. xTB metadynamics biases along **Cartesian RMSD** relative to
> previously visited structures; the bond distances, angles and
> dihedrals computed in analysis are **post-hoc structural descriptors**,
> not the quantity that was biased. Recovering a true free-energy
> surface would require reweighting the metadynamics bias, which is out
> of scope here.

---

## 1. Objective

1. Run a gas-phase xTB metadynamics simulation.
2. Collect and visualise the `scoord.*` structures (≈1 ps apart).
3. Compute structural collective variables (CVs) from those structures.
4. Extract the potential energy `Epot` from the xTB log.
5. Match each structure to the nearest-in-time `Epot` value.
6. Plot the structural changes and the sampled potential-energy landscape.

## 2. Software requirements

| Tool        | Purpose                                  |
|-------------|------------------------------------------|
| xTB 6.7.1   | optimization + metadynamics MD           |
| Python 3    | analysis scripts                         |
| ASE         | reading Turbomole/XYZ, RMSD geometry I/O |
| NumPy       | numerics, Kabsch alignment               |
| pandas      | tabular merge / CSV                      |
| Matplotlib  | plotting (no seaborn)                    |
| VMD / Avogadro (optional) | manual visualization       |

```bash
pip install ase numpy pandas matplotlib
module load xtb/6.7.1   # on the HPC cluster
```

## 3. Directory structure

```
xtb_metadynamics/
├── input/        # starting geometry (molecule.xyz / coord)
├── run/          # metadyn.inp, Slurm script, xTB outputs live here
├── analysis/     # CSVs and the assembled trajectory
├── figures/      # PNG plots
├── scripts/      # all Python + the master run_analysis.sh
└── README.md
```

## 4. Running xTB

All commands below are run from `run/`. Set `--chrg` and `--uhf` for
your system (`--uhf` is the number of **unpaired electrons**).

```bash
# 2a. Optimize the starting geometry (GFN2-xTB, tight)
xtb molecule.xyz --opt tight --gfn 2 --chrg 0 --uhf 0 > optimization.log
#    -> the optimized geometry is written to 'xtbopt.xyz'

# 2b. Convert the optimized geometry to Turbomole 'coord' (ASE; Bohr)
python ../scripts/prepare_input.py xtbopt.xyz coord
#    Use this 'coord' (or xtbopt.xyz directly) as the metadynamics input.

# 4. Run metadynamics
xtb coord --md --input metadyn.inp --gfn 2 --chrg 0 --uhf 0 > output.log
#    or submit the Slurm job:
sbatch submit_xtb_mtd.slurm

# 5. Validate
bash validate_run.sh output.log
```

## 5. Analysis

```bash
cd scripts
./run_analysis.sh          # runs all five stages in order
```

Individual stages:

```bash
python convert_scoord.py        --run-dir ../run --out-dir ../analysis
python calculate_cvs.py         --xyz ../analysis/scoord_1ps.xyz --dt-ps 1.0
python extract_epot.py          --log ../run/output.log
python merge_structure_energy.py --tolerance-ps 0.5
python plot_analysis.py
```

Visualise the structures:

```bash
ase gui ../analysis/scoord_1ps.xyz      # step through frames
vmd     ../analysis/scoord_1ps.xyz      # load as a multi-frame XYZ
#   Avogadro: File -> Open -> scoord_1ps.xyz, then use the animation slider
```

## 6. Atom indexing (read this before editing CVs)

**ASE atom indices are ZERO-BASED.** The atom your visualizer or the
Turbomole file labels "atom 1" is **index 0** in `calculate_cvs.py`.

Edit the CONFIG block at the top of `scripts/calculate_cvs.py`:

```python
DIST1_ATOMS    = (0, 1)     # bond that may break/form
DIST2_ATOMS    = (1, 2)     # second bond (partner in a concerted step)
ANGLE_ATOMS    = (0, 1, 2)  # i-j-k, vertex = middle index
DIHEDRAL_ATOMS = (0, 1, 2, 3)
```

The defaults are **placeholders** — they are not chemically meaningful
for an arbitrary molecule. Choose indices that track the process you
expect (see "Choosing CVs" below).

### Choosing chemically meaningful CVs

| Process                         | Suggested CV                               |
|---------------------------------|--------------------------------------------|
| Bond breaking                   | the breaking-bond distance                 |
| Bond formation                  | the forming-bond distance                  |
| Proton transfer                 | `d(H, donor) − d(H, acceptor)`             |
| Concerted break + form          | `d(breaking bond) − d(forming bond)`       |
| Conformational transition       | a dihedral angle                           |

Plotting an arbitrary pair of atom distances is **not** automatically a
meaningful CV analysis — the CV must reflect the chemistry you expect.

## 7. Expected outputs

```
run/metadyn.inp, run/optimization.log, run/output.log
run/xtb.trj, run/scoord.*, run/mdrestart, run/xtbmdok
analysis/scoord_1ps.xyz, analysis/scoord_1ps.traj
analysis/collective_variables.csv
analysis/epot.csv
analysis/trajectory_analysis.csv
figures/rmsd_vs_time.png
figures/distances_vs_time.png
figures/distance_difference_vs_time.png
figures/dihedral_vs_time.png
figures/epot_vs_time.png
figures/epot_vs_rmsd.png
figures/epot_vs_cv.png
figures/structural_map_2d.png        (primary 2D map; scatter)
figures/radius_of_gyration_vs_time.png
figures/max_pair_distance_vs_time.png
figures/estimated_components_vs_time.png
figures/energy_landscape_2d.png      (only if sampling is dense enough)
figures/energy_landscape_2d.png
```

## 8. Known limitations

* **Two cadences.** `xtb.trj` is now dumped every 100 fs (`dump=100.0`),
  far finer than the ~1 ps `scoord.*` snapshots. The sparse `scoord.*`
  set is fine for inspection, but fast events (a bond breaking in tens of
  fs) can fall *between* snapshots; use the full `run/xtb.trj` for
  detailed transition analysis.
* **`save` ≠ snapshot frequency.** `save` sets the maximum number of
  structures retained for the RMSD bias criterion; it does not control
  the `scoord.*` write frequency, and a larger `save` is not by itself
  "more aggressive flooding" (per-structure push strength is `kpush`).
* **Snapshot timing is approximate.** `--dt-ps` / `--t0-ps` in
  `calculate_cvs.py` are assumptions; the time axis is derived from each
  file's numeric suffix (so a missing `scoord.N` does not shift later
  times), but it remains approximate until verified against the actual
  xTB output cadence.
* **Atom count does not prove integrity.** Consistent atom counts and
  element ordering are *required* for trajectory analysis but do **not**
  prove the molecule stayed connected — dissociation usually preserves
  the atom count while fragments separate.
* **Connectivity checks are heuristic.** `max_pair_distance_A` and
  `estimated_components` (covalent-radii cutoff, `connectivity_scale`
  default 1.3) are rough fragmentation indicators only. Real reactions
  change connectivity, weakly bound complexes are naturally
  multi-component, and the cutoff is approximate — inspect flagged frames
  visually rather than treating every multi-component frame as an error.
* **Post-hoc descriptors.** The distances/angles/dihedrals are computed
  *after* the run and are **not** the coordinate xTB biased (which is
  Cartesian RMSD against retained references).
* **Biased energies.** `Epot` comes from a biased run; do not read the
  landscape as thermodynamic free energy.
* **Parameter sensitivity.** `kpush`, `alp`, `save`, `temp` and `time`
  in `metadyn.inp` are conservative initial testing values, not
  universally correct production settings, and generally need tuning per
  system. Larger `kpush` increases the push away from visited structures
  and the risk of unphysical dissociation.

## 9. Potential energy vs free energy

* **Sampled potential-energy landscape (what we plot):** instantaneous
  `Epot` from a biased MD run, projected onto a structural coordinate.
  Affected by the bias and by kinetic sampling.
* **Free-energy surface (what we do NOT produce):** the unbiased
  potential of mean force along a CV, obtained only after reweighting
  out the metadynamics bias and integrating over all other degrees of
  freedom. Not computed in this workflow.
