# July 3 Assignment — Complete Progress Record

*Verified against files on disk under `xtb_metadynamics/sweep/` on 2026-07-10.*

---

## 1. Project objective

The broader project is to establish and validate **xTB metadynamics workflows**
for small water systems — enhanced-sampling MD driven by a root-mean-square-
deviation bias — and to build the analysis, preparation and visualization
tooling around them. The longer-term aim is to use these validated reference
trajectories to **assess graph-neural-network (GNN) surrogate potentials for
accelerated enhanced sampling**, so the emphasis in this phase is on correct,
reproducible workflows and a defensible interpretation of what the simulations
actually show.

---

## 2. Work completed since July 3

### Bias-strength testing
**What / why:** A three-level bias sweep was run on gas-phase clusters
(H2O)1–4 at `kpush` = 0.008 (low), 0.02 (med), 0.05 (high), 20 ps, NVT 300 K,
GFN2-xTB, to test how the metadynamics bias strength affects cluster stability.
**Observed** (`analysis/per_run_summary.csv`, `figures/intact_heatmap.png`):
higher `kpush` breaks clusters sooner (e.g. the trimer first fragments at
1.7 → 0.2 → 0.1 ps across low → med → high). The single water is the most
resistant (only fragments at high bias); all multi-molecule clusters separate
even at low bias.
**Conclusion:** The bias preferentially breaks the *weakest available bond* —
the covalent O–H (~5 eV) for the lone molecule, but the weak intermolecular
hydrogen bond (~0.2 eV) for clusters, which is why clusters come apart at even
the gentlest bias.

### Timestep stability
**What / why:** Investigated whether a 1 fs timestep is safe under stronger
bias. **Observed:** stronger bias with a 1 fs step drove instability; a 0.5 fs
step was stable. **Conclusion:** all production runs use **0.5 fs**, appropriate
for unconstrained O–H bonds under an aggressive bias.

### Temperature dependence
**What / why:** At fixed bias (`kpush` = 0.02) the water trimer was run at
**200, 250, 298, 350 K** to isolate the thermostat-temperature effect from the
bias effect (`run_temp_sweep.py`, `figures/temperature_vs_time.png`,
`analysis/cv_T{200,250,298,350}.csv`).
**Observed:** maximum O–O separation reached **80.6 / 133.5 / 138.3 / 143.3 Å**
at 200 / 250 / 298 / 350 K — higher temperature drives **faster and farther**
separation. At every temperature the maximum intramolecular O–H stayed
**≈1.03–1.04 Å**.
**Conclusion:** The process is **intact molecules drifting apart, not O–H
dissociation** — the extra thermal energy simply helps whole molecules escape
the hydrogen-bond network sooner. (Single 3-molecule trajectories carry some
statistical scatter, so the ordering is a trend, not a strict monotonic law.)

### Water-cluster preparation
**What / why:** Implemented a **repeat-and-equilibrate** workflow
(`prepare_box.py`): place N waters on a loose 3-D grid wider than the van der
Waals contact distance, write a periodic cell, run **unbiased NVT MD** to let
the thermostat condense them, and use the equilibrated structure as the
metadynamics start.
**Observed:** for the 8-water system the minimum O–O distance contracted from
**≈3.88 Å to ≈2.68 Å**, i.e. from a loose grid to genuine hydrogen-bond
contacts (recorded in `FINDINGS_phase2.md`).
**Conclusion:** the workflow produces a **more condensed, hydrogen-bonded
starting configuration** without external packing tools.
A **Packmol input generator** was also written (`pack_water.py`, with
`packmol/water_8.inp`), but Packmol execution was deferred — see §4. Packmol is
**not** a blocking task, because repeat-plus-NVT-equilibration was an accepted
preparation alternative in the assignment.

### Periodic-boundary implementation
**What / why:** Added periodic xTB coordinate generation using `$periodic 3`
and `$cell` (in `prepare_box.py` / `run_pbc.py`) so molecules cannot evaporate
to infinity. **Two implementation facts were discovered and handled:**
- **GFN2-xTB cannot run under PBC** in the installed xTB 6.7.1
  (`scf: Multipoles not available with PBC`). **GFN1-xTB was used** for all
  periodic calculations.
- **`xtb.trj` does not store the periodic cell.** The cell therefore had to be
  **manually restored** during analysis, and all periodic distances evaluated
  with the **minimum-image convention** (implemented as the `--cell` option in
  `cv_analysis.py`).

### Corrected periodic metadynamics
**Preliminary diagnostic (superseded):** the first 8-water PBC test used a
**7.76 Å box**; under bias it produced physically unrealistic O–H distances,
indicating the box was too small and/or the bias too aggressive. **This run
was superseded** and is not a final result.
**Corrected run:** repeated with a **larger 9.20 Å box** and a **reduced bias
`kpush` = 0.008**, 20 ps, 0.5 fs, GFN1
(`pbc_runs/n8_pbc/`, analyzed in `analysis/cv_n8_PBC_v2.csv`,
`figures/pbc_vs_gas.png`).
**Observed:** the maximum O–O distance stayed **≈6–7 Å, capped at 7.69 Å**,
always inside the 9.20 Å box, whereas the gas-phase reference cluster separates
to **>100 Å**. The oxygen-neighbour coordination remained substantial
(time-averaged ≈1.9) rather than collapsing to zero.
**Conclusion (stated carefully):** PBC **removes the gas-phase evaporation
artifact by spatial confinement** — the system stays bounded and the
oxygen-neighbour network is largely maintained. This is **not** yet a claim of
clean molecular integrity or true liquid-like behaviour: the same corrected
trajectory shows large assigned-O–H excursions under GFN1 in a dense box, which
the current nearest-O analysis cannot cleanly separate from proton reassignment.
Distinguish four separate claims: (i) spatial confinement — demonstrated;
(ii) molecular integrity under PBC — not cleanly demonstrated; (iii) maintenance
of the oxygen-neighbour network — largely supported; (iv) true liquid-like
behaviour — not claimed.

### CV design and interpretation
**What / why:** Built a trajectory-analysis workflow (`cv_analysis.py`) to
separate **chemical dissociation** from **physical separation**:
- **assigned/nearest O–H distance** — molecular-integrity diagnostic
  (grows only if a covalent bond breaks);
- **intermolecular O–O distance** — spatial-separation diagnostic;
- **smooth oxygen-neighbour coordination number** (O–O switching function);
- **minimum-image distances** for PBC trajectories.
**Validation** (`analysis/cv_n1_high.csv`, `analysis/cv_n3_low.csv`):
- single water, high bias → assigned O–H grew to ~1315 Å → **genuine O–H
  dissociation**;
- three waters, low bias → O–H stayed ≈**1.06 Å** while O–O exceeded **100 Å**
  → **intact molecules drifting apart**, not bond breaking.
**Terminology (careful):** the smooth O–O metric is an **oxygen-neighbour
coordination number**, *not* an exact hydrogen-bond count, because **no angular
O–H···O criterion** is applied. Also, nearest-O–H analysis can keep a short
distance even when a proton is reassigned, so **per-oxygen O–H coordination** is
identified as a stronger future reactive metric (prototyped in
`analyze_controls.py`).

### Visualization and analysis
**What / why:** Built an ASE-based visualization workflow (`make_movie.py`):
render each frame with covalent-radius **bonds** into a PNG stack and combine
into an animated GIF.
**Produced:** `figures/movie_n3_low.gif` (gas trimer — starts compact, then
three intact waters drift to the corners, each keeping its O–H bonds) and
`figures/movie_n8_PBC.gif` (periodic box — stays spatially confined).
**Conclusion:** the animations make the **dissociation-vs-drift distinction
visible** — in the gas trimer the molecules separate while retaining their two
O–H bonds each (drift), which is exactly what the CV analysis reports.

---

## 3. Key technical findings

1. **The metadynamics bias breaks the weakest available bond** — covalent O–H
   for a lone molecule, weak hydrogen bonds for clusters; clusters therefore
   evaporate even at the lowest bias.
2. **A large O–O separation is not, by itself, dissociation.** Checking the
   intramolecular O–H distance disambiguates: lone water **dissociates**;
   clusters **drift apart** intact.
3. **Higher temperature accelerates and extends separation** (max O–O 80.6 →
   143.3 Å over 200 → 350 K) while molecules stay intact (O–H ≈1.03 Å).
4. **GFN2-xTB is incompatible with PBC** in this xTB build; **GFN1** is required
   for periodic runs, so gas (GFN2) and periodic (GFN1) results are not at an
   identical level of theory.
5. **Periodic boundaries remove the evaporation artifact by confinement** —
   corrected 9.20 Å box keeps max O–O ≤ 7.69 Å versus >100 Å in the gas phase.
6. **`xtb.trj` lacks cell metadata**, so periodic analysis must restore the cell
   and use minimum-image distances — without this, wrapped atoms give spurious
   huge O–H/O–O distances.

---

## 4. Problems encountered and how they were resolved

- **Packmol runtime issue:** the PyPI `packmol` wheel ships a `packmol.exe`
  dynamically linked against `libgfortran-5.dll` / `libquadmath-0.dll`, which
  are absent on this machine, so it crashes (confirmed error:
  `error while loading shared libraries: libgfortran-5.dll`). `uvx packmol`
  ships the same broken binary. **Resolution:** deferred Packmol
  (`conda install -c conda-forge packmol` bundles the DLLs) and used
  repeat-plus-NVT-equilibration, an accepted alternative.
- **GFN2 incompatible with PBC:** `scf: Multipoles not available with PBC`.
  **Resolution:** used GFN1-xTB for all periodic runs.
- **Missing cell metadata in `xtb.trj`:** frames read back with no cell.
  **Resolution:** restore the known cell and compute minimum-image distances
  (`cv_analysis.py --cell`).
- **Small-box instability:** the first 7.76 Å PBC box produced unphysical O–H
  distances under bias. **Resolution:** larger 9.20 Å box + gentler
  `kpush` = 0.008.
- **SCF / dynamics instability of a dense box run gas-phase:** a PBC-condensed
  box is unphysical without periodicity — run gas-phase it either fails SCF
  (near-zero HOMO–LUMO gap) or the atoms rocket apart at the first MD step.
  **Resolution (for the optional controls extension):** start both arms from a
  GFN1 geometry-optimized 8-water cluster instead of an MD-condensed box.
- **Unit trap:** the `$cell` line in a Turbomole `coord` is in **Bohr**; it must
  be divided by 1.8897 for the Ångström value used in minimum-image analysis
  (a 14.67 vs 7.76 Å error was caught and corrected this way).

---

## 5. July 3 completion table

| July 3 task | Final status |
|---|---|
| Test different `kpush` bias strengths | **Complete** |
| Investigate timestep stability and use 0.5 fs | **Complete** |
| Test different NVT temperatures | **Complete** |
| Create and test a periodic structure | **Complete** |
| Prepare multiple water molecules | **Complete** (repeat + NVT equilibration; Packmol deferred, not blocking) |
| Plot key O–H distances | **Complete** |
| Produce CV / energy-surface analysis | **Complete** |
| Generate ASE/OVITO movie or PNG stack | **Complete** |
| Determine CVs for multiple waters | **Complete** |
| Distinguish dissociation from molecular drift | **Complete** |

**All required July 3 items are complete.** Packmol is **not** a blocking item.

---

## 6. Additional validation work (optional, not part of the completed assignment)

A **controls-and-replicates** extension was designed to strengthen the case by
separating thermal/initialization instability from bias-induced behaviour and
by testing reproducibility. Intended 2×2 design, each cell × 3 **independent
initial-configuration replicas**:

| Boundary | Unbiased (kpush = 0) | Biased (kpush = 0.008) |
|---|---|---|
| Gas phase | baseline stability | bias-induced separation |
| PBC | baseline condensed | biased but confined |

Corrections incorporated into the analysis (`analyze_controls.py`) **before**
interpreting any results:
- **same GFN1-optimized 8-water starting geometry** for both gas and PBC arms,
  placed in a larger shared cell (gas arm removes PBC but keeps coordinates);
- a short unbiased **stability smoke test** first;
- **connectivity-based fragmentation** (largest connected-component size /
  fragment count), with **max O–O kept only as a descriptive observable** — for
  an 8-water cluster `max O–O > 3.5 Å` is *not* a fragmentation criterion;
- the O–O switching-function metric renamed **oxygen-neighbour coordination**
  (no angular H-bond criterion applied);
- **per-oxygen O–H coordination** added so proton transfer / hydronium–hydroxide
  formation is not hidden by nearest-O reassignment;
- the three runs described as **independent initial-configuration replicas**
  (xTB does not expose an MD velocity seed).

**Current on-disk state:** the driver (`run_controls.py`) and corrected analysis
(`analyze_controls.py`) exist, and a **stability smoke test of all four cells
(seed 0) has passed** using the corrected GFN1-optimized shared starting
geometry — all four ran cleanly with no SCF failure or blow-up, and the gas
unbiased trajectory kept every oxygen at exactly 2 hydrogens through the early
frames while expanding gradually (span 4.0 → 6.5 Å), confirming the earlier
step-1 instability is resolved. The **full 12-run grid, the mean±std summary,
and the consolidated 3-panel figure are not yet produced**. This section is
**planned/ready optional validation**, explicitly **not** part of the completed
July 3 assignment; the setup is validated and ready to run when desired.

---

## 7. Methodological limitations

- **Finite, small systems** (1–8 water molecules); an 8-water cluster is a small
  model of condensed water.
- **Short trajectories** (20 ps) and, for the temperature and single-condition
  runs, **single trajectories** — statistical scatter is present.
- **Sensitivity to box size and bias** — the periodic result depends on choosing
  a sensible cell and a gentle bias; a too-small box gives unphysical behaviour.
- **Limited replicas** — the reproducibility (controls + replicas) extension is
  not yet complete.
- **PBC confinement is not, by itself, proof of liquid-like behaviour** —
  it demonstrates spatial bounding, not thermodynamic condensed-phase structure.
- **GFN1 was required for PBC** (GFN2 unsupported), so gas (GFN2) and periodic
  (GFN1) comparisons are not strictly at one level of theory; GFN1 also
  describes hydrogen bonding less accurately than GFN2.
- **Packmol deferred** (missing Windows Fortran runtime); repeat-plus-NVT was
  used instead.

---

## 8. To Show the Professor

*All paths verified present on disk. Ordered by importance.*

1. **`figures/intact_heatmap.png`** (+ `analysis/per_run_summary.csv`)
   - Demonstrates: bias-strength effect across cluster sizes (fraction of the
     run each cluster stayed intact).
   - Say aloud: *"Higher kpush breaks clusters sooner, and the lone water is the
     only species that resists — because it can only come apart by breaking a
     covalent O–H bond."*

2. **`figures/temperature_vs_time.png`** (+ `analysis/cv_T{200,250,298,350}.csv`)
   - Demonstrates: temperature dependence of separation at fixed bias.
   - Say aloud: *"Higher temperature drives faster and farther separation, from
     81 Å at 200 K to 143 Å at 350 K, while O–H stays near 1.03 Å — so it is
     drifting apart, not dissociation."*

3. **`figures/pbc_vs_gas.png`** (+ `analysis/cv_n8_PBC_v2.csv`)
   - Demonstrates: gas-phase evaporation versus bounded periodic behaviour.
   - Say aloud: *"With the corrected 9.20 Å box and weaker bias, the periodic
     system stays spatially bounded below 8 Å while the gas-phase cluster
     separates past 100 Å."*

4. **O–H versus O–O dissociation/drift comparison**
   (`analysis/cv_n1_high.csv` vs `analysis/cv_n3_low.csv`)
   - Demonstrates: the CV that separates dissociation from drift.
   - Say aloud: *"A big O–O distance alone is ambiguous — I check the
     intramolecular O–H: the monomer's O–H blew up to over 1000 Å, so it truly
     dissociated, whereas the trimer's O–H stayed at 1.06 Å while O–O exceeded
     100 Å, so those molecules just drifted apart intact."*

5. **`figures/movie_n3_low.gif`** (fallback frames:
   `figures/movie_n3_low_frames/frame_0000.png` and `frame_0030.png`)
   - Demonstrates: intact molecules drifting apart, bonds retained.
   - Say aloud: *"In the animation each water keeps its two O–H bonds as the
     three molecules drift to the corners — visual confirmation of drift, not
     dissociation."*

6. **Repeat-and-equilibrate preparation** (`prepare_box.py`; condensation
   result ~3.9 → 2.7 Å recorded in `FINDINGS_phase2.md`)
   - Demonstrates: how multiple waters were prepared without Packmol.
   - Say aloud: *"I place waters on a loose grid and let unbiased NVT condense
     them — the minimum O–O contracts from about 3.9 to 2.7 Å, a real
     hydrogen-bond distance."*

7. **Corrected PBC input** (`pbc_runs/n8_pbc/input.coord` and
   `pbc_runs/n8_pbc/md.inp`)
   - Demonstrates: the exact corrected settings — **9.20 Å box, kpush = 0.008,
     0.5 fs, GFN1**.
   - Say aloud: *"This is the corrected periodic setup: a 9.2 Å cell, gentle
     bias, half-femtosecond step, and GFN1 because GFN2 cannot run under PBC."*

8. **`figures/movie_n8_PBC.gif`** (optional, fallback frames in
   `figures/movie_n8_PBC_frames/`)
   - Demonstrates: the periodic system staying spatially confined.
   - Say aloud: *"Under PBC nothing leaves the box — the system stays condensed
     instead of evaporating."*

9. **Concise summary table** (below, §"Summary comparison") — gas vs PBC,
   temperature, O–H integrity, O–O separation, oxygen-neighbour coordination.

*(The unbiased-controls / replicas 3-panel figure does **not** yet exist on
disk — present it only as planned validation work.)*

### Summary comparison

| Condition | O–H integrity | max O–O | O-neighbour coord. | Interpretation |
|---|---|---|---|---|
| Gas monomer, high bias | **broken** (~1315 Å) | n/a | n/a | dissociation |
| Gas trimer, low bias | intact (~1.06 Å) | >100 Å | → 0 | drift apart |
| Gas trimer, 200→350 K | intact (~1.03 Å) | 81→143 Å | → 0 | drift, faster hotter |
| PBC 8-water, kpush 0.008 | not cleanly resolved (GFN1) | ≤7.69 Å (bounded) | ≈1.9 retained | spatially confined |

### Suggested meeting order (3–5 minutes)

1. **Assignment and question** — bias, temperature, periodic structure, water
   preparation, CVs; and the core question: is a large O–O separation
   dissociation or drift?
2. **Initial gas-phase problem** — clusters evaporate under the bias
   (`intact_heatmap.png`).
3. **CV diagnosis** — O–H vs O–O separates dissociation from drift
   (`cv_n1_high.csv` vs `cv_n3_low.csv`; `movie_n3_low.gif`).
4. **PBC correction** — periodicity confines the system; note GFN2→GFN1 and the
   box-size fix (`pbc_vs_gas.png`, `cv_n8_PBC_v2.csv`).
5. **Temperature result** — higher T, faster/farther drift, molecules intact
   (`temperature_vs_time.png`).
6. **Limitations and next validation step** — small systems, GFN1-for-PBC, and
   the planned unbiased-controls + replicas extension.

---

## 9. Paste-ready workdoc entry

> **xTB metadynamics — progress since July 3 (completed 2026-07-10).**
> All July 3 assignment items were completed. A three-level bias sweep
> (`kpush` = 0.008 / 0.02 / 0.05) on gas-phase water clusters (H2O)1–4 showed
> that the metadynamics bias preferentially breaks the weakest available bond:
> the lone water resists and only dissociates at high bias (breaking a covalent
> O–H), whereas hydrogen-bonded clusters separate even at the lowest bias.
> A 1 fs timestep was found to be unstable under stronger bias; 0.5 fs was
> adopted for all production runs. A temperature sweep of the water trimer at
> 200 / 250 / 298 / 350 K (fixed bias) showed that higher temperature drives
> faster and farther separation (maximum O–O 80.6 → 143.3 Å) while
> intramolecular O–H distances remain ≈1.03 Å, confirming intact-molecule drift
> rather than dissociation.
>
> Multiple water molecules were prepared with a repeat-and-equilibrate
> workflow — a loose grid condensed by unbiased NVT MD, reducing the minimum
> O–O distance from ≈3.88 to ≈2.68 Å (a hydrogen-bonded configuration). A
> Packmol input generator was also prepared but Packmol execution was deferred
> because the Windows executable lacked the required Fortran runtime libraries;
> repeat-plus-NVT-equilibration served as the accepted preparation route.
>
> A periodic workflow was implemented using `$periodic 3` / `$cell`. Two xTB
> facts were established: GFN2-xTB cannot run under periodic boundaries in this
> build (multipoles unavailable), so GFN1-xTB was used; and `xtb.trj` does not
> retain the periodic cell, so the cell was restored during analysis and all
> periodic distances evaluated with the minimum-image convention. An initial
> 7.76 Å periodic box produced unphysical O–H distances under bias and was
> superseded; the corrected run used a 9.20 Å box, kpush = 0.008, 0.5 fs and
> GFN1. In the corrected run the maximum O–O distance stayed bounded (≤7.69 Å,
> within the box) while the gas-phase reference separated beyond 100 Å,
> demonstrating that periodic boundaries remove the gas-phase evaporation
> artifact by spatial confinement; molecular integrity under GFN1 in a dense
> box was not cleanly resolved and true liquid-like behaviour is not claimed.
>
> A collective-variable analysis was developed to separate chemical
> dissociation from physical separation using assigned O–H distance (molecular
> integrity), intermolecular O–O distance (spatial separation) and a smooth
> oxygen-neighbour coordination number (O–O based, not an angular hydrogen-bond
> count), with minimum-image distances for periodic trajectories. It was
> validated on a single-water high-bias trajectory (genuine O–H dissociation)
> and a three-water low-bias trajectory (O–H ≈1.06 Å with O–O > 100 Å, i.e.
> intact drift). An ASE-based visualization workflow produced bonded PNG frames
> and animated GIFs for the gas and periodic trajectories, making the
> dissociation-versus-drift distinction directly visible.
>
> Methodological caveats: small finite systems (1–8 waters), short (20 ps) and
> largely single trajectories, sensitivity to box size and bias, GFN1 required
> for PBC, and Packmol deferred. As additional validation (not part of the
> completed assignment), an unbiased-controls and independent-initial-
> configuration-replicas study is planned, using a shared GFN1-optimized
> 8-water starting geometry, connectivity-based fragmentation metrics, an
> oxygen-neighbour coordination number, and per-oxygen O–H coordination to
> expose proton transfer; only a partial stability smoke test has been run so
> far.
