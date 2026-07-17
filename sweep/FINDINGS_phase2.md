# Phase 2 — packing, periodic boundaries, temperature, CV design, movies

This extends the earlier bias×size sweep (`FINDINGS.md`) with the methodological
tasks that were assigned next. All scripts and outputs are under `sweep/`.

## Task-by-task summary

### 1. Input preparation / packing
- **Packmol** (`pack_water.py`) writes a proper Packmol `.inp` + PDB template
  and converts the packed box to a periodic Turbomole `coord`. **Caveat:** the
  PyPI `packmol` wheel ships a `packmol.exe` that is dynamically linked against
  `libgfortran-5.dll` / `libquadmath-0.dll`, which are absent on this machine,
  so it crashes. Use `conda install -c conda-forge packmol` (bundles the DLLs)
  to enable this path.
- **No-install alternative used instead** (`prepare_box.py`, professor's option
  2): place N waters on a loose grid wider than the vdW contact distance, then
  run a short **unbiased NVT MD** so the thermostat condenses them. Verified:
  minimum O-O contracts from ~3.9 A (loose) to ~2.7 A (a real H-bond) — a
  liquid-like condensed box, ready for metadynamics.

### 2. Periodic boundary conditions (the evaporation fix)
- **KEY TECHNICAL FINDING:** GFN2-xTB **cannot run under PBC** in xtb 6.7.1 —
  it stops with `scf: Multipoles not available with PBC` (GFN2's multipole
  electrostatics are not implemented for periodic cells). Periodic runs must use
  **GFN1-xTB** (`--gfn 1`), which uses monopole electrostatics and supports the
  cell. So gas-phase (GFN2) and periodic (GFN1) results are NOT at the same
  level of theory — an important caveat for any comparison.
- **RESULT (`pbc_vs_gas.png`):** an 8-water periodic box (edge 9.2 A) run for
  20 ps at gentle bias (kpush=0.008) stays **condensed the whole time** — max
  O-O oscillates around 6-7 A and never exceeds the box, whereas the gas-phase
  trimer evaporates to >100 A. **Periodicity removes the evaporation channel.**
  The coordination CV (H-bonds per water) fluctuates ~1.3-3.0 and stays intact,
  i.e. the network reorganizes rather than falling apart — which is what we
  want metadynamics to sample.
- **Practical caveat:** with only 8 waters the periodic box is small (density
  and bias both matter). A too-small box or too-strong bias pushes atoms into
  unphysical overlaps; the roomier 0.6 g/cm^3 box + gentle bias behaves well.

### 3. Temperature axis (`run_temp_sweep.py`, `temperature_vs_time.png`)
Trimer at fixed kpush=0.02, NVT T in {200, 250, 298, 350} K. Higher temperature
-> the cluster drifts apart faster and farther (max O-O reached):

| T (K) | 200 | 250 | 298 | 350 |
|-------|-----|-----|-----|-----|
| max O-O (A) | 81 | 133 | 138 | 143 |

At every temperature the process is **drifting apart, not dissociation** (O-H
bonds stay intact) — the extra thermal energy just helps intact molecules
escape the H-bonds sooner.

### 4. CV design — dissociation vs drifting apart (`cv_analysis.py`)
The assigned question: *a large max O-O distance — is it a water DISSOCIATING
or the molecules DRIFTING APART?* Answered with a complementary CV pair, made
PBC-aware (minimum-image distances, since xtb.trj does not store the cell):

- **max intramolecular O-H** — grows only if a covalent O-H bond breaks
  (DISSOCIATION).
- **max O-O** — grows on both (ambiguous alone).
- **mean coordination number** (smooth switching function on O...O) — the
  principled structural CV; measures how intact the H-bond network is.

Classification: `dissociated` if max O-H >= 1.3 A; else `drifting_apart` if
max O-O >= 3.5 A; else `intact`. Validated on existing runs:

| run | max O-H | max O-O | verdict |
|-----|--------:|--------:|---------|
| (H2O)1 high bias | **1315 A** | 0 | **dissociated** (O-H broke) |
| (H2O)3 low bias  | 1.06 A | 105 A | **drifting apart** (intact molecules) |

So: the lone water **dissociates** (must break a covalent bond); the clusters
**drift apart / evaporate** (weak H-bonds break, molecules survive). Checking
the intramolecular O-H is what disambiguates the two.

### 5. Visualization (`make_movie.py`)
ASE renders each frame with bonds (covalent-radii) into a PNG stack, combined
into an animated GIF with PIL:
- `figures/movie_n3_low.gif` — gas trimer: starts compact, then three intact
  waters drift to the corners (drifting apart, each keeps its O-H bonds).
- `figures/movie_n8_PBC.gif` — periodic box: stays condensed (nothing leaves
  the frame), H-bond network rearranging.
Plus the earlier line plots: O-H distance vs time, energy-vs-CV surface.

## Bottom line for the write-up
1. Gas-phase metadynamics on small water clusters mostly **evaporates** them;
   the bias breaks the weakest bond (H-bond for clusters, covalent O-H only for
   the lone molecule).
2. **PBC fixes the evaporation** — molecules stay condensed and the H-bond
   network samples configurations — but forces GFN1 (GFN2 unsupported under
   PBC), and needs a sensibly sized box + gentle bias.
3. Temperature adds a second, independent knob: higher T evaporates faster.
4. The right CVs to report are **intramolecular O-H (dissociation) + O-O /
   coordination (drifting / network integrity)**, not O-O alone.
