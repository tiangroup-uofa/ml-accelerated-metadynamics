# Larger vacuum water clusters — droplet stability & bias chemistry

Implements the **July-10 to-do list**: go beyond the earlier n = 1..8 work to
larger free (vacuum) clusters and ask two questions.

- **Step 1 / 3 — droplet vs evaporation.** Keep the gentle settings that were
  stable before (low `kpush` = 0.008, step 0.5 fs, NVT 300 K) and scale the
  cluster up (**n = 20 → 30 → 50**). Does a bigger free cluster hold together
  as a **droplet**, or do molecules keep peeling off to infinity like the tiny
  n = 2..4 clusters did?
- **Step 2 — raise the bias.** Fix the size (**n = 30**) and increase `kpush`
  (low → med → high = 0.008 → 0.02 → 0.05). Do covalent **O–H bonds break**,
  or do **protons transfer** between molecules (autoionization signature)?

Everything is **GFN2-xTB in vacuum** (no cell, no `$periodic`) — so, unlike the
periodic n=8 runs that were forced onto GFN1, these keep the more accurate GFN2.
Timestep 0.5 fs (the value found stable under bias in the earlier work).

## Reproduce

```bash
export XTB_BIN=$(which xtb)          # any xtb 6.7.1; see note below
python build_large_clusters.py       # geometries/water_{20,30,50}.xyz
python run_large.py --step 1 --sizes 20 30 50 --time-ps 3   # droplet ladder
python run_large.py --step 2 --sizes 30 --time-ps 3         # raise kpush at n=30
python analyze_large.py              # per-run CVs -> analysis/cv_*.csv
python plot_large.py                 # figures/droplet_stability.png, bias_chemistry_n30.png
```

**xtb note.** This machine (Apple-Silicon macOS) has no native xtb; the bundled
`tools/xtb-6.7.1/bin/xtb.exe` is a Windows binary. xtb 6.7.1 was installed for
these runs via conda-forge (`micromamba create -n xtb -c conda-forge xtb`),
which provides an `osx-arm64` build of the **same 6.7.1** version. `run_large.py`
reads the binary from `$XTB_BIN`; on an HPC node just `module load xtb` and
`export XTB_BIN=$(which xtb)`.

## Collective variables (analyze_large.py)

Droplet vs evaporation:
- **Rg** — radius of gyration of the oxygens; flat = stable droplet, growing =
  falling apart.
- **n_evaporated** — waters with no other O within 3.5 Å (left the surface).
- **frac_in_main** — fraction of waters in the largest O–O connected cluster
  (1.0 = one intact droplet).

Bond breaking / proton transfer:
- **max_intra_OH** — longest H-to-nearest-O distance; ≥1.3 Å = a broken O–H.
- **n_transfer** — protons whose nearest oxygen changed from the start frame
  = proton that changed owner (the per-oxygen reassignment metric flagged as
  the stronger reactive CV in `JULY3_PROGRESS_RECORD.md`).

## Results

See `FINDINGS_large.md` (written from the analysis CSVs once the runs finish).

## Caveats

- Short (3 ps) exploratory runs, single trajectory each — trends, not statistics.
- Vacuum GFN2 clusters, still small models of bulk water.
- `n_transfer` counts nearest-O reassignment; a genuine transfer and a large
  librational excursion can both trip it, so corroborate with `max_intra_OH`
  and visual inspection before calling an event a true proton transfer.
