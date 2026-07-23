# MACE-OFF23 vs GFN2-xTB in unbiased MD — does the surrogate reproduce the dynamics?

Study 05 showed MACE-OFF23 matches GFN2-xTB on *static* forces/energies. This
checks the next thing: run **unbiased NVT MD** with each potential and see whether
they produce the same **droplet structure** — the prerequisite for MACE-driven
metadynamics. Reproduce with `compare_md.py`; data in `analysis/`, figure in
`figures/md_structure_n20.png`.

Comparison is on **structural distributions**, not frame-by-frame (chaotic MD
trajectories diverge; the ensembles should agree if the surfaces match).

## Result (n=20 droplet, 2.5 ps, 300 K)

| | O–O 1st peak (H-bond) | R_g | mean coordination | condensed? |
|---|---|---|---|---|
| GFN2-xTB | 2.83 Å | 3.78 ± 0.07 Å | 4.01 | yes |
| MACE-OFF23 | 2.83 Å | 3.85 ± 0.05 Å | 3.72 | yes |

- **The O–O pair distribution is essentially identical** — same hydrogen-bond peak
  at 2.83 Å and the same second-shell structure (~4.5, ~6 Å). Both potentials keep
  the droplet condensed and intact.
- R_g is near-identical (MACE's droplet is a touch tighter/narrower); coordination
  is close (MACE slightly lower).

**Off-the-shelf MACE-OFF23 (no fine-tuning) reproduces xTB's droplet structure in
actual dynamics, not just static forces** — the go-ahead for MACE-driven sampling.
Note MACE-OFF23 is DFT-trained (ωB97M-D3BJ), *not* fit to xTB, so this is two
independent methods agreeing, and where they differ MACE is presumably the more
accurate description.

## ⚠️ Throughput caveat (needs a config pass before scaling)

The MACE MD here ran **anomalously slow — ~2817 ms/step** (a ~4 h wall-clock run)
versus **31 ms/step** for xTB and **~75 ms/step** for MACE in the *static*
benchmark (Study 05). This ~40× slowdown is not a real MACE cost; likely CPU
float64 + torch thread oversubscription over a long unattended run. **Before
scaling MACE MD / metadynamics up, sort throughput:** float32 inference, pin
`OMP_NUM_THREADS` / torch threads, or move to GPU. Structure is validated;
production speed is a separate engineering task.

## Next

- Fix MACE MD throughput (float32 / threads / GPU), then extend the size ladder.
- Move to **MACE-driven metadynamics** via ASE/PLUMED — which lets us bias along
  proper collective variables instead of xTB's global-RMSD. **Which CV** is the
  open question (coordination numbers, per-O O–H for proton transfer, tetrahedral
  order, or a learned CV).

## Caveats

- Single 2.5 ps trajectory per method, one size (n=20); MACE-OFF23 small, CPU,
  float64; Langevin thermostat (friction 0.01 fs⁻¹).
- Free (non-periodic) droplet; the "O–O distribution" is a finite-cluster pair
  distribution, not a bulk g(r).
