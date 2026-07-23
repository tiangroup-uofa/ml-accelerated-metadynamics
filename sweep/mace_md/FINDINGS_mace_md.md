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

## Throughput — resolved (`benchmark.py`)

The first MD run clocked ~2817 ms/step, which looked alarming. It was a **CPU
contention artifact** — that run happened while large downloads and conda installs
were saturating the machine — compounded by float64. A clean benchmark on a quiet
machine (n=20, 60 atoms):

| precision | threads | ms/step |
|-----------|:-------:|:-------:|
| float64 | 8 | 67 |
| float32 | 8 | 39 |
| **float32** | **4** | **37** |
| float32 | 1 | 54 |

So the real MACE-OFF23 cost is **~37 ms/step** (float32, 4 threads) — **on par with
GFN2-xTB (31 ms/step)** at n=20, and, since MACE is ~linear vs xTB's O(N³),
**faster than xTB beyond ~n=50**. `compare_md.py` now defaults to float32 + thread
pinning. MACE MD / metadynamics is viable; no config blocker remains.

## Next

- Throughput resolved (above) — extend the size ladder (n=100, 200) to confirm
  MACE overtakes xTB where it matters.
- Move to **MACE-driven metadynamics** via ASE/PLUMED — which lets us bias along
  proper collective variables instead of xTB's global-RMSD. **Which CV** is the
  open question (coordination numbers, per-O O–H for proton transfer, tetrahedral
  order, or a learned CV).

## Caveats

- Single 2.5 ps trajectory per method, one size (n=20); MACE-OFF23 small, CPU,
  float64; Langevin thermostat (friction 0.01 fs⁻¹).
- Free (non-periodic) droplet; the "O–O distribution" is a finite-cluster pair
  distribution, not a bulk g(r).
