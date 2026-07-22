# MACE-OFF23 vs GFN2-xTB on the water droplets — first validation

First step of the MLIP-surrogate direction: compare **MACE-OFF23 (small)**
against **GFN2-xTB** on the droplet trajectories we already generated
(`sweep/large/runs/n{20,30,50}/low`, 30 frames each). Reproduce with
`compare_mace_xtb.py`; data in `analysis/mace_xtb_metrics.csv`, figures in
`figures/`.

**Framing:** MACE-OFF23 is DFT-trained (ωB97M-D3BJ), GFN2-xTB is semiempirical,
so this measures the **discrepancy between two surrogates**, not MACE-against-
truth (MACE is the more accurate one). We compare the physically meaningful
quantities: force components (same units) and *relative* energies (absolute
references cancel over a fixed-composition trajectory).

## Results

| n | atoms | force MAE | force RMSE | force r | rel-E RMSE | rel-E r | MACE ms/frame | ×xTB |
|---|---|---|---|---|---|---|---|---|
| 20 | 60 | 0.142 | 0.185 eV/Å | 0.992 | 3.29 meV/atom | 0.974 | 75 | 2.5× |
| 30 | 90 | 0.146 | 0.190 eV/Å | 0.992 | 2.53 meV/atom | 0.987 | 96 | 1.6× |
| 50 | 150 | 0.152 | 0.198 eV/Å | 0.991 | 2.30 meV/atom | 0.983 | 172 | 1.0× |

## Two takeaways

**1. The two surfaces agree closely in shape.** Force components correlate at
**r ≈ 0.99** (tight parity plot, `figures/force_parity.png`) and relative
energies track at **r ≈ 0.97–0.99, ~2–3 meV/atom** (`figures/relative_energy.png`).
There is a systematic ~0.15 eV/Å force spread — expected for two different levels
of theory — but the **landscape shape is consistent**, so MACE-OFF23 is a
sensible drop-in for the xTB PES on these systems (and, being DFT-trained,
presumably the more accurate description of the two).

**2. The scaling crossover is already visible.** MACE is ~linear-scaling and xTB
is ~O(N³) (SCF diagonalisation). MACE's cost relative to xTB falls
**2.5× → 1.6× → 1.0×** over n=20 → 30 → 50: by 150 atoms they are equal speed on
this CPU, and **beyond ~n=50 MACE wins**. This is the concrete evidence for the
"use the MLIP where xTB scaling hurts" thesis — the payoff regime is larger
droplets, exactly as expected. (Absolute timings are CPU/precision-dependent —
CPU, float64; float32/GPU would shift the crossover — but the *scaling trend* is
the robust signal.)

## Implications / next

- MACE-OFF23 is trustworthy enough on these droplets to **drive MD/metadynamics
  with it** (via ASE/PLUMED, not xTB's built-in RMSD bias). That unlocks
  **proper collective-variable biasing** instead of the chemically-blind global
  RMSD — which is the real open question (see below).
- Extend the size ladder past n=50 (n=100, 200, …) to map the crossover and the
  large-system regime where MACE is the only affordable option.
- **Open question (flagged by the group): which CVs to bias.** Candidates:
  coordination numbers (O–O and per-O O–H, the latter for proton transfer),
  local tetrahedral order, or a *learned* CV (TICA/VAMPnets/autoencoder on these
  trajectories). This is the next thing to pin down.

## Caveats

- MACE-OFF23 **small** model, CPU, float64; short single trajectories (30 frames
  each) at geometries sampled from *xTB* dynamics (so the comparison is on the
  xTB-visited region of configuration space).
- Timings include per-call ASE/calculator overhead; not a rigorous benchmark.
- No periodic systems here (free droplets); MACE-OFF23 is a non-periodic organic
  model — for condensed/PBC work a different reference/model choice is needed.
