# MACE-OFF23 vs GFN2-xTB validation

First step of the MLIP-surrogate direction: how does the **MACE-OFF23** foundation
model compare to **GFN2-xTB** on the water droplets we already ran?

```bash
# needs mace-torch + xtb-python in the env (see ../../SETUP.md)
micromamba run -n xtb python compare_mace_xtb.py
```

Evaluates both potentials on `../large/runs/n{20,30,50}/low/xtb.trj`, comparing
force components and relative energies (the absolute energy references differ:
DFT-trained MACE vs semiempirical xTB), and records per-frame timing.

**Result:** force parity **r ≈ 0.99**, relative energies **~2–3 meV/atom**, and a
**speed crossover** — MACE goes 2.5×→1.6×→1.0× the xTB cost over n=20→30→50, i.e.
MACE (≈linear) overtakes xTB (≈O(N³)) by ~150 atoms. Full write-up:
[`FINDINGS_mace.md`](FINDINGS_mace.md). Outputs in `analysis/`, `figures/`.
