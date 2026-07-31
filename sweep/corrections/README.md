# Post-training correction models (MACE-OFF23 → GFN2-xTB)

Fit and benchmark lightweight post-training corrections that make MACE-OFF23
reproduce a reference method (GFN2-xTB here — a cheap stand-in for the eventual
DFT reference) **without retraining the foundation model**, using the existing
`CorrectedCalculator` API (`../diels_alder/neb/corrections.py`).

Four corrections:

| model | form | conservative? | params |
|---|---|:---:|---|
| global | F′ = α F, E′ = α E | yes | 1 |
| affine | F′ = α F, E′ = α E + β | yes | 2 |
| element | F′ᵢ = α₍Zᵢ₎ Fᵢ | **no** | #elements |
| delta | ΔE = Σ pair potentials, F′ = F − ∇ΔE (`pairwise_delta.py`) | yes | 48 (linear) |

## Pipeline

```bash
# 1. dataset: configs with meaningful forces (water MD frames + rattled DA path)
python build_geoms.py
micromamba run -n macemd python eval_ref.py --calc mace --out data/mace.npz
micromamba run -n macemd python eval_ref.py --calc xtb  --out data/xtb.npz   # separate process!

# 2. fit + force metrics (train/val, per-domain)
python fit_corrections.py            # -> fitted_corrections.json

# 3. downstream benchmarks
micromamba run -n macemd python eval_da.py                   # Diels-Alder energetics
micromamba run -n macemd python eval_water.py --which xtb    # droplet reference
micromamba run -n macemd python eval_water.py --which mace   # baseline + 4 corrections

# 4. figures + summary table
python plot_corrections.py           # figures/ + summary_table.md
```

`build_corr.py` reconstructs the fitted corrections as `CorrectedCalculator`s so
every benchmark uses identical models. MACE and xtb-python are always evaluated in
**separate processes** (both link OpenMP). Conclusions + recommendation in
[`FINDINGS_corrections.md`](FINDINGS_corrections.md).
