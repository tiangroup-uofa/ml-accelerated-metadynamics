# kpush should scale with system size — the per-atom normalization

Tian's question: the RMSD-bias `kpush` "should scale to a per-molecule/atom
basis." This derives *why* and confirms it — the same nominal `kpush` is
genuinely weaker on a bigger cluster, and the fix is `kpush ∝ N`.

## The law (derivation + numerical demo, `derive_bias_scaling.py`)

xTB's bias is a sum of Gaussians in **root-mean-square** deviation:

$$V(x) = \sum_\text{ref} k_\text{push}\,e^{-\alpha\,\mathrm{RMSD}^2},\qquad
\mathrm{RMSD}^2 = \tfrac{1}{N}\sum_i |x_i - x_i^\text{ref}|^2$$

The bias force on atom $i$ is

$$F_i = k_\text{push}\,\alpha\,e^{-\alpha\,\mathrm{RMSD}^2}\;\frac{2}{N}\,(x_i - x_i^\text{ref})$$

— it carries an explicit **1/N**. So the same `kpush` pushes each atom $1/N$ as
hard in a larger system; the per-atom bias effect is diluted by size. To keep it
constant, **`kpush` must scale ∝ N** (per-atom); per-molecule is the same with
$N = 3\times(\#\text{waters})$.

`derive_bias_scaling.py` confirms this on real water geometries (perturbations at
a fixed RMSD, so only the 1/N gradient term varies):

![Fixed kpush follows the 1/N line (diluted); N-scaled kpush keeps the per-atom
force constant across n=3→96 atoms.](figures/kpush_scaling_law.png){width=520}

| N atoms | per-atom force, fixed kpush | N-scaled kpush |
|--------:|:---------------------------:|:--------------:|
| 3  | 1.00 (ref) | 1.00 |
| 12 | 0.26 | 1.03 |
| 48 | 0.066 | 1.05 |
| 96 | 0.033 | 1.04 |

`F × N` is constant for fixed kpush; N-scaled kpush holds the per-atom force flat.

## Empirical confirmation (n=30 droplet)

Prediction: the trimer (n=3, 9 atoms) first fragments at `kpush = 0.008`, so by
the N-scaling the 30-water droplet (90 atoms) should reach the same per-atom bias
at `kpush ≈ 0.008 × 90/9 = 0.08`. We already knew n=30 stays fully intact through
`kpush = 0.05`. Running it at the predicted values:

| kpush | max O–O (Å) | min coordination | molecules evaporated | in droplet |
|------:|:-----------:|:----------------:|:--------------------:|:----------:|
| 0.008 / 0.02 / 0.05 | ~13.4 | 4.07–4.32 | 0 | 100% |
| **0.08** (N-scaled) | 17.8 | 3.54 | **1** | 93% |
| 0.12 | 15.5 | 3.54 | 1 | 97% |

![The droplet is intact up to kpush 0.05 and starts shedding a molecule at
~0.08 — the N-scaled prediction.](figures/kpush_confirmation_n30.png){width=520}

**The onset of fragmentation appears right at the N-scaled `kpush` (~0.08)**,
where it was absent at the un-scaled values — confirming that raising `kpush` in
proportion to size restores the bias effect the dilution had removed.

::: note
**Honest caveat.** The onset is soft (one molecule sheds, not full breakup),
because size changes *two* things at once: it dilutes the per-atom bias force
(1/N, corrected here) **and** it raises cohesion (interior waters have more
H-bonds). The per-atom `kpush ∝ N` law fixes the first exactly; the second means
a large droplet still resists somewhat more than the bare force-scaling predicts.
No O–H dissociation or proton transfer at these levels — still physical
evaporation.
:::

## Takeaway

Report and set `kpush` **per atom** (or per molecule), not as a raw number:
`kpush_eff = kpush_ref × N / N_ref`. This makes bias strength comparable across
cluster sizes — the practical rule behind the "droplets don't break" observation
in [FINDINGS_large.md](../large/FINDINGS_large.md).

## Reproduce

```bash
python derive_bias_scaling.py     # the 1/N law + figure (no xtb needed)
# empirical: n=30 metadynamics at the N-scaled kpush
#   xtb input.xyz --md --input metadyn.inp --gfn 2   (kpush=0.08, 0.12)
python ../large/analyze_large.py --run n30_k0.08
```
