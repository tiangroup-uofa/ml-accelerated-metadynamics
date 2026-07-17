# Larger vacuum water clusters — droplet stability & bias chemistry

GFN2-xTB, vacuum (no PBC), NVT 300 K, step 0.5 fs, 3 ps exploratory runs.
Answers the July-10 to-do list (steps 1/3 and 2). Data: `analysis/cv_n*.csv`;
figures: `figures/droplet_stability.png`, `figures/bias_chemistry_n30.png`.

## Step 1/3 — does a bigger free cluster hold as a droplet? YES.

Low bias (kpush=0.008), sizes n=20/30/50:

| n | Rg start→end (Å) | frac in droplet | evaporated | max O–H (Å) |
|---|---|---|---|---|
| 20 | 3.61 → 3.96 | 1.00 | 0 | 1.10 (intact) |
| 30 | 4.45 → 4.18 | 1.00 | 0 | 1.11 (intact) |
| 50 | 4.99 → 4.95 | 1.00 | 0 | 1.16 (intact) |

Every size stays a **single, compact, connected droplet** — stable radius of
gyration, nobody evaporates, all O–H bonds intact. This is the **opposite** of
the earlier n=2–4 clusters, which evaporated even at this same low bias
(`../FINDINGS.md`). **Evaporation is suppressed by size.**

*Why:* in a 2–4 water cluster every molecule is surface with few hydrogen bonds,
so the RMSD bias peels one off immediately; in a 20–50 water droplet, interior
molecules sit in a full H-bond cage (oxygen-neighbour coordination ≈ 4–5 here)
and cohesion beats the bias. A free droplet forms and holds in vacuum — no
periodic box needed to stop evaporation, unlike the n=8 PBC workaround (which
also forced GFN1); here we keep **GFN2** and still get a bounded droplet.

## Step 2 — raise kpush at n=30: when do bonds break / protons transfer?

Fixed n=30, kpush = 0.008 → 0.02 → 0.05:

| kpush | Rg range (Å) | O-neighbour coord | max O–H (Å) | proton transfers |
|---|---|---|---|---|
| low 0.008 | 4.17–4.66 | 4.32–5.07 | 1.11 | 0 |
| med 0.02 | 4.21–4.66 | 4.25–5.03 | 1.11 | 0 |
| high 0.05 | 4.21–4.75 | 4.07–4.87 | 1.10 | 0 |

**Over these 3 ps, raising the bias does essentially nothing to the droplet** —
no fragmentation, no O–H dissociation, no proton transfer at any bias. Higher
kpush only very slightly widens the excursions (coordination dips to 4.07 vs
4.32). So the "bond-breaking crossover" that was dramatic for small clusters
(monomer dissociates at kpush=0.05) **does not occur for the 30-water droplet**
at the same nominal kpush.

### The important caveat (and why this is actually useful)

This is **not** proof the droplet is indestructible. The xtb bias is a function
of **global Cartesian RMSD**, which is spread over **all** degrees of freedom.
A 30-water droplet has ~10× the atoms of a dimer, so the *same nominal kpush*
delivers a **much smaller per-atom push** — i.e. kpush=0.05 is effectively a
**gentle** bias at n=30 but an aggressive one at n=2. The droplet's apparent
robustness is therefore as much about **RMSD being size-dependent** as about
intrinsic cohesion. To reach the bond-breaking / proton-transfer regime in a
droplet you would need a **much larger kpush (scaled up with system size)**,
a longer run, or a **local/targeted bias** rather than global RMSD.

This is the concrete empirical basis for the **"normalize the bias per degree
of freedom"** heuristic (high-level direction 1): a single kpush number does not
transfer across cluster sizes, and this dataset shows exactly why and by roughly
how much.

## Bottom line

1. **Step 1/3:** free water droplets (n=20–50) are stable in vacuum GFN2 NVT at
   low bias — evaporation is suppressed by size, no PBC needed.
2. **Step 2:** the same nominal kpush that shreds small clusters barely touches a
   30-water droplet, because global-RMSD bias is diluted over system size — no
   bond breaking or proton transfer in 3 ps. Reaching that regime needs
   size-scaled bias.
3. Both results argue for a **size-normalized kpush** as the first modelling step.

## Caveats

- Short (3 ps) single trajectories — trends, not statistics.
- `n_transfer` / coordination are nearest-O / switching-function metrics, not
  angular H-bond counts; corroborate any future "event" with O–H distance.
- Vacuum GFN2 droplets of 20–50 waters are still small models of bulk water.
