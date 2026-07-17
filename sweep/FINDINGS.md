# Bias-strength / cluster-size sweep of gas-phase xTB metadynamics

**System:** gas-phase water clusters (H2O)_n, n = 1..4
**Method:** GFN2-xTB metadynamics (RMSD bias), NVT 300 K, 20 ps, step 0.5 fs
**Sweep:** cluster size {1,2,3,4} x bias strength kpush {0.008 low, 0.02 med, 0.05 high} = 12 runs

All runs, geometries, scripts and figures live under `sweep/`. Reproduce with:

```
python build_clusters.py      # starting geometries
python run_sweep.py           # the 12 metadynamics runs
python analyze_sweep.py       # per-run intactness table
python plot_sweep.py          # heatmap + fragmentation/spread time series
python plot_chemistry.py      # H-bond distance & energy-vs-CV plots
```

---

## Headline result

`figures/intact_heatmap.png` — fraction of the 20 ps trajectory each cluster
stayed **intact** (one connected piece, all O-H bonds present):

| cluster | low (0.008) | med (0.02) | high (0.05) |
|---------|:-----------:|:----------:|:-----------:|
| (H2O)1  | **1.00**    | **1.00**   | 0.21 |
| (H2O)2  | 0.00        | 0.07       | 0.02 |
| (H2O)3  | 0.10        | 0.03       | 0.01 |
| (H2O)4  | 0.15        | 0.12       | 0.06 |

## Two clear trends

**1. Bias strength: more bias -> breaks sooner (every size).**
Reading across any row, higher kpush lowers the intact fraction / shortens the
time to break. Clearest for the trimer, which first loses its single-piece
structure at **1.7 ps (low) -> 0.2 ps (med) -> 0.1 ps (high)**. The single
water only dissociates at all once the bias is high (kpush=0.05, breaks at
4.2 ps); at low/med it survives the full run.
See `figures/hbond_distance_vs_time_n{2,3,4}.png`: the intermolecular O...H
bond climbs out of the ~1.5-2.5 A H-bond band earliest for the high-bias (red)
run and latest for low-bias (green).

**2. What breaks depends on what's holding the system together.**
- The **monomer** can only come apart by breaking a strong **covalent O-H
  bond (~5 eV)**, so it resists the bias and only fails at high kpush.
- The **clusters** come apart by breaking a weak **intermolecular hydrogen
  bond (~0.2 eV)**, which the RMSD bias snaps almost immediately even at low
  kpush. Crucially, each water stays internally intact (`final_oh_intact` is
  mostly True for the clusters) — the *cluster* evaporates, the *molecules*
  survive. The two/three/four monomers drift tens-to-hundreds of A apart
  (`spread_vs_time.png`).
- Among the clusters, the **tetramer is the most robust** (highest intact
  fractions), consistent with each molecule having more hydrogen bonds to hold
  it in the ring.

## Energy along the separation CV

`figures/energy_vs_cv_n{2,3,4}.png` and `energy_vs_cv_all.png` plot potential
energy vs the collective variable = max O-O distance. Energy rises steeply
while the cluster is compact, then **flattens to a plateau** as the fragments
separate — the dissociation asymptote of non-interacting monomers. Note this
is a **sampled potential-energy landscape from a biased run, not a free-energy
surface** (recovering the latter needs reweighting of the metadynamics bias;
out of scope, same caveat as the main project README).

## Practical takeaway

For gas-phase metadynamics on hydrogen-bonded clusters, even a small RMSD bias
preferentially rips the weak intermolecular bonds and evaporates the cluster
before it can usefully explore intramolecular rearrangements. To study cluster
conformers rather than evaporation one would need to **restrain the O-O
distances** (a wall/repulsive-sphere potential) or use a much gentler bias —
a natural next experiment.
