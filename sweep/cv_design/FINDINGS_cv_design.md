# Collective-variable scoping for water metadynamics

Tian's "hard one": which CVs to bias in MACE-driven metadynamics. This is a first
pass — compute a battery of candidate CVs on trajectories spanning the distinct
physical states, then ask which are informative, non-redundant, and capture the
processes we care about. Reproduce with `cv_scan.py`; data in `analysis/`.

## Candidate CVs (7) and the states tested

| CV | what it measures |
|----|------------------|
| `Rg` | radius of gyration of O atoms — condensation / size |
| `asphericity` | gyration-tensor shape (0 = sphere) |
| `max_OO` | largest O–O distance — spread / an atom leaving |
| `coord_OO` | mean O–O switching-function coordination — network density |
| `n_hbond_pm` | H-bonds per molecule (O···H < 2.5 Å **and** O–H···O > 150°) |
| `q_tet` | tetrahedral order (Errington–Debenedetti) — local liquid structure |
| `max_OH` | max intramolecular O–H — dissociation (chemistry) |

States: stable droplet (n30), fragmenting droplet (n30, kpush 0.08), big droplet
(n50), drifting small cluster (n3), monomer dissociation (n1).

## What separates the states

Between-state / within-state variance ratio (higher = more discriminating):

| CV | separation | defined on |
|----|:----------:|:----------:|
| **coord_OO** | **50.9** | all states |
| **n_hbond_pm** | **12.6** | all states |
| asphericity | 4.9 | droplets only |
| q_tet | 1.4 | droplets only |
| max_OO | 1.0 | all |
| Rg | 1.0 | all |
| max_OH | 0.8 | all |

`coord_OO` and the angular `n_hbond_pm` are by far the strongest state
discriminators. `max_OH` looks weak here **only because just one state (the
monomer) dissociates** — but it is the *sole* CV that fires on that state, so it
is essential as an orthogonal chemical coordinate, not redundant.

## Redundancy — the 7 CVs collapse to ~3 directions

Strongly correlated pairs (|r| > 0.9):

- `Rg ≈ max_OO` (r = +1.00) — both are just size/spread → keep one.
- `coord_OO ≈ n_hbond_pm` (r = +0.97) and `≈ −asphericity` (r = −0.90) — one
  "network-integrity" family.

PCA confirms it: **PC1 (43%) = network integrity** (coord / n_hbond / −asphericity),
**PC2 (30%) = size + dissociation** (Rg / max_OO / −max_OH); together 73% in two
components. In that 2-D CV space the states cleanly separate — droplets cluster at
high integrity, the drifting cluster traces a path across as its network breaks,
and the dissociating monomer splits off along the O–H axis
(`figures/cv_pca.png`, `figures/cv_correlation.png`).

## Recommended CV set (minimal, non-redundant)

1. **Network integrity** — mean O–O coordination *or* angular H-bond count
   (`coord_OO` / `n_hbond_pm`; pick one, they're redundant). The primary
   structural coordinate.
2. **Size / spread** — `Rg` (or `max_OO`; redundant). Separates condensed vs
   dispersed / evaporation.
3. **Chemistry** — `max_OH`, plus a **per-oxygen O–H coordination** to catch
   proton transfer / autoionization (orthogonal to the two structural CVs).

`q_tet` and `asphericity` add little beyond (1) here and are only defined for
droplets, so they're not primary choices.

## Important caveat & next step

This scan separates **different systems/states**, so a chunk of the discriminating
power (esp. `coord_OO`) reflects system-size differences, not slow dynamics
*within* one droplet. For choosing what to **bias**, the sharper question is which
CV captures the **slow reorganization within a single droplet** — that needs a
**time-lagged analysis (TICA / VAMP)** on one system's trajectory, or a learned CV
(autoencoder / VAMPnet). That is the natural next step, on top of this
physically-motivated shortlist.

## Reproduce

```bash
python cv_scan.py     # CV battery on 5 trajectories -> table, correlation, PCA
```
