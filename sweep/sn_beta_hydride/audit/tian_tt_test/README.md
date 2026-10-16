# Audit of Tian's tt-test structures and saved MACE-POLAR results

No calculator was run to make anything in this directory. Every number is
recomputed from saved coordinates and total energies. To regenerate it all
(seconds; numpy/ASE/matplotlib only):

```bash
cd sweep/sn_beta_hydride && python audit_tian_tt_test.py
```

| file | content |
|---|---|
| `input_validation.json` | full pair validation of `input/tian_tt_test/{reactant,product}.xyz` |
| `structure_validation.json` | single-structure validation of Tian's POLAR TS and both side minima |
| `tian_polar_summary.json` | file integrity, XYZ/JSON energy cross-check, both-reference-state energetics, scan maximum (kept separate), saved saddle evidence, conformer diagnostics, agreement with Tian's JSON and his handoff table |
| `structure_views.png` | x/y/z views of R, TS and P in one Sn-centred frame. **Qualitative only.** |

## Evidence classes

| class | items |
|---|---|
| **Verified from saved raw data** | file hashes (12/12 match Tian's record); composition and atom order; every distance below; Tian's POLAR arithmetic, under both reference states |
| **Strong, but not independently certified** | POLAR saddle and frequency: Sella converged (fmax 6.9e-4 eV/Å); one significant imaginary mode, −923.1 cm⁻¹; overlap 0.75 with δ; H\* carries 85% of the mode; ±mode side minima are reactant-like and product-like. Our pipeline has **not** recomputed the Hessian, and the eigenvector was not saved. |
| **Preliminary / table only** | MACE-OMOL 102.8 kJ/mol and GFN2-xTB 101.9 kJ/mol: there are no raw files on any branch |
| **Not established** | a DFT reference for this exact cluster and state; 77–78 kJ/mol as a comparable reference; global-minimum conformers; the solvent effect for this model |

## MACE-POLAR-1-S, vacuum, electronic energies (kJ/mol)

| quantity | vs `relaxed_endpoint` | vs `connected_side_minimum` |
|---|---|---|
| barrier E(TS) − E(R) | **112.97** | **126.41** |
| reverse barrier E(TS) − E(P) | 142.99 | 145.57 |
| ΔE = E(P) − E(R) | **−30.02** | **−19.17** |

- **Tian's handoff table** (126.4 / −19.2) uses the `connected_side_minimum`
  reference. His JSON headline (112.97 / −30.02) uses `relaxed_endpoint`.
- **The relaxed-scan maximum is not a TS.** It sits 118.46 kJ/mol above the
  relaxed reactant and 5.49 kJ/mol above the Sella saddle.

## Geometry (from the coordinates, Å)

| | C2–H\* | C1–H\* | C1–O1 | C2–O2 | Sn–O1 | Sn–O2 | Sn–O_water | hydride state |
|---|---|---|---|---|---|---|---|---|
| reactant | 1.105 | 2.112 | 1.218 | 1.387 | 2.339 | 2.043 | 2.311 | reactant-like |
| TS | 1.314 | 1.348 | 1.297 | 1.316 | 2.107 | 2.128 | 2.320 | shared |
| product | 2.115 | 1.100 | 1.385 | 1.220 | 1.992 | 2.473 | 2.377 | product-like |

## Conformer sensitivity

| side minimum vs relaxed input | RMSD whole | RMSD reactive core | ΔE | torsions changed ≥ 30° | flag |
|---|---|---|---|---|---|
| reactant side | 1.245 Å | 0.083 Å | −13.44 kJ/mol | C1–C2–C4–O5 −70°, C2–C4–O5–H15 +53° | **MATERIAL** (conformational) |
| product side | 0.996 Å | 0.242 Å | −2.58 kJ/mol | none | below 4.18 kJ/mol |

The 13.4 kJ/mol drop on the reactant side comes from a backbone rotation and
rotation of the caps. The reactive core is unchanged. It is a lower **local**
minimum, not a global minimum: no conformer search has been done.
