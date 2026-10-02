# Authoritative Sn-beta input structures

**Status: EMPTY. Waiting on the original structures (Tian / original CPMD work).**

Do not put reconstructed, hand-built, or "approximately right" Sn-beta geometries
here. Every stage reads these files and treats them as ground truth.

## What goes here

| file | content |
|---|---|
| `reactant.xyz` | the state just before C2 → C1 hydride transfer, hydride (H\*) on **C2**. Expected from the literature description (Ali 2025 thesis §4.2 / Fig. 4.1, not verified against the paper): open-chain glucose deprotonated at C2–O and bound to Sn, with the water formed at the site still on Sn, on a Sn(–O–SiH₃)₃ cluster. Confirm against the received file. |
| `product.xyz` | the same system after C2 → C1 hydride transfer, H\* on **C1** |

Both files must:

- contain **one** structure each (multi-frame files are rejected);
- have the **same atoms in the same order** (index *i* is the same atom in both);
- use Å; for a periodic model, use extended XYZ with `Lattice="..."` and `pbc="T T T"`, with the identical cell in both files.

## After adding the files, fill in `../config.json`

1. `system.charge`, `system.spin_multiplicity` (2S+1) and `system.expected_n_atoms`.
2. `atom_map` (**zero-based** ASE indices): `C1`, `C2`, `H_transfer` (H₂ in the
   paper's Fig. 1c), `H_C1` (H₁), `Sn`, `O1` (C1 oxygen), `O2` (C2 oxygen) and
   `O_water` (the water formed at Sn).
3. `fixed_atoms`, if the model is a cluster with frozen termination atoms.
4. The CV parameters `r0`, `n`, `m` (= CPMD d⁰, p, p+q) taken from the original CPMD input, then set `parameters_confirmed: true`. The atoms (C1/C2 with H₁ + H₂) are already set from a secondary source (Ali 2025 thesis Fig. 4.1c).

Then run the validation, which also gives you the CV values:

```bash
/usr/bin/python3 ../validate_structure.py --config ../config.json --json ../outputs/input_validation.json
```

## Provenance (fill in)

- Source (person / file / trajectory frame):
- Original level of theory (CPMD functional, cutoff, cell):
- Cluster cut or periodic? Termination scheme:
- Any edits made after receipt (and why):
