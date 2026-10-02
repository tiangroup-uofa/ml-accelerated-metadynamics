# Authoritative Sn-beta input structures

**Status: EMPTY. Waiting on the original structures (Tian / original CPMD work).**

Do not put reconstructed, hand-built, or "approximately right" Sn-beta geometries
here. Every stage reads these files and treats them as ground truth.

## What goes here

| file | content |
|---|---|
| `reactant.xyz` | glucose coordinated to the partially hydrolyzed Sn-beta site, hydride (H\*) on **C2** |
| `product.xyz` | the same system after C2 → C1 hydride transfer, H\* on **C1** |

Both files must:

- contain **one** structure each (multi-frame files are rejected);
- have the **same atoms in the same order** (index *i* is the same atom in both);
- use Å; for a periodic model, use extended XYZ with `Lattice="..."` and `pbc="T T T"`, with the identical cell in both files.

## After adding the files, fill in `../config.json`

1. `system.charge`, `system.spin_multiplicity` (2S+1) and `system.expected_n_atoms`.
2. `atom_map` (**zero-based** ASE indices): `C1`, `C2`, `H_transfer`, `H_C1`, `Sn`.
3. `fixed_atoms`, if the model is a cluster with frozen termination atoms.
4. The `CV1`/`CV2` definitions (`group`, `r0`, `n`, `m`) taken from the original work. The shipped values are an **unconfirmed** inference.

Then run the validation, which also gives you the CV values:

```bash
/usr/bin/python3 ../validate_structure.py --config ../config.json --json ../outputs/input_validation.json
```

## Provenance (fill in)

- Source (person / file / trajectory frame):
- Original level of theory (CPMD functional, cutoff, cell):
- Cluster cut or periodic? Termination scheme:
- Any edits made after receipt (and why):
