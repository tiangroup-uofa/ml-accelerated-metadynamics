# `tian_tt_test` — reconstructed Sn-beta / glucose vacuum cluster

> **Reconstructed, not literature coordinates.** T. Tian built this model by hand,
> with agent-assisted SMILES conversion. It is our reading of the cluster drawing
> in Mushrif *et al.*, PCCP 17, 4961 (2015), Fig. 1b/c, as reproduced in Ali's
> 2025 thesis, Fig. 4.1. It is **not** the paper's or the CPMD run's geometry,
> since neither gives coordinates.

| | |
|---|---|
| Source | branch `tt-test`, commit `b97ef46`, `glucose/structures/` |
| Files | `reactant.xyz` and `product.xyz` are **byte-identical** copies of Tian's `reactant_/product_mace-polar_polar-1-s.xyz`; `atom_map_tian.json` is his `atom_map.json` |
| Model | 42 atoms, C₆H₂₂O₁₀Si₃Sn. Open-chain D-glucose (O2 deprotonated) bound to Sn(OSiH₃)₃(H₂O)(O2)(O1). Vacuum: no solvent, no PBC, no frozen atoms. |
| Charge / multiplicity | 0 / 1. **Inferred** by Tian: consistent with Sn(IV) plus four anionic O and 230 electrons, but not stated in any literature source. |
| Relaxation state | Local minima of **MACE-POLAR-1-S** near the hand-built guess. Every method re-relaxes them before any path calculation. |
| Atom map (0-based) | C1 = 1, C2 = 2, H\* = 13, H_C1 = 12, Sn = 23, O1 = 0, O2 = 3, O_water = 24 |
| Config | `../../config_tian_tt_test.json` |

`provenance.json` has the SHA-256 of each file (computed with CRLF → LF,
because a Windows checkout with `core.autocrlf` rewrites line endings).
`tests/test_tian_tt_test.py` checks them against both this record and Tian's
own `glucose/structures/provenance.json`.

## Known quirks (documented, not edited)

- **The XYZ comment line of both relaxed files is wrong.** It was inherited
  from the unrelaxed guess and reads "REACTANT, hand-built guess (unrelaxed)",
  even in `product.xyz`. The coordinates and the stored `energy=` are the
  relaxed ones. The files are kept byte-identical.
- **Validation passes with 0 errors and 5 expected warnings:**
  - reconstructed structures
  - charge/spin inferred
  - "Sn–O2 bond broken": Sn–O2 goes from 2.04 Å to 2.47 Å, just past the
    2.46 Å bond criterion. This is the intended swap of the O1/O2 roles at Sn.
  - cap hydrogens H34 and H36 move 2.6–2.8 Å between endpoints, because the
    SiH₃ groups rotated during the separate relaxations.

## How Tian built it (his scripts, `glucose/structures/scripts/`)

1. **`build_guess.py`:**
   - RDKit, 300 ETKDG conformers, MMFF-relaxed.
   - Among the 40 lowest, the one with the best ligand clearance (conformer
     199); its O2–C2–C1–O1 dihedral is set to 0°.
   - Sn and the ligands are placed geometrically.
2. **`relax_polar.py`:**
   - Reactant: FIRE to 0.2 eV/Å, then BFGS to 0.02 eV/Å.
   - Product guess: only H13 is moved from C2 onto C1, on the relaxed reactant.
   - The product is then relaxed the same way.
3. **`ts_sella.py`:**
   - 15-point relaxed scan along δ = d(C2–H\*) − d(C1–H\*), then Sella
     (internal coordinates, fmax 1e-3) from the scan maximum.
   - Hessian, then ±0.1 Å imaginary-mode displacement and relaxation.
   - This is **not** a CI-NEB.

Audit artifacts derived from these files are in
[`../../audit/tian_tt_test/`](../../audit/tian_tt_test/README.md).
