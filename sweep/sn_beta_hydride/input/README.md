# Sn-beta input structures and structure policy

## Policy

1. **Authoritative structures are preferred.** These are original CPMD frames or
   published coordinates, with known provenance. They go directly in `input/`
   as `reactant.xyz` / `product.xyz` and are configured in `../config.json`
   (`provenance.structure_class = "authoritative"`).
   **Status: none received yet.** `../config.json` is deliberately unconfigured.
2. **Reconstructed structures are allowed only when explicitly labelled.**
   These are hand-built or agent-assisted models. Each set gets its own
   subdirectory and its own config, with a `provenance` block. Validation fails
   unless that block states:
   - `structure_class: "reconstructed"`
   - `not_literature_coordinates: true`
   - `constructed_by`
   - `construction_method`
   - `charge_spin_status` (`inferred` | `literature` | `computed`)
3. **Validation first, then relaxation with the method in use.** Production
   runs on reconstructed structures require both:
   - every stage validates first (hard errors refuse the run)
   - `run_path.py` refuses `--allow-unoptimized` for reconstructed sets, so
     the endpoints must first be relaxed with the same method
     (`optimize_endpoints.py`, or `run_protocol.py`, which does it).
4. Every result JSON carries the config's `provenance` block. Stages print a
   RECONSTRUCTED banner.

Never edit received coordinates in place. If a structure needs a fix, add a new
file, a new provenance entry and the reason.

## Datasets

| directory | class | config | status |
|---|---|---|---|
| `input/` (`reactant.xyz`, `product.xyz`) | authoritative | `../config.json` | **empty**, waiting on the original structures |
| [`tian_tt_test/`](tian_tt_test/README.md) | **reconstructed** | `../config_tian_tt_test.json` | 42-atom vacuum cluster from Tian (tt-test, b97ef46); validated; charge 0 / singlet **inferred** |

## Requirements for any structure pair

Both files must:

- contain **one** structure each (multi-frame files are rejected)
- have the **same atoms in the same order** (index *i* is the same atom in both)
- use Å. For a periodic model, use extended XYZ with `Lattice="..."` and
  `pbc="T T T"`, with an identical cell in both files.

The config must set:

1. `system.charge`, `system.spin_multiplicity` (2S+1) and
   `system.expected_n_atoms`. `system.expected_formula` is optional.
2. `atom_map` (**zero-based** ASE indices): `C1`, `C2`, `H_transfer` (H₂ in the
   paper's Fig. 1c), `H_C1` (H₁), `Sn`, `O1` (the C1 oxygen), `O2` (the C2
   oxygen) and `O_water` (the water formed at Sn).
3. `fixed_atoms`, if the model is a cluster with frozen termination atoms.
4. Optionally `capping`, to check the termination groups (e.g. three O–SiH₃
   on Sn).
5. The CV parameters `r0`, `n`, `m` (= CPMD d⁰, p, p+q), once they are known
   from the original CPMD input.

Then validate:

```bash
python ../validate_structure.py --config ../config.json --json ../outputs/input_validation.json
```

## Provenance of the authoritative set (fill in when received)

- Source (person / file / trajectory frame):
- Original level of theory (CPMD functional, cutoff, cell):
- Cluster cut or periodic? Termination scheme:
- Any edits made after receipt (and why):
