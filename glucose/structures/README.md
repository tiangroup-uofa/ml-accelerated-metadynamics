# Glucose hydride transfer: local vacuum structures

Isolated copy of the local results in `sweep/sn_beta_hydride/build/`.
The original directory is unchanged. Coordinates and result JSONs were copied
byte-for-byte; `provenance.json` records source paths and SHA-256 hashes.

Reference: Mushrif, Varghese & Krishnamurthy, *Solvation dynamics and energetics
of intramolecular hydride transfer reactions in biomass conversion*, PCCP
**17**, 4961 (2015), DOI: https://doi.org/10.1039/C4CP05063K.

## Model and provenance

These are **locally constructed and relaxed structures, not the paper's
coordinates**. The starting builder interprets the published cluster drawing;
the existing literature audit says the full paper/SI was not accessible.
No authoritative coordinates were available.

The 42-atom model contains open-chain D-glucose bound to a SiH₃-capped Sn site:
Sn(OSiH₃)₃(H₂O)(O2)(O1). O2 starts as an alkoxide and C1=O coordinates to Sn.
The bound water is retained, but there is no bulk solvent, periodic boundary
condition, or frozen atom. Charge 0 and singlet multiplicity 1 are inferred
model choices. Atom ordering is shared throughout; see `atom_map.json`
(zero-based indices, transferring H = 13, C1 = 1, C2 = 2, Sn = 23).

## Saved structures

| File | Meaning |
|---|---|
| `reactant_mace-polar_polar-1-s.xyz` | Initial, locally relaxed glucose-side endpoint |
| `product_mace-polar_polar-1-s.xyz` | Final, locally relaxed hydride-transfer endpoint |
| `ts_mace-polar_polar-1-s.xyz` | Vacuum Sella saddle |
| `ts_reactant_side_mace-polar_polar-1-s.xyz` | Minimum reached after negative/positive-mode displacement toward reactant |
| `ts_product_side_mace-polar_polar-1-s.xyz` | Minimum reached on the product side |
| `scan_mace-polar_polar-1-s.xyz` / `.csv` | 15-point relaxed scan |
| `reactant_guess.xyz`, `product_guess.xyz` | Unrelaxed starting guesses, not the deliverable minima |

## Findings from the saved records

MACE-POLAR-1 `polar-1-s`, CPU, float64, four threads:

- Both original endpoints converged below 0.02 eV/Å; six O remain within
  2.8 Å of Sn.
- H moves from C2 to C1: C2–H / C1–H distances are 1.105 / 2.112 Å
  initially, and 2.115 / 1.100 Å finally. Carbonyl/alkoxide bond lengths swap.
- Product minus reactant potential energy: **−30.02 kJ/mol**.
- Forward / reverse saddle barriers relative to those endpoints:
  **112.97 / 142.99 kJ/mol**. Scan maximum: 118.46 kJ/mol.
- Sella converged in 43 steps; unprojected maximum force = 0.000688 eV/Å.
  TS C2–H / C1–H = 1.314 / 1.348 Å.
- Central-difference, mass-weighted Hessian: **one significant imaginary
  mode, −923.09 cm⁻¹** (counting below −50 cm⁻¹). Its overlap with the
  hydride-coordinate gradient is 0.750; H* carries 85.4% of the Cartesian
  mode displacement weight.

**Connectivity caveat:** displacing the TS by ±0.1 Å along its imaginary mode
and relaxing reaches the expected hydride-bonding sides, but not the exact
original conformers. The reactant-side minimum is 13.44 kJ/mol lower
(RMSD 1.245 Å), and the product-side minimum 2.58 kJ/mol lower (RMSD 0.996 Å).
Thus the saddle is supported as a hydride-transfer first-order saddle, but
an exact path connection to the original endpoint geometries is not established.
Using the lower side minima gives barriers of about 126.41 / 145.57 kJ/mol
and reaction energy −19.16 kJ/mol (derived from saved energies).
The displacement-and-relaxation check is **not a full IRC**.

These are vacuum potential energies, without ZPE or thermal corrections.
They are not metadynamics free-energy barriers or a quantitative reproduction
of the reference paper. No new calculations were run during this isolation.

## Setup and reproduction

The result JSONs record: `mace-torch 0.3.16`, `torch 2.12.0`,
`ase 3.28.0`, and `graph_longrange 0.4.0`. Use Python 3.10+ with NumPy,
SciPy, ASE, PyTorch, mace-torch, and Sella. RDKit is only needed to rebuild
the geometric guess. Model weights are downloaded/cached by MACE.

For mace-torch 0.3.16, the existing working setup used this specific long-range
dependency; newer releases previously failed with
`unexpected keyword argument 'force_pbc_evaluator'`:

```bash
python -m pip install "mace-torch==0.3.16" "ase==3.28.0" numpy scipy sella
python -m pip install --no-deps \
  "git+https://github.com/WillBaldwin0/graph_electrostatics@v0.4.0"
```

Use a compatible PyTorch installation; the versions above are recorded run
metadata, not a newly tested environment lock.

The working scripts and all their local helper modules are in `scripts/`;
they do not import code from the original study directory. Their only functional
isolation change redirects inputs/outputs into a separate workspace and seeds
missing XYZ/JSON/CSV inputs from this archive. The archive is not overwritten.

From the project root:

```bash
# Optional: choose a fresh absolute output directory.
export GLUCOSE_WORKDIR="$PWD/glucose/structures/runs/reproduction"

# Re-relax the original geometric guess, then scan/search/check the saddle.
python glucose/structures/scripts/relax_polar.py
python glucose/structures/scripts/ts_sella.py

# Alternatively, use the archived scan and endpoints to repeat saddle validation.
# Run this in a fresh workspace if you want the archived endpoint energies.
python glucose/structures/scripts/ts_sella.py --reuse-scan
```

Without `GLUCOSE_WORKDIR`, outputs go to `runs/polar-1-s/`.
Scripts default to `--model mace-polar --variant polar-1-s --threads 4`.
Endpoint optimization uses FIRE (0.2 eV/Å) then BFGS (0.02 eV/Å);
scan BFGS uses 0.03 eV/Å; internal-coordinate Sella uses 0.001 eV/Å.
The Hessian step is 0.001 Å. To rebuild the initial guess, install RDKit and
run `scripts/build_guess.py` in a fresh workspace.
