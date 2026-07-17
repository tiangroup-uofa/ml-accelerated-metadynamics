# Setup

## xtb binary (not committed)

The xtb 6.7.1 program itself is **not** in this repository — the `tools/`
directory (~123 MB) is git-ignored because it held a platform-specific
(Windows) build. Install xtb 6.7.1 yourself:

- **conda / mamba (macOS arm64, Linux):**
  `conda install -c conda-forge xtb`  (or `micromamba create -n xtb -c conda-forge xtb`)
- **HPC:** `module load xtb/6.7.1`

Then tell the scripts where it is:

```bash
export XTB_BIN=$(which xtb)
```

The newer drivers under `sweep/large/` and `sweep/diels_alder/` read `$XTB_BIN`
(falling back to `xtb` on `PATH`). Some of the original scripts under `sweep/`
were written for the bundled Windows path and may need the same treatment.

## Python

Analysis needs Python 3 with `ase numpy pandas matplotlib` (scipy + `xtb-python`
+ `sella` additionally for the Diels-Alder transition-state search):

```bash
pip install ase numpy pandas matplotlib scipy
# for the DA TS refinement (sweep/diels_alder/refine_ts_sella.py):
conda install -c conda-forge xtb-python
pip install sella
```
