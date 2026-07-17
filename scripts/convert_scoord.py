#!/usr/bin/env python3
"""
convert_scoord.py
=================
Collect the metadynamics reference structures that xTB writes to
``scoord.1``, ``scoord.2``, ... (Turbomole ``coord`` format, atomic
units / Bohr) and assemble them into a single multi-frame trajectory.

Outputs
-------
    scoord_1ps.xyz   -- multi-frame extended-XYZ (Angstrom)
    scoord_1ps.traj  -- ASE binary trajectory

Notes
-----
* ``scoord.*`` files are sorted NUMERICALLY, so scoord.2 precedes
  scoord.10 (lexicographic sorting would get this wrong).
* Every frame is validated to contain the same number and ordering of
  atoms; a mismatch raises an explicit error rather than failing
  silently.
* Each frame stores its source filename in ``atoms.info['source_file']``.

Usage
-----
    python convert_scoord.py --run-dir ../run --out-dir ../analysis
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ase.io import read, write
from ase.io.trajectory import Trajectory


def numeric_key(path: Path) -> int:
    """Return the integer suffix of a scoord.<N> filename for sorting."""
    m = re.search(r"\.(\d+)$", path.name)
    if m is None:
        # files without a numeric suffix sort last
        return sys.maxsize
    return int(m.group(1))


def find_scoord_files(run_dir: Path) -> list[Path]:
    """Find and numerically sort all scoord.* files in run_dir."""
    files = [p for p in run_dir.glob("scoord.*") if numeric_key(p) != sys.maxsize]
    if not files:
        raise FileNotFoundError(
            f"No 'scoord.*' files found in {run_dir.resolve()}.\n"
            "Did the xTB metadynamics run actually produce reference "
            "structures? Check that 'save' > 0 in metadyn.inp."
        )
    files.sort(key=numeric_key)
    return files


def read_turbomole(path: Path):
    """Read a single Turbomole coord/scoord file via ASE.

    ASE reads Turbomole 'coord' files and converts Bohr -> Angstrom
    automatically, so the returned Atoms object is in Angstrom.
    """
    try:
        atoms = read(str(path), format="turbomole")
    except Exception as exc:  # noqa: BLE001 - we want a helpful message
        raise RuntimeError(
            f"Failed to read '{path}' as a Turbomole coord file: {exc}\n"
            "Verify the file begins with '$coord' and ends with '$end'."
        ) from exc
    atoms.info["source_file"] = path.name
    return atoms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="../run", type=Path,
                        help="Directory containing scoord.* files (default ../run)")
    parser.add_argument("--out-dir", default="../analysis", type=Path,
                        help="Directory to write outputs (default ../analysis)")
    parser.add_argument("--xyz-name", default="scoord_1ps.xyz")
    parser.add_argument("--traj-name", default="scoord_1ps.traj")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_scoord_files(run_dir)
    print(f"Found {len(files)} scoord.* files in {run_dir}")
    print("Numeric order (first 5):",
          [f.name for f in files[:5]], "..." if len(files) > 5 else "")

    frames = []
    ref_symbols = None
    for idx, path in enumerate(files):
        atoms = read_turbomole(path)

        if ref_symbols is None:
            ref_symbols = list(atoms.get_chemical_symbols())
        else:
            symbols = list(atoms.get_chemical_symbols())
            if len(symbols) != len(ref_symbols):
                raise ValueError(
                    f"Atom-count mismatch: '{path.name}' has {len(symbols)} "
                    f"atoms but the first frame had {len(ref_symbols)}. "
                    "Frames cannot be compared. This often indicates "
                    "fragmentation under an overly aggressive bias."
                )
            if symbols != ref_symbols:
                raise ValueError(
                    f"Atom-ordering / element mismatch in '{path.name}'.\n"
                    f"  expected: {ref_symbols}\n"
                    f"  found:    {symbols}\n"
                    "All frames must share identical atom ordering for "
                    "RMSD and per-atom analyses to be valid."
                )
        frames.append(atoms)

    xyz_path = out_dir / args.xyz_name
    traj_path = out_dir / args.traj_name

    # Multi-frame extended XYZ. extxyz preserves atoms.info comments.
    write(str(xyz_path), frames, format="extxyz")
    print(f"Wrote {len(frames)} frames -> {xyz_path}")

    with Trajectory(str(traj_path), mode="w") as traj:
        for atoms in frames:
            traj.write(atoms)
    print(f"Wrote {len(frames)} frames -> {traj_path}")

    print(f"\nAtoms per frame: {len(ref_symbols)}")
    print(f"Chemical formula: {frames[0].get_chemical_formula()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
