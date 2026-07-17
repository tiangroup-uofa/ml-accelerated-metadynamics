#!/usr/bin/env python3
"""
prepare_input.py
================
Convert a starting geometry between extended-XYZ and Turbomole ``coord``
format using ASE, for use as the xTB metadynamics input.

ASE handles the Angstrom <-> Bohr conversion: extended-XYZ is in
Angstrom, Turbomole ``coord`` files are in Bohr. You generally do NOT
need to convert units by hand.

Usage
-----
    # XYZ -> Turbomole coord
    python prepare_input.py molecule.xyz coord

    # Turbomole coord -> XYZ
    python prepare_input.py coord molecule.xyz
"""
from __future__ import annotations

import sys
from pathlib import Path

from ase.io import read, write


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not src.exists():
        print(f"ERROR: input {src} not found.", file=sys.stderr)
        return 1

    atoms = read(str(src))

    # Choose output format from the destination name.
    if dst.name == "coord" or dst.suffix == "":
        write(str(dst), atoms, format="turbomole")
    else:
        write(str(dst), atoms)

    print(f"Converted {src} ({atoms.get_chemical_formula()}, "
          f"{len(atoms)} atoms) -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
