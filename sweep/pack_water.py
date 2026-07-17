#!/usr/bin/env python3
"""
pack_water.py
=============
Prepare periodic water boxes with Packmol (https://m3g.github.io/packmol/).

This does two things:
  1. Always writes a Packmol input file (water_<N>.inp) and a single-water
     template (water_single.pdb) into packmol/, so the packing is fully
     specified and reproducible even before Packmol is installed.
  2. If a real Packmol executable is available (via --packmol PATH or found
     on PATH), it runs it to produce the packed box water_<N>_packed.pdb,
     then converts that to an xTB Turbomole 'coord' with periodic cell.

Box size is chosen for liquid-like density (~1 g/cm^3) so the NVT metadynamics
starts from a condensed configuration that cannot trivially evaporate.

Usage:
    python pack_water.py --n 8                 # just write the .inp
    python pack_water.py --n 8 --packmol C:/path/to/packmol.exe   # + run it
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACK = HERE / "packmol"

# One rigid water in PDB format (A). O at origin, standard geometry.
OH = 0.9584
HOH = np.radians(104.45 / 2.0)
WATER_PDB = f"""HEADER    single water template
ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H1  HOH A   1     {OH*np.sin(HOH):7.3f} {0.0:7.3f} {OH*np.cos(HOH):7.3f}  1.00  0.00           H
ATOM      3  H2  HOH A   1     {-OH*np.sin(HOH):7.3f} {0.0:7.3f} {OH*np.cos(HOH):7.3f}  1.00  0.00           H
END
"""

# Mass of water (g/mol) and Avogadro for density -> box-size math.
M_WATER = 18.015
NA = 6.02214076e23


def box_length_for_density(n: int, rho_g_cm3: float = 1.0) -> float:
    """Cubic box edge (Angstrom) holding n waters at target density."""
    mass_g = n * M_WATER / NA                       # total mass in grams
    vol_cm3 = mass_g / rho_g_cm3
    vol_A3 = vol_cm3 * 1.0e24                        # cm^3 -> A^3
    return float(vol_A3 ** (1.0 / 3.0))


def write_inputs(n: int, rho: float, pad: float = 1.5):
    PACK.mkdir(parents=True, exist_ok=True)
    (PACK / "water_single.pdb").write_text(WATER_PDB)
    L = box_length_for_density(n, rho)
    # keep molecules 'pad' A inside the walls so PBC images don't clash
    lo, hi = pad, L - pad
    inp = f"""# Packmol input: {n} waters in a cubic box, ~{rho} g/cm^3, edge {L:.3f} A
tolerance 2.0
filetype pdb
output water_{n}_packed.pdb

structure water_single.pdb
  number {n}
  inside box {lo:.3f} {lo:.3f} {lo:.3f} {hi:.3f} {hi:.3f} {hi:.3f}
end structure
"""
    (PACK / f"water_{n}.inp").write_text(inp)
    print(f"n={n}: density {rho} g/cm^3 -> cubic box edge L = {L:.3f} A")
    print(f"  wrote {PACK / f'water_{n}.inp'}")
    print(f"  wrote {PACK / 'water_single.pdb'}")
    return L


def find_packmol(explicit: str | None) -> str | None:
    if explicit:
        p = Path(explicit)
        return str(p) if p.exists() else None
    return shutil.which("packmol")


def run_packmol(n: int, packmol: str):
    inp = PACK / f"water_{n}.inp"
    print(f"  running: {packmol} < {inp.name}")
    with inp.open() as fh:
        proc = subprocess.run([packmol], stdin=fh, cwd=str(PACK),
                              capture_output=True, text=True)
    (PACK / f"packmol_{n}.log").write_text(proc.stdout + proc.stderr)
    packed = PACK / f"water_{n}_packed.pdb"
    if proc.returncode != 0 or not packed.exists():
        print(f"  Packmol FAILED (rc={proc.returncode}); see packmol_{n}.log",
              file=sys.stderr)
        return None
    print(f"  packed -> {packed}")
    return packed


def pdb_to_periodic_coord(pdb: Path, L: float, out: Path):
    """Convert a packed PDB to xTB Turbomole 'coord' with a cubic $cell.

    coord is in Bohr; PBC declared with $periodic 3 and a cubic cell.
    """
    from ase.io import read
    ANG2BOHR = 1.8897259886
    atoms = read(str(pdb))
    pos = atoms.get_positions() * ANG2BOHR
    sym = atoms.get_chemical_symbols()
    Lb = L * ANG2BOHR
    lines = ["$coord"]
    for (x, y, z), s in zip(pos, sym):
        lines.append(f"{x:20.12f}{y:20.12f}{z:20.12f}   {s.lower()}")
    lines += [
        "$periodic 3",
        "$cell",
        f" {Lb:.9f} {Lb:.9f} {Lb:.9f} 90. 90. 90.",
        "$end",
        "",
    ]
    out.write_text("\n".join(lines))
    print(f"  wrote periodic coord -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of waters")
    ap.add_argument("--density", type=float, default=1.0, help="g/cm^3")
    ap.add_argument("--packmol", default=None,
                    help="path to packmol executable (else search PATH)")
    args = ap.parse_args()

    L = write_inputs(args.n, args.density)

    pk = find_packmol(args.packmol)
    if pk is None:
        print("\nPackmol not found. Input files are ready; re-run with "
              "--packmol PATH once installed:")
        print(f"  python pack_water.py --n {args.n} --packmol C:/path/to/packmol.exe")
        return 0

    packed = run_packmol(args.n, pk)
    if packed is None:
        return 1
    pdb_to_periodic_coord(packed, L, PACK / f"water_{args.n}_periodic.coord")
    print("\nDone. Use the periodic coord as the metadynamics input.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
