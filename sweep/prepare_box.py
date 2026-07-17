#!/usr/bin/env python3
"""
prepare_box.py
==============
Prepare a condensed, periodic water box for metadynamics WITHOUT Packmol,
using the "repeat + equilibrate" recipe (professor's option 2):

  1. Place N waters on a loose cubic grid inside a periodic cell, spaced
     WIDER than the van der Waals contact distance so there are no clashes.
     (Grid spacing is chosen so the cell is a bit larger than the target
     liquid-density box; the molecules therefore start slightly too far
     apart -- exactly the intended starting point.)
  2. Write a periodic Turbomole 'coord' ($periodic 3 / $cell).
  3. Run a SHORT unbiased NVT MD (no metadynamics) so the thermostat pulls
     the molecules together into a condensed, liquid-like configuration.
  4. The final frame of that MD is the equilibrated box used to start
     metadynamics.

This is the no-install alternative to Packmol; the resulting condensed box
plays the same role (a realistic starting configuration that will not
trivially evaporate under PBC).

Usage:
    python prepare_box.py --n 8 --density 1.0 --equil-ps 5
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

HERE = Path(__file__).resolve().parent
OUT = HERE / "boxes"
XTB = (HERE.parent / "tools" / "xtb-6.7.1" / "bin" / "xtb.exe").resolve()

OH = 0.9584
HOH = np.radians(104.45 / 2.0)
M_WATER = 18.015
NA = 6.02214076e23
ANG2BOHR = 1.8897259886


def single_water(origin, rng) -> Atoms:
    """One water at origin with a random orientation (well-defined seed)."""
    o = np.array([0.0, 0.0, 0.0])
    h1 = np.array([OH * np.sin(HOH), 0.0, OH * np.cos(HOH)])
    h2 = np.array([-OH * np.sin(HOH), 0.0, OH * np.cos(HOH)])
    pos = np.array([o, h1, h2])
    # random rotation via a random unit quaternion -> rotation matrix
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1 - 2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1 - 2*(x*x+y*y)],
    ])
    pos = pos @ R.T + np.asarray(origin)
    return Atoms("OH2", positions=pos)


def box_length(n: int, rho: float) -> float:
    mass_g = n * M_WATER / NA
    vol_A3 = (mass_g / rho) * 1.0e24
    return float(vol_A3 ** (1.0 / 3.0))


def build_loose_grid(n: int, rho: float, expand: float = 1.25,
                     seed: int = 0) -> tuple[Atoms, float]:
    """Place n waters on a loose grid in a cell slightly LARGER (by `expand`)
    than the liquid-density box, so molecules start beyond vdW contact."""
    L_liq = box_length(n, rho)
    L = L_liq * expand
    rng = np.random.default_rng(seed)
    # smallest cubic grid that holds n points
    m = int(np.ceil(n ** (1.0 / 3.0)))
    spacing = L / m
    coords = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if len(coords) < n:
                    coords.append(((i + 0.5) * spacing,
                                   (j + 0.5) * spacing,
                                   (k + 0.5) * spacing))
    atoms = Atoms()
    for c in coords:
        atoms += single_water(c, rng)
    atoms.set_cell([L, L, L])
    atoms.set_pbc(True)
    return atoms, L


def write_periodic_coord(atoms: Atoms, L: float, out: Path) -> None:
    pos = atoms.get_positions() * ANG2BOHR
    sym = atoms.get_chemical_symbols()
    Lb = L * ANG2BOHR
    lines = ["$coord"]
    for (x, y, z), s in zip(pos, sym):
        lines.append(f"{x:20.12f}{y:20.12f}{z:20.12f}   {s.lower()}")
    lines += ["$periodic 3", "$cell",
              f" {Lb:.9f} {Lb:.9f} {Lb:.9f} 90. 90. 90.", "$end", ""]
    out.write_text("\n".join(lines))


def equilibrate(coord: Path, run_dir: Path, temp: float, ps: float,
                gfn: int = 1) -> Path:
    """Short unbiased NVT MD to condense the box. Returns final-frame xyz.

    NOTE: uses GFN1-xTB by default. GFN2-xTB CANNOT run under periodic
    boundary conditions in xtb 6.7.1 ("Multipoles not available with PBC")
    because GFN2's multipole electrostatics are not implemented for PBC.
    GFN1 uses monopole-only electrostatics and supports the periodic cell.
    """
    md_inp = run_dir / "equil.inp"
    md_inp.write_text(
        f"$md\n   temp={temp}\n   time={ps}\n   step=0.5\n   dump=50.0\n"
        f"   hmass=1\n   shake=0\n   nvt=true\n$end\n"
    )
    log = run_dir / "equil.log"
    print(f"  equilibrating {ps} ps NVT @ {temp} K (unbiased, GFN{gfn}) ...")
    with log.open("w") as fh:
        proc = subprocess.run(
            [str(XTB), coord.name, "--md", "--input", md_inp.name,
             "--gfn", str(gfn), "--chrg", "0", "--uhf", "0"],
            cwd=str(run_dir), stdout=fh, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"equilibration MD failed; see {log}")
    trj = run_dir / "xtb.trj"
    frames = read(str(trj), index=":", format="extxyz")
    final = frames[-1] if isinstance(frames, list) else frames
    out_xyz = run_dir / "equilibrated.xyz"
    write(str(out_xyz), final)
    print(f"  condensed box -> {out_xyz.name} ({len(final)} atoms)")
    return out_xyz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--equil-ps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-equilibrate", action="store_true",
                    help="only build the loose grid + coord, skip the MD")
    args = ap.parse_args()

    run_dir = OUT / f"n{args.n}"
    run_dir.mkdir(parents=True, exist_ok=True)

    atoms, L = build_loose_grid(args.n, args.density, seed=args.seed)
    L_liq = box_length(args.n, args.density)
    print(f"n={args.n}: liquid-density edge {L_liq:.2f} A, "
          f"loose starting edge {L:.2f} A ({len(atoms)} atoms)")
    write(str(run_dir / "loose_grid.xyz"), atoms)
    coord = run_dir / "start.coord"
    write_periodic_coord(atoms, L, coord)
    print(f"  wrote {coord}")

    if args.no_equilibrate:
        print("  (skipping equilibration as requested)")
        return 0

    if not XTB.exists():
        print(f"ERROR: xtb not found at {XTB}", file=sys.stderr)
        return 1
    equilibrate(coord, run_dir, args.temp, args.equil_ps)
    print("\nDone. equilibrated.xyz is the condensed box for metadynamics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
