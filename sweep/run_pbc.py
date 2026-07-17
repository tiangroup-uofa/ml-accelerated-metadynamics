#!/usr/bin/env python3
"""
run_pbc.py
==========
Run metadynamics on the equilibrated PERIODIC water box (from prepare_box.py)
and, for contrast, on the SAME cluster in the gas phase -- both at GFN1-xTB so
the comparison is at a single level of theory.

Why GFN1 for both: GFN2-xTB cannot run under periodic boundary conditions in
xtb 6.7.1 ("Multipoles not available with PBC"). To compare gas vs periodic
fairly we therefore use GFN1 in BOTH cases.

The point of the comparison: under PBC the molecules cannot evaporate to
infinity (the box + minimum-image convention keeps them condensed), so the
metadynamics explores structural rearrangements instead of just flying apart --
directly addressing the gas-phase evaporation problem found earlier.

Usage:
    python run_pbc.py --n 8 --kpush 0.02 --time 20 --temp 300
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ase.io import read, write

HERE = Path(__file__).resolve().parent
BOXES = HERE / "boxes"
OUT = HERE / "pbc_runs"
XTB = (HERE.parent / "tools" / "xtb-6.7.1" / "bin" / "xtb.exe").resolve()
ANG2BOHR = 1.8897259886


def metadyn_inp(temp, time_ps, kpush, alp=1.2, save=10):
    return (f"$md\n   temp={temp}\n   time={time_ps}\n   step=0.5\n"
            f"   dump=100.0\n   hmass=1\n   shake=0\n   nvt=true\n$end\n"
            f"$metadyn\n   save={save}\n   kpush={kpush}\n   alp={alp}\n$end\n")


def run_xtb(args, cwd, log):
    with Path(log).open("w") as fh:
        p = subprocess.run([str(XTB), *args], cwd=str(cwd),
                           stdout=fh, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"xtb failed rc={p.returncode}; see {log}")


def read_cell_line(src: Path) -> str:
    """Extract the '$cell' data line (in Bohr) from a Turbomole coord."""
    lines = src.read_text().splitlines()
    for i, l in enumerate(lines):
        if l.strip().startswith("$cell"):
            return lines[i + 1].strip()
    raise ValueError(f"no $cell block in {src}")


def periodic_coord_from_box(n: int, dst: Path):
    """Build a periodic coord from the EQUILIBRATED geometry, reusing the
    exact $cell (Bohr) that start.coord / the equilibration MD used."""
    src = BOXES / f"n{n}" / "start.coord"
    cell_line = read_cell_line(src)
    equil = read(str(BOXES / f"n{n}" / "equilibrated.xyz"))
    pos = equil.get_positions() * ANG2BOHR
    sym = equil.get_chemical_symbols()
    lines = ["$coord"]
    for (x, y, z), s in zip(pos, sym):
        lines.append(f"{x:20.12f}{y:20.12f}{z:20.12f}   {s.lower()}")
    lines += ["$periodic 3", "$cell", f" {cell_line}", "$end", ""]
    dst.write_text("\n".join(lines))
    # return the cubic edge in Angstrom (for downstream MIC analysis)
    return float(cell_line.split()[0]) / ANG2BOHR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--kpush", type=float, default=0.02)
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--gas", action="store_true",
                    help="also run the same molecules gas-phase (no cell)")
    args = ap.parse_args()

    if not (BOXES / f"n{args.n}" / "equilibrated.xyz").exists():
        print(f"ERROR: run prepare_box.py --n {args.n} first.", file=sys.stderr)
        return 1

    # --- periodic run ---
    pdir = OUT / f"n{args.n}_pbc"
    pdir.mkdir(parents=True, exist_ok=True)
    coord = pdir / "input.coord"
    edge_A = periodic_coord_from_box(args.n, coord)
    (pdir / "md.inp").write_text(metadyn_inp(args.temp, args.time, args.kpush))
    print(f"[PBC] n={args.n} kpush={args.kpush} {args.time} ps GFN1 "
          f"box={edge_A:.2f} A")
    print(f"      -> analyze with: cv_analysis.py --cell {edge_A:.3f}")
    run_xtb(["input.coord", "--md", "--input", "md.inp", "--gfn", "1",
             "--chrg", "0", "--uhf", "0"], pdir, pdir / "output.log")
    print(f"  done -> {pdir/'xtb.trj'}")

    # --- optional gas-phase run of the same geometry (GFN1, no cell) ---
    if args.gas:
        gdir = OUT / f"n{args.n}_gas"
        gdir.mkdir(parents=True, exist_ok=True)
        write(str(gdir / "input.xyz"),
              read(str(BOXES / f"n{args.n}" / "equilibrated.xyz")))
        (gdir / "md.inp").write_text(metadyn_inp(args.temp, args.time, args.kpush))
        print(f"[GAS] n={args.n} kpush={args.kpush} {args.time} ps GFN1 ...")
        run_xtb(["input.xyz", "--md", "--input", "md.inp", "--gfn", "1",
                 "--chrg", "0", "--uhf", "0"], gdir, gdir / "output.log")
        print(f"  done -> {gdir/'xtb.trj'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
