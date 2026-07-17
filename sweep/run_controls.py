#!/usr/bin/env python3
"""
run_controls.py
===============
Controls + replicas experiment to strengthen the phase-2 conclusions.

Grid (same 8-water system throughout, GFN1 so gas and PBC are one theory level):

    boundary : {gas, pbc}
    bias     : {0.0 (unbiased control), 0.008 (metadynamics)}
    seed     : {0, 1, 2}   -> 3 independent replicas

= 4 cells x 3 seeds = 12 runs, each 20 ps NVT @ 300 K, step 0.5 fs.

Replicas: xtb has no MD velocity-seed flag, so independent replicas are made
by re-preparing the box with different --seed values in prepare_box.py (which
randomizes molecular orientations), producing genuinely independent starting
configurations that are then equilibrated and run.

The unbiased (kpush=0) control separates instability caused by temperature /
initialization from instability specifically induced by the metadynamics bias.

Each run lives in controls/<boundary>_k<bias>_s<seed>/.

Usage:
    python run_controls.py                 # prepare boxes + run everything
    python run_controls.py --prep-only     # only build/equilibrate the boxes
    python run_controls.py --cells gas_0.0 pbc_0.008   # subset
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ase.io import read, write

HERE = Path(__file__).resolve().parent
OUT = HERE / "controls"
BOXES = HERE / "controls_boxes"
XTB = (HERE.parent / "tools" / "xtb-6.7.1" / "bin" / "xtb.exe").resolve()
PREP = HERE / "prepare_box.py"
ANG2BOHR = 1.8897259886

N_WATERS = 8
# Lower density -> a LARGER shared box (~13 A). This single geometry is used
# for BOTH the gas and PBC arms so they differ only in boundary condition.
# A dense (0.6-1.0 g/cm^3) box is stable under PBC but ROCKETS APART when run
# gas-phase (the compressed configuration is unphysical without periodicity),
# so we deliberately start lower-density. PBC still confines; gas can expand.
DENSITY = 0.30         # ~13 A box for 8 waters
TEMP = 300.0
TIME_PS = 20.0
EQUIL_PS = 2.0         # gentle: settle contacts without over-compressing
SEEDS = [0, 1, 2]
BIASES = [0.0, 0.008]
BOUNDARIES = ["gas", "pbc"]


def md_inp(kpush: float) -> str:
    md = (f"$md\n   temp={TEMP}\n   time={TIME_PS}\n   step=0.5\n"
          f"   dump=100.0\n   hmass=1\n   shake=0\n   nvt=true\n$end\n")
    if kpush > 0:
        md += f"$metadyn\n   save=10\n   kpush={kpush}\n   alp=1.2\n$end\n"
    return md


def run_xtb(args, cwd, log):
    with Path(log).open("w") as fh:
        p = subprocess.run([str(XTB), *args], cwd=str(cwd),
                           stdout=fh, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def prepare_box(seed: int) -> tuple[Path, float]:
    """Build a clean shared starting geometry for one replica (seed):

      1. loose random grid of 8 waters (prepare_box.py --no-equilibrate),
         which has pristine 0.96 A O-H bonds;
      2. GFN1 geometry OPTIMIZATION -> a physically valid, connected
         8-water cluster (all O with 2 H, O-O ~2.6-5 A).

    We optimize rather than MD-equilibrate because a raw dense/low-density
    box run gas-phase is numerically unstable (atoms rocket apart / a water
    dissociates during equilibration). The optimized cluster is stable for
    BOTH the gas and PBC arms. Returns (start_xyz, box_edge_Angstrom).
    The box edge is chosen large enough to enclose the optimized cluster
    with padding, so PBC confines without crushing.
    """
    bdir = BOXES / f"seed{seed}"
    start = bdir / "start.xyz"
    opt = bdir / "xtbopt.xyz"
    if not opt.exists():
        bdir.mkdir(parents=True, exist_ok=True)
        # 1. loose grid only (no MD)
        rc = subprocess.run(
            [sys.executable, str(PREP), "--n", str(N_WATERS),
             "--density", str(DENSITY), "--seed", str(seed),
             "--no-equilibrate"], cwd=str(HERE)).returncode
        if rc != 0:
            raise RuntimeError(f"loose-grid prep failed for seed {seed}")
        import shutil
        shutil.copy(HERE / "boxes" / f"n{N_WATERS}" / "loose_grid.xyz", start)
        # 2. GFN1 optimize
        run_xtb(["start.xyz", "--opt", "loose", "--gfn", "1",
                 "--chrg", "0", "--uhf", "0"], bdir, bdir / "opt.log")
        if not opt.exists():
            raise RuntimeError(f"GFN1 opt produced no xtbopt.xyz (seed {seed})")

    # choose a cubic cell that encloses the cluster + ~4 A padding
    atoms = read(str(opt))
    span = (atoms.get_positions().max(axis=0)
            - atoms.get_positions().min(axis=0)).max()
    edge = float(span + 4.0)
    return opt, edge


def make_periodic_coord(equil_xyz: Path, edge_A: float, dst: Path):
    atoms = read(str(equil_xyz))
    # centre the cluster in the box so it is not split across the boundary
    atoms.positions -= atoms.get_center_of_mass()
    atoms.positions += edge_A / 2.0
    pos = atoms.get_positions() * ANG2BOHR
    sym = atoms.get_chemical_symbols()
    Lb = edge_A * ANG2BOHR
    lines = ["$coord"]
    for (x, y, z), s in zip(pos, sym):
        lines.append(f"{x:20.12f}{y:20.12f}{z:20.12f}   {s.lower()}")
    lines += ["$periodic 3", "$cell",
              f" {Lb:.9f} {Lb:.9f} {Lb:.9f} 90. 90. 90.", "$end", ""]
    dst.write_text("\n".join(lines))


def run_cell(boundary: str, bias: float, seed: int,
             equil_xyz: Path, edge_A: float) -> str:
    tag = f"{boundary}_k{bias}_s{seed}"
    rdir = OUT / tag
    rdir.mkdir(parents=True, exist_ok=True)
    if (rdir / "xtb.trj").exists():
        return f"{tag}: exists, skip"
    (rdir / "md.inp").write_text(md_inp(bias))
    if boundary == "pbc":
        make_periodic_coord(equil_xyz, edge_A, rdir / "input.coord")
        inp = "input.coord"
    else:
        write(str(rdir / "input.xyz"), read(str(equil_xyz)))
        inp = "input.xyz"
    rc = run_xtb([inp, "--md", "--input", "md.inp", "--gfn", "1",
                  "--chrg", "0", "--uhf", "0"], rdir, rdir / "output.log")
    ok = (rdir / "xtb.trj").exists()
    return f"{tag}: {'ok' if (rc == 0 and ok) else 'FAILED rc=' + str(rc)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--one-per-cell", action="store_true",
                    help="smoke test: only seed 0 of each cell")
    args = ap.parse_args()

    seeds = [0] if args.one_per_cell else args.seeds

    # prepare (and cache) one equilibrated box per seed
    boxes = {}
    for s in seeds:
        print(f"[prep] equilibrating box seed {s} ...")
        equil, edge = prepare_box(s)
        boxes[s] = (equil, edge)
        print(f"  seed {s}: box edge {edge:.2f} A")
    if args.prep_only:
        return 0

    results = []
    for s in seeds:
        equil, edge = boxes[s]
        for boundary in BOUNDARIES:
            for bias in BIASES:
                print(f"[run] {boundary} kpush={bias} seed={s} ...")
                results.append(run_cell(boundary, bias, s, equil, edge))
                print("   " + results[-1])
    print("\n=== summary ===")
    for r in results:
        print(" ", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
