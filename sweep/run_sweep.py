#!/usr/bin/env python3
"""
run_sweep.py
============
Drive a 2D sweep of gas-phase xTB metadynamics over:

    * cluster size   n = 1, 2, 3, 4  (H2O)_n
    * bias strength  kpush = 0.008 (low), 0.02 (med), 0.05 (high)

= 12 runs. Each run:
  1. Optimizes the cluster starting geometry once (GFN2-xTB, tight).
     (The optimized geometry is cached per size and reused across kpush.)
  2. Runs 20 ps NVT metadynamics at the chosen kpush.

Every run executes in its OWN directory under runs/, because xTB writes
fixed-name outputs (xtb.trj, xtbmd.log, scoord.*, mdrestart) that would
otherwise collide.

Usage:
    python run_sweep.py                 # run everything (skips finished runs)
    python run_sweep.py --force         # re-run even if outputs exist
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEOM = HERE / "geometries"
RUNS = HERE / "runs"
XTB = (HERE.parent / "tools" / "xtb-6.7.1" / "bin" / "xtb.exe").resolve()

SIZES = [1, 2, 3, 4]
KPUSH = {"low": 0.008, "med": 0.02, "high": 0.05}

SIM_TIME_PS = 20.0
TEMP_K = 300
STEP_FS = 0.5          # 0.5 fs: safer with free (unconstrained) O-H bonds
DUMP_FS = 100.0
SAVE = 10
ALP = 1.2


def metadyn_input(kpush: float) -> str:
    return f"""$md
   temp={TEMP_K}
   time={SIM_TIME_PS}
   step={STEP_FS}
   dump={DUMP_FS}
   hmass=1
   shake=0
   nvt=true
$end
$metadyn
   save={SAVE}
   kpush={kpush}
   alp={ALP}
$end
"""


def run_xtb(args: list[str], cwd: Path, logfile: Path) -> None:
    """Run xtb in cwd, capturing stdout to logfile. Raises on failure."""
    with logfile.open("w") as fh:
        proc = subprocess.run(
            [str(XTB), *args],
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"xtb failed ({proc.returncode}) in {cwd}; see {logfile}")


def optimize_cluster(n: int, force: bool) -> Path:
    """Optimize (H2O)_n once; return path to the optimized xyz. Cached."""
    opt_dir = RUNS / f"n{n}" / "opt"
    opt_dir.mkdir(parents=True, exist_ok=True)
    opt_xyz = opt_dir / "xtbopt.xyz"
    if opt_xyz.exists() and not force:
        print(f"  [n={n}] optimized geometry cached, reusing")
        return opt_xyz
    start = opt_dir / "start.xyz"
    shutil.copy(GEOM / f"water_{n}.xyz", start)
    print(f"  [n={n}] optimizing starting geometry ...")
    run_xtb(["start.xyz", "--opt", "tight", "--gfn", "2",
             "--chrg", "0", "--uhf", "0"],
            cwd=opt_dir, logfile=opt_dir / "optimization.log")
    if not opt_xyz.exists():
        raise RuntimeError(f"optimization produced no xtbopt.xyz in {opt_dir}")
    return opt_xyz


def run_metadyn(n: int, label: str, kpush: float, opt_xyz: Path,
                force: bool) -> Path:
    run_dir = RUNS / f"n{n}" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    ok_marker = run_dir / "xtbmdok"
    trj = run_dir / "xtb.trj"
    if trj.exists() and not force:
        print(f"  [n={n} {label} kpush={kpush}] already done, skipping")
        return run_dir
    # fresh inputs
    shutil.copy(opt_xyz, run_dir / "input.xyz")
    (run_dir / "metadyn.inp").write_text(metadyn_input(kpush))
    print(f"  [n={n} {label} kpush={kpush}] running {SIM_TIME_PS:.0f} ps MTD ...")
    run_xtb(["input.xyz", "--md", "--input", "metadyn.inp", "--gfn", "2",
             "--chrg", "0", "--uhf", "0"],
            cwd=run_dir, logfile=run_dir / "output.log")
    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run finished runs")
    ap.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    args = ap.parse_args()

    if not XTB.exists():
        print(f"ERROR: xtb not found at {XTB}", file=sys.stderr)
        return 1

    print(f"xTB: {XTB}")
    print(f"Sweep: sizes={args.sizes}  kpush={KPUSH}  time={SIM_TIME_PS} ps\n")

    for n in args.sizes:
        opt_xyz = optimize_cluster(n, args.force)
        for label, kpush in KPUSH.items():
            run_metadyn(n, label, kpush, opt_xyz, args.force)
    print("\nAll runs complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
