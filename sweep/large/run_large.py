#!/usr/bin/env python3
"""
run_large.py
============
Drive gas-phase (vacuum) xTB metadynamics on the LARGER water clusters built
by build_large_clusters.py, implementing the July-10 to-do list:

  step 1/3 : low kpush, short NVT, sizes n = 20/30/50 -- does a bigger free
             cluster hold together as a DROPLET or keep evaporating?
  step 2   : fix the size, RAISE kpush -- do bonds break / proton-transfer
             events appear as the bias gets stronger?

All runs are GFN2-xTB, NVT, step 0.5 fs (the timestep found stable earlier),
vacuum (no cell / no $periodic -- so GFN2 is available, unlike the PBC runs
which were forced onto GFN1). Each run gets its own directory because xTB
writes fixed-name outputs (xtb.trj, mdrestart, scoord.*).

xTB binary: set env var XTB_BIN, else the default below (the micromamba
conda-forge xtb 6.7.1 installed for this machine). On an HPC node just
`export XTB_BIN=$(which xtb)` after `module load xtb`.

Usage:
    python run_large.py --step 1 --sizes 20 30 50 --time-ps 3
    python run_large.py --step 2 --sizes 30 --time-ps 3
    python run_large.py --step 1 --sizes 20 --time-ps 0.2 --tag timing  # quick probe
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEOM = HERE / "geometries"
RUNS = HERE / "runs"

# xtb binary: $XTB_BIN, else whatever `xtb` is on PATH (conda env / module load).
XTB = Path(os.environ.get("XTB_BIN") or shutil.which("xtb") or "xtb")

KPUSH = {"low": 0.008, "med": 0.02, "high": 0.05}
TEMP_K = 300
STEP_FS = 0.5
DUMP_FS = 100.0
SAVE = 10
ALP = 1.2


def metadyn_input(kpush: float, time_ps: float) -> str:
    return f"""$md
   temp={TEMP_K}
   time={time_ps}
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


def run_xtb(args: list[str], cwd: Path, logfile: Path) -> float:
    """Run xtb in cwd, stdout -> logfile. Returns wall-clock seconds.
    Raises on non-zero exit."""
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
    env.setdefault("OMP_STACKSIZE", "1G")
    t0 = time.time()
    with logfile.open("w") as fh:
        proc = subprocess.run([str(XTB), *args], cwd=str(cwd),
                              stdout=fh, stderr=subprocess.STDOUT,
                              text=True, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"xtb failed ({proc.returncode}) in {cwd}; see {logfile}")
    return dt


def optimize_cluster(n: int, force: bool) -> Path:
    """GFN2 optimize (H2O)_n once; cache the optimized xyz."""
    opt_dir = RUNS / f"n{n}" / "opt"
    opt_dir.mkdir(parents=True, exist_ok=True)
    opt_xyz = opt_dir / "xtbopt.xyz"
    if opt_xyz.exists() and not force:
        print(f"  [n={n}] optimized geometry cached")
        return opt_xyz
    shutil.copy(GEOM / f"water_{n}.xyz", opt_dir / "start.xyz")
    print(f"  [n={n}] optimizing (GFN2) ...", flush=True)
    dt = run_xtb(["start.xyz", "--opt", "--gfn", "2", "--chrg", "0", "--uhf", "0"],
                 cwd=opt_dir, logfile=opt_dir / "optimization.log")
    if not opt_xyz.exists():
        raise RuntimeError(f"no xtbopt.xyz in {opt_dir}")
    print(f"  [n={n}] optimized in {dt:.0f} s")
    return opt_xyz


def run_metadyn(n: int, label: str, kpush: float, time_ps: float,
                opt_xyz: Path, force: bool, tag: str | None) -> Path:
    dirname = tag if tag else label
    run_dir = RUNS / f"n{n}" / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    trj = run_dir / "xtb.trj"
    if trj.exists() and not force:
        print(f"  [n={n} {dirname}] already done, skipping")
        return run_dir
    shutil.copy(opt_xyz, run_dir / "input.xyz")
    (run_dir / "metadyn.inp").write_text(metadyn_input(kpush, time_ps))
    nsteps = int(time_ps * 1000 / STEP_FS)
    print(f"  [n={n} {dirname} kpush={kpush}] {time_ps} ps "
          f"({nsteps} steps) GFN2 vacuum ...", flush=True)
    dt = run_xtb(["input.xyz", "--md", "--input", "metadyn.inp",
                  "--gfn", "2", "--chrg", "0", "--uhf", "0"],
                 cwd=run_dir, logfile=run_dir / "output.log")
    print(f"  [n={n} {dirname}] done in {dt:.0f} s "
          f"({dt / nsteps * 1000:.1f} ms/step)")
    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=[1, 2], required=True,
                    help="1/3 = low-bias size ladder; 2 = raise kpush")
    ap.add_argument("--sizes", type=int, nargs="*", default=[20, 30, 50])
    ap.add_argument("--time-ps", type=float, default=3.0)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="step-2 bias levels (default med high)")
    ap.add_argument("--tag", default=None,
                    help="override run-dir name (for a quick timing probe)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not XTB.exists():
        print(f"ERROR: xtb not found at {XTB}\n"
              f"Set XTB_BIN to your xtb binary.", file=sys.stderr)
        return 1
    print(f"xTB: {XTB}")

    labels = args.labels or (["low"] if args.step == 1 else ["med", "high"])
    print(f"step {args.step}: sizes={args.sizes} labels={labels} "
          f"time={args.time_ps} ps\n")

    for n in args.sizes:
        opt_xyz = optimize_cluster(n, args.force)
        for label in labels:
            run_metadyn(n, label, KPUSH[label], args.time_ps,
                        opt_xyz, args.force, args.tag)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
