#!/usr/bin/env python3
"""
run_temp_sweep.py
=================
Temperature axis for the metadynamics study: at a fixed bias (kpush) and
fixed cluster size, vary the NVT thermostat temperature and see how it
affects cluster stability. Complements run_sweep.py (which varied kpush).

The thermostat temperature sets the kinetic energy available to cross
barriers, so higher T should make the cluster break up sooner/more, on top
of whatever the bias is doing. This isolates the temperature effect from the
bias effect.

Default: (H2O)3 at kpush=0.02 (med), T in {200, 250, 298, 350} K, 20 ps.
Gas-phase GFN2 (matches the main sweep). Reuses the optimized geometry
produced by run_sweep.py if present.

Usage:
    python run_temp_sweep.py                 # defaults
    python run_temp_sweep.py --n 3 --kpush 0.02
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TRUNS = HERE / "temp_runs"
GEOM = HERE / "geometries"
XTB = (HERE.parent / "tools" / "xtb-6.7.1" / "bin" / "xtb.exe").resolve()

TEMPS = [200, 250, 298, 350]
DEF_N = 3
DEF_KPUSH = 0.02
TIME_PS = 20.0


def metadyn_inp(temp, kpush):
    return (f"$md\n   temp={temp}\n   time={TIME_PS}\n   step=0.5\n"
            f"   dump=100.0\n   hmass=1\n   shake=0\n   nvt=true\n$end\n"
            f"$metadyn\n   save=10\n   kpush={kpush}\n   alp=1.2\n$end\n")


def run_xtb(args, cwd, log):
    with Path(log).open("w") as fh:
        p = subprocess.run([str(XTB), *args], cwd=str(cwd),
                           stdout=fh, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"xtb failed rc={p.returncode}; see {log}")


def get_opt_geom(n: int, work: Path) -> Path:
    """Reuse run_sweep's optimized geometry if available, else optimize."""
    cached = RUNS / f"n{n}" / "opt" / "xtbopt.xyz"
    if cached.exists():
        dst = work / "input.xyz"
        shutil.copy(cached, dst)
        return dst
    # optimize fresh
    shutil.copy(GEOM / f"water_{n}.xyz", work / "start.xyz")
    run_xtb(["start.xyz", "--opt", "tight", "--gfn", "2",
             "--chrg", "0", "--uhf", "0"], work, work / "opt.log")
    shutil.copy(work / "xtbopt.xyz", work / "input.xyz")
    return work / "input.xyz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEF_N)
    ap.add_argument("--kpush", type=float, default=DEF_KPUSH)
    ap.add_argument("--temps", type=int, nargs="*", default=TEMPS)
    args = ap.parse_args()

    print(f"Temperature sweep: (H2O){args.n}  kpush={args.kpush}  "
          f"T={args.temps} K\n")
    for T in args.temps:
        rdir = TRUNS / f"n{args.n}_k{args.kpush}_T{T}"
        rdir.mkdir(parents=True, exist_ok=True)
        if (rdir / "xtb.trj").exists():
            print(f"  T={T} K already done, skipping")
            continue
        geom = get_opt_geom(args.n, rdir)
        (rdir / "md.inp").write_text(metadyn_inp(T, args.kpush))
        print(f"  T={T} K running ...")
        run_xtb([geom.name, "--md", "--input", "md.inp", "--gfn", "2",
                 "--chrg", "0", "--uhf", "0"], rdir, rdir / "output.log")
    print("\nTemperature sweep complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
