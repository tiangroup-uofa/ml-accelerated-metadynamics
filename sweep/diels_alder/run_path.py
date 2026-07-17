#!/usr/bin/env python3
"""
run_path.py
===========
Optimize the Diels-Alder endpoints and run the xtb RMSD-biased reaction-path
search (https://xtb-docs.readthedocs.io/en/latest/path.html).

Steps:
  1. GFN2 --opt the reactant and product guesses -> start.xyz, end.xyz
     (same atom ordering is preserved from build_diels_alder.py, which the
     --path interpolation requires).
  2. xtb start.xyz --path end.xyz --input path.inp
     path.inp uses the tutorial settings (kpush=0.003, kpull=-0.015,
     ppull=0.05, alp=1.2, npoint=25, anopt=10).
  3. Parse the forward barrier, reaction energy and TS estimate from the log.

This is the SAME RMSD-bias machinery as the metadynamics runs, but aimed at a
known bond-forming reaction: it directly answers "can we drive an actual
reaction (two new C-C sigma bonds) in this xtb setup?".

xtb binary via $XTB_BIN (see ../large/run_large.py for the default).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# xtb binary: $XTB_BIN, else whatever `xtb` is on PATH (conda env / module load).
XTB = Path(os.environ.get("XTB_BIN") or shutil.which("xtb") or "xtb")

PATH_INP = """$path
   nrun=1
   npoint=25
   anopt=10
   kpush=0.003
   kpull=-0.015
   ppull=0.05
   alp=1.2
$end
"""


def run_xtb(args, cwd, log):
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")   # tiny system; leave cores for MD
    with open(log, "w") as fh:
        p = subprocess.run([str(XTB), *args], cwd=str(cwd),
                           stdout=fh, stderr=subprocess.STDOUT, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"xtb failed ({p.returncode}); see {log}")


def opt(raw: str, out: str):
    d = HERE / "opt_tmp"
    d.mkdir(exist_ok=True)
    shutil.copy(HERE / raw, d / raw)
    run_xtb([raw, "--opt", "--gfn", "2", "--chrg", "0", "--uhf", "0"],
            d, d / f"opt_{raw}.log")
    shutil.copy(d / "xtbopt.xyz", HERE / out)
    print(f"  optimized {raw} -> {out}")


def parse_path_log(log: Path):
    txt = log.read_text()
    def grab(pat):
        m = re.search(pat, txt)
        return m.group(1) if m else None
    barrier = grab(r"forward barrier\s*\(kcal\)\s*:\s*([-\d.]+)")
    if barrier is None:
        barrier = grab(r"barrier\s*\(kcal\)\s*[:=]?\s*([-\d.]+)")
    dE = grab(r"reaction energy\s*\(kcal\)\s*:\s*([-\d.]+)")
    return barrier, dE


def main() -> int:
    if not XTB.exists():
        print(f"ERROR: xtb not found at {XTB}; set XTB_BIN", file=sys.stderr)
        return 1
    if not (HERE / "reactant_raw.xyz").exists():
        print("Run build_diels_alder.py first.", file=sys.stderr)
        return 1

    print("optimizing endpoints (GFN2) ...")
    opt("reactant_raw.xyz", "start.xyz")
    opt("product_raw.xyz", "end.xyz")

    (HERE / "path.inp").write_text(PATH_INP)
    print("running reaction-path search (xtb --path) ...")
    run_xtb(["start.xyz", "--path", "end.xyz", "--input", "path.inp",
             "--gfn", "2", "--chrg", "0", "--uhf", "0"],
            HERE, HERE / "path.log")

    barrier, dE = parse_path_log(HERE / "path.log")
    print("\n=== Diels-Alder path result ===")
    print(f"  forward barrier : {barrier} kcal/mol")
    print(f"  reaction energy : {dE} kcal/mol")
    ts = HERE / "xtbpath_ts.xyz"
    print(f"  TS structure    : {'xtbpath_ts.xyz' if ts.exists() else '(not written)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
