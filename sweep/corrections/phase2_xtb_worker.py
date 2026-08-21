#!/usr/bin/env python3
"""
phase2_xtb_worker.py — GFN2-xTB energies + forces for a list of .xyz files.

Runs inside the `xtbenv` micromamba environment (xtb-python + ase only); called
as a subprocess by phase2_lib.xtb_eval so xtb-python and MACE/torch never share
a process (both link OpenMP).

    micromamba run -n xtbenv python phase2_xtb_worker.py in.json out.json

Settings match the A1 stability gate: GFN2-xTB, electronic_temperature=1000 K,
max_iterations=500, accuracy=1.0.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from ase.io import read

os.environ.setdefault("OMP_NUM_THREADS", "4")


def main():
    in_json, out_json = sys.argv[1], sys.argv[2]
    paths = json.loads(open(in_json).read())

    from xtb.ase.calculator import XTB

    energies, forces, errors = [], [], []
    for p in paths:
        try:
            at = read(p)
            at.calc = XTB(method="GFN2-xTB", electronic_temperature=1000.0,
                          max_iterations=500, accuracy=1.0)
            energies.append(float(at.get_potential_energy()))
            forces.append(at.get_forces().tolist())
        except Exception as e:
            errors.append({"file": p, "error": repr(e)})
            energies.append(None)
            forces.append(None)

    with open(out_json, "w") as fh:
        json.dump({"energies": energies, "forces": forces, "errors": errors}, fh)
    print(f"[xtb] {len(paths) - len(errors)}/{len(paths)} OK")


if __name__ == "__main__":
    main()
