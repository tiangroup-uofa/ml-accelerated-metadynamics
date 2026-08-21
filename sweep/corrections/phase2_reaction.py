#!/usr/bin/env python3
"""
phase2_reaction.py — per-iteration relaxed concerted reaction scan.

Identical protocol to phaseC_reaction.py (same cscan frames, same FixBondLengths
+ FIRE relaxation, float64) so barriers are comparable to the verified Phase C
references: raw MACE 36.0 | Delta-A 14.1 | Delta-B 9.9 | Delta-C 15.0 | xTB 6.7.

Note this is a *constrained scan* barrier, a different estimator from the Sella
saddle barrier reported by the loop; both are tracked separately.

Prints one JSON line.
    /usr/bin/python3 phase2_reaction.py --delta phase2/iter00/delta.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path

import numpy as np
from ase.constraints import FixBondLengths
from ase.io import read
from ase.optimize import FIRE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diels_alder" / "neb"))

DA = HERE.parent / "diels_alder"
FORMING = [[0, 5], [3, 4]]
EV2KCAL = 23.060548


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True)
    ap.add_argument("--save-max", default=None,
                    help="write the scan-maximum geometry here (Sella seed)")
    a = ap.parse_args()

    import torch

    torch.set_num_threads(4)
    from mace.calculators import mace_off

    from corrections import CorrectedCalculator, Delta
    from pairwise_delta import PairwiseDelta

    base = mace_off(model="small", device="cpu", default_dtype="float64")
    delta = PairwiseDelta.from_dict(json.loads(Path(a.delta).read_text()))

    coord, E, relaxed = [], [], []
    for f in sorted(glob.glob(str(DA / "cscan" / "f_*.xyz"))):
        at = read(f)
        at.calc = CorrectedCalculator(base, Delta(delta))
        at.set_constraint(FixBondLengths(FORMING))
        FIRE(at, logfile=None).run(fmax=0.03, steps=300)
        at.set_constraint()
        p = at.get_positions()
        coord.append(0.5 * (np.linalg.norm(p[0] - p[5]) + np.linalg.norm(p[3] - p[4])))
        E.append(float(at.get_potential_energy()))
        relaxed.append(at)

    coord = np.array(coord)
    rel = (np.array(E) - E[0]) * EV2KCAL
    i = int(np.argmax(rel))
    if a.save_max:
        from ase.io import write as _w
        _w(a.save_max, relaxed[i])
    print(json.dumps({
        "barrier_kcal": float(rel[i]),
        "ts_forming": float(coord[i]),
        "reaction_energy_kcal": float(rel[-1]),
        "scan_max_index": i,
        "coord": coord.tolist(),
        "rel_kcal": rel.tolist(),
    }))


if __name__ == "__main__":
    main()
