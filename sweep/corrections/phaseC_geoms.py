#!/usr/bin/env python3
"""
phaseC_geoms.py — Phase C controlled dataset (targeted barrier-region calibration)
==================================================================================
Builds the geometry sets for the baseline-vs-barrier-enriched Delta experiment,
with a STRICT no-leakage split BY PARENT reaction-path geometry.

19 concerted-scan points (../diels_alder/cscan/f_*.xyz) are the "parents".
  - TEST parents (8, spanning all regions) are held out entirely: their rattles
    appear ONLY in the reactive test set, never in any training set.
  - TRAIN parents (11) supply all training rattles.
So a test geometry is never a perturbation of any training geometry.

Roles written (manifest.csv records role/parent/region/forming_CC per config):
  baseline       : 4 rattles × 11 train parents          (Model A base)
  barrier_extra  : 8 rattles × 4 BARRIER train parents   (targeted enrichment, +B)
  generic_extra  : 8 rattles × 4 NON-barrier train parents (generic control, +C)
  test           : 6 rattles × 8 held-out test parents   (reactive test set)
  water          : 30 droplet MD frames (shared training; equilibrium anchor)

Models: A = baseline+water ; B = baseline+barrier_extra+water ;
        C(control) = baseline+generic_extra+water. Enrichments add the SAME count
(32) so B vs C isolates data DISTRIBUTION (targeted vs generic).

Rattle = 0.10 Å RMS (matches Phase B), single global seeded rng so all rattles are
distinct. Output: phaseC/geoms/*.xyz + phaseC/manifest.csv.
"""
from __future__ import annotations
import os, glob, csv
from pathlib import Path
import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
DA = HERE.parent / "diels_alder"
LARGE = HERE.parent / "large"
OUT = HERE / "phaseC" / "geoms"
AMP = 0.10
RNG = np.random.default_rng(7)

# parent index -> role in the split (indices into sorted cscan f_00..f_18,
# forming C-C 3.30 -> 1.54). Test parents span every region; barrier = 1.7-2.4 Å.
TEST_PARENTS = [2, 4, 6, 8, 11, 13, 15, 18]          # 3.10,2.91,2.71,2.52,2.22,2.03,1.83,1.54
TRAIN_PARENTS = [0, 1, 3, 5, 7, 9, 10, 12, 14, 16, 17]
BARRIER_TRAIN = [10, 12, 14, 16]                      # 2.32,2.13,1.93,1.74 (barrier region)
GENERIC_TRAIN = [0, 5, 9, 17]                         # 3.30,2.81,2.42,1.64 (non-barrier)


def region(r):
    if r > 2.9:  return "reactant_basin"
    if r > 2.4:  return "pre_TS_rising"
    if r > 1.95: return "TS_region"
    if r > 1.7:  return "post_TS"
    return "product_basin"


def forming(at):
    p = at.get_positions()
    return 0.5 * (np.linalg.norm(p[0]-p[5]) + np.linalg.norm(p[3]-p[4]))


def rattle(base):
    at = base.copy()
    d = RNG.normal(0, 1, at.positions.shape)
    d *= AMP / np.sqrt((d**2).sum(1).mean())
    at.positions += d
    return at


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(OUT / "*.xyz")):
        os.remove(f)
    parents = [read(f) for f in sorted(glob.glob(str(DA / "cscan" / "f_*.xyz")))]
    rows = []
    k = 0

    def emit(base, role, parent_idx):
        nonlocal k
        at = rattle(base); r = forming(at)
        fn = f"g_{k:04d}.xyz"; write(str(OUT / fn), at)
        rows.append(dict(file=fn, role=role, parent=parent_idx,
                         forming_CC=round(r, 3), region=region(r))); k += 1

    for pi in TRAIN_PARENTS:                       # baseline
        for _ in range(4): emit(parents[pi], "baseline", pi)
    for pi in BARRIER_TRAIN:                        # targeted enrichment
        for _ in range(8): emit(parents[pi], "barrier_extra", pi)
    for pi in GENERIC_TRAIN:                        # generic control enrichment
        for _ in range(8): emit(parents[pi], "generic_extra", pi)
    for pi in TEST_PARENTS:                         # held-out reactive test
        for _ in range(6): emit(parents[pi], "test", pi)

    # water frames (shared training anchor)
    wk = 0
    for n in (20, 30, 50):
        frames = read(str(LARGE / "runs" / f"n{n}" / "low" / "xtb.trj"), index=":", format="xyz")
        idx = np.linspace(len(frames)//4, len(frames)-1, 10).round().astype(int)
        for j in idx:
            at = frames[int(j)]; fn = f"w_{wk:03d}.xyz"; write(str(OUT / fn), at)
            rows.append(dict(file=fn, role="water", parent=-1,
                             forming_CC=-1, region="water")); wk += 1

    with open(HERE / "phaseC" / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print("roles:", dict(Counter(r["role"] for r in rows)))
    print("test-set region counts:",
          dict(Counter(r["region"] for r in rows if r["role"] == "test")))
    print(f"train parents {TRAIN_PARENTS}\ntest parents  {TEST_PARENTS} (held out)")
    print(f"barrier-enrich parents {BARRIER_TRAIN}, generic parents {GENERIC_TRAIN}")
    print(f"wrote {len(rows)} configs -> {OUT}")


if __name__ == "__main__":
    main()
