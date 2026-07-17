#!/usr/bin/env python3
"""
calculate_cvs.py
================
Compute structural collective variables (CVs) for each frame of the
metadynamics snapshot trajectory produced by ``convert_scoord.py``.

Computed per frame
-------------------
* aligned Cartesian RMSD from the first frame (Kabsch superposition)
* distance_1   = |r(A1) - r(A2)|
* distance_2   = |r(B1) - r(B2)|
* distance_difference = distance_1 - distance_2
* angle (deg)  defined by three atoms
* dihedral (deg) defined by four atoms (skipped if too few atoms)
* radius of gyration (Angstrom)
* max pairwise atomic distance (Angstrom)        [fragmentation diagnostic]
* estimated number of connected components        [fragmentation diagnostic]

Approximate time per snapshot is derived from the NUMERIC SUFFIX of the
source file (scoord.1, scoord.2, ..., scoord.10), not from list order, so
a missing snapshot does not shift the time axis:

    approx_time_ps = t0_ps + (snapshot_number - first_snapshot_number) * dt_ps

Timing remains APPROXIMATE until verified against the actual xTB output.

Output
------
    collective_variables.csv

IMPORTANT -- ATOM INDEXING
--------------------------
ASE uses ZERO-BASED atom indices. The atom that a visualizer or a
Turbomole file lists as "atom 1" is index 0 here. Set the indices in
the CONFIG block below to match YOUR system; the defaults are
placeholders only and are NOT meaningful for an arbitrary molecule.

Usage
-----
    python calculate_cvs.py \
        --xyz ../analysis/scoord_1ps.xyz \
        --out ../analysis/collective_variables.csv \
        --dt-ps 1.0 --t0-ps 1.0 --connectivity-scale 1.3
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import covalent_radii
from ase.io import read

# =====================================================================
# CONFIG: zero-based atom indices.  EDIT THESE FOR YOUR SYSTEM.
# ---------------------------------------------------------------------
# distance_1: bond / contact you expect to break or form
DIST1_ATOMS = (0, 1)
# distance_2: a second bond / contact (e.g. the partner in a concerted
#             bond-break/bond-form, or the second leg of a proton transfer)
DIST2_ATOMS = (1, 2)
# angle: three atoms i-j-k, vertex is the middle index (j)
ANGLE_ATOMS = (0, 1, 2)
# dihedral: four atoms i-j-k-l
DIHEDRAL_ATOMS = (0, 1, 2, 3)
# =====================================================================


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Minimum RMSD between point sets P and Q after optimal rotation.

    Both sets are first centred on their centroids, then the optimal
    rotation is found via SVD (the Kabsch algorithm). Returns RMSD in
    the same length units as the inputs (Angstrom here).
    """
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    # Covariance matrix and its SVD
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    # Correct for a possible reflection so we get a proper rotation
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_rot = Pc @ R.T
    diff = P_rot - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def radius_of_gyration(atoms) -> float:
    """Mass-weighted radius of gyration in Angstrom."""
    masses = atoms.get_masses()
    pos = atoms.get_positions()
    com = np.average(pos, axis=0, weights=masses)
    d2 = ((pos - com) ** 2).sum(axis=1)
    return float(np.sqrt(np.average(d2, weights=masses)))


def max_pair_distance(atoms) -> float:
    """Largest distance between any two atoms (Angstrom)."""
    pos = atoms.get_positions()
    mx = 0.0
    for i in range(len(pos)):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        if d.size:
            mx = max(mx, float(d.max()))
    return mx


def estimate_components(atoms, scale: float) -> int:
    """Estimate the number of connected components (HEURISTIC).

    Two atoms i, j are considered bonded when their separation is below
    scale * (r_cov[i] + r_cov[j]) using ASE covalent radii. The number of
    connected components is then found by union-find. This is only a rough
    fragmentation indicator: real reactions may legitimately change
    connectivity, weakly bound complexes are naturally multi-component,
    and the covalent cutoff is approximate.
    """
    pos = atoms.get_positions()
    Z = atoms.get_atomic_numbers()
    radii = covalent_radii[Z]
    n = len(atoms)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) <= scale * (radii[i] + radii[j]):
                union(i, j)
    return len({find(i) for i in range(n)})


def snapshot_number(source_file: str):
    """Extract the integer suffix from a 'scoord.<N>' style filename.

    Returns None if there is no numeric suffix.
    """
    m = re.search(r"\.(\d+)$", source_file)
    return int(m.group(1)) if m else None


def validate_frames(frames) -> list[str]:
    """Ensure all frames share atom count and chemical-symbol ordering."""
    ref = list(frames[0].get_chemical_symbols())
    n = len(ref)
    for i, atoms in enumerate(frames):
        sym = list(atoms.get_chemical_symbols())
        if len(sym) != n:
            raise ValueError(
                f"Frame {i} has {len(sym)} atoms, expected {n}. "
                "Cannot compute consistent CVs across frames."
            )
        if sym != ref:
            raise ValueError(
                f"Frame {i} chemical-symbol ordering differs from frame 0."
            )
    return ref


def check_indices(name: str, idx, natoms: int) -> None:
    for a in idx:
        if a < 0 or a >= natoms:
            raise IndexError(
                f"{name}: atom index {a} is out of range for a system "
                f"with {natoms} atoms (valid 0..{natoms-1}). "
                "Remember ASE indices are zero-based."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xyz", default="../analysis/scoord_1ps.xyz", type=Path)
    parser.add_argument("--out", default="../analysis/collective_variables.csv", type=Path)
    parser.add_argument("--dt-ps", type=float, default=1.0,
                        help="Approximate time between snapshots in ps. "
                             "VERIFY against your scoord write interval; "
                             "this is only assumed, not measured.")
    parser.add_argument("--t0-ps", type=float, default=1.0,
                        help="Approximate time of the FIRST snapshot in ps.")
    parser.add_argument("--connectivity-scale", type=float, default=1.3,
                        help="Covalent-radii scale factor for the "
                             "connected-components heuristic (default 1.3).")
    parser.add_argument("--allow-index-fallback", action="store_true",
                        help="If a source filename has no numeric suffix, "
                             "fall back to list order for timing instead "
                             "of failing.")
    args = parser.parse_args()

    xyz_path = args.xyz.resolve()
    if not xyz_path.exists():
        print(f"ERROR: {xyz_path} not found. Run convert_scoord.py first.",
              file=sys.stderr)
        return 1

    frames = read(str(xyz_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) == 0:
        print("ERROR: no frames read from XYZ.", file=sys.stderr)
        return 1

    symbols = validate_frames(frames)
    natoms = len(symbols)
    print(f"Loaded {len(frames)} frames, {natoms} atoms each "
          f"({frames[0].get_chemical_formula()}).")

    # Validate configured indices against the actual system
    check_indices("DIST1_ATOMS", DIST1_ATOMS, natoms)
    check_indices("DIST2_ATOMS", DIST2_ATOMS, natoms)
    check_indices("ANGLE_ATOMS", ANGLE_ATOMS, natoms)
    have_dihedral = natoms >= 4
    if have_dihedral:
        check_indices("DIHEDRAL_ATOMS", DIHEDRAL_ATOMS, natoms)
    else:
        print("NOTE: fewer than 4 atoms; dihedral will be reported as NaN.")

    ref_pos = frames[0].get_positions()

    # Resolve snapshot numbers from filenames up front, so timing derives
    # from the scoord index rather than list position.
    source_files = [a.info.get("source_file", f"frame_{i}")
                    for i, a in enumerate(frames)]
    snap_numbers = [snapshot_number(s) for s in source_files]

    missing = [source_files[i] for i, n in enumerate(snap_numbers) if n is None]
    if missing and not args.allow_index_fallback:
        print("ERROR: the following frames have no numeric snapshot suffix "
              "(expected 'scoord.<N>'):", file=sys.stderr)
        for m in missing:
            print(f"   {m}", file=sys.stderr)
        print("Re-run convert_scoord.py on scoord.* files, or pass "
              "--allow-index-fallback to use list order for timing.",
              file=sys.stderr)
        return 1
    if missing:
        print(f"NOTE: {len(missing)} frame(s) lack a numeric suffix; using "
              "list order for their timing (--allow-index-fallback).")

    # First snapshot number among those that have one; used as the time anchor.
    valid_numbers = [n for n in snap_numbers if n is not None]
    first_number = min(valid_numbers) if valid_numbers else 0

    rows = []
    for i, atoms in enumerate(frames):
        pos = atoms.get_positions()

        rmsd = kabsch_rmsd(pos, ref_pos)
        # ASE get_distance/get_angle/get_dihedral take zero-based indices
        d1 = atoms.get_distance(*DIST1_ATOMS)
        d2 = atoms.get_distance(*DIST2_ATOMS)
        angle = atoms.get_angle(*ANGLE_ATOMS)
        dih = atoms.get_dihedral(*DIHEDRAL_ATOMS) if have_dihedral else np.nan
        rg = radius_of_gyration(atoms)
        maxpair = max_pair_distance(atoms)
        ncomp = estimate_components(atoms, args.connectivity_scale)

        snap = snap_numbers[i]
        if snap is not None:
            approx_time = args.t0_ps + (snap - first_number) * args.dt_ps
        else:
            # fallback: list order (only reached with --allow-index-fallback)
            approx_time = args.t0_ps + i * args.dt_ps

        rows.append({
            "frame": i,
            "snapshot_number": snap if snap is not None else np.nan,
            "source_file": source_files[i],
            "approx_time_ps": approx_time,
            "rmsd_A": rmsd,
            "distance_1_A": d1,
            "distance_2_A": d2,
            "distance_difference_A": d1 - d2,
            "angle_deg": angle,
            "dihedral_deg": dih,
            "radius_of_gyration_A": rg,
            "max_pair_distance_A": maxpair,
            "estimated_components": ncomp,
        })

    df = pd.DataFrame(rows)
    nmulti = int((df["estimated_components"] > 1).sum())
    if nmulti:
        print(f"NOTE: {nmulti} frame(s) estimated as multi-component "
              f"(scale={args.connectivity_scale}). This is a HEURISTIC "
              "fragmentation indicator only; inspect those frames visually.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out.resolve()}")
    print("\nColumn summary:")
    print(df.describe(include="all").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
