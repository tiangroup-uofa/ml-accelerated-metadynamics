#!/usr/bin/env python3
"""
reaction_coordinate.py
======================
Collective variables (CVs) for the C2 -> C1 intramolecular hydride transfer,
defined entirely by ``config.json``. No global atom index appears in this file:
every atom is referenced by a name from ``config["atom_map"]`` (or an element
selector), so the same code serves the real Sn-beta structure and any test system.

CV types (all distances are minimum-image when the structure is periodic)
------------------------------------------------------------------------
``distance``             d(a, b) = |r_a - r_b|                                 [Å]

``distance_difference``  delta = d(plus[0], plus[1]) - d(minus[0], minus[1])    [Å]
                         For the hydride transfer, with plus = (C2, H*) and
                         minus = (C1, H*): delta < 0 when H* sits on C2
                         (reactant), delta > 0 when it sits on C1 (product),
                         delta ~ 0 near a symmetric transfer TS.

``coordination``         CN = sum_{i in center} sum_{j in group, j != i} s(r_ij)
                         with the PLUMED-style rational switching function
                             s(r) = (1 - x^n) / (1 - x^m),  x = (r - d0) / r0
                         s = 1 for r <= d0 and s -> n/m at x = 1 (removable
                         singularity). Dimensionless; ~1 per bonded partner,
                         -> 0 for distant partners. Not normalized by |center|.

``group`` / ``center`` may be a list of atom names / integer indices, or an
element selector ``{"element": "H"}`` (optionally ``"exclude": [...]``).

Reference basins (``config["reference_basins"]``) are the approximate CPMD
values quoted for Mushrif et al. (2015). They are DIAGNOSTIC ONLY: being close
to them does not prove a structure is correct, and the CV definitions they refer
to must be confirmed against the original work before they are compared.

Usage
-----
    python reaction_coordinate.py frames.xyz --config config.json --csv cvs.csv
    from reaction_coordinate import ReactionCoordinates
    rc = ReactionCoordinates.from_config(cfg, natoms=len(atoms))
    rc.compute(atoms)          # -> {"CV1": ..., "CV2": ..., ...}
    rc.compute_many(frames)    # -> list of dicts (one per frame)
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_io import ConfigError, index_of, load_config, write_json  # noqa: E402
from geometry import distance, distance_matrix  # noqa: E402

CV_TYPES = ("distance", "distance_difference", "coordination")


def switching(r, r0: float, n: int = 6, m: int = 12, d0: float = 0.0):
    """Rational switching function s(r) = (1 - x^n)/(1 - x^m), x = (r - d0)/r0."""
    r = np.asarray(r, float)
    x = (r - d0) / r0
    out = np.empty_like(x)
    near = np.abs(x - 1.0) < 1e-8
    below = x <= 0.0
    reg = ~(near | below)
    out[reg] = (1.0 - x[reg] ** n) / (1.0 - x[reg] ** m)
    out[near] = n / m
    out[below] = 1.0
    return out


class ReactionCoordinates:
    """Resolved CV definitions for a system with a fixed atom count/ordering."""

    def __init__(self, specs: dict, natoms: int, symbols=None):
        self.natoms = natoms
        self.specs = specs              # name -> resolved spec (indices only)
        self.symbols = symbols

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: dict, natoms: int, symbols=None, names=None):
        raw = cfg.get("cvs")
        if not isinstance(raw, dict) or not raw:
            raise ConfigError("config has no 'cvs' section")
        names = list(raw) if names is None else list(names)
        specs = {}
        for name in names:
            if name not in raw:
                raise ConfigError(f"CV '{name}' not defined in config")
            specs[name] = _resolve(cfg, name, raw[name], natoms, symbols)
        return cls(specs, natoms, symbols)

    @property
    def names(self):
        return list(self.specs)

    # ------------------------------------------------------------------ #
    def _check(self, atoms):
        if len(atoms) != self.natoms:
            raise ValueError(f"CVs were configured for {self.natoms} atoms, "
                             f"got a structure with {len(atoms)}")

    def compute(self, atoms) -> dict:
        self._check(atoms)
        D = None
        out = {}
        for name, s in self.specs.items():
            t = s["type"]
            if t == "distance":
                out[name] = distance(atoms, *s["atoms"])
            elif t == "distance_difference":
                out[name] = distance(atoms, *s["plus"]) - distance(atoms, *s["minus"])
            elif t == "coordination":
                if D is None:
                    D = distance_matrix(atoms)
                tot = 0.0
                for i in s["center"]:
                    js = [j for j in s["group"] if j != i]
                    if js:
                        tot += float(np.sum(switching(D[i, js], s["r0"], s["n"], s["m"], s["d0"])))
                out[name] = tot
            else:  # pragma: no cover - guarded in _resolve
                raise ConfigError(f"unknown CV type {t}")
        return out

    def compute_many(self, frames) -> list[dict]:
        return [self.compute(a) for a in frames]

    def gradient(self, atoms, name: str, h: float = 1e-5) -> np.ndarray:
        """Cartesian gradient dCV/dR (natoms x 3) by central differences.

        Only atoms that enter the CV are displaced, so this is cheap."""
        s = self.specs[name]
        involved = sorted(set(s.get("atoms", [])) | set(s.get("plus", []))
                          | set(s.get("minus", [])) | set(s.get("center", []))
                          | set(s.get("group", [])))
        g = np.zeros((len(atoms), 3))
        sub = ReactionCoordinates({name: s}, self.natoms, self.symbols)
        for i in involved:
            for k in range(3):
                a = atoms.copy()
                a.positions[i, k] += h
                fp = sub.compute(a)[name]
                a.positions[i, k] -= 2 * h
                fm = sub.compute(a)[name]
                g[i, k] = (fp - fm) / (2 * h)
        return g

    def describe(self) -> dict:
        return {n: dict(s) for n, s in self.specs.items()}


# --------------------------------------------------------------------------- #
def _resolve_group(cfg, ref, natoms, symbols, what):
    if isinstance(ref, dict) and "element" in ref:
        if symbols is None:
            raise ConfigError(f"{what}: element selector needs the structure's symbols")
        excl = {index_of(cfg, r, natoms) for r in ref.get("exclude", [])}
        idx = [i for i, s in enumerate(symbols) if s == ref["element"] and i not in excl]
        if not idx:
            raise ConfigError(f"{what}: no atoms of element {ref['element']}")
        return idx
    if isinstance(ref, (list, tuple)):
        idx = [index_of(cfg, r, natoms) for r in ref]
        if len(set(idx)) != len(idx):
            raise ConfigError(f"{what}: duplicate atoms {ref}")
        if not idx:
            raise ConfigError(f"{what}: empty atom group")
        return idx
    raise ConfigError(f"{what}: expected a list of atoms or {{'element': X}}, got {ref!r}")


def _pair(cfg, ref, natoms, what):
    if not isinstance(ref, (list, tuple)) or len(ref) != 2:
        raise ConfigError(f"{what}: expected exactly two atoms, got {ref!r}")
    i, j = (index_of(cfg, r, natoms) for r in ref)
    if i == j:
        raise ConfigError(f"{what}: both atoms are index {i}")
    return [i, j]


def _resolve(cfg, name, spec, natoms, symbols):
    if not isinstance(spec, dict) or "type" not in spec:
        raise ConfigError(f"CV '{name}' must be an object with a 'type'")
    t = spec["type"]
    if t not in CV_TYPES:
        raise ConfigError(f"CV '{name}': unknown type '{t}' (known: {CV_TYPES})")
    out = {"type": t}
    if t == "distance":
        out["atoms"] = _pair(cfg, spec.get("atoms"), natoms, f"CV '{name}'")
    elif t == "distance_difference":
        out["plus"] = _pair(cfg, spec.get("plus"), natoms, f"CV '{name}'.plus")
        out["minus"] = _pair(cfg, spec.get("minus"), natoms, f"CV '{name}'.minus")
    else:  # coordination
        out["center"] = _resolve_group(cfg, spec.get("center"), natoms, symbols, f"CV '{name}'.center")
        out["group"] = _resolve_group(cfg, spec.get("group"), natoms, symbols, f"CV '{name}'.group")
        for key, default in (("r0", None), ("n", 6), ("m", 12), ("d0", 0.0)):
            v = spec.get(key, default)
            if v is None:
                raise ConfigError(f"CV '{name}': switching parameter '{key}' is not set")
            out[key] = v
        if not (out["r0"] > 0):
            raise ConfigError(f"CV '{name}': r0 must be > 0")
        if not (isinstance(out["n"], int) and isinstance(out["m"], int) and out["m"] > out["n"] > 0):
            raise ConfigError(f"CV '{name}': need integers m > n > 0 (got n={out['n']}, m={out['m']})")
    if spec.get("status"):
        out["status"] = spec["status"]
    return out


# --------------------------------------------------------------------------- #
#  reference-basin diagnostics
# --------------------------------------------------------------------------- #
def basin_diagnostics(values: dict, cfg: dict) -> dict | None:
    """Euclidean distance (in the CVs listed by each basin) from ``values`` to
    each configured reference basin. DIAGNOSTIC ONLY — never a pass/fail."""
    rb = cfg.get("reference_basins")
    if not rb:
        return None
    tol = rb.get("tolerance")
    out = {"note": "diagnostic only; CV definitions must match the reference work",
           "source": rb.get("source"), "basins": {}}
    for bname, target in (rb.get("basins") or {}).items():
        keys = [k for k in target if k in values]
        if not keys:
            continue
        dist = math.sqrt(sum((values[k] - target[k]) ** 2 for k in keys))
        out["basins"][bname] = {"target": target, "distance": dist,
                                "within_tolerance": (dist <= tol) if tol is not None else None}
    if out["basins"]:
        out["nearest"] = min(out["basins"], key=lambda b: out["basins"][b]["distance"])
    return out


def write_csv(path, rows: list[dict], extra: list[dict] | None = None) -> None:
    """Write one row per frame: frame index, optional extra columns, CVs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(**(extra[i] if extra else {}), **r) for i, r in enumerate(rows)]
    keys = ["frame"] + [k for k in rows[0]] if rows else ["frame"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"frame": i, **r})


def main(argv=None) -> int:
    from ase.io import read

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("structure", help="XYZ / extXYZ / ASE-readable file (all frames are used)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", default=None, help="write per-frame CVs to CSV")
    ap.add_argument("--json", default=None, help="write per-frame CVs to JSON")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    frames = read(args.structure, index=":")
    rc = ReactionCoordinates.from_config(cfg, len(frames[0]), frames[0].get_chemical_symbols())
    rows = rc.compute_many(frames)
    for i, r in enumerate(rows):
        print(f"frame {i:4d}  " + "  ".join(f"{k}={v:8.4f}" for k, v in r.items()))
    if args.csv:
        write_csv(args.csv, rows)
    if args.json:
        write_json(args.json, {"structure": args.structure, "definitions": rc.describe(),
                               "frames": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
