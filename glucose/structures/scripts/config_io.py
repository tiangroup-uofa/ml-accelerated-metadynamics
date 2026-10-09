#!/usr/bin/env python3
"""
config_io.py
============
Load ``config.json`` and resolve the reactive-atom *names* used everywhere else
(``"C1"``, ``"C2"``, ``"H_transfer"``, ...) to zero-based ASE indices.

No module in this benchmark hard-codes a global atom index: every index comes
from ``config["atom_map"]``. Until the authoritative Sn-beta structures arrive
those entries are ``null`` and anything that needs them fails loudly with
``ConfigError``.

Relative paths in the config are resolved against the config file's directory,
so the same config works from any working directory.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"


class ConfigError(ValueError):
    """The configuration is incomplete or inconsistent."""


def load_config(path=None) -> dict:
    """Read a config JSON. Keys starting with ``_`` are comments and ignored."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"config {path} is not valid JSON: {e}") from e
    cfg = _strip_comments(cfg)
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_dir"] = str(path.resolve().parent)
    return cfg


def _strip_comments(obj):
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def resolve_path(cfg: dict, p) -> Path:
    """Resolve a config-relative path."""
    p = Path(p)
    if p.is_absolute():
        return p
    return Path(cfg.get("_config_dir", HERE)) / p


def section(cfg: dict, name: str) -> dict:
    return copy.deepcopy(cfg.get(name, {}) or {})


# --------------------------------------------------------------------------- #
#  atom map
# --------------------------------------------------------------------------- #
def atom_map(cfg: dict, natoms: int | None = None, require=None) -> dict:
    """Return ``{name: index}`` for the configured reactive atoms.

    ``require`` lists names that must be set; by default every name in the map.
    Raises ``ConfigError`` for unset (null), non-integer, negative, duplicated
    or (if ``natoms`` is given) out-of-range indices.
    """
    raw = cfg.get("atom_map")
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("config has no 'atom_map' section")
    names = list(raw) if require is None else list(require)
    out, problems = {}, []
    for name in names:
        if name not in raw:
            problems.append(f"atom_map has no entry '{name}'")
            continue
        v = raw[name]
        if v is None:
            problems.append(f"atom_map['{name}'] is null (not configured yet)")
            continue
        if isinstance(v, bool) or not isinstance(v, int):
            problems.append(f"atom_map['{name}'] = {v!r} is not an integer index")
            continue
        if v < 0:
            problems.append(f"atom_map['{name}'] = {v} is negative (indices are zero-based)")
            continue
        if natoms is not None and v >= natoms:
            problems.append(f"atom_map['{name}'] = {v} is out of range for {natoms} atoms")
            continue
        out[name] = v
    seen = {}
    for name, i in out.items():
        if i in seen:
            problems.append(f"atom_map['{name}'] and atom_map['{seen[i]}'] both point to index {i}")
        seen[i] = name
    if problems:
        raise ConfigError("invalid atom mapping:\n  - " + "\n  - ".join(problems))
    return out


def index_of(cfg: dict, ref, natoms: int | None = None) -> int:
    """An atom reference is either a name from ``atom_map`` or a raw integer."""
    if isinstance(ref, bool):
        raise ConfigError(f"atom reference {ref!r} is not a name or index")
    if isinstance(ref, int):
        if ref < 0 or (natoms is not None and ref >= natoms):
            raise ConfigError(f"atom index {ref} out of range for {natoms} atoms")
        return ref
    if isinstance(ref, str):
        return atom_map(cfg, natoms, require=[ref])[ref]
    raise ConfigError(f"atom reference {ref!r} is not a name or index")


def fixed_atoms(cfg: dict, natoms: int) -> list[int]:
    """Indices held fixed (e.g. cluster-termination atoms). Names or ints allowed."""
    refs = cfg.get("fixed_atoms") or []
    idx = sorted({index_of(cfg, r, natoms) for r in refs})
    return idx


def charge_and_multiplicity(cfg: dict) -> tuple[int, int]:
    """Total charge and spin multiplicity (2S+1). Both must be set explicitly:
    the MACE calculators silently default to q=0, mult=1 if they are missing."""
    sysc = cfg.get("system", {}) or {}
    q, m = sysc.get("charge"), sysc.get("spin_multiplicity")
    problems = []
    if q is None or isinstance(q, bool) or not isinstance(q, int):
        problems.append(f"system.charge must be an integer (got {q!r})")
    if m is None or isinstance(m, bool) or not isinstance(m, int) or m < 1:
        problems.append(f"system.spin_multiplicity must be an integer >= 1 (got {m!r})")
    if problems:
        raise ConfigError("; ".join(problems))
    return q, m


def output_root(cfg: dict, model_tag: str | None = None) -> Path:
    root = resolve_path(cfg, cfg.get("output_dir", "outputs"))
    return root / model_tag if model_tag else root


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return repr(o)
