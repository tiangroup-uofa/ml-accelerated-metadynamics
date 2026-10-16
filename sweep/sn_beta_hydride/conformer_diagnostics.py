#!/usr/bin/env python3
"""
conformer_diagnostics.py
========================
Is a change in a reference energy chemistry or conformation?

Compares two structures of the SAME chemical state (e.g. a relaxed input
reactant and the minimum reached from the saddle on the reactant side):

* whole-system Kabsch RMSD, reactive-core RMSD (atom-map atoms, or
  ``config["conformer"]["core"]``) and the RMSD of everything else
* the torsions that changed most: one representative proper dihedral per
  rotatable non-metal heavy-atom bond (end atoms chosen heavy-first, so a
  hydroxyl H stands in only where no heavy neighbour exists), wrapped to
  [-180, 180]
* the energy offset E(other) - E(ref) in kJ/mol, flagged as MATERIAL when
  |offset| >= ``material_offset_kJ_mol`` (default 4.184 kJ/mol = 1 kcal/mol)

The verdict is descriptive. A lower minimum found this way is only *a* lower
local minimum: nothing here searches conformers or establishes a global
minimum.

    python conformer_diagnostics.py --config config_tian_tt_test.json \
        --ref a.xyz --other b.xyz --e-ref -1.0 --e-other -1.1 [--json out.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_io import load_config, write_json  # noqa: E402
from energetics import EV2KJ  # noqa: E402
from geometry import bond_set, kabsch_rmsd  # noqa: E402
from hydride import core_indices  # noqa: E402

DEFAULTS = {"material_offset_kJ_mol": 4.184, "core_similar_A": 0.25,
            "dihedral_change_deg": 30.0, "metals": ["Sn"], "top_dihedrals": 10}


def _opts(cfg):
    return {**DEFAULTS, **(cfg.get("conformer") or {})}


def representative_dihedrals(atoms, bond_scale: float = 1.2, metals=("Sn",)):
    """One (i, j, k, l) per bond j-k between non-metal heavy atoms that both have
    another neighbour. Metal-containing bonds are excluded: covalent radii make
    Sn 'bond' to everything within ~3 Å, which is not a torsion axis."""
    syms = atoms.get_chemical_symbols()
    metal = {i for i, s in enumerate(syms) if s in metals}
    bonds = {b for b in bond_set(atoms, bond_scale) if not (set(b) & metal)}
    nb = {i: set() for i in range(len(atoms))}
    for i, j in bonds:
        nb[i].add(j)
        nb[j].add(i)
    heavy = lambda i: syms[i] != "H"  # noqa: E731

    def pick(center, exclude):
        cands = sorted(nb[center] - {exclude}, key=lambda a: (not heavy(a), a))
        return cands[0] if cands else None

    out = []
    for j, k in sorted(bonds):
        if not (heavy(j) and heavy(k)):
            continue
        i, l = pick(j, k), pick(k, j)
        if i is None or l is None or i == l:
            continue
        out.append((i, j, k, l))
    return out


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def compare(ref, other, cfg: dict, e_ref=None, e_other=None, label="") -> dict:
    """Conformer comparison of ``other`` against ``ref`` (same atoms, same order)."""
    if ref.get_chemical_symbols() != other.get_chemical_symbols():
        raise ValueError("conformer comparison needs identical atom ordering")
    o = _opts(cfg)
    n = len(ref)
    core = core_indices(cfg, n)
    rest = [i for i in range(n) if i not in set(core)]
    P, Q = other.get_positions(), ref.get_positions()
    res = {
        "label": label,
        "rmsd_whole_A": kabsch_rmsd(P, Q),
        "rmsd_core_A": kabsch_rmsd(P[core], Q[core]) if len(core) >= 3 else None,
        "rmsd_noncore_A": kabsch_rmsd(P[rest], Q[rest]) if len(rest) >= 3 else None,
        "core_atoms": core,
    }
    bs = float((cfg.get("validation") or {}).get("bond_scale", 1.2))
    syms = ref.get_chemical_symbols()
    changes = []
    for q in representative_dihedrals(ref, bs, o["metals"]):
        a, b = ref.get_dihedral(*q), other.get_dihedral(*q)
        d = _wrap(b - a)
        changes.append({"atoms": list(q), "label": "-".join(f"{syms[i]}{i}" for i in q),
                        "ref_deg": round(_wrap(a), 1), "other_deg": round(_wrap(b), 1),
                        "change_deg": round(d, 1)})
    changes.sort(key=lambda c: -abs(c["change_deg"]))
    big = [c for c in changes if abs(c["change_deg"]) >= o["dihedral_change_deg"]]
    res["n_dihedrals_checked"] = len(changes)
    res["changed_dihedrals"] = big[: int(o["top_dihedrals"])]
    res["n_changed_dihedrals"] = len(big)
    res["dihedral_threshold_deg"] = o["dihedral_change_deg"]

    core_similar = res["rmsd_core_A"] is not None and res["rmsd_core_A"] <= o["core_similar_A"]
    res["core_similar"] = bool(core_similar)
    res["core_similar_threshold_A"] = o["core_similar_A"]
    if e_ref is not None and e_other is not None:
        off = (float(e_other) - float(e_ref)) * EV2KJ
        res["energy_offset_kJ_mol"] = off
        res["material"] = bool(abs(off) >= o["material_offset_kJ_mol"])
        res["material_threshold_kJ_mol"] = o["material_offset_kJ_mol"]
    else:
        res["energy_offset_kJ_mol"] = None
        res["material"] = None
    res["interpretation"] = _interpret(res)
    res["note"] = ("descriptive only: a lower structure found this way is a lower LOCAL "
                   "minimum, not evidence of the global minimum")
    return res


def _interpret(r):
    parts = []
    if r["core_similar"]:
        parts.append(f"reactive core essentially unchanged (core RMSD {r['rmsd_core_A']:.3f} Å)")
    elif r["rmsd_core_A"] is not None:
        parts.append(f"reactive core differs (core RMSD {r['rmsd_core_A']:.3f} Å)")
    if r["n_changed_dihedrals"]:
        top = r["changed_dihedrals"][0]
        parts.append(f"{r['n_changed_dihedrals']} torsion(s) changed by >= "
                     f"{r['dihedral_threshold_deg']:.0f} deg (largest {top['label']} "
                     f"{top['change_deg']:+.0f} deg)")
    if r["energy_offset_kJ_mol"] is not None:
        off = r["energy_offset_kJ_mol"]
        if r["material"]:
            kind = "conformational" if r["core_similar"] else "structural"
            parts.append(f"MATERIAL {kind} energy offset {off:+.2f} kJ/mol (>= "
                         f"{r['material_threshold_kJ_mol']:.2f}): barriers depend on which "
                         "structure is the reference")
        else:
            parts.append(f"energy offset {off:+.2f} kJ/mol (below the materiality threshold)")
    return "; ".join(parts)


def main(argv=None) -> int:
    from ase.io import read

    ap = argparse.ArgumentParser(description="Conformer-sensitivity diagnostic for two structures")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--e-ref", type=float, default=None, help="eV (default: energy stored in file)")
    ap.add_argument("--e-other", type=float, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    R, O = read(a.ref), read(a.other)

    def _e(atoms, given):
        if given is not None:
            return given
        try:
            return atoms.get_potential_energy()
        except Exception:
            return None
    res = compare(R, O, cfg, _e(R, a.e_ref), _e(O, a.e_other), f"{a.other} vs {a.ref}")
    print(res["interpretation"])
    if a.json:
        write_json(a.json, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
