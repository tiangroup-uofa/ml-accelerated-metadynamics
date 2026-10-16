#!/usr/bin/env python3
"""
connectivity.py
===============
+/- imaginary-mode connectivity check for a saddle point, and the TS
certification label that uses it.

    *** This is a connectivity SANITY CHECK, NOT an IRC. ***

Procedure (ported from Tian's glucose/structures/scripts/ts_sella.py, step 4):

1. take the dominant imaginary mode's Cartesian displacement vector from the
   Hessian, normalized so the atom that moves most is displaced by
   ``mode_displacement_A`` (default 0.1 Å, Tian's value)
2. build TS + step and TS - step (copies; the TS itself is never modified)
3. relax both with an ordinary minimizer (default BFGS, fmax 0.02 eV/Å)
4. label each relaxed structure from the mapped hydride coordinate
   (``hydride.classify_hydride_state``: reactant-like / product-like / ...)
5. compare each side with the relaxed input endpoint of the same state:
   whole and reactive-core RMSD, changed torsions, energy offset
   (``conformer_diagnostics.compare``)

The saddle "connects" when one side is reactant-like and the other
product-like. A steepest-descent path (IRC) can still end in different
minima: a minimizer started 0.1 Å off the saddle follows its own trajectory.
The side minima are the ``connected_side_minimum`` reference state of
``energetics.py``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from conformer_diagnostics import compare as conformer_compare
from energetics import EV2KJ
from geometry import kabsch_rmsd, max_force
from hydride import classify_hydride_state, core_distances, core_indices
from workflow import apply_fixed, make_optimizer, strip_calc

DEFAULTS = {"enabled": True, "mode_displacement_A": 0.1, "optimizer": "BFGS",
            "fmax": 0.02, "steps": 1500}
NOT_IRC = ("+/- imaginary-mode displacement + local relaxation: a connectivity sanity "
           "check, NOT an intrinsic reaction coordinate (IRC)")


def options(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("connectivity") or {})}


def displaced_pair(ts, mode, amplitude_A: float):
    """(ts + step, ts - step) with the mode scaled so its largest per-atom
    displacement equals ``amplitude_A``. Returns copies and the step array."""
    mode = np.asarray(mode, float).reshape(len(ts), 3)
    lead = float(np.linalg.norm(mode, axis=1).max())
    if not np.isfinite(lead) or lead <= 0:
        raise ValueError("imaginary-mode vector is zero or not finite")
    step = float(amplitude_A) * mode / lead
    plus, minus = strip_calc(ts), strip_calc(ts)
    plus.positions = ts.get_positions() + step
    minus.positions = ts.get_positions() - step
    return plus, minus, step


def _relax(atoms, calc, cfg, opts, stem: Path | None):
    a = strip_calc(atoms)
    apply_fixed(a, cfg)
    a.calc = calc
    traj = log = None
    if stem is not None:
        traj, log = stem.with_suffix(".traj"), stem.with_suffix(".log")
    opt = make_optimizer(opts["optimizer"], a, traj, log)
    t0 = time.time()
    conv = bool(opt.run(fmax=float(opts["fmax"]), steps=int(opts["steps"])))
    rec = {"converged": conv, "steps": int(opt.get_number_of_steps()),
           "energy_eV": float(a.get_potential_energy()),
           "fmax_eV_A": max_force(a.get_forces()), "seconds": round(time.time() - t0, 2)}
    return strip_calc(a), rec


def connectivity_check(ts, mode, new_calc, cfg: dict, endpoints: dict | None = None,
                       out: Path | None = None, ts_energy_eV: float | None = None) -> dict:
    """Run the +/- mode relaxation check.

    ``new_calc()`` returns a fresh calculator; ``endpoints`` optionally maps
    "reactant"/"product" to {"atoms", "energy_eV", "source"} for comparison."""
    from ase.io import write

    opts = options(cfg)
    ts_pos0 = ts.get_positions().copy()
    plus, minus, step = displaced_pair(ts, mode, opts["mode_displacement_A"])
    out = Path(out) if out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    sides = {}
    for sign, start in (("+", plus), ("-", minus)):
        tag = "plus" if sign == "+" else "minus"
        if out:
            write(str(out / f"conn_{tag}_start.xyz"), start)
        calc = new_calc()
        relaxed, rec = _relax(start, calc, cfg, opts, (out / f"conn_{tag}") if out else None)
        relaxed.info.update({"energy_eV": rec["energy_eV"], "connectivity_side": sign,
                             "note": "relaxed from TS +/- imaginary mode; not an IRC endpoint"})
        if out:
            write(str(out / f"conn_{tag}_relaxed.xyz"), relaxed)
        st = classify_hydride_state(relaxed, cfg)
        side = {"sign": sign, **rec,
                "start_state": classify_hydride_state(start, cfg)["state"],
                "hydride_state": st, "state": st["state"],
                "core_distances_A": core_distances(relaxed, cfg),
                "rmsd_from_ts_A": kabsch_rmsd(relaxed.get_positions(), ts_pos0),
                "files": ({"start": str(out / f"conn_{tag}_start.xyz"),
                           "relaxed": str(out / f"conn_{tag}_relaxed.xyz")} if out else None)}
        if ts_energy_eV is not None:
            side["energy_below_ts_kJ_mol"] = (ts_energy_eV - rec["energy_eV"]) * EV2KJ
        if endpoints:
            side["vs_endpoints"] = {}
            core = core_indices(cfg, len(relaxed))
            for role, ep in endpoints.items():
                P, Q = relaxed.get_positions(), ep["atoms"].get_positions()
                side["vs_endpoints"][role] = {"rmsd_whole_A": kabsch_rmsd(P, Q),
                                              "rmsd_core_A": kabsch_rmsd(P[core], Q[core])}
            match = {"reactant-like": "reactant", "product-like": "product"}.get(st["state"])
            if match and match in endpoints:
                ep = endpoints[match]
                side["matched_endpoint"] = match
                side["conformer_vs_matched_endpoint"] = conformer_compare(
                    ep["atoms"], relaxed, cfg, ep.get("energy_eV"), rec["energy_eV"],
                    f"{match}-side minimum vs relaxed input {match}")
        sides[sign] = side
        sides[sign]["_atoms"] = relaxed
    if not np.allclose(ts.get_positions(), ts_pos0):
        raise RuntimeError("connectivity check modified the TS geometry")

    by_state = {s["state"]: k for k, s in sides.items()}
    reactant_sign = by_state.get("reactant-like")
    product_sign = by_state.get("product-like")
    connects = reactant_sign is not None and product_sign is not None and \
        reactant_sign != product_sign
    both_converged = all(s["converged"] for s in sides.values())
    if connects and both_converged:
        verdict = ("CONNECTS: one side relaxes to a reactant-like and the other to a "
                   "product-like minimum (" + NOT_IRC + ")")
    elif connects:
        verdict = "CONNECTS (states) but a side relaxation did NOT converge; energies unreliable"
    else:
        verdict = ("DOES NOT CONNECT reactant-like and product-like basins: side states "
                   f"{sides['+']['state']} (+) / {sides['-']['state']} (-)")
    atoms_out = {k: s.pop("_atoms") for k, s in sides.items()}
    result = {
        "method_note": NOT_IRC,
        "settings": opts,
        "mode_step_max_atom_A": float(np.linalg.norm(step, axis=1).max()),
        "sides": sides,
        "reactant_side": reactant_sign, "product_side": product_sign,
        "connects": bool(connects), "both_converged": both_converged,
        "verdict": verdict,
        "ts_unchanged": True,
    }
    result["_atoms"] = atoms_out
    return result


def side_minimum_reference(conn: dict | None):
    """(reactant_eV, product_eV, reason) of the connected side minima."""
    if conn is None:
        return None, None, "connectivity check not run"
    if not conn["connects"]:
        return None, None, "saddle does not connect reactant-like and product-like basins"
    if not conn["both_converged"]:
        return None, None, "a side relaxation did not converge"
    return (conn["sides"][conn["reactant_side"]]["energy_eV"],
            conn["sides"][conn["product_side"]]["energy_eV"], None)


def ts_certification(*, stationary: bool, fmax: float, stationary_fmax: float,
                     n_imaginary: int, overlap, min_overlap: float, geometry_valid: bool,
                     connectivity: dict | None) -> str:
    """One label for the evidence a saddle has. 'CERTIFIED' requires ALL of:
    stationarity, exactly one significant imaginary mode, that mode aligned with
    the hydride-transfer coordinate, a valid geometry, and +/- mode relaxation
    reaching reactant-like and product-like basins (converged)."""
    if not geometry_valid:
        return "NOT CERTIFIED: geometry fails structure validation"
    if not stationary:
        return (f"NOT CERTIFIED: not a stationary point (fmax {fmax:.2e} > "
                f"{stationary_fmax} eV/Å)")
    if n_imaginary != 1:
        return f"NOT CERTIFIED: {n_imaginary} significant imaginary modes (need exactly 1)"
    if overlap is None or overlap < min_overlap:
        return (f"NOT CERTIFIED: imaginary mode not aligned with the hydride transfer "
                f"(overlap {overlap} < {min_overlap})")
    if connectivity is None:
        return ("NOT CERTIFIED: first-order saddle candidate with a hydride-transfer mode; "
                "+/- mode connectivity not checked")
    if not connectivity["connects"]:
        return "NOT CERTIFIED: " + connectivity["verdict"]
    if not connectivity["both_converged"]:
        return "NOT CERTIFIED: +/- mode side relaxation did not converge"
    return ("CERTIFIED: stationary first-order saddle; imaginary mode is the hydride transfer; "
            "+/- mode relaxation reaches reactant-like and product-like minima (not an IRC)")
