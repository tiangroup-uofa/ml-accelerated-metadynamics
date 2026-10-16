#!/usr/bin/env python3
"""
energetics.py
=============
Barriers and reaction energies with an EXPLICIT reference state.

The tt-test audit found two defensible references for the same saddle that
differ by 13.4 kJ/mol (MACE-POLAR): the relaxed input endpoints, and the
minima reached by relaxing away from the saddle along its imaginary mode. A
single unlabeled "barrier" hides that. Everything here is therefore keyed by
reference state, and no function returns a bare barrier number.

Reference states
  relaxed_endpoint        E_R / E_P = the method's own relaxed input endpoints
                          (optimize_endpoints.py)
  connected_side_minimum  E_R / E_P = the minima reached from the saddle by the
                          -/+ imaginary-mode displacement + relaxation
                          (connectivity.py; a sanity check, NOT an IRC)

For each available state:
  barrier_forward  = E_TS - E_R
  barrier_reverse  = E_TS - E_P
  reaction_energy  = E_P  - E_R
All are electronic potential-energy differences of ONE method in ONE
environment (no ZPE, no thermal terms, no free energies).

The saddle's status travels with the numbers: unless ``ts_status`` says the
point was certified, every barrier block is labelled as an energy difference
to an UNCERTIFIED stationary point.
"""
from __future__ import annotations

EV2KJ = 96.48533212          # CODATA 2018: 1 eV = 96.485 332 12 kJ/mol
EV2KCAL = 23.060548          # as in workflow.py

REFERENCE_STATES = ("relaxed_endpoint", "connected_side_minimum")
REFERENCE_DESCRIPTIONS = {
    "relaxed_endpoint": "E(TS) - E(reactant) with the reactant/product being this method's "
                        "relaxed input endpoints",
    "connected_side_minimum": "E(TS) - E(minimum reached from the TS by negative/positive "
                              "imaginary-mode displacement + relaxation); NOT an IRC endpoint",
}
CERTIFIED = "CERTIFIED"


def diff(e_to: float, e_from: float) -> dict:
    """e_to - e_from in eV, kJ/mol and kcal/mol."""
    d = float(e_to) - float(e_from)
    return {"eV": d, "kJ_mol": d * EV2KJ, "kcal_mol": d * EV2KCAL}


def reference_state(name: str, reactant_eV=None, product_eV=None, source=None,
                    unavailable_reason=None) -> dict:
    if name not in REFERENCE_STATES:
        raise ValueError(f"unknown reference state {name!r}; use one of {REFERENCE_STATES}")
    ok = reactant_eV is not None and product_eV is not None
    return {"name": name, "available": ok, "description": REFERENCE_DESCRIPTIONS[name],
            "reactant_eV": None if reactant_eV is None else float(reactant_eV),
            "product_eV": None if product_eV is None else float(product_eV),
            "source": source,
            "unavailable_reason": None if ok else (unavailable_reason or "not computed")}


def energetics(ts_eV: float, states: list[dict], ts_status: str, ts_source=None,
               material_kJ_mol: float = 4.184) -> dict:
    """Labelled barriers / reaction energies for every reference state.

    ``states`` are ``reference_state(...)`` dicts. ``ts_status`` is the
    certification label of the saddle (see ``connectivity.ts_certification``).
    ``material_kJ_mol`` flags a reference-state dependence at or above it."""
    names = [s["name"] for s in states]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate reference states: {names}")
    certified = ts_status.startswith(CERTIFIED)
    label = ("E(certified first-order saddle) - E(reference)" if certified else
             "E(UNCERTIFIED stationary point) - E(reference); not a certified barrier")
    out = {
        "units_note": "electronic potential energies of one method, one environment; "
                      "no ZPE/thermal terms; not free energies",
        "ts": {"energy_eV": float(ts_eV), "status": ts_status, "certified": certified,
               "source": ts_source},
        "reference_states": {s["name"]: s for s in states},
        "barrier_forward": {}, "barrier_reverse": {}, "reaction_energy": {},
        "quantity_label": label,
        "policy": "No single unlabeled barrier is reported; every value names its reference state.",
    }
    for s in states:
        n = s["name"]
        if not s["available"]:
            for k in ("barrier_forward", "barrier_reverse", "reaction_energy"):
                out[k][n] = None
            continue
        out["barrier_forward"][n] = diff(ts_eV, s["reactant_eV"])
        out["barrier_reverse"][n] = diff(ts_eV, s["product_eV"])
        out["reaction_energy"][n] = diff(s["product_eV"], s["reactant_eV"])
    spread = {}
    for k in ("barrier_forward", "barrier_reverse", "reaction_energy"):
        vals = [v["kJ_mol"] for v in out[k].values() if v is not None]
        spread[k] = (max(vals) - min(vals)) if len(vals) > 1 else None
    out["reference_state_spread_kJ_mol"] = spread
    out["reference_state_dependent"] = any(v is not None and v >= material_kJ_mol
                                           for v in spread.values())
    out["material_threshold_kJ_mol"] = material_kJ_mol
    return out


def format_table(en: dict) -> str:
    """Human-readable lines, one per (quantity, reference state)."""
    lines = [f"  [{en['quantity_label']}]"]
    for k, lab in (("barrier_forward", "barrier fwd"), ("barrier_reverse", "barrier rev"),
                   ("reaction_energy", "dE(P-R)    ")):
        for n, v in en[k].items():
            txt = "n/a (" + (en["reference_states"][n]["unavailable_reason"] or "") + ")" \
                if v is None else f"{v['kJ_mol']:9.2f} kJ/mol ({v['kcal_mol']:7.2f} kcal/mol)"
            lines.append(f"  {lab}  vs {n:24s} {txt}")
    return "\n".join(lines)
