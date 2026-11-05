"""
DFT-budget experiment framework: how many DFT labels recover a useful barrier?

STATUS: framework only. It has been exercised on SYNTHETIC data in tests; it must
not be run for science until Tian's real Gaussian reference set exists. The
hypothesis "~2 endpoints + ~10 TS-region points suffice" is a HYPOTHESIS; nothing
here assumes TS-enriched sampling wins -- the strategies are compared on equal
terms and the DFT error map decides.

Leakage rules (enforced, tested)
--------------------------------
* The unit of splitting is the GROUP (one Gaussian job = one path point,
  including all its optimisation steps), never the record, so consecutive steps
  of one optimisation cannot straddle train and evaluation.
* Held-out metrics are computed only on groups with no labelled record.
* Point selection uses LABEL-FREE information only: path coordinate s and an
  explicitly supplied `ts_center_s` that must come from the MACE pathway
  (e.g. `mace_max_energy_location`), never from DFT energies/forces.
* The barrier error is computed against the full DFT reference (by design), and
  `ts_in_training` / `reactant_in_training` flags say whether those two points
  were themselves labelled, so a barrier "recovered" by training on the TS
  point is visible as such.

Reuses the existing correction classes (sweep/diels_alder/neb/corrections.py).
Known limits carried over from that code: GlobalScale/Affine/ElementScale are
force-fits; ElementScale is non-conservative; sweep/corrections/PairwiseDelta has
no Sn/Si pair types and is deliberately NOT used here (extending it is a
methodology change that needs review).
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from .dft_compare import EV_TO_KJ_MOL, ReferenceSetError, cosine, finals, path_coordinate

_CORR = Path(__file__).resolve().parents[2] / "diels_alder" / "neb" / "corrections.py"
KINDS = ("none", "global", "affine", "element")


def load_corrections():
    import sys
    if "_da_corrections" in sys.modules:
        return sys.modules["_da_corrections"]
    spec = importlib.util.spec_from_file_location("_da_corrections", _CORR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_da_corrections"] = mod        # py3.9 dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


class LeakageError(ValueError):
    pass


@dataclass
class Subset:
    name: str
    strategy: str
    train_groups: tuple
    train_rids: tuple
    label_s: tuple             # path coordinate of each labelled group

    @property
    def n_labels(self) -> int:
        return len(self.train_rids)


def assert_no_leakage(train_rids, eval_rids, group_of):
    t, e = set(train_rids), set(eval_rids)
    if t & e:
        raise LeakageError(f"records in both train and eval: {sorted(t & e)[:3]}")
    tg = {group_of[r] for r in t}
    eg = {group_of[r] for r in e}
    if tg & eg:
        raise LeakageError(f"groups straddle train/eval (correlated optimisation steps): {sorted(tg & eg)[:3]}")


# ---------------------------------------------------------------- selection
def mace_max_energy_location(final_records, e_mace_eV_by_rid):
    """Label-free TS-region guess: path coordinate of the highest-MACE-energy point.
    This is a sampling LOCATION only -- it is never reported as a barrier or TS."""
    s_of = path_coordinate(final_records)
    top = max(final_records, key=lambda r: e_mace_eV_by_rid[r.rid])
    return s_of[top.group]


def _mk(name, strategy, records, groups, labels="final"):
    by_group = {}
    for r in records:
        by_group.setdefault(r.group, []).append(r)
    s_of = path_coordinate(finals(records))
    rids, gs = [], []
    for g in groups:
        recs = by_group[g]
        sel = [r for r in recs if r.is_final] if labels == "final" else recs
        rids += [r.rid for r in sel]; gs.append(g)
    return Subset(name, strategy, tuple(gs), tuple(rids), tuple(s_of[g] for g in gs))


def _endpoint_groups(records):
    fr = finals(records)
    gs = [r.group for r in fr if r.role in ("reactant", "product")]
    if len(gs) != 2:
        raise ReferenceSetError("need exactly one reactant and one product group")
    return gs


def endpoints_only(records, labels="final"):
    return _mk("endpoints", "endpoints_only", records, _endpoint_groups(records), labels)


def uniform(records, n_total, labels="final"):
    """n_total points evenly spaced in s (includes both endpoints)."""
    if n_total < 2:
        raise ReferenceSetError("uniform needs n_total >= 2")
    fr = finals(records); s_of = path_coordinate(fr)
    targets = np.linspace(0, 1, n_total)
    chosen = []
    for t in targets:
        g = min((r.group for r in fr if r.group not in chosen), key=lambda g: abs(s_of[g] - t))
        chosen.append(g)
    return _mk(f"uniform{n_total}", "uniform", records, chosen, labels)


def ts_enriched(records, n_ts, ts_center_s, labels="final"):
    """Endpoints + the n_ts interior points nearest `ts_center_s` (label-free)."""
    fr = finals(records); s_of = path_coordinate(fr)
    ends = _endpoint_groups(records)
    interior = sorted((r.group for r in fr if r.group not in ends), key=lambda g: abs(s_of[g] - ts_center_s))
    if n_ts > len(interior):
        raise ReferenceSetError(f"only {len(interior)} interior points available, asked for {n_ts}")
    return _mk(f"ts_enriched{n_ts}", "ts_enriched", records, ends + interior[:n_ts], labels)


# ---------------------------------------------------------------- fit/eval
def fit_correction(kind, train_records, evaluated):
    C = load_corrections()
    if kind == "none":
        return C.Identity()
    bf = [evaluated[r.rid][1] for r in train_records]
    rf = [r.f_dft for r in train_records]
    be = [evaluated[r.rid][0] for r in train_records]
    re_ = [r.e_dft_eV for r in train_records]
    if kind == "global":
        return C.GlobalScale.fit(bf, rf)
    if kind == "affine":
        return C.Affine.fit(bf, rf, be, re_)
    if kind == "element":
        return C.ElementScale.fit(bf, rf, [_sym(r) for r in train_records])
    raise ReferenceSetError(f"unknown correction kind {kind}")


def _sym(r):
    return Atoms(numbers=r.numbers).get_chemical_symbols()


def evaluate_subset(kind, subset, records, evaluated):
    """Fit `kind` on the subset's labels; score held-out groups and the barrier."""
    rec = {r.rid: r for r in records}
    group_of = {r.rid: r.group for r in records}
    train = [rec[i] for i in subset.train_rids]
    train_groups = set(subset.train_groups)
    held = [r for r in records if r.group not in train_groups]
    if not held:
        raise ReferenceSetError("no held-out groups left: every group is labelled")
    assert_no_leakage(subset.train_rids, [r.rid for r in held], group_of)
    corr = fit_correction(kind, train, evaluated)

    def corrected(r):
        a = Atoms(numbers=r.numbers, positions=r.positions)
        return corr.apply(a, evaluated[r.rid][0], evaluated[r.rid][1])

    ref = [r for r in finals(records) if r.role == "reactant"]
    ts = [r for r in finals(records) if r.role == "ts"]
    if len(ref) != 1 or len(ts) != 1:
        raise ReferenceSetError("evaluate_subset needs exactly one final reactant and one final ts record")
    cr, ct = corrected(ref[0]), corrected(ts[0])
    e_err, f_rmse, f_cos = [], [], []
    for r in held:
        e, f = corrected(r)
        e_err.append(((e - cr[0]) - (r.e_dft_eV - ref[0].e_dft_eV)) * EV_TO_KJ_MOL)
        f_rmse.append(float(np.sqrt(((f - r.f_dft) ** 2).mean())))
        f_cos.append(cosine(r.f_dft, f))
    dft_b = (ts[0].e_dft_eV - ref[0].e_dft_eV) * EV_TO_KJ_MOL
    mace_b = (ct[0] - cr[0]) * EV_TO_KJ_MOL
    prod = [r for r in finals(records) if r.role == "product"]
    cp = corrected(prod[0]) if prod else None
    tg = set(subset.train_groups)
    return {"strategy": subset.strategy, "subset": subset.name, "correction": kind,
            "n_labels": subset.n_labels, "label_groups": list(subset.train_groups),
            "label_s": [round(x, 4) for x in subset.label_s],
            "n_heldout_records": len(held),
            "heldout_energy_rmse_kJ_mol": float(np.sqrt(np.mean(np.square(e_err)))),
            "heldout_force_rmse_eV_A": float(np.mean(f_rmse)),
            "heldout_force_cos": float(np.nanmean(f_cos)),
            "reference_state": "dft_geometry_single_point",
            "barrier_forward_dft_kJ_mol": float(dft_b), "barrier_forward_corrected_kJ_mol": float(mace_b),
            "barrier_error_kJ_mol": float(mace_b - dft_b),
            "reaction_energy_error_kJ_mol": (None if cp is None else float(((cp[0] - cr[0]) - (prod[0].e_dft_eV - ref[0].e_dft_eV)) * EV_TO_KJ_MOL)),
            "ts_in_training": ts[0].group in tg, "reactant_in_training": ref[0].group in tg,
            "product_in_training": bool(prod) and prod[0].group in tg,
            # a barrier error is only a HELD-OUT estimate when the TS point itself was not labelled;
            # the reactant is the barrier's reference and is labelled by every endpoint-based strategy
            "barrier_error_is_heldout_wrt_ts": ts[0].group not in tg,
            # ElementScale leaves energies unchanged: its barrier equals the uncorrected one by construction
            "energy_is_corrected": kind in ("global", "affine")}


def run_budget_experiment(records, evaluated, subsets, kinds=("none", "affine")):
    return [evaluate_subset(k, s, records, evaluated) for s in subsets for k in kinds]


def plot_budget(results, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    for kind in sorted({r["correction"] for r in results}):
        rows = [r for r in results if r["correction"] == kind]
        fig, ax = plt.subplots(figsize=(6, 4))
        for strat in sorted({r["strategy"] for r in rows}):
            sub = sorted([r for r in rows if r["strategy"] == strat], key=lambda r: r["n_labels"])
            ln, = ax.plot([r["n_labels"] for r in sub], [abs(r["barrier_error_kJ_mol"]) for r in sub], "-", label=strat)
            for r in sub:   # hollow marker = the TS point itself was labelled, so NOT a held-out barrier estimate
                held = r["barrier_error_is_heldout_wrt_ts"]
                ax.plot(r["n_labels"], abs(r["barrier_error_kJ_mol"]), "o", color=ln.get_color(), mfc=(ln.get_color() if held else "white"))
        ax.plot([], [], "o", mfc="white", color="gray", label="hollow: TS point was labelled (not held out)")
        ax.set_xlabel("number of DFT labels"); ax.set_ylabel("|barrier error| (kJ/mol)")
        ax.set_title(f"barrier error vs DFT labels  [{kind}]"); ax.grid(alpha=.3); ax.legend()
        p = f"{out_prefix}_vs_nlabels_{kind}.png"; fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig); out.append(p)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [f"{r['subset']}|{r['correction']}" for r in results]
    ax.bar(range(len(results)), [abs(r["barrier_error_kJ_mol"]) for r in results],
           color=["C0" if r["barrier_error_is_heldout_wrt_ts"] else "C3" for r in results])
    ax.set_xticks(range(len(results))); ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("|barrier error| (kJ/mol)"); ax.set_title("barrier error vs sampling strategy (red = TS was labelled)")
    p = f"{out_prefix}_vs_strategy.png"; fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig); out.append(p)
    return out
