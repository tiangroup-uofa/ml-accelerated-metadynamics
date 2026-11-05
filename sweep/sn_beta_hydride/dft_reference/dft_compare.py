"""
DFT (Gaussian) vs MACE comparison along an R -> TS -> P path.

Nothing here invents reference data: records come only from parsed Gaussian
logs (see gaussian_parser.py) listed in an explicit manifest.

Manifest (JSON), every field explicit:
{
  "system": {"charge": 0, "multiplicity": 1},
  "basis_note": "free text: Sn ECP/basis actually used (not parsed from the log)",
  "structures": [
    {"file": "r.log",  "role": "reactant", "pathway": "A", "point": 0},
    {"file": "p01.log","role": "path",     "pathway": "A", "point": 1},
    {"file": "ts.log", "role": "ts",       "pathway": "A", "point": 3},
    {"file": "p.log",  "role": "product",  "pathway": "A", "point": 6}
  ]
}
Each log is one *group* (pathway, point); every optimisation step inside it is a
record of that group. Groups -- not records -- are the unit for train/eval
splits, because steps of one optimisation are highly correlated.

Barrier convention. A DFT-vs-MACE barrier computed from DFT *geometries* is the
`dft_geometry_single_point` reference state: E(TS)-E(R) with MACE evaluated at
the DFT-optimised geometries. It is NOT the MACE-relaxed barrier of
energetics.py (`relaxed_endpoint` / `connected_side_minimum`) and must never be
reported next to those without its label.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gaussian_parser import GaussianJob, HARTREE_TO_KJ_MOL, HARTREE_TO_EV, parse_gaussian_log

EV_TO_KJ_MOL = 96.48533212331
ROLES = ("reactant", "product", "ts", "path")
REFERENCE_STATE = "dft_geometry_single_point"


class ReferenceSetError(ValueError):
    pass


@dataclass
class Record:
    rid: str
    group: str                 # "pathway:point"
    pathway: str
    point: int
    role: str
    step: int
    is_final: bool
    numbers: np.ndarray
    positions: np.ndarray
    e_dft_eV: float
    f_dft: np.ndarray          # eV/A
    ts_n_imag: int | None = None   # from a DFT freq job, if the log has one


# ----------------------------------------------------------------- loading
def load_reference_set(directory, manifest_path=None, *, expect_charge=None, expect_multiplicity=None):
    d = Path(directory)
    mp = Path(manifest_path) if manifest_path else d / "manifest.json"
    if not mp.exists():
        raise ReferenceSetError(f"manifest not found: {mp} (roles/points must be explicit, never guessed)")
    man = json.loads(mp.read_text())
    sysm = man.get("system") or {}
    q, mult = sysm.get("charge"), sysm.get("multiplicity")
    if q is None or mult is None:
        raise ReferenceSetError("manifest.system must give charge and multiplicity")
    for want, have, nm in ((expect_charge, q, "charge"), (expect_multiplicity, mult, "multiplicity")):
        if want is not None and want != have:
            raise ReferenceSetError(f"manifest {nm}={have} differs from the model/config setting {want}")
    records, numbers0, seen_groups, jobs = [], None, set(), []
    for ent in man["structures"]:
        role = ent["role"]
        if role not in ROLES:
            raise ReferenceSetError(f"unknown role {role!r}; allowed {ROLES}")
        job = parse_gaussian_log(d / ent["file"])
        jobs.append(job)
        if (job.charge, job.multiplicity) != (q, mult):
            raise ReferenceSetError(f"{ent['file']}: Gaussian charge/mult {(job.charge, job.multiplicity)} "
                                    f"!= manifest {(q, mult)}")
        ack = set(man.get("acknowledged_warnings", []))
        unack = sorted({w.split("]")[0].lstrip("[") for w in job.warnings if w.startswith("[")} - ack)
        if unack:
            raise ReferenceSetError(f"{ent['file']}: unacknowledged calculation-setting warning(s) {unack}: {job.warnings}. "
                                    "List the tags under manifest 'acknowledged_warnings' only after deciding they are acceptable")
        group = f"{ent['pathway']}:{int(ent['point'])}"
        if group in seen_groups:
            raise ReferenceSetError(f"duplicate group {group}")
        seen_groups.add(group)
        if numbers0 is None:
            numbers0 = job.steps[0].numbers
        elif not np.array_equal(numbers0, job.steps[0].numbers):
            raise ReferenceSetError(f"{ent['file']}: atom order/elements differ from the first structure")
        for s in job.steps:
            records.append(Record(f"{ent['file']}#{s.index}", group, str(ent["pathway"]), int(ent["point"]),
                                  role, s.index, s.index == len(job.steps) - 1, s.numbers, s.positions,
                                  s.energy_eV, s.forces_eV_A, job.n_imaginary if role == "ts" else None))
    return records, man, jobs


def finals(records):
    return [r for r in records if r.is_final]


def path_coordinate(final_records):
    """Fractional cumulative Kabsch-RMSD arc length along the points of ONE pathway
    (ordered by `point`). Geometry-only, so it is label-free."""
    fr = sorted(final_records, key=lambda r: r.point)
    s = [0.0]
    for a, b in zip(fr[:-1], fr[1:]):
        s.append(s[-1] + _kabsch_rmsd(a.positions, b.positions))
    s = np.array(s)
    return {r.group: (float(x / s[-1]) if s[-1] > 0 else 0.0) for r, x in zip(fr, s)}


def _kabsch_rmsd(a, b):
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(b0.T @ a0)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((b0 @ r - a0) ** 2).sum(1).mean()))


# ----------------------------------------------------------------- metrics
def cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def force_metrics(f_ref, f_pred, subsets=None):
    """Global and per-subset force metrics (eV/A). `subsets`: {name: [atom idx]}."""
    f_ref, f_pred = np.asarray(f_ref, float), np.asarray(f_pred, float)
    if f_ref.shape != f_pred.shape:
        raise ReferenceSetError(f"force shapes differ {f_ref.shape} vs {f_pred.shape}")
    err = f_pred - f_ref
    out = {"cos": cosine(f_ref, f_pred),
           "norm_ratio": float(np.linalg.norm(f_pred) / np.linalg.norm(f_ref)) if np.linalg.norm(f_ref) else float("nan"),
           "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
           "per_atom_err": np.linalg.norm(err, axis=1)}
    for name, idx in (subsets or {}).items():
        idx = list(idx)
        out[f"{name}_rmse"] = float(np.sqrt((err[idx] ** 2).mean()))
        out[f"{name}_cos"] = cosine(f_ref[idx], f_pred[idx])
    return out


def align_energies(e_dft_eV, e_mace_eV, ref_index=None, mode="reference_point"):
    """Relative energies in kJ/mol. mode='reference_point' subtracts each method's
    own energy at `ref_index` (the reactant); 'mean_offset' removes the mean
    difference instead. The alignment mode is returned and must be reported."""
    d, m = np.asarray(e_dft_eV, float), np.asarray(e_mace_eV, float)
    if mode == "reference_point":
        if ref_index is None:
            raise ReferenceSetError("reference_point alignment needs ref_index")
        d_rel, m_rel = d - d[ref_index], m - m[ref_index]
    elif mode == "mean_offset":
        d_rel, m_rel = d - d.mean(), m - m.mean()
    else:
        raise ReferenceSetError(f"unknown alignment mode {mode}")
    return d_rel * EV_TO_KJ_MOL, m_rel * EV_TO_KJ_MOL, mode


def energy_metrics(d_rel_kJ, m_rel_kJ):
    e = np.asarray(m_rel_kJ) - np.asarray(d_rel_kJ)
    return {"rmse_kJ_mol": float(np.sqrt((e ** 2).mean())), "max_abs_kJ_mol": float(np.abs(e).max()),
            "mean_signed_kJ_mol": float(e.mean()), "errors_kJ_mol": e}


def barrier_comparison(records, e_mace_eV_by_rid):
    """Barrier / reaction energy at DFT geometries, explicitly labelled.

    Needs exactly one final 'reactant', 'ts', 'product' record in the pathway.
    The DFT TS is labelled DFT_TS_FREQ_VERIFIED only if its log carried a
    frequency job with exactly one imaginary mode; otherwise DFT_TS_UNVERIFIED.
    A path maximum is never promoted to a TS here: the TS role must be given."""
    def one(role):
        c = [r for r in finals(records) if r.role == role]
        if len(c) != 1:
            raise ReferenceSetError(f"barrier comparison needs exactly one final '{role}' record, found {len(c)}")
        return c[0]
    r, t, p = one("reactant"), one("ts"), one("product")
    g = lambda x: e_mace_eV_by_rid[x.rid]
    # one imaginary frequency is NECESSARY, not sufficient: the mode's identity (H transfer) is not checked here
    status = "DFT_TS_ONE_IMAGINARY_MODE_MODE_IDENTITY_UNCHECKED" if t.ts_n_imag == 1 else "DFT_TS_UNVERIFIED"
    out = {"reference_state": REFERENCE_STATE, "dft_ts_status": status,
           "note": "MACE evaluated at DFT geometries; not comparable to relaxed_endpoint/connected_side_minimum"}
    for nm, (a, b) in {"barrier_forward": (t, r), "barrier_reverse": (t, p), "reaction_energy": (p, r)}.items():
        dft = (a.e_dft_eV - b.e_dft_eV) * EV_TO_KJ_MOL
        mace = (g(a) - g(b)) * EV_TO_KJ_MOL
        out[nm] = {"dft_kJ_mol": float(dft), "mace_kJ_mol": float(mace), "error_kJ_mol": float(mace - dft)}
    return out


# ------------------------------------------------------- full-path analysis
def evaluate_records(records, evaluate, expect_numbers=None):
    """`evaluate(numbers, positions) -> (energy_eV, forces eV/A)`; the caller builds
    the calculator with the SAME charge/spin as the DFT jobs."""
    res = {}
    for r in records:
        if expect_numbers is not None and not np.array_equal(r.numbers, expect_numbers):
            raise ReferenceSetError(f"{r.rid}: atom ordering differs from the MACE reference structure")
        e, f = evaluate(r.numbers, r.positions)
        res[r.rid] = (float(e), np.asarray(f, float))
    return res


def analyse(records, evaluated, *, subsets=None, align="reference_point"):
    """Per-record table + summary. `records` should be one pathway."""
    ref = [r for r in finals(records) if r.role == "reactant"]
    if align == "reference_point" and len(ref) != 1:
        raise ReferenceSetError("reference_point alignment needs exactly one final reactant record")
    ordered = sorted(records, key=lambda r: (r.point, r.step))
    e_d = [r.e_dft_eV for r in ordered]
    e_m = [evaluated[r.rid][0] for r in ordered]
    ri = ordered.index(ref[0]) if ref else None
    d_rel, m_rel, mode = align_energies(e_d, e_m, ri, align)
    s_of = path_coordinate(finals(records))
    rows = []
    for r, dr, mr in zip(ordered, d_rel, m_rel):
        fm = force_metrics(r.f_dft, evaluated[r.rid][1], subsets)
        row = {"rid": r.rid, "group": r.group, "role": r.role, "step": r.step, "is_final": r.is_final,
               "s": s_of.get(r.group), "dft_rel_kJ_mol": float(dr), "mace_rel_kJ_mol": float(mr),
               "energy_err_kJ_mol": float(mr - dr)}
        row.update({k: v for k, v in fm.items() if k != "per_atom_err"})
        row["per_atom_err"] = fm["per_atom_err"]
        rows.append(row)
    em = energy_metrics(d_rel, m_rel)
    summary = {"alignment": mode, "n_records": len(rows), "energy": {k: v for k, v in em.items() if k != "errors_kJ_mol"},
               "force_cos_mean": float(np.nanmean([x["cos"] for x in rows])),
               "force_rmse_mean": float(np.mean([x["rmse"] for x in rows]))}
    try:
        summary["barrier_comparison"] = barrier_comparison(records, {r.rid: evaluated[r.rid][0] for r in records})
    except ReferenceSetError as e:
        summary["barrier_comparison"] = {"reference_state": REFERENCE_STATE, "unavailable": str(e)}
    return rows, summary


def plot_path_diagnostics(tables, out_png, title="DFT vs MACE along path"):
    """tables: {model_label: rows}. Five panels vs path coordinate s (finals only
    for lines; optimisation-step records are drawn as faint dots)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 1, figsize=(7, 13), sharex=True)
    first = next(iter(tables.values()))
    fin = sorted([r for r in first if r["is_final"]], key=lambda r: r["s"])
    ax[0].plot([r["s"] for r in fin], [r["dft_rel_kJ_mol"] for r in fin], "k-o", label="DFT")
    for lab, rows in tables.items():
        f = sorted([r for r in rows if r["is_final"]], key=lambda r: r["s"])
        x = [r["s"] for r in f]
        ax[0].plot(x, [r["mace_rel_kJ_mol"] for r in f], "-s", label=lab)
        ax[1].plot(x, [r["energy_err_kJ_mol"] for r in f], "-s", label=lab)
        ax[2].plot(x, [r["cos"] for r in f], "-s", label=lab)
        ax[3].plot(x, [r["norm_ratio"] for r in f], "-s", label=lab)
        key = "core_rmse" if "core_rmse" in f[0] else "rmse"
        ax[4].plot(x, [r[key] for r in f], "-s", label=lab)
    for a, y in zip(ax, ["E rel. reactant (kJ/mol)", "MACE - DFT (kJ/mol)", "force cosine",
                         "|F_MACE| / |F_DFT|", "force RMSE (eV/A)"]):
        a.set_ylabel(y); a.grid(alpha=.3)
    ax[2].axhline(1, c="gray", lw=.5); ax[3].axhline(1, c="gray", lw=.5)
    ax[4].set_xlabel("path coordinate s (cumulative RMSD, 0=R, 1=P)")
    ax[0].legend(); ax[0].set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    return out_png
