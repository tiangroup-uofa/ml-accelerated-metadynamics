#!/usr/bin/env python3
"""
validate_structure.py
=====================
Strict pre-calculation validation of reactant/product (or any single) structure.
Every stage script runs this first and refuses to start a calculation when it
reports errors — a wrong input should cost seconds, not a day of NEB.

ERRORS (hard fail)
  * file missing / unreadable by ASE / more than one frame / zero atoms
  * non-finite coordinates; periodic flags with a degenerate cell
  * duplicate atoms (d < duplicate_A) or impossible contacts (d < min_distance_A)
  * isolated atoms (no neighbour within isolated_scale x covalent sum),
    unless listed in validation.allowed_isolated
  * required elements (e.g. Sn) absent; atom count != system.expected_n_atoms
  * reactive-atom mapping unset / out of range / duplicated / wrong element
  * charge or multiplicity unset, or electron-count parity inconsistent with
    the multiplicity
  * endpoint bond expectations violated (default: H* bonded to C2 and not C1 in
    the reactant, the reverse in the product) — catches swapped files and wrong
    H indices; severity configurable
  * reactant/product atom counts or element ordering differ

WARNINGS (reported, do not block)
  * short contacts (d < short_contact_scale x covalent sum)
  * bond changes between endpoints other than the expected ones
  * large displacement of non-reactive atoms between endpoints (possible
    permutation or a different conformer)
  * fixed atoms that differ between endpoints
  * no O around Sn within the shell; CV values far from the reference basins

DIAGNOSTICS (info only): formula, element counts, cell/PBC, fragments, Sn
coordination shell (neighbours, distances), reactive-atom distances, CVs.

Usage
-----
    python validate_structure.py --config config.json            # reactant+product from config
    python validate_structure.py --reactant r.xyz --product p.xyz --json report.json
    python validate_structure.py --single ts.xyz                  # one structure, no role
Exit code 0 = PASS/WARN, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_io import (ConfigError, atom_map, charge_and_multiplicity,  # noqa: E402
                       fixed_atoms, index_of, load_config, resolve_path, write_json)
from geometry import (bond_set, covalent_cutoffs, distance, distance_matrix,  # noqa: E402
                      fragments, per_atom_displacement, uses_pbc)

DEFAULTS = {
    "duplicate_A": 0.10,
    "min_distance_A": 0.50,
    "short_contact_scale": 0.70,
    "bond_scale": 1.20,
    "isolated_scale": 1.60,
    "allowed_isolated": [],
    "require_elements": [],
    "sn_shell_A": 2.80,
    "endpoint_bond_severity": "error",
    "max_nonreactive_displacement_A": 2.0,
    "fixed_atom_tolerance_A": 1e-3,
}


class ValidationError(RuntimeError):
    """Raised by ``require_valid`` when a report contains errors."""


class Report:
    def __init__(self, label):
        self.label = label
        self.errors, self.warnings, self.info = [], [], {}

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    @property
    def status(self):
        return "FAIL" if self.errors else ("WARN" if self.warnings else "PASS")

    def to_dict(self):
        return {"label": self.label, "status": self.status, "errors": self.errors,
                "warnings": self.warnings, "info": self.info}


def vcfg(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update(cfg.get("validation") or {})
    sysc = cfg.get("system") or {}
    if sysc.get("require_elements") is not None:
        out["require_elements"] = sysc["require_elements"]
    return out


def _name_of(cfg):
    inv = {}
    for k, v in (cfg.get("atom_map") or {}).items():
        if isinstance(v, int) and not isinstance(v, bool):
            inv.setdefault(v, k)
    return inv


# --------------------------------------------------------------------------- #
#  loading
# --------------------------------------------------------------------------- #
def load_structure(path, rep: Report):
    """Read exactly one structure; record problems in ``rep``. Returns Atoms or None."""
    from ase.io import read

    path = Path(path)
    rep.info["file"] = str(path)
    if not path.exists():
        rep.error(f"file does not exist: {path}")
        return None
    try:
        frames = read(str(path), index=":")
    except Exception as e:  # ASE raises many types
        rep.error(f"ASE could not parse {path}: {type(e).__name__}: {e}")
        return None
    if len(frames) == 0:
        rep.error(f"{path} contains no structures")
        return None
    if len(frames) > 1:
        rep.error(f"{path} contains {len(frames)} frames; an endpoint must be a single structure")
        return None
    return frames[0]


# --------------------------------------------------------------------------- #
#  single structure
# --------------------------------------------------------------------------- #
def validate_atoms(atoms, cfg: dict, label="structure", role=None, rep=None) -> Report:
    """Validate one Atoms object. ``role`` in {None, "reactant", "product"}."""
    rep = rep or Report(label)
    v = vcfg(cfg)
    n = len(atoms)
    syms = atoms.get_chemical_symbols()
    rep.info.update({
        "n_atoms": n,
        "formula": atoms.get_chemical_formula(mode="hill") if n else "",
        "element_counts": dict(sorted(Counter(syms).items())),
        "has_Sn": "Sn" in syms,
        "pbc": [bool(x) for x in atoms.pbc],
        "cell_lengths_A": [float(x) for x in atoms.cell.lengths()],
        "cell_volume_A3": float(abs(atoms.cell.volume)),
        "role": role,
    })
    if n == 0:
        rep.error("structure has zero atoms")
        return rep

    exp_n = (cfg.get("system") or {}).get("expected_n_atoms")
    if exp_n is not None and n != exp_n:
        rep.error(f"atom count {n} != system.expected_n_atoms {exp_n}")
    for el in v["require_elements"]:
        if el not in syms:
            rep.error(f"required element {el} is absent (formula {rep.info['formula']})")

    pos = atoms.get_positions()
    if not np.all(np.isfinite(pos)):
        bad = np.where(~np.all(np.isfinite(pos), axis=1))[0].tolist()
        rep.error(f"non-finite coordinates on atom(s) {bad[:20]}")
        return rep
    if np.any(atoms.pbc) and abs(atoms.cell.volume) < 1e-8:
        rep.error("periodic boundary flags set but the cell is degenerate (zero volume)")
        return rep

    _check_charge_spin(atoms, cfg, rep)
    _check_distances(atoms, cfg, v, rep)
    _check_mapping(atoms, cfg, rep)
    _sn_environment(atoms, cfg, v, rep)
    if role in ("reactant", "product") and not rep.info.get("mapping_errors"):
        _check_endpoint_bonds(atoms, cfg, v, role, rep)
    _cv_diagnostics(atoms, cfg, role, rep)
    return rep


def _check_charge_spin(atoms, cfg, rep):
    try:
        q, mult = charge_and_multiplicity(cfg)
    except ConfigError as e:
        rep.error(f"charge/multiplicity not configured: {e}")
        return
    n_elec = int(atoms.numbers.sum()) - q
    rep.info["charge"] = q
    rep.info["spin_multiplicity"] = mult
    rep.info["n_electrons"] = n_elec
    if n_elec < 0:
        rep.error(f"charge {q} leaves a negative electron count")
    elif (n_elec % 2) == (mult % 2):
        # even electrons need an odd multiplicity and vice versa
        rep.error(f"{n_elec} electrons is inconsistent with spin multiplicity {mult} "
                  f"(parity mismatch; check system.charge / spin_multiplicity)")


def _check_distances(atoms, cfg, v, rep):
    n = len(atoms)
    D = distance_matrix(atoms)
    syms = atoms.get_chemical_symbols()
    names = _name_of(cfg)
    lab = lambda i: f"{syms[i]}{i}" + (f"({names[i]})" if i in names else "")
    iu = np.triu_indices(n, 1)
    d = D[iu]
    if d.size:
        k = int(np.argmin(d))
        rep.info["min_distance_A"] = float(d[k])
        rep.info["min_distance_pair"] = [int(iu[0][k]), int(iu[1][k])]
    cov = covalent_cutoffs(atoms.numbers, 1.0)[iu]
    dup = d < v["duplicate_A"]
    clash = (d < v["min_distance_A"]) & ~dup
    short = (d < v["short_contact_scale"] * cov) & ~dup & ~clash
    for mask, kind, fn in ((dup, "duplicate/overlapping atoms", rep.error),
                           (clash, "impossibly short contact", rep.error),
                           (short, "suspiciously short contact", rep.warn)):
        idx = np.where(mask)[0]
        for k in idx[:25]:
            i, j = int(iu[0][k]), int(iu[1][k])
            fn(f"{kind}: {lab(i)}-{lab(j)} = {d[k]:.3f} Å")
        if len(idx) > 25:
            fn(f"... and {len(idx) - 25} more {kind}")

    # isolated atoms: no neighbour within isolated_scale x covalent sum
    C = covalent_cutoffs(atoms.numbers, v["isolated_scale"])
    np.fill_diagonal(C, -1.0)
    has_nb = (D < C).any(axis=1) if n > 1 else np.array([False])
    allowed = {index_of(cfg, r, n) for r in v["allowed_isolated"]}
    iso = [i for i in range(n) if not has_nb[i]]
    rep.info["isolated_atoms"] = iso
    for i in iso:
        if i in allowed:
            rep.warn(f"isolated atom {lab(i)} (allowed by validation.allowed_isolated)")
        else:
            nn = float(np.min(np.delete(D[i], i))) if n > 1 else float("nan")
            rep.error(f"isolated atom {lab(i)}: nearest neighbour {nn:.2f} Å")

    bonds = bond_set(atoms, v["bond_scale"], D)
    frags = fragments(n, bonds)
    rep.info["n_bonds"] = len(bonds)
    rep.info["n_fragments"] = len(frags)
    rep.info["fragment_sizes"] = [len(f) for f in frags]


def _check_mapping(atoms, cfg, rep):
    n = len(atoms)
    syms = atoms.get_chemical_symbols()
    try:
        amap = atom_map(cfg, n)
    except ConfigError as e:
        rep.error(str(e))
        rep.info["mapping_errors"] = True
        return
    rep.info["atom_map"] = {k: {"index": i, "element": syms[i]} for k, i in amap.items()}
    bad = False
    for name, el in (cfg.get("expected_elements") or {}).items():
        if name in amap and syms[amap[name]] != el:
            rep.error(f"reactive atom '{name}' (index {amap[name]}) is {syms[amap[name]]}, "
                      f"expected {el}")
            bad = True
    rep.info["mapping_errors"] = bad
    # reactive distances (diagnostic)
    rd = {}
    keys = list(amap)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            rd[f"{keys[a]}-{keys[b]}"] = round(distance(atoms, amap[keys[a]], amap[keys[b]]), 4)
    rep.info["reactive_distances_A"] = rd


def _sn_environment(atoms, cfg, v, rep):
    syms = atoms.get_chemical_symbols()
    sn = [i for i, s in enumerate(syms) if s == "Sn"]
    if not sn:
        return
    D = distance_matrix(atoms)
    names = _name_of(cfg)
    env = []
    for i in sn:
        order = np.argsort(D[i])
        shell = [int(j) for j in order if j != i and D[i, j] <= v["sn_shell_A"]]
        nbs = [{"index": j, "element": syms[j], "name": names.get(j),
                "distance_A": round(float(D[i, j]), 4)} for j in shell]
        n_o = sum(1 for x in nbs if x["element"] == "O")
        env.append({"Sn_index": i, "shell_cutoff_A": v["sn_shell_A"],
                    "coordination": len(nbs), "n_O": n_o, "neighbours": nbs,
                    "nearest_O_A": next((x["distance_A"] for x in nbs if x["element"] == "O"),
                                        None)})
        if n_o == 0:
            rep.warn(f"Sn{i} has no O neighbour within {v['sn_shell_A']} Å "
                     f"(framework Sn should be O-coordinated)")
    rep.info["Sn_environment"] = env


def _endpoint_rules(cfg, role):
    eb = cfg.get("endpoint_bonds")
    if eb is None:
        return {}
    return eb.get(role) or {}


def _check_endpoint_bonds(atoms, cfg, v, role, rep):
    rules = _endpoint_rules(cfg, role)
    if not rules:
        return
    sev = rep.error if v["endpoint_bond_severity"] == "error" else rep.warn
    C = covalent_cutoffs(atoms.numbers, v["bond_scale"])
    for kind in ("bonded", "not_bonded"):
        for a, b in rules.get(kind, []):
            try:
                i, j = index_of(cfg, a, len(atoms)), index_of(cfg, b, len(atoms))
            except ConfigError as e:
                rep.error(str(e))
                continue
            d = distance(atoms, i, j)
            is_b = d < C[i, j]
            if kind == "bonded" and not is_b:
                sev(f"{role}: expected {a}-{b} bonded but d = {d:.3f} Å "
                    f"(cutoff {C[i, j]:.2f}); wrong index, wrong file, or swapped endpoints?")
            if kind == "not_bonded" and is_b:
                sev(f"{role}: expected {a}-{b} NOT bonded but d = {d:.3f} Å "
                    f"(cutoff {C[i, j]:.2f}); wrong index, wrong file, or swapped endpoints?")


def _cv_diagnostics(atoms, cfg, role, rep):
    if not cfg.get("cvs") or rep.info.get("mapping_errors"):
        return
    from reaction_coordinate import ReactionCoordinates, basin_diagnostics

    try:
        rc = ReactionCoordinates.from_config(cfg, len(atoms), atoms.get_chemical_symbols())
        vals = rc.compute(atoms)
    except ConfigError as e:
        rep.error(f"CV definition problem: {e}")
        return
    rep.info["cvs"] = {k: round(float(x), 5) for k, x in vals.items()}
    bd = basin_diagnostics(vals, cfg)
    if bd:
        rep.info["reference_basins"] = bd
        if role in ("reactant", "product") and bd.get("nearest") and bd["nearest"] != role:
            rep.warn(f"{role}: CVs are nearer the '{bd['nearest']}' reference basin "
                     f"(diagnostic only; check the CV definitions and the structure)")


# --------------------------------------------------------------------------- #
#  pair
# --------------------------------------------------------------------------- #
def compare_endpoints(R, P, cfg, rep: Report):
    """Consistency of a reactant/product pair (same atoms, same order)."""
    v = vcfg(cfg)
    if len(R) != len(P):
        rep.error(f"reactant has {len(R)} atoms, product has {len(P)}")
        return
    sr, sp = R.get_chemical_symbols(), P.get_chemical_symbols()
    mism = [i for i, (a, b) in enumerate(zip(sr, sp)) if a != b]
    if mism:
        ex = ", ".join(f"{i}:{sr[i]}/{sp[i]}" for i in mism[:10])
        rep.error(f"element ordering differs at {len(mism)} position(s) "
                  f"(index:reactant/product) {ex}")
        return
    if [bool(x) for x in R.pbc] != [bool(x) for x in P.pbc]:
        rep.error("reactant and product have different PBC flags")
    if uses_pbc(R) and not np.allclose(R.cell.array, P.cell.array, atol=1e-4):
        rep.error("reactant and product have different cells")

    n = len(R)
    names = _name_of(cfg)
    try:
        amap = atom_map(cfg, n)
    except ConfigError:
        amap = {}
    reactive = set(amap.values())
    fixed = []
    try:
        fixed = fixed_atoms(cfg, n)
    except ConfigError as e:
        rep.error(str(e))

    # topology change
    br, bp = bond_set(R, v["bond_scale"]), bond_set(P, v["bond_scale"])
    broken, formed = sorted(br - bp), sorted(bp - br)
    lab = lambda i: f"{sr[i]}{i}" + (f"({names[i]})" if i in names else "")
    rep.info["bonds_broken"] = [[lab(i), lab(j)] for i, j in broken]
    rep.info["bonds_formed"] = [[lab(i), lab(j)] for i, j in formed]
    expected = set()
    for kind in ("break", "form"):
        for a, b in (cfg.get("expected_bond_changes") or {}).get(kind, []):
            try:
                i, j = sorted((index_of(cfg, a, n), index_of(cfg, b, n)))
                expected.add((kind, i, j))
            except ConfigError:
                pass
    for kind, lst in (("break", broken), ("form", formed)):
        for i, j in lst:
            if (kind, i, j) not in expected:
                rep.warn(f"unexpected bond {'broken' if kind == 'break' else 'formed'} "
                         f"between endpoints: {lab(i)}-{lab(j)}")
    for kind, i, j in sorted(expected):
        lst = broken if kind == "break" else formed
        if (i, j) not in lst:
            rep.warn(f"expected bond to {kind} ({lab(i)}-{lab(j)}) does not change between endpoints")

    # displacement
    disp = per_atom_displacement(P, R, align=not fixed)
    rep.info["endpoint_rmsd_A"] = float(np.sqrt(np.mean(disp ** 2)))
    k = int(np.argmax(disp))
    rep.info["max_displacement"] = {"atom": lab(k), "A": float(disp[k])}
    nonreact = [i for i in range(n) if i not in reactive]
    big = [i for i in nonreact if disp[i] > v["max_nonreactive_displacement_A"]]
    for i in big[:20]:
        rep.warn(f"non-reactive atom {lab(i)} moves {disp[i]:.2f} Å between endpoints "
                 f"(possible atom permutation or different conformer)")
    if fixed:
        dfix = np.linalg.norm(R.positions[fixed] - P.positions[fixed], axis=1)
        if np.max(dfix) > v["fixed_atom_tolerance_A"]:
            rep.warn(f"fixed atoms differ between endpoints by up to {np.max(dfix):.4f} Å")


def validate_pair(reactant, product, cfg: dict) -> dict:
    """Validate a reactant/product pair (paths or Atoms). Returns a JSON-able dict
    with per-structure reports and a pair report; ``status`` is the worst."""
    reps, atoms = {}, {}
    for role, src in (("reactant", reactant), ("product", product)):
        rep = Report(role)
        a = load_structure(src, rep) if isinstance(src, (str, Path)) else src
        if a is not None:
            validate_atoms(a, cfg, role, role, rep)
        reps[role], atoms[role] = rep, a
    pair = Report("pair")
    if atoms["reactant"] is not None and atoms["product"] is not None:
        compare_endpoints(atoms["reactant"], atoms["product"], cfg, pair)
    else:
        pair.error("cannot compare endpoints: at least one failed to load")
    out = {"reactant": reps["reactant"].to_dict(), "product": reps["product"].to_dict(),
           "pair": pair.to_dict()}
    out["status"] = _worst([r["status"] for r in out.values() if isinstance(r, dict)])
    out["n_errors"] = sum(len(r["errors"]) for k, r in out.items() if isinstance(r, dict))
    out["n_warnings"] = sum(len(r["warnings"]) for k, r in out.items() if isinstance(r, dict))
    return out


def validate_single(src, cfg: dict, label="structure", role=None) -> dict:
    rep = Report(label)
    a = load_structure(src, rep) if isinstance(src, (str, Path)) else src
    if a is not None:
        validate_atoms(a, cfg, label, role, rep)
    d = rep.to_dict()
    d["n_errors"], d["n_warnings"] = len(d["errors"]), len(d["warnings"])
    return d


def _worst(statuses):
    for s in ("FAIL", "WARN"):
        if s in statuses:
            return s
    return "PASS"


def require_valid(report: dict, what="input") -> None:
    """Raise ``ValidationError`` (after printing) if the report has errors."""
    if report["status"] == "FAIL":
        print_report(report)
        raise ValidationError(f"{what} failed validation with {report['n_errors']} error(s); "
                              "refusing to start the calculation")


def print_report(report: dict) -> None:
    parts = [report] if "label" in report else [report[k] for k in ("reactant", "product", "pair")]
    for r in parts:
        i = r["info"]
        head = f"[{r['status']}] {r['label']}"
        if "formula" in i:
            head += f": {i.get('file', '(in memory)')}  {i['formula']}  N={i['n_atoms']}"
        print(head)
        if "min_distance_A" in i:
            print(f"    min distance {i['min_distance_A']:.3f} Å  fragments {i.get('n_fragments')}"
                  f"  Sn present: {i.get('has_Sn')}")
        if i.get("cvs"):
            print("    CVs: " + "  ".join(f"{k}={v:.3f}" for k, v in i["cvs"].items()))
        for env in i.get("Sn_environment", []):
            nb = ", ".join(f"{x['element']}{x['index']}@{x['distance_A']:.2f}"
                           for x in env["neighbours"][:8])
            print(f"    Sn{env['Sn_index']}: CN={env['coordination']} (O: {env['n_O']}) {nb}")
        if r["label"] == "pair" and "endpoint_rmsd_A" in i:
            print(f"    endpoint RMSD {i['endpoint_rmsd_A']:.3f} Å; max displacement "
                  f"{i['max_displacement']['atom']} {i['max_displacement']['A']:.2f} Å")
            print(f"    bonds broken {i['bonds_broken']}  formed {i['bonds_formed']}")
        for e in r["errors"]:
            print(f"    ERROR: {e}")
        for w in r["warnings"]:
            print(f"    warning: {w}")
    if "status" in report and "label" not in report:
        print(f"==> overall: {report['status']} ({report['n_errors']} errors, "
              f"{report['n_warnings']} warnings)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Strict structure validation (Sn-beta hydride benchmark)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--reactant", default=None)
    ap.add_argument("--product", default=None)
    ap.add_argument("--single", default=None, help="validate one structure (no endpoint role)")
    ap.add_argument("--role", default=None, choices=["reactant", "product"],
                    help="with --single: apply that endpoint's bond expectations")
    ap.add_argument("--json", default=None, help="write the machine-readable report here")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.single:
        report = validate_single(args.single, cfg, Path(args.single).name, args.role)
    else:
        st = cfg.get("structures") or {}
        r = args.reactant or (resolve_path(cfg, st["reactant"]) if st.get("reactant") else None)
        p = args.product or (resolve_path(cfg, st["product"]) if st.get("product") else None)
        if r is None or p is None:
            print("no reactant/product given (config.structures or --reactant/--product)")
            return 1
        report = validate_pair(r, p, cfg)
    print_report(report)
    if args.json:
        write_json(args.json, report)
        print(f"report -> {args.json}")
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
