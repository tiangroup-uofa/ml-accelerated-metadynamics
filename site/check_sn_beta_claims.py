#!/usr/bin/env python3
"""
check_sn_beta_claims.py — verify the public Sn-β Current work page against its evidence.

    python3 site/check_sn_beta_claims.py            # source + rendered HTML + stale wording
    python3 site/check_sn_beta_claims.py --links    # also check internal links, assets, anchors

Every published number is DERIVED here from the merged benchmark data
(sweep/sn_beta_hydride/audit/omol_benchmark/*.json incl. cross_evaluation.json, which
sweep/sn_beta_hydride/cross_evaluate.py generates, and the input structures) and must then appear in
site/current-work.qmd AND in docs/current-work.html. Bar widths, gauge positions and SVG marker
coordinates are recomputed from the same values. Pre-benchmark wording that must not survive
anywhere on the site is listed in STALE. Exit status 1 on any failure. Needs numpy-free Python
plus ASE (for the Sn–O distances); run it with the interpreter that has ASE.
"""
from __future__ import annotations

import html
import json
import math
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

from ase.io import read

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SN = REPO / "sweep" / "sn_beta_hydride"
AUD = SN / "audit" / "omol_benchmark"
DOCS = REPO / "docs"
QMD = HERE / "current-work.qmd"
HTML = DOCS / "current-work.html"
MINUS = "−"
BAR_SCALE = 140.0


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def f(v, nd):
    s = f"{abs(v):.{nd}f}"
    return (MINUS + s) if v < 0 else s


def width(v, scale=BAR_SCALE):
    return f"{abs(v) / scale * 100:.1f}"


def text_of(path):
    t = Path(path).read_text(encoding="utf-8")
    t = re.sub(r"<(script|style)\b.*?</\1>", " ", t, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", t))


# --------------------------------------------------------------------------- #
#  evidence
# --------------------------------------------------------------------------- #
cmpj = J(AUD / "comparison_omol_vs_polar.json")
runs = J(AUD / "reassessment_corrected_rules.json")
omol, omol4, polar = runs[0], runs[1], runs[2]
xtb = J(AUD / "xtb_sanity__xtb_sanity.json")
en = cmpj["energetics_kJ_mol"]
O, P = en["OMOL"], en["POLAR (Tian's run)"]
RE, CS = "relaxed_endpoint", "connected_side_minimum"
om_vt = J(AUD / "omol_1024_full_protocol" / "vib" / "validate_ts.json")
m4_vt = J(AUD / "omol_4M_ladder" / "vib" / "validate_ts.json")
om_path = J(AUD / "omol_1024_full_protocol" / "path" / "path.json")
om_ref = J(AUD / "omol_1024_full_protocol" / "ts" / "refine_ts.json")
seed = J(AUD / "phase3_seed_from_polar_ts__refine_ts.json")
rep = (SN / "OMOL_BENCHMARK_REPORT.md").read_text(encoding="utf-8")

claims: list[tuple[str, str]] = []
regexes: list[tuple[str, str]] = []


def need(label, needle):
    claims.append((label, needle))


def bar(label, val_str, w, kind=None):
    regexes.append((label, rf'<span class="val">{re.escape(val_str)}(?:<small>[^<]*</small>)?</span>'
                           rf'<span class="bar {kind or "(?:omol|polar)"}[^"]*" style="--w:{w}%"'))


# ---- scorecard values and bars
for nm, d in (("OMOL", O), ("POLAR", P)):
    k = "omol" if nm == "OMOL" else "polar"
    for ref in (RE, CS):
        v = d["barrier_fwd"][ref]; need(f"{nm} barrier {ref}", f(v, 2)); bar(f"{nm} barrier bar {ref}", f(v, 2), width(v), k)
        e = d["dE"][ref]; need(f"{nm} dE {ref}", f(e, 2)); bar(f"{nm} dE bar {ref}", f(e, 2), width(e), k)
    off = d["R_relaxed_minus_R_prime"]; need(f"{nm} conformer offset", f(off, 1)); bar(f"{nm} offset bar", f(off, 1), width(off), k)
for ref in (RE, CS):
    need(f"OMOL-POLAR {ref}", f(en["OMOL_minus_POLAR"]["barrier_fwd"][ref], 1))
need("headline spread 36–41", "36–41")
need("R' offset POLAR 13.44", f(P["R_relaxed_minus_R_prime"], 2))
need("barrier shift 13.4→", f(P["barrier_fwd"][CS] - P["barrier_fwd"][RE], 1))
need("barrier shift 18.4→", f(O["barrier_fwd"][CS] - O["barrier_fwd"][RE], 1))

# ---- reference-conformer dumbbell (positions on the 0–140 kJ/mol axis, shifts, labels)
for nm, d in (("POLAR", P), ("OMOL", O)):
    a, b = d["barrier_fwd"][RE], d["barrier_fwd"][CS]
    need(f"dumbbell {nm} segment", f'--a:{width(a)}%;--b:{width(b)}%')
    need(f"dumbbell {nm} ep dot", f'style="--x:{width(a)}%" data-v="{a:.2f}"')
    need(f"dumbbell {nm} cs dot", f'style="--x:{width(b)}%" data-v="{b:.2f}"')
    need(f"dumbbell {nm} shift", f'<span class="dlab">+{d["R_relaxed_minus_R_prime"]:.1f}')

# ---- saddles
need("OMOL nu", f(omol["nu_imag_cm"], 0)); need("POLAR nu", f(-923.09, 0))
need("POLAR re-run nu", f(polar["nu_imag_cm"], 2))
need("OMOL overlap", f(omol["hydride_overlap"], 2)); need("POLAR overlap", f(polar["hydride_overlap"], 2))
need("TS RMSD whole", f(cmpj["conformer_offsets"]["OMOL TS vs POLAR TS"]["rmsd_whole_A"], 2))
need("TS RMSD core", f(cmpj["conformer_offsets"]["OMOL TS vs POLAR TS"]["rmsd_core_A"], 3))
need("R-R' core POLAR", f(cmpj["conformer_offsets"]["POLAR R' vs POLAR R"]["rmsd_core_A"], 3))
need("R-R' core OMOL", f(cmpj["conformer_offsets"]["OMOL R' vs OMOL R"]["rmsd_core_A"], 3))
for lab, key in (("POLAR", "POLAR R' vs POLAR R"), ("OMOL", "OMOL R' vs OMOL R")):
    need(f"{lab} torsion", f(round(cmpj["conformer_offsets"][key]["largest_dihedral"]["change_deg"]), 0) + "°")
    assert cmpj["conformer_offsets"][key]["largest_dihedral"]["label"] == "C1-C2-C4-O5"

# ---- statuses
for r in runs:
    assert r["statuses"]["saddle"] == "CERTIFIED", r["run"]
assert omol["statuses"]["neb_path"] == "UNCONVERGED_DIAGNOSTIC_ONLY" and polar["statuses"]["reproduced"] == "REPRODUCED"
assert max(r["max_drift_vs_saved_kJ_mol"] for r in runs) == 0.0
assert all(set(s["side_coordination"].values()) == {"PRESERVED"} for s in runs)

# ---- NEB lane and Sella lane (+ gauge geometry)
pr = omol["ts_provenance"]
need("NEB fmax", f(pr["neb_fmax_final"], 3)); need("NEB target", f(pr["neb_fmax_target"], 2)); need("NEB steps", str(pr["neb_steps"]))
assert om_path["converged"] is False and om_path["steps"] == pr["neb_steps"]
need("Sella steps", f"in {omol['sella_nsteps']} steps")
sf = omol["stationary_fmax_eV_A"]; need("Sella fmax", f"{sf * 1e4:.1f}×10⁻⁴")
need("POLAR Sella fmax", f"{polar['stationary_fmax_eV_A'] * 1e4:.1f}×10⁻⁴"); assert polar["sella_nsteps"] == 0
lg = lambda x, lo, hi: (math.log10(x) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * 100
need("NEB gauge fill", f"--w:{lg(pr['neb_fmax_final'], 0.01, 1):.1f}%"); need("NEB gauge tick", f"--x:{lg(pr['neb_fmax_target'], 0.01, 1):.1f}%")
need("Sella gauge fill", f"--w:{lg(sf, 1e-4, 1):.1f}%"); need("Sella gauge tick", f"--x:{lg(1e-3, 1e-4, 1):.1f}%")
assert abs(seed["final"]["energy_eV"] - om_ref["final"]["energy_eV"]) < 1e-6        # independent seed -> same saddle
need("seed RMSD 0.001", "RMSD 0.001")
assert "0.001" in rep

# ---- same-geometry cross-check: read from the COMMITTED, script-generated artifact
#      (bars are sized from the DISPLAYED one-decimal value, as on the page)
#      (sweep/sn_beta_hydride/cross_evaluate.py -> audit/omol_benchmark/cross_evaluation.json)
xe = J(AUD / "cross_evaluation.json")
xd = xe["single_point_differences_kJ_mol"]
o_at_p = [xd["omol_at_polar_geometries"]["TS-R"], xd["omol_at_polar_geometries"]["TS-R'"]]
o_own = [xd["omol_at_omol_geometries"]["TS-R"], xd["omol_at_omol_geometries"]["TS-R'"]]
p_own = [xd["polar_at_polar_geometries"]["TS-R"], xd["polar_at_polar_geometries"]["TS-R'"]]
p_at_o = [xd["polar_at_omol_geometries"]["TS-R"], xd["polar_at_omol_geometries"]["TS-R'"]]
# own-geometry single points must equal the relaxed barriers of the scorecard (same structures, same settings)
assert abs(p_own[0] - P["barrier_fwd"][RE]) < 1e-3 and abs(p_own[1] - P["barrier_fwd"][CS]) < 1e-3
assert abs(o_own[0] - O["barrier_fwd"][RE]) < 1e-3 and abs(o_own[1] - O["barrier_fwd"][CS]) < 1e-3
gaps = []
for i, lab in enumerate(("TS−R", "TS−R′")):
    for who, pv, ov in (("POLAR structures", p_own[i], o_at_p[i]), ("OMOL structures", p_at_o[i], o_own[i])):
        need(f"geo POLAR {lab} {who}", f"<span class=\"v\">{pv:.1f}</span><span class=\"bar polar\" style=\"--w:{width(round(pv, 1))}%\"")
        need(f"geo OMOL {lab} {who}", f"<span class=\"v\">{ov:.1f}</span><span class=\"bar omol\" style=\"--w:{width(round(ov, 1))}%\"")
        g = round(pv - ov); gaps.append(g); need(f"geo gap {lab} {who}", f"<strong>{g}</strong> kJ/mol")
need("gap range", f"{min(gaps)} to {max(gaps)} kJ/mol")
need("cross-eval source", "audit/omol_benchmark/cross_evaluation.json")

# ---- 4M checkpoint
e4 = m4_vt["energetics"]
d_re = e4["barrier_forward"][RE]["kJ_mol"] - O["barrier_fwd"][RE]
d_cs = e4["barrier_forward"][CS]["kJ_mol"] - O["barrier_fwd"][CS]
need("4M shift RE", "+" + f(d_re, 2)); need("4M shift CS", MINUS + f(-d_cs, 2).lstrip(MINUS))
need("4M dE old", f(O["dE"][RE], 2)); need("4M dE new", f(e4["reaction_energy"][RE]["kJ_mol"], 2))
need("4M dE shift", f(e4["reaction_energy"][RE]["kJ_mol"] - O["dE"][RE], 1))

# ---- xTB distances, track coordinates and verdicts
kr = xtb["roles"]["reactant"]["final"]["key_sn_o"]; kp = xtb["roles"]["product"]["final"]["key_sn_o"]
assert xtb["roles"]["reactant"]["final"]["verdict"] == "STRUCTURALLY_INVALID" and xtb["roles"]["product"]["final"]["verdict"] == "PRESERVED"
assert "STRUCTURALLY INVALID FOR BARRIER COMPARISON" == xtb["status"]
X = lambda d: 70 + (d - 2.0) * 540 / 1.4
need("xTB reactant ref", f"input {kr['O_water']['ref_A']:.2f} Å"); need("xTB reactant now", f"xTB-relaxed {kr['O_water']['now_A']:.2f} Å")
need("xTB product ref", f"input {kp['O_water']['ref_A']:.2f} Å"); need("xTB product now", f"xTB-relaxed {kp['O_water']['now_A']:.2f} Å")
need("xTB O1", f"{kr['O1']['ref_A']:.2f} → {kr['O1']['now_A']:.2f} Å")
for tag, k in (("r", kr), ("p", kp)):
    a, b = X(round(k["O_water"]["ref_A"], 2)), X(round(k["O_water"]["now_A"], 2))
    need(f"svg {tag} input x", f"cx=\"{a:.1f}\""); need(f"svg {tag} shift", f"translateX({b - a:.1f}px)")
need("svg cutoff", f"x1=\"{X(2.8):.1f}\"")

# ---- input-structure Sn–O facts
cfg = J(SN / "config_tian_tt_test.json"); m = cfg["atom_map"]
for role in ("reactant", "product"):
    at = read(SN / "input" / "tian_tt_test" / f"{role}.xyz")
    d = at.get_distances(m["Sn"], [i for i in range(len(at)) if at.numbers[i] == 8])
    assert sum(x <= 2.80 for x in d) == 6, role
    assert sum(x <= 2.46 for x in d) == (6 if role == "reactant" else 5), (role, sorted(d)[:7])
    if role == "reactant":
        need("reactant Sn-O1/O2", f"Sn–O1 {at.get_distance(m['Sn'], m['O1']):.2f} → ")
    else:
        need("product Sn-O1/O2", f"{at.get_distance(m['Sn'], m['O1']):.2f} Å, Sn–O2 2.04 → {at.get_distance(m['Sn'], m['O2']):.2f} Å")
        need("product 2.47 beyond 2.46", "2.46 Å bond criterion")

# --------------------------------------------------------------------------- #
#  stale / forbidden wording, site-wide
# --------------------------------------------------------------------------- #
STALE = [
    "Not re-certified", "not re-certified", "Not yet run", "not yet run", "others pending",
    "awaiting standardized", "has **not** independently recomputed", "not independently certified",
    "No stage has been run", "Why no method comparison", "six O around Sn", "no results yet",
    "No results yet", "in preparation", "waiting on authoritative", "no certified result",
    "Pipeline built and tested; waiting", "Sn-β infrastructure (no results)", "no benchmark result yet",
    "Next experiments", "has not been independently re-certified", "not yet in hand",
]
SITE_SOURCES = ["current-work.qmd", "index.qmd", "methods.qmd", "research-map.qmd", "research-map.json",
                "research-map.svg", "_quarto.yml", "README.md"]
SITE_RENDERED = ["current-work.html", "index.html", "methods.html", "research-map.html"]

fails: list[str] = []


def check_text(name, blob, raw=None):
    raw = raw if raw is not None else blob
    for label, needle in claims:
        if needle not in raw and needle not in blob:
            fails.append(f"[{name}] missing claim '{label}': {needle!r}")
    for label, pat in regexes:
        if not re.search(pat, raw):
            fails.append(f"[{name}] bar/value mismatch '{label}': {pat[:90]}")


qmd = QMD.read_text(encoding="utf-8")
check_text("qmd", qmd)
if HTML.exists():
    raw_html = HTML.read_text(encoding="utf-8")
    check_text("html", text_of(HTML), raw_html)
else:
    fails.append(f"missing rendered page {HTML} (run quarto render)")

for fn in SITE_SOURCES:
    t = (HERE / fn).read_text(encoding="utf-8")
    for s in STALE:
        if s in t:
            fails.append(f"[stale in {fn}] {s!r}")
for fn in SITE_RENDERED:
    if (DOCS / fn).exists():
        t = text_of(DOCS / fn)
        for s in STALE:
            if s in t:
                fails.append(f"[stale in docs/{fn}] {s!r}")
for fn in ("search.json",):
    if (DOCS / fn).exists():
        t = (DOCS / fn).read_text(encoding="utf-8")
        for s in ("Not re-certified", "not re-certified", "Why no method comparison", "Next experiments"):
            if s in t:
                fails.append(f"[stale in docs/{fn}] {s!r}")

# --------------------------------------------------------------------------- #
#  optional: internal links, assets, anchors in the rendered site
# --------------------------------------------------------------------------- #
class Collect(HTMLParser):
    def __init__(self):
        super().__init__(); self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"): self.ids.add(a["id"])
        if tag == "a" and a.get("name"): self.ids.add(a["name"])
        for k in ("href", "src", "srcset"):
            if a.get(k): self.links.append((tag, k, a[k]))


if "--links" in sys.argv:
    pages = {p.name: Collect() for p in DOCS.glob("*.html")}
    for p in DOCS.glob("*.html"):
        pages[p.name].feed(p.read_text(encoding="utf-8"))
    n = 0
    for name in ("current-work.html", "index.html", "methods.html", "research-map.html"):
        c = pages[name]
        for tag, k, v in c.links:
            for u in ([x.strip().split(" ")[0] for x in v.split(",")] if k == "srcset" else [v]):
                if re.match(r"(https?:|mailto:|data:|javascript:)", u) or not u:
                    continue
                n += 1
                path, _, frag = u.partition("#")
                tgt = (DOCS / path) if path else DOCS / name
                if path and not tgt.exists():
                    fails.append(f"[link {name}] missing target {u}")
                elif frag:
                    tname = Path(path).name if path else name
                    if tname.endswith(".html") and tname in pages and frag not in pages[tname].ids:
                        fails.append(f"[link {name}] missing anchor {u}")
    print(f"checked {n} internal links/assets in 4 pages")
    for need_ in ("assets/sn_beta_reaction.png", "assets/sn_beta_reaction_stacked.png",
                  "assets/sn_beta_energy_profile.png", "assets/sn_beta_structure_views.png"):
        if not (DOCS / need_).exists():
            fails.append(f"missing figure docs/{need_}")

print(f"{len(claims)} number claims + {len(regexes)} bar/value pairs checked against qmd and html; "
      f"{len(STALE)} stale phrases scanned across {len(SITE_SOURCES)} sources and {len(SITE_RENDERED)} rendered pages")
if fails:
    print(f"\nFAILED ({len(fails)}):")
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("OK: every published Sn-β number matches the evidence; no stale wording found")
