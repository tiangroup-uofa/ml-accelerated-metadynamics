#!/usr/bin/env python3
"""
build_guess.py
==============
Hand-built STARTING GUESS for the reactant of the Sn-beta C2->C1 hydride
transfer, reproduced from Mushrif, Varghese, Krishnamurthy, PCCP 17, 4961
(2015), Fig. 1b/c. The paper gives no coordinates; this is a geometric build,
to be relaxed with MACE-POLAR (relax_polar.py).

Model (Fig. 1b/c): open-chain D-glucose deprotonated at O2 (C2-O-Sn covalent),
C1=O coordinated to Sn, the water formed from Sn-OH + the O2 proton still on
Sn, and three O-SiH3 caps: Sn(OSiH3)3(H2O)(O2)(O1), octahedral, 42 atoms,
neutral (Sn(IV) + 4 anionic O), closed shell (inferred, not stated).

Needs RDKit; run with an ephemeral env, e.g.
    uv run --with rdkit --with numpy python build_guess.py
"""
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdCIPLabeler, rdMolTransforms

# Keep archived structures untouched; all rerun files live in a separate workspace.
import os
import shutil

ARCHIVE = Path(__file__).resolve().parent.parent
HERE = Path(os.environ.get("GLUCOSE_WORKDIR", str(ARCHIVE / "runs" / "polar-1-s"))).resolve()
if HERE == ARCHIVE or ARCHIVE in HERE.parents and "runs" not in HERE.relative_to(ARCHIVE).parts:
    raise ValueError("Use a separate workspace, not the structure archive or scripts directory")
HERE.mkdir(parents=True, exist_ok=True)
for pattern in ("*.xyz", "*.json", "*.csv"):
    for source in ARCHIVE.glob(pattern):
        target = HERE / source.name
        if not target.exists():
            shutil.copy2(source, target)
# open-chain D-glucose, C2-O deprotonated. SMILES order: O1 C1 C2 O2 C3 O3 C4 O4 C5 O5 C6 O6
SMILES = "O=C[C@H]([O-])[C@@H](O)[C@H](O)[C@H](O)CO"
O1, C1, C2, O2 = 0, 1, 2, 3
N_CONF, SEED = 300, 20150108

# target bond lengths (A) / angles for the Sn cluster
SN_O2, SN_O1 = 2.05, 2.25          # covalent alkoxide, dative carbonyl
SN_OSI, O_SI, SI_H, SN_O_SI = 1.95, 1.63, 1.48, 140.0
SN_OW, OW_H, SN_OW_H = 2.20, 0.97, 127.5


def unit(v):
    return v / np.linalg.norm(v)


def perp_basis(a):
    """Two unit vectors orthogonal to unit vector a."""
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    q = unit(np.cross(a, t))
    return q, np.cross(a, q)


def place_sn(X):
    """Sn in the O1-C1-C2-O2 plane, closing a 5-ring, away from the carbons."""
    o1, o2, c1, c2 = X[O1], X[O2], X[C1], X[C2]
    d = np.linalg.norm(o1 - o2)
    u = unit(o1 - o2)
    x = (SN_O2**2 - SN_O1**2 + d**2) / (2 * d)
    h = np.sqrt(max(SN_O2**2 - x**2, 0.0))
    away = (o1 + o2) / 2 - (c1 + c2) / 2
    v = unit(away - np.dot(away, u) * u)
    return o2 + x * u + h * v


def osih3(sn, s, phi):
    """O-SiH3 on site direction s; phi rotates the Si about the Sn-O axis."""
    o = sn + SN_OSI * s
    q, r = perp_basis(s)
    w = np.cos(phi) * q + np.sin(phi) * r
    b = np.radians(180.0 - SN_O_SI)
    si = o + O_SI * (np.cos(b) * s + np.sin(b) * w)
    a = unit(o - si)                        # Si -> O
    q, r = perp_basis(a)
    hs = []
    for k in range(3):
        ang = np.radians(60.0 + 120.0 * k) + phi
        dvec = np.cos(np.radians(109.47)) * a + np.sin(np.radians(109.47)) * (
            np.cos(ang) * q + np.sin(ang) * r)
        hs.append(si + SI_H * dvec)
    return [("O", o), ("Si", si)] + [("H", h) for h in hs]


def water(sn, s, phi):
    o = sn + SN_OW * s
    q, r = perp_basis(s)
    w = np.cos(phi) * q + np.sin(phi) * r
    b = np.radians(180.0 - SN_OW_H)
    return [("O", o)] + [("H", o + OW_H * (np.cos(b) * s + sg * np.sin(b) * w))
                         for sg in (1, -1)]


def build_cluster(X):
    sn = place_sn(X)
    e1, e2 = unit(X[O1] - sn), unit(X[O2] - sn)
    n = unit(np.cross(e1, e2))
    # remaining octahedral sites: trans to O1 (water), trans to O2, and the two axial
    sites = {"water": -e1, "trans_O2": -e2, "ax+": n, "ax-": -n}
    phis = np.radians(np.arange(0, 360, 30))
    best = None
    for pw in phis:
        for p1 in phis:
            for p2 in phis:
                for p3 in phis:
                    lig = (water(sn, sites["water"], pw) + osih3(sn, sites["trans_O2"], p1)
                           + osih3(sn, sites["ax+"], p2) + osih3(sn, sites["ax-"], p3))
                    L = np.array([p for _, p in lig])
                    dmin = np.min(np.linalg.norm(L[:, None] - X[None], axis=-1))
                    ll = np.linalg.norm(L[:, None] - L[None], axis=-1)
                    # only non-bonded ligand-ligand pairs (different groups)
                    grp = np.repeat([0, 1, 2, 3], [3, 5, 5, 5])
                    lmin = np.min(ll[grp[:, None] != grp[None]])
                    score = min(dmin, lmin)
                    if best is None or score > best[0]:
                        best = (score, sn, lig)
    return best


def main():
    # check D-glucose on the neutral parent (same chiral tags): removing the O2
    # proton changes CIP priorities at C3, not the spatial configuration
    ref = Chem.MolFromSmiles(SMILES.replace("[O-]", "O"))
    rdCIPLabeler.AssignCIPLabels(ref)
    cip = {f"C{k}": ref.GetAtomWithIdx(i).GetProp("_CIPCode")
           for k, i in ((2, 2), (3, 4), (4, 6), (5, 8))}
    assert cip == {"C2": "R", "C3": "S", "C4": "R", "C5": "R"}, cip   # D-glucose
    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
    h_c1 = [a.GetIdx() for a in mol.GetAtomWithIdx(C1).GetNeighbors() if a.GetSymbol() == "H"][0]
    h_c2 = [a.GetIdx() for a in mol.GetAtomWithIdx(C2).GetNeighbors() if a.GetSymbol() == "H"][0]

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONF, randomSeed=SEED)
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    cands = []
    for cid, (_, e) in zip(cids, res):
        conf = mol.GetConformer(cid)
        # syn chelate: rotate the C1 side (O1, H1) to dihedral 0
        rdMolTransforms.SetDihedralDeg(conf, O2, C2, C1, O1, 0.0)
        X = conf.GetPositions()
        cands.append((e, cid, X))
    cands.sort(key=lambda t: t[0])
    e0 = cands[0][0]

    out = []
    for e, cid, X in cands[:40]:            # lowest-MMFF conformers only
        score, sn, lig = build_cluster(X)
        out.append((score, e - e0, cid, X, sn, lig))
        print(f"conf {cid:3d}  dE_MMFF {e - e0:6.1f} kcal/mol  min contact {score:.2f} A")
    out.sort(key=lambda t: -t[0])
    score, de, cid, X, sn, lig = out[0]
    print(f"picked conf {cid}: min contact {score:.2f} A, dE_MMFF {de:.1f} kcal/mol")

    symbols = sym + ["Sn"] + [s for s, _ in lig]
    pos = np.vstack([X, sn[None], np.array([p for _, p in lig])])
    n_glc = len(sym)
    amap = {"C1": C1, "C2": C2, "H_transfer": h_c2, "H_C1": h_c1, "Sn": n_glc,
            "O1": O1, "O2": O2, "O_water": n_glc + 1}
    with open(HERE / "reactant_guess.xyz", "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write("Sn-beta glucose hydride transfer REACTANT, hand-built guess (unrelaxed); "
                "Mushrif et al. PCCP 2015 Fig. 1b\n")
        for s, p in zip(symbols, pos):
            f.write(f"{s:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
    (HERE / "atom_map.json").write_text(json.dumps(
        {"_comment": "zero-based indices, same order in reactant and product",
         "atom_map": amap, "n_atoms": len(symbols), "charge": 0, "spin_multiplicity": 1,
         "rdkit_conformer": cid, "min_contact_A": round(float(score), 3)}, indent=2) + "\n")
    print(f"wrote reactant_guess.xyz ({len(symbols)} atoms), atom_map.json: {amap}")


if __name__ == "__main__":
    main()
