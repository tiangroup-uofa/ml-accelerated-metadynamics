"""
Sn-site coordination integrity relative to a reference structure.

Why relative: the RECONSTRUCTED input product already has Sn-O2 = 2.47 Å (five O
inside 2.46 Å, six inside 2.80 Å) -- the reaction itself moves O2 out of the
inner shell. An absolute "six Sn-O < 2.46 Å" test would reject the reference. So
a method's relaxed structure is judged against the INPUT structure of the SAME
role (reactant vs input reactant, product vs input product), with one rule applied
identically to every method (MACE-OMOL, MACE-POLAR, GFN2-xTB).

REFERENCE CONVENTION (single rule, used everywhere): the reference is the INPUT structure of the
same role (reactant/product), not a model-relaxed one -- the same reference for every method
and for connectivity side-minima (connectivity.side_coordination), xtb_sanity and
compare_methods. The metadynamics monitor must use it too.

Verdicts
--------
STRUCTURALLY_INVALID  at least one of:
    * a heavy shell atom (Z>1, within `shell_A` of Sn in the reference) is outside the shell
      (ligand dissociation, e.g. the bound water leaving);
    * a heavy atom entered the shell that was not in it (coordination-number change);
    * a heavy-atom covalent bond away from Sn broke or formed (Si-O cap cleavage, C-O/C-C
      scission) or a hydrogen came within SN_H_CLOSE_A of Sn (hydride on Sn);
    * the bound water lost its integrity (O_water no longer has exactly two H);
    * the hydride state differs from the expected one (reactant relaxed into
      product-like or vice versa, or detached).
WEAKENED              no invalid condition, but a Sn-O distance lengthened by
                      more than `weaken_A` (an ANALYSIS threshold, not a physical law).
PRESERVED             none of the above.

Thresholds are analysis parameters; they are recorded in every result.
"""
from __future__ import annotations

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from config_io import atom_map
from hydride import classify_hydride_state

SHELL_A = 2.80
WEAKEN_A = 0.30
HX_BOND_SCALE = 1.2
SN_H_CLOSE_A = 2.0


def sn_shell(atoms, sn_index: int, shell_A: float = SHELL_A) -> dict:
    """{atom index: Sn distance} for every HEAVY (Z>1) atom within shell_A of Sn.

    Hydrogens are excluded on purpose: a ligand water rotating about its Sn-O axis
    moves its H atoms across any fixed Sn-H sphere without changing the coordination
    (found when the first draft flagged an intact xTB product water). Water
    integrity is checked separately via `water_integrity`."""
    pos = atoms.get_positions()
    d = np.linalg.norm(pos - pos[sn_index], axis=1)
    return {int(i): float(d[i]) for i in np.where(d <= shell_A)[0]
            if i != sn_index and atoms.numbers[i] > 1}


def heavy_bonds(atoms, sn_index: int, scale: float = HX_BOND_SCALE, shell_A: float = SHELL_A) -> set:
    """Covalent graph over heavy atoms (Z>1). Pairs involving Sn use the Sn shell cutoff
    (Sn-O bonds are long and soft); all other pairs use scale x (r_i + r_j)."""
    pos, nums = atoms.get_positions(), atoms.numbers
    heavy = [i for i in range(len(atoms)) if nums[i] > 1]
    out = set()
    for a_ in range(len(heavy)):
        for b_ in range(a_ + 1, len(heavy)):
            i, j = heavy[a_], heavy[b_]
            d = float(np.linalg.norm(pos[i] - pos[j]))
            cut = shell_A if sn_index in (i, j) else scale * (covalent_radii[nums[i]] + covalent_radii[nums[j]])
            if d < cut:
                out.add((i, j))
    return out


def water_integrity(atoms, o_water: int, bond_scale: float = HX_BOND_SCALE) -> dict:
    cut = bond_scale * (covalent_radii[atomic_numbers["O"]] + covalent_radii[atomic_numbers["H"]])
    pos, nums = atoms.get_positions(), atoms.numbers
    d = np.linalg.norm(pos - pos[o_water], axis=1)
    hs = [int(i) for i in np.where((d < cut) & (nums == 1))[0]]
    return {"n_H": len(hs), "H_indices": hs, "intact": len(hs) == 2, "cutoff_A": float(cut)}


def check_coordination(atoms, reference, cfg: dict, *, expected_hydride_state: str | None = None,
                       shell_A: float = SHELL_A, weaken_A: float = WEAKEN_A) -> dict:
    if len(atoms) != len(reference) or not np.array_equal(atoms.numbers, reference.numbers):
        raise ValueError("coordination check needs identical atom ordering to the reference")
    m = atom_map(cfg, len(atoms), require=["Sn", "O1", "O2", "O_water"])
    sn = m["Sn"]
    ref_sh, sh = sn_shell(reference, sn, shell_A), sn_shell(atoms, sn, shell_A)
    sym = atoms.get_chemical_symbols()
    lost = sorted(set(ref_sh) - set(sh))
    gained = sorted(set(sh) - set(ref_sh))
    d_ref = lambda i: float(np.linalg.norm(reference.positions[i] - reference.positions[sn]))
    d_now = lambda i: float(np.linalg.norm(atoms.positions[i] - atoms.positions[sn]))
    lengthened = {int(i): d_now(i) - d_ref(i) for i in ref_sh if d_now(i) - d_ref(i) > weaken_A}
    key = {n: m[n] for n in ("O1", "O2", "O_water")}
    key_d = {n: {"ref_A": d_ref(i), "now_A": d_now(i), "in_shell": i in sh} for n, i in key.items()}
    water = water_integrity(atoms, m["O_water"])
    hyd = classify_hydride_state(atoms, cfg)
    # heavy-atom bond changes away from Sn (e.g. Si-O cap cleavage, C-O/C-C scission) and hydride on Sn
    b_ref, b_now = heavy_bonds(reference, sn, HX_BOND_SCALE, shell_A), heavy_bonds(atoms, sn, HX_BOND_SCALE, shell_A)
    non_sn = lambda bs: {b for b in bs if sn not in b}
    broken, formed = sorted(non_sn(b_ref) - non_sn(b_now)), sorted(non_sn(b_now) - non_sn(b_ref))
    sn_h_ref = {i for i in range(len(atoms)) if atoms.numbers[i] == 1 and
                np.linalg.norm(reference.positions[i] - reference.positions[sn]) < SN_H_CLOSE_A}
    sn_h_now = {i for i in range(len(atoms)) if atoms.numbers[i] == 1 and
                np.linalg.norm(atoms.positions[i] - atoms.positions[sn]) < SN_H_CLOSE_A}
    hyd_on_sn = sorted(sn_h_now - sn_h_ref)
    reasons = []
    if broken:
        reasons.append("heavy-atom bond(s) broken: " + ", ".join(f"{sym[i]}{i}-{sym[j]}{j}" for i, j in broken))
    if formed:
        reasons.append("heavy-atom bond(s) formed: " + ", ".join(f"{sym[i]}{i}-{sym[j]}{j}" for i, j in formed))
    if hyd_on_sn:
        reasons.append("hydrogen within %.1f A of Sn (hydride on Sn?): " % SN_H_CLOSE_A + ", ".join(f"H{i}" for i in hyd_on_sn))
    if lost:
        reasons.append("shell atom(s) left the Sn shell: " + ", ".join(f"{sym[i]}{i}" for i in lost))
    if gained:
        reasons.append("atom(s) entered the Sn shell: " + ", ".join(f"{sym[i]}{i}" for i in gained))
    if not water["intact"]:
        reasons.append(f"bound water not intact (O_water has {water['n_H']} H)")
    if expected_hydride_state and hyd["state"] != expected_hydride_state:
        reasons.append(f"hydride state {hyd['state']} != expected {expected_hydride_state}")
    verdict = "STRUCTURALLY_INVALID" if reasons else ("WEAKENED" if lengthened else "PRESERVED")
    return {"verdict": verdict, "reasons": reasons,
            "n_shell_reference": len(ref_sh), "n_shell_now": len(sh),
            "lost_from_shell": lost, "gained_in_shell": gained,
            "heavy_bonds_broken": [list(b) for b in broken], "heavy_bonds_formed": [list(b) for b in formed],
            "hydrogen_near_sn": hyd_on_sn,
            "lengthened_gt_threshold_A": lengthened,
            "key_sn_o": key_d, "water": water, "hydride": hyd,
            "thresholds": {"shell_A": shell_A, "weaken_A": weaken_A,
                           "note": "analysis thresholds, applied identically to every method"}}
