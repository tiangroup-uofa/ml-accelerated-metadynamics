#!/usr/bin/env python3
"""
Phase 1B: xTB vs MACE on identical frozen xTB geometries (R/TS/P).

Eliminates path-search methodology entirely. Both calculators evaluate the exact
same R/TS/P geometries (from native xTB), frozen (no relaxation).

Report:
  - absolute energies
  - relative energies
  - force norms / max atomic force
  - barrier height on identical geometries (ΔE‡_xTB vs ΔE‡_MACE@xTBgeom)
  - reaction energy on identical geometries
  - MACE force norm at xTB TS

Purpose: answer whether the energetic disagreement persists on identical geometries.

Run via: python phase1_identical_geoms.py
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
REACTANT = HERE / "start.xyz"
TS = HERE / "ts_opt.xyz"
PRODUCT = HERE / "end.xyz"

# Verify structures exist
for path in [REACTANT, TS, PRODUCT]:
    if not path.exists():
        raise FileNotFoundError(f"{path}")
    print(f"OK {path.name}")


def evaluate_subprocess(geom_file, calc_name):
    """Evaluate via subprocess to avoid OpenMP/conda conflicts."""
    script = f"""
import sys, json, os, numpy as np
os.environ["OMP_NUM_THREADS"] = "4"
from ase.io import read
geom = read("{geom_file}")
if "{calc_name}" == "xtb":
    from xtb.ase.calculator import XTB
    calc = XTB(method="GFN2-xTB", electronic_temperature=1000.0, max_iterations=500)
else:
    import torch; torch.set_num_threads(4)
    from mace.calculators import mace_off
    calc = mace_off(model="small", device="cpu", default_dtype="float64")
geom.calc = calc
E = float(geom.get_potential_energy())
F = geom.get_forces()
fmax = float(np.max(np.linalg.norm(F, axis=1)))
fnorm = float(np.linalg.norm(F))
print(json.dumps({{"E": E, "fmax": fmax, "fnorm": fnorm}}))
"""
    result = subprocess.run(
        ["/usr/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"{calc_name} evaluation failed")
    return json.loads(result.stdout.strip())


def main():
    # Read structures for geometry info
    R = read(str(REACTANT))
    TS_geom = read(str(TS))
    P = read(str(PRODUCT))

    print("\n=== Phase 1B: identical-geometry comparison ===")
    print(f"Reactant: {REACTANT.name} ({len(R)} atoms, C6H10)")
    print(f"TS (xTB):  {TS.name} ({len(TS_geom)} atoms, forming C–C: ?)")
    print(f"Product:  {PRODUCT.name} ({len(P)} atoms, C6H10)")

    # Check forming C–C distance at TS
    d05 = float(np.linalg.norm(TS_geom.positions[0] - TS_geom.positions[5]))
    d34 = float(np.linalg.norm(TS_geom.positions[3] - TS_geom.positions[4]))
    ts_forming_mean = 0.5 * (d05 + d34)
    print(f"  TS forming C–C: {d05:.3f} / {d34:.3f} Å (mean {ts_forming_mean:.3f})")

    print("\n=== Evaluating xTB ===")
    E_xtb_r = evaluate_subprocess(str(REACTANT), "xtb")
    E_xtb_ts = evaluate_subprocess(str(TS), "xtb")
    E_xtb_p = evaluate_subprocess(str(PRODUCT), "xtb")
    print(f"R:  E={E_xtb_r['E']:.6f} eV, fmax={E_xtb_r['fmax']:.4f}, fnorm={E_xtb_r['fnorm']:.4f}")
    print(f"TS: E={E_xtb_ts['E']:.6f} eV, fmax={E_xtb_ts['fmax']:.4f}, fnorm={E_xtb_ts['fnorm']:.4f}")
    print(f"P:  E={E_xtb_p['E']:.6f} eV, fmax={E_xtb_p['fmax']:.4f}, fnorm={E_xtb_p['fnorm']:.4f}")

    print("\n=== Evaluating MACE ===")
    E_mace_r = evaluate_subprocess(str(REACTANT), "mace")
    E_mace_ts = evaluate_subprocess(str(TS), "mace")
    E_mace_p = evaluate_subprocess(str(PRODUCT), "mace")
    print(f"R:  E={E_mace_r['E']:.6f} eV, fmax={E_mace_r['fmax']:.4f}, fnorm={E_mace_r['fnorm']:.4f}")
    print(f"TS: E={E_mace_ts['E']:.6f} eV, fmax={E_mace_ts['fmax']:.4f}, fnorm={E_mace_ts['fnorm']:.4f}")
    print(f"P:  E={E_mace_p['E']:.6f} eV, fmax={E_mace_p['fmax']:.4f}, fnorm={E_mace_p['fnorm']:.4f}")

    # Convert to kcal/mol
    EV2KCAL = 23.060548

    # Barrier heights
    barrier_xtb = (E_xtb_ts["E"] - E_xtb_r["E"]) * EV2KCAL
    barrier_mace = (E_mace_ts["E"] - E_mace_r["E"]) * EV2KCAL

    # Reaction energy
    rxn_xtb = (E_xtb_p["E"] - E_xtb_r["E"]) * EV2KCAL
    rxn_mace = (E_mace_p["E"] - E_mace_r["E"]) * EV2KCAL

    print("\n=== RESULTS ===")
    print(f"xTB barrier (identical geometries):            {barrier_xtb:.1f} kcal/mol")
    print(f"MACE barrier (identical xTB geometries):       {barrier_mace:.1f} kcal/mol")
    print(f"Difference:                                     {barrier_mace - barrier_xtb:+.1f} kcal/mol")
    print()
    print(f"xTB reaction energy:                            {rxn_xtb:.1f} kcal/mol")
    print(f"MACE reaction energy (identical geoms):         {rxn_mace:.1f} kcal/mol")
    print(f"Difference:                                     {rxn_mace - rxn_xtb:+.1f} kcal/mol")
    print()
    print(f"MACE force norm at xTB TS geometry:             {E_mace_ts['fnorm']:.4f} eV/Å (all atoms)")
    print(f"MACE fmax at xTB TS geometry:                   {E_mace_ts['fmax']:.4f} eV/Å (max atom)")
    print(f"xTB fmax at TS geometry (should be small):      {E_xtb_ts['fmax']:.4f} eV/Å")

    # Write JSON
    result = {
        "comment": "xTB and MACE evaluated on identical frozen xTB R/TS/P geometries",
        "TS_forming_CC_mean": float(ts_forming_mean),
        "xtb": {
            "R": {"E_eV": E_xtb_r["E"], "fmax": E_xtb_r["fmax"], "fnorm": E_xtb_r["fnorm"]},
            "TS": {"E_eV": E_xtb_ts["E"], "fmax": E_xtb_ts["fmax"], "fnorm": E_xtb_ts["fnorm"]},
            "P": {"E_eV": E_xtb_p["E"], "fmax": E_xtb_p["fmax"], "fnorm": E_xtb_p["fnorm"]},
            "barrier_kcal": float(barrier_xtb),
            "reaction_energy_kcal": float(rxn_xtb),
        },
        "mace": {
            "R": {"E_eV": E_mace_r["E"], "fmax": E_mace_r["fmax"], "fnorm": E_mace_r["fnorm"]},
            "TS": {"E_eV": E_mace_ts["E"], "fmax": E_mace_ts["fmax"], "fnorm": E_mace_ts["fnorm"]},
            "P": {"E_eV": E_mace_p["E"], "fmax": E_mace_p["fmax"], "fnorm": E_mace_p["fnorm"]},
            "barrier_kcal": float(barrier_mace),
            "reaction_energy_kcal": float(rxn_mace),
        },
        "barrier_difference_kcal": float(barrier_mace - barrier_xtb),
        "reaction_energy_difference_kcal": float(rxn_mace - rxn_xtb),
    }

    out = HERE / "phase1_identical_geoms.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
