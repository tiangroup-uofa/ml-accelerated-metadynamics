#!/usr/bin/env python3
"""
phaseC_water.py — C5 equilibrium water sanity check
===================================================
Short unbiased NVT MD on the n=20 droplet with raw MACE, Delta-A (baseline),
Delta-B (barrier-enriched), and xTB (reference); compare O–O distribution, radius
of gyration, condensation. Checks whether reactive barrier enrichment damages the
already-good water behaviour.
  micromamba run -n macemd python phaseC_water.py --which xtb
  micromamba run -n macemd python phaseC_water.py --which mace
"""
from __future__ import annotations
import argparse, os, sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diels_alder" / "neb"))
sys.path.insert(0, str(HERE))
RUNS = HERE.parent / "large" / "runs"
BINS = np.linspace(2.2, 8.0, 60); CTR = 0.5*(BINS[1:]+BINS[:-1])


def oo_rg(atoms):
    sym = np.array(atoms.get_chemical_symbols()); O = atoms.get_positions()[sym == "O"]
    com = O.mean(0); rg = float(np.sqrt(((O-com)**2).sum(1).mean()))
    d = [np.linalg.norm(O[a]-O[b]) for a in range(len(O)) for b in range(a+1, len(O))]
    return np.array(d), rg


def run_md(atoms0, calc, ps=1.5, equil=0.3, temp=300, step=0.5, sample=10):
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    at = atoms0.copy(); at.calc = calc
    MaxwellBoltzmannDistribution(at, temperature_K=temp)
    dyn = Langevin(at, timestep=step*units.fs, temperature_K=temp, friction=0.01/units.fs)
    n = int(ps*1000/step); neq = int(equil*1000/step); every = int(sample/step)
    dd, rgs = [], []
    for i in range(n):
        dyn.run(1)
        if i >= neq and i % every == 0:
            d, rg = oo_rg(at); dd.append(d); rgs.append(rg)
    hist, _ = np.histogram(np.concatenate(dd), bins=BINS, density=True)
    return hist, np.array(rgs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--which", choices=["mace", "xtb"], required=True)
    a = ap.parse_args(); os.environ["OMP_NUM_THREADS"] = "4"
    start = read(str(RUNS / "n20" / "opt" / "xtbopt.xyz"))
    out = {}
    if a.which == "xtb":
        from xtb.ase.calculator import XTB
        h, rg = run_md(start, XTB(method="GFN2-xTB"))
        out["xtb"] = dict(hist=h.tolist(), rg_mean=float(rg.mean()), rg_std=float(rg.std()),
                          peak=float(CTR[h.argmax()]))
        print(f"xtb: O-O peak {out['xtb']['peak']:.2f} Å, Rg {out['xtb']['rg_mean']:.2f}")
    else:
        import torch; torch.set_num_threads(4)
        from mace.calculators import mace_off
        from corrections import CorrectedCalculator, Identity, Delta
        from pairwise_delta import PairwiseDelta
        base = mace_off(model="small", device="cpu", default_dtype="float32")
        d = json.loads((HERE / "phaseC" / "fitted_deltas.json").read_text())
        models = {"raw_MACE": Identity(),
                  "A_baseline": Delta(PairwiseDelta.from_dict(d["A_baseline"])),
                  "B_barrier": Delta(PairwiseDelta.from_dict(d["B_barrier"]))}
        for name, corr in models.items():
            h, rg = run_md(start, CorrectedCalculator(base, corr))
            out[name] = dict(hist=h.tolist(), rg_mean=float(rg.mean()), rg_std=float(rg.std()),
                             peak=float(CTR[h.argmax()]))
            print(f"{name:11s}: O-O peak {out[name]['peak']:.2f} Å, "
                  f"Rg {out[name]['rg_mean']:.2f}±{out[name]['rg_std']:.2f}")
    f = HERE / "phaseC" / f"water_{a.which}.json"
    f.write_text(json.dumps({"ctr": CTR.tolist(), **out})); print(f"saved -> {f}")


if __name__ == "__main__":
    main()
