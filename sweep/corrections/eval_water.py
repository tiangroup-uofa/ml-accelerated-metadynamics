#!/usr/bin/env python3
"""
eval_water.py
=============
Water-droplet benchmark for the corrections: short unbiased NVT MD with each
corrected MACE calculator (and xTB as the reference), comparing droplet structure
— O–O pair distribution (RDF analogue), radius of gyration, and a scalar
structural-agreement distance to the xTB reference.

Run TWICE in separate processes (MACE and xtb-python both use OpenMP):
    micromamba run -n macemd python eval_water.py --which xtb    # reference
    micromamba run -n macemd python eval_water.py --which mace   # baseline + 4 corrections
"""
from __future__ import annotations
import argparse, os, sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RUNS = HERE.parent / "large" / "runs"
BINS = np.linspace(2.2, 8.0, 60)
CTR = 0.5 * (BINS[1:] + BINS[:-1])


def oo_and_rg(atoms):
    sym = np.array(atoms.get_chemical_symbols())
    O = atoms.get_positions()[sym == "O"]
    com = O.mean(0); rg = float(np.sqrt(((O - com) ** 2).sum(1).mean()))
    d = []
    for a in range(len(O)):
        for b in range(a + 1, len(O)):
            d.append(np.linalg.norm(O[a] - O[b]))
    return np.array(d), rg


def run_md(atoms0, calc, ps=1.5, equil_ps=0.3, temp=300, step=0.5, sample=10):
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    at = atoms0.copy(); at.calc = calc
    MaxwellBoltzmannDistribution(at, temperature_K=temp)
    dyn = Langevin(at, timestep=step*units.fs, temperature_K=temp, friction=0.01/units.fs)
    n = int(ps*1000/step); neq = int(equil_ps*1000/step); every = int(sample/step)
    dd, rgs = [], []
    for i in range(n):
        dyn.run(1)
        if i >= neq and i % every == 0:
            d, rg = oo_and_rg(at); dd.append(d); rgs.append(rg)
    hist, _ = np.histogram(np.concatenate(dd), bins=BINS, density=True)
    return hist, np.array(rgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["mace", "xtb"], required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    start = read(str(RUNS / f"n{a.n}" / "opt" / "xtbopt.xyz"))
    RES = HERE / "results"; RES.mkdir(exist_ok=True)
    out = {}

    if a.which == "xtb":
        from xtb.ase.calculator import XTB
        hist, rg = run_md(start, XTB(method="GFN2-xTB"))
        out["xtb"] = dict(hist=hist.tolist(), rg_mean=float(rg.mean()),
                          rg_std=float(rg.std()), peak=float(CTR[hist.argmax()]))
        print(f"xtb: O-O peak {out['xtb']['peak']:.2f} Å, Rg {out['xtb']['rg_mean']:.2f}")
    else:
        import torch; torch.set_num_threads(a.threads)
        from mace.calculators import mace_off
        from build_corr import load_corrections, corrected_calc
        base = mace_off(model="small", device="cpu", default_dtype="float32")
        corr = load_corrections()
        for name in corr:
            hist, rg = run_md(start, corrected_calc(base, name, corr))
            out[name] = dict(hist=hist.tolist(), rg_mean=float(rg.mean()),
                             rg_std=float(rg.std()), peak=float(CTR[hist.argmax()]))
            print(f"{name:9s}: O-O peak {out[name]['peak']:.2f} Å, "
                  f"Rg {out[name]['rg_mean']:.2f}±{out[name]['rg_std']:.2f}")
    f = RES / f"water_{a.which}.json"
    f.write_text(json.dumps({"ctr": CTR.tolist(), **out}))
    print(f"saved -> {f}")


if __name__ == "__main__":
    main()
