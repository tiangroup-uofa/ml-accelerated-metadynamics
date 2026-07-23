#!/usr/bin/env python3
"""
compare_md.py
=============
Second step of the MLIP-surrogate direction: does MACE-OFF23 give the same
*dynamics/structure* as GFN2-xTB, not just the same static forces (Study 05)?

We run **unbiased NVT MD** on a water droplet with each potential from the same
start, then compare **structural distributions** — the right comparison for
chaotic MD, since individual trajectories diverge but the ensembles should agree
if the two surfaces describe the same liquid:

  * O–O pair-distance distribution (the finite-cluster analogue of g(r)) — H-bond
    peak position/height.
  * radius of gyration distribution — droplet size / density.
  * mean oxygen-neighbour coordination.
  * energy stability (both should be stable, no blow-up / evaporation).

This validates MACE-OFF23 for *dynamics*, the prerequisite for MACE-driven
metadynamics.

Run in the clean `macemd` env (see ../../SETUP.md):
    micromamba run -n macemd python compare_md.py --n 20 --ps 2.5 --equil-ps 0.5
"""
from __future__ import annotations
import argparse, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "large" / "runs"
OUT = HERE / "analysis"; FIG = HERE / "figures"
R0_OO, SW_P, SW_Q = 3.2, 8, 14


def switch(r):
    x = r / R0_OO; x = np.where(np.isclose(x, 1.0), 1.0 + 1e-9, x)
    return (1.0 - x ** SW_P) / (1.0 - x ** SW_Q)


def frame_metrics(pos, o_idx):
    o = pos[o_idx]
    com = o.mean(0); rg = float(np.sqrt(((o - com) ** 2).sum(1).mean()))
    # all O-O distances
    d = []
    coord = np.zeros(len(o))
    for a in range(len(o)):
        for b in range(a + 1, len(o)):
            r = float(np.linalg.norm(o[a] - o[b])); d.append(r)
            s = switch(r); coord[a] += s; coord[b] += s
    return rg, float(coord.mean()), np.array(d)


def run_md(calc_name, atoms0, ps, equil_ps, temp, step_fs, sample_fs):
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = atoms0.copy()
    if calc_name == "mace":
        from mace.calculators import mace_off
        # float32 is ~1.8x faster than float64 and structurally indistinguishable
        # here; on-par with xTB at n=20 and faster beyond (see benchmark.py).
        atoms.calc = mace_off(model="small", device="cpu", default_dtype="float32")
    else:
        from xtb.ase.calculator import XTB
        atoms.calc = XTB(method="GFN2-xTB")
    sym = np.array(atoms.get_chemical_symbols())
    o_idx = np.where(sym == "O")[0]
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    dyn = Langevin(atoms, timestep=step_fs * units.fs, temperature_K=temp,
                   friction=0.01 / units.fs)
    nsteps = int(ps * 1000 / step_fs); nequil = int(equil_ps * 1000 / step_fs)
    every = int(sample_fs / step_fs)
    rgs, coords, dds, energies = [], [], [], []
    t0 = time.time()
    for i in range(nsteps):
        dyn.run(1)
        if i >= nequil and i % every == 0:
            pos = atoms.get_positions()
            rg, co, dd = frame_metrics(pos, o_idx)
            rgs.append(rg); coords.append(co); dds.append(dd)
            energies.append(atoms.get_potential_energy())
    dt = time.time() - t0
    print(f"  [{calc_name}] {nsteps} steps in {dt:.0f}s ({dt/nsteps*1000:.0f} ms/step), "
          f"{len(rgs)} samples")
    return dict(rg=np.array(rgs), coord=np.array(coords),
                dd=np.concatenate(dds), E=np.array(energies),
                ms_step=dt / nsteps * 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ps", type=float, default=2.5)
    ap.add_argument("--equil-ps", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--step-fs", type=float, default=0.5)
    ap.add_argument("--sample-fs", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch/OMP threads (4 was fastest for small droplets)")
    args = ap.parse_args()
    import os as _os; _os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch; torch.set_num_threads(args.threads)
    from ase.io import read
    start = read(str(RUNS / f"n{args.n}" / "opt" / "xtbopt.xyz"))
    print(f"droplet n={args.n} ({len(start)} atoms), unbiased NVT {args.ps} ps @ {args.temp} K")
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    res = {name: run_md(name, start, args.ps, args.equil_ps, args.temp,
                        args.step_fs, args.sample_fs) for name in ("xtb", "mace")}

    # O-O pair distribution (shared bins)
    hi = max(res["xtb"]["dd"].max(), res["mace"]["dd"].max())
    bins = np.linspace(2.2, min(hi, 9.0), 60); ctr = 0.5 * (bins[1:] + bins[:-1])
    import pandas as pd
    df = pd.DataFrame({"r_A": ctr})
    for name in ("xtb", "mace"):
        h, _ = np.histogram(res[name]["dd"], bins=bins, density=True)
        df[f"pOO_{name}"] = h
    df.to_csv(OUT / f"pairdist_n{args.n}.csv", index=False)
    summ = {name: dict(rg_mean=float(res[name]["rg"].mean()),
                       rg_std=float(res[name]["rg"].std()),
                       coord_mean=float(res[name]["coord"].mean()),
                       E_drift=float(res[name]["E"][-1] - res[name]["E"][0]),
                       ms_step=res[name]["ms_step"]) for name in res}
    pd.DataFrame(summ).T.to_csv(OUT / f"md_summary_n{args.n}.csv")

    # first-peak (H-bond) position of the O-O distribution
    def peak(name):
        h = df[f"pOO_{name}"].values; k = int(np.argmax(h[:len(h)//2])); return ctr[k]
    print("\n=== MACE-OFF23 vs GFN2-xTB unbiased MD (structure) ===")
    for name in ("xtb", "mace"):
        s = summ[name]
        print(f"  {name:4s}: Rg {s['rg_mean']:.2f}±{s['rg_std']:.2f} A  "
              f"coord {s['coord_mean']:.2f}  O-O 1st peak {peak(name):.2f} A  "
              f"{s['ms_step']:.0f} ms/step")

    # plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(ctr, df.pOO_xtb, color="tab:red", label="GFN2-xTB")
    a1.plot(ctr, df.pOO_mace, color="tab:blue", label="MACE-OFF23")
    a1.set_xlabel("O–O distance (Å)"); a1.set_ylabel("pair-distance density")
    a1.set_title(f"O–O distribution (n={args.n})"); a1.legend()
    a2.hist(res["xtb"]["rg"], bins=20, alpha=.55, color="tab:red", density=True, label="GFN2-xTB")
    a2.hist(res["mace"]["rg"], bins=20, alpha=.55, color="tab:blue", density=True, label="MACE-OFF23")
    a2.set_xlabel("radius of gyration (Å)"); a2.set_ylabel("density")
    a2.set_title("Droplet size distribution"); a2.legend()
    fig.tight_layout(); fig.savefig(FIG / f"md_structure_n{args.n}.png", dpi=140)
    print(f"  wrote {FIG/f'md_structure_n{args.n}.png'} + CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
