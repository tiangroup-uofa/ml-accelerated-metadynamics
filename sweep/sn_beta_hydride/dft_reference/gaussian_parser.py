"""
Strict Gaussian (g09/g16) text-output parser for the Sn-beta DFT reference set.

STATUS: written from the documented Gaussian log layout and exercised on
SYNTHETIC fixtures (tests/test_dft_reference.py). It has NOT been run on a real
Gaussian09 log of this project -- Tian's dataset has not arrived. Re-verify the
first real file by eye (see `describe_job`) before trusting any number.

Design rules
------------
* Fail loudly. A job without "Normal termination", a step without an energy or
  forces, a truncated block, an unsupported method, or a geometry frame that
  cannot be paired with the force frame all raise `GaussianParseError`.
* Forces are paired with the INPUT-orientation geometry. Gaussian prints the
  "Forces (Hartrees/Bohr)" block in the input frame ("Axes restored to original
  set"), so pairing them with "Standard orientation" coordinates would silently
  rotate one of the two. Standard orientation is accepted only with `nosymm`.
* Gaussian's printed forces are forces (= -gradient). No sign flip is applied to
  them. `gradient_to_force` exists for gradient-convention sources (e.g. the
  fchk "Cartesian Gradient"), and is tested.
* Units: Hartree -> eV, Hartree/Bohr -> eV/Angstrom.
* Only SCF energies (HF / DFT, `SCF Done`) are supported. Post-HF energies
  (MP2, CC, CASSCF ...) are rejected rather than silently using the SCF energy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
HARTREE_BOHR_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANG      # 51.42208...
HARTREE_TO_KJ_MOL = 2625.4996394799

_UNSUPPORTED_METHODS = ("mp2", "mp3", "mp4", "ccsd", "cisd", "casscf", "qci", "ci(", "eomccsd",
                        "b2plyp", "mpw2plyp", "dsdpbe")     # double hybrids print non-SCF totals
_ORIENT_TAGS = (("Input orientation:", "input"),
                ("Z-Matrix orientation:", "zmatrix"),
                ("Standard orientation:", "standard"))


class GaussianParseError(ValueError):
    """Raised for any incomplete, ambiguous or unsupported Gaussian output."""


def gradient_to_force(gradient):
    """Gradient (dE/dx) -> force (-dE/dx). Gaussian's printed forces are already
    forces; use this only for gradient-convention sources such as fchk."""
    return -np.asarray(gradient, dtype=float)


@dataclass
class GaussianStep:
    index: int
    numbers: np.ndarray            # (N,) atomic numbers, input order
    positions: np.ndarray          # (N,3) Angstrom
    energy_hartree: float
    energy_eV: float
    forces_eV_A: np.ndarray        # (N,3) eV/Angstrom  (= -gradient)
    orientation: str               # "input" | "zmatrix" | "standard"


@dataclass
class GaussianJob:
    path: str
    version: str | None
    revision: str | None
    route: str
    method: str | None
    basis: str | None
    basis_is_gen: bool
    charge: int
    multiplicity: int
    natoms: int
    steps: list
    n_normal_terminations: int
    is_opt: bool
    opt_stationary_point_found: bool | None
    frequencies_cm1: list
    warnings: list = field(default_factory=list)
    has_ecp: bool = False

    @property
    def final(self) -> GaussianStep:
        return self.steps[-1]

    @property
    def n_imaginary(self) -> int | None:
        if not self.frequencies_cm1:
            return None
        return int(sum(1 for f in self.frequencies_cm1 if f < 0.0))


def describe_job(job: GaussianJob) -> str:
    """Human-readable one-glance summary; eyeball this on the first real file."""
    return (f"{job.path}: {job.version} rev {job.revision} | route '{job.route}' | "
            f"method={job.method} basis={job.basis} gen={job.basis_is_gen} | "
            f"q={job.charge} mult={job.multiplicity} | N={job.natoms} | "
            f"steps={len(job.steps)} | opt_stationary={job.opt_stationary_point_found} | "
            f"n_imag={job.n_imaginary} | final E={job.final.energy_hartree:.8f} Eh"
            + (f" | WARNINGS: {job.warnings}" if job.warnings else ""))


def _read_orientation(lines, i, path):
    """`lines[i]` holds the orientation title. Return (numbers, xyz, next_index)."""
    j = i + 1
    if j + 3 >= len(lines):
        raise GaussianParseError(f"{path}: truncated orientation block at line {i + 1}")
    j += 4                                   # dashes, 2 header lines, dashes
    nums, xyz = [], []
    while j < len(lines) and not lines[j].lstrip().startswith("----"):
        tok = lines[j].split()
        if len(tok) != 6:
            raise GaussianParseError(f"{path}: malformed orientation row at line {j + 1}: {lines[j]!r}")
        nums.append(int(tok[1])); xyz.append([float(t) for t in tok[3:6]])
        j += 1
    if j >= len(lines):
        raise GaussianParseError(f"{path}: orientation block not terminated (file truncated)")
    if any(n <= 0 for n in nums):
        raise GaussianParseError(f"{path}: dummy/ghost atom (Z<=0) in orientation block; unsupported")
    return np.array(nums, int), np.array(xyz, float), j + 1


def _read_forces(lines, i, path):
    """`lines[i]` is the 'Forces (Hartrees/Bohr)' title."""
    j = i + 3                                # 'Number Number X Y Z' header, dashes
    rows = []
    while j < len(lines) and not lines[j].lstrip().startswith("----"):
        tok = lines[j].split()
        if len(tok) != 5:
            raise GaussianParseError(f"{path}: malformed force row at line {j + 1}: {lines[j]!r}")
        rows.append([float(t) for t in tok[2:5]])
        j += 1
    if j >= len(lines):
        raise GaussianParseError(f"{path}: force block not terminated (file truncated)")
    return np.array(rows, float), j + 1


def _route(lines):
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#") and i > 0 and lines[i - 1].lstrip().startswith("----"):
            parts, j = [], i
            while j < len(lines) and not lines[j].lstrip().startswith("----"):
                parts.append(lines[j].strip()); j += 1
            return " ".join(parts)
    return ""


def parse_gaussian_log(path, *, allow_multi_job: bool = False) -> GaussianJob:
    path = str(path)
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()

    if "Error termination" in text:
        k = text.index("Error termination")
        raise GaussianParseError(f"{path}: Gaussian reported an error: {text[k:k + 120].splitlines()[0]!r}")
    n_term = len(re.findall(r"Normal termination of Gaussian", text))
    n_jobs = len(re.findall(r"Entering Gaussian System", text))
    if n_term == 0:
        raise GaussianParseError(f"{path}: no 'Normal termination of Gaussian' -- job incomplete or truncated")
    if n_jobs > 1 and not allow_multi_job:
        raise GaussianParseError(f"{path}: {n_jobs} chained jobs (Link1) in one file; split them or pass allow_multi_job=True")
    if n_term != max(n_jobs, 1):
        raise GaussianParseError(f"{path}: {n_jobs} jobs but {n_term} normal terminations -- some job incomplete")

    m = re.search(r"^\s*Gaussian\s+(\d+):\s+(\S+)", text, re.M)
    version, revision = (f"G{m.group(1)}", m.group(2)) if m else (None, None)
    route = _route(lines)
    rl = route.lower()
    if not route:
        raise GaussianParseError(f"{path}: route section ('# ...') not found")
    for bad in _UNSUPPORTED_METHODS:
        if bad in rl:
            raise GaussianParseError(f"{path}: method '{bad}' in route '{route}': only SCF (HF/DFT) energies are supported")

    mm = re.search(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", text)
    if not mm:
        raise GaussianParseError(f"{path}: 'Charge = .. Multiplicity = ..' line not found")
    charge, mult = int(mm.group(1)), int(mm.group(2))

    meth = re.search(r"(?:^|[\s(])([A-Za-z0-9()+\-*]+)/([A-Za-z0-9()+\-*,]+)", route)
    method, basis = (meth.group(1), meth.group(2)) if meth else (None, None)
    basis_is_gen = bool(basis and basis.lower() in ("gen", "genecp")) or " gen" in rl

    # Each warning is "[tag] message". load_reference_set REFUSES any tag the manifest does not
    # list under "acknowledged_warnings" -- these change what the energy means, so they are policy gates.
    warnings = []
    if "empiricaldispersion" in rl or re.search(r"\bem=", rl):
        warnings.append("[dispersion] empirical dispersion in route: confirm 'SCF Done' includes it before comparing")
    if basis_is_gen:
        warnings.append("[gen_basis] general basis/ECP: record the Sn basis/ECP in the manifest (not parsed from the log)")
    if "scrf" in rl:
        warnings.append("[solvent] implicit solvent (SCRF) in route: energies are not vacuum energies")
    has_ecp = bool(re.search(r"Pseudopotential Parameters|ECP\s+parameters|PSEUDO", text))

    for bad in ("modredundant", "scan", "irc"):
        if re.search(rf"\b{bad}\b", rl):
            raise GaussianParseError(f"{path}: '{bad}' job in route '{route}': scan/IRC logs hold many unrelated "
                                     "geometries and one 'Stationary point found'; unsupported (split into single jobs)")
    is_opt = bool(re.search(r"\bopt\b|\bopt=|\bopt\(", rl))
    nosymm = "nosymm" in rl
    stat = ("Stationary point found" in text) if is_opt else None

    mn = re.search(r"NAtoms=\s*(\d+)", text)
    natoms_hdr = int(mn.group(1)) if mn else None

    # ---- event scan: orientation(s) -> SCF Done -> Forces => one step ----
    steps, pend, energy = [], {}, None
    i = 0
    while i < len(lines):
        ln = lines[i]
        tag = next((t for title, t in _ORIENT_TAGS if title in ln), None)
        if tag:
            nums, xyz, i = _read_orientation(lines, i, path)
            if pend and not pend.get("done") and tag in pend and not np.allclose(pend[tag][1], xyz, atol=1e-6):
                raise GaussianParseError(f"{path}: a geometry block was followed by another, different one with no "
                                         "energy/forces in between (a step lost or 'force' not requested)")
            pend = {} if (pend.get("done") or not pend) else pend
            pend[tag] = (nums, xyz)
            energy = None
            continue
        ms = re.search(r"SCF Done:\s+E\(([^)]+)\)\s*=\s*(-?\d+\.\d+)", ln)
        if ms:
            energy = float(ms.group(2)); i += 1; continue
        if "Forces (Hartrees/Bohr)" in ln:
            f, i = _read_forces(lines, i, path)
            if energy is None:
                raise GaussianParseError(f"{path}: forces block with no preceding 'SCF Done' energy (line {i})")
            frame = next((k for k in ("input", "zmatrix") if k in pend), None)
            if frame is None:
                if "standard" in pend and nosymm:
                    frame = "standard"
                    if "[standard_frame]" not in " ".join(warnings):
                        warnings.append("[standard_frame] only 'Standard orientation' printed; assumed equal to the input frame "
                                        "because of nosymm -- NOT verified against a real log; check the first one")
                elif "standard" in pend:
                    raise GaussianParseError(
                        f"{path}: only 'Standard orientation' coordinates available; Gaussian forces are in the input frame. "
                        "Re-run with 'nosymm' (or output Input orientation) to pair them safely")
                else:
                    raise GaussianParseError(f"{path}: forces block without a geometry block")
            nums, xyz = pend[frame]
            if f.shape[0] != len(nums):
                raise GaussianParseError(f"{path}: {f.shape[0]} force rows vs {len(nums)} atoms")
            steps.append(GaussianStep(len(steps), nums, xyz, energy, energy * HARTREE_TO_EV,
                                      f * HARTREE_BOHR_TO_EV_ANG, frame))
            pend["done"] = True
            energy = None
            continue
        i += 1

    if not steps:
        raise GaussianParseError(f"{path}: no complete (geometry, energy, forces) step found "
                                 "(missing 'force' keyword, or not a Gaussian SCF job?)")
    if sum(1 for z in steps[0].numbers if z > 36) and not has_ecp and not re.search(r"def2|sdd|lanl|cep|stuttgart|ecp", rl):
        warnings.append("[heavy_no_ecp] element with Z>36 but no pseudopotential block/ECP basis seen: confirm the Sn basis is all-electron")
    # a trailing geometry-only block after convergence is normal; one with a
    # DIFFERENT geometry and no forces means a step was lost
    for k in ("input", "zmatrix", "standard"):
        if k in pend and not pend.get("done"):
            if not np.allclose(pend[k][1], steps[-1].positions, atol=1e-4) and k != "standard":
                raise GaussianParseError(f"{path}: trailing geometry with no energy/forces (step lost)")
    if natoms_hdr is not None and natoms_hdr != len(steps[0].numbers):
        raise GaussianParseError(f"{path}: NAtoms={natoms_hdr} but orientation has {len(steps[0].numbers)}")
    for s in steps:
        if not np.array_equal(s.numbers, steps[0].numbers):
            raise GaussianParseError(f"{path}: atom ordering/elements change between steps")
    if is_opt and not stat:
        raise GaussianParseError(f"{path}: optimization job without 'Stationary point found' (not converged)")

    freqs = []
    for mf in re.finditer(r"Frequencies --\s+(.*)", text):
        freqs += [float(x) for x in mf.group(1).split()]

    return GaussianJob(path, version, revision, route, method, basis, basis_is_gen, charge, mult,
                       len(steps[0].numbers), steps, n_term, is_opt, stat, freqs, warnings, has_ecp)
