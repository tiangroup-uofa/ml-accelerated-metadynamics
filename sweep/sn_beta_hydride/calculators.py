#!/usr/bin/env python3
"""
calculators.py
==============
ASE calculators for the MACE foundation models tested on the Sn-beta hydride
transfer: **MACE-OMOL** and **MACE-POLAR**. The scientific workflow only ever
sees a ``ModelHandle``; everything model-specific lives in the ``_build_*``
functions and the ``MODELS`` registry below.

Verified against the installed mace-torch 0.3.16 (``mace.calculators``):

* ``mace_omol(model="extra_large" | path | URL, device, default_dtype, **kw)``
  ScaleShiftMACE, head ``omol``, Z = 1..83 (Sn included). Total charge and spin
  enter through categorical embeddings.
* ``mace_polar(model="polar-1-s" | "polar-1-m" | "polar-1-l" | path | URL, ...)``
  PolarMACE (``model_type="PolarMACE"``), Z = 1..83. Needs the external
  ``graph_longrange`` package; with mace-torch 0.3.16 only graph_electrostatics
  **tag v0.4.0** is API-compatible (PyPI 0.4.4 and GitHub HEAD raise
  ``unexpected keyword argument 'force_pbc_evaluator'``).

Both read ``atoms.info["charge"]`` (total charge) and ``atoms.info["spin"]``
(spin **multiplicity** 2S+1) and *silently default to 0 / 1 when absent*. The
``ChargeSpinCalculator`` wrapper therefore injects the configured values on
every call and refuses atoms that carry conflicting values.

The existing MACE-OFF23 workflows (``sweep/diels_alder/neb/calculators.py`` etc.)
are untouched.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from config_io import ConfigError, charge_and_multiplicity

GRAPH_LONGRANGE_FIX = ('pip install --no-deps '
                       '"git+https://github.com/WillBaldwin0/graph_electrostatics@v0.4.0"')


# --------------------------------------------------------------------------- #
#  model-specific builders (the ONLY place that knows about each model's API)
# --------------------------------------------------------------------------- #
def _build_omol(variant, device, dtype, options):
    from mace.calculators import mace_omol

    return mace_omol(model=variant, device=device, default_dtype=dtype, **options)


def check_polar_dependency() -> str:
    """Raise a clear error if graph_longrange is missing or API-incompatible."""
    # mace must be imported FIRST: its __init__ sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD,
    # without which e3nn (imported by graph_longrange) fails to load its constants
    # under torch >= 2.6 ("Weights only load failed").
    import mace  # noqa: F401

    try:
        import graph_longrange  # noqa: F401
        from graph_longrange.features import GTOElectrostaticFeatures
    except ImportError as e:
        raise RuntimeError(
            "MACE-POLAR needs the 'graph_longrange' package, which is not importable "
            f"({e}). Install the version compatible with mace-torch 0.3.16:\n  "
            + GRAPH_LONGRANGE_FIX) from e
    params = inspect.signature(GTOElectrostaticFeatures.precompute_geometry).parameters
    if "force_pbc_evaluator" not in params:
        raise RuntimeError(
            "The installed 'graph_longrange' is API-incompatible with this mace-torch "
            "(precompute_geometry has no 'force_pbc_evaluator'; MACE-POLAR energy "
            "calls would fail). Install the compatible tag:\n  " + GRAPH_LONGRANGE_FIX)
    return getattr(graph_longrange, "__file__", "?")


def _build_polar(variant, device, dtype, options):
    check_polar_dependency()
    from mace.calculators import mace_polar

    return mace_polar(model=variant, device=device, default_dtype=dtype, **options)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    pretty: str
    builder: object
    variants: tuple
    default_variant: str
    reads_charge_spin: bool = True


MODELS = {
    "mace-omol": ModelSpec("mace-omol", "MACE-OMOL-0", _build_omol,
                           ("extra_large",), "extra_large"),
    "mace-polar": ModelSpec("mace-polar", "MACE-POLAR-1", _build_polar,
                            ("polar-1-s", "polar-1-m", "polar-1-l"), "polar-1-s"),
}
DTYPES = ("float32", "float64")


# --------------------------------------------------------------------------- #
#  config parsing
# --------------------------------------------------------------------------- #
@dataclass
class CalcSettings:
    model: str
    variant: str
    device: str = "cpu"
    dtype: str = "float64"
    threads: int | None = None
    options: dict = field(default_factory=dict)
    model_path: str | None = None       # local checkpoint overrides `variant`

    @property
    def tag(self) -> str:
        return f"{self.model}_{self.variant}".replace("/", "_")


def parse_calc_settings(cfg: dict, model=None, variant=None, device=None,
                        dtype=None) -> CalcSettings:
    """Build ``CalcSettings`` from ``config["calculator"]`` + CLI overrides."""
    c = dict(cfg.get("calculator") or {})
    model = (model or c.get("model") or "").lower()
    if model not in MODELS:
        raise ConfigError(f"unknown calculator model '{model}'. Known: {sorted(MODELS)}")
    spec = MODELS[model]
    per_model = (c.get("models") or {}).get(model, {})
    variant = variant or per_model.get("variant") or spec.default_variant
    model_path = per_model.get("model_path")
    if model_path is None and variant not in spec.variants:
        raise ConfigError(f"{model}: unknown variant '{variant}'. Known: {spec.variants} "
                          "(or set calculator.models.<model>.model_path)")
    dtype = dtype or c.get("dtype", "float64")
    if dtype not in DTYPES:
        raise ConfigError(f"dtype must be one of {DTYPES}, got '{dtype}'")
    options = dict(per_model.get("options") or {})
    for bad in ("model", "device", "default_dtype"):
        if bad in options:
            raise ConfigError(f"calculator option '{bad}' must be set via the dedicated field")
    return CalcSettings(model=model, variant=variant, device=device or c.get("device", "cpu"),
                        dtype=dtype, threads=c.get("threads"), options=options,
                        model_path=model_path)


# --------------------------------------------------------------------------- #
#  per-image wrapper
# --------------------------------------------------------------------------- #
class ChargeSpinCalculator(Calculator):
    """Evaluate a shared base calculator with the configured charge/multiplicity.

    Cheap to instantiate (the model is shared), so NEB can have one per image
    (``allow_shared_calculator=False``, as in the Diels-Alder NEB). Counts
    ``calculate()`` invocations in ``n_calls`` (one call = one energy+forces).
    An optional ``correction`` with ``.apply(atoms, E, F) -> (E, F)`` (e.g. the
    ``ForceCorrection`` classes in ``sweep/diels_alder/neb/corrections.py``) is
    the hook for later corrections; none is used in this benchmark yet.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, base, charge: int, multiplicity: int, correction=None, **kw):
        super().__init__(**kw)
        self.base = base
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.correction = correction
        self.n_calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        for key, want in (("charge", self.charge), ("spin", self.multiplicity)):
            have = atoms.info.get(key)
            if have is not None and int(have) != want:
                raise ValueError(f"atoms.info['{key}']={have} conflicts with the configured "
                                 f"value {want}; refusing to evaluate")
        a = atoms.copy()
        a.info["charge"] = self.charge
        a.info["spin"] = self.multiplicity
        self.base.reset()          # never reuse a cached result across wrappers
        a.calc = self.base
        e = float(a.get_potential_energy())
        f = np.array(a.get_forces())
        if self.correction is not None:
            e, f = self.correction.apply(atoms, e, f)
        self.n_calls += 1
        self.results["energy"] = e
        self.results["free_energy"] = e
        self.results["forces"] = f


class ModelHandle:
    """A loaded foundation model + the system's charge/multiplicity."""

    def __init__(self, settings: CalcSettings, base, charge: int, multiplicity: int):
        self.settings = settings
        self.base = base
        self.charge = charge
        self.multiplicity = multiplicity
        self._calcs = []

    def new_calc(self, correction=None) -> ChargeSpinCalculator:
        c = ChargeSpinCalculator(self.base, self.charge, self.multiplicity, correction)
        self._calcs.append(c)
        return c

    @property
    def total_calls(self) -> int:
        return sum(c.n_calls for c in self._calcs)

    @property
    def supported_numbers(self) -> set[int]:
        return {int(z) for z in self.base.z_table.zs}

    def check_elements(self, atoms) -> None:
        """Fail loudly if the structure contains an element the model lacks."""
        missing = sorted({s for s, z in zip(atoms.get_chemical_symbols(), atoms.numbers)
                          if int(z) not in self.supported_numbers})
        if missing:
            raise ValueError(f"{self.settings.model} ({self.settings.variant}) does not "
                             f"support element(s) {missing}")

    def info(self) -> dict:
        from importlib.metadata import PackageNotFoundError, version

        def _v(pkg):
            try:
                return version(pkg)
            except PackageNotFoundError:
                return None

        model = self.base.models[0]
        zs = sorted(self.supported_numbers)
        return {
            "model": self.settings.model,
            "pretty": MODELS[self.settings.model].pretty,
            "variant": self.settings.variant,
            "model_path": self.settings.model_path,
            "model_class": type(model).__name__,
            "head": getattr(self.base, "head", None),
            "r_max_A": float(self.base.r_max),
            "z_range": [zs[0], zs[-1]], "n_elements": len(zs),
            "device": self.settings.device, "dtype": self.settings.dtype,
            "charge": self.charge, "spin_multiplicity": self.multiplicity,
            "mace_torch": _v("mace-torch"), "torch": _v("torch"), "ase": _v("ase"),
            "graph_longrange": _v("graph_longrange"),
        }


def build_model(settings: CalcSettings, charge: int, multiplicity: int) -> ModelHandle:
    if settings.threads:
        os.environ["OMP_NUM_THREADS"] = str(settings.threads)
        import torch

        torch.set_num_threads(int(settings.threads))
    spec = MODELS[settings.model]
    variant = settings.model_path or settings.variant
    if settings.model_path and not Path(settings.model_path).exists():
        raise FileNotFoundError(f"model_path does not exist: {settings.model_path}")
    base = spec.builder(variant, settings.device, settings.dtype, dict(settings.options))
    return ModelHandle(settings, base, charge, multiplicity)


def model_from_config(cfg: dict, model=None, variant=None, device=None, dtype=None) -> ModelHandle:
    """Config -> loaded ``ModelHandle`` (charge/multiplicity must be configured)."""
    settings = parse_calc_settings(cfg, model, variant, device, dtype)
    q, m = charge_and_multiplicity(cfg)
    return build_model(settings, q, m)


def add_calc_args(ap) -> None:
    """Standard CLI overrides shared by every stage script."""
    ap.add_argument("--config", default=None, help="config JSON (default: ./config.json)")
    ap.add_argument("--model", default=None, choices=sorted(MODELS))
    ap.add_argument("--variant", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=DTYPES)
