from __future__ import annotations

import re
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from ..components.density import GED, Normal, SkewT, StudentT
from ..components.mean import ARMA, ARX, HARX
from ..components.vol import GARCH, GJRGARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .routine import Routine

_CACHE: Dict[str, Routine] = {}

_DENSITY_MAP = {
    "Normal": Normal,
    "StudentT": StudentT,
    "SkewT": SkewT,
    "GED": GED,
}


def _parse_uid(uid: str) -> CompositeSpec:
    components = []
    for part in uid.split("+"):
        if part.startswith("ARMA("):
            match = re.fullmatch(r"ARMA\((\d+),(\d+)\)", part)
            if match is None:
                raise RuntimeError(f"generic kernel could not parse mean component '{part}'")
            components.append(ARMA(int(match.group(1)), int(match.group(2))))
        elif part.startswith("ARX("):
            match = re.fullmatch(r"ARX\((\d+)\)", part)
            if match is None:
                raise RuntimeError(f"generic kernel could not parse mean component '{part}'")
            components.append(ARX(int(match.group(1))))
        elif part.startswith("HARX("):
            match = re.fullmatch(r"HARX\(([\d,]+)\)", part)
            if match is None:
                raise RuntimeError(f"generic kernel could not parse mean component '{part}'")
            horizons = tuple(int(token) for token in match.group(1).split(","))
            components.append(HARX(horizons))
        elif part.startswith("GJR-GARCH("):
            match = re.fullmatch(r"GJR-GARCH\((\d+),(\d+)\)", part)
            if match is None:
                raise RuntimeError(f"generic kernel could not parse volatility component '{part}'")
            components.append(GJRGARCH(int(match.group(1)), int(match.group(2))))
        elif part.startswith("GARCH("):
            match = re.fullmatch(r"GARCH\((\d+),(\d+)\)", part)
            if match is None:
                raise RuntimeError(f"generic kernel could not parse volatility component '{part}'")
            components.append(GARCH(int(match.group(1)), int(match.group(2))))
        else:
            density_cls = _DENSITY_MAP.get(part)
            if density_cls is None:
                raise RuntimeError(f"generic kernel does not know how to parse '{part}'")
            components.append(density_cls())
    return CompositeSpec(*components)
def _build(uid: str) -> Routine:
    template = _parse_uid(uid)

    def fit(
        y: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = False,
        verbose: bool = False,
        x: object = None,
        **_: object,
    ) -> EstimationResult:
        raise NotImplementedError(
            f"{uid} does not yet expose a contract-compliant `fit()` path. "
            "This family currently only supports fixed-parameter workflows "
            "such as `loglikelihood()`, `filter()`, `fix()`, `forecast()`, and "
            "`simulate()`. A public fitting path will be enabled only after the "
            "family has dedicated analytical derivative kernels and an honest "
            "optimization policy instead of generic numerical-difference fallback."
        )

    return Routine(uid=uid, fit=fit, n_params=template.total_params)


def get_routine(uid: str) -> Routine:
    if uid not in _CACHE:
        _CACHE[uid] = _build(uid)
    return _CACHE[uid]
