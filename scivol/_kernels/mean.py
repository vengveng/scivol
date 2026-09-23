"""Standalone mean routine registry; mathematics live in the native mean jet."""

import re

from ..components.mean import ARMA, ARX, HARX
from .._mean_fit import DENSITIES, fit_mean_spec
from .routine import Routine


def get_routine(uid):
    match = re.fullmatch(r"(ARMA|ARX|HARX)\(([0-9,]+)\)\+([A-Za-z]+)", uid)
    if match is None:
        raise ValueError(f"Invalid standalone mean specification: {uid}")
    family, orders, density = match.groups()
    orders = tuple(map(int, orders.split(",")))
    mean = (
        ARMA(*orders)
        if family == "ARMA"
        else ARX(*orders) if family == "ARX" else HARX(orders)
    )
    density = next(cls() for cls in DENSITIES if cls.__name__ == density)
    spec = mean + density

    def fit(data, **kwargs):
        return fit_mean_spec(spec, data, **kwargs)

    return Routine(
        uid=uid, fit=fit, n_params=spec.total_params + (density.n_params > 0)
    )
