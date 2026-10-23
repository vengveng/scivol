"""Caller-built causal ARX designs for the private Normal profile transport."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .mean_cases import linear_case
from .profile_cases import profile_case


LAYOUTS = (
    (1, 2, True, 1, None),
    (2, 1, True, 5, 8),
    (2, 1, False, 2, 2),
    (3, 0, False, 3, 5),
    (7, 2, True, 9, None),
)
Y = np.asarray([(((11 * t + 5) % 29) - 14) / 9 + 0.03 * ((t % 3) - 1) for t in range(28)])


def causal_design(y, external, order, hold_back, *, observation_history=(), horizons=None):
    """Independent retained AR lags or HAR windows, then caller exogenous data."""
    windows = tuple(range(1, order + 1)) if horizons is None else tuple(horizons)
    width = len(windows)
    design = np.zeros((len(y), width + external.shape[1]), dtype=np.float64)
    for row in range(hold_back, len(y)):
        for column, horizon in enumerate(windows):
            lags = (horizon,) if horizons is None else range(1, horizon + 1)
            terms = [float(y[row - lag]) if row - lag >= hold_back
                     else float(observation_history[hold_back - (row - lag) - 1])
                     for lag in lags if row - lag >= hold_back or len(observation_history)]
            design[row, column] = (terms[0] if horizons is None else math.fsum(terms) / len(terms)) if terms else 0.0
        design[row, width:] = external[row]
    return design


def reflection_values(z):
    coefficients = np.zeros(len(z))
    for stage, coordinate in enumerate(z):
        previous = coefficients.copy()
        kappa = math.tanh(float(coordinate))
        for row in range(stage):
            coefficients[row] = previous[row] - kappa * previous[stage - row - 1]
        coefficients[stage] = kappa
    return coefficients


def arx_profile_case(order=2, exog=1, intercept=True, hold_back=5, first=8):
    if order < 1 or exog < 0 or hold_back < order:
        raise ValueError("Declare positive AR order, nonnegative exog width and sufficient physical trim")
    y = Y.copy()
    # Discarded data must not leak into the first retained lag feature.
    y[:hold_back] = [(-1) ** row * (row + 1) * 1e200 for row in range(hold_back)]
    external = np.asarray([[(float((t + 3 * j) % 7) - 3) / (j + 4)
                            for j in range(exog)] for t in range(len(y))], dtype=np.float64)
    design = causal_design(y, external, order, hold_back)
    ar_z = np.asarray([(-1) ** j * 0.21 / (j + 1) for j in range(order)])
    if order == 2 and intercept:
        ar_z = np.arctanh(np.asarray([0.86, -0.4]))
    z = np.r_[([0.07] if intercept else []), ar_z,
              [(-1) ** j * 0.03 / (j + 1) for j in range(exog)]]
    theta = z.copy()
    start = int(intercept)
    theta[start:start + order] = reflection_values(ar_z)
    label = f"arx-p{order}-x{exog}-i{int(intercept)}-hold{hold_back}-first{first}"
    case = profile_case(linear_case("normal", intercept=intercept, y=y, design=design,
                                   parameters=np.r_[theta, 1.3], hold_back=hold_back,
                                   first=first, stop=27, case_id=label))
    case.update(linear_ar_order=order, generated_design_width=order, input_exogenous_width=exog,
                input_exogenous=external, original_z_seed=z, recursion_begin=0,
                conditioning="caller_built_trim_then_reset_causal_design",
                theta_names=tuple((["intercept"] if intercept else [])
                                  + [f"ar[{lag}]" for lag in range(1, order + 1)]
                                  + [f"x[{column}]" for column in range(1, exog + 1)]))
    return case


def arx_profile_cases():
    return tuple(arx_profile_case(*layout) for layout in LAYOUTS)


def as_original_z(case):
    result = deepcopy(case)
    z = result["original_z_seed"].copy()
    result.update(parameters=z, original_z=z.copy(), transformed=True,
                  case_id=case["case_id"] + "-original-z")
    return result


def plateau_case(sign, coordinate):
    result = as_original_z(arx_profile_case(1, 2, True, 3, 6))
    result["parameters"][1] = sign * coordinate
    result["original_z"] = result["parameters"].copy()
    result["case_id"] = f"profile-arx-original-z-plateau-{sign:+d}-{coordinate}"
    return result
