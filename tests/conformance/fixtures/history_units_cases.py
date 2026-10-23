"""History fitting/internal units, independent of native layouts and unit maps."""

from copy import deepcopy
import math

import numpy as np

from .history_cases import DENSITIES, as_chart, case, modes
from .profile_arx_cases import causal_design


VOLATILITIES = ("egarch", "aparch", "garch", "gjr")
MODELS = tuple(prefix + volatility for volatility in VOLATILITIES
               for prefix in ("", "linear_", "arma_", "arx_", "harx_"))
DECLARATIONS = tuple((model, density, mode) for model in MODELS
                     for density in DENSITIES for mode in modes(model))


def unit_case(model, density, mode, *, chart=False, divisor=4.0, causal_intercept=True):
    declared = case(model, density, mode, causal_intercept=causal_intercept)
    if chart:
        declared = as_chart(declared)
    external = declared.get("input_exogenous", declared.get("design", np.empty((len(declared["y"]), 0))))
    divisors = np.asarray([3.0 if column % 2 == 0 else 0.5 for column in range(external.shape[1])])
    declared.update(public_scale=2.0, internal_response_scale=divisor, internal_design_scales=divisors,
                    case_id=declared["case_id"] + f"-units-r{divisor}")
    return declared


def fitting_case(declared):
    """Public scaling defines the fitted data, but leaves the implicit log seed zero."""
    result = deepcopy(declared)
    scale = declared["public_scale"]
    result["y"] *= scale
    result["h0"] *= scale * scale
    for field in ("observation_history", "innovation_history"):
        result[field] = tuple(scale * value for value in declared[field])
    if declared.get("causal_kind"):
        result["design"] = causal_design(
            result["y"], result["input_exogenous"], result["causal_order"], result["hold_back"],
            observation_history=result["observation_history"],
            horizons=result["causal_horizons"] if result["causal_kind"] == "harx" else None)
    result["transformed"] = False
    if declared["model"].endswith("egarch"):
        result["implicit_log_variance"] = 0.0
    return result


def design_divisors(declared):
    generated = (len(declared["causal_horizons"]) if declared.get("causal_kind") == "harx"
                 else declared["causal_order"] if declared.get("causal_kind") == "arx" else 0)
    return np.r_[np.full(generated, declared["internal_response_scale"]), declared["internal_design_scales"]]


def internal_case(declared):
    result = fitting_case(declared)
    divisor = declared["internal_response_scale"]
    result["y"] /= divisor
    result["h0"] /= divisor * divisor
    for field in ("observation_history", "innovation_history"):
        result[field] = tuple(value / divisor for value in result[field])
    if "design" in result:
        result["design"] /= design_divisors(declared)
    if "input_exogenous" in result:
        result["input_exogenous"] /= declared["internal_design_scales"]
    if declared["model"].endswith("egarch"):
        result["implicit_log_variance"] = -2 * math.log(divisor)
    return result


def internal_theta_map(declared):
    from oracles.garch import jax_modules
    _, jnp, _ = jax_modules()
    family = declared["model"].rsplit("_", 1)[-1]
    if family not in VOLATILITIES:
        raise ValueError("No declared history unit map for this volatility")
    names = declared["theta_names"]
    width = names.index("omega")
    beta = np.asarray([index for index, name in enumerate(names) if name.startswith("beta[")], dtype=int)
    delta = names.index("delta") if family == "aparch" else None
    divisor = declared["internal_response_scale"]
    factors = np.ones(len(declared["parameters"]))
    if width:
        start = int(declared["intercept"])
        if start:
            factors[0] = 1 / divisor
        if declared["model"].startswith("linear_"):
            factors[start:width] = design_divisors(declared) / divisor
    log_divisor = math.log(divisor)

    def mapping(theta):
        scaled = theta * jnp.asarray(factors)
        if family == "egarch":
            omega = theta[width] - 2 * log_divisor * (1 - jnp.sum(theta[beta]))
        elif family == "aparch":
            omega = theta[width] * jnp.exp(-theta[delta] * log_divisor)
        else:
            omega = theta[width] / (divisor * divisor)
        return scaled.at[width].set(omega)

    return mapping
