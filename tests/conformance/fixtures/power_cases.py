"""Independent, explicit zero-mean power-model conditioning and parameter points."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .garch_cases import DENSITIES, REQUESTS, SHAPES, SHAPE_NAMES, Y


FAMILIES = ("egarch", "aparch")
ORDERS = ((1, 1), (3, 2), (2, 3), (4, 3))
SHAPE_Z = {
    "normal": (), "studentt": (5.1,), "skewt": (5.1, -0.21),
    "ged": (0.24,), "gh": (0.18, -2.32, 0.1),
    "nig": (0.18, -2.32), "hyperbolic": (0.18, -2.32),
    "variance_gamma": (0.18, -3.8),
}


def parameter_names(model, density, p, q):
    names = (["omega"] + [f"alpha[{i + 1}]" for i in range(p)]
             + [f"gamma[{i + 1}]" for i in range(p)]
             + [f"beta[{i + 1}]" for i in range(q)])
    if model == "aparch":
        names.append("delta")
    return tuple(names + list(SHAPE_NAMES[density]))


def case(model="egarch", density="normal", p=1, q=1, *, parameters=None,
         y=Y, h0=0.93, first=0, stop=None, hold_back=0, case_id=None):
    if model not in FAMILIES or density not in DENSITIES:
        raise ValueError("Unknown power model or density")
    if p <= 0 or q <= 0:
        raise ValueError("The declared standalone power scope has positive p and q")
    y = np.asarray(y, dtype=np.float64)
    stop = len(y) if stop is None else stop
    if y.ndim != 1 or not 0 <= hold_back <= first < stop <= len(y):
        raise ValueError("Invalid explicit retained/scored sample")
    if parameters is None:
        alpha = [0.085 / (i + 1) for i in range(p)]
        gamma = [-0.16 / (i + 1) for i in range(p)]
        beta = [0.38 / (i + 1) for i in range(q)]
        parameters = [(-0.11 if model == "egarch" else 0.11)] + alpha + gamma + beta
        parameters += ([1.7] if model == "aparch" else []) + list(SHAPES[density])
    names = parameter_names(model, density, p, q)
    if len(parameters) != len(names):
        raise ValueError("Power parameter width does not match the declared order")
    return dict(
        case_id=case_id or f"{model}-{density}-{p}{q}", model=model, density=density,
        p=p, q=q, y=y.copy(), h0=float(h0), parameters=np.asarray(parameters, dtype=np.float64),
        theta_names=names, first=first, stop=stop, hold_back=hold_back,
        scored_rows=tuple(range(first, stop)), transformed=False, original_z=None,
        chart=None, normalization="additive_nll", mean="zero", units="input",
        seed_policy="explicit_theta_independent_variance_at_retained_zero",
        missing_lags="zero_contribution", expected_valid_mask=7, power_log_units=0.0,
    )


def with_parameters(template, parameters, **updates):
    declared = deepcopy(template)
    declared["parameters"] = np.asarray(parameters, dtype=np.float64)
    declared.update(updates)
    return declared


def ordinary_cases():
    return tuple(case(model, density, p, q) for model in FAMILIES
                 for density in DENSITIES for p, q in ORDERS)


def chart_case(model, density, p=1, q=1):
    if model == "egarch":
        z = ([-0.11] + [0.085 / (i + 1) for i in range(p)]
             + [-0.16 / (i + 1) for i in range(p)] + [0.23 / (i + 1) for i in range(q)])
    else:
        z = ([-1.7] + [-1.1 + 0.13 * i for i in range(p)]
             + [-0.16 / (i + 1) for i in range(p)]
             + [-0.8 + 0.11 * i for i in range(q)]
             + ([-1.2] if density in {"studentt", "skewt"} else [1.3]))
    z = np.asarray(z + list(SHAPE_Z[density]), dtype=np.float64)
    return with_parameters(case(model, density, p, q), z, transformed=True,
                           original_z=z.copy(), chart="power_unshifted_exact_slack_v1",
                           case_id=f"{model}-{density}-{p}{q}-original-z")


def long_horizon_case(model, density):
    y = np.tile([0.61, -0.83, 0.17, -0.29, 1.07, -0.46, 0.31, -0.72], 128)
    point = ([-2e-4, 2e-4, -1e-4, 0.9995] if model == "egarch"
             else [2e-5, 1e-4, -0.12, 0.9995, 1.7])
    return case(model, density, parameters=point + list(SHAPES[density]), y=y, h0=0.72,
                first=17, case_id=f"{model}-{density}-long-near-boundary")


def seed_delta_case(p=1, q=1):
    point = [1e200] + [0.0] * (2 * p) + [0.5] + [0.0] * (q - 1) + [2.0]
    declared = case("aparch", p=p, q=q, parameters=point, y=np.zeros(3), h0=1e200,
                    case_id=f"aparch-normal-{p}{q}-large-seed-delta")
    # Fixed units keep AD intermediates representable without changing equations.
    declared["power_log_units"] = math.log(declared["h0"]) / 2.0
    return declared


def small_power_case(density="normal", p=1, q=1):
    declared = case("aparch", density, p, q, y=[0.0, -0.37, 0.0, 0.81, -0.19])
    declared["parameters"][1 + 2 * p + q] = 0.37
    declared["case_id"] += "-small-delta-fixed-zeros"
    return declared


def tiny_scale_case(model, p=1, q=1):
    scale = 2.0 ** -40
    declared = case(model, p=p, q=q, y=np.asarray(Y) * scale, h0=0.93 * scale * scale)
    point = declared["parameters"].copy()
    if model == "egarch":
        point[0] += (1.0 - sum(point[1 + 2 * p:1 + 2 * p + q])) * 2.0 * math.log(scale)
    else:
        point[0] *= scale ** point[1 + 2 * p + q]
    declared["parameters"] = point
    declared["case_id"] += "-below-historical-variance-floor"
    return declared


def domain_cases():
    """True domain violations, distinct from floating-point support or kinks."""
    results = []
    for model in FAMILIES:
        for density in DENSITIES:
            declared = case(model, density)
            point = declared["parameters"].copy()
            if density == "normal":
                point[1 + 2 * declared["p"]] = 1.1
            elif density == "studentt":
                point[-1] = 2.0
            elif density == "skewt":
                point[-1] = 1.0
            elif density == "ged":
                point[-1] = 0.0
            else:
                point[-len(SHAPES[density])] = 1.0
            results.append(with_parameters(declared, point, case_id=declared["case_id"] + "-domain"))
    for index, value, label in ((0, 0.0, "omega"), (1, -0.1, "alpha"),
                                (2, 1.0, "gamma"), (4, 0.0, "delta")):
        declared = case("aparch")
        point = declared["parameters"].copy()
        point[index] = value
        results.append(with_parameters(declared, point, case_id=f"aparch-normal-invalid-{label}"))
    for density in ("studentt", "skewt"):
        declared = case("aparch", density)
        point = declared["parameters"].copy()
        point[4] = point[5]
        results.append(with_parameters(declared, point, case_id=f"aparch-{density}-moment-tail-pole"))
    return tuple(results)
