"""Binary64 inputs independent of native layouts, starts, seeds and samples."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


DENSITIES = ("normal", "studentt", "skewt", "ged", "gh", "nig", "hyperbolic", "variance_gamma")
FAMILIES = ("garch", "gjr")
ORDERS = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (2, 3), (4, 3))
REQUESTS = tuple(range(1, 8))
Y = (0.37, -0.81, 1.23, -0.19, 0.62, -1.41, 0.08, -0.54)
SHAPES = {
    "normal": (), "studentt": (7.4,), "skewt": (7.4, -0.23),
    "ged": (0.8,), "gh": (0.18, 2.1, 0.35), "nig": (0.18, 2.1),
    "hyperbolic": (0.18, 2.1), "variance_gamma": (0.18, 2.4),
}
SHAPE_NAMES = {
    "normal": (), "studentt": ("nu",), "skewt": ("nu", "lambda"),
    "ged": ("nu",), "gh": ("rho", "zeta", "lambda"),
    "nig": ("rho", "zeta"), "hyperbolic": ("rho", "zeta"),
    "variance_gamma": ("rho", "lambda"),
}


def parameter_names(model, density, p, q):
    names = ["omega"] + [f"alpha[{i + 1}]" for i in range(p)]
    if model == "gjr":
        names += [f"gamma[{i + 1}]" for i in range(p)]
    return tuple(names + [f"beta[{i + 1}]" for i in range(q)] + list(SHAPE_NAMES[density]))


def case(model="garch", density="normal", p=1, q=1, *, parameters=None,
         y=Y, h0=0.93, first=0, stop=None, hold_back=0, case_id=None):
    y = np.asarray(y, dtype=np.float64)
    if parameters is None:
        alpha = [0.085 / (i + 1) for i in range(p)]
        gamma = [-0.025 / (i + 1) for i in range(p)] if model == "gjr" else []
        beta = [0.38 / (i + 1) for i in range(q)]
        parameters = [0.11] + alpha + gamma + beta + list(SHAPES[density])
    stop = y.size if stop is None else stop
    return {
        "case_id": case_id or f"{model}-{density}-{p}{q}",
        "model": model, "density": density, "p": p, "q": q,
        "parameters": np.asarray(parameters, dtype=np.float64),
        "theta_names": parameter_names(model, density, p, q),
        "y": y.copy(), "h0": float(h0), "first": first, "stop": stop,
        "hold_back": hold_back, "scored_rows": tuple(range(first, stop)),
        "transformed": False, "chart": None, "original_z": None,
        "scope": "full_data_likelihood", "normalization": "additive_nll",
        "units": "input", "seed_policy": "explicit_fixed_variance_at_retained_zero",
        "missing_lags": "zero_contribution", "expected_valid_mask": 7,
    }


def ordinary_cases():
    return tuple(case(model, density, p, q) for model in FAMILIES
                 for density in DENSITIES for p, q in ORDERS)


def zero_order_cases():
    """Inventory only until the chief declares the new public zero-order domain."""
    return tuple(case(model, density, p, q) for model in FAMILIES
                 for density in DENSITIES for p, q in ((0, 0), (0, 2), (2, 0)))


def with_parameters(template, parameters, **updates):
    result = deepcopy(template)
    result["parameters"] = np.asarray(parameters, dtype=np.float64)
    result.update(updates)
    return result


def chart_case(model, density, p=1, q=1):
    count = (2 if model == "gjr" else 1) * p + q
    z = [-1.7] + [(-0.9 + 0.23 * i) for i in range(count)]
    shapes = {
        "normal": [], "studentt": [5.1], "skewt": [5.1, -0.21],
        "ged": [0.24], "gh": [0.18, -2.32, 0.1],
        "nig": [0.18, -2.32], "hyperbolic": [0.18, -2.32],
        "variance_gamma": [0.18, -3.8],
    }
    z = np.asarray(z + shapes[density], dtype=np.float64)
    return with_parameters(case(model, density, p, q), z,
                           transformed=True, chart="native_interior_v1", original_z=z.copy(),
                           case_id=f"{model}-{density}-{p}{q}-original-z")


def long_horizon_case(model, density):
    pattern = np.asarray([0.61, -0.83, 0.17, -0.29, 1.07, -0.46, 0.31, -0.72])
    y = np.tile(pattern, 128)
    vol = [2.0e-5, 2.0e-4] + ([1.0e-4] if model == "gjr" else []) + [0.99965]
    return case(model, density, parameters=vol + list(SHAPES[density]), y=y,
                h0=0.72, case_id=f"{model}-{density}-long-horizon-near-boundary")


def tiny_scale_case(model, density):
    template = case(model, density)
    theta = template["parameters"].copy()
    theta[0] *= 1e-20
    return case(model, density, parameters=theta, y=np.asarray(Y) * 1e-10,
                h0=0.93e-20, case_id=f"{model}-{density}-below-historical-floor")


def rejection_cases():
    results = []
    for model in FAMILIES:
        for density in DENSITIES:
            template = case(model, density)
            point = template["parameters"].copy()
            if density == "normal":
                point[0] = 0.0
            elif density == "studentt":
                point[-1] = 2.0
            elif density == "skewt":
                point[-1] = 1.0
            elif density == "ged":
                point[-1] = 0.0
            else:
                point[-len(SHAPES[density])] = 1.0
            results.append(with_parameters(template, point, expected_valid_mask=0,
                                           case_id=template["case_id"] + "-outside-domain"))
    for index, value in ((0, -0.01), (1, -0.01), (2, -0.01), (2, 0.96)):
        template = case()
        point = template["parameters"].copy()
        point[index] = value
        results.append(with_parameters(template, point, expected_valid_mask=0,
                                       case_id=f"garch-invalid-coefficient-{index}-{value}"))
    template = case("gjr")
    point = template["parameters"].copy()
    point[2] = -point[1] - 0.01
    results.append(with_parameters(template, point, expected_valid_mask=0,
                                   case_id="gjr-negative-combined-shock"))
    return tuple(results)
