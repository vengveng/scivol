"""Declared constant-scale mean inputs, without native starts or sample decisions."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .garch_cases import DENSITIES, REQUESTS, SHAPES, SHAPE_NAMES


Y = (0.37, -0.81, 1.23, -0.19, 0.62, -1.41, 0.08, -0.54, 0.93, -0.67, 0.28, -1.09)
LAYOUTS = (
    ("arma11", "arma", 1, 1, 0, True),
    ("arma32", "arma", 3, 2, 0, True),
    ("arma00", "arma", 0, 0, 0, True),
    ("arma00-no-intercept", "arma", 0, 0, 0, False),
    ("arma04-no-intercept", "arma", 0, 4, 0, False),
    ("arma40-no-intercept", "arma", 4, 0, 0, False),
    ("linear-empty", "linear", 0, 0, 0, False),
    ("linear-intercept-only", "linear", 0, 0, 0, True),
    ("linear-three-plus-intercept", "linear", 0, 0, 3, True),
    ("linear-four-explicit-constant", "linear", 0, 0, 4, False),
    ("linear-wide-seven", "linear", 0, 0, 7, False),
    ("linear-wide-seventeen", "linear", 0, 0, 17, False),
)
CHART_LAYOUTS = tuple(layout for layout in LAYOUTS if layout[0] in {
    "arma11", "arma32", "arma00-no-intercept", "arma04-no-intercept",
    "arma40-no-intercept", "linear-three-plus-intercept",
    "linear-four-explicit-constant", "linear-wide-seventeen",
})
SHAPE_COORDINATES = {
    "normal": (), "studentt": (5.1,), "skewt": (5.1, -0.21),
    "ged": (0.24,), "gh": (0.18, -2.32, 0.1), "nig": (0.18, -2.32),
    "hyperbolic": (0.18, -2.32), "variance_gamma": (0.18, -3.8),
}


def external_design(rows, columns, *, constant_column=False):
    time = np.arange(rows, dtype=np.float64)[:, None]
    index = np.arange(columns, dtype=np.float64)[None, :]
    design = 0.6 * np.sin((time + 1.0) * (index + 1.0) * 0.37) + 0.2 * np.cos(index + time * 0.23)
    if constant_column and columns:
        design[:, 0] = 1.0
    return np.ascontiguousarray(design)


def mean_count(case):
    return int(case["intercept"]) + (case["p"] + case["q"] if case["model"] == "arma"
                                     else case["design"].shape[1])


def names(model, density, p, q, columns, intercept):
    result = ["intercept"] if intercept else []
    if model == "arma":
        result += [f"ar[{i + 1}]" for i in range(p)]
        result += [f"ma[{i + 1}]" for i in range(q)]
    else:
        result += [f"coefficient[{i + 1}]" for i in range(columns)]
    return tuple(result + ["sigma2"] + list(SHAPE_NAMES[density]))


def _case(model, density, p, q, intercept, y, design, parameters,
          first, stop, hold_back, case_id):
    y = np.asarray(y, dtype=np.float64).copy()
    begin = max(1, p, q) if model == "arma" else 0
    stop = len(y) if stop is None else stop
    expected_first = hold_back + begin if first is None else first
    result = {
        "case_id": case_id, "model": model, "density": density, "p": p, "q": q,
        "intercept": bool(intercept), "y": y, "design": np.ascontiguousarray(design),
        "parameters": np.asarray(parameters, dtype=np.float64).copy(),
        "theta_names": names(model, density, p, q, design.shape[1], intercept),
        "hold_back": hold_back, "stop": stop, "recursion_begin": begin,
        "first_scored_original_position": expected_first,
        "scored_rows": tuple(range(expected_first, stop)),
        "transformed": False, "original_z": None, "chart": None,
        "scope": "full_data_likelihood", "normalization": "additive_nll",
        "units": "input", "conditioning": "zero_presample_ma_raw_retained_ar",
        "expected_valid_mask": 7,
    }
    if first is not None:
        result["first"] = first
    return result


def arma_case(density="normal", p=1, q=1, *, intercept=True, parameters=None,
              y=Y, first=None, stop=None, hold_back=0, case_id=None):
    if parameters is None:
        parameters = ([0.03] if intercept else [])
        parameters += [0.09 * (-1.0) ** i / (i + 1) for i in range(p)]
        parameters += [-0.13 / (i + 1) for i in range(q)]
        parameters += [1.3] + list(SHAPES[density])
    return _case("arma", density, p, q, intercept, y, np.empty((len(y), 0)), parameters,
                 first, stop, hold_back, case_id or f"arma-{p}{q}-{intercept}-{density}")


def linear_case(density="normal", columns=4, *, intercept=False, design=None,
                constant_column=False, parameters=None, y=Y, first=None, stop=None,
                hold_back=0, case_id=None):
    design = (external_design(len(y), columns, constant_column=constant_column)
              if design is None else np.ascontiguousarray(design, dtype=np.float64))
    if parameters is None:
        coefficients = [0.025 * np.cos(i + 0.3) / (i + 1) for i in range(design.shape[1])]
        parameters = ([0.03] if intercept else []) + coefficients + [1.3] + list(SHAPES[density])
    return _case("linear", density, 0, 0, intercept, y, design, parameters,
                 first, stop, hold_back, case_id or f"linear-{design.shape[1]}-{intercept}-{density}")


def layout_case(layout, density):
    label, model, p, q, columns, intercept = layout
    if model == "arma":
        return arma_case(density, p, q, intercept=intercept, case_id=f"{label}-{density}")
    return linear_case(density, columns, intercept=intercept,
                       constant_column="explicit-constant" in label, case_id=f"{label}-{density}")


def ordinary_cases():
    return tuple(layout_case(layout, density) for layout in LAYOUTS for density in DENSITIES)


def chart_case(template):
    result = deepcopy(template)
    count = mean_count(result)
    z = result["parameters"][:count].copy()
    if result["model"] == "arma":
        ma_start = int(result["intercept"]) + result["p"]
        z[ma_start:] = [0.13 / (i + 1) for i in range(result["q"])]
    z = np.r_[z, 0.7, SHAPE_COORDINATES[result["density"]]]
    result.update(parameters=z.copy(), transformed=True, original_z=z.copy(),
                  chart="constant_mean_interior_v1", case_id=result["case_id"] + "-original-z")
    return result


def original_z_cases():
    return tuple(chart_case(layout_case(layout, density)) for layout in CHART_LAYOUTS for density in DENSITIES)


def scored_suffix(template, first, stop=None, *, hold_back=None):
    result = deepcopy(template)
    stop = result["stop"] if stop is None else stop
    if hold_back is not None:
        result["hold_back"] = hold_back
    result.update(first=first, stop=stop, first_scored_original_position=first,
                  scored_rows=tuple(range(first, stop)), case_id=result["case_id"] + f"-rows{first}:{stop}")
    return result


def expected_dispatch(case):
    if case["model"] == "arma":
        return int(case["intercept"] and case["p"] == case["q"] == 1)
    return int(mean_count(case) == 4)


def long_ma_case():
    y = np.tile(np.array([0.37, -0.81, 1.23, -0.19, 0.62, -1.41, 0.08, -0.54]), 96)
    return arma_case(p=0, q=1, parameters=[0.01, -0.998, 1.3], y=y,
                     case_id="arma01-near-invertibility-long")
