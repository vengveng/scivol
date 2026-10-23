"""Joint inputs declared without native layouts, starts or sample metadata."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .garch_cases import DENSITIES, REQUESTS, SHAPES, SHAPE_NAMES
from .mean_cases import SHAPE_COORDINATES, Y, external_design


FAMILIES = ("arma_garch", "arma_gjr", "linear_garch", "linear_gjr")
LAYOUTS = (
    ("arma11-vol11", "arma", 1, 1, 1, 1, 0, True),
    ("arma32-vol23", "arma", 3, 2, 2, 3, 0, True),
    ("arma04-vol32", "arma", 0, 4, 3, 2, 0, False),
    ("arma40-vol12", "arma", 4, 0, 1, 2, 0, False),
    ("arma00-vol43", "arma", 0, 0, 4, 3, 0, False),
    ("arma11-no-intercept-vol21", "arma", 1, 1, 2, 1, 0, False),
    ("linear-three-plus-intercept-vol11", "linear", 0, 0, 1, 1, 3, True),
    ("linear-four-explicit-constant-vol11", "linear", 0, 0, 1, 1, 4, False),
    ("linear-six-plus-intercept-vol11", "linear", 0, 0, 1, 1, 6, True),
    ("linear-seven-no-intercept-vol11", "linear", 0, 0, 1, 1, 7, False),
    ("linear-wide-seventeen-vol32", "linear", 0, 0, 3, 2, 17, False),
    ("linear-empty-vol21", "linear", 0, 0, 2, 1, 0, False),
    ("linear-intercept-only-vol12", "linear", 0, 0, 1, 2, 0, True),
)
CHART_LAYOUTS = tuple(item for item in LAYOUTS if item[0] in {
    "arma11-vol11", "arma32-vol23", "arma04-vol32",
    "linear-three-plus-intercept-vol11", "linear-six-plus-intercept-vol11",
    "linear-wide-seventeen-vol32",
})


def mean_width(case):
    return int(case["intercept"]) + (case["p_ar"] + case["q_ma"]
        if case["model"].startswith("arma_") else case["design"].shape[1])


def volatility_width(case):
    return 1 + (2 if case["model"].endswith("gjr") else 1) * case["p_vol"] + case["q_vol"]


def case(model="arma_garch", density="normal", p_ar=1, q_ma=1,
         p_vol=1, q_vol=1, *, columns=3, intercept=True, y=Y, design=None,
         parameters=None, h0=0.93, first=None, stop=None, hold_back=0,
         constant_column=False, case_id=None):
    if model not in FAMILIES:
        raise ValueError(model)
    arma, leverage = model.startswith("arma_"), model.endswith("gjr")
    if not arma:
        p_ar = q_ma = 0
    y = np.asarray(y, dtype=np.float64).copy()
    design = (np.empty((len(y), 0)) if arma else
              external_design(len(y), columns, constant_column=constant_column)
              if design is None else np.asarray(design, dtype=np.float64))
    design = np.ascontiguousarray(design).copy()
    names = ["intercept"] if intercept else []
    mean = [0.03] if intercept else []
    if arma:
        names += [f"ar[{i + 1}]" for i in range(p_ar)] + [f"ma[{i + 1}]" for i in range(q_ma)]
        mean += [0.09 * (-1.0) ** i / (i + 1) for i in range(p_ar)]
        mean += [-0.13 / (i + 1) for i in range(q_ma)]
    else:
        names += [f"coefficient[{i + 1}]" for i in range(design.shape[1])]
        mean += [0.025 * np.cos(i + 0.3) / (i + 1) for i in range(design.shape[1])]
    names += ["omega"] + [f"alpha[{i + 1}]" for i in range(p_vol)]
    if leverage:
        names += [f"gamma[{i + 1}]" for i in range(p_vol)]
    names += [f"beta[{i + 1}]" for i in range(q_vol)] + list(SHAPE_NAMES[density])
    if parameters is None:
        parameters = mean + [0.11] + [0.085 / (i + 1) for i in range(p_vol)]
        parameters += [-0.025 / (i + 1) for i in range(p_vol)] if leverage else []
        parameters += [0.38 / (i + 1) for i in range(q_vol)] + list(SHAPES[density])
    begin = max(p_ar, q_ma, p_vol, q_vol) if arma else 0
    stop = len(y) if stop is None else stop
    first_position = hold_back + begin if first is None else first
    row_ids = np.arange(len(y), dtype=np.uint64)
    result = dict(
        case_id=case_id or f"{model}-ar{p_ar}ma{q_ma}-vol{p_vol}{q_vol}-k{design.shape[1]}-{intercept}-{density}",
        model=model, density=density, p_ar=p_ar, q_ma=q_ma, p_vol=p_vol, q_vol=q_vol,
        intercept=bool(intercept), design=design, y=y, rows=row_ids,
        parameters=np.asarray(parameters, dtype=np.float64).copy(), theta_names=tuple(names),
        h0=float(h0), hold_back=hold_back, stop=stop, recursion_begin=begin,
        first_scored_original_position=first_position,
        scored_positions=tuple(range(first_position, stop)),
        scored_rows=tuple(row_ids[first_position:stop]),
        transformed=False, original_z=None, chart=None,
        scope="full_data_likelihood", normalization="additive_nll", units="input",
        seed_policy="explicit_fixed_conditioning_variances",
        conditioning="zero_innovations_raw_retained_ar" if arma else "seed_at_retained_zero",
        expected_valid_mask=7,
    )
    if first is not None:
        result["first"] = first
    return result


def layout_case(layout, leverage, density):
    label, mean, p_ar, q_ma, p_vol, q_vol, columns, intercept = layout
    return case(f"{mean}_{'gjr' if leverage else 'garch'}", density,
                p_ar, q_ma, p_vol, q_vol, columns=columns, intercept=intercept,
                constant_column="explicit-constant" in label,
                case_id=f"{label}-{'gjr' if leverage else 'garch'}-{density}")


def ordinary_cases():
    return tuple(layout_case(layout, leverage, density) for layout in LAYOUTS
                 for leverage in (False, True) for density in DENSITIES)


def chart_case(template):
    result = deepcopy(template)
    width = mean_width(result)
    mean_z = result["parameters"][:width].copy()
    if result["model"].startswith("arma_"):
        start = int(result["intercept"]) + result["p_ar"]
        mean_z[start:] = [0.13 / (i + 1) for i in range(result["q_ma"])]
    count = volatility_width(result) - 1
    z = np.r_[mean_z, -1.7, [-0.9 + 0.23 * i for i in range(count)],
              SHAPE_COORDINATES[result["density"]]]
    result.update(parameters=z.copy(), original_z=z.copy(), transformed=True,
                  chart="joint_interior_v1", case_id=result["case_id"] + "-original-z")
    return result


def original_z_cases():
    return tuple(chart_case(layout_case(layout, leverage, density)) for layout in CHART_LAYOUTS
                 for leverage in (False, True) for density in DENSITIES)


def scored_suffix(template, first, stop=None, *, hold_back=None):
    result = deepcopy(template)
    stop = result["stop"] if stop is None else stop
    if hold_back is not None:
        result["hold_back"] = hold_back
    result.update(first=first, stop=stop, first_scored_original_position=first,
                  scored_positions=tuple(range(first, stop)),
                  scored_rows=tuple(result["rows"][first:stop]),
                  case_id=result["case_id"] + f"-rows{first}:{stop}")
    return result


def expected_dispatch(case):
    if case["transformed"]:
        # The shipped chart helpers retain runtime order metadata.
        return 0
    common_vol = case["p_vol"] == case["q_vol"] == 1
    if case["model"].startswith("arma_"):
        return int(common_vol and case["intercept"] and case["p_ar"] == case["q_ma"] == 1)
    return int(common_vol and mean_width(case) in (4, 7))


def long_horizon_case(model, density):
    y = np.tile(np.array([0.37, -0.81, 1.23, -0.19, 0.62, -1.41, 0.08, -0.54]), 32)
    result = case(model, density, p_ar=0, q_ma=1, columns=0, y=y, h0=0.72,
                  case_id=f"{model}-{density}-long-near-boundary")
    width = mean_width(result)
    result["parameters"][:width] = [0.01, -0.998] if model.startswith("arma_") else [0.01]
    vol = [2e-5, 2e-4] + ([1e-4] if model.endswith("gjr") else []) + [0.99965]
    result["parameters"][width:width + volatility_width(result)] = vol
    return result


def scaled_case(template, factor):
    result = deepcopy(template)
    width = mean_width(result)
    result["y"] *= factor
    if result["model"].startswith("arma_"):
        if result["intercept"]:
            result["parameters"][0] *= factor
    else:
        result["parameters"][:width] *= factor
    result["parameters"][width] *= factor * factor
    result["h0"] *= factor * factor
    result["case_id"] += f"-units{factor}"
    return result
