"""Declared joint mean/power inputs and different ARMA/linear startup policies."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .garch_cases import DENSITIES, REQUESTS, SHAPES
from .mean_cases import Y, external_design
from .power_cases import chart_case as power_chart_case, parameter_names as power_names


FAMILIES = ("arma_egarch", "arma_aparch", "linear_egarch", "linear_aparch")
LAYOUTS = (
    ("arma11-vol11", "arma", 1, 1, 1, 1, 0, True),
    ("arma32-vol23", "arma", 3, 2, 2, 3, 0, True),
    ("arma02-no-intercept-vol32", "arma", 0, 2, 3, 2, 0, False),
    ("arma30-no-intercept-vol43", "arma", 3, 0, 4, 3, 0, False),
    ("arma00-no-intercept-vol43", "arma", 0, 0, 4, 3, 0, False),
    ("arma11-no-intercept-vol32", "arma", 1, 1, 3, 2, 0, False),
    ("linear-three-plus-intercept-vol11", "linear", 0, 0, 1, 1, 3, True),
    ("linear-four-explicit-constant-vol11", "linear", 0, 0, 1, 1, 4, False),
    ("linear-five-no-intercept-vol23", "linear", 0, 0, 2, 3, 5, False),
    ("linear-wide-seventeen-vol32", "linear", 0, 0, 3, 2, 17, False),
    ("linear-empty-vol43", "linear", 0, 0, 4, 3, 0, False),
    ("linear-intercept-only-vol32", "linear", 0, 0, 3, 2, 0, True),
)
CHART_LAYOUTS = tuple(item for item in LAYOUTS if item[0] != "linear-empty-vol43")


def mean_width(case):
    return int(case["intercept"]) + (case["p_ar"] + case["q_ma"]
        if case["model"].startswith("arma_") else case["design"].shape[1])


def volatility_width(case):
    return 1 + 2 * case["p_vol"] + case["q_vol"] + int(case["model"].endswith("aparch"))


def case(model="arma_egarch", density="normal", p_ar=None, q_ma=None,
         p_vol=1, q_vol=1, *, columns=3, intercept=True, y=Y, design=None,
         parameters=None, h0=0.93, first=None, stop=None, hold_back=0,
         constant_column=False, case_id=None):
    if model not in FAMILIES or density not in DENSITIES:
        raise ValueError("Unknown joint power model or density")
    arma = model.startswith("arma_")
    p_ar = (1 if arma else 0) if p_ar is None else p_ar
    q_ma = (1 if arma else 0) if q_ma is None else q_ma
    if min(p_ar, q_ma) < 0 or min(p_vol, q_vol) <= 0 or (not arma and (p_ar or q_ma)):
        raise ValueError("Invalid declared mean or volatility order")
    y = np.asarray(y, dtype=np.float64).copy()
    if y.ndim != 1:
        raise ValueError("Joint observations must be one-dimensional")
    if arma:
        if design is not None and np.asarray(design).shape != (len(y), 0):
            raise ValueError("ARMA cannot consume external design columns")
        design = np.empty((len(y), 0))
    elif design is None:
        design = external_design(len(y), columns, constant_column=constant_column)
    design = np.ascontiguousarray(design, dtype=np.float64).copy()
    if design.ndim != 2 or design.shape[0] != len(y):
        raise ValueError("Design rows must match the observations")
    names, mean = (["intercept"], [0.03]) if intercept else ([], [])
    if arma:
        names += [f"ar[{i + 1}]" for i in range(p_ar)] + [f"ma[{i + 1}]" for i in range(q_ma)]
        mean += [0.09 * (-1.0) ** i / (i + 1) for i in range(p_ar)]
        mean += [-0.13 / (i + 1) for i in range(q_ma)]
    else:
        names += [f"coefficient[{i + 1}]" for i in range(design.shape[1])]
        mean += [0.025 * np.cos(i + 0.3) / (i + 1) for i in range(design.shape[1])]
    volatility = model.split("_", 1)[1]
    names += list(power_names(volatility, density, p_vol, q_vol))
    if parameters is None:
        parameters = mean + [-0.11 if volatility == "egarch" else 0.11]
        parameters += [0.085 / (i + 1) for i in range(p_vol)]
        parameters += [-0.16 / (i + 1) for i in range(p_vol)]
        parameters += [0.38 / (i + 1) for i in range(q_vol)]
        parameters += ([1.7] if volatility == "aparch" else []) + list(SHAPES[density])
    if len(parameters) != len(names):
        raise ValueError("Joint parameter width does not match the declared layout")
    begin = max(p_ar, q_ma, p_vol, q_vol) if arma else 0
    stop = len(y) if stop is None else stop
    first_position = hold_back + begin if first is None else first
    if not 0 <= hold_back <= first_position < stop <= len(y) or first_position < hold_back + begin:
        raise ValueError("Invalid retained, conditioned or scored sample")
    rows = np.arange(len(y), dtype=np.uint64)
    result = dict(
        case_id=case_id or f"{model}-ar{p_ar}ma{q_ma}-vol{p_vol}{q_vol}-k{design.shape[1]}-{intercept}-{density}",
        model=model, density=density, p_ar=p_ar, q_ma=q_ma, p_vol=p_vol, q_vol=q_vol,
        intercept=bool(intercept), y=y, design=design, rows=rows,
        parameters=np.asarray(parameters, dtype=np.float64).copy(), theta_names=tuple(names),
        h0=float(h0), hold_back=hold_back, stop=stop, recursion_begin=begin,
        first_scored_original_position=first_position,
        scored_positions=tuple(range(first_position, stop)), scored_rows=tuple(rows[first_position:stop]),
        transformed=False, original_z=None, chart=None, expected_valid_mask=7,
        normalization="additive_nll", units="input", power_log_units=0.0,
        seed_policy="seed_histories_then_first_recurrence" if arma else "fixed_first_variance",
        innovation_conditioning="zero_before_recursion_begin" if arma else "omitted_unavailable_lags",
    )
    if first is not None:
        result["first"] = first
    return result


def layout_case(layout, volatility, density):
    label, mean, ar, ma, p, q, columns, intercept = layout
    return case(f"{mean}_{volatility}", density, ar, ma, p, q, columns=columns,
                intercept=intercept, constant_column="explicit-constant" in label,
                case_id=f"{label}-{volatility}-{density}")


def ordinary_cases():
    return tuple(layout_case(layout, volatility, density) for layout in LAYOUTS
                 for volatility in ("egarch", "aparch") for density in DENSITIES)


def chart_case(template):
    declared = deepcopy(template)
    width = mean_width(declared)
    mean_z = declared["parameters"][:width].copy()
    if declared["model"].startswith("arma_"):
        ma = int(declared["intercept"]) + declared["p_ar"]
        mean_z[ma:] = [0.13 / (i + 1) for i in range(declared["q_ma"])]
    power = power_chart_case(declared["model"].split("_", 1)[1], declared["density"],
                             declared["p_vol"], declared["q_vol"])
    z = np.r_[mean_z, power["original_z"]]
    declared.update(parameters=z.copy(), original_z=z.copy(), transformed=True,
                    chart="joint_power_shared_unshifted_chart_v1", case_id=declared["case_id"] + "-original-z")
    return declared


def original_z_cases():
    return tuple(chart_case(layout_case(layout, volatility, density)) for layout in CHART_LAYOUTS
                 for volatility in ("egarch", "aparch") for density in DENSITIES)


def scored_suffix(template, first, stop=None):
    declared = deepcopy(template)
    stop = declared["stop"] if stop is None else stop
    if not declared["hold_back"] + declared["recursion_begin"] <= first < stop <= len(declared["y"]):
        raise ValueError("Invalid explicit score suffix")
    declared.update(first=first, stop=stop, first_scored_original_position=first,
                    scored_positions=tuple(range(first, stop)), scored_rows=tuple(declared["rows"][first:stop]),
                    case_id=declared["case_id"] + f"-rows{first}:{stop}")
    return declared


def expected_dispatch(declared):
    if (declared["p_vol"], declared["q_vol"]) != (1, 1):
        return 0
    if declared["model"].startswith("arma_"):
        return int(declared["intercept"] and (declared["p_ar"], declared["q_ma"]) == (1, 1))
    return 2 if mean_width(declared) == 4 else 0


def fixed_zero_row_case(model, density, *, empty_mean=False):
    if not model.startswith("linear_"):
        raise ValueError("Structural design-zero fixtures require a linear model")
    declared = case(model, density, columns=0 if empty_mean else 5, intercept=False,
                    p_vol=3, q_vol=2, y=[0.0, -0.37, 0.81, 0.0, -0.19, 0.62])
    declared["design"][[0, 3]] = 0.0
    if density == "variance_gamma":
        declared["parameters"][mean_width(declared) + volatility_width(declared)] = 0.0
    if model.endswith("aparch"):
        declared["parameters"][mean_width(declared) + volatility_width(declared) - 1] = 0.37
    declared["case_id"] += "-structurally-fixed-zeros"
    return declared


def long_horizon_case(model, density="normal"):
    y = np.tile([0.61, -0.83, 0.17, -0.29, 1.07, -0.46, 0.31, -0.72], 64)
    declared = case(model, density, p_ar=0, q_ma=1 if model.startswith("arma_") else 0,
                    columns=1, intercept=False, y=y, first=17, h0=0.72)
    width = mean_width(declared)
    declared["parameters"][:width] = [-0.998] if model.startswith("arma_") else [0.01]
    vol = ([-2e-4, 2e-4, -1e-4, 0.9995] if model.endswith("egarch")
           else [2e-5, 1e-4, -0.12, 0.9995, 1.7])
    declared["parameters"][width:width + volatility_width(declared)] = vol
    declared["case_id"] += "-long-near-boundary"
    return declared


def historical_aparch_small_unit_case(density, *, scaled=False):
    shapes = {"normal": (), "gh": (.1, 1.3, .4), "variance_gamma": (.1, 3.)}
    if density not in shapes:
        raise ValueError("Unknown historical conditioned APARCH density")
    y = np.asarray([.2, -.4, .5, -1., .7, .3])
    theta = np.asarray([.03, .12, -.08, .08, .07, .15, .82, 1.7, *shapes[density]])
    if scaled:
        theta[0] *= 1e-5
        theta[3] *= 1e-5 ** theta[7]
        y *= 1e-5
    return case("arma_aparch", density, 1, 1, 1, 1, y=y, parameters=theta,
                h0=float(np.mean(y * y)), first=1, stop=6, hold_back=0,
                case_id=f"historical-arma11-aparch11-{density}-{'1e-5' if scaled else 'unit'}")
