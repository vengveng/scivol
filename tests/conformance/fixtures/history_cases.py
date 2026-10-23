"""History fixtures declared independently of native sample/layout metadata."""

from copy import deepcopy
from functools import lru_cache
import math

import numpy as np

from . import garch_cases, joint_cases, joint_power_cases, mean_cases, power_cases
from .profile_arx_cases import causal_design

DENSITIES = ("skewt", "ged", "gh", "nig", "hyperbolic", "variance_gamma")
MODELS = ("garch", "gjr", "egarch", "aparch", "arma", "linear",
          "arma_garch", "arma_gjr", "arma_egarch", "arma_aparch",
          "linear_garch", "linear_gjr", "linear_egarch", "linear_aparch")
CAUSAL_MODELS = tuple(f"{mean}_{vol}" for mean in ("arx", "harx")
                      for vol in ("constant", "garch", "gjr", "egarch", "aparch"))
Y = np.asarray([2.75, -1.125, .75, -.375, 1.625, -.875,
                .5, 1.25, -1.5, .625, 1.875, -.25])
OBSERVATIONS = (1.125, -.75, .5)
INNOVATIONS = (-.375, .625, -.875)
ORDERS = {"wide": (2, 3, 3, 2), "common": (1, 1, 1, 1),
          "ma_long": (2, 3, 1, 2), "vol_long": (2, 1, 3, 2)}


def required_histories(model, layout="wide"):
    ar, ma, p, _ = ORDERS[layout]
    if model.startswith(("arx_", "harx_")):
        mean, vol = model.split("_", 1)
        return (2 if mean == "arx" else 3), (0 if vol == "constant" else p)
    if model == "arma":
        return ar, ma
    if model == "linear":
        return 0, 0
    if model.startswith("arma_"):
        return ar, max(ma, p)
    return 0, p


def modes(model):
    observations, innovations = required_histories(model)
    return (("empty",)
            + (("observations",) if observations else ())
            + (("innovations",) if innovations else ())
            + (("both",) if observations and innovations else ())
            + (("zero",) if observations or innovations else ()))


def case(model, density, mode="both", layout="wide", *, causal_intercept=True, causal_first_offset=2):
    if density not in DENSITIES or mode not in modes(model):
        raise ValueError("Unknown history fixture declaration")
    ar, ma, p, q = ORDERS[layout]
    observation_count, innovation_count = required_histories(model, layout)
    observations = OBSERVATIONS[:observation_count] if mode in ("observations", "both") else ()
    innovations = INNOVATIONS[:innovation_count] if mode in ("innovations", "both") else ()
    if mode == "zero":
        observations, innovations = (0.0,) * observation_count, (0.0,) * innovation_count
    causal = model.startswith(("arx_", "harx_"))
    begin = max(ar, ma, p, q) if model.startswith("arma_") else max(ar, ma) if model == "arma" else 0
    hold = 4 if causal else 1
    first = hold + begin + (causal_first_offset if causal else 1)
    options = dict(y=Y, first=first, stop=10, hold_back=hold)
    native_model = model
    if causal:
        mean, vol = model.split("_", 1)
        native_model = "linear" if vol == "constant" else "linear_" + vol
        external = np.asarray([[.25 + (row % 3) / 8] for row in range(len(Y))])
        design = causal_design(Y, external, 2, hold, observation_history=observations,
                               horizons=(1, 3) if mean == "harx" else None)
        options.update(design=design, columns=3, intercept=causal_intercept)
    if native_model in ("garch", "gjr"):
        declared = garch_cases.case(native_model, density, p, q, h0=1.25, **options)
    elif native_model in ("egarch", "aparch"):
        declared = power_cases.case(native_model, density, p, q, h0=1.25, **options)
    elif native_model == "arma":
        declared = mean_cases.arma_case(density, ar, ma, **options)
    elif native_model == "linear":
        declared = mean_cases.linear_case(density, **options)
    elif native_model.endswith(("_garch", "_gjr")):
        declared = joint_cases.case(native_model, density, ar, ma, p, q, h0=1.25, **options)
    else:
        declared = joint_power_cases.case(native_model, density,
                                          ar if native_model.startswith("arma_") else 0,
                                          ma if native_model.startswith("arma_") else 0,
                                          p, q, h0=1.25, **options)
    declared.update(observation_history=observations, innovation_history=innovations,
                    recursion_begin=begin, history_model=model,
                    case_id=f"history-{model}-{density}-{mode}-{layout}")
    if causal:
        declared.update(causal_kind=mean, causal_order=2, causal_horizons=(1, 3) if mean == "harx" else (),
                        input_exogenous=external)
    return declared


def as_chart(declared):
    model = declared["model"]
    if model in ("garch", "gjr", "egarch", "aparch"):
        factory = garch_cases if model in ("garch", "gjr") else power_cases
        mapped = factory.chart_case(model, declared["density"], declared["p"], declared["q"])
        result = deepcopy(declared)
        for key in ("parameters", "original_z", "transformed", "chart"):
            result[key] = deepcopy(mapped[key])
        result["case_id"] += "-original-z"
        return result
    factory = (mean_cases if model in ("arma", "linear") else joint_cases
               if model.endswith(("_garch", "_gjr")) else joint_power_cases)
    result = factory.chart_case(declared)
    if declared.get("causal_kind") == "arx":
        start, order = int(declared["intercept"]), declared["causal_order"]
        coordinates = np.asarray([1.2 * (-1) ** lag / (lag + 1) for lag in range(order)])
        result["parameters"][start:start + order] = coordinates
        result["original_z"][start:start + order] = coordinates
    return result


def declarations(causal=False):
    return tuple((model, density, mode) for model in (CAUSAL_MODELS if causal else MODELS)
                 for density in DENSITIES for mode in modes(model))


HISTORICAL_MEANX_CASES = (
    ("arx_egarch11", "skewt"), ("arx_egarch11", "ged"),
    ("harx_egarch21", "studentt"), ("harx_egarch21", "skewt"), ("harx_egarch21", "ged"),
    ("harx_gjr21", "studentt"), ("harx_gjr21", "skewt"), ("harx_gjr21", "ged"),
)


# These three routines preserve historical fixture-generation rounding only.
# Likelihood and derivative expectations use the independent shared oracles.
def _historical_har_history_means(y: np.ndarray, horizon: int) -> np.ndarray:
    """Causal window means, without subtracting an expired global prefix."""
    if y.size < 2:
        return np.zeros(y.size, dtype=np.float64)
    width = min(horizon, y.size)
    ordinary = _historical_well_conditioned_prefix_means(y, width)
    if ordinary is not None:
        return ordinary
    return _historical_block_history_means(y, width)


def _historical_well_conditioned_prefix_means(
    y: np.ndarray, width: int,
) -> np.ndarray | None:
    # Retain ordinary design rounding. Rebase when the global prefix has lost
    # at least half its significant bits, or its accumulation is out of range.
    times = np.arange(1, y.size)
    starts = np.maximum(times - width, 0)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        prefix = np.r_[0., np.cumsum(y)]
        absolute_prefix = np.r_[0., np.cumsum(np.abs(y))]
        sums = prefix[times] - prefix[starts]
        stable = np.isfinite(sums) & (
            np.abs(sums) > np.sqrt(np.finfo(float).eps) * absolute_prefix[times]
        )
        if not np.all(stable):
            return None
        result = np.zeros(y.size, dtype=np.float64)
        result[1:] = sums / np.minimum(times, width)
        return result


def _historical_block_history_means(y: np.ndarray, width: int) -> np.ndarray:
    n = y.size
    result = np.zeros(n, dtype=np.float64)
    magnitude = float(np.max(np.abs(y)))
    scale = float(width) if magnitude > np.finfo(float).max / width else 1.0
    forward = np.zeros((n + 1, 2), dtype=np.float64)
    backward = np.zeros((n + 1, 2), dtype=np.float64)

    def accumulate(indices: range, output: np.ndarray, offset: int) -> None:
        total = correction = 0.0
        for index in indices:
            value = float(y[index]) / scale
            updated = total + value
            correction += ((total - updated) + value if abs(total) >= abs(value)
                           else (value - updated) + total)
            total = updated
            output[index + offset] = total, correction

    for start in range(0, n, width):
        stop = min(start + width, n)
        accumulate(range(start, stop), forward, 1)
        accumulate(range(stop - 1, start - 1, -1), backward, 0)
    for time in range(1, n):
        count = min(time, width)
        if time <= width:
            terms = forward[time]
        elif time % width == 0:
            terms = backward[time - width]
        else:
            terms = (*backward[time - width], *forward[time])
        result[time] = math.fsum(terms) * (scale / count)
    return result


def _historical_mean_features(kind, y, x):
    features = np.zeros((len(y), 3 if kind == "arx" else 4), dtype=np.float64)
    features[:, 0] = 1.0
    if kind == "arx":
        features[1:, 1] = y[:-1]
    else:
        for column, horizon in enumerate((1, 5), start=1):
            features[:, column] = _historical_har_history_means(y, horizon)
    features[:, -1:] = x
    return np.ascontiguousarray(features)


@lru_cache(maxsize=3)
def _historical_meanx_series(recipe):
    """Frozen seeded data recipes; not likelihood simulators or AD oracles."""
    har = recipe.startswith("harx")
    egarch = "egarch" in recipe
    kind = "harx" if har else "arx"
    n = 300 if har else 280
    seed = {"arx_egarch11": 20260512, "harx_egarch21": 20260513,
            "harx_gjr21": 20260603}[recipe]
    rng = np.random.default_rng(seed)
    burn, total = 300, n + 300
    x = rng.standard_normal((total, 1)).astype(np.float64)
    y, residuals, variance = (np.zeros(total, dtype=np.float64) for _ in range(3))
    draws = rng.standard_normal(total).astype(np.float64)
    if egarch:
        mean = np.asarray([.01, .30, .09, .12] if har else [.02, .22, .15])
        omega, alpha, gamma, beta = ((-.20, [.06, .03], [.02, -.01], [.89]) if har
                                     else (-.15, [.10], [-.04], [.91]))
        logh = np.zeros(total, dtype=np.float64)
        variance[0] = 1.0
        residuals[0] = np.sqrt(variance[0]) * draws[0]
        features = _historical_mean_features(kind, y[:1], x[:1])
        y[0] = float(features[0] @ mean) + residuals[0]
        for t in range(1, total):
            value = omega
            for lag, (a, g) in enumerate(zip(alpha, gamma), start=1):
                if t >= lag:
                    z = residuals[t - lag] / np.sqrt(variance[t - lag])
                    value += a * (abs(z) - np.sqrt(2.0 / np.pi)) + g * z
            for lag, b in enumerate(beta, start=1):
                if t >= lag:
                    value += b * logh[t - lag]
            logh[t] = value
            variance[t] = np.exp(value)
            residuals[t] = np.sqrt(variance[t]) * draws[t]
            features = _historical_mean_features(kind, y[:t + 1], x[:t + 1])
            y[t] = float(features[t] @ mean) + residuals[t]
    else:
        omega, alpha, gamma, beta = 3e-6, [.03, .015], [.02, .01], [.88]
        persistence = float(sum(alpha) + .5 * sum(gamma) + sum(beta))
        # Original data-generation floors are retained, never used in an objective.
        variance[0] = omega / max(1.0 - persistence, 1e-3)
        for t in range(total):
            if t:
                value = omega
                for lag in range(len(alpha)):
                    if t > lag:
                        e = residuals[t - 1 - lag]
                        value += alpha[lag] * e**2 + gamma[lag] * (1.0 if e < 0.0 else 0.0) * e**2
                for lag in range(len(beta)):
                    if t > lag:
                        value += beta[lag] * variance[t - 1 - lag]
                variance[t] = max(value, 1e-12)
            mean = .01 + float(np.dot([.12], x[t]))
            for coefficient, horizon in zip([.35, .10], (1, 5)):
                width = min(horizon, t)
                if width:
                    mean += coefficient * float(np.mean(y[t - width:t]))
            residuals[t] = np.sqrt(variance[t]) * draws[t]
            y[t] = mean + residuals[t]
    return np.ascontiguousarray(y[burn:]), np.ascontiguousarray(x[burn:])


def historical_meanx_case(recipe, density, *, causal):
    """Retain all old rows; causal transport adds only a discarded prefix."""
    if (recipe, density) not in HISTORICAL_MEANX_CASES:
        raise ValueError("Unknown historical meanx declaration")
    y, external = _historical_meanx_series(recipe)
    har, egarch = recipe.startswith("harx"), "egarch" in recipe
    kind = "harx" if har else "arx"
    features = _historical_mean_features(kind, y, external)
    if recipe == "arx_egarch11":
        point = [.02, .22, .15, -.15, .10, -.04, .91]
    elif egarch:
        point = [.01, .30, .09, .12, -.20, .06, .03, .02, -.01, .89]
    else:
        point = [.01, .35, .10, .12, 3e-6, .03, .015, .02, .01, .88]
    shape = ([8.0] if density == "studentt" else
             [8.0, -.05 if egarch else -.10] if density == "skewt" else
             [1.6 if egarch else 1.7])
    hold = (5 if har else 1) if causal else 0
    observations = np.r_[np.zeros(hold), y]
    inputs = np.vstack([np.zeros((hold, 1)), external])
    design = np.vstack([np.zeros((hold, features.shape[1] - 1)), features[:, 1:]])
    factory = joint_power_cases if egarch else joint_cases
    declared = factory.case("linear_egarch" if egarch else "linear_gjr", density,
                            p_vol=2 if har else 1, q_vol=1, y=observations, design=design,
                            parameters=point + shape, h0=float(np.mean(y * y)),
                            hold_back=hold, first=hold, stop=len(observations),
                            case_id=f"historical-{recipe}-{density}-{'causal' if causal else 'linear'}")
    declared.update(historical_recipe=recipe, historical_y=y.copy(), historical_exogenous=external.copy(),
                    historical_features=features, historical_row_offset=hold,
                    observation_history=(), innovation_history=())
    if causal:
        declared.update(causal_kind=kind, causal_order=1 if not har else 0,
                        causal_horizons=(1, 5) if har else (), input_exogenous=inputs)
    if egarch:
        declared["historical_tolerances"] = (
            (3e-5, 6e-6), (2e-2 if density == "skewt" else 2e-3, 3e-4 if har else 2e-4),
            (.12 if density == "skewt" else .08 if har else .07,
             8.0 if density == "skewt" else .08 if har else .07))
    else:
        declared["historical_tolerances"] = (((2e-6, 3e-6), (4e-5, 6e-6), (1e-3, 1.5e-2))
                                            if density == "ged" else
                                            ((3e-6, 4e-6), (5e-5, 8e-6), (2e-3, 2.5e-2)))
    return declared


def causal_ar_plateau_case(volatility, density, sign, coordinate):
    if volatility not in ("constant", "gjr", "egarch", "aparch") or density not in DENSITIES:
        raise ValueError("Unknown causal AR plateau declaration")
    if sign not in (-1, 1):
        raise ValueError("Plateau sign must be -1 or 1")
    hold = 3
    external = np.asarray([[.25 + (row % 3) / 8] for row in range(len(Y))])
    observations = (.625,)
    innovations = () if volatility == "constant" else INNOVATIONS
    design = causal_design(Y, external, 1, hold, observation_history=observations)
    options = dict(y=Y, design=design, columns=2, intercept=True, hold_back=hold, first=hold, stop=10)
    if volatility == "constant":
        declared = mean_cases.linear_case(density, **options)
    else:
        factory = joint_cases if volatility == "gjr" else joint_power_cases
        declared = factory.case("linear_" + volatility, density, p_vol=3, q_vol=2, h0=1.25, **options)
    declared.update(causal_kind="arx", causal_order=1, causal_horizons=(),
                    input_exogenous=external, observation_history=observations,
                    innovation_history=innovations)
    declared = as_chart(declared)
    declared["parameters"][1] = sign * coordinate
    declared["original_z"] = declared["parameters"].copy()
    declared["case_id"] = f"arx1-{volatility}-{density}-plateau-{sign:+d}-{coordinate}"
    return declared

