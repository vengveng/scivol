"""Public exact Normal profiles in fitting units, using existing independent oracles."""

from copy import deepcopy
import ctypes
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pytest

from fixtures.history_cases import OBSERVATIONS
from fixtures.mean_cases import arma_case, external_design, linear_case, scored_suffix
from fixtures.profile_arx_cases import causal_design
from fixtures.profile_cases import ordinary_cases, profile_case, tail_case
from oracles.garch import jax_modules
from oracles.profile import coordinate_map, full_normal_jet, jet, nuisance, pullback


SENTINEL = -987654321.125
DOUBLE = ctypes.POINTER(ctypes.c_double)
U32 = ctypes.POINTER(ctypes.c_uint32)
U64 = ctypes.POINTER(ctypes.c_uint64)
SIZE = ctypes.c_size_t


@lru_cache(maxsize=1)
def bridge():
    path = os.environ.get("SCIVOLAPI_PROFILE_UNITS_TEST_BRIDGE")
    if not path:
        raise RuntimeError("Set SCIVOLAPI_PROFILE_UNITS_TEST_BRIDGE; public profile/units tests cannot skip it.")
    library = ctypes.CDLL(str(Path(path).resolve(strict=True)))
    library.scivolapi_profile_units_evaluate.argtypes = [
        ctypes.c_uint, SIZE, SIZE, ctypes.c_uint, U64, SIZE,
        DOUBLE, SIZE, DOUBLE, SIZE, SIZE, SIZE, SIZE,
        ctypes.c_double, ctypes.c_double, DOUBLE, DOUBLE, SIZE,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, U32, U64,
    ]
    library.scivolapi_profile_units_evaluate.restype = ctypes.c_int
    library.scivolapi_profile_units_evaluate_conditioned.argtypes = (
        library.scivolapi_profile_units_evaluate.argtypes + [DOUBLE, SIZE])
    library.scivolapi_profile_units_evaluate_conditioned.restype = ctypes.c_int
    return library


def native(case, mask, *, chart, response_divisor, caller):
    y = np.ascontiguousarray(case["y"], dtype=np.float64)
    x = np.ascontiguousarray(case.get("external_design", case["design"]), dtype=np.float64)
    source = np.ascontiguousarray(case["parameters"], dtype=np.float64)
    horizons = np.ascontiguousarray(case.get("horizons", ()), dtype=np.uint64)
    divisors = np.array([4.0 if i % 2 == 0 else 0.5 for i in range(x.shape[1])])
    k = len(source)
    f, evidence = np.full(1, SENTINEL), np.full(1, SENTINEL)
    g, h, theta = np.full(k, SENTINEL), np.full((k, k), SENTINEL), np.full(k + 1, SENTINEL)
    validity, metadata = np.zeros(4, dtype=np.uint32), np.zeros(6, dtype=np.uint64)
    family = case.get("family", 0 if case["model"] == "arma" else 1)
    arguments = (
        family, case.get("ar_lags", case["p"]), case["q"], case["intercept"],
        horizons.ctypes.data_as(U64), len(horizons), y.ctypes.data_as(DOUBLE), len(y),
        x.ctypes.data_as(DOUBLE), x.shape[1], case["hold_back"], case["scored_rows"][0], case["stop"],
        2.0, response_divisor, divisors.ctypes.data_as(DOUBLE), source.ctypes.data_as(DOUBLE), k,
        chart, mask, caller, f.ctypes.data_as(DOUBLE), g.ctypes.data_as(DOUBLE), h.ctypes.data_as(DOUBLE),
        theta.ctypes.data_as(DOUBLE), evidence.ctypes.data_as(DOUBLE), validity.ctypes.data_as(U32),
        metadata.ctypes.data_as(U64),
    )
    if "observation_history" in case:
        history = np.ascontiguousarray(case["observation_history"], dtype=np.float64)
        status = bridge().scivolapi_profile_units_evaluate_conditioned(
            *arguments, history.ctypes.data_as(DOUBLE), len(history))
    else:
        status = bridge().scivolapi_profile_units_evaluate(*arguments)
    return dict(status=status, valid_mask=int(validity[0]), reasons=validity[1:],
                f=float(f[0]), g=g, h=h, theta=theta, evidence=float(evidence[0]), metadata=metadata)


def causal_case(*, har=False, intercept=True, order=2, columns=2,
                horizons=(1, 4), observation_history=()):
    y = np.array([0.37, -0.81, 1.23, -0.19, 0.62, -1.41, 0.08, -0.54,
                  0.44, 1.02, -0.32, 0.71, -0.93, 0.26, 0.51, -0.12])
    horizons = tuple(horizons) if har else ()
    generated = len(horizons) if har else order
    hold_back = max(2, max(horizons) if har else order)
    external = external_design(len(y), columns)
    design = causal_design(2 * y, external, order, hold_back,
                           observation_history=2 * np.asarray(observation_history),
                           horizons=horizons if har else None)
    parameters = ([0.125] if intercept else []) + [0.1] * (generated + columns) + [1.3]
    case = profile_case(linear_case(columns=generated+columns, intercept=intercept, y=y,
                                   design=design, parameters=parameters, first=8, stop=14,
                                   hold_back=hold_back))
    case.update(family=3 if har else 2, ar_lags=0 if har else order,
                horizons=horizons, external_design=external,
                linear_ar_order=0 if har else order,
                case_id=f"profile-units-{'harx' if har else 'arx'}-{intercept}")
    if len(observation_history):
        case["observation_history"] = tuple(observation_history)
    return case


def causal_history_case(*, har=False, intercept=True, first_offset=0):
    history = OBSERVATIONS[:3 if har else 2]
    case = causal_case(har=har, intercept=intercept, horizons=(1, 3), observation_history=history)
    case = scored_suffix(case, case["hold_back"] + first_offset)
    if not har:
        start = int(intercept)
        case["parameters"][start:start + 2] = (0.7, -0.4)
    case["case_id"] = (f"profile-units-history-{'har13' if har else 'arx2'}"
                       f"-i{int(intercept)}-first-plus{first_offset}")
    return case


def history_cases():
    return tuple(causal_history_case(har=har, intercept=intercept, first_offset=offset)
                 for har in (False, True) for intercept in (False, True) for offset in (0, 1))


def cases():
    original = ordinary_cases()
    result = [scored_suffix(original[i], 7, 11, hold_back=2) for i in (0, 1, 3, 6, 8)]
    result += [causal_case(har=har, intercept=intercept)
               for har in (False, True) for intercept in (False, True)]
    return result


def oracle_case(case):
    expected = deepcopy(case)
    expected["y"] *= 2.0
    if "observation_history" in expected:
        expected["observation_history"] = 2 * np.asarray(expected["observation_history"])
    expected["transformed"] = False
    return expected


def public_profile_pullback(case, theta_jet, chart):
    if not chart:
        return theta_jet
    oracle = dict(case, transformed=True, original_z=case["parameters"].copy())
    return pullback(oracle, theta_jet)


def expected_jet(case, output, chart):
    theta_jet = jet(oracle_case(case), theta=output["theta"][:-1])
    return public_profile_pullback(case, theta_jet, chart)


def exercise_profile_units(case, chart, response_divisor):
    full = native(case, 7, chart=chart, response_divisor=response_divisor, caller=False)
    assert full["status"] == 0 and full["valid_mask"] == 7
    if not chart:
        np.testing.assert_array_equal(full["theta"][:-1], case["parameters"])
    expected = expected_jet(case, full, chart)
    exact = nuisance(oracle_case(case), full["theta"][:-1])
    np.testing.assert_allclose(full["theta"][-1], float(exact["variance"]), rtol=4e-13, atol=0)
    np.testing.assert_allclose(full["evidence"], float(exact["f"]), rtol=5e-14, atol=3e-12)
    for mask in range(1, 8):
        for caller in (False, True):
            output = native(case, mask, chart=chart, response_divisor=response_divisor, caller=caller)
            assert output["status"] == 0 and output["valid_mask"] == mask
            assert tuple(output["reasons"]) == tuple(0 if mask & bit else 3 for bit in (1, 2, 4))
            np.testing.assert_array_equal(output["metadata"],
                [len(case["parameters"]), len(case["parameters"])+1, case["hold_back"],
                 case["scored_rows"][0], len(case["scored_rows"]), 2 if chart else 0])
            np.testing.assert_array_equal(output["theta"], full["theta"])
            np.testing.assert_allclose(output["evidence"], float(exact["f"]), rtol=5e-14, atol=3e-12)
            for bit, name, target in zip((1, 2, 4), ("f", "g", "h"), expected):
                if mask & bit:
                    np.testing.assert_allclose(output[name], target, rtol=4e-10 if bit != 1 else 5e-14,
                                               atol=8e-11 if bit != 1 else 3e-12)
                    np.testing.assert_allclose(output[name], full[name], rtol=4e-10 if bit != 1 else 5e-14,
                                               atol=8e-11 if bit != 1 else 3e-12)
                else:
                    np.testing.assert_array_equal(output[name], np.full_like(output[name], SENTINEL))
    return full, expected


@pytest.mark.parametrize("case", cases(), ids=lambda case: case["case_id"])
@pytest.mark.parametrize("chart", (False, True))
@pytest.mark.parametrize("response_divisor", (3.0, 0.25))
def test_public_profile_units_all_masks(case, chart, response_divisor):
    exercise_profile_units(case, chart, response_divisor)


def assert_full_normal_schur(case, output, chart):
    oracle = oracle_case(case)
    value, gradient, hessian = full_normal_jet(oracle, output["theta"][:-1], output["theta"][-1])
    np.testing.assert_allclose(gradient[-1], 0.0, rtol=0, atol=5e-13)
    schur = hessian[:-1, :-1] - np.outer(hessian[:-1, -1], hessian[-1, :-1]) / hessian[-1, -1]
    target = public_profile_pullback(case, (value, gradient[:-1], schur), chart)
    for name, expected in zip(("f", "g", "h"), target):
        np.testing.assert_allclose(output[name], expected, rtol=4e-10 if name != "f" else 5e-14,
                                   atol=8e-11 if name != "f" else 3e-12)
    return hessian, target


@pytest.mark.parametrize("case", history_cases(), ids=lambda case: case["case_id"])
@pytest.mark.parametrize("chart", (False, True))
@pytest.mark.parametrize("response_divisor", (3.0, 0.25))
def test_public_profile_units_supplied_causal_history_all_masks_and_schur(case, chart, response_divisor):
    output, expected = exercise_profile_units(case, chart, response_divisor)
    full_hessian, schur_target = assert_full_normal_schur(case, output, chart)
    # The nuisance cross block must be active, so a conditional Hessian cannot pass.
    assert np.max(np.abs(full_hessian[:-1, -1])) > 1e-5
    theta_jet = jet(oracle_case(case), theta=output["theta"][:-1])
    assert not np.allclose(theta_jet[2], full_hessian[:-1, :-1], rtol=1e-12, atol=1e-12)
    for profile, schur in zip(expected, schur_target):
        np.testing.assert_allclose(profile, schur, rtol=4e-10, atol=8e-11)
    if chart:
        jax, jnp, _ = jax_modules()
        z = jnp.asarray(case["parameters"])
        mapping = coordinate_map(case)
        np.testing.assert_allclose(output["theta"][:-1], np.asarray(mapping(z)), rtol=4e-14, atol=4e-15)
        jacobian = np.asarray(jax.jacfwd(mapping)(z))
        curvature_term = expected[2] - jacobian.T @ theta_jet[2] @ jacobian
        if case["family"] == 2:
            assert np.max(np.abs(curvature_term)) > 1e-6
        else:
            # HAR coefficients retain their identity chart, with no AR restriction.
            np.testing.assert_array_equal(curvature_term, np.zeros_like(curvature_term))


@pytest.mark.parametrize("har", (False, True))
@pytest.mark.parametrize("chart", (False, True))
def test_public_profile_units_history_is_used_without_discarded_row_leakage(har, chart):
    case = causal_history_case(har=har)
    baseline = native(case, 7, chart=chart, response_divisor=3.0, caller=True)
    assert baseline["status"] == 0 and baseline["valid_mask"] == 7
    for history in ((), (0.0,) * len(case["observation_history"]), case["observation_history"][::-1]):
        variant = deepcopy(case)
        variant["observation_history"] = history
        variant["design"] = causal_design(2 * variant["y"], variant["external_design"], 2,
            variant["hold_back"], observation_history=2 * np.asarray(history),
            horizons=variant["horizons"] if har else None)
        actual = native(variant, 7, chart=chart, response_divisor=3.0, caller=True)
        assert actual["status"] == 0 and actual["valid_mask"] == 7
        for name, target in zip(("f", "g", "h"), expected_jet(variant, actual, chart)):
            np.testing.assert_allclose(actual[name], target, rtol=4e-10 if name != "f" else 5e-14,
                                       atol=8e-11 if name != "f" else 3e-12)
        exact = nuisance(oracle_case(variant), actual["theta"][:-1])
        np.testing.assert_allclose(actual["theta"][-1], float(exact["variance"]), rtol=4e-13, atol=0)
        assert abs(actual["f"] - baseline["f"]) > 1e-7
        assert abs(actual["theta"][-1] - baseline["theta"][-1]) > 1e-7
        if not len(history):
            legacy = deepcopy(variant)
            legacy.pop("observation_history")
            old_entry = native(legacy, 7, chart=chart, response_divisor=3.0, caller=True)
            assert old_entry["status"] == 0 and old_entry["valid_mask"] == 7
            for name in ("f", "g", "h", "theta", "evidence", "metadata", "reasons"):
                np.testing.assert_array_equal(actual[name], old_entry[name])
    excluded = deepcopy(case)
    excluded["y"][:case["hold_back"]] = 1e100
    excluded["y"][case["stop"]:] = -1e100
    excluded["external_design"][:case["hold_back"]] = -1e100
    excluded["external_design"][case["stop"]:] = 1e100
    actual = native(excluded, 7, chart=chart, response_divisor=3.0, caller=True)
    assert actual["status"] == 0 and actual["valid_mask"] == 7
    for name in ("f", "g", "h", "theta", "evidence", "metadata", "reasons"):
        np.testing.assert_array_equal(actual[name], baseline[name])


@pytest.mark.parametrize("family", ("arma", "arx"))
def test_public_profile_units_original_z_tail(family):
    if family == "arma":
        case = tail_case(18.0)
    else:
        case = causal_case(order=1, columns=1)
        case["parameters"][1] = 18.0
    output = native(case, 7, chart=True, response_divisor=3.0, caller=True)
    assert output["status"] == 0 and output["valid_mask"] == 7
    expected = expected_jet(case, output, True)
    assert output["g"][1] != 0.0 and output["h"][1, 1] != 0.0
    np.testing.assert_allclose(output["g"][1], expected[1][1], rtol=4e-9, atol=1e-29)
    np.testing.assert_allclose(output["h"][1, 1], expected[2][1, 1], rtol=4e-9, atol=1e-29)


def test_public_profile_units_independent_full_normal_schur():
    case = profile_case(arma_case())
    output = native(case, 7, chart=False, response_divisor=3.0, caller=True)
    assert output["status"] == 0 and output["valid_mask"] == 7
    assert_full_normal_schur(case, output, False)


@pytest.mark.parametrize("response_divisor", (0.5, 2.0))
def test_public_profile_units_nuisance_rounds_once(response_divisor):
    # native() applies public_scale=2. The declared public residuals are
    # 2^-537 and 2^-538, giving exact variance (5/8)*denorm_min.
    case = profile_case(linear_case(columns=0, intercept=False,
                                   y=[2.0 ** -538, 2.0 ** -539], parameters=[1.3]))
    exact = nuisance(oracle_case(case))
    assert float(exact["variance"]) == np.nextafter(0.0, 1.0)
    for chart in (False, True):
        for mask in range(1, 8):
            output = native(case, mask, chart=chart, response_divisor=response_divisor, caller=True)
            assert output["status"] == 0 and output["valid_mask"] == mask
            assert output["theta"][-1] == np.nextafter(0.0, 1.0)
            if mask & 1:
                np.testing.assert_allclose(output["f"], float(exact["f"]), rtol=5e-14, atol=3e-12)
