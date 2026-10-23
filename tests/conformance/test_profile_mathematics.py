"""Private exact-profile transport and tests, never public causal sample binding."""

from __future__ import annotations

from copy import deepcopy
import ctypes
from functools import lru_cache
import math
import os
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from fixtures.mean_cases import arma_case, linear_case, scored_suffix
from fixtures.profile_cases import (REQUESTS, exact_binary_range_cases, expected_dispatch, long_horizon_case,
                                     ordinary_cases, original_z_cases, profile_case,
                                     range_case, suffix_cases, tail_case, zero_sse_case)
from oracles.mean import numpy_residuals
from oracles.profile import (coordinate_map, full_normal_jet, jet, linear_closed_jet, linear_exact_jet,
                              mp_jet, nuisance, pullback, rounded_variance_backward_error)


SENTINEL = -987654321.125
DOUBLE = ctypes.POINTER(ctypes.c_double)
U32 = ctypes.POINTER(ctypes.c_uint32)
U64 = ctypes.POINTER(ctypes.c_uint64)
SIZE = ctypes.c_size_t


@lru_cache(maxsize=1)
def profile_bridge():
    path = os.environ.get("SCIVOLAPI_PROFILE_TEST_BRIDGE")
    if not path:
        raise RuntimeError("Set SCIVOLAPI_PROFILE_TEST_BRIDGE; a required private profile bridge cannot be skipped.")
    library = ctypes.CDLL(str(Path(path).resolve(strict=True)))
    library.scivolapi_profile_probe_create.argtypes = [
        ctypes.c_uint, SIZE, SIZE, DOUBLE, SIZE, DOUBLE, SIZE, ctypes.c_uint,
        SIZE, SIZE, SIZE, SIZE, ctypes.c_uint, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, SIZE,
    ]
    library.scivolapi_profile_probe_create.restype = ctypes.c_int
    library.scivolapi_profile_probe_destroy.argtypes = [ctypes.c_void_p]
    library.scivolapi_profile_probe_destroy.restype = None
    library.scivolapi_profile_probe_metadata.argtypes = [ctypes.c_void_p, U64, U32]
    library.scivolapi_profile_probe_metadata.restype = ctypes.c_int
    library.scivolapi_profile_probe_map.argtypes = [ctypes.c_void_p, DOUBLE, SIZE, DOUBLE, DOUBLE]
    library.scivolapi_profile_probe_map.restype = ctypes.c_int
    library.scivolapi_profile_probe_evaluate.argtypes = [
        ctypes.c_void_p, DOUBLE, SIZE, ctypes.c_uint, DOUBLE, DOUBLE, DOUBLE,
        DOUBLE, U32, U64, DOUBLE,
    ]
    library.scivolapi_profile_probe_evaluate.restype = ctypes.c_int
    return library


class ProfileProblem:
    def __init__(self, case):
        if "h0" in case or case.get("density", "normal") != "normal":
            raise ValueError("This transport accepts only an unseeded Normal variance profile")
        self.library = profile_bridge()
        self.handle = ctypes.c_void_p()
        y = np.ascontiguousarray(case["y"], dtype=np.float64)
        design = np.ascontiguousarray(case["design"], dtype=np.float64)
        if y.ndim != 1 or design.ndim != 2 or design.shape[0] != len(y):
            raise ValueError("Profile design must be an observation-by-column matrix")
        absent = SIZE(-1).value
        error = ctypes.create_string_buffer(1024)
        family = {"arma": 0, "linear": 1}[case["model"]]
        if case["model"] == "linear" and case.get("linear_ar_order", 0):
            family = 2
        ar_argument = case["linear_ar_order"] if family == 2 else case["p"]
        status = self.library.scivolapi_profile_probe_create(
            family, ar_argument, case["q"],
            y.ctypes.data_as(DOUBLE), len(y), design.ctypes.data_as(DOUBLE), design.shape[1],
            case["intercept"], case.get("hold_back", 0), case.get("first", absent),
            case.get("stop", len(y)), case.get("recursion_begin_override", absent),
            case.get("transformed", False), case.get("fixed", True),
            ctypes.byref(self.handle), error, len(error),
        )
        if status:
            raise ValueError(f"Private profile preparation failed ({status}): {error.value.decode()}")
        metadata = np.zeros(6, dtype=np.uint64)
        dispatch = np.zeros(1, dtype=np.uint32)
        status = self.library.scivolapi_profile_probe_metadata(
            self.handle, metadata.ctypes.data_as(U64), dispatch.ctypes.data_as(U32))
        assert status == 0
        self.width, self.statistical_count, self.first, self.count, self.hold_back, self.begin = map(int, metadata)
        self.dispatch = int(dispatch[0])

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.library.scivolapi_profile_probe_destroy(self.handle)
        self.handle = ctypes.c_void_p()

    def map(self, z):
        point = np.ascontiguousarray(z, dtype=np.float64)
        theta = np.full(self.width, SENTINEL)
        jacobian = np.full((self.width, self.width), SENTINEL)
        status = self.library.scivolapi_profile_probe_map(
            self.handle, point.ctypes.data_as(DOUBLE), len(point),
            theta.ctypes.data_as(DOUBLE), jacobian.ctypes.data_as(DOUBLE))
        return {"status": status, "theta": theta, "jacobian": jacobian}

    def evaluate(self, parameters, request):
        point = np.ascontiguousarray(parameters, dtype=np.float64)
        value = np.full(1, SENTINEL)
        gradient = np.full(self.width, SENTINEL)
        hessian = np.full((self.width, self.width), SENTINEL)
        theta = np.full(self.width, SENTINEL)
        validity = np.array([0, 3, 3, 3, 1, 0, 0], dtype=np.uint32)
        count = np.zeros(1, dtype=np.uint64)
        reconstructed = np.full(2, SENTINEL)
        status = self.library.scivolapi_profile_probe_evaluate(
            self.handle, point.ctypes.data_as(DOUBLE), len(point), request,
            value.ctypes.data_as(DOUBLE), gradient.ctypes.data_as(DOUBLE),
            hessian.ctypes.data_as(DOUBLE), theta.ctypes.data_as(DOUBLE),
            validity.ctypes.data_as(U32), count.ctypes.data_as(U64), reconstructed.ctypes.data_as(DOUBLE))
        return {"f": float(value[0]), "g": gradient, "h": hessian, "theta": theta,
                "status": status, "valid_mask": int(validity[0]), "reasons": validity[1:4],
                "nuisance": {"status": int(validity[4]), "reason": int(validity[5]),
                             "n_eff": int(count[0]), "sse": float(reconstructed[0]) if validity[6] & 1 else None,
                             "variance": float(reconstructed[1]) if validity[6] & 2 else None},
                "dispatch": self.dispatch, "derivative_width": self.width,
                "statistical_parameter_count": self.statistical_count,
                "scored_rows": tuple(range(self.first, self.first + self.count)), "sentinel": SENTINEL}


@pytest.fixture
def profile_native():
    def evaluate(case, request):
        with ProfileProblem(case) as problem:
            return problem.evaluate(case["parameters"], request)
    return evaluate


def assert_unrequested(output, mask):
    for bit, name in ((1, "f"), (2, "g"), (4, "h")):
        if not mask & bit:
            np.testing.assert_array_equal(output[name], np.full_like(output[name], SENTINEL))


def assert_jet(output, expected, mask):
    assert output["status"] == 0 and output["valid_mask"] == mask
    assert_unrequested(output, mask)
    for bit, name, target in zip((1, 2, 4), ("f", "g", "h"), expected):
        if mask & bit:
            assert np.isfinite(output[name]).all()
            np.testing.assert_allclose(output[name], target, rtol=4e-10 if bit != 1 else 5e-14,
                                       atol=8e-11 if bit != 1 else 3e-12)


def assert_nuisance_same(left, right):
    for name in ("status", "reason", "n_eff"):
        assert left[name] == right[name]
    for name in ("sse", "variance"):
        if left[name] is None or right[name] is None:
            assert left[name] is right[name]
        else:
            np.testing.assert_allclose(left[name], right[name], rtol=4e-13, atol=0)


def exercise_requests(native, case, expected):
    full = native(case, 7)
    exact = nuisance(case, full["theta"])
    assert full["derivative_width"] == len(case["theta_names"])
    assert full["statistical_parameter_count"] == full["derivative_width"] + 1
    assert full["dispatch"] == expected_dispatch(case)
    assert full["scored_rows"] == case["scored_rows"]
    np.testing.assert_allclose(full["f"], float(exact["f"]), rtol=5e-14, atol=3e-12)
    np.testing.assert_allclose(full["nuisance"]["sse"], float(exact["sse"]), rtol=4e-14, atol=0)
    np.testing.assert_allclose(full["nuisance"]["variance"], float(exact["variance"]), rtol=4e-14, atol=0)
    assert full["nuisance"]["status"] == 0 and full["nuisance"]["reason"] == 1
    assert full["nuisance"]["n_eff"] == len(case["scored_rows"])
    for mask in REQUESTS:
        output = native(case, mask)
        assert_jet(output, expected, mask)
        assert_nuisance_same(output["nuisance"], full["nuisance"])
        for bit, name in ((1, "f"), (2, "g"), (4, "h")):
            if mask & bit:
                np.testing.assert_allclose(output[name], full[name], rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("case", ordinary_cases(), ids=lambda case: case["case_id"])
def test_profile_theta_all_masks(profile_native, case):
    exercise_requests(profile_native, case, jet(case))


@pytest.mark.parametrize("case", original_z_cases(), ids=lambda case: case["case_id"])
def test_profile_original_z_all_masks(profile_native, case):
    actual = profile_native(case, 7)
    exercise_requests(profile_native, case, pullback(case, jet(case, theta=actual["theta"])))


@pytest.mark.parametrize("case", ordinary_cases(), ids=lambda case: case["case_id"])
def test_profile_full_normal_schur_identity(profile_native, case):
    actual = profile_native(case, 7)
    f, g, h = full_normal_jet(case, actual["theta"], actual["nuisance"]["variance"])
    assert abs(g[-1]) < 2e-12
    schur = h[:-1, :-1] - np.outer(h[:-1, -1], h[-1, :-1]) / h[-1, -1]
    assert_jet(actual, (f, g[:-1], schur), 7)
    if len(g) > 1:
        assert np.max(abs(h[:-1, -1])) > 1e-5
        assert not np.allclose(schur, h[:-1, :-1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ordinary_cases() + original_z_cases(), ids=lambda case: case["case_id"])
def test_profile_fixed_runtime_parity(profile_native, case):
    runtime = deepcopy(case)
    runtime["fixed"] = False
    for mask in REQUESTS:
        left, right = profile_native(case, mask), profile_native(runtime, mask)
        assert left["dispatch"] == expected_dispatch(case) and right["dispatch"] == 0
        assert left["status"] == right["status"] == 0
        assert left["valid_mask"] == right["valid_mask"] == mask
        for bit, name in ((1, "f"), (2, "g"), (4, "h")):
            if mask & bit:
                np.testing.assert_allclose(left[name], right[name], rtol=4e-13, atol=4e-13)
        assert_nuisance_same(left["nuisance"], right["nuisance"])
        assert_unrequested(left, mask)
        assert_unrequested(right, mask)


@pytest.mark.parametrize("case", suffix_cases(), ids=lambda case: case["case_id"])
def test_profile_original_position_suffix_holdback(profile_native, case):
    exercise_requests(profile_native, case, jet(case))


@pytest.mark.parametrize("case", tuple(case for case in ordinary_cases() if case["model"] == "linear"),
                         ids=lambda case: case["case_id"])
def test_profile_linear_closed_form(profile_native, case):
    exercise_requests(profile_native, case, linear_closed_jet(case))


@pytest.mark.parametrize("case", (
    profile_case(arma_case()),
    profile_case(arma_case(p=0, q=0)),
    profile_case(linear_case(columns=2)),
), ids=lambda case: case["case_id"])
def test_profile_scalar_mp_derivatives(profile_native, case):
    expected = mp_jet(case)
    assert_jet(profile_native(case, 7), expected, 7)
    for actual, target in zip(jet(case), expected):
        np.testing.assert_allclose(actual, target, rtol=4e-11, atol=3e-11)


def test_profile_scored_suffix_does_not_reset_ma(profile_native):
    case = scored_suffix(profile_case(arma_case(parameters=[0.1, 0.2, 0.4, 1.3])), 6, 11)
    reset = deepcopy(case)
    reset.update(recursion_begin=6, recursion_begin_override=6)
    left, right = profile_native(case, 7), profile_native(reset, 7)
    assert_jet(left, jet(case), 7)
    assert_jet(right, jet(reset), 7)
    assert abs(left["f"] - right["f"]) > 1e-4
    assert left["nuisance"]["n_eff"] == right["nuisance"]["n_eff"] == 5


def test_profile_explicit_zero_order_recursion_begin(profile_native):
    case = profile_case(arma_case(p=0, q=0, first=0))
    case.update(recursion_begin=0, recursion_begin_override=0)
    exercise_requests(profile_native, case, jet(case))


def test_profile_constant_mean_reductions(profile_native):
    cases = (
        profile_case(arma_case(p=0, q=0, parameters=[0.03, 1.3], first=3, stop=11, hold_back=2)),
        profile_case(linear_case(columns=0, intercept=True, parameters=[0.03, 1.3],
                                 first=3, stop=11, hold_back=2)),
        profile_case(linear_case(columns=1, design=np.ones((12, 1)), parameters=[0.03, 1.3],
                                 first=3, stop=11, hold_back=2)),
    )
    for mask in REQUESTS:
        outputs = [profile_native(case, mask) for case in cases]
        for output in outputs:
            assert_jet(output, linear_closed_jet(cases[1]), mask)
            assert_nuisance_same(output["nuisance"], outputs[0]["nuisance"])


def test_profile_holdback_is_not_only_score_selection(profile_native):
    base = profile_case(arma_case(parameters=[0.1, 0.2, 0.4, 1.3]))
    full_history = scored_suffix(base, 7, 11)
    held_history = scored_suffix(base, 7, 11, hold_back=2)
    left, right = profile_native(full_history, 7), profile_native(held_history, 7)
    assert_jet(left, jet(full_history), 7)
    assert_jet(right, jet(held_history), 7)
    assert left["scored_rows"] == right["scored_rows"]
    assert abs(left["f"] - right["f"]) > 1e-5


def test_profile_conditioning_after_score_start_is_rejected():
    case = scored_suffix(profile_case(arma_case()), 3, 11)
    case.update(recursion_begin=6, recursion_begin_override=6)
    with pytest.raises(ValueError, match=r"failed \(1\)"):
        ProfileProblem(case)


def test_profile_empty_score_selection_is_rejected():
    case = scored_suffix(profile_case(arma_case()), 11, 11)
    with pytest.raises(ValueError, match=r"failed \(1\)"):
        ProfileProblem(case)


def test_profile_near_boundary_recursive_curvature(profile_native):
    case = long_horizon_case()
    expected = jet(case)
    assert np.max(abs(expected[2])) > 1000
    assert_jet(profile_native(case, 7), mp_jet(case), 7)
    exercise_requests(profile_native, case, expected)


@pytest.mark.parametrize("scale", (2.0 ** -400, 2.0 ** 400))
def test_profile_linear_scale_equivariance(profile_native, scale):
    case = profile_case(linear_case(columns=3, intercept=True))
    scaled = deepcopy(case)
    scaled["y"] *= scale
    scaled["parameters"] *= scale
    left, right = profile_native(case, 7), profile_native(scaled, 7)
    assert left["status"] == right["status"] == 0
    count = len(case["scored_rows"])
    np.testing.assert_allclose(right["f"], left["f"] + count * math.log(scale), rtol=5e-14, atol=3e-12)
    np.testing.assert_allclose(right["g"], left["g"] / scale, rtol=4e-11, atol=0)
    np.testing.assert_allclose(right["h"], left["h"] / scale ** 2, rtol=4e-11, atol=0)
    np.testing.assert_allclose(right["nuisance"]["variance"], left["nuisance"]["variance"] * scale ** 2,
                               rtol=4e-14, atol=0)
    exercise_requests(profile_native, scaled, jet(scaled))


@pytest.mark.parametrize("case", (zero_sse_case("arma"), zero_sse_case("linear"), zero_sse_case("linear", 4)),
                         ids=lambda case: case["case_id"])
def test_profile_zero_sse_is_unbounded_not_floored(profile_native, case):
    np.testing.assert_array_equal(numpy_residuals(case)[np.array(case["scored_rows"]) - case["hold_back"]], 0)
    for mask in REQUESTS:
        output = profile_native(case, mask)
        assert output["status"] == 4 and output["valid_mask"] == 0
        assert output["nuisance"] == {"status": 4, "reason": 2, "n_eff": len(case["scored_rows"]),
                                     "sse": 0.0, "variance": None}
        assert_unrequested(output, mask)


@pytest.mark.parametrize("scale", (2.0 ** -600, 2.0 ** 600))
def test_profile_raw_sse_range_failure_is_not_zero_sse(profile_native, scale):
    case = range_case(scale)
    assert mp.isfinite(nuisance(case)["f"])
    for mask in REQUESTS:
        output = profile_native(case, mask)
        assert output["status"] == 5 and output["valid_mask"] == 0
        assert output["nuisance"]["status"] == 5 and output["nuisance"]["reason"] == 3
        assert output["nuisance"]["variance"] is None
        assert output["nuisance"]["sse"] == (0.0 if scale < 1 else None)
        assert_unrequested(output, mask)


@pytest.mark.parametrize("quotient_rounding", (False, True))
def test_profile_subnormal_variance_has_explicit_backward_error(profile_native, quotient_rounding):
    case = range_case(2.0 ** -537, quotient_rounding=quotient_rounding)
    exact = nuisance(case)
    for mask in REQUESTS:
        output = profile_native(case, mask)
        assert_jet(output, (float(exact["f"]), np.empty(0), np.empty((0, 0))), mask)
        assert output["nuisance"]["variance"] == np.nextafter(0.0, 1.0)
        assert output["nuisance"]["reason"] == 1 and output["nuisance"]["status"] == 0
    error = rounded_variance_backward_error(case, output["nuisance"]["variance"])
    if quotient_rounding:
        assert float(error["relative_variance_error"]) == 0.5
        assert 0.1 < float(error["full_minus_profile_f"]) < 0.12
    else:
        assert error["relative_variance_error"] == 0 and error["full_minus_profile_f"] == 0


def test_profile_positive_sse_unrepresentable_quotient(profile_native):
    case = range_case(2.0 ** -537)
    case["y"] = np.array([2.0 ** -537, 0.0, 0.0])
    assert nuisance(case)["variance"] > 0
    for mask in REQUESTS:
        output = profile_native(case, mask)
        assert output["status"] == 5 and output["valid_mask"] == 0
        assert output["nuisance"]["sse"] == np.nextafter(0.0, 1.0)
        assert output["nuisance"]["status"] == 5 and output["nuisance"]["reason"] == 3
        assert output["nuisance"]["variance"] is None
        assert_unrequested(output, mask)


def test_profile_h_range_failure_preserves_f_g(profile_native):
    case = profile_case(linear_case(columns=1, design=[[1e200]], y=[1.0], parameters=[0.0, 1.3]))
    expected = {"f": (math.log(2 * math.pi) + 1) / 2, "g": np.array([-1e200])}
    for mask in REQUESTS:
        output = profile_native(case, mask)
        assert output["valid_mask"] == mask & 3
        assert output["status"] == (5 if mask & 4 else 0)
        assert output["nuisance"] == {"status": 0, "reason": 1, "n_eff": 1, "sse": 1.0, "variance": 1.0}
        for bit, name in ((1, "f"), (2, "g")):
            if mask & bit:
                np.testing.assert_allclose(output[name], expected[name], rtol=3e-14, atol=0)
        if mask & 4:
            assert output["reasons"][2] == 5
        assert_unrequested(output, mask)


def test_profile_original_z_tail_curvature(profile_native):
    case = tail_case(18.0)
    output = profile_native(case, 7)
    expected = pullback(case, jet(case, theta=output["theta"]))
    assert_jet(output, expected, 7)
    assert expected[1][1] != 0 and expected[2][1, 1] != 0
    np.testing.assert_allclose(output["g"][1], expected[1][1], rtol=4e-9, atol=1e-29)
    np.testing.assert_allclose(output["h"][1, 1], expected[2][1, 1], rtol=4e-9, atol=1e-29)
    with ProfileProblem(case) as problem:
        mapped = problem.map(case["original_z"])
    from oracles.garch import jax_modules
    jax, jnp, _ = jax_modules()
    expected_j = np.asarray(jax.jacfwd(coordinate_map(case))(jnp.asarray(case["original_z"])))
    assert mapped["status"] == 0
    np.testing.assert_allclose(mapped["jacobian"][1, 1], expected_j[1, 1], rtol=3e-12, atol=1e-29)


def test_profile_rounded_chart_boundary_is_not_clipped(profile_native):
    case = tail_case(20.0)
    for mask in REQUESTS:
        output = profile_native(case, mask)
        assert output["theta"][1] == 1.0
        assert output["status"] == 5 and output["valid_mask"] == 0
        assert output["nuisance"]["variance"] is None
        assert_unrequested(output, mask)


@pytest.mark.parametrize("axis,value", ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)))
def test_profile_theta_reflection_boundary_is_domain_failure(profile_native, axis, value):
    case = profile_case(arma_case(intercept=False, parameters=[0.2, -0.3, 1.3]))
    case["parameters"][axis] = value
    for mask in REQUESTS:
        output = profile_native(case, mask)
        assert output["status"] == 4 and output["valid_mask"] == 0
        assert output["nuisance"]["variance"] is None
        assert_unrequested(output, mask)


def test_profile_full_theta_axis_is_rejected_without_writes():
    case = ordinary_cases()[0]
    with ProfileProblem(case) as problem:
        output = problem.evaluate(np.r_[case["parameters"], 1.0], 7)
    assert output["status"] == 1 and output["valid_mask"] == 0
    for name in ("f", "g", "h"):
        np.testing.assert_array_equal(output[name], np.full_like(output[name], SENTINEL))


@pytest.mark.parametrize("case", exact_binary_range_cases(), ids=lambda case: case["case_id"])
def test_profile_exact_binary_range(profile_native, case):
    expected = linear_exact_jet(case)
    label = case["range_requirement"]
    try:
        diagnostic_sse = float(expected["sse"])
    except OverflowError:
        diagnostic_sse = None
    variance = float(expected["variance"])
    assert math.isfinite(variance) and variance > 0
    assert np.isfinite(expected["g"]).all() and np.isfinite(expected["h"]).all()
    if "sse-sum-overflow" in label or "square-overflow" in label:
        assert diagnostic_sse is None and variance == 2.0 ** 1022
    elif label == "recoverable-subnormal-double-round":
        assert diagnostic_sse == variance == np.nextafter(0.0, 1.0)
        assert diagnostic_sse / expected["n_eff"] == 0.0
        backward = rounded_variance_backward_error(case, variance)
        assert abs(float(backward["relative_variance_error"]) - 0.6) < 1e-15
        assert 0.095 < float(backward["full_minus_profile_f"]) < 0.096
    elif label == "subnormal-gradient-erased-by-division":
        assert expected["g"][0] == -2.0 * np.nextafter(0.0, 1.0)
    elif "exact-zero" in label:
        assert expected["exact_h"][0][0] == 0
    else:
        assert expected["exact_h"][0][0] != 0
        assert np.sign(expected["h"][0, 0]) == (1 if label.endswith("positive") else -1)
    for mask in REQUESTS:
        output = profile_native(case, mask)
        assert output["status"] == 0 and output["valid_mask"] == mask, (label, mask, output["status"], output["valid_mask"])
        assert output["nuisance"] == {"status": 0, "reason": 1, "n_eff": expected["n_eff"],
                                     "sse": diagnostic_sse, "variance": variance}
        assert output["derivative_width"] == len(case["parameters"])
        assert output["statistical_parameter_count"] == len(case["parameters"]) + 1
        assert_unrequested(output, mask)
        if mask & 1:
            np.testing.assert_allclose(output["f"], expected["f"], rtol=5e-14, atol=3e-12)
        if mask & 2:
            if label == "subnormal-gradient-erased-by-division":
                np.testing.assert_array_equal(output["g"], expected["g"])
            else:
                np.testing.assert_allclose(output["g"], expected["g"], rtol=4e-13, atol=0)
        if mask & 4:
            if "exact-zero" in label:
                np.testing.assert_array_equal(output["h"], np.zeros_like(expected["h"]))
            else:
                np.testing.assert_allclose(output["h"], expected["h"], rtol=4e-13, atol=0)
