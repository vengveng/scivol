"""Independent fixed-input DCC density-ratio checks; no public registration assumed.

The local ctypes records mirror only the test transport ABI. Collection loads
neither JAX nor a bridge. Chief executes AD/native cases serially.
"""

from __future__ import annotations

from copy import deepcopy
import ctypes
from fractions import Fraction
from functools import lru_cache
import os
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
import pytest

from fixtures.dependence_cases import (
    LAWS, LAYOUTS, REQUESTS, SHAPES, SIMPLEX_ANCHOR_POINTS, case, exact_sum_cases,
    expected_dispatch, extreme_student_hessian_case, history_case, identity_case,
    long_case, ordinary_cases, scored_suffix, shape_case, simplex_anchor_case,
)
from oracles.dependence import (
    affine_simplex_slack, gamma_germ, jax_modules, jet, mp_affine_simplex_slack,
    mp_gamma_constant, mp_jet, mp_normalize, mp_row_ratio, mp_state_path, mp_value,
    retained_residuals, scipy_value, simplex_anchor, state_jet,
)


_DOUBLE = ctypes.POINTER(ctypes.c_double)
_SIZE = ctypes.c_size_t
_U32 = ctypes.c_uint32


class _Config(ctypes.Structure):
    _fields_ = [("p", _SIZE), ("q", _SIZE), ("series", _SIZE), ("law", _U32), ("fixed", _U32)]


class _Input(ctypes.Structure):
    _fields_ = [("data", _DOUBLE), ("count", _SIZE)]


class _Output(ctypes.Structure):
    _fields_ = [("data", _DOUBLE), ("count", _SIZE)]


class _Sample(ctypes.Structure):
    _fields_ = [("residuals", _Input), ("target", _Input), ("seed", _Input),
                ("residual_history", _Input), ("correlation_history", _Input),
                ("recursion_begin", _SIZE), ("first_scored", _SIZE)]


class _Requirements(ctypes.Structure):
    _fields_ = [("bytes", _SIZE), ("alignment", _SIZE), ("parameters", _SIZE), ("dispatch", ctypes.c_int)]


class _Buffers(ctypes.Structure):
    _fields_ = [("value", _Output), ("gradient", _Output), ("hessian", _Output)]


class _Validity(ctypes.Structure):
    _fields_ = [(name, _U32) for name in ("status", "requested", "valid_mask", "f_reason", "g_reason", "h_reason")]


@lru_cache(maxsize=1)
def bridge():
    path = os.environ.get("SCIVOLAPI_DEPENDENCE_TEST_BRIDGE")
    if not path:
        raise RuntimeError("Set SCIVOLAPI_DEPENDENCE_TEST_BRIDGE to the new private dependence probe; no legacy fallback.")
    library = ctypes.CDLL(str(Path(path).resolve(strict=True)))
    library.scivolapi_dependence_probe_requirements.argtypes = [ctypes.POINTER(_Config), _U32, _U32, ctypes.POINTER(_Requirements)]
    library.scivolapi_dependence_probe_sample_status.argtypes = [ctypes.POINTER(_Config), ctypes.POINTER(_Sample)]
    library.scivolapi_dependence_probe_parameter_status.argtypes = [ctypes.POINTER(_Config), _Input]
    library.scivolapi_dependence_probe_evaluate.argtypes = [
        ctypes.POINTER(_Config), ctypes.POINTER(_Sample), _Input, _U32, _U32, _U32,
        ctypes.c_void_p, _SIZE, ctypes.POINTER(_Buffers), ctypes.POINTER(_Validity)]
    for name in ("requirements", "sample_status", "parameter_status", "evaluate"):
        getattr(library, "scivolapi_dependence_probe_" + name).restype = _U32
    return library


class NativeDependence:
    def __init__(self, declared):
        self.library = bridge()
        self.case = declared
        self.config = _Config(declared["p_corr"], declared["q_corr"], declared["series"],
                              {"normal": 0, "studentt": 1}[declared["law"]], declared["fixed"])
        arrays = [retained_residuals(declared), declared["target"], declared["seed"],
                  declared["residual_history"], declared["correlation_history"]]
        self.inputs = [np.ascontiguousarray(value, dtype=np.float64) for value in arrays]
        self.sample = _Sample(*[_Input(value.ctypes.data_as(_DOUBLE), value.size) for value in self.inputs],
                              declared["recursion_begin"], declared["first_scored"])

    def requirements(self, request, *, coordinates=0):
        needed = _Requirements()
        status = self.library.scivolapi_dependence_probe_requirements(
            ctypes.byref(self.config), coordinates, request, ctypes.byref(needed))
        return status, needed

    def sample_status(self):
        return self.library.scivolapi_dependence_probe_sample_status(ctypes.byref(self.config), ctypes.byref(self.sample))

    def parameter_status(self, point=None):
        point = np.ascontiguousarray(self.case["parameters"] if point is None else point, dtype=np.float64)
        return self.library.scivolapi_dependence_probe_parameter_status(
            ctypes.byref(self.config), _Input(point.ctypes.data_as(_DOUBLE), point.size))

    def evaluate(self, request=7, *, prevalidated=False, bytes_delta=0, misaligned=False,
                 output_counts=None, point=None):
        status, needed = self.requirements(request)
        if status:
            raise ValueError(f"Native dependence requirements failed ({status})")
        width = len(self.case["theta_names"])
        assert needed.parameters == width
        assert needed.alignment >= ctypes.alignment(ctypes.c_double)
        point = np.ascontiguousarray(self.case["parameters"] if point is None else point, dtype=np.float64)
        sentinel = -987654321.125
        f, g, h = np.full(1, sentinel), np.full(width, sentinel), np.full((width, width), sentinel)
        counts = (1, width, width * width) if output_counts is None else output_counts
        outputs = _Buffers(*[_Output(value.ctypes.data_as(_DOUBLE), count)
                             for value, count in zip((f, g, h), counts, strict=True)])
        capacity = needed.bytes + bytes_delta
        assert capacity >= 0
        arena = np.full(capacity + needed.alignment + 64, 165, dtype=np.uint8)
        start = 32 + (-(arena.ctypes.data + 32) % needed.alignment) + int(misaligned)
        validity = _Validity()
        status = self.library.scivolapi_dependence_probe_evaluate(
            ctypes.byref(self.config), ctypes.byref(self.sample), _Input(point.ctypes.data_as(_DOUBLE), point.size),
            0, request, prevalidated, ctypes.c_void_p(arena.ctypes.data + start), capacity,
            ctypes.byref(outputs), ctypes.byref(validity))
        np.testing.assert_array_equal(arena[:start], np.full(start, 165, dtype=np.uint8))
        np.testing.assert_array_equal(arena[start + capacity:], np.full(len(arena) - start - capacity, 165, dtype=np.uint8))
        assert status == validity.status
        return dict(f=f[0], g=g, h=h, status=status, valid_mask=validity.valid_mask,
                    requested=validity.requested, sentinel=sentinel,
                    reasons=np.array([validity.f_reason, validity.g_reason, validity.h_reason]),
                    dispatch=needed.dispatch, workspace_bytes=needed.bytes)


@pytest.fixture
def dependence():
    return lambda declared, request=7, **kwargs: NativeDependence(declared).evaluate(request, **kwargs)


def assert_untouched(actual, mask):
    for bit, name in ((1, "f"), (2, "g"), (4, "h")):
        if not mask & bit:
            value = np.asarray(actual[name])
            sentinel = np.full_like(value, actual["sentinel"])
            np.testing.assert_array_equal(value.view(np.uint64), sentinel.view(np.uint64))
            assert actual["reasons"][(1, 2, 4).index(bit)] == 3


def assert_close(actual, expected, mask, *, value_atol=3e-9, rtol=3e-8):
    for bit, name, reference in zip((1, 2, 4), ("f", "g", "h"), expected):
        if mask & bit:
            assert np.isfinite(actual[name]).all() and np.isfinite(reference).all(), name
            np.testing.assert_allclose(actual[name], reference, rtol=0 if bit == 1 else rtol,
                                       atol=value_atol if bit == 1 else rtol,
                                       err_msg=f"dependence {name} mask={mask}")


def assert_shape_close(actual, expected, mask):
    if mask & 2:
        np.testing.assert_allclose(actual["g"][-1], expected[1][-1], rtol=3e-8, atol=0)
    if mask & 4:
        np.testing.assert_allclose(actual["h"][-1], expected[2][-1], rtol=3e-8, atol=0)
        np.testing.assert_allclose(actual["h"][:, -1], expected[2][:, -1], rtol=3e-8, atol=0)


def exercise_masks(dependence, declared, expected, *, strict_shape=False):
    full = dependence(declared, 7)
    assert full["status"] == 0 and full["valid_mask"] == 7
    assert_close(full, expected, 7)
    if strict_shape:
        assert_shape_close(full, expected, 7)
    for mask in REQUESTS:
        actual = full if mask == 7 else dependence(declared, mask)
        assert actual["status"] == 0 and actual["valid_mask"] == mask
        assert actual["requested"] == mask
        assert_untouched(actual, mask)
        assert_close(actual, expected, mask)
        if strict_shape:
            assert_shape_close(actual, expected, mask)
        for bit, name in ((1, "f"), (2, "g"), (4, "h")):
            if mask & bit:
                np.testing.assert_allclose(actual[name], full[name], rtol=2e-12 if bit != 1 else 0,
                                           atol=2e-12 if bit != 1 else 2e-10)
        prevalidated = dependence(declared, mask, prevalidated=True)
        assert prevalidated["status"] == 0 and prevalidated["valid_mask"] == mask
        assert_untouched(prevalidated, mask)
        assert_close(prevalidated, expected, mask)
        if strict_shape:
            assert_shape_close(prevalidated, expected, mask)
    return full


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
def test_dependence_theta_all_masks_general_fixed_match_independent_ad(dependence, declared):
    actual = exercise_masks(dependence, declared, jet(declared))
    assert actual["dispatch"] == expected_dispatch(declared)
    np.testing.assert_allclose(actual["h"], actual["h"].T, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("p,q,n", LAYOUTS)
def test_dependence_runtime_fixed_flag_parity_and_actual_dispatch(dependence, law, p, q, n):
    runtime = case(law, p, q, n)
    fixed = deepcopy(runtime)
    fixed["fixed"] = True
    for mask in REQUESTS:
        a, b = dependence(runtime, mask), dependence(fixed, mask)
        assert a["status"] == b["status"] == 0 and a["valid_mask"] == b["valid_mask"] == mask
        assert a["dispatch"] == 0 and b["dispatch"] == expected_dispatch(fixed)
        assert_untouched(a, mask)
        assert_untouched(b, mask)
        assert_close(b, (a["f"], a["g"], a["h"]), mask, value_atol=2e-10, rtol=2e-12)


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("p,q,n", ((1, 1, 2), (3, 2, 3), (2, 3, 5)))
def test_dependence_oracle_scalar_mp_refines_and_matches_scipy(law, p, q, n):
    declared = case(law, p, q, n)
    low, high = mp_value(declared, digits=60), mp_value(declared, digits=100)
    with mp.workdps(100):
        assert abs(low - high) < mp.mpf("1e-48")
    assert np.isfinite(float(high))
    assert abs(scipy_value(declared) - float(high)) <= 3e-10


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("n", (2, 3, 7))
def test_dependence_complete_additive_constants_match_mp(dependence, law, n):
    declared = case(law, series=n)
    expected = float(mp_value(declared))
    for mask in (1, 3, 5, 7):
        actual = dependence(declared, mask)
        assert actual["status"] == 0 and actual["valid_mask"] == mask
        assert_close(actual, (expected, None, None), 1)


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("n", (2, 3, 7))
@pytest.mark.parametrize("zero", (False, True))
def test_dependence_identity_normal_zero_student_nonzero_not_marginal_likelihood(dependence, law, n, zero):
    declared = identity_case(law, n, zero=zero)
    expected = jet(declared)
    actual = exercise_masks(dependence, declared, expected)
    value = float(mp_value(declared))
    assert abs(actual["f"] - value) <= 3e-9
    if law == "normal":
        assert abs(value) <= 1e-80 and abs(actual["f"]) <= 3e-14
    else:
        assert abs(value) > 1e-6
    if zero:
        np.testing.assert_allclose(actual["g"][:2], [0.0, 0.0], rtol=0, atol=3e-14)


@pytest.mark.parametrize("law", LAWS)
def test_dependence_external_histories_seed_target_and_scored_prefix(dependence, law):
    declared = history_case(law)
    expected = jet(declared)
    exercise_masks(dependence, declared, expected)
    changed = deepcopy(declared)
    changed["residuals"][:declared["hold_back"] + declared["recursion_begin"]] *= 100.0
    changed["residuals"][declared["stop"]:] *= 100.0
    exercise_masks(dependence, changed, expected)
    for field in ("residual_history", "correlation_history", "seed", "target"):
        item = deepcopy(declared)
        if field == "seed":
            # Explicit Q histories supersede seed; the supplied seed is still validated.
            item[field] *= 3.0
            exercise_masks(dependence, item, expected)
            continue
        item[field] *= 1.3
        reference = jet(item)
        assert abs(reference[0] - expected[0]) > 1e-8
        exercise_masks(dependence, item, reference)
    prefix = deepcopy(declared)
    prefix["residuals"][3] *= 3.0
    prefix_jet = jet(prefix)
    assert abs(prefix_jet[0] - expected[0]) > 1e-6
    exercise_masks(dependence, prefix, prefix_jet)


@pytest.mark.parametrize("law", LAWS)
def test_dependence_score_selection_is_additive_without_resetting_state(dependence, law):
    declared = history_case(law)
    suffix = exercise_masks(dependence, declared, jet(declared))
    all_rows = scored_suffix(declared, declared["hold_back"] + declared["recursion_begin"])
    prefix = scored_suffix(all_rows, all_rows["hold_back"] + all_rows["recursion_begin"], 5)
    all_output, prefix_output = dependence(all_rows), dependence(prefix)
    assert all_output["status"] == prefix_output["status"] == 0
    for name in ("f", "g", "h"):
        np.testing.assert_allclose(all_output[name] - prefix_output[name], suffix[name], rtol=3e-11, atol=3e-10)


@pytest.mark.parametrize("law", LAWS)
def test_dependence_independent_q_r_state_sensitivities(law):
    declared = history_case(law, p=3, q=2, series=3)
    states, first, second = state_jet(declared)
    with mp.workdps(100):
        point = tuple(mp.mpf(float(v)) for v in declared["parameters"])
        expected = np.array([np.array(q.tolist(), dtype=float) for q in mp_state_path(declared, point)])
    np.testing.assert_allclose(states, expected, rtol=3e-14, atol=3e-15)
    p, q = declared["p_corr"], declared["q_corr"]
    initial_slopes = np.stack([np.outer(e, e) - declared["target"] for e in declared["residual_history"]]
                             + [state - declared["target"] for state in declared["correlation_history"]], axis=-1)
    np.testing.assert_allclose(first[0, ..., :p + q], initial_slopes, rtol=3e-14, atol=3e-15)
    np.testing.assert_array_equal(second[0], np.zeros_like(second[0]))
    assert np.max(abs(second[1:])) > 1e-5
    correlations, r_first, r_second = state_jet(declared, correlations=True)
    n = declared["series"]
    for row in range(len(states)):
        with mp.workdps(100):
            reference = np.array(mp_normalize(mp.matrix(expected[row].tolist())).tolist(), dtype=float)
        np.testing.assert_allclose(correlations[row], reference, rtol=3e-14, atol=3e-15)
    index = np.arange(n)
    np.testing.assert_array_equal(correlations[:, index, index], np.ones((len(states), n)))
    np.testing.assert_array_equal(r_first[:, index, index], np.zeros_like(r_first[:, index, index]))
    np.testing.assert_array_equal(r_second[:, index, index], np.zeros_like(r_second[:, index, index]))
    if law == "studentt":
        np.testing.assert_array_equal(first[..., -1], np.zeros_like(first[..., -1]))
        np.testing.assert_array_equal(second[..., -1], np.zeros_like(second[..., -1]))


@pytest.mark.parametrize("nu", SHAPES, ids=lambda value: repr(value))
@pytest.mark.parametrize("p,q,n", ((1, 1, 2), (2, 1, 3), (1, 1, 7)))
def test_dependence_student_shape_and_all_mixed_h_match_ad_and_mp(dependence, nu, p, q, n):
    declared = shape_case(nu, p, q, n)
    low, high = mp_jet(declared, digits=90), mp_jet(declared, digits=140)
    for left, right in zip(low, high):
        assert np.isfinite(left).all() and np.isfinite(right).all()
        np.testing.assert_allclose(left, right, rtol=3e-13, atol=0)
    ad = jet(declared)
    assert_close(dict(f=ad[0], g=ad[1], h=ad[2]), high, 7)
    actual = exercise_masks(dependence, declared, high, strict_shape=True)
    # No absolute tolerance may erase tiny asymptotic shape or cross derivatives.
    for output in (dict(g=ad[1], h=ad[2]), actual):
        np.testing.assert_allclose(output["g"][-1], high[1][-1], rtol=3e-8, atol=0)
        np.testing.assert_allclose(output["h"][-1], high[2][-1], rtol=3e-8, atol=0)
        np.testing.assert_allclose(output["h"][:, -1], high[2][:, -1], rtol=3e-8, atol=0)
    assert np.max(abs(high[2][:-1, -1])) > 0


@pytest.mark.parametrize("n", (2, 3, 7))
@pytest.mark.parametrize("nu", (2.0 + 2.0 ** -30, 128.0, 1e15))
def test_dependence_gamma_constant_twojet_mp_refinement(n, nu):
    low, high = gamma_germ(nu, n, digits=90), gamma_germ(nu, n, digits=140)
    np.testing.assert_allclose(low, high, rtol=3e-13, atol=0)
    with mp.workdps(140):
        point = mp.mpf(float(nu))
        if n == 2:
            identity = 2 * (mp.loggamma((point + 1) / 2) - mp.loggamma(point / 2)) - mp.log(point / 2)
            assert abs(identity - mp_gamma_constant(point, n)) < mp.mpf("1e-110")


@pytest.mark.parametrize("argument", (1e-18, 1e-8, 0.001, 0.01))
def test_dependence_oracle_small_tail_taylor_has_value_and_derivative_bounds(argument):
    with mp.workdps(350):
        x = mp.mpf(float(argument))
        exact = lambda value: mp.log1p(value) - value
        polynomial = lambda value: mp.fsum((-1) ** (k + 1) * value ** k / k for k in range(2, 13))
        bounds = (x ** 13 / (13 * (1 - x)), x ** 12 / (1 - x),
                  x ** 11 * (12 / (1 - x) + x / (1 - x) ** 2))
        for degree, bound in enumerate(bounds):
            left = exact(x) if degree == 0 else mp.diff(exact, x, degree)
            right = polynomial(x) if degree == 0 else mp.diff(polynomial, x, degree)
            assert abs(left - right) <= bound


@pytest.mark.parametrize("law", LAWS)
def test_dependence_long_horizon_near_persistence_keeps_large_state_sensitivities(dependence, law):
    declared = long_case(law)
    _, first, second = state_jet(declared)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert np.max(abs(first)) > 30 and np.max(abs(second)) > 1000
    expected = jet(declared)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9
    exercise_masks(dependence, declared, expected)


@pytest.mark.parametrize("declared", exact_sum_cases(), ids=lambda item: item["case_id"])
def test_dependence_exact_positive_sum_admission_not_host_longdouble(dependence, declared):
    with mp.workdps(140):
        total = mp.fsum(mp.mpf(float(value)) for value in declared["parameters"])
        admitted = total < 1
    assert admitted == declared["case_id"].endswith("admit")
    problem = NativeDependence(declared)
    assert problem.parameter_status() == (0 if admitted else 4)
    for mask in REQUESTS:
        actual = dependence(declared, mask)
        assert_untouched(actual, mask)
        if admitted:
            assert actual["status"] == 0 and actual["valid_mask"] == mask
            assert_close(actual, (0.0, np.zeros(3), np.zeros((3, 3))), mask)
        else:
            assert actual["status"] == 4 and actual["valid_mask"] == 0


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("kind", ("negative-a", "negative-b", "persistence", "nonfinite", "nu-boundary"))
def test_dependence_parameter_domain_rejections_do_not_manufacture_outputs(dependence, law, kind):
    declared = case(law)
    if kind == "negative-a":
        declared["parameters"][0] = -0.01
    elif kind == "negative-b":
        declared["parameters"][1] = -0.01
    elif kind == "persistence":
        declared["parameters"][1] = 1.1
    elif kind == "nonfinite":
        declared["parameters"][0] = np.nan
    elif law == "studentt":
        declared["parameters"][-1] = 2.0
    else:
        declared["parameters"][0] = -0.01
    for mask in REQUESTS:
        actual = dependence(declared, mask)
        assert actual["status"] == 4 and actual["valid_mask"] == 0
        assert_untouched(actual, mask)
        for name in ("f", "g", "h"):
            np.testing.assert_array_equal(actual[name], np.full_like(np.asarray(actual[name]), actual["sentinel"]))


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("which", ("target", "seed", "history"))
def test_dependence_individual_supplied_matrices_need_not_be_spd(dependence, law, which):
    declared = case(law, residuals=[[0.1, -0.2]], target=np.eye(2), seed=np.eye(2))
    if which == "target":
        declared["target"] = np.array([[1.0, 1.0], [1.0, 1.0]])
    elif which == "seed":
        declared["seed"] = np.array([[1.0, 1.1], [1.1, 1.0]])
        declared["parameters"][:2] = [0.0, 0.1]
    else:
        declared["correlation_history"] = np.array([[[1.0, 1.1], [1.1, 1.0]]])
        declared["parameters"][:2] = [0.0, 0.1]
    assert NativeDependence(declared).sample_status() == 0
    exercise_masks(dependence, declared, jet(declared))


@pytest.mark.parametrize("law", LAWS)
def test_dependence_empty_q_histories_repeat_supplied_seed_not_target(dependence, law):
    declared = case(law, p=3, q=2, residuals=retained_residuals(case(law))[:4])
    expected = jet(declared)
    exercise_masks(dependence, declared, expected)
    changed = deepcopy(declared)
    changed["seed"] = changed["target"].copy()
    reference = jet(changed)
    assert abs(reference[0] - expected[0]) > 1e-8
    exercise_masks(dependence, changed, reference)


@pytest.mark.parametrize("law", LAWS)
def test_dependence_only_scored_states_need_positive_diagonal_and_spd(dependence, law):
    declared = case(law, residuals=np.zeros((4, 2)), target=np.eye(2), seed=-np.eye(2),
                    first=3, parameters=[0.0, 0.8] + ([8.0] if law == "studentt" else []))
    with mp.workdps(100):
        point = tuple(mp.mpf(float(value)) for value in declared["parameters"])
        states = mp_state_path(declared, point)
        assert all(states[t][0, 0] < 0 for t in range(3)) and states[3][0, 0] > 0
    assert NativeDependence(declared).sample_status() == 0
    expected = jet(declared)
    exercise_masks(dependence, declared, expected)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9


def test_dependence_fixed_residual_student_shape_is_not_fixed_uniform_copula(dependence):
    from scipy import stats

    declared = case("studentt", parameters=[0.0, 0.0, 4.0])
    uniforms = stats.t.cdf(declared["residuals"] / np.sqrt(6.0 / 8.0), df=8.0)
    transformed = deepcopy(declared)
    transformed["residuals"] = np.sqrt(2.0 / 4.0) * stats.t.ppf(uniforms, df=4.0)
    fixed_residual_value, fixed_uniform_value = float(mp_value(declared)), float(mp_value(transformed))
    assert abs(fixed_residual_value - fixed_uniform_value) > 1e-6
    actual = dependence(declared, 1)
    assert actual["status"] == 0 and actual["valid_mask"] == 1
    assert_close(actual, (fixed_residual_value, None, None), 1)


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("scale", (2.0 ** -200, 2.0 ** -1000, 2.0 ** -1040))
def test_dependence_positive_tiny_q_has_no_diagonal_floor(dependence, law, scale):
    target = scale * np.array([[1.0, 0.25], [0.25, 1.0]])
    declared = case(law, residuals=[[0.1, -0.2]], target=target, seed=target,
                    parameters=[0.0, 0.0] + ([8.0] if law == "studentt" else []))
    expected = float(mp_value(declared))
    actual = dependence(declared, 1)
    assert actual["status"] == 0 and actual["valid_mask"] == 1
    assert_close(actual, (expected, None, None), 1)


@pytest.mark.parametrize("law", LAWS)
def test_dependence_disparate_diagonal_scales_normalize_without_root_product(dependence, law):
    scales = np.array([2.0 ** -500, 2.0 ** 500])
    target = np.array([[1.0, 0.25], [0.25, 1.0]]) * scales[:, None] * scales[None, :]
    declared = case(law, residuals=[[0.1, -0.2]], target=target, seed=target,
                    parameters=[0.0, 0.0] + ([8.0] if law == "studentt" else []))
    actual = dependence(declared, 1)
    assert actual["status"] == 0 and actual["valid_mask"] == 1
    assert_close(actual, (float(mp_value(declared)), None, None), 1)


@pytest.mark.parametrize("law", LAWS)
def test_dependence_near_singular_scored_r_keeps_exact_derivatives(dependence, law):
    rho = 1 - 2.0 ** -10
    target = np.array([[1.0, rho], [rho, 1.0]])
    declared = case(law, residuals=[[0.1, -0.2], [0.3, -0.4]], target=target, seed=target,
                    parameters=[0.0, 0.0] + ([8.0] if law == "studentt" else []))
    expected = mp_jet(declared)
    ad = jet(declared)
    assert_close(dict(f=ad[0], g=ad[1], h=ad[2]), expected, 7)
    exercise_masks(dependence, declared, expected)


def test_dependence_student_nu_boundary_raw_ratio_overflow_has_finite_mp_value(dependence):
    declared = case("studentt", residuals=[[1e154, 0.0]], target=np.eye(2), seed=np.eye(2),
                    parameters=[0.0, 0.0, np.nextafter(2.0, np.inf)])
    low, high = mp_value(declared, digits=90), mp_value(declared, digits=140)
    with mp.workdps(140):
        assert abs(low - high) < mp.mpf("1e-75")
    assert abs(float(high) - 372.03179295009113) <= 5e-13
    for mask in REQUESTS:
        actual = dependence(declared, mask)
        assert_untouched(actual, mask)
        if mask & 1:
            assert actual["valid_mask"] & 1 and actual["reasons"][0] == 0
            assert_close(actual, (float(high), None, None), 1)
        assert not actual["valid_mask"] & ~mask


def test_dependence_h_overflow_preserves_finite_f_g_not_fabricated_h(dependence):
    declared = case("normal", residuals=[[0.1, 0.2]], target=np.eye(2) * 1e-200,
                    seed=np.eye(2) * 1e-200, residual_history=[[1.0, 1.0]], parameters=[0.0, 0.5])
    expected = mp_jet(declared, request=3, digits=110)
    assert np.isfinite(expected[1]).all() and np.max(abs(expected[1])) > 1e190
    for mask in REQUESTS:
        actual = dependence(declared, mask)
        assert_untouched(actual, mask)
        valid = mask & 3
        assert actual["valid_mask"] == valid
        if mask & 4:
            assert actual["status"] == 5 and actual["reasons"][2] == 5
        else:
            assert actual["status"] == 0
        if valid:
            assert_close(actual, expected, valid)
            if mask & 2:
                np.testing.assert_allclose(actual["g"], expected[1], rtol=3e-11, atol=0)


@pytest.mark.parametrize("law", LAWS)
def test_dependence_singular_scored_state_is_not_jittered_or_padded(dependence, law):
    declared = case(law, residuals=[[0.1, -0.2]], target=np.ones((2, 2)), seed=np.ones((2, 2)))
    assert NativeDependence(declared).sample_status() == 0
    for mask in REQUESTS:
        actual = dependence(declared, mask)
        assert actual["status"] == 5 and actual["valid_mask"] == 0
        assert_untouched(actual, mask)


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("fault", ("short-arena", "misaligned-arena", "short-f", "short-g", "short-h"))
def test_dependence_checked_workspace_and_output_admission_is_write_free(dependence, law, fault):
    declared = case(law)
    width = len(declared["parameters"])
    kwargs = ({"bytes_delta": -1} if fault == "short-arena" else
              {"misaligned": True} if fault == "misaligned-arena" else
              {"output_counts": (0, width, width * width)} if fault == "short-f" else
              {"output_counts": (1, width - 1, width * width)} if fault == "short-g" else
              {"output_counts": (1, width, width * width - 1)})
    actual = dependence(declared, 7, **kwargs)
    assert actual["status"] == (1 if fault == "misaligned-arena" else 2)
    assert actual["valid_mask"] == 0
    for name in ("f", "g", "h"):
        np.testing.assert_array_equal(actual[name], np.full_like(np.asarray(actual[name]), actual["sentinel"]))


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("fault", ("asymmetric-target", "asymmetric-history", "nonfinite", "empty-score", "bad-origin"))
def test_dependence_immutable_sample_admission_rejects_invalid_inputs(law, fault):
    declared = history_case(law)
    if fault == "asymmetric-target":
        declared["target"][0, 1] += 0.01
    elif fault == "asymmetric-history":
        declared["correlation_history"][0, 0, 1] += 0.01
    elif fault == "nonfinite":
        declared["residual_history"][0, 0] = np.nan
    elif fault == "empty-score":
        declared["first_scored"] = len(retained_residuals(declared))
    else:
        declared["recursion_begin"] = declared["first_scored"] + 1
    assert NativeDependence(declared).sample_status() == 1


@pytest.mark.parametrize("law", LAWS)
def test_dependence_request_specific_workspace_orders_and_no_chart(law):
    problem = NativeDependence(case(law))
    needed = [problem.requirements(mask)[1] for mask in (1, 2, 4)]
    assert 0 < needed[0].bytes < needed[1].bytes < needed[2].bytes
    for mask in REQUESTS:
        status, result = problem.requirements(mask)
        assert status == 0
        assert result.bytes == needed[2 if mask & 4 else 1 if mask & 2 else 0].bytes
        assert problem.requirements(mask, coordinates=1)[0] == 3
    assert problem.requirements(0)[0] == problem.requirements(8)[0] == 1


@pytest.mark.parametrize("p,q,n", ((0, 1, 2), (1, 0, 2), (1, 1, 1)))
def test_dependence_zero_orders_single_series_are_explicitly_outside_scope(p, q, n):
    declared = case(parameters=[], p=1, q=1)
    declared.update(p_corr=p, q_corr=q, series=n)
    assert NativeDependence(declared).requirements(1)[0] == 1


def test_dependence_suite_has_no_legacy_import():
    assert not any(name == "scivol" or name.startswith("scivol.") for name in sys.modules)


@pytest.mark.parametrize("label,coefficients,expected_hex", SIMPLEX_ANCHOR_POINTS,
                         ids=[point[0] for point in SIMPLEX_ANCHOR_POINTS])
def test_dependence_simplex_anchor_rounds_exact_rational_not_a_rounded_sum(label, coefficients, expected_hex):
    expected_exact = {
        "ordinary-tie": Fraction(1, 2) - Fraction(1, 2 ** 55),
        "tiny-exact": Fraction(1, 2 ** 107),
        "tiny-rounded-up": Fraction(1, 2 ** 107) - Fraction(1, 2 ** 161),
        "tiny-rounded-down": Fraction(1, 2 ** 107) - Fraction(3, 2 ** 161),
        "boundary": Fraction(0),
    }[label]
    anchor = simplex_anchor(coefficients)
    assert anchor.exact_slack == expected_exact
    assert anchor.theta0 == tuple(coefficients)
    assert anchor.value.hex() == float.fromhex(expected_hex).hex()
    if "rounded" in label or label == "ordinary-tie":
        assert Fraction.from_float(anchor.value) != anchor.exact_slack
    if label.startswith("tiny"):
        assert 1.0 - float(sum((Fraction.from_float(value) for value in coefficients), Fraction(0))) == 0.0
        assert anchor.value > 0.0


@pytest.mark.parametrize("kind", ("exact", "rounded-up", "rounded-down"))
def test_dependence_simplex_affine_germ_keeps_value_slope_and_zero_curvature(kind):
    declared = simplex_anchor_case(kind=kind)
    anchor = simplex_anchor(declared["parameters"])
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(anchor.theta0)
    function = lambda theta: affine_simplex_slack(theta, anchor)
    assert float(function(point)).hex() == anchor.value.hex()
    np.testing.assert_array_equal(jax.grad(function)(point), -np.ones(len(point)))
    np.testing.assert_array_equal(jax.jacfwd(jax.grad(function))(point), np.zeros((len(point), len(point))))
    with mp.workdps(140):
        mp_point = tuple(mp.mpf(value) for value in anchor.theta0)
        mp_function = lambda *theta: mp_affine_simplex_slack(theta, anchor)
        assert mp_function(*mp_point) == mp.mpf(anchor.value)
        for row in range(len(mp_point)):
            first = tuple(int(i == row) for i in range(len(mp_point)))
            assert mp.diff(mp_function, mp_point, first) == -1
            for col in range(len(mp_point)):
                second = tuple(int(i == row) + int(i == col) for i in range(len(mp_point)))
                assert mp.diff(mp_function, mp_point, second) == 0


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("kind", ("exact", "rounded-up", "rounded-down"))
@pytest.mark.parametrize("fixed", (False, True), ids=("runtime", "fixed-flag"))
def test_dependence_portable_slack_anchor_nonproportional_states_full_crossblocks(dependence, law, kind, fixed):
    declared = simplex_anchor_case(law, kind, fixed=fixed)
    width = declared["p_corr"] + declared["q_corr"]
    anchor = simplex_anchor(declared["parameters"][:width])
    assert anchor.exact_slack > 0 and anchor.value > 0
    assert np.all(declared["residuals"] != 0.0)
    assert declared["target"][0, 1] > 0 > declared["seed"][0, 1]
    with mp.workdps(170):
        mp_point = tuple(mp.mpf(float(value)) for value in declared["parameters"])
        states = mp_state_path(declared, mp_point)
        normalized = mp_normalize(states[0])
        assert normalized[0, 1] > 0
        assert mp.det(normalized) > 0
    low, high = mp_jet(declared, digits=110), mp_jet(declared, digits=170)
    for left, right in zip(low, high):
        assert np.isfinite(left).all() and np.isfinite(right).all()
        np.testing.assert_allclose(left, right, rtol=3e-13, atol=0)
    assert abs(high[0]) > 1e-6
    assert np.max(abs(high[1][:width])) > 1e25
    assert np.max(abs(high[2][:width, :width])) > 1e55
    ad = jet(declared)
    assert_close(dict(f=ad[0], g=ad[1], h=ad[2]), high, 7)
    assert abs(scipy_value(declared) - high[0]) <= 3e-10
    if law == "studentt":
        assert_shape_close(dict(g=ad[1], h=ad[2]), high, 6)
        assert np.max(abs(high[2][:-1, -1])) > 1e25
    actual = exercise_masks(dependence, declared, high, strict_shape=law == "studentt")
    assert actual["dispatch"] == expected_dispatch(declared)


@pytest.mark.parametrize("fixed", (False, True), ids=("runtime", "fixed"))
def test_dependence_student_extreme_residual_h_keeps_scaled_correlation_curvature(dependence, fixed):
    declared = extreme_student_hessian_case(fixed=fixed)
    # Binary64 AD of the unscaled local second partial can lose this same term.
    low, high = mp_jet(declared, digits=110), mp_jet(declared, digits=170)
    for left, right in zip(low, high):
        assert np.isfinite(left).all() and np.isfinite(right).all()
        np.testing.assert_allclose(left, right, rtol=3e-13, atol=0)
    with mp.workdps(170):
        point = tuple(mp.mpf(float(value)) for value in declared["parameters"])
        state = mp_state_path(declared, point)[0]
        correlation = mp_normalize(state)
        assert state[0, 0] == state[1, 1]
        rho = correlation[0, 1]
        target_rho = mp.mpf(float(declared["target"][0, 1]))
        seed_rho = mp.mpf(float(declared["seed"][0, 1]))
        slopes = ((1 - target_rho) / state[0, 0], (seed_rho - target_rho) / state[0, 0])
        determinant = 1 - rho * rho
        square = mp.mpf(float(declared["residuals"][0, 0])) ** 2 / determinant
        unscaled = -(point[-1] + 2) / (2 * (point[-1] - 2 + square) ** 2)
        assert 0 < abs(unscaled) < mp.mpf(float(np.nextafter(0.0, 1.0))) / 2
        square_a = 2 * square * rho / determinant * slopes[0]
        missing_term = -(point[-1] + 2) / 2 * (square_a / (point[-1] - 2 + square)) ** 2
        assert abs(missing_term) > mp.mpf("0.01")
        # Separate rho-only large-E identity; its finite-E correction is below binary64 precision.
        curvature = (point[-1] + 1) * (1 + rho * rho) / (determinant * determinant)
        independent_block = np.array([[float(curvature * left * right) for right in slopes] for left in slopes])
    np.testing.assert_allclose(high[2][:2, :2], independent_block, rtol=3e-13, atol=0)
    actual = exercise_masks(dependence, declared, high, strict_shape=True)
    assert actual["dispatch"] == expected_dispatch(declared)
    np.testing.assert_allclose(actual["h"][:2, :2], independent_block, rtol=3e-12, atol=0)
