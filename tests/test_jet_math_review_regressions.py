"""Review range witnesses through the fresh native image's checked ABI.

The parent fresh-image runner supplies ``scivol._native_core``. Importing this
module or running only the oracle proofs does not load an extension or build one.
All oracle inputs are exact supplied doubles, with no clipping or finite differences.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
import pytest

from test_native_anchored_mean_chart import Jet


SENTINELS = (123.0, 456.0, 789.0)
CASES = [
    ("ged", 1), ("ged", 4),
    ("normal_gradient", 1), ("normal_gradient", 4),
    ("joint_normal", 1), ("arma_normal", 3),
    ("student_tail", 0), ("student_tail", 1), ("student_tail", 4),
]


@dataclass(frozen=True)
class Witness:
    kind: str
    columns: int
    density: int
    theta: NDArray[np.float64]
    data: NDArray[np.float64]
    design: NDArray[np.float64] | None
    seed: float
    representable: int


def _sample(kind: str, columns: int) -> Witness:
    if kind == "ged":
        tail, data, design_value, seed, density, representable = (
            [1., .1, .1, 4.], [1e-252], 1e100, 1e-152, 3, 7)
    elif kind == "normal_gradient":
        tail, data, design_value, seed, density, representable = (
            [1e-200], [1e-200], 1e250, 0., 0, 3)
    elif kind == "student_tail":
        tail, data, design_value, seed, density, representable = (
            [1e-320, 8.], [.1], 1., 0., 1, 1)
    else:
        tail, data, design_value, seed, density, representable = (
            [5e307, .1, .5], [0., 1e155], 1., 1e308, 0, 7)
    theta = np.array([0.] * columns + tail, dtype=np.float64)
    observations = np.array(data, dtype=np.float64)
    design = np.zeros((len(data), columns), dtype=np.float64)
    if columns:
        design[:, 0] = design_value
        if kind == "joint_normal":
            design[0, 0] = 0.
    return Witness(kind, columns, density, theta, observations,
                   None if kind == "arma_normal" else design, seed, representable)


def _unary(point: Jet, value: mp.mpf, first: mp.mpf, second: mp.mpf) -> Jet:
    width = len(point.gradient)
    return Jet(value, tuple(first * entry for entry in point.gradient), tuple(
        first * point.hessian[i * width + j]
        + second * point.gradient[i] * point.gradient[j]
        for i in range(width) for j in range(width)))


def _log(point: Jet) -> Jet:
    return _unary(point, mp.log(point.value), 1 / point.value, -1 / point.value**2)


def _lgamma(point: Jet) -> Jet:
    return _unary(point, mp.loggamma(point.value), mp.digamma(point.value),
                  mp.polygamma(1, point.value))


def _student_squared_nll(square: Jet | mp.mpf, variance: Jet, nu: Jet) -> Jet:
    half = mp.mpf(".5")
    a = nu - 2
    log_variance = _log(variance)
    normalizer = (half * (mp.log(mp.pi) + _log(a))
                  + _lgamma(nu * half) - _lgamma((nu + 1) * half))
    log_tail = _log(a * variance + square) - log_variance - _log(a)
    return normalizer + half * log_variance + half * (nu + 1) * log_tail


def _hansen_nll(residual: Jet, variance: Jet, nu: Jet, skew: Jet) -> Jet:
    half = mp.mpf(".5")
    log_c = (_lgamma((nu + 1) * half) - _lgamma(nu * half)
             - half * (mp.log(mp.pi) + _log(nu - 2)))
    c = mp.exp(log_c.value)
    shift = 4 * skew * _unary(log_c, c, c, c) * (nu - 2) / (nu - 1)
    scale2 = 1 + 3 * skew * skew - shift * shift
    scale = mp.sqrt(scale2.value)
    scale = _unary(scale2, scale, 1 / (2 * scale), -1 / (4 * scale**3))
    root = mp.sqrt(variance.value)
    root = _unary(variance, root, 1 / (2 * root), -1 / (4 * root**3))
    centered = scale * residual + shift * root
    side = mp.sign(centered.value)
    assert side != 0
    adjusted = centered / (1 + side * skew)
    return _student_squared_nll(adjusted * adjusted, variance, nu) - _log(scale)


def _density_nll(residual: Jet, variance: Jet, shape: list[Jet], density: int) -> Jet:
    half = mp.mpf(".5")
    log_variance = _log(variance)
    if density == 0:
        return half * (mp.log(2 * mp.pi) + log_variance + residual * residual / variance)
    nu = shape[0]
    if density == 1:
        return _student_squared_nll(residual * residual, variance, nu)
    inverse_nu = nu.reciprocal()
    log_scale = half * (_lgamma(inverse_nu) - _lgamma(3 * inverse_nu))
    normalizer = mp.log(2) + log_scale + _lgamma(inverse_nu) - _log(nu)
    log_power = nu * (_log(residual * mp.sign(residual.value))
                      - half * log_variance - log_scale)
    power = mp.exp(log_power.value)
    return normalizer + half * log_variance + _unary(log_power, power, power, power)


def _oracle(case: Witness, digits: int) -> tuple[NDArray[np.float64], ...]:
    with mp.workdps(digits):
        width = case.theta.size
        theta = [Jet.variable(float(value), width, i)
                 for i, value in enumerate(case.theta)]
        objective = Jet.constant(0, width)
        previous_residual = Jet.constant(0, width)
        previous_variance = Jet.constant(case.seed, width)
        first = 1 if case.kind == "arma_normal" else 0
        for t in range(first, case.data.size):
            if case.kind == "arma_normal":
                residual = (float(case.data[t]) - theta[0]
                            - theta[1] * float(case.data[t - 1])
                            - theta[2] * previous_residual)
            else:
                residual = Jet.constant(float(case.data[t]), width) - sum(
                    (theta[j] * float(case.design[t, j]) for j in range(case.columns)), 0)
            if case.kind in {"ged", "joint_normal", "arma_normal"}:
                omega, alpha, beta = theta[case.columns:case.columns + 3]
                variance = (Jet.constant(case.seed, width) if t == 0 else
                            omega + alpha * previous_residual * previous_residual
                            + beta * previous_variance)
                shape = theta[case.columns + 3:]
            else:
                variance, shape = theta[case.columns], theta[case.columns + 1:]
            objective += _density_nll(residual, variance, shape, case.density)
            previous_residual, previous_variance = residual, variance
        return (np.array([float(objective.value)]),
                np.array([float(value) for value in objective.gradient]),
                np.array([float(value) for value in objective.hessian]).reshape(width, width))


@pytest.fixture(scope="module", params=CASES, ids=[f"{kind}-{columns}col" for kind, columns in CASES])
def witness(request):
    case = _sample(*request.param)
    expected = _oracle(case, 120)
    for actual, refined in zip(expected, _oracle(case, 200)):
        np.testing.assert_array_equal(actual, refined)
    return case, expected


def test_review_oracle_proves_request_domains(witness):
    case, expected = witness
    finite_mask = sum(bit for bit, output in zip((1, 2, 4), expected)
                      if np.isfinite(output).all())
    assert finite_mask == case.representable
    with mp.workdps(200):
        e = mp.mpf(float(case.data[-1]))
        h = mp.mpf(case.seed if case.kind in {"ged", "joint_normal", "arma_normal"}
                   else float(case.theta[case.columns]))
        if case.kind == "ged":
            x = mp.mpf(float(case.design[0, 0]))
            b4 = (mp.gamma(mp.mpf(1) / 4) / mp.gamma(mp.mpf(3) / 4))**2
            local_curvature = 12 * (e / mp.sqrt(h))**2 / b4
            assert float(local_curvature) == 0.
            assert np.isfinite(float(x / mp.sqrt(h)))
            np.testing.assert_allclose(expected[2][0, 0], float(12 * e**2 * x**2 / (b4 * h**2)),
                                       rtol=2e-15, atol=0.)
            np.testing.assert_allclose(expected[2][0, 0], 1.3708397431333905, rtol=2e-15, atol=0.)
        elif case.kind == "normal_gradient":
            x, residual, variance = map(Fraction, map(float, (
                case.design[0, 0], case.data[0], case.theta[case.columns])))
            assert expected[1][0] == float(-x * residual / variance)
            assert expected[1][case.columns] == float(
                Fraction(1, 2) / variance - residual**2 / (2 * variance**2))
            assert x**2 / variance > Fraction(float(np.finfo(float).max))
            with np.errstate(over="ignore"):
                assert np.isinf(case.design[0, 0] / np.sqrt(case.theta[case.columns]))
        elif case.kind in {"joint_normal", "arma_normal"}:
            assert e**2 > mp.mpf(float(np.finfo(float).max))
            beta = case.columns + 2
            np.testing.assert_allclose(expected[1][beta], -49.5, rtol=2e-15, atol=0.)
            np.testing.assert_allclose(expected[2][beta, beta], 99.5, rtol=2e-15, atol=0.)
        else:
            assert e**2 / (6 * h) > mp.mpf(float(np.finfo(float).max))
            assert np.isneginf(expected[1][case.columns])
            assert np.isposinf(expected[2][case.columns, case.columns])
            np.testing.assert_allclose(expected[0][0], 2919.329047691347, rtol=0., atol=2e-12)


@pytest.fixture(scope="module")
def native_core():
    from scivol import _native_core
    return _native_core


def _arma_request(native, case: Witness, mask: int, fixed: bool,
                  outputs: tuple[NDArray[np.float64], ...]) -> bool:
    assert native._arma_garch_dispatch_kind(1, 1, 1, 1, 0) == 1
    if fixed:
        size = native._arma_garch_workspace_bytes(1, 1, 1, 1, 0, 0, mask)
    else:
        # Fixed dispatch needs no arena; the runtime sibling has its own planner.
        library = ctypes.CDLL(native.__file__)
        size_type, unsigned, pointer = ctypes.c_size_t, ctypes.c_uint32, ctypes.c_void_p
        planner = library.vlk_runtime_arma_garch_jet_workspace_bytes
        planner.argtypes = [size_type] * 4 + [unsigned] * 3
        planner.restype = size_type
        size = planner(1, 1, 1, 1, 0, 0, mask)
        assert size > 0
    workspace = np.empty(max(size, 1), dtype=np.uint8)
    arguments = (case.theta.ctypes.data, case.theta.size, case.data.ctypes.data,
                 case.data.size, case.seed, 1, 1, 1, 1, 0, 0, mask,
                 workspace.ctypes.data, workspace.nbytes,
                 *(array.ctypes.data for array in outputs))
    if fixed:
        status, valid = native._arma_garch_jet(*arguments)
    else:
        # The forced-runtime export belongs to the same fresh extension image.
        evaluate = library.vlk_runtime_arma_garch_jet_evaluate
        evaluate.argtypes = [pointer, size_type, pointer, size_type, ctypes.c_double,
                             size_type, size_type, size_type, size_type, unsigned, unsigned, unsigned,
                             pointer, size_type, pointer, pointer, pointer, ctypes.POINTER(unsigned)]
        evaluate.restype = ctypes.c_int32
        validity = unsigned(0)
        status = evaluate(*arguments, ctypes.byref(validity))
        valid = validity.value
    assert (status, valid) == (0, mask), (case.kind, mask, fixed, status, valid)
    return True


def _request(native, case: Witness, mask: int, fixed: bool):
    width = case.theta.size
    outputs = (np.full(1, SENTINELS[0]), np.full(width, SENTINELS[1]),
               np.full((width, width), SENTINELS[2]))
    if case.kind in {"normal_gradient", "student_tail"}:
        descriptor = (case.data.size, case.columns, 0, 0, 0, 0, case.density,
                      True, fixed, 0, 0, 0)
        assert native._mean_dispatch_kind(descriptor) == int(fixed and case.columns == 4)
        size = native._mean_sizes(descriptor, 0)[0]
        workspace = np.empty(max(size, 1), dtype=np.uint8)
        valid = native._mean_jet(
            descriptor, case.theta.ctypes.data, case.data.ctypes.data,
            case.design.ctypes.data if case.columns else 0, mask, False,
            workspace.ctypes.data, workspace.nbytes,
            *(array.ctypes.data for array in outputs), 0)
    elif case.kind == "arma_normal":
        valid = _arma_request(native, case, mask, fixed, outputs)
    else:
        plan = (0, case.density, case.columns, 1, 1, case.data.size, fixed)
        size = native._linear_joint_sizes(plan, mask, False, (0, 0, 0, 0))[0]
        workspace = np.empty(max(size, 1), dtype=np.uint8)
        valid = native._linear_joint_jet(
            plan, case.theta.ctypes.data, case.data.ctypes.data, case.design.ctypes.data,
            case.seed, mask, False, workspace.ctypes.data, workspace.nbytes,
            *(array.ctypes.data for array in outputs), 0, 0)
    return bool(valid), outputs


@pytest.mark.parametrize("fixed", [False, True], ids=["runtime", "fixed-enabled"])
def test_review_witness_all_sparse_requests(native_core, witness, fixed):
    case, expected = witness
    successful = {}
    for mask in range(1, 8):
        valid, outputs = _request(native_core, case, mask, fixed)
        # Failed requested buffers have no output guarantee. Unrequested ones do.
        for bit, output, sentinel in zip((1, 2, 4), outputs, SENTINELS):
            if not mask & bit:
                np.testing.assert_array_equal(output, sentinel)
        assert valid == ((mask & case.representable) == mask), (case.kind, fixed, mask)
        if not valid:
            continue
        successful[mask] = outputs
        for bit, output, reference in zip((1, 2, 4), outputs, expected):
            if mask & bit:
                assert np.isfinite(output).all()
                np.testing.assert_allclose(
                    output, reference, rtol=0. if bit == 1 else 3e-10,
                    atol=2e-11 if bit == 1 else 8e-323,
                    err_msg=f"{case.kind}, fixed={fixed}, mask={mask}, output={bit}")
    full = successful[case.representable]
    for mask, outputs in successful.items():
        for bit, output, reference in zip((1, 2, 4), outputs, full):
            if mask & bit:
                np.testing.assert_allclose(output, reference,
                                           rtol=0. if bit == 1 else 3e-12,
                                           atol=2e-11 if bit == 1 else 8e-323)


@pytest.mark.parametrize("p,q,dispatch", [(1, 1, 1), (3, 2, 0)], ids=["fixed11", "runtime32"])
def test_standalone_normal_uses_supplied_squares_all_masks(native_core, p, q, dispatch):
    pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import garch_value_grad_hess

    squared = np.array([.25, 1., 2.25, .0625, 4., .5625, 0., 1.5625], dtype=np.float64)
    theta = np.r_[.2, np.full(p, .12 / p), np.full(q, .65 / q)]
    seed = float(squared.mean())
    value, gradient, hessian = garch_value_grad_hess(theta, squared, p, q, "normal")
    expected = (np.array([value + .5 * squared.size * np.log(2 * np.pi)]), gradient, hessian)
    assert native_core._garch_dispatch_kind(p, q, 0, 0) == dispatch
    # Normal does not require raw residuals; even their overflowing squares are irrelevant.
    raw_variants = (np.sqrt(squared), np.full(squared.size, 1e155), None)
    for mask in range(1, 8):
        size = native_core._garch_workspace_bytes(p, q, 0, 0, mask)
        workspace = np.empty(max(size, 1), dtype=np.uint8)
        baseline = None
        for raw in raw_variants:
            outputs = (np.full(1, SENTINELS[0]), np.full(theta.size, SENTINELS[1]),
                       np.full((theta.size, theta.size), SENTINELS[2]))
            status, valid = native_core._garch_jet(
                theta.ctypes.data, 0 if raw is None else raw.ctypes.data,
                squared.ctypes.data, squared.size, seed, p, q, 0, 0, mask,
                workspace.ctypes.data, workspace.nbytes,
                *(array.ctypes.data for array in outputs))
            assert (status, valid) == (0, mask)
            for bit, output, reference, sentinel in zip((1, 2, 4), outputs, expected, SENTINELS):
                if not mask & bit:
                    np.testing.assert_array_equal(output, sentinel)
                    continue
                assert np.isfinite(output).all()
                np.testing.assert_allclose(output, reference,
                                           rtol=0. if bit == 1 else 3e-12,
                                           atol=2e-11 if bit == 1 else 3e-12)
            if baseline is not None:
                for output, reference in zip(outputs, baseline):
                    np.testing.assert_array_equal(output, reference)
            baseline = outputs


def _conditioned_tail_oracle(case: Witness, digits: int) -> tuple[NDArray[np.float64], ...]:
    with mp.workdps(digits):
        width = case.theta.size
        theta = [Jet.variable(float(value), width, i)
                 for i, value in enumerate(case.theta)]
        variance = Jet.constant(case.seed, width)
        if case.density == 1:
            square = mp.mpf(float(case.data[0] * case.data[0]))
            objective = _student_squared_nll(square, variance, theta[-1])
        else:
            residual = Jet.constant(float(case.data[0]), width)
            objective = _hansen_nll(residual, variance, theta[-2], theta[-1])
        return (np.array([float(objective.value)]),
                np.array([float(value) for value in objective.gradient]),
                np.array([float(value) for value in objective.hessian]).reshape(width, width))


@pytest.fixture(scope="module", params=[(1, 1, 1), (3, 2, 0)], ids=["fixed11", "runtime32"])
def conditioned_tail(request):
    p, q, dispatch = request.param
    theta = np.r_[1., np.full(p, .1 / p), np.full(q, .1 / q), 8.]
    data = np.array([.1], dtype=np.float64)
    cases = [Witness("conditioned_student", 0, 1, theta, data, None, 1e-320, 7),
             Witness("conditioned_skewt", 0, 2, np.r_[theta, 0.], data, None, 1e-320, 7)]
    expected = {}
    for case in cases:
        expected[case.density] = _conditioned_tail_oracle(case, 120)
        for actual, refined in zip(expected[case.density], _conditioned_tail_oracle(case, 200)):
            np.testing.assert_array_equal(actual, refined)
    return p, q, dispatch, cases, expected


def test_conditioned_tail_oracle_proves_finite_shape_jets(conditioned_tail):
    p, q, _, cases, expected = conditioned_tail
    volatility = 1 + p + q
    for case in cases:
        value, gradient, hessian = expected[case.density]
        assert all(np.isfinite(output).all() for output in (value, gradient, hessian))
        np.testing.assert_array_equal(gradient[:volatility], 0.)
        np.testing.assert_array_equal(hessian[:volatility, :], 0.)
        np.testing.assert_array_equal(hessian[:, :volatility], 0.)
    student, skewt = expected[1], expected[2]
    np.testing.assert_allclose(student[0][0], 2919.329047691347, rtol=0., atol=2e-12)
    np.testing.assert_allclose(student[1][-1], 364.48211232224835, rtol=3e-12, atol=0.)
    np.testing.assert_allclose(student[2][-1, -1], -.046781092381029318, rtol=3e-12, atol=0.)
    for bit, actual, reference in zip((1, 2, 4), (skewt[0], skewt[1][:-1], skewt[2][:-1, :-1]), student):
        np.testing.assert_allclose(actual, reference, rtol=0. if bit == 1 else 3e-12,
                                   atol=2e-11 if bit == 1 else 8e-323)
    np.testing.assert_allclose(skewt[1][-1], -9., rtol=3e-12, atol=0.)
    np.testing.assert_allclose(skewt[2][-1, -1], 14.25, rtol=3e-12, atol=0.)
    np.testing.assert_allclose(skewt[2][-2, -1], -1., rtol=3e-12, atol=0.)


def test_conditioned_student_skewt_shape_tail_all_masks(native_core, conditioned_tail):
    p, q, dispatch, cases, expected = conditioned_tail
    volatility = 1 + p + q
    successful = {}
    for case in cases:
        assert native_core._garch_dispatch_kind(p, q, case.density, 0) == dispatch
        squared = case.data * case.data
        successful[case.density] = {}
        for mask in range(1, 8):
            size = native_core._garch_workspace_bytes(p, q, case.density, 0, mask)
            workspace = np.empty(max(size, 1), dtype=np.uint8)
            outputs = (np.full(1, SENTINELS[0]), np.full(case.theta.size, SENTINELS[1]),
                       np.full((case.theta.size, case.theta.size), SENTINELS[2]))
            status, valid = native_core._garch_jet(
                case.theta.ctypes.data, case.data.ctypes.data, squared.ctypes.data,
                squared.size, case.seed, p, q, case.density, 0, mask,
                workspace.ctypes.data, workspace.nbytes,
                *(array.ctypes.data for array in outputs))
            assert (status, valid) == (0, mask), (case.kind, p, q, mask, status, valid)
            for bit, output, reference, sentinel in zip((1, 2, 4), outputs, expected[case.density], SENTINELS):
                if not mask & bit:
                    np.testing.assert_array_equal(output, sentinel)
                    continue
                assert np.isfinite(output).all()
                np.testing.assert_allclose(output, reference,
                                           rtol=0. if bit == 1 else 3e-12,
                                           atol=2e-11 if bit == 1 else 8e-323)
            if mask & 2:
                np.testing.assert_array_equal(outputs[1][:volatility], 0.)
            if mask & 4:
                np.testing.assert_array_equal(outputs[2][:volatility, :], 0.)
                np.testing.assert_array_equal(outputs[2][:, :volatility], 0.)
            successful[case.density][mask] = outputs
        for mask, outputs in successful[case.density].items():
            for bit, output, reference in zip((1, 2, 4), outputs, successful[case.density][7]):
                if mask & bit:
                    np.testing.assert_allclose(output, reference,
                                               rtol=0. if bit == 1 else 3e-12,
                                               atol=2e-11 if bit == 1 else 8e-323)
    for mask in range(1, 8):
        student = successful[1][mask]
        skewt = successful[2][mask]
        common = (skewt[0], skewt[1][:-1], skewt[2][:-1, :-1])
        for bit, output, reference in zip((1, 2, 4), common, student):
            if mask & bit:
                np.testing.assert_allclose(output, reference,
                                           rtol=0. if bit == 1 else 3e-12,
                                           atol=2e-11 if bit == 1 else 8e-323)
