from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scivol import (
    ARMA,
    GED,
    GARCH,
    GeneralizedHyperbolic,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
)
from scivol import _native_core
from scivol._evaluation import filter_spec
from scivol._gh import standardized_gh_rvs
from scivol._native_fit import (
    NativeModel,
    _describe_model,
    _initial_theta,
    _jet_workspace,
    _theta_to_coordinates,
)


DensityFactory = Callable[[], object]

_DENSITIES: tuple[tuple[str, DensityFactory], ...] = (
    ("normal", Normal),
    ("studentt", StudentT),
    ("skewt", SkewT),
    ("ged", GED),
    ("gh", GeneralizedHyperbolic),
    ("nig", NormalInverseGaussian),
    ("hyperbolic", Hyperbolic),
    ("vg", VarianceGamma),
)
_REQUESTS = tuple(range(1, 8))
_HALF_LOG_2PI = 0.9189385332046727


def _returns(seed: int = 20260827, n: int = 700) -> np.ndarray:
    generator = np.random.default_rng(seed)
    burn = 300
    innovations = generator.standard_t(7, size=n + burn) * np.sqrt(5.0 / 7.0)
    return _filter_garch_innovations(innovations, burn)


def _gh_returns(seed: int = 20260827, n: int = 1200) -> np.ndarray:
    generator = np.random.default_rng(seed)
    burn = 300
    innovations = standardized_gh_rvs(
        generator, n + burn, 0.15, 1.3, 0.4
    )
    return _filter_garch_innovations(innovations, burn)


def _filter_garch_innovations(
    innovations: np.ndarray, burn: int
) -> np.ndarray:
    residuals = np.empty(innovations.size, dtype=np.float64)
    variance = np.empty(innovations.size, dtype=np.float64)
    variance[0] = 2e-5 / (1.0 - 0.08 - 0.88)
    residuals[0] = np.sqrt(variance[0]) * innovations[0]
    for time_index in range(1, innovations.size):
        variance[time_index] = (
            2e-5
            + 0.08 * residuals[time_index - 1] ** 2
            + 0.88 * variance[time_index - 1]
        )
        residuals[time_index] = (
            np.sqrt(variance[time_index]) * innovations[time_index]
        )
    return np.ascontiguousarray(residuals[burn:], dtype=np.float64)


def _joint_returns(
    seed: int = 778,
    n: int = 1400,
    p_ar: int = 2,
    q_ma: int = 2,
    p_vol: int = 3,
    q_vol: int = 2,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    burn = 500
    total = n + burn
    innovations = generator.standard_t(7, total) * np.sqrt(5.0 / 7.0)
    ar = np.array([0.28, -0.12], dtype=np.float64)[:p_ar]
    ma = np.array([0.20, -0.08], dtype=np.float64)[:q_ma]
    alpha = np.full(p_vol, 0.08 / p_vol, dtype=np.float64)
    beta = np.full(q_vol, 0.85 / q_vol, dtype=np.float64)
    maximum_lag = max(p_ar, q_ma, p_vol, q_vol)
    variance = np.full(total, 2e-5 / 0.07, dtype=np.float64)
    residuals = np.zeros(total, dtype=np.float64)
    observations = np.zeros(total, dtype=np.float64)
    for time_index in range(maximum_lag, total):
        variance[time_index] = 2e-5
        for lag, coefficient in enumerate(alpha, start=1):
            variance[time_index] += coefficient * residuals[time_index - lag] ** 2
        for lag, coefficient in enumerate(beta, start=1):
            variance[time_index] += coefficient * variance[time_index - lag]
        residuals[time_index] = np.sqrt(variance[time_index]) * innovations[time_index]
        observations[time_index] = 0.001 + residuals[time_index]
        for lag, coefficient in enumerate(ar, start=1):
            observations[time_index] += coefficient * observations[time_index - lag]
        for lag, coefficient in enumerate(ma, start=1):
            observations[time_index] += coefficient * residuals[time_index - lag]
    return np.ascontiguousarray(observations[burn:], dtype=np.float64)


def _model_and_parameters(
    kind: str,
    density_factory: DensityFactory,
) -> tuple[NativeModel, np.ndarray, np.ndarray]:
    data = _returns(n=48)
    if kind == "garch":
        spec = GARCH(2, 1) + density_factory()
    else:
        spec = ARMA(2, 1) + GARCH(2, 1) + density_factory()
    model = _describe_model(spec)
    theta, _ = _initial_theta(spec, model, data, None)
    if kind == "arma_garch":
        theta[:4] = np.array([0.002, 0.15, -0.05, -0.10])
    return model, data, theta


def _evaluate_mask(
    model: NativeModel,
    data: np.ndarray,
    parameters: np.ndarray,
    request: int,
    *,
    transformed: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    value = np.full(1, np.nan, dtype=np.float64)
    gradient = np.full(model.n_params, np.nan, dtype=np.float64)
    hessian = np.full((model.n_params, model.n_params), np.nan, dtype=np.float64)
    workspace = _jet_workspace(model, transformed=transformed)
    initial_variance = float(np.mean(data * data))

    if model.kind == "garch":
        squared = np.ascontiguousarray(data * data, dtype=np.float64)
        status, valid = _native_core._garch_jet(
            parameters.ctypes.data,
            data.ctypes.data,
            squared.ctypes.data,
            data.size,
            initial_variance,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    else:
        status, valid = _native_core._arma_garch_jet(
            parameters.ctypes.data,
            model.n_params,
            data.ctypes.data,
            data.size,
            initial_variance,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )

    assert status == 0
    assert valid == request
    return float(value[0]), gradient, hessian


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize("kind", ("garch", "arma_garch"))
@pytest.mark.parametrize("transformed", (False, True))
def test_every_native_request_matches_the_full_jet(
    density_name: str,
    density_factory: DensityFactory,
    kind: str,
    transformed: bool,
) -> None:
    model, data, theta = _model_and_parameters(kind, density_factory)
    parameters = _theta_to_coordinates(model, theta) if transformed else theta
    full_value, full_gradient, full_hessian = _evaluate_mask(
        model, data, parameters, 7, transformed=transformed
    )

    for request in _REQUESTS:
        value, gradient, hessian = _evaluate_mask(
            model, data, parameters, request, transformed=transformed
        )
        if request & 1:
            assert value == pytest.approx(full_value, rel=2e-12, abs=2e-12)
        if request & 2:
            assert_allclose(gradient, full_gradient, rtol=2e-11, atol=2e-11)
        if request & 4:
            assert_allclose(hessian, full_hessian, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize(
    "density_factory,oracle_name",
    ((Normal, "normal"), (StudentT, "studentt"), (SkewT, "skewt"), (GED, "ged")),
)
def test_joint_theta_jet_matches_independent_ad(
    density_factory: DensityFactory,
    oracle_name: str,
) -> None:
    pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import arma_garch_value_grad_hess

    model, data, theta = _model_and_parameters("arma_garch", density_factory)
    if oracle_name == "ged":
        theta[-1] = 0.8
    value, gradient, hessian = _evaluate_mask(
        model, data, theta, 7, transformed=False
    )
    expected_value, expected_gradient, expected_hessian = (
        arma_garch_value_grad_hess(
            theta,
            data,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            dist=oracle_name,
        )
    )
    n_eff = data.size - model.maximum_lag
    if oracle_name == "normal":
        expected_value += _HALF_LOG_2PI

    assert value == pytest.approx(n_eff * expected_value, rel=2e-9, abs=2e-8)
    assert_allclose(gradient, n_eff * expected_gradient, rtol=2e-7, atol=2e-7)
    assert_allclose(hessian, n_eff * expected_hessian, rtol=3e-6, atol=3e-6)


def test_garch_ged_theta_jet_matches_ad_below_legacy_shape_padding() -> None:
    pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import garch_value_grad_hess

    model, data, theta = _model_and_parameters("garch", GED)
    theta[-1] = 0.8
    value, gradient, hessian = _evaluate_mask(
        model, data, theta, 7, transformed=False
    )
    expected = garch_value_grad_hess(
        theta, data * data, model.p_vol, model.q_vol, dist="ged"
    )

    assert value == pytest.approx(expected[0], rel=2e-9, abs=2e-8)
    assert_allclose(gradient, expected[1], rtol=2e-7, atol=2e-7)
    assert_allclose(hessian, expected[2], rtol=3e-6, atol=3e-6)
    filtered = filter_spec(GARCH(2, 1) + GED(), data, theta)
    assert -filtered.log_likelihood == pytest.approx(value, rel=2e-10, abs=2e-9)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize("kind", ("garch", "arma_garch"))
def test_native_density_matrices_fit_credibly(
    density_name: str,
    density_factory: DensityFactory,
    kind: str,
) -> None:
    data = _gh_returns() if density_name == "gh" else _returns()
    spec = (
        GARCH(2, 1) + density_factory()
        if kind == "garch"
        else ARMA(1, 1) + GARCH(2, 1) + density_factory()
    )
    result = spec.fit(data, solver="native", maxiter=700)

    assert result.converged, result.convergence_message
    assert result.fit_info.optimizer_mode == 0
    assert result.fit_info.kkt_residual <= result.fit_info.kkt_tolerance
    assert result.fit_info.maximum_constraint_violation <= result.fit_info.kkt_tolerance
    assert result.fit_info.objective_parity_error <= 1e-7
    assert result.fit_info.engine == "cpp-function-jet"
    assert np.isfinite(result.log_likelihood)


def test_joint_fit_has_cross_component_curvature_and_inference() -> None:
    generator = np.random.default_rng(42)
    n, burn = 1200, 300
    omega, alpha, beta = 2e-5, 0.08, 0.85
    c, phi, theta_ma = 0.001, 0.25, -0.15
    variance = np.empty(n + burn)
    residual = np.empty(n + burn)
    observations = np.empty(n + burn)
    variance[0] = omega / (1.0 - alpha - beta)
    residual[0] = 0.0
    observations[0] = c
    for time in range(1, n + burn):
        variance[time] = (
            omega + alpha * residual[time - 1] ** 2 + beta * variance[time - 1]
        )
        residual[time] = np.sqrt(variance[time]) * generator.normal()
        observations[time] = (
            c
            + phi * observations[time - 1]
            + theta_ma * residual[time - 1]
            + residual[time]
        )

    data = np.ascontiguousarray(observations[burn:], dtype=np.float64)
    result = (ARMA(1, 1) + GARCH(1, 1) + Normal()).fit(
        data, solver="native", maxiter=1000
    )

    assert result.converged, result.convergence_message
    assert result.hessian is not None
    assert np.any(np.abs(result.hessian[:3, 3:6]) > 1e-8)
    assert result.cov_matrix is not None
    assert result.std_errors is not None
    assert np.all(np.isfinite(result.std_errors))


@pytest.mark.parametrize(
    "spec,data",
    (
        (GARCH(3, 2) + Normal(), _returns(seed=778, n=900)),
        (
            ARMA(2, 2) + GARCH(3, 2) + StudentT(),
            _joint_returns(),
        ),
        (ARMA(0, 0) + GARCH(1, 1) + Normal(), _returns(seed=779, n=900)),
    ),
    ids=("garch-runtime-fallback", "joint-runtime-fallback", "zero-mean-orders"),
)
def test_native_runtime_orders_beyond_precompiled_cases(
    spec: object, data: np.ndarray
) -> None:
    result = spec.fit(data, solver="native", maxiter=1000)

    assert result.converged, result.convergence_message
    assert result.fit_info.objective_parity_error <= 1e-7


@pytest.mark.parametrize("density_factory", (Normal, StudentT, GeneralizedHyperbolic))
def test_legacy_joint_results_report_the_complete_likelihood(
    density_factory: DensityFactory,
) -> None:
    data = _returns(seed=400, n=450)
    spec = ARMA(1, 1) + GARCH(1, 1) + density_factory()
    result = spec.fit(data, log_mode=True)
    filtered = filter_spec(spec, result.data, result.params)

    assert result.log_likelihood == pytest.approx(
        filtered.log_likelihood, rel=2e-11, abs=2e-9
    )
    assert abs(result.log_likelihood) > 100.0


def test_native_solver_retains_a_legacy_variance_gamma_optimum() -> None:
    data = _returns()
    legacy = (ARMA(1, 1) + GARCH(2, 1) + VarianceGamma()).fit(
        data, log_mode=True
    )
    native = (ARMA(1, 1) + GARCH(2, 1) + VarianceGamma()).fit(
        data,
        solver="native",
        initial_params=legacy.params,
        maxiter=700,
    )
    legacy_at_common_sample = filter_spec(
        legacy.spec, native.data, legacy.params
    ).log_likelihood

    assert native.converged, native.convergence_message
    assert native.log_likelihood >= legacy_at_common_sample - 1e-8


def test_native_failure_is_quiet_and_persistent(capsys: pytest.CaptureFixture[str]) -> None:
    result = (ARMA(1, 1) + GARCH(2, 1) + SkewT()).fit(
        _returns(n=450), solver="native", maxiter=1
    )
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert not result.converged
    assert result.fit_info.warnings
    assert "Strict credibility check failed" in result.convergence_message


def test_native_rejects_an_open_boundary_likelihood() -> None:
    generator = np.random.default_rng(20260827)
    independent = np.ascontiguousarray(
        generator.standard_t(7, size=700) * 0.035,
        dtype=np.float64,
    )
    result = (GARCH(2, 1) + Normal()).fit(
        independent, solver="native", maxiter=700
    )

    assert not result.converged
    assert result.fit_info.boundary_reached
    assert any("open model boundary" in warning for warning in result.fit_info.warnings)


def test_native_fits_are_reentrant_across_threads() -> None:
    datasets = [
        _joint_returns(
            seed=900 + seed,
            n=700,
            p_ar=1,
            q_ma=1,
            p_vol=1,
            q_vol=1,
        )
        for seed in range(4)
    ]

    def fit(data: np.ndarray) -> np.ndarray:
        result = (ARMA(1, 1) + GARCH(1, 1) + StudentT()).fit(
            data, solver="native", maxiter=700
        )
        assert result.converged, result.convergence_message
        return result.params

    sequential = [fit(data) for data in datasets]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(fit, datasets))

    for expected, actual in zip(sequential, concurrent):
        assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_existing_default_engine_is_unchanged() -> None:
    result = (GARCH(1, 1) + Normal()).fit(_returns(n=400))

    assert result.fit_info.engine is None
    assert result.fit_info.solver != "native-slsqp"
