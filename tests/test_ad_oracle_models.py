from __future__ import annotations

import importlib.util
from typing import Any
import warnings

import numpy as np
import pytest

from scivol import ARMA, GARCH, GJRGARCH, Normal, SkewT, StudentT, _core
from scivol._devtools.ad_oracle import (
    arma_gjr_garch_logspace_value_grad_hess,
    arma_gjr_garch_value_grad_hess,
    arma_garch_logspace_value_grad_hess,
    arma_garch_value_grad_hess,
    arma_normal_logspace_value_grad_hess,
    arma_normal_value_grad_hess,
    garch_logspace_value_grad_hess,
    garch_value_grad_hess,
    gjr_garch_logspace_value_grad_hess,
    gjr_garch_value_grad_hess,
)
from scivol._kernels.transforms import (
    jacobian_garch,
    jacobian_gjr_garch,
    jacobian_arma_garch_normal,
    jacobian_arma_gjr_garch_normal,
    jacobian_arma_gjr_garch_skewt,
    jacobian_arma_gjr_garch_studentt,
    jacobian_arma_garch_skewt,
    jacobian_arma_garch_studentt,
    log_hessian_arma_garch_normal,
    log_hessian_arma_gjr_garch_normal,
    log_hessian_arma_gjr_garch_skewt,
    log_hessian_arma_gjr_garch_studentt,
    log_hessian_arma_garch_skewt,
    log_hessian_arma_garch_studentt,
    log_hessian_arma_normal,
    log_hessian_garch,
    log_hessian_gjr_garch,
    pack_garch,
    pack_gjr_garch,
    second_derivatives_garch,
    second_derivatives_gjr_garch,
    unpack_arma_normal,
    unpack_arma_garch_normal,
    unpack_arma_gjr_garch_normal,
    unpack_arma_gjr_garch_skewt,
    unpack_arma_gjr_garch_studentt,
    unpack_arma_garch_skewt,
    unpack_arma_garch_studentt,
    unpack_garch,
    unpack_garch_skewt,
    unpack_garch_studentt,
    unpack_gjr_garch,
    unpack_gjr_garch_skewt,
    unpack_gjr_garch_studentt,
)


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle model checks", allow_module_level=True)


def _as_cptr(arr: np.ndarray) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


OMEGA_Z_CASES = (-25.0, 0.5)
GJR_THETA_Z_DIAGNOSTIC_CASES = (
    pytest.param(
        0.0032521979978232802,
        0.00875595033429564,
        0.00874570778187854,
        0.023377366177619045,
        1875,
        20260506,
        id="low-persistence",
    ),
    pytest.param(
        1.981199923829673e-05,
        0.18799708069672755,
        9.950000149250003e-09,
        0.8070029143282723,
        3000,
        20260507,
        id="near-unit-persistence",
    ),
    pytest.param(
        2.9097371911520058e-06,
        0.03827824785487638,
        0.03802240370777959,
        0.9236993484373406,
        3000,
        20260508,
        id="z-offender-22517",
    ),
)


def _softplus_theta_from_z(z_omega: float) -> float:
    if z_omega > 20.0:
        return float(z_omega)
    return float(np.log1p(np.exp(z_omega)))


def _simulate_gjr_normal_series(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    n: int,
    seed: int,
    burn_in: int = 500,
) -> np.ndarray:
    persistence = alpha + 0.5 * gamma + beta
    if persistence >= 1.0:
        raise ValueError(f"Scenario is nonstationary: persistence={persistence}")

    rng = np.random.default_rng(seed)
    total_n = n + burn_in
    y = np.zeros(total_n, dtype=np.float64)
    sigma2 = np.zeros(total_n, dtype=np.float64)

    sigma2[0] = omega / (1.0 - persistence)
    y[0] = np.sqrt(sigma2[0]) * rng.standard_normal()

    for t in range(1, total_n):
        e_prev = y[t - 1]
        ind = 1.0 if e_prev < 0.0 else 0.0
        sigma2[t] = omega + alpha * e_prev * e_prev + gamma * ind * e_prev * e_prev + beta * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], 1e-12)
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

    return y[burn_in:]


def _fit_gjr11_normal_debug(
    series: np.ndarray,
    *,
    log_mode: bool,
    debug_capture: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    capture = {} if debug_capture is None else debug_capture
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (GJRGARCH(1, 1) + Normal()).fit(
            series,
            solver="slsqp",
            log_mode=log_mode,
            verbose=False,
            n_jobs=1,
            debug_capture=capture,
            **kwargs,
        )
    return result, capture


@pytest.fixture(scope="module")
def arma_series() -> np.ndarray:
    rng = np.random.default_rng(2026)
    y = np.zeros(320, dtype=np.float64)
    c, phi1, phi2, theta1 = 0.01, 0.25, -0.10, 0.20
    eps_prev = 0.0
    for t in range(2, len(y)):
        shock = rng.normal(scale=0.1)
        mu_t = c + phi1 * y[t - 1] + phi2 * y[t - 2] + theta1 * eps_prev
        y[t] = mu_t + shock
        eps_prev = shock
    return y


@pytest.fixture(scope="module")
def arma_garch_series() -> np.ndarray:
    rng = np.random.default_rng(2027)
    y = np.zeros(360, dtype=np.float64)
    sigma2 = np.zeros_like(y)
    eps = np.zeros_like(y)
    c, phi, theta = 0.0, 0.20, -0.15
    omega, alpha, beta = 2e-6, 0.06, 0.92
    sigma2[0] = omega / (1.0 - alpha - beta)
    for t in range(1, len(y)):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        y[t] = c + phi * y[t - 1] + theta * eps[t - 1] + eps[t]
    return y


@pytest.fixture(scope="module")
def arma_gjr_garch_series() -> np.ndarray:
    rng = np.random.default_rng(2029)
    y = np.zeros(360, dtype=np.float64)
    sigma2 = np.zeros_like(y)
    eps = np.zeros_like(y)
    c, phi, theta = 0.0, 0.18, -0.12
    omega, alpha, gamma, beta = 2e-6, 0.04, 0.05, 0.88
    sigma2[0] = omega / (1.0 - alpha - 0.5 * gamma - beta)
    for t in range(1, len(y)):
        ind = 1.0 if eps[t - 1] < 0.0 else 0.0
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + gamma * ind * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        y[t] = c + phi * y[t - 1] + theta * eps[t - 1] + eps[t]
    return y


@pytest.fixture(scope="module")
def garch_skewt_series() -> np.ndarray:
    rng = np.random.default_rng(2028)
    y = np.zeros(420, dtype=np.float64)
    sigma2 = np.zeros_like(y)
    omega, alpha, beta = 1.5e-6, 0.05, 0.92
    nu, lam = 8.0, -0.25
    sigma2[0] = omega / (1.0 - alpha - beta)
    t_scale = np.sqrt((nu - 2.0) / nu)

    for t in range(len(y)):
        if t > 0:
            sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
        z_raw = rng.standard_t(nu) * t_scale
        z = z_raw * (1.0 - lam) if z_raw < 0.0 else z_raw * (1.0 + lam)
        y[t] = np.sqrt(sigma2[t]) * z

    return y


class TestOmegaTransformRegression:
    @pytest.mark.parametrize("z_omega", OMEGA_Z_CASES)
    def test_garch_softplus_transform_matches_closed_form_and_c(self, z_omega: float) -> None:
        z = np.array([z_omega, -1.2, -0.4, 0.25, -0.35], dtype=np.float64)
        theta_py = pack_garch(z, 2, 2)
        theta_c = np.zeros_like(theta_py)
        _core._pack_garch_pq(_as_cptr(z), _as_cptr(theta_c), 2, 2)

        J_py = jacobian_garch(theta_py, 2, 2)
        J_c = np.zeros((z.size, z.size), dtype=np.float64)
        _core._jacobian_garch_pq(_as_cptr(theta_py), _as_cptr(J_c), 2, 2)

        second = second_derivatives_garch(theta_py, 2, 2, dist="normal")
        expected_omega = _softplus_theta_from_z(z_omega)
        expected_first = -np.expm1(-expected_omega)
        expected_second = expected_first * (1.0 - expected_first)

        np.testing.assert_allclose(theta_py[0], expected_omega, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(theta_c, theta_py, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(J_c, J_py, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(J_py[0, 0], expected_first, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(second[0, 0, 0], expected_second, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("z_omega", OMEGA_Z_CASES)
    def test_gjr_softplus_transform_matches_closed_form_and_c(self, z_omega: float) -> None:
        z = np.array([z_omega, -1.0, -0.6, -0.2, 0.3, -0.4, 0.1], dtype=np.float64)
        theta_py = pack_gjr_garch(z, 2, 2)
        theta_c = np.zeros_like(theta_py)
        _core._pack_gjr_garch_pq(_as_cptr(z), _as_cptr(theta_c), 2, 2)

        J_py = jacobian_gjr_garch(theta_py, 2, 2)
        J_c = np.zeros((z.size, z.size), dtype=np.float64)
        _core._jacobian_gjr_garch_pq(_as_cptr(theta_py), _as_cptr(J_c), 2, 2)

        second = second_derivatives_gjr_garch(theta_py, 2, 2, dist="normal")
        expected_omega = _softplus_theta_from_z(z_omega)
        expected_first = -np.expm1(-expected_omega)
        expected_second = expected_first * (1.0 - expected_first)

        np.testing.assert_allclose(theta_py[0], expected_omega, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(theta_c, theta_py, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(J_c, J_py, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(J_py[0, 0], expected_first, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(second[0, 0, 0], expected_second, rtol=1e-12, atol=1e-15)


class TestARMAOracle:
    def test_arma11_normal_matches_c_nll_and_grad(self, arma_series: np.ndarray) -> None:
        params = np.array([0.01, 0.25, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_series)
        grad = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_normal_value_grad_hess(params, arma_series, 1, 1)
        value_c = _core._arma_nll_11_normal(_as_cptr(params), _as_cptr(arma_series), _as_cptr(resid), arma_series.size)
        _core._arma_nll_grad_11_normal(
            _as_cptr(params), _as_cptr(arma_series), _as_cptr(resid), _as_cptr(grad), arma_series.size
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-8, atol=1e-10)
        assert np.all(np.isfinite(hess_ad))

    def test_arma22_normal_matches_pq_nll_and_grad(self, arma_series: np.ndarray) -> None:
        params = np.array([0.01, 0.25, -0.10, 0.20, -0.05], dtype=np.float64)
        resid = np.zeros_like(arma_series)
        e0 = np.zeros(2, dtype=np.float64)
        grad = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_normal_value_grad_hess(params, arma_series, 2, 2)
        value_c = _core._arma_nll_pq_normal(
            _as_cptr(params), _as_cptr(arma_series), _as_cptr(resid), _as_cptr(e0), arma_series.size, 2, 2
        )
        _core._arma_nll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_series),
            _as_cptr(resid),
            _as_cptr(e0),
            _as_cptr(grad),
            arma_series.size,
            2,
            2,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-8, atol=1e-10)
        assert np.all(np.isfinite(hess_ad))

    def test_arma22_normal_theta_hessian_matches_ad(self, arma_series: np.ndarray) -> None:
        params = np.array([0.01, 0.25, -0.10, 0.20, -0.05], dtype=np.float64)
        resid = np.zeros_like(arma_series)
        e0 = np.zeros(2, dtype=np.float64)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._arma_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_series),
            _as_cptr(resid),
            _as_cptr(e0),
            _as_cptr(hess_c),
            arma_series.size,
            2,
            2,
        )
        _, _, hess_ad = arma_normal_value_grad_hess(params, arma_series, 2, 2)

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-7, atol=1e-8)

    def test_arma22_normal_log_matches_c_grad_and_hess(self, arma_series: np.ndarray) -> None:
        params = np.array([0.01, 0.25, -0.10, 0.20, -0.05], dtype=np.float64)
        z = unpack_arma_normal(params, 2, 2)
        resid = np.zeros_like(arma_series)
        e0 = np.zeros(2, dtype=np.float64)
        grad_z = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_normal_logspace_value_grad_hess(z, arma_series, 2, 2)
        value_c = _core._log_arma_nll_pq_normal(
            _as_cptr(z), _as_cptr(arma_series), _as_cptr(resid), _as_cptr(e0), arma_series.size, 2, 2
        )
        _core._log_arma_nll_grad_pq_normal(
            _as_cptr(z),
            _as_cptr(arma_series),
            _as_cptr(resid),
            _as_cptr(e0),
            _as_cptr(grad_z),
            arma_series.size,
            2,
            2,
        )
        grad_theta = np.zeros_like(params)
        _core._arma_nll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_series),
            _as_cptr(resid),
            _as_cptr(e0),
            _as_cptr(grad_theta),
            arma_series.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._arma_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_series),
            _as_cptr(resid),
            _as_cptr(e0),
            _as_cptr(hess_theta),
            arma_series.size,
            2,
            2,
        )
        hess_z = log_hessian_arma_normal(params, grad_theta, hess_theta, 2, 2)

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-7, atol=1e-8)

    def test_arma22_normal_fit_runs_in_log_mode(self, arma_series: np.ndarray) -> None:
        spec = ARMA(2, 2) + Normal()
        result = spec.fit(arma_series, solver="slsqp", log_mode=True, verbose=False)
        assert result.params is not None
        assert len(result.params) == 5
        assert np.all(np.isfinite(result.params))

    def test_arma22_normal_fit_runs_in_log_mode_with_trust(self, arma_series: np.ndarray) -> None:
        spec = ARMA(2, 2) + Normal()
        result = spec.fit(arma_series, solver="trust", log_mode=True, verbose=False)
        assert result.params is not None
        assert len(result.params) == 5
        assert np.all(np.isfinite(result.params))


class TestARMAGARCHOracle:
    def test_arma_garch11_normal_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        grad = np.zeros_like(params)
        h0 = float(np.mean(arma_garch_series ** 2))

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "normal")
        value_c = _core._arma_garch_nll_11_normal(
            _as_cptr(params), _as_cptr(arma_garch_series), _as_cptr(resid), _as_cptr(sigma2), h0, arma_garch_series.size
        )
        _core._arma_garch_nll_grad_11_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad),
            h0,
            arma_garch_series.size,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-6, atol=1e-8)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch11_normal_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        hess = np.zeros((params.size, params.size), dtype=np.float64)
        h0 = float(np.mean(arma_garch_series ** 2))

        _core._arma_garch_hess_11_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess),
            h0,
            arma_garch_series.size,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "normal")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch11_studentt_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92, 8.0], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        grad = np.zeros_like(params)
        h0 = float(np.mean(arma_garch_series ** 2))

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "studentt")
        value_c = _core._arma_garch_nll_11_studentt(
            _as_cptr(params), _as_cptr(arma_garch_series), _as_cptr(resid), _as_cptr(sigma2), h0, arma_garch_series.size
        )
        _core._arma_garch_nll_grad_11_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad),
            h0,
            arma_garch_series.size,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-6, atol=1e-8)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch11_studentt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92, 8.0], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        hess = np.zeros((params.size, params.size), dtype=np.float64)
        h0 = float(np.mean(arma_garch_series ** 2))

        _core._arma_garch_hess_11_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess),
            h0,
            arma_garch_series.size,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "studentt")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_normal_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "normal")
        value_c = _core._arma_garch_nll_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        value_grad_c = _core._arma_garch_nll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-6, atol=1e-8)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch_pq_normal_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        hess = np.zeros((params.size, params.size), dtype=np.float64)

        _core._arma_garch_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "normal")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_studentt_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "studentt")
        value_c = _core._arma_garch_nll_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        value_grad_c = _core._arma_garch_nll_grad_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-6, atol=1e-8)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch_pq_studentt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        hess = np.zeros((params.size, params.size), dtype=np.float64)

        _core._arma_garch_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "studentt")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch11_skewt_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92, 8.0, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        grad = np.zeros_like(params)
        h0 = float(np.mean(arma_garch_series ** 2))

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "skewt")
        value_c = _core._arma_garch_nll_11_skewt(
            _as_cptr(params), _as_cptr(arma_garch_series), _as_cptr(resid), _as_cptr(sigma2), h0, arma_garch_series.size
        )
        value_grad_c = _core._arma_garch_nll_grad_11_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad),
            h0,
            arma_garch_series.size,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-5, atol=1e-7)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch11_skewt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92, 8.0, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        hess = np.zeros((params.size, params.size), dtype=np.float64)
        h0 = float(np.mean(arma_garch_series ** 2))

        _core._arma_garch_hess_11_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess),
            h0,
            arma_garch_series.size,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "skewt")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-5, atol=1e-7)

    def test_arma_garch_pq_skewt_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "skewt")
        value_c = _core._arma_garch_nll_pq_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        value_grad_c = _core._arma_garch_nll_grad_pq_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-5, atol=1e-7)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch_pq_skewt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        hess = np.zeros((params.size, params.size), dtype=np.float64)

        _core._arma_garch_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "skewt")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-5, atol=1e-7)

    def test_arma_garch_pq_normal_log_gradient_matches_chain_rule(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad_z = np.zeros_like(params)
        z = unpack_arma_garch_normal(params, 1, 1, 2, 2)

        _, grad_theta, _ = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "normal")
        grad_z_ref = jacobian_arma_garch_normal(params, 1, 1, 2, 2).T @ grad_theta

        _core._log_arma_garch_nll_grad_pq_normal(
            _as_cptr(z),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_z),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        np.testing.assert_allclose(grad_z, grad_z_ref, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_normal_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        z = unpack_arma_garch_normal(params, 1, 1, 2, 2)

        _, _, hess_ad = arma_garch_logspace_value_grad_hess(z, arma_garch_series, 1, 1, 2, 2, "normal")
        grad_theta = np.zeros_like(params)
        _core._arma_garch_nll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._arma_garch_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_z = log_hessian_arma_garch_normal(params, grad_theta, hess_theta, 1, 1, 2, 2)

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("omega_z", OMEGA_Z_CASES)
    def test_arma_garch_pq_normal_log_parity_holds_across_omega_scales(
        self,
        arma_garch_series: np.ndarray,
        omega_z: float,
    ) -> None:
        omega = _softplus_theta_from_z(omega_z)
        params = np.array([0.0, 0.15, -0.10, omega, 0.04, 0.02, 0.70, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad_z = np.zeros_like(params)
        z = unpack_arma_garch_normal(params, 1, 1, 2, 2)

        value_ad, grad_ad, hess_ad = arma_garch_logspace_value_grad_hess(
            z, arma_garch_series, 1, 1, 2, 2, "normal"
        )
        value_c = _core._log_arma_garch_nll_pq_normal(
            _as_cptr(z),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        _core._log_arma_garch_nll_grad_pq_normal(
            _as_cptr(z),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_z),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )

        grad_theta = np.zeros_like(params)
        _core._arma_garch_nll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._arma_garch_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_z = log_hessian_arma_garch_normal(params, grad_theta, hess_theta, 1, 1, 2, 2)

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_studentt_log_gradient_matches_chain_rule(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad_z = np.zeros_like(params)
        z = unpack_arma_garch_studentt(params, 1, 1, 2, 2)

        _, grad_theta, _ = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "studentt")
        grad_z_ref = jacobian_arma_garch_studentt(params, 1, 1, 2, 2).T @ grad_theta

        _core._log_arma_garch_nll_grad_pq_studentt(
            _as_cptr(z),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_z),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        np.testing.assert_allclose(grad_z, grad_z_ref, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_studentt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        z = unpack_arma_garch_studentt(params, 1, 1, 2, 2)

        _, _, hess_ad = arma_garch_logspace_value_grad_hess(z, arma_garch_series, 1, 1, 2, 2, "studentt")
        grad_theta = np.zeros_like(params)
        _core._arma_garch_nll_grad_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._arma_garch_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_z = log_hessian_arma_garch_studentt(params, grad_theta, hess_theta, 1, 1, 2, 2)

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_skewt_log_gradient_matches_chain_rule(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad_z = np.zeros_like(params)
        z = unpack_arma_garch_skewt(params, 1, 1, 2, 2)
        _, grad_theta, _ = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "skewt")
        grad_z_ref = jacobian_arma_garch_skewt(params, 1, 1, 2, 2).T @ grad_theta
        _, grad_z_ad, _ = arma_garch_logspace_value_grad_hess(z, arma_garch_series, 1, 1, 2, 2, "skewt")
        _core._log_arma_garch_nll_grad_pq_skewt(
            _as_cptr(z),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_z),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )

        np.testing.assert_allclose(grad_z, grad_z_ref, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(grad_z_ref, grad_z_ad, rtol=1e-5, atol=1e-7)

    def test_arma_garch_pq_skewt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        z = unpack_arma_garch_skewt(params, 1, 1, 2, 2)

        _, _, hess_ad = arma_garch_logspace_value_grad_hess(z, arma_garch_series, 1, 1, 2, 2, "skewt")
        grad_theta = np.zeros_like(params)
        _core._arma_garch_nll_grad_pq_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._arma_garch_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_z = log_hessian_arma_garch_skewt(params, grad_theta, hess_theta, 1, 1, 2, 2)

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-5, atol=1e-7)

    def test_generic_arma_garch_normal_fit_runs_in_log_mode(self, arma_garch_series: np.ndarray) -> None:
        from scivol import ARMA, GARCH, Normal

        spec = ARMA(1, 1) + GARCH(2, 2) + Normal()
        result = spec.fit(arma_garch_series, solver="slsqp", log_mode=True)

        assert result.params is not None
        assert len(result.params) == 8
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_generic_arma_garch_normal_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        from scivol import ARMA, GARCH, Normal

        spec = ARMA(1, 1) + GARCH(2, 2) + Normal()
        result = spec.fit(arma_garch_series, solver="trust", log_mode=True)

        assert result.params is not None
        assert len(result.params) == 8
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_generic_arma_garch_studentt_fit_runs_in_constrained_mode(self, arma_garch_series: np.ndarray) -> None:
        from scivol import ARMA, GARCH, StudentT

        spec = ARMA(1, 1) + GARCH(2, 2) + StudentT()
        result = spec.fit(arma_garch_series, solver="slsqp", log_mode=False)

        assert result.params is not None
        assert len(result.params) == 9
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_generic_arma_garch_studentt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        from scivol import ARMA, GARCH, StudentT

        spec = ARMA(1, 1) + GARCH(2, 2) + StudentT()
        result = spec.fit(arma_garch_series, solver="trust", log_mode=True)

        assert result.params is not None
        assert len(result.params) == 9
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_generic_arma_garch_skewt_fit_runs_in_log_mode(self, arma_garch_series: np.ndarray) -> None:
        from scivol import ARMA, GARCH, SkewT

        spec = ARMA(1, 1) + GARCH(2, 2) + SkewT()
        result = spec.fit(arma_garch_series, solver="slsqp", log_mode=True)

        assert result.params is not None
        assert len(result.params) == 10
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_generic_arma_garch_skewt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        from scivol import ARMA, GARCH, SkewT

        spec = ARMA(1, 1) + GARCH(2, 2) + SkewT()
        result = spec.fit(arma_garch_series, solver="trust", log_mode=True)

        assert result.params is not None
        assert len(result.params) == 10
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0


class TestARMAGJRGARCHOracle:
    _DIST_CASES = {
        "normal": {
            "params_11": np.array([0.0, 0.18, -0.12, 2e-6, 0.04, 0.05, 0.88], dtype=np.float64),
            "params_pq": np.array([0.0, 0.12, -0.08, 2e-6, 0.04, 0.02, 0.03, 0.01, 0.65, 0.15], dtype=np.float64),
            "unpack": unpack_arma_gjr_garch_normal,
            "jacobian": jacobian_arma_gjr_garch_normal,
            "log_hessian": log_hessian_arma_gjr_garch_normal,
            "density_cls": Normal,
        },
        "studentt": {
            "params_11": np.array([0.0, 0.18, -0.12, 2e-6, 0.04, 0.05, 0.88, 8.0], dtype=np.float64),
            "params_pq": np.array([0.0, 0.12, -0.08, 2e-6, 0.04, 0.02, 0.03, 0.01, 0.65, 0.15, 8.5], dtype=np.float64),
            "unpack": unpack_arma_gjr_garch_studentt,
            "jacobian": jacobian_arma_gjr_garch_studentt,
            "log_hessian": log_hessian_arma_gjr_garch_studentt,
            "density_cls": StudentT,
        },
        "skewt": {
            "params_11": np.array([0.0, 0.18, -0.12, 2e-6, 0.04, 0.05, 0.88, 8.0, -0.2], dtype=np.float64),
            "params_pq": np.array([0.0, 0.12, -0.08, 2e-6, 0.04, 0.02, 0.03, 0.01, 0.65, 0.15, 8.5, -0.2], dtype=np.float64),
            "unpack": unpack_arma_gjr_garch_skewt,
            "jacobian": jacobian_arma_gjr_garch_skewt,
            "log_hessian": log_hessian_arma_gjr_garch_skewt,
            "density_cls": SkewT,
        },
    }

    @pytest.mark.parametrize("dist", ["normal", "studentt", "skewt"])
    def test_arma_gjr_garch11_matches_c_nll_grad_and_hessian(
        self,
        arma_gjr_garch_series: np.ndarray,
        dist: str,
    ) -> None:
        params = self._DIST_CASES[dist]["params_11"]
        resid = np.zeros_like(arma_gjr_garch_series)
        sigma2 = np.zeros_like(arma_gjr_garch_series)
        grad = np.zeros_like(params)
        hess = np.zeros((params.size, params.size), dtype=np.float64)
        h0 = float(np.mean(arma_gjr_garch_series ** 2))

        value_ad, grad_ad, hess_ad = arma_gjr_garch_value_grad_hess(
            params, arma_gjr_garch_series, 1, 1, 1, 1, dist
        )
        value_c = getattr(_core, f"_arma_gjr_garch_nll_11_{dist}")(
            _as_cptr(params),
            _as_cptr(arma_gjr_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            h0,
            arma_gjr_garch_series.size,
        )
        value_grad_c = getattr(_core, f"_arma_gjr_garch_nll_grad_11_{dist}")(
            _as_cptr(params),
            _as_cptr(arma_gjr_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad),
            h0,
            arma_gjr_garch_series.size,
        )
        getattr(_core, f"_arma_gjr_garch_hess_11_{dist}")(
            _as_cptr(params),
            _as_cptr(arma_gjr_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess),
            h0,
            arma_gjr_garch_series.size,
        )

        rtol = 1e-4 if dist == "skewt" else 1e-6
        atol = 1e-6 if dist == "skewt" else 1e-8
        np.testing.assert_allclose(value_c, value_ad, rtol=rtol, atol=1e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=rtol, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=rtol, atol=atol)
        np.testing.assert_allclose(hess, hess_ad, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dist", ["normal", "studentt", "skewt"])
    def test_arma_gjr_garch_pq_matches_c_nll_grad_and_hessian(
        self,
        arma_gjr_garch_series: np.ndarray,
        dist: str,
    ) -> None:
        params = self._DIST_CASES[dist]["params_pq"]
        resid = np.zeros_like(arma_gjr_garch_series)
        sigma2 = np.zeros_like(arma_gjr_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_gjr_garch_series ** 2), dtype=np.float64)
        grad = np.zeros_like(params)
        hess = np.zeros((params.size, params.size), dtype=np.float64)

        value_ad, grad_ad, hess_ad = arma_gjr_garch_value_grad_hess(
            params, arma_gjr_garch_series, 1, 1, 2, 2, dist
        )
        value_c = getattr(_core, f"_arma_gjr_garch_nll_pq_{dist}")(
            _as_cptr(params),
            _as_cptr(arma_gjr_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            arma_gjr_garch_series.size,
            1,
            1,
            2,
            2,
        )
        value_grad_c = getattr(_core, f"_arma_gjr_garch_nll_grad_pq_{dist}")(
            _as_cptr(params),
            _as_cptr(arma_gjr_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad),
            arma_gjr_garch_series.size,
            1,
            1,
            2,
            2,
        )
        getattr(_core, f"_arma_gjr_garch_hess_pq_{dist}")(
            _as_cptr(params),
            _as_cptr(arma_gjr_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess),
            arma_gjr_garch_series.size,
            1,
            1,
            2,
            2,
        )

        rtol = 1e-4 if dist == "skewt" else 1e-6
        atol = 1e-6 if dist == "skewt" else 1e-8
        np.testing.assert_allclose(value_c, value_ad, rtol=rtol, atol=1e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=rtol, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=rtol, atol=atol)
        np.testing.assert_allclose(hess, hess_ad, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dist", ["normal", "studentt", "skewt"])
    def test_arma_gjr_garch_pq_log_gradient_and_hessian_match_ad(
        self,
        arma_gjr_garch_series: np.ndarray,
        dist: str,
    ) -> None:
        case = self._DIST_CASES[dist]
        params = case["params_pq"]
        resid = np.zeros_like(arma_gjr_garch_series)
        sigma2 = np.zeros_like(arma_gjr_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_gjr_garch_series ** 2), dtype=np.float64)
        grad_theta = np.zeros_like(params)
        grad_z = np.zeros_like(params)
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        z = case["unpack"](params, 1, 1, 2, 2)

        _, grad_z_ad, hess_z_ad = arma_gjr_garch_logspace_value_grad_hess(
            z, arma_gjr_garch_series, 1, 1, 2, 2, dist
        )
        getattr(_core, f"_arma_gjr_garch_nll_grad_pq_{dist}")(
            _as_cptr(params),
            _as_cptr(arma_gjr_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_theta),
            arma_gjr_garch_series.size,
            1,
            1,
            2,
            2,
        )
        getattr(_core, f"_arma_gjr_garch_hess_pq_{dist}")(
            _as_cptr(params),
            _as_cptr(arma_gjr_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess_theta),
            arma_gjr_garch_series.size,
            1,
            1,
            2,
            2,
        )
        getattr(_core, f"_log_arma_gjr_garch_nll_grad_pq_{dist}")(
            _as_cptr(z),
            _as_cptr(arma_gjr_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_z),
            arma_gjr_garch_series.size,
            1,
            1,
            2,
            2,
        )
        grad_z_ref = case["jacobian"](params, 1, 1, 2, 2).T @ grad_theta
        hess_z = case["log_hessian"](params, grad_theta, hess_theta, 1, 1, 2, 2)

        rtol = 1e-5 if dist == "skewt" else 1e-6
        atol = 1e-7 if dist == "skewt" else 1e-8
        np.testing.assert_allclose(grad_z, grad_z_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(grad_z, grad_z_ad, rtol=rtol, atol=atol)
        np.testing.assert_allclose(hess_z, hess_z_ad, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        ("dist", "expected_len"),
        [("normal", 10), ("studentt", 11), ("skewt", 12)],
    )
    def test_generic_arma_gjr_garch_fit_runs_in_log_mode_with_trust(
        self,
        arma_gjr_garch_series: np.ndarray,
        dist: str,
        expected_len: int,
    ) -> None:
        density_cls = self._DIST_CASES[dist]["density_cls"]
        spec = ARMA(1, 1) + GJRGARCH(2, 2) + density_cls()
        result = spec.fit(arma_gjr_garch_series, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == expected_len
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0


class TestLogspaceHessianOracle:
    def test_garch22_normal_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid2 = arma_garch_series * arma_garch_series
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15], dtype=np.float64)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._garch_ll_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(hess_c),
            resid2.size,
            2,
            2,
        )
        _, _, hess_ad = garch_value_grad_hess(params, resid2, 2, 2, "normal")

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-7, atol=1e-8)

    def test_garch22_studentt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid2 = arma_garch_series * arma_garch_series
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15, 8.5], dtype=np.float64)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._garch_ll_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(hess_c),
            resid2.size,
            2,
            2,
        )
        _, _, hess_ad = garch_value_grad_hess(params, resid2, 2, 2, "studentt")

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-6, atol=1e-6)

    def test_garch22_variance_recursion_matches_manual(self, arma_garch_series: np.ndarray) -> None:
        resid2 = arma_garch_series * arma_garch_series
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15], dtype=np.float64)
        sigma2_c = np.zeros_like(resid2)
        sigma2_c[0] = np.mean(resid2)
        sigma2_py = np.zeros_like(resid2)
        sigma2_py[0] = np.mean(resid2)

        _core._garch_variance_pq(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2_c),
            resid2.size,
            2,
            2,
        )

        for t in range(1, resid2.size):
            sigma2_py[t] = (
                params[0]
                + params[1] * resid2[t - 1]
                + (params[2] * resid2[t - 2] if t >= 2 else 0.0)
                + params[3] * sigma2_py[t - 1]
                + (params[4] * sigma2_py[t - 2] if t >= 2 else 0.0)
            )

        np.testing.assert_allclose(sigma2_c, sigma2_py, rtol=1e-12, atol=1e-12)

    def test_garch22_normal_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15], dtype=np.float64)
        z = unpack_garch(params, 2, 2)

        _, _, hess_ad = garch_logspace_value_grad_hess(z, resid2, 2, 2, "normal")
        _, grad_theta, _ = garch_value_grad_hess(params, resid2, 2, 2, "normal")
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._garch_ll_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid2.size,
            2,
            2,
        )
        hess_z = log_hessian_garch(params, grad_theta, hess_theta, 2, 2, dist="normal")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("omega_z", OMEGA_Z_CASES)
    def test_garch22_normal_log_parity_holds_across_omega_scales(
        self,
        arma_garch_series: np.ndarray,
        omega_z: float,
    ) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        omega = _softplus_theta_from_z(omega_z)
        params = np.array([omega, 0.05, 0.03, 0.70, 0.15], dtype=np.float64)
        z = unpack_garch(params, 2, 2)

        value_ad, grad_ad, hess_ad = garch_logspace_value_grad_hess(z, resid2, 2, 2, "normal")
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        grad_z = np.zeros_like(z)
        value_c = _core._log_garch_ll_pq_normal(
            _as_cptr(z), _as_cptr(resid2), _as_cptr(sigma2), resid2.size, 2, 2
        )
        _core._log_garch_ll_grad_pq_normal(
            _as_cptr(z), _as_cptr(resid2), _as_cptr(sigma2), _as_cptr(grad_z), resid2.size, 2, 2
        )

        grad_theta = np.zeros_like(params)
        _core._garch_ll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(grad_theta),
            resid2.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._garch_ll_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid2.size,
            2,
            2,
        )
        hess_z = log_hessian_garch(params, grad_theta, hess_theta, 2, 2, dist="normal")

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-8)

    def test_garch22_studentt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15, 8.5], dtype=np.float64)
        z = unpack_garch_studentt(params, 2, 2)

        _, _, hess_ad = garch_logspace_value_grad_hess(z, resid2, 2, 2, "studentt")
        grad_theta = np.zeros_like(params)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        _core._garch_ll_grad_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(grad_theta),
            resid2.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._garch_ll_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid2.size,
            2,
            2,
        )
        hess_z = log_hessian_garch(params, grad_theta, hess_theta, 2, 2, dist="studentt")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-6)

    def test_garch22_skewt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15, 8.5, -0.2], dtype=np.float64)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._garch_ll_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_c),
            resid.size,
            2,
            2,
        )
        _, _, hess_ad = garch_value_grad_hess(params, resid2, 2, 2, "skewt", resid=resid)

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-5, atol=1e-6)

    def test_garch22_skewt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15, 8.5, -0.2], dtype=np.float64)
        z = unpack_garch_skewt(params, 2, 2)

        _, _, hess_ad = garch_logspace_value_grad_hess(z, resid2, 2, 2, "skewt", resid=resid)
        grad_theta = np.zeros_like(params)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        _core._garch_ll_grad_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_theta),
            resid.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._garch_ll_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid.size,
            2,
            2,
        )
        hess_z = log_hessian_garch(params, grad_theta, hess_theta, 2, 2, dist="skewt")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-5, atol=1e-6)

    def test_garch22_skewt_fused_log_gradient_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15, 8.5, -0.2], dtype=np.float64)
        z = unpack_garch_skewt(params, 2, 2)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        grad_z = np.zeros_like(z)

        value_c = _core._log_garch_ll_pq_skewt(
            _as_cptr(z),
            _as_cptr(resid),
            _as_cptr(sigma2),
            resid.size,
            2,
            2,
        )
        _core._log_garch_ll_grad_pq_skewt(
            _as_cptr(z),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_z),
            resid.size,
            2,
            2,
        )
        value_ad, grad_ad, _ = garch_logspace_value_grad_hess(z, resid2, 2, 2, "skewt", resid=resid)

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-5, atol=1e-6)

    def test_gjr22_normal_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10], dtype=np.float64)
        z = unpack_gjr_garch(params, 2, 2)

        _, _, hess_ad = gjr_garch_logspace_value_grad_hess(z, resid, 2, 2, "normal")
        _, grad_theta, _ = gjr_garch_value_grad_hess(params, resid, 2, 2, "normal")
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._gjr_garch_ll_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid.size,
            2,
            2,
        )
        hess_z = log_hessian_gjr_garch(params, grad_theta, hess_theta, 2, 2, dist="normal")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("omega_z", OMEGA_Z_CASES)
    def test_gjr22_normal_log_parity_holds_across_omega_scales(
        self,
        arma_garch_series: np.ndarray,
        omega_z: float,
    ) -> None:
        resid = arma_garch_series
        omega = _softplus_theta_from_z(omega_z)
        params = np.array([omega, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10], dtype=np.float64)
        z = unpack_gjr_garch(params, 2, 2)

        value_ad, grad_ad, hess_ad = gjr_garch_logspace_value_grad_hess(z, resid, 2, 2, "normal")
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        grad_z = np.zeros_like(z)
        value_c = _core._log_gjr_garch_ll_pq_normal(
            _as_cptr(z), _as_cptr(resid), _as_cptr(sigma2), resid.size, 2, 2
        )
        _core._log_gjr_garch_ll_grad_pq_normal(
            _as_cptr(z), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(grad_z), resid.size, 2, 2
        )

        grad_theta = np.zeros_like(params)
        _core._gjr_garch_ll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_theta),
            resid.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._gjr_garch_ll_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid.size,
            2,
            2,
        )
        hess_z = log_hessian_gjr_garch(params, grad_theta, hess_theta, 2, 2, dist="normal")

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-8)

    def test_garch22_normal_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GARCH(2, 2) + Normal()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 5
        assert np.all(np.isfinite(result.params))

    def test_garch22_skewt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GARCH(2, 2) + SkewT()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 7
        assert np.all(np.isfinite(result.params))

    def test_gjr22_normal_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GJRGARCH(2, 2) + Normal()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 7
        assert np.all(np.isfinite(result.params))

    @pytest.mark.parametrize(
        ("omega", "alpha", "gamma", "beta", "n_obs", "seed"),
        GJR_THETA_Z_DIAGNOSTIC_CASES,
    )
    def test_gjr11_normal_debug_capture_matches_mapped_start_evaluations(
        self,
        omega: float,
        alpha: float,
        gamma: float,
        beta: float,
        n_obs: int,
        seed: int,
    ) -> None:
        series = _simulate_gjr_normal_series(omega, alpha, gamma, beta, n=n_obs, seed=seed)
        _, capture = _fit_gjr11_normal_debug(series, log_mode=False)

        theta_start = np.asarray(capture["theta_start"], dtype=np.float64)
        theta_from_z_start = np.asarray(capture["theta_from_z_start"], dtype=np.float64)
        z_theta = np.asarray(capture["z_start_eval"]["theta"], dtype=np.float64)
        theta_eval = capture["theta_from_z_start_eval"]
        z_eval = capture["z_start_eval"]

        np.testing.assert_allclose(theta_from_z_start, z_theta, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            capture["theta_start_l1_diff"],
            float(np.abs(theta_start - theta_from_z_start).sum()),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            capture["theta_start_max_abs_diff"],
            float(np.max(np.abs(theta_start - theta_from_z_start))),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            theta_eval["objective_per_obs"],
            z_eval["theta_objective_per_obs"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(theta_eval["gradient_per_obs"], dtype=np.float64),
            np.asarray(z_eval["theta_gradient_per_obs"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(theta_eval["hessian_per_obs"], dtype=np.float64),
            np.asarray(z_eval["theta_hessian_per_obs"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        ("omega", "alpha", "gamma", "beta", "n_obs", "seed"),
        GJR_THETA_Z_DIAGNOSTIC_CASES,
    )
    def test_gjr11_normal_debug_capture_reports_zero_gap_for_explicitly_matched_starts(
        self,
        omega: float,
        alpha: float,
        gamma: float,
        beta: float,
        n_obs: int,
        seed: int,
    ) -> None:
        series = _simulate_gjr_normal_series(omega, alpha, gamma, beta, n=n_obs, seed=seed)
        _, baseline_capture = _fit_gjr11_normal_debug(series, log_mode=False)

        theta_start = np.asarray(baseline_capture["theta_start"], dtype=np.float64)
        matched_z_start = unpack_gjr_garch(theta_start, 1, 1)

        _, matched_capture = _fit_gjr11_normal_debug(
            series,
            log_mode=True,
            debug_theta_start=theta_start,
            debug_z_start=matched_z_start,
        )

        np.testing.assert_allclose(
            matched_capture["runtime_theta_start_l1_diff"],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            matched_capture["runtime_theta_start_max_abs_diff"],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(matched_capture["runtime_theta_start"], dtype=np.float64),
            np.asarray(matched_capture["runtime_theta_from_z_start"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )

    def test_garch22_studentt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GARCH(2, 2) + StudentT()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 6
        assert np.all(np.isfinite(result.params))

    def test_gjr22_studentt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10, 8.5], dtype=np.float64)
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._gjr_garch_ll_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_c),
            resid.size,
            2,
            2,
        )
        _, _, hess_ad = gjr_garch_value_grad_hess(params, resid, 2, 2, "studentt")

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-6, atol=1e-6)

    def test_gjr22_studentt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10, 8.5], dtype=np.float64)
        z = unpack_gjr_garch_studentt(params, 2, 2)

        _, _, hess_ad = gjr_garch_logspace_value_grad_hess(z, resid, 2, 2, "studentt")
        grad_theta = np.zeros_like(params)
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        _core._gjr_garch_ll_grad_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_theta),
            resid.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._gjr_garch_ll_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid.size,
            2,
            2,
        )
        hess_z = log_hessian_gjr_garch(params, grad_theta, hess_theta, 2, 2, dist="studentt")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-6)

    def test_gjr22_studentt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GJRGARCH(2, 2) + StudentT()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 8
        assert np.all(np.isfinite(result.params))

    def test_gjr22_skewt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10, 8.5, -0.2], dtype=np.float64)
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._gjr_garch_ll_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_c),
            resid.size,
            2,
            2,
        )
        _, _, hess_ad = gjr_garch_value_grad_hess(params, resid, 2, 2, "skewt")

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-5, atol=1e-6)

    def test_gjr22_skewt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10, 8.5, -0.2], dtype=np.float64)
        z = unpack_gjr_garch_skewt(params, 2, 2)

        _, _, hess_ad = gjr_garch_logspace_value_grad_hess(z, resid, 2, 2, "skewt")
        grad_theta = np.zeros_like(params)
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        _core._gjr_garch_ll_grad_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_theta),
            resid.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._gjr_garch_ll_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid.size,
            2,
            2,
        )
        hess_z = log_hessian_gjr_garch(params, grad_theta, hess_theta, 2, 2, dist="skewt")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-5, atol=1e-6)

    def test_gjr22_skewt_fused_log_gradient_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10, 8.5, -0.2], dtype=np.float64)
        z = unpack_gjr_garch_skewt(params, 2, 2)
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        grad_z = np.zeros_like(z)

        value_c = _core._log_gjr_garch_ll_pq_skewt(
            _as_cptr(z),
            _as_cptr(resid),
            _as_cptr(sigma2),
            resid.size,
            2,
            2,
        )
        _core._log_gjr_garch_ll_grad_pq_skewt(
            _as_cptr(z),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_z),
            resid.size,
            2,
            2,
        )
        value_ad, grad_ad, _ = gjr_garch_logspace_value_grad_hess(z, resid, 2, 2, "skewt")

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-5, atol=1e-6)

    def test_gjr22_skewt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GJRGARCH(2, 2) + SkewT()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 9
        assert np.all(np.isfinite(result.params))


class TestGARCHSkewTOracle:
    @pytest.mark.parametrize(
        "params",
        [
            np.array([1.5e-6, 0.05, 0.92, 8.0, -0.25], dtype=np.float64),
            np.array([2.0e-6, 0.07, 0.88, 12.0, 0.20], dtype=np.float64),
        ],
    )
    def test_garch11_skewt_matches_c_objective_and_gradient(
        self,
        garch_skewt_series: np.ndarray,
        params: np.ndarray,
    ) -> None:
        resid2 = garch_skewt_series * garch_skewt_series
        sigma2 = np.zeros_like(garch_skewt_series)
        sigma2[0] = np.mean(resid2)
        grad = np.zeros_like(params)

        _core._garch_variance_11(_as_cptr(params[:3]), _as_cptr(resid2), _as_cptr(sigma2), garch_skewt_series.size)
        value_c = -_core._skewt_ll(
            _as_cptr(garch_skewt_series),
            _as_cptr(sigma2),
            garch_skewt_series.size,
            float(params[3]),
            float(params[4]),
        )
        value_grad_c = _core._garch_ll_grad_11_skewt(
            _as_cptr(params), _as_cptr(garch_skewt_series), _as_cptr(grad), garch_skewt_series.size
        )
        value_ad, grad_ad, hess_ad = garch_value_grad_hess(
            params, resid2, 1, 1, dist="skewt", resid=garch_skewt_series
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=1e-6, atol=1e-4)
        np.testing.assert_allclose(grad, grad_ad, rtol=2e-6, atol=1e-4)
        assert np.all(np.isfinite(hess_ad))
