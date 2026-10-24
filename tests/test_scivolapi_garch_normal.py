"""scivolapi GARCH(p,q)-Normal: jet against the AD oracle, request masks, paths, fits."""

from __future__ import annotations

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
from scivol._devtools.ad_oracle import _jax_modules, _value_grad_hess, garch_value_grad_hess  # noqa: E402

ORDERS = [(1, 1), (2, 1), (1, 2), (2, 3)]
LOG_2PI = np.log(2.0 * np.pi)


def model_for(p: int, q: int) -> "api.Model":
    spec = api.ModelSpec()
    spec.p, spec.q = p, q
    return api.Model(spec)


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    return np.random.default_rng(7).standard_normal(1500) * 1.3


def interior_theta(p: int, q: int) -> np.ndarray:
    return np.array([0.08] + [0.10 / p] * p + [0.80 / q] * q)


def assert_close(actual, expected, rtol):
    scale = max(1.0, float(np.max(np.abs(expected))))
    assert float(np.max(np.abs(np.asarray(actual) - expected))) <= rtol * scale


@pytest.mark.parametrize("p,q", ORDERS)
def test_theta_jet_matches_ad_oracle(series, p, q):
    model, theta = model_for(p, q), interior_theta(p, q)
    jet = api.Jet(model, series)
    value, gradient, hessian = garch_value_grad_hess(theta, series * series, p, q, "normal")
    value += 0.5 * LOG_2PI * series.size  # the oracle omits the Gaussian constant
    full = jet.theta(theta, api.VALUE | api.GRADIENT | api.HESSIAN)
    assert full["status"] == "ok"
    assert full["value"] == pytest.approx(value, rel=1e-13)
    assert_close(full["gradient"], gradient, 1e-12)
    assert_close(full["hessian"], hessian, 1e-12)


@pytest.mark.parametrize("p,q", ORDERS)
def test_every_sparse_request_equals_the_full_request(series, p, q):
    model, theta = model_for(p, q), interior_theta(p, q)
    jet = api.Jet(model, series)
    full = jet.theta(theta, 15)
    for mask in range(1, 16):
        part = jet.theta(theta, mask)
        assert part["status"] == "ok"
        for key, bit in (("value", 1), ("gradient", 2), ("hessian", 4), ("variance", 8)):
            if mask & bit:
                assert np.array_equal(part[key], full[key])
            else:
                assert key not in part


@pytest.mark.parametrize("p,q", ORDERS)
def test_z_jet_includes_chart_curvature(series, p, q):
    jax, jnp, _ = _jax_modules()
    model, theta = model_for(p, q), interior_theta(p, q)
    jet = api.Jet(model, series)
    z = model.to_z(theta)
    limit = 1.0 - 8.0 * (p + q + 1) * np.finfo(float).eps
    r2 = jnp.asarray(series * series)

    def objective(zj):
        omega = np.finfo(float).tiny + jnp.exp(zj[0])
        e = jnp.exp(zj[1:])
        coefficients = limit * e / (1.0 + jnp.sum(e))
        alpha, beta = coefficients[:p], coefficients[p:]
        seed = jnp.mean(r2)

        def step(history, t):
            h = omega
            for i in range(p):
                h = h + jnp.where(t > i, alpha[i] * r2[t - 1 - i], 0.0)
            for j in range(q):
                h = h + jnp.where(t > j, beta[j] * history[j], 0.0)
            return jnp.concatenate([h[None], history[:-1]]), h

        _, tail = jax.lax.scan(step, jnp.full((q,), seed), jnp.arange(1, series.size))
        h = jnp.concatenate([seed[None], tail])
        return 0.5 * jnp.sum(jnp.log(h) + r2 / h)

    _, gradient, hessian = _value_grad_hess(objective, z)
    full = jet.z(z, api.VALUE | api.GRADIENT | api.HESSIAN)
    assert full["status"] == "ok"
    assert_close(full["gradient"], gradient, 1e-12)
    assert_close(full["hessian"], hessian, 1e-12)
    assert np.allclose(model.to_theta(z), theta, rtol=1e-14, atol=0.0)


def test_paths_reproduce_the_likelihood(series):
    model, theta = model_for(1, 1), interior_theta(1, 1)
    out = api.Jet(model, series).theta(theta, api.VALUE | api.PATHS)
    h, e = out["variance"], out["residual"]
    assert np.array_equal(e, series)
    assert out["innovation"] == pytest.approx(e / np.sqrt(h), rel=1e-15)
    assert out["value"] == pytest.approx(0.5 * np.sum(LOG_2PI + np.log(h) + e * e / h), rel=1e-13)


def test_jet_rejects_points_where_the_likelihood_is_undefined(series):
    jet = api.Jet(model_for(1, 1), series)
    assert jet.theta(np.array([-0.1, 0.1, 0.8]), api.VALUE)["status"] == "not_evaluable"
    assert jet.theta(np.array([0.1, -0.1, 0.8]), api.VALUE)["status"] == "not_evaluable"


# --------------------------------------------------------------------------- #
# Complete fits through the public API
# --------------------------------------------------------------------------- #

from scivol import GARCH, Normal  # noqa: E402


def simulate_garch(n: int, omega: float, alpha: float, beta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    h = omega / (1.0 - alpha - beta)
    for t in range(n):
        y[t] = np.sqrt(h) * rng.standard_normal()
        h = omega + alpha * y[t] * y[t] + beta * h
    return y


@pytest.fixture(scope="module")
def garch_data() -> np.ndarray:
    return simulate_garch(5000, 0.05, 0.08, 0.90, seed=11)


def test_fit_recovers_known_parameters(garch_data):
    result = (GARCH(1, 1) + Normal()).fit(garch_data, engine="scivolapi")
    assert result.converged
    assert result.fit_info.inference_status == "available"
    omega, alpha, beta = result.params
    se = result.std_errors
    assert abs(alpha - 0.08) < 4 * se[1] and abs(beta - 0.90) < 4 * se[2]
    assert abs(omega - 0.05) < 4 * se[0]


def test_fit_matches_the_production_likelihood(garch_data):
    new = (GARCH(1, 1) + Normal()).fit(garch_data, engine="scivolapi")
    old = (GARCH(1, 1) + Normal()).fit(garch_data)
    assert new.log_likelihood == pytest.approx(old.log_likelihood, abs=1e-6)


def test_theta_and_z_paths_run_independently_and_agree(garch_data):
    theta = (GARCH(1, 1) + Normal()).fit(garch_data, engine="scivolapi", coordinates="theta")
    z = (GARCH(1, 1) + Normal()).fit(garch_data, engine="scivolapi", coordinates="z")
    assert theta.fit_info.optimizer_phases == ("theta:slsqp", "theta:newton")
    assert z.fit_info.optimizer_phases == ("z:slsqp",)
    assert theta.converged and z.converged
    assert theta.log_likelihood == pytest.approx(z.log_likelihood, abs=1e-6)


def test_fit_is_exactly_equivariant_to_data_scale(garch_data):
    base = (GARCH(1, 1) + Normal()).fit(garch_data, engine="scivolapi")
    scaled = (GARCH(1, 1) + Normal()).fit(10.0 * garch_data, engine="scivolapi")
    assert scaled.params[0] == pytest.approx(100.0 * base.params[0], rel=1e-8)
    assert scaled.params[1:] == pytest.approx(base.params[1:], rel=1e-8)
    assert scaled.log_likelihood == pytest.approx(base.log_likelihood - garch_data.size * np.log(10.0), abs=1e-6)


def test_reported_paths_reproduce_the_reported_likelihood(garch_data):
    result = (GARCH(1, 1) + Normal()).fit(garch_data, engine="scivolapi")
    h = result.sigma2
    assert result.log_likelihood == pytest.approx(
        -0.5 * np.sum(LOG_2PI + np.log(h) + garch_data**2 / h), abs=1e-8)


def test_boundary_estimate_reports_no_standard_errors():
    white_noise = np.random.default_rng(3).standard_normal(3000) * 0.7  # true alpha = beta = 0
    result = (GARCH(1, 1) + Normal()).fit(white_noise, engine="scivolapi")
    assert result.fit_info.inference_status == "at_boundary"
    assert result.cov_matrix is None
