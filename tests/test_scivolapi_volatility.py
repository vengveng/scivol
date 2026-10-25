"""scivolapi volatility recursions beyond GARCH: AD oracle, masks, charts, fits."""

from __future__ import annotations

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
from tests._oracle import (  # noqa: E402
    _jax_aparch_power_moment,
    _jax_modules,
    aparch_value_grad_hess,
    _jax_negative_second_moment,
    egarch_value_grad_hess,
    gjr_garch_value_grad_hess,
)

ORDERS = [(1, 1), (2, 1), (1, 2)]
GJR_DENSITIES = {
    "normal": ([], "normal"), "student_t": ([6.0], "studentt"), "ged": ([1.3], "ged"), "skew_t": ([6.0, -0.3], "skewt"),
}


def gjr_model(density: str, p: int = 1, q: int = 1) -> "api.Model":
    spec = api.ModelSpec()
    spec.volatility = api.VolatilityKind.gjr
    spec.p, spec.q = p, q
    spec.density = getattr(api.DensityKind, density)
    return api.Model(spec)


def gjr_point(p: int, q: int, shape) -> np.ndarray:
    return np.array([0.08] + [0.05 / p] * p + [0.10 / p] * p + [0.80 / q] * q + list(shape))


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    return np.random.default_rng(5).standard_t(5, 1000) * 0.9


def assert_close(actual, expected, rtol):
    scale = max(1.0, float(np.max(np.abs(expected))))
    assert float(np.max(np.abs(np.asarray(actual) - expected))) <= rtol * scale


@pytest.mark.parametrize("density", list(GJR_DENSITIES))
@pytest.mark.parametrize("p,q", ORDERS)
def test_gjr_jet_matches_ad_oracle(series, density, p, q):
    shape, oracle_name = GJR_DENSITIES[density]
    theta = gjr_point(p, q, shape)
    value, gradient, hessian = gjr_garch_value_grad_hess(theta, series, p, q, oracle_name)
    if density == "normal":
        value += 0.5 * np.log(2.0 * np.pi) * series.size  # the oracle omits the Gaussian constant
    out = api.Jet(gjr_model(density, p, q), series).theta(theta, 7)
    assert out["status"] == "ok"
    assert out["value"] == pytest.approx(value, rel=1e-13)
    assert_close(out["gradient"], gradient, 1e-12)
    assert_close(out["hessian"], hessian, 1e-12)


def test_gjr_every_sparse_request_equals_the_full_request(series):
    jet = api.Jet(gjr_model("student_t", 2, 1), series)
    theta = gjr_point(2, 1, [6.0])
    full = jet.theta(theta, 15)
    for mask in range(1, 16):
        part = jet.theta(theta, mask)
        for key, bit in (("value", 1), ("gradient", 2), ("hessian", 4), ("variance", 8)):
            if mask & bit:
                assert np.array_equal(part[key], full[key])


def test_gjr_without_leverage_is_garch(series):
    spec = api.ModelSpec()
    garch = api.Jet(api.Model(spec), series).theta(np.array([0.08, 0.1, 0.8]), 7)
    gjr = api.Jet(gjr_model("normal"), series).theta(np.array([0.08, 0.1, 0.0, 0.8]), 7)
    assert gjr["value"] == pytest.approx(garch["value"], rel=1e-15)
    keep = [0, 1, 3]
    assert_close(gjr["gradient"][keep], garch["gradient"], 1e-14)
    assert_close(gjr["hessian"][np.ix_(keep, keep)], garch["hessian"], 1e-14)


def test_gjr_z_jet_includes_chart_curvature(series):
    """The simplex holds c+ = (1 - k) alpha, c- = k (alpha + gamma), beta with k = 1/2."""
    jax, jnp, _ = _jax_modules()
    model = gjr_model("normal", 1, 2)
    jet = api.Jet(model, series)
    theta = np.array([0.08, 0.05, -0.02, 0.5, 0.3])  # negative leverage inside the domain
    z = model.to_z(theta)
    assert np.allclose(model.to_theta(z), theta, rtol=1e-13, atol=1e-16)
    limit, kappa = 1.0 - 8.0 * 5 * np.finfo(float).eps, 0.5

    def chart(u):
        e = jnp.exp(u[1:])
        v = limit * e / (1.0 + jnp.sum(e))
        alpha = v[0] / (1.0 - kappa)
        return jnp.stack([np.finfo(float).tiny + jnp.exp(u[0]), alpha, v[1] / kappa - alpha, v[2], v[3]])

    full = jet.theta(theta, api.GRADIENT | api.HESSIAN)
    jac = np.asarray(jax.jacfwd(chart)(jnp.asarray(z)))
    curvature = np.asarray(jax.hessian(lambda u: jnp.dot(jnp.asarray(full["gradient"]), chart(u)))(jnp.asarray(z)))
    out = jet.z(z, api.GRADIENT | api.HESSIAN)
    assert_close(out["gradient"], jac.T @ full["gradient"], 1e-12)
    assert_close(out["hessian"], jac.T @ full["hessian"] @ jac + curvature, 1e-11)


def test_gjr_chart_differentiates_the_skewed_persistence_weight(series):
    """With a skewed density kappa = E[z^2 1(z<0)] depends on the shapes, and so
    do alpha = c+ / (1 - kappa) and gamma = c- / kappa - alpha."""
    jax, jnp, jsp = _jax_modules()
    model = gjr_model("skew_t", 1, 1)
    jet = api.Jet(model, series)
    theta = np.array([0.08, 0.05, 0.08, 0.85, 6.0, -0.3])
    z = model.to_z(theta)
    assert np.allclose(model.to_theta(z), theta, rtol=1e-12, atol=1e-15)
    limit = 1.0 - 8.0 * 4 * np.finfo(float).eps
    tiny, two, bound = np.finfo(float).tiny, np.nextafter(2.0, 3.0), np.nextafter(1.0, 0.0)

    def chart(u):
        nu, lam = two + jnp.exp(u[4]), bound * jnp.tanh(u[5])
        kappa = _jax_negative_second_moment(jnp, jsp, jnp.stack([nu, lam]), "skewt")
        e = jnp.exp(u[1:4])
        v = limit * e / (1.0 + jnp.sum(e))
        alpha = v[0] / (1.0 - kappa)
        return jnp.stack([tiny + jnp.exp(u[0]), alpha, v[1] / kappa - alpha, v[2], nu, lam])

    full = jet.theta(theta, api.GRADIENT | api.HESSIAN)
    jac = np.asarray(jax.jacfwd(chart)(jnp.asarray(z)))
    curvature = np.asarray(jax.hessian(lambda u: jnp.dot(jnp.asarray(full["gradient"]), chart(u)))(jnp.asarray(z)))
    out = jet.z(z, api.GRADIENT | api.HESSIAN)
    assert out["status"] == "ok"
    # The oracle's kappa is a fixed quadrature, accurate to about 1e-12.
    assert_close(out["gradient"], jac.T @ full["gradient"], 1e-9)
    assert_close(out["hessian"], jac.T @ full["hessian"] @ jac + curvature, 1e-8)


def test_gjr_domain_admits_negative_leverage_down_to_minus_alpha(series):
    model = gjr_model("normal")
    assert model.feasible(np.array([0.1, 0.05, 0.3, 0.79]))      # 0.05 + 0.15 + 0.79 < 1
    assert not model.feasible(np.array([0.1, 0.05, 0.4, 0.79]))  # 0.05 + 0.20 + 0.79 > 1
    assert model.feasible(np.array([0.1, 0.05, -0.04, 0.9]))     # alpha + gamma >= 0
    assert not model.feasible(np.array([0.1, 0.05, -0.06, 0.9]))
    jet = api.Jet(model, series)
    assert jet.theta(np.array([0.1, 0.05, -0.06, 0.9]), api.VALUE)["status"] == "not_evaluable"


# --------------------------------------------------------------------------- #
# EGARCH
# --------------------------------------------------------------------------- #

EGARCH_DENSITIES = {
    "normal": ([], "normal"), "student_t": ([6.0], "studentt"), "ged": ([1.3], "ged"), "skew_t": ([6.0, -0.2], "skewt"),
}


def egarch_model(density: str, p: int = 1, q: int = 1) -> "api.Model":
    spec = api.ModelSpec()
    spec.volatility = api.VolatilityKind.egarch
    spec.p, spec.q = p, q
    spec.density = getattr(api.DensityKind, density)
    return api.Model(spec)


def egarch_point(p: int, q: int, shape) -> np.ndarray:
    beta = [0.9] if q == 1 else [0.6, 0.25]
    return np.array([-0.05] + [0.12 / p] * p + [-0.06 / p] * p + beta + list(shape))


@pytest.mark.parametrize("density", list(EGARCH_DENSITIES))
@pytest.mark.parametrize("p,q", ORDERS)
def test_egarch_jet_matches_ad_oracle(series, density, p, q):
    """E|z| enters the recursion, so the shape derivatives run through it too."""
    shape, oracle_name = EGARCH_DENSITIES[density]
    theta = egarch_point(p, q, shape)
    value, gradient, hessian = egarch_value_grad_hess(theta, series, oracle_name, p, q)
    if density == "normal":
        value += 0.5 * np.log(2.0 * np.pi) * series.size  # the oracle omits the Gaussian constant
    out = api.Jet(egarch_model(density, p, q), series).theta(theta, 7)
    assert out["status"] == "ok"
    assert out["value"] == pytest.approx(value, rel=1e-13)
    assert_close(out["gradient"], gradient, 1e-12)
    assert_close(out["hessian"], hessian, 1e-12)


def test_egarch_every_sparse_request_equals_the_full_request(series):
    jet = api.Jet(egarch_model("skew_t", 2, 2), series)
    theta = egarch_point(2, 2, [6.0, -0.2])
    full = jet.theta(theta, 15)
    for mask in range(1, 16):
        part = jet.theta(theta, mask)
        for key, bit in (("value", 1), ("gradient", 2), ("hessian", 4), ("variance", 8)):
            if mask & bit:
                assert np.array_equal(part[key], full[key])


def test_egarch_domain_is_the_stable_lag_polynomial():
    model = egarch_model("normal", 1, 2)
    assert model.feasible(np.array([0.0, 0.1, -0.05, 1.2, -0.4]))  # roots 1/0.67, 1/0.53: stable
    assert not model.feasible(np.array([0.0, 0.1, -0.05, 0.5, 0.6]))  # a root inside the unit circle
    assert model.feasible(np.array([0.0, 0.1, -0.05, -0.9, 0.0]))


def test_egarch_z_jet_includes_reflection_chart_curvature(series):
    jax, jnp, _ = _jax_modules()
    model = egarch_model("student_t", 1, 2)
    jet = api.Jet(model, series)
    theta = egarch_point(1, 2, [6.0])
    z = model.to_z(theta)
    assert np.allclose(model.to_theta(z), theta, rtol=1e-13, atol=1e-15)
    bound = np.nextafter(1.0, 0.0)

    def chart(u):
        k1, k2 = bound * jnp.tanh(u[3]), bound * jnp.tanh(u[4])
        beta = jnp.stack([k1 - k2 * k1, k2])  # Levinson for q = 2
        nu = np.nextafter(2.0, 3.0) + jnp.exp(u[5])
        return jnp.concatenate([u[:3], beta, nu[None]])

    full = jet.theta(theta, api.GRADIENT | api.HESSIAN)
    jac = np.asarray(jax.jacfwd(chart)(jnp.asarray(z)))
    curvature = np.asarray(jax.hessian(lambda u: jnp.dot(jnp.asarray(full["gradient"]), chart(u)))(jnp.asarray(z)))
    out = jet.z(z, api.GRADIENT | api.HESSIAN)
    assert_close(out["gradient"], jac.T @ full["gradient"], 1e-12)
    assert_close(out["hessian"], jac.T @ full["hessian"] @ jac + curvature, 1e-11)


# --------------------------------------------------------------------------- #
# APARCH
# --------------------------------------------------------------------------- #

APARCH_DENSITIES = EGARCH_DENSITIES
APARCH_CASES = [(1, 1, 1.4), (2, 1, 2.3), (1, 2, 0.8)]


def aparch_model(density: str, p: int = 1, q: int = 1) -> "api.Model":
    spec = api.ModelSpec()
    spec.volatility = api.VolatilityKind.aparch
    spec.p, spec.q = p, q
    spec.density = getattr(api.DensityKind, density)
    return api.Model(spec)


def aparch_point(p: int, q: int, delta: float, shape) -> np.ndarray:
    return np.array([0.05] + [0.08 / p] * p + [0.3 / p] * p + [0.85 / q] * q + [delta] + list(shape))


def check_aparch_against_oracle(data, density, p, q, delta):
    shape, oracle_name = APARCH_DENSITIES[density]
    theta = aparch_point(p, q, delta, shape)
    value, gradient, hessian = aparch_value_grad_hess(
        theta, data, p, q, oracle_name, initial_variance=float(np.mean(data * data)))
    out = api.Jet(aparch_model(density, p, q), data).theta(theta, 7)
    assert out["status"] == "ok"
    assert out["value"] == pytest.approx(value, rel=1e-13)
    assert_close(out["gradient"], gradient, 1e-12)
    assert_close(out["hessian"], hessian, 1e-12)


# The APARCH oracle recompiles per call (25-50 s at n = 1000): one short case per
# density routinely, the full matrix with --run-slow.
@pytest.mark.parametrize("density", list(APARCH_DENSITIES))
def test_aparch_jet_matches_ad_oracle(series, density):
    check_aparch_against_oracle(series[:200], density, 2, 1, 1.4)


@pytest.mark.slow
@pytest.mark.parametrize("density", list(APARCH_DENSITIES))
@pytest.mark.parametrize("p,q,delta", APARCH_CASES)
def test_aparch_jet_matches_ad_oracle_everywhere(series, density, p, q, delta):
    check_aparch_against_oracle(series, density, p, q, delta)


@pytest.mark.parametrize("delta", [0.6, 1.4, 2.5])
def test_aparch_derivatives_are_continuous_at_exact_zero_returns(series, delta):
    """(|e| - gamma e)^delta and all its derivatives vanish at e = 0 for delta > 0."""
    exact, near = series.copy(), series.copy()
    exact[::9] = 0.0
    near[::9] = 1e-300
    theta = aparch_point(1, 1, delta, [6.0])
    at = api.Jet(aparch_model("student_t"), exact).theta(theta, 7)
    beside = api.Jet(aparch_model("student_t"), near).theta(theta, 7)
    assert at["status"] == beside["status"] == "ok"
    assert at["value"] == pytest.approx(beside["value"], rel=1e-14)
    assert_close(at["gradient"], beside["gradient"], 1e-12)
    assert_close(at["hessian"], beside["hessian"], 1e-12)


def test_aparch_every_sparse_request_equals_the_full_request(series):
    jet = api.Jet(aparch_model("skew_t", 2, 1), series)
    theta = aparch_point(2, 1, 1.4, [6.0, -0.2])
    full = jet.theta(theta, 15)
    for mask in range(1, 16):
        part = jet.theta(theta, mask)
        for key, bit in (("value", 1), ("gradient", 2), ("hessian", 4), ("variance", 8)):
            if mask & bit:
                assert np.array_equal(part[key], full[key])


def test_aparch_chart_differentiates_the_power_moment_weights(series):
    """alpha_i = v_i / M_i with M_i = E[(|z| - gamma_i z)^delta] depending on gamma_i,
    delta and the shapes."""
    jax, jnp, jsp = _jax_modules()
    model = aparch_model("student_t", 2, 1)
    jet = api.Jet(model, series)
    theta = np.array([0.05, 0.05, 0.03, 0.3, -0.2, 0.8, 1.4, 7.0])
    z = model.to_z(theta)
    assert np.allclose(model.to_theta(z), theta, rtol=1e-12, atol=1e-15)
    limit = 1.0 - 8.0 * 4 * np.finfo(float).eps
    tiny, two, bound = np.finfo(float).tiny, np.nextafter(2.0, 3.0), np.nextafter(1.0, 0.0)

    def chart(u):
        g1, g2 = bound * jnp.tanh(u[3]), bound * jnp.tanh(u[4])
        delta, nu = tiny + jnp.exp(u[6]), two + jnp.exp(u[7])
        shape = jnp.stack([nu])
        m1 = _jax_aparch_power_moment(jnp, jsp, g1, delta, shape, "studentt")
        m2 = _jax_aparch_power_moment(jnp, jsp, g2, delta, shape, "studentt")
        e = jnp.exp(jnp.stack([u[1], u[2], u[5]]))
        v = limit * e / (1.0 + jnp.sum(e))
        return jnp.stack([tiny + jnp.exp(u[0]), v[0] / m1, v[1] / m2, g1, g2, v[2], delta, nu])

    full = jet.theta(theta, api.GRADIENT | api.HESSIAN)
    jac = np.asarray(jax.jacfwd(chart)(jnp.asarray(z)))
    curvature = np.asarray(jax.hessian(lambda u: jnp.dot(jnp.asarray(full["gradient"]), chart(u)))(jnp.asarray(z)))
    out = jet.z(z, api.GRADIENT | api.HESSIAN)
    assert out["status"] == "ok"
    assert_close(out["gradient"], jac.T @ full["gradient"], 1e-11)
    assert_close(out["hessian"], jac.T @ full["hessian"] @ jac + curvature, 1e-10)


def test_aparch_domain_needs_the_power_moment():
    model = aparch_model("student_t")
    assert model.feasible(np.array([0.05, 0.08, 0.3, 0.85, 1.4, 6.0]))
    assert not model.feasible(np.array([0.05, 0.08, 0.3, 0.85, 6.5, 6.0]))  # delta >= nu: no moment
    assert not model.feasible(np.array([0.05, 0.08, 0.3, 0.95, 1.4, 6.0]))  # persistence >= 1


# --------------------------------------------------------------------------- #
# Complete fits through the public API
# --------------------------------------------------------------------------- #

from scivol import APARCH, EGARCH, GED, GJRGARCH, Normal, SkewT, StudentT  # noqa: E402


@pytest.fixture(scope="module")
def gjr_data() -> np.ndarray:
    rng = np.random.default_rng(4)
    y, h = np.empty(4000), 1.0
    z = rng.standard_t(6, 4000) * np.sqrt(4.0 / 6.0)
    for t in range(4000):
        y[t] = np.sqrt(h) * z[t]
        h = 0.05 + (0.03 + 0.10 * (y[t] < 0.0)) * y[t] ** 2 + 0.90 * h
    return y


# Maximized log-likelihoods from the previous production engine on the same
# data and likelihood: the fit must reach at least these values.
PRODUCTION_LOG_LIKELIHOOD = {
    "gjr_Normal": -6672.984987994012,
    "gjr_StudentT": -6565.1803279297865,
    "gjr_GED": -6578.702792687239,
    "egarch_Normal": -4861.208404301725,
    "aparch_Normal": -4551.4396952828265,
    "egarch_StudentT": -4671.111941620415,
    "aparch_StudentT": -4380.192499173566,
    "egarch_SkewT": -4670.948308403579,
    "aparch_SkewT": -4380.036350027015,
}


@pytest.mark.parametrize("density", [Normal, StudentT, GED])
def test_gjr_fit_reaches_the_production_optimum(gjr_data, density):
    result = (GJRGARCH(1, 1) + density()).fit(gjr_data)
    assert result.converged and result.fit_info.inference_status == "available"
    assert result.log_likelihood >= PRODUCTION_LOG_LIKELIHOOD["gjr_" + density.__name__] - 1e-6


def test_gjr_fit_recovers_the_leverage_effect(gjr_data):
    result = (GJRGARCH(1, 1) + StudentT()).fit(gjr_data)
    gamma, se = result.params[2], result.std_errors[2]
    assert abs(gamma - 0.10) < 4.0 * se


@pytest.fixture(scope="module")
def egarch_data() -> np.ndarray:
    rng = np.random.default_rng(8)
    y, x = np.empty(4000), 0.0
    z = rng.standard_t(6, 4000) * np.sqrt(4.0 / 6.0)
    for t in range(4000):
        y[t] = np.exp(0.5 * x) * z[t]
        x = -0.02 + 0.12 * (abs(z[t]) - 0.8) - 0.06 * z[t] + 0.95 * x
    return y


@pytest.mark.parametrize("density", [Normal, StudentT, SkewT])
def test_egarch_fit_reaches_the_production_optimum(egarch_data, density):
    result = (EGARCH(1, 1) + density()).fit(egarch_data)
    assert result.converged and result.fit_info.inference_status == "available"
    assert result.log_likelihood >= PRODUCTION_LOG_LIKELIHOOD["egarch_" + density.__name__] - 1e-6


def test_egarch_fit_is_exactly_equivariant_to_data_scale(egarch_data):
    base = (EGARCH(1, 1) + StudentT()).fit(egarch_data)
    scaled = (EGARCH(1, 1) + StudentT()).fit(10.0 * egarch_data)
    beta = base.params[3]
    assert scaled.params[0] == pytest.approx(base.params[0] + 2.0 * np.log(10.0) * (1.0 - beta), abs=1e-7)
    assert scaled.params[1:] == pytest.approx(base.params[1:], rel=1e-6)
    assert scaled.log_likelihood == pytest.approx(base.log_likelihood - egarch_data.size * np.log(10.0), abs=1e-6)


@pytest.fixture(scope="module")
def aparch_data() -> np.ndarray:
    rng = np.random.default_rng(9)
    y, s = np.empty(4000), 1.0
    z = rng.standard_t(6, 4000) * np.sqrt(4.0 / 6.0)
    for t in range(4000):
        y[t] = s ** (1.0 / 1.3) * z[t]
        s = 0.03 + 0.07 * (abs(y[t]) - 0.4 * y[t]) ** 1.3 + 0.90 * s
    return y


@pytest.mark.parametrize("density", [Normal, StudentT, SkewT])
def test_aparch_fit_reaches_the_production_optimum(aparch_data, density):
    result = (APARCH(1, 1) + density()).fit(aparch_data)
    assert result.converged and result.fit_info.inference_status == "available"
    assert result.log_likelihood >= PRODUCTION_LOG_LIKELIHOOD["aparch_" + density.__name__] - 1e-6


def test_aparch_fit_is_exactly_equivariant_to_data_scale(aparch_data):
    base = (APARCH(1, 1) + StudentT()).fit(aparch_data)
    scaled = (APARCH(1, 1) + StudentT()).fit(10.0 * aparch_data)
    delta = base.params[4]
    assert scaled.params[0] == pytest.approx(base.params[0] * 10.0 ** delta, rel=1e-6)
    assert scaled.params[1:] == pytest.approx(base.params[1:], rel=1e-6)
    assert scaled.log_likelihood == pytest.approx(base.log_likelihood - aparch_data.size * np.log(10.0), abs=1e-6)
