"""scivolapi volatility recursions beyond GARCH: AD oracle, masks, charts, fits."""

from __future__ import annotations

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
from scivol._devtools.ad_oracle import _jax_modules, gjr_garch_value_grad_hess  # noqa: E402

ORDERS = [(1, 1), (2, 1), (1, 2)]
GJR_DENSITIES = {"normal": ([], "normal"), "student_t": ([6.0], "studentt"), "ged": ([1.3], "ged")}


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


def test_gjr_domain_admits_negative_leverage_down_to_minus_alpha(series):
    model = gjr_model("normal")
    assert model.feasible(np.array([0.1, 0.05, 0.3, 0.79]))      # 0.05 + 0.15 + 0.79 < 1
    assert not model.feasible(np.array([0.1, 0.05, 0.4, 0.79]))  # 0.05 + 0.20 + 0.79 > 1
    assert model.feasible(np.array([0.1, 0.05, -0.04, 0.9]))     # alpha + gamma >= 0
    assert not model.feasible(np.array([0.1, 0.05, -0.06, 0.9]))
    jet = api.Jet(model, series)
    assert jet.theta(np.array([0.1, 0.05, -0.06, 0.9]), api.VALUE)["status"] == "not_evaluable"


# --------------------------------------------------------------------------- #
# Complete fits through the public API
# --------------------------------------------------------------------------- #

from scivol import GED, GJRGARCH, Normal, StudentT  # noqa: E402


@pytest.fixture(scope="module")
def gjr_data() -> np.ndarray:
    rng = np.random.default_rng(4)
    y, h = np.empty(4000), 1.0
    z = rng.standard_t(6, 4000) * np.sqrt(4.0 / 6.0)
    for t in range(4000):
        y[t] = np.sqrt(h) * z[t]
        h = 0.05 + (0.03 + 0.10 * (y[t] < 0.0)) * y[t] ** 2 + 0.90 * h
    return y


@pytest.mark.parametrize("density", [Normal, StudentT, GED])
def test_gjr_fit_matches_the_production_likelihood(gjr_data, density):
    new = (GJRGARCH(1, 1) + density()).fit(gjr_data, engine="scivolapi")
    old = (GJRGARCH(1, 1) + density()).fit(gjr_data)
    assert new.converged and new.fit_info.inference_status == "available"
    assert new.log_likelihood == pytest.approx(old.log_likelihood, abs=1e-6)


def test_gjr_fit_recovers_the_leverage_effect(gjr_data):
    result = (GJRGARCH(1, 1) + StudentT()).fit(gjr_data, engine="scivolapi")
    gamma, se = result.params[2], result.std_errors[2]
    assert abs(gamma - 0.10) < 4.0 * se
