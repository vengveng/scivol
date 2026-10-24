"""scivolapi GARCH(p,q) with non-Normal densities: AD oracle, masks, charts, reductions."""

from __future__ import annotations

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
from scivol._devtools.ad_oracle import _jax_modules, garch_value_grad_hess  # noqa: E402

ORACLE_NAME = {"student_t": "studentt", "skew_t": "skewt", "ged": "ged"}
SHAPES = {  # shape points per density, including near-limit values
    "student_t": [[8.0], [2.3], [150.0]],
    "skew_t": [[7.0, -0.3], [3.1, 0.6], [60.0, 0.05]],
    "ged": [[1.4], [0.7], [3.5]],
}
ORDERS = [(1, 1), (2, 1), (1, 2)]


def model_for(density: str, p: int = 1, q: int = 1) -> "api.Model":
    spec = api.ModelSpec()
    spec.p, spec.q = p, q
    spec.density = getattr(api.DensityKind, density)
    return api.Model(spec)


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.standard_t(5, 1200) * 0.9


def point(p: int, q: int, shape) -> np.ndarray:
    return np.array([0.08] + [0.10 / p] * p + [0.80 / q] * q + list(shape))


def assert_close(actual, expected, rtol):
    scale = max(1.0, float(np.max(np.abs(expected))))
    assert float(np.max(np.abs(np.asarray(actual) - expected))) <= rtol * scale


@pytest.mark.parametrize("density", list(ORACLE_NAME))
@pytest.mark.parametrize("p,q", ORDERS)
def test_theta_jet_matches_ad_oracle(series, density, p, q):
    model = model_for(density, p, q)
    jet = api.Jet(model, series)
    for shape in SHAPES[density]:
        theta = point(p, q, shape)
        value, gradient, hessian = garch_value_grad_hess(
            theta, series * series, p, q, ORACLE_NAME[density], resid=series)
        full = jet.theta(theta, api.VALUE | api.GRADIENT | api.HESSIAN)
        assert full["status"] == "ok", shape
        assert full["value"] == pytest.approx(value, rel=1e-12), shape
        assert_close(full["gradient"], gradient, 1e-12)
        assert_close(full["hessian"], hessian, 1e-12)


@pytest.mark.parametrize("density", list(ORACLE_NAME))
def test_every_sparse_request_equals_the_full_request(series, density):
    model = model_for(density, 2, 1)
    jet = api.Jet(model, series)
    theta = point(2, 1, SHAPES[density][0])
    full = jet.theta(theta, 15)
    for mask in range(1, 16):
        part = jet.theta(theta, mask)
        assert part["status"] == "ok"
        for key, bit in (("value", 1), ("gradient", 2), ("hessian", 4), ("variance", 8)):
            if mask & bit:
                assert np.array_equal(part[key], full[key])


def jax_theta(jnp, jax, model, z):
    """Independent JAX statement of the charts declared by the model's bounds."""
    lower, upper = model.bounds()
    names = model.parameter_names
    coefficients = [i for i, n in enumerate(names) if n.startswith(("alpha", "beta"))]
    k = len(coefficients)
    limit = 1.0 - 8.0 * (k + 1) * np.finfo(float).eps
    e = jnp.exp(z[coefficients[0]:coefficients[-1] + 1])
    simplex = limit * e / (1.0 + jnp.sum(e))
    out = []
    for i in range(len(names)):
        if i in coefficients:
            out.append(simplex[i - coefficients[0]])
        elif np.isfinite(upper[i]):
            out.append(upper[i] * jnp.tanh(z[i]))
        elif np.isfinite(lower[i]):
            out.append(lower[i] + jax.nn.softplus(z[i]))
        else:
            out.append(z[i])
    return jnp.stack(out)


@pytest.mark.parametrize("density", list(ORACLE_NAME))
def test_z_jet_includes_chart_curvature(series, density):
    jax, jnp, _ = _jax_modules()
    model = model_for(density, 1, 2)
    jet = api.Jet(model, series)
    theta = point(1, 2, SHAPES[density][0])
    z = model.to_z(theta)
    assert np.allclose(model.to_theta(z), theta, rtol=1e-13, atol=0.0)

    full_theta = jet.theta(theta, api.GRADIENT | api.HESSIAN)
    jac = np.asarray(jax.jacfwd(lambda v: jax_theta(jnp, jax, model, v))(jnp.asarray(z)))
    curv = np.asarray(jax.hessian(
        lambda v: jnp.dot(jnp.asarray(full_theta["gradient"]), jax_theta(jnp, jax, model, v)))(jnp.asarray(z)))
    expected_g = jac.T @ full_theta["gradient"]
    expected_h = jac.T @ full_theta["hessian"] @ jac + curv

    full_z = jet.z(z, api.GRADIENT | api.HESSIAN)
    assert full_z["status"] == "ok"
    assert_close(full_z["gradient"], expected_g, 1e-12)
    assert_close(full_z["hessian"], expected_h, 1e-11)


def test_skew_t_at_zero_skew_is_student_t(series):
    skew, student = api.Jet(model_for("skew_t"), series), api.Jet(model_for("student_t"), series)
    a = skew.theta(point(1, 1, [6.5, 0.0]), 7)
    b = student.theta(point(1, 1, [6.5]), 7)
    assert a["value"] == pytest.approx(b["value"], rel=1e-14)
    assert_close(a["gradient"][:4], b["gradient"], 1e-12)
    assert_close(a["hessian"][:4, :4], b["hessian"], 1e-11)


def test_ged_with_shape_two_is_normal(series):
    ged, normal = api.Jet(model_for("ged"), series), api.Jet(model_for("normal"), series)
    a = ged.theta(point(1, 1, [2.0]), 7)
    b = normal.theta(point(1, 1, []), 7)
    assert a["value"] == pytest.approx(b["value"], rel=1e-14)
    assert_close(a["gradient"][:3], b["gradient"], 1e-12)
    assert_close(a["hessian"][:3, :3], b["hessian"], 1e-11)


def test_ged_derivatives_exist_at_exact_zero_returns(series):
    data = series.copy()
    data[::7] = 0.0
    jet = api.Jet(model_for("ged"), data)
    theta = point(1, 1, [0.8])
    value, gradient, hessian = garch_value_grad_hess(theta, data * data, 1, 1, "ged")
    out = jet.theta(theta, 7)
    assert out["status"] == "ok"
    assert out["value"] == pytest.approx(value, rel=1e-12)
    assert_close(out["gradient"], gradient, 1e-10)
    assert_close(out["hessian"], hessian, 1e-9)


@pytest.mark.parametrize("density,bad", [("student_t", [2.0]), ("skew_t", [5.0, 1.0]), ("ged", [0.0])])
def test_shapes_outside_the_domain_are_not_evaluable(series, density, bad):
    jet = api.Jet(model_for(density), series)
    assert jet.theta(point(1, 1, bad), api.VALUE)["status"] == "not_evaluable"


# --------------------------------------------------------------------------- #
# Complete fits through the public API
# --------------------------------------------------------------------------- #

from scivol import GARCH, GED, SkewT, StudentT  # noqa: E402

COMPONENTS = {"student_t": StudentT, "skew_t": SkewT, "ged": GED}


def simulate(n: int, innovations: np.ndarray) -> np.ndarray:
    y, h = np.empty(n), 0.05 / 0.02
    for t in range(n):
        y[t] = np.sqrt(h) * innovations[t]
        h = 0.05 + 0.08 * y[t] * y[t] + 0.90 * h
    return y


@pytest.fixture(scope="module")
def t_data() -> np.ndarray:
    rng = np.random.default_rng(1)
    return simulate(4000, rng.standard_t(5, 4000) * np.sqrt(3.0 / 5.0))


@pytest.mark.parametrize("density", list(COMPONENTS))
def test_fit_matches_the_production_likelihood(t_data, density):
    new = (GARCH(1, 1) + COMPONENTS[density]()).fit(t_data, engine="scivolapi")
    old = (GARCH(1, 1) + COMPONENTS[density]()).fit(t_data)
    assert new.converged and new.fit_info.inference_status == "available"
    assert new.log_likelihood == pytest.approx(old.log_likelihood, abs=1e-6)


def test_student_t_fit_recovers_the_tail_index(t_data):
    result = (GARCH(1, 1) + StudentT()).fit(t_data, engine="scivolapi")
    assert abs(result.params[3] - 5.0) < 4.0 * result.std_errors[3]
    assert abs(result.params[1] - 0.08) < 4.0 * result.std_errors[1]
