"""scivolapi joint mean-volatility models: AD oracle, masks, charts, kinks, domain, fits."""

from __future__ import annotations

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
from scivol._devtools.ad_oracle import (  # noqa: E402
    _egarch_skewt_quadrature_rule,
    _jax_aparch_power_moment,
    _jax_hansen_skewt_abs_moment,
    _jax_modules,
    _jax_student_log_normalizer,
    _jax_vg_abs_moment,
)
from scivol._devtools.joint_mean_oracle import _density_nll  # noqa: E402

FAMILY = {"garch": 0, "gjr": 1, "egarch": 2, "aparch": 3}
DENSITY_INDEX = {
    "normal": 0, "student_t": 1, "skew_t": 2, "ged": 3,
    "generalized_hyperbolic": 4, "nig": 5, "hyperbolic": 6, "variance_gamma": 7,
}
ORACLE_NAMES = ("normal", "studentt", "skewt", "ged", "gh", "nig", "hyperbolic", "vg")
SHAPES = {
    "normal": [], "student_t": [6.0], "skew_t": [6.0, -0.25], "ged": [1.5],
    "generalized_hyperbolic": [-0.3, 1.4, -0.7], "nig": [-0.2, 1.1], "hyperbolic": [0.4, 0.6],
    "variance_gamma": [-0.3, 2.5],
}


def mean_spec(constant=True, ar=0, har=(), exog=0, ma=0) -> "api.MeanSpec":
    spec = api.MeanSpec()
    spec.constant, spec.ar, spec.har, spec.exog, spec.ma = constant, ar, list(har), exog, ma
    return spec


def model_for(mean, volatility="garch", density="normal", p=1, q=1) -> "api.Model":
    spec = api.ModelSpec()
    spec.mean = mean
    spec.volatility = getattr(api.VolatilityKind, volatility)
    spec.p, spec.q = p, q
    spec.density = getattr(api.DensityKind, density)
    return api.Model(spec)


def mean_point(mean) -> list[float]:
    point = [0.05] if mean.constant else []
    point += [0.3, -0.1, 0.05, 0.02][: mean.ar]
    point += [0.2, 0.1, 0.05][: len(mean.har)]
    point += [0.5, -0.3, 0.2][: mean.exog]
    point += [0.25, -0.1, 0.05][: mean.ma]
    return point


def volatility_point(volatility, p, q, density) -> list[float]:
    shape = SHAPES[density]
    if volatility == "garch":
        return [0.08] + [0.10 / p] * p + [0.80 / q] * q + shape
    if volatility == "gjr":
        return [0.08] + [0.05 / p] * p + [0.10 / p] * p + [0.80 / q] * q + shape
    if volatility == "egarch":
        return [0.02] + [0.15 / p] * p + [-0.08 / p] * p + [0.90 / q] * q + shape
    return [0.05] + [0.08 / p] * p + [0.3 / p] * p + [0.85 / q] * q + [1.6] + shape


@pytest.fixture(scope="module")
def data():
    """An AR(1)-GARCH(1,1) series with t innovations and two exogenous columns."""
    rng = np.random.default_rng(11)
    n = 180
    x = rng.standard_normal((n, 2))
    y = np.zeros(n)
    h, e = 1.0, 0.0
    for t in range(n):
        h = 0.05 + 0.1 * e * e + 0.85 * h
        e = np.sqrt(h) * rng.standard_t(6) * np.sqrt(4 / 6)
        y[t] = 0.1 + 0.3 * (y[t - 1] if t else 0.0) + x[t] @ [0.4, -0.2] + e
    return y, x


def design(y, x, mean):
    """Regressor rows of the scored rows, built independently of the library."""
    lags = max([mean.ar] + list(mean.har)) if (mean.ar or mean.har) else 0
    rows = []
    for t in range(lags, len(y)):
        row = [1.0] if mean.constant else []
        row += [y[t - i] for i in range(1, mean.ar + 1)]
        row += [np.mean(y[t - w:t]) for w in mean.har]
        row += list(x[t, : mean.exog])
        rows.append(row)
    return lags, np.array(rows).reshape(len(y) - lags, -1)


def row_terms(y, x, mean, volatility, p, q, density):
    """Each scored row's negative log-likelihood as a JAX function of theta.

    Scored rows start after the mean's largest y lag; the variance of the first
    scored row is mean(y^2) over all rows; pre-sample residuals are zero.
    """
    jax, jnp, jsp = _jax_modules()
    family, d = FAMILY[volatility], DENSITY_INDEX[density]
    lags, rows = design(y, x, mean)
    scored = jnp.asarray(y[lags:])
    k, m = rows.shape[1], rows.shape[1] + mean.ma
    n = len(y) - lags
    h0 = float(np.mean(y * y))
    beta_start = m + 1 + (1 if family == 0 else 2) * p
    end = beta_start + q + int(family == 3)
    name = ORACLE_NAMES[d]

    def objective(par):
        b, ma = par[:k], par[k:m]
        residuals = []
        for s in range(n):
            e = scored[s] - jnp.dot(jnp.asarray(rows[s]), b)
            for j in range(min(mean.ma, s)):
                e = e - ma[j] * residuals[s - 1 - j]
            residuals.append(e)
        residuals = jnp.stack(residuals)
        alpha = par[m + 1: m + 1 + p]
        gamma = par[m + 1 + p: beta_start]
        beta, shape = par[beta_start: beta_start + q], par[end:]
        if family == 3:
            delta = par[end - 1]
            history = [jnp.asarray(h0) ** (delta / 2)]
        elif family == 2:
            if d == 1:
                moment = 2 * jnp.exp(_jax_student_log_normalizer(jnp, jsp, shape[0])) / (1 + 1 / (shape[0] - 2))
            elif d == 2:
                nodes, weights = _egarch_skewt_quadrature_rule()
                moment = _jax_hansen_skewt_abs_moment(jnp, jsp, *shape, jnp.asarray(nodes), jnp.asarray(weights))
            elif d == 7:
                moment = _jax_vg_abs_moment(jnp, jsp, *shape)
            else:
                moment = _jax_aparch_power_moment(jnp, jsp, 0.0, 1.0, shape, name)
            history = [jnp.log(h0)]
        else:
            history = [jnp.asarray(h0)]
        variance = [jnp.asarray(h0)]
        for t in range(1, n):
            state = par[m]
            for lag in range(min(t, p)):
                e = residuals[t - 1 - lag]
                if family == 0:
                    state += alpha[lag] * e * e
                elif family == 1:
                    state += (alpha[lag] + gamma[lag] * (e < 0)) * e * e
                elif family == 2:
                    z = e / jnp.sqrt(variance[t - 1 - lag])
                    state += alpha[lag] * (jnp.abs(z) - moment) + gamma[lag] * z
                else:
                    shock = (jnp.where(e >= 0, 1.0, -1.0) - gamma[lag]) * e
                    state += alpha[lag] * shock ** delta
            for lag in range(min(t, q)):
                state += beta[lag] * history[t - 1 - lag]
            history.append(state)
            variance.append(jnp.exp(state) if family == 2 else (state ** (2 / delta) if family == 3 else state))
        variance = jnp.stack(variance)
        z = residuals / jnp.sqrt(variance)
        one = np.zeros(1, dtype=bool)
        return 0.5 * jnp.log(variance) + jnp.stack(
            [_density_nll(jnp, jsp, z[t:t + 1], shape, d, one) for t in range(n)])

    return objective


def oracle(theta, y, x, mean, volatility, p, q, density):
    """Joint negative log-likelihood, gradient and Hessian by JAX."""
    jax, jnp, _ = _jax_modules()
    rows = row_terms(y, x, mean, volatility, p, q, density)

    def objective(par):
        return jnp.sum(rows(par))

    point = jnp.asarray(theta)
    return float(objective(point)), np.asarray(jax.grad(objective)(point)), np.asarray(jax.hessian(objective)(point))


def oracle_scores(theta, y, x, mean, volatility, p, q, density):
    """Each scored row's gradient (rows x dimension) by JAX."""
    jax, jnp, _ = _jax_modules()
    return np.asarray(jax.jacfwd(row_terms(y, x, mean, volatility, p, q, density))(jnp.asarray(theta)))


def assert_close(actual, expected, rtol):
    scale = max(1.0, float(np.max(np.abs(expected))))
    assert float(np.max(np.abs(np.asarray(actual) - expected))) <= rtol * scale


def check_against_oracle(data, mean, volatility, density, p=1, q=1, tolerance=1e-11):
    y, x = data
    model = model_for(mean, volatility, density, p, q)
    theta = np.array(mean_point(mean) + volatility_point(volatility, p, q, density))
    assert model.feasible(theta)
    value, gradient, hessian = oracle(theta, y, x, mean, volatility, p, q, density)
    out = api.Jet(model, y, np.ascontiguousarray(x[:, : mean.exog]) if mean.exog else None).theta(theta, 7)
    assert out["status"] == "ok"
    assert out["value"] == pytest.approx(value, rel=tolerance)
    assert_close(out["gradient"], gradient, tolerance)
    assert_close(out["hessian"], hessian, tolerance)


CASES = [  # (mean, volatility, density, p, q)
    (dict(ar=1, ma=1), "garch", "normal", 1, 1),
    (dict(ar=2, ma=1), "gjr", "student_t", 1, 1),
    (dict(ar=1, exog=2), "egarch", "skew_t", 1, 1),
    (dict(har=(1, 5)), "aparch", "ged", 1, 1),
    (dict(ma=2), "garch", "skew_t", 2, 1),
    (dict(constant=False, ar=1, ma=1), "egarch", "normal", 1, 2),
    (dict(ar=1, har=(3,), exog=1, ma=1), "gjr", "ged", 1, 2),
    (dict(ma=1), "aparch", "student_t", 2, 1),
    (dict(), "garch", "nig", 1, 1),
    (dict(ar=1, ma=1), "egarch", "student_t", 2, 1),
    (dict(ar=2, exog=1), "aparch", "normal", 1, 2),
]


@pytest.mark.parametrize("mean,volatility,density,p,q", CASES)
def test_joint_jet_matches_ad_oracle(data, mean, volatility, density, p, q):
    check_against_oracle(data, mean_spec(**mean), volatility, density, p, q)


@pytest.mark.slow
@pytest.mark.parametrize("density", list(DENSITY_INDEX))
@pytest.mark.parametrize("volatility", list(FAMILY))
def test_joint_jet_matches_ad_oracle_everywhere(data, volatility, density):
    tolerance = 1e-10 if DENSITY_INDEX[density] >= 4 else 1e-11  # the GH oracle's quadrature
    check_against_oracle(data, mean_spec(ar=1, exog=1, ma=1), volatility, density, 1, 1, tolerance)


def test_every_sparse_request_equals_the_full_request(data):
    y, x = data
    mean = mean_spec(ar=1, exog=2, ma=1)
    model = model_for(mean, "gjr", "skew_t")
    theta = np.array(mean_point(mean) + volatility_point("gjr", 1, 1, "skew_t"))
    jet = api.Jet(model, y, x)
    full = jet.theta(theta, 15)
    for request in range(1, 16):
        out = jet.theta(theta, request)
        assert out["valid"] == request
        for key in ("value", "gradient", "hessian", "residual", "variance"):
            if key in out:
                assert np.array_equal(out[key], full[key]), (request, key)


def test_scored_rows_start_after_the_largest_lag(data):
    y, x = data
    mean = mean_spec(ar=2, har=(1, 5))
    jet = api.Jet(model_for(mean), y)
    assert jet.first_scored == 5
    theta = np.array(mean_point(mean) + volatility_point("garch", 1, 1, "normal"))
    out = jet.theta(theta, api.PATHS)
    assert len(out["residual"]) == len(y) - 5
    assert out["variance"][0] == pytest.approx(np.mean(y * y), rel=1e-15)


def test_a_mean_without_coefficients_is_the_volatility_model(data):
    y, _ = data
    theta = np.array(volatility_point("egarch", 1, 1, "student_t"))
    plain = api.Jet(model_for(mean_spec(constant=False), "egarch", "student_t"), y).theta(theta, 7)
    constant = api.Jet(model_for(mean_spec(), "egarch", "student_t"), y).theta(np.r_[0.0, theta], 7)
    assert constant["value"] == pytest.approx(plain["value"], rel=1e-14)
    assert_close(constant["gradient"][1:], plain["gradient"], 1e-13)
    assert_close(constant["hessian"][1:, 1:], plain["hessian"], 1e-13)


def test_z_jet_includes_the_reflection_charts(data):
    jax, jnp, _ = _jax_modules()
    y, x = data
    mean = mean_spec(ar=2, exog=1, ma=2)
    model = model_for(mean, "garch", "student_t")
    jet = api.Jet(model, y, np.ascontiguousarray(x[:, :1]))
    theta = np.array([0.05, 0.3, -0.1, 0.5, 0.25, -0.1] + volatility_point("garch", 1, 1, "student_t"))
    z = model.to_z(theta)
    assert np.allclose(model.to_theta(z), theta, rtol=1e-13, atol=1e-15)
    bound = np.nextafter(1.0, 0.0)

    def levinson2(u):
        k1, k2 = bound * jnp.tanh(u[0]), bound * jnp.tanh(u[1])
        return jnp.stack([k1 - k2 * k1, k2])

    def chart(u):
        ar, ma = levinson2(u[1:3]), -levinson2(u[4:6])
        omega = np.finfo(float).tiny + jnp.exp(u[6])
        logits = jnp.r_[u[7:9], 0.0]
        persistence = (1 - 8 * 3 * np.finfo(float).eps) * jax.nn.softmax(logits)[:-1]
        nu = np.nextafter(2.0, 3.0) + jnp.exp(u[9])
        return jnp.concatenate([u[:1], ar, u[3:4], ma, omega[None], persistence, nu[None]])

    assert np.allclose(np.asarray(chart(jnp.asarray(z))), theta, rtol=1e-12)
    full = jet.theta(theta, api.GRADIENT | api.HESSIAN)
    jac = np.asarray(jax.jacfwd(chart)(jnp.asarray(z)))
    curvature = np.asarray(jax.hessian(lambda u: jnp.dot(jnp.asarray(full["gradient"]), chart(u)))(jnp.asarray(z)))
    out = jet.z(z, api.GRADIENT | api.HESSIAN)
    assert_close(out["gradient"], jac.T @ full["gradient"], 1e-12)
    assert_close(out["hessian"], jac.T @ full["hessian"] @ jac + curvature, 1e-11)


def test_domain_is_stationary_ar_and_har_with_invertible_ma():
    arma = model_for(mean_spec(ar=2, ma=1))
    tail = volatility_point("garch", 1, 1, "normal")
    assert arma.feasible(np.array([0.0, 1.2, -0.4, 0.5] + tail))
    assert not arma.feasible(np.array([0.0, 0.5, 0.6, 0.5] + tail))  # an AR root inside the unit circle
    assert not arma.feasible(np.array([0.0, 0.3, 0.1, 1.2] + tail))  # MA not invertible
    assert arma.feasible(np.array([0.0, 0.3, 0.1, -0.9] + tail))
    har = model_for(mean_spec(har=(1, 5, 22)))
    assert har.feasible(np.array([0.0, 0.3, 0.3, 0.3] + tail))
    assert not har.feasible(np.array([0.0, 0.4, 0.4, 0.4] + tail))  # implied AR sums past one
    lower, upper = model_for(mean_spec(ar=1, ma=1)).bounds()
    assert upper[1] == upper[2] == np.nextafter(1.0, 0.0) and np.isnan(lower[0])


def test_exact_data_scaling(data):
    y, x = data
    mean = mean_spec(ar=1, exog=2, ma=1)
    for volatility in FAMILY:
        model = model_for(mean, volatility, "student_t")
        theta = np.array(mean_point(mean) + volatility_point(volatility, 1, 1, "student_t"))
        factor = 3.7
        scaled = model.rescale(theta, factor)
        base = api.Jet(model, y, x).theta(theta, api.VALUE)["value"]
        moved = api.Jet(model, factor * y, x).theta(scaled, api.VALUE)["value"]
        assert moved == pytest.approx(base + (len(y) - 1) * np.log(factor), rel=1e-12)


# --------------------------------------------------------------------------- #
# Kinks: an exact zero residual
# --------------------------------------------------------------------------- #


def zero_residual_case(volatility, density, tail):
    """A constant-mean model whose constant equals one observation exactly."""
    y = np.random.default_rng(3).standard_normal(60)
    model = model_for(mean_spec(), volatility, density)
    theta = np.array([y[20]] + tail)
    return api.Jet(model, y).theta(theta, 7)


def test_gjr_hessian_does_not_exist_at_a_zero_residual():
    out = zero_residual_case("gjr", "normal", [0.08, 0.05, 0.10, 0.80])
    assert out["valid"] == api.VALUE | api.GRADIENT and out["status"] == "numerical_range"
    smooth = zero_residual_case("gjr", "normal", [0.08, 0.05, 0.0, 0.80])  # no leverage: smooth
    assert smooth["valid"] == 7


def test_egarch_derivatives_do_not_exist_at_a_zero_residual():
    out = zero_residual_case("egarch", "normal", [0.02, 0.15, -0.08, 0.90])
    assert out["valid"] == api.VALUE


@pytest.mark.parametrize("delta,valid", [(2.5, 7), (1.6, 3), (0.9, 1)])
def test_aparch_derivatives_at_a_zero_residual_follow_delta(delta, valid):
    out = zero_residual_case("aparch", "normal", [0.05, 0.08, 0.3, 0.85, delta])
    assert out["valid"] == valid


@pytest.mark.parametrize("nu,valid", [(2.5, 7), (2.0, 7), (1.5, 3), (0.8, 1)])
def test_ged_derivatives_at_a_zero_residual_follow_nu(nu, valid):
    out = zero_residual_case("garch", "ged", [0.08, 0.10, 0.80, nu])
    assert out["valid"] == valid


def test_ged_curvature_at_nu_two_is_the_normal_one():
    y = np.random.default_rng(3).standard_normal(60)
    tail = [0.08, 0.10, 0.80]
    theta = np.array([y[20]] + tail)
    ged = api.Jet(model_for(mean_spec(), "garch", "ged"), y).theta(np.r_[theta, 2.0], 7)
    normal = api.Jet(model_for(mean_spec(), "garch", "normal"), y).theta(theta, 7)
    assert_close(ged["hessian"][:-1, :-1], normal["hessian"], 1e-12)


# --------------------------------------------------------------------------- #
# Fits
# --------------------------------------------------------------------------- #


def simulate_arma_garch(n, seed):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    h, e_prev, y_prev = 0.1, 0.0, 0.0
    for t in range(n):
        h = 0.02 + 0.08 * e_prev ** 2 + 0.88 * h
        e = np.sqrt(h) * rng.standard_normal()
        y[t] = 0.05 + 0.4 * y_prev + e + 0.2 * e_prev
        e_prev, y_prev = e, y[t]
    return y


def test_arma_garch_fit_recovers_the_parameters():
    y = simulate_arma_garch(4000, 7)
    model = model_for(mean_spec(ar=1, ma=1))
    result = api.fit(model, y)
    assert result.success, result.warnings
    np.testing.assert_allclose(result.theta, [0.05, 0.4, 0.2, 0.02, 0.08, 0.88], atol=0.06)
    assert result.scored == len(y) - 1
    for coordinates in (api.Coordinates.theta, api.Coordinates.z):
        options = api.FitOptions()
        options.coordinates = coordinates
        other = api.fit(model, y, options)
        assert other.success, (coordinates, other.warnings)
        assert other.log_likelihood == pytest.approx(result.log_likelihood, abs=1e-6)


@pytest.mark.parametrize("components", ["arma", "arx", "harx"])
def test_public_fit_agrees_with_production(components):
    """Same model up to the conditioning convention: estimates agree closely."""
    from scivol import ARMA, ARX, GARCH, GJRGARCH, HARX

    rng = np.random.default_rng(23)
    y = simulate_arma_garch(2500, 29)
    x = rng.standard_normal((len(y), 1))
    if components == "arma":
        spec, kwargs = ARMA(1, 1) + GJRGARCH(1, 1), {}  # Normal data: StudentT would run nu to infinity
    elif components == "arx":
        spec, kwargs = ARX(1) + GARCH(1, 1), {"x": x}
    else:
        spec, kwargs = HARX([1, 5]) + GARCH(1, 1), {"x": x}
    ours = spec.fit(y, engine="scivolapi", **kwargs)
    production = spec.fit(y, **kwargs)
    assert ours.optimization_result.success, ours.optimization_result.message
    assert len(ours.sigma2) == len(ours.resid) == len(ours.data)
    np.testing.assert_allclose(ours.optimization_result.x, production.optimization_result.x, atol=0.02)


def test_public_fit_keeps_the_index_aligned_with_the_scored_rows():
    import pandas as pd
    from scivol import ARX, GARCH

    y = simulate_arma_garch(800, 31)
    series = pd.Series(y, index=pd.date_range("2020-01-01", periods=len(y), freq="D"))
    result = (ARX(3) + GARCH(1, 1)).fit(series, engine="scivolapi", hold_back=10)
    assert len(result.sigma2) == len(y) - 13
    assert result.sigma2.index[0] == series.index[13]
