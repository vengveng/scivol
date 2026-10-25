"""scivolapi simulation and forecasting: continuations, series, exact forecasts."""

from __future__ import annotations

import math

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
from tests.test_scivolapi_mean import mean_point, mean_spec, model_for, volatility_point  # noqa: E402


@pytest.fixture(scope="module")
def data():
    """AR(1)-GJR(1,1) returns with t innovations and one exogenous column."""
    rng = np.random.default_rng(3)
    n = 400
    x = rng.standard_normal((n, 1))
    y = np.zeros(n)
    h, e = 1.0, 0.0
    for t in range(n):
        h = 0.05 + (0.05 + 0.10 * (e < 0)) * e * e + 0.85 * h
        e = np.sqrt(h) * rng.standard_t(6) * np.sqrt(4 / 6)
        y[t] = 0.1 + 0.3 * (y[t - 1] if t else 0.0) + 0.4 * x[t, 0] + e
    return y, x


CASES = [  # (mean, volatility, density, p, q)
    (dict(ar=1, ma=1), "garch", "normal", 1, 1),
    (dict(ar=2, exog=1), "gjr", "student_t", 1, 1),
    (dict(har=(1, 5), ma=1), "egarch", "ged", 1, 2),
    (dict(ar=1), "aparch", "skew_t", 2, 1),
    (dict(constant=False), "gjr", "nig", 2, 2),
]


def setup(data, mean, volatility, density, p, q):
    y, x = data
    mean = mean_spec(**mean)
    model = model_for(mean, volatility, density, p, q)
    theta = np.array(mean_point(mean) + volatility_point(volatility, p, q, density))
    assert model.feasible(theta)
    rows = np.ascontiguousarray(x[:, : mean.exog]) if mean.exog else None
    return model, theta, mean, y, rows


def extend(y, rows, model, theta, value, row):
    """The jet on the sample with one more observation; `value` keeps mean(y^2), so the seed is unchanged."""
    y1 = np.append(y, value)
    x1 = None if rows is None else np.vstack([rows, row])
    return api.Jet(model, y1, x1).theta(theta, api.PATHS)


@pytest.mark.parametrize("mean,volatility,density,p,q", CASES)
def test_one_step_forecast_is_the_likelihood_recursion(data, mean, volatility, density, p, q):
    model, theta, mean, y, rows = setup(data, mean, volatility, density, p, q)
    future = np.ones((1, mean.exog)) * 0.7 if mean.exog else None
    f = api.forecast(model, theta, y, 1, rows, future)
    value = np.sqrt(np.mean(y * y))  # appending it leaves mean(y^2), the variance seed, unchanged
    out = extend(y, rows, model, theta, value, None if future is None else future[0])
    assert f["residual_variance"][0] == pytest.approx(out["variance"][-1], rel=1e-14)
    assert f["mean"][0] == pytest.approx(value - out["residual"][-1], rel=1e-13, abs=1e-15)
    assert f["variance"][0] == pytest.approx(f["residual_variance"][0], rel=1e-15)


@pytest.mark.parametrize("mean,volatility,density,p,q", CASES)
def test_a_continuation_is_the_likelihood_recursion(data, mean, volatility, density, p, q):
    """The jet on the sample extended by a generated path reproduces the path.

    The extended sample has a different variance seed, whose effect has decayed
    to rounding after 400 rows.
    """
    model, theta, mean, y, rows = setup(data, mean, volatility, density, p, q)
    horizon = 4
    future = np.random.default_rng(0).standard_normal((horizon, mean.exog)) if mean.exog else None
    z = np.array([[0.3, -1.2, 2.0, -0.4]])
    s = api.simulate(model, theta, y, z, rows, future)
    np.testing.assert_allclose(s["residual"], np.sqrt(s["variance"]) * z, rtol=1e-15)
    np.testing.assert_allclose(s["y"], s["mean"] + s["residual"], rtol=1e-15)
    x1 = None if rows is None else np.vstack([rows, future])
    out = api.Jet(model, np.append(y, s["y"][0]), x1).theta(theta, api.PATHS)
    np.testing.assert_allclose(out["variance"][-horizon:], s["variance"][0], rtol=1e-12)
    np.testing.assert_allclose(out["residual"][-horizon:], s["residual"][0], rtol=1e-11)


def test_paths_continue_from_one_filtered_state(data):
    model, theta, mean, y, rows = setup(data, dict(ar=1, ma=1), "gjr", "student_t", 1, 1)
    z = np.random.default_rng(1).standard_normal((3, 6))
    s = api.simulate(model, theta, y, z)
    alone = [api.simulate(model, theta, y, z[i:i + 1]) for i in range(3)]
    for i in range(3):
        for key in ("y", "mean", "residual", "variance"):
            np.testing.assert_array_equal(s[key][i], alone[i][key][0])
    assert np.all(s["variance"][:, 0] == s["variance"][0, 0])  # the first row is known


def garch_reference(theta, y, z):
    """GARCH(1,1) without a mean, by hand: filter with h_0 = mean(y^2), then continue."""
    omega, alpha, beta = theta
    h = np.mean(y * y)
    for t in range(1, len(y)):
        h = omega + alpha * y[t - 1] ** 2 + beta * h
    e_last, out = y[-1], []
    for zk in z:
        h = omega + alpha * e_last ** 2 + beta * h
        e_last = np.sqrt(h) * zk
        out.append((h, e_last))
    return np.array(out)


def test_garch_continuation_matches_a_hand_recursion(data):
    y, _ = data
    model = model_for(mean_spec(constant=False))
    theta = np.array([0.05, 0.1, 0.85])
    z = np.array([[0.5, -2.0, 1.0, 0.1, -0.3]])
    s = api.simulate(model, theta, y, z)
    reference = garch_reference(theta, y, z[0])
    np.testing.assert_allclose(s["variance"][0], reference[:, 0], rtol=1e-13)
    np.testing.assert_allclose(s["residual"][0], reference[:, 1], rtol=1e-13)


def test_garch_forecast_has_the_closed_form(data):
    y, _ = data
    model = model_for(mean_spec(constant=False))
    omega, alpha, beta = theta = np.array([0.05, 0.1, 0.85])
    f = api.forecast(model, theta, y, 30)
    level = omega / (1 - alpha - beta)
    k = np.arange(30)
    np.testing.assert_allclose(f["residual_variance"],
                               level + (alpha + beta) ** k * (f["residual_variance"][0] - level), rtol=1e-12)
    np.testing.assert_array_equal(f["mean"], 0.0)


def test_arma_forecast_variance_uses_the_psi_weights(data):
    y, _ = data
    model = model_for(mean_spec(ar=1, ma=1))
    c, phi, ma, omega, alpha, beta = theta = np.array([0.1, 0.5, 0.3, 0.05, 0.1, 0.85])
    f = api.forecast(model, theta, y, 6)
    psi = np.array([1.0] + [(phi + ma) * phi ** (j - 1) for j in range(1, 6)])
    expected = [np.sum(psi[: k + 1] ** 2 * f["residual_variance"][k::-1]) for k in range(6)]
    np.testing.assert_allclose(f["variance"], expected, rtol=1e-13)
    mean = [f["mean"][0]]
    for _ in range(5):
        mean.append(c + phi * mean[-1])
    np.testing.assert_allclose(f["mean"], mean, rtol=1e-13)


@pytest.mark.parametrize("volatility,density,draw", [
    ("gjr", "student_t", lambda rng, n: rng.standard_t(6, n) * np.sqrt(4 / 6)),
    ("egarch", "normal", lambda rng, n: rng.standard_normal(n)),
    ("egarch", "ged", None),
])
def test_exact_forecasts_agree_with_simulated_averages(data, volatility, density, draw):
    model, theta, mean, y, rows = setup(data, dict(ar=1), volatility, density, 1, 1)
    rng = np.random.default_rng(7)
    paths, horizon = 200_000, 8
    if draw is None:  # GED(1.5) by its gamma representation
        nu = theta[-1]
        g = rng.gamma(1 / nu, 1.0, paths * horizon) ** (1 / nu)
        z = g * rng.choice([-1.0, 1.0], paths * horizon)
        z /= np.sqrt(math.gamma(3 / nu) / math.gamma(1 / nu))
    else:
        z = draw(rng, paths * horizon)
    z = z.reshape(paths, horizon)
    f = api.forecast(model, theta, y, horizon)
    s = api.simulate(model, theta, y, z)
    se = s["variance"].std(0) / np.sqrt(paths)
    assert np.all(np.abs(s["variance"].mean(0) - f["residual_variance"]) < 5 * se + 1e-11 * f["residual_variance"])
    se_y = s["y"].std(0) / np.sqrt(paths)
    assert np.all(np.abs(s["y"].mean(0) - f["mean"]) < 5 * se_y)
    np.testing.assert_allclose(s["y"].var(0), f["variance"], rtol=0.03)


def test_egarch_forecasts_beyond_one_step_need_finite_exponential_moments(data):
    model, theta, mean, y, rows = setup(data, dict(ar=1), "egarch", "student_t", 1, 1)
    assert api.forecast(model, theta, y, 1)["residual_variance"][0] > 0
    with pytest.raises(ValueError, match="does not exist"):
        api.forecast(model, theta, y, 2)


def test_aparch_forecasts_beyond_one_step_are_path_averages(data):
    model, theta, mean, y, rows = setup(data, dict(ar=1), "aparch", "normal", 1, 1)
    with pytest.raises(ValueError, match="innovations"):
        api.forecast(model, theta, y, 3)
    z = np.random.default_rng(2).standard_normal((1000, 3))
    f = api.forecast(model, theta, y, 3, z=z)
    s = api.simulate(model, theta, y, z)
    assert f["simulated"] and not api.forecast(model, theta, y, 1)["simulated"]
    np.testing.assert_allclose(f["residual_variance"][1:], s["variance"].mean(0)[1:], rtol=1e-13)
    assert f["residual_variance"][0] == s["variance"][0, 0]


def test_series_start_at_the_stationary_values():
    model = model_for(mean_spec(ar=1))
    c, phi, omega, alpha, beta = theta = np.array([0.2, 0.5, 0.05, 0.1, 0.85])
    z = np.random.default_rng(4).standard_normal((1, 200_000))
    s = api.simulate_series(model, theta, z)
    assert s["variance"][0, 0] == pytest.approx(omega / (1 - alpha - beta), rel=1e-14)
    assert s["mean"][0, 0] == pytest.approx(c / (1 - phi), rel=1e-14)  # its lag sits at the unconditional mean
    assert np.mean(s["y"]) == pytest.approx(c / (1 - phi), rel=0.05)
    assert np.mean(s["residual"] ** 2) == pytest.approx(omega / (1 - alpha - beta), rel=0.05)
    with pytest.raises(ValueError, match="domain"):
        api.simulate_series(model, np.array([0.2, 0.5, 0.05, 0.3, 0.75]), z)


# The Python layer: results and fixed-parameter evaluation.

from scivol import _evaluation as native_eval  # noqa: E402


def test_a_fitted_result_filters_and_forecasts_natively(data):
    from scivol import ARX, GJRGARCH, StudentT

    y, x = data
    spec = ARX(lags=2) + GJRGARCH(1, 1) + StudentT()
    result = spec.fit(y, x=x)
    state = result.filter()
    np.testing.assert_allclose(state.residuals, result.resid, rtol=1e-14)
    np.testing.assert_allclose(state.sigma2, result.sigma2, rtol=1e-14)
    assert state.log_likelihood == pytest.approx(result.log_likelihood, rel=1e-13)
    future = np.full((5, 1), 0.2)
    fc = result.forecast(5, x=future)
    model = result._native_model
    out = api.forecast(model, np.array(result.params), y, 5, np.ascontiguousarray(x), future)
    np.testing.assert_array_equal(fc.mean, out["mean"])
    np.testing.assert_array_equal(fc.residual_variance, out["residual_variance"])
    assert fc.distribution == "StudentT" and fc.nu == pytest.approx(result.params[-1])
    assert fc.quantile(0.05).shape == (5,)
    with pytest.raises(ValueError, match="exogenous"):
        result.forecast(5)


def test_fixed_parameter_evaluation_matches_the_fit(data):
    from scivol import ARMA, EGARCH, GED

    y, _ = data
    spec = ARMA(1, 1) + EGARCH(1, 1) + GED()
    result = spec.fit(y)
    theta = np.array(result.params)
    assert native_eval.loglikelihood(spec, y, theta) == pytest.approx(result.log_likelihood, rel=1e-14)
    np.testing.assert_allclose(native_eval.hessian(spec, y, theta), -np.array(result.hessian), rtol=1e-12)
    assert np.max(np.abs(native_eval.score(spec, y, theta))) < 1e-3
    state = native_eval.filter_state(spec, y, theta)
    np.testing.assert_allclose(state.sigma2, result.sigma2, rtol=1e-14)
    np.testing.assert_allclose(state.mean + state.residuals, y[1:], rtol=1e-14)


def test_simulated_series_are_reproducible_and_stationary():
    from scivol import GARCH, Normal

    spec = GARCH(1, 1) + Normal()
    theta = np.array([0.05, 0.1, 0.85])
    a = native_eval.simulate(spec, 50_000, theta, seed=3)
    b = native_eval.simulate(spec, 50_000, theta, seed=3)
    np.testing.assert_array_equal(a.data, b.data)
    np.testing.assert_allclose(a.residuals, np.sqrt(a.sigma2) * a.innovations, rtol=1e-15)
    assert np.mean(a.data ** 2) == pytest.approx(1.0, rel=0.1)


@pytest.mark.parametrize("volatility,density,shape", [
    ("egarch", "normal", []),
    ("egarch", "ged", [1.3]),
    ("egarch", "nig", [-0.3, 1.2]),
    ("egarch", "hyperbolic", [0.4, 0.8]),
    ("egarch", "generalized_hyperbolic", [-0.3, 1.4, -0.7]),
    ("egarch", "variance_gamma", [-0.3, 2.5]),
    ("gjr", "skew_t", [6.0, -0.3]),
])
def test_drawn_innovations_follow_the_likelihoods_density(data, volatility, density, shape):
    """Exact forecasts (exponential and negative moments) against averages over paths drawn by
    inverting the native distribution function: the two must describe the same density."""
    from scivol.components.density import GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, Normal, \
        SkewT, VarianceGamma
    from scivol.components.vol import EGARCH, GJRGARCH

    y, _ = data
    names = {"normal": Normal, "ged": GED, "nig": NormalInverseGaussian, "hyperbolic": Hyperbolic,
             "generalized_hyperbolic": GeneralizedHyperbolic, "variance_gamma": VarianceGamma, "skew_t": SkewT}
    spec = (EGARCH if volatility == "egarch" else GJRGARCH)(1, 1) + names[density]()
    model = model_for(mean_spec(constant=False), volatility, density)
    theta = np.array(volatility_point(volatility, 1, 1, "normal") + shape)
    paths, horizon = 100_000, 4
    z = native_eval.innovations(model.innovation_density(theta), paths, horizon, 11)
    assert abs(z.mean()) < 5 / np.sqrt(z.size) and z.var() == pytest.approx(1.0, abs=0.02)
    f = api.forecast(model, theta, y, horizon)
    s = api.simulate(model, theta, y, z)
    se = s["variance"].std(0) / np.sqrt(paths)
    assert np.all(np.abs(s["variance"].mean(0) - f["residual_variance"]) < 5 * se + 1e-11 * f["residual_variance"])
