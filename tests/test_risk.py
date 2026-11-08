"""Expected shortfall, tail means, value-at-risk backtests and news impact curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy import stats as sps
from scipy.special import xlogy

from scivol import (APARCH, ARMA, DCC, EGARCH, GARCH, GED, GJRGARCH, GeneralizedHyperbolic, Hyperbolic, Normal,
                    NormalInverseGaussian, SkewT, StudentT, VarianceGamma, stats)

LAWS = [
    Normal().distribution(),
    StudentT().distribution(eta=0.2),
    StudentT().distribution(eta=0.45),
    StudentT().distribution(eta=0.001),
    SkewT().distribution(eta=0.15, lam=-0.4),
    SkewT().distribution(eta=0.3, lam=0.7),
    SkewT().distribution(eta=0.0, lam=0.5),
    GED().distribution(nu=0.6),
    GED().distribution(nu=1.0),
    GED().distribution(nu=5.0),
    GeneralizedHyperbolic().distribution(rho=-0.3, zeta=1.2, **{"lambda": -0.8}),
    NormalInverseGaussian().distribution(rho=0.4, zeta=0.7),
    Hyperbolic().distribution(rho=-0.5, zeta=2.0),
    VarianceGamma().distribution(rho=0.2, **{"lambda": 2.5}),
]
LEVELS = np.array([1e-6, 1e-3, 0.01, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999])


def _quadrature_tail_mean(law, u: float) -> float:
    """E[Z | Z <= q] by adaptive quadrature of z f(z), split where the density has kinks."""
    q = float(law.ppf(u))
    breaks = [0.0]
    if law.name == "SkewT":
        breaks.append(float(law.ppf((1.0 - law.shapes["lam"]) / 2.0)))
    points = sorted(b for b in breaks if b < q) + [q]
    f = lambda z: z * law.pdf(z)
    total = integrate.quad(f, -np.inf, points[0], epsabs=0.0, epsrel=1e-13, limit=400)[0]
    for left, right in zip(points[:-1], points[1:]):
        total += integrate.quad(f, left, right, epsabs=0.0, epsrel=1e-13, limit=400)[0]
    return total / u


@pytest.mark.parametrize("law", LAWS, ids=repr)
def test_tail_means_match_quadrature(law):
    for u, value in zip(LEVELS, law.tail_mean(LEVELS)):
        assert value == pytest.approx(_quadrature_tail_mean(law, u), rel=1e-9, abs=1e-14)


def test_tail_means_close_forms():
    z = sps.norm.ppf(0.01)
    assert Normal().distribution().tail_mean(0.01) == pytest.approx(-sps.norm.pdf(z) / 0.01, rel=1e-14)
    nu, u = 5.0, 0.025
    x = sps.t.ppf(u, nu)
    expected = -np.sqrt((nu - 2) / nu) * (nu + x * x) / (nu - 1) * sps.t.pdf(x, nu) / u
    assert StudentT().distribution(eta=1 / nu).tail_mean(u) == pytest.approx(expected, rel=1e-13)
    # The law has mean zero, so near u = 1 the tail mean is minus the small upper tail's moment.
    law, u = SkewT().distribution(eta=0.2, lam=0.5), 1 - 1e-12
    upper = integrate.quad(lambda z: z * law.pdf(z), float(law.ppf(u)), np.inf, epsabs=0.0, epsrel=1e-12)[0]
    assert law.tail_mean(u) == pytest.approx(-upper / u, rel=1e-6)


def test_expected_shortfall_of_a_forecast():
    y = (GARCH() + StudentT()).simulate(1500, [0.05, 0.08, 0.9, 0.15], seed=4).data
    result = (ARMA(1) + GARCH() + StudentT()).fit(y)
    forecast = result.forecast(3, simulations=20000, seed=1)
    law = result.innovations
    sigma = np.sqrt(forecast.residual_variance[0])
    assert forecast.es(0.01)[0] == pytest.approx(-(forecast.mean[0] + sigma * law.tail_mean(0.01)), rel=1e-13)
    es, var = forecast.es([0.01, 0.05]), forecast.var([0.01, 0.05])
    assert es.shape == (3, 2) and np.all(es > var) and np.all(es[:, 0] > es[:, 1])
    paths = forecast.simulated_paths()[:, 2]
    cutoff = np.quantile(paths, 0.05)
    assert es[2, 1] == pytest.approx(-paths[paths <= cutoff].mean(), rel=1e-14)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        forecast.es(1.0)


def _reference_backtests(returns, var, level, lags=4):
    hits = (returns < -var).astype(int)
    n, x = hits.size, hits.sum()
    kupiec = 2 * (xlogy(x, x / n) + xlogy(n - x, 1 - x / n) - xlogy(x, level) - xlogy(n - x, 1 - level))
    counts = np.array([[np.sum((hits[:-1] == i) & (hits[1:] == j)) for j in (0, 1)] for i in (0, 1)])
    rows = counts.sum(axis=1)
    markov = sum(xlogy(counts[i, j], counts[i, j] / rows[i]) for i in (0, 1) for j in (0, 1))
    ones, zeros = counts[:, 1].sum(), counts[:, 0].sum()
    independence = 2 * (markov - xlogy(ones, ones / (n - 1)) - xlogy(zeros, zeros / (n - 1)))
    coverage = 2 * (markov - xlogy(ones, level) - xlogy(zeros, 1 - level))
    h = hits - level
    design = np.column_stack([np.ones(n - lags), var[lags:], *[h[lags - k:n - k] for k in range(1, lags + 1)]])
    response = h[lags:]
    dq = response @ design @ np.linalg.solve(design.T @ design, design.T @ response) / (level * (1 - level))
    return kupiec, independence, coverage, dq


@pytest.mark.parametrize("scale", [1.0, 1.3, 1.8])
def test_backtests_match_their_definitions(scale):
    rng = np.random.default_rng(int(scale * 10))
    var = 1.5 + 0.5 * rng.random(600)
    returns = scale * rng.standard_normal(600)
    kupiec, independence, coverage, dq = _reference_backtests(returns, var, 0.05)
    k = stats.kupiec(returns, var, 0.05)
    c = stats.christoffersen(returns, var, 0.05)
    d = stats.dynamic_quantile(returns, var, 0.05)
    assert k.statistic == pytest.approx(kupiec, rel=1e-11) and k.df == 1
    assert k.p_value == pytest.approx(sps.chi2.sf(kupiec, 1), rel=1e-10)
    assert k.violations == int(np.sum(returns < -var)) and k.observations == 600
    assert c.statistic == pytest.approx(coverage, rel=1e-11) and c.df == 2
    assert c.independence.statistic == pytest.approx(independence, rel=1e-10, abs=1e-12)
    assert sum(c.transitions.values()) == 599
    assert d.statistic == pytest.approx(dq, rel=1e-10) and d.df == 6 and d.rows == 596


def test_backtests_without_violations():
    returns, var = np.zeros(250), np.ones(250)
    k = stats.kupiec(pd.Series(returns), var, 0.01)
    assert k.violations == 0 and k.statistic == pytest.approx(-2 * 250 * np.log(0.99), rel=1e-14)
    c = stats.christoffersen(returns, var, 0.01)
    assert c.status == "not_estimable" and c.independence.status == "not_estimable"
    assert not c.rejects(0.05)
    assert stats.dynamic_quantile(returns, var, 0.01).status == "not_estimable"


def test_backtest_inputs_are_checked():
    with pytest.raises(ValueError, match="same length"):
        stats.kupiec(np.zeros(10), np.ones(9), 0.01)
    with pytest.raises(ValueError, match="tail probability"):
        stats.christoffersen(np.zeros(10), np.ones(10), 1.0)
    with pytest.raises(ValueError, match="missing or infinite"):
        stats.dynamic_quantile(np.r_[np.nan, np.zeros(20)], np.ones(21), 0.05)


def test_a_rolling_backtest_from_one_step_forecasts():
    y = (GARCH() + StudentT()).simulate(1300, [0.05, 0.08, 0.9, 0.15], seed=8).data
    rolling = (GARCH() + StudentT()).fit_rolling(y[:-1], window=1000, step=50)
    var = np.array([w.result.forecast(1).var(0.05)[0] for w in rolling])
    realized = y[[w.stop for w in rolling]]
    assert stats.kupiec(realized, var, 0.05).status == "ok"


# The interface details the replications ran into

def test_dcc_regressors_name_only_the_series_that_have_them():
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(rng.standard_normal((600, 3)), columns=["a", "b", "c"])
    x = rng.standard_normal((600, 1))
    marginals = [GARCH().fit(returns["a"]), GARCH().fit(returns["b"]), (ARMA(0) + GARCH()).fit(returns["c"], x=x)]
    result = DCC().fit(returns, univariate_results=marginals)
    forecast = result.forecast(2, x={"c": np.zeros((2, 1))}, simulations=200, seed=1)
    assert forecast.mean.shape == (2, 3)
    with pytest.raises(KeyError, match="unknown series"):
        result.forecast(2, x={"d": np.zeros((2, 1))}, simulations=200, seed=1)
    with pytest.raises(ValueError, match=r"series c: Forecasts need the regressors x\[1\]"):
        result.forecast(2, simulations=200, seed=1)
    with pytest.raises(ValueError, match=r"series c: Simulations need the regressors x\[1\]"):
        result.simulate(3)


def test_parameters_outside_the_domain_are_named():
    y = GARCH().simulate(500, [0.1, 0.1, 0.8], seed=1).data
    with pytest.raises(ValueError, match=r"omega > 0 \(it is -0.1\)"):
        GARCH().fix(y, [-0.1, 0.08, 0.9])
    with pytest.raises(ValueError, match=r"eta < 0.5"):
        (GARCH() + StudentT()).fix(y, [0.1, 0.1, 0.8, 0.6])


def test_a_rolling_window_of_one_named_series_gives_its_result():
    y = pd.Series(GARCH().simulate(700, [0.1, 0.1, 0.8], seed=2).data, name="SPX")
    rolling = GARCH().fit_rolling(y, window=500, step=100)
    assert list(rolling[0].batch) == ["SPX"] and rolling[0].result is rolling[0].batch["SPX"]
    frame = pd.DataFrame({"a": y, "b": y})
    with pytest.raises(ValueError, match="several series"):
        GARCH().fit_rolling(frame, window=500, step=200)[0].result


# News impact curves

@pytest.fixture(scope="module")
def leveraged():
    return (GJRGARCH() + StudentT()).simulate(3000, [0.05, 0.04, 0.1, 0.85, 0.15], seed=3).data


def _absolute_moment(law, power=1.0, gamma=0.0):
    return integrate.quad(lambda z: (abs(z) - gamma * z) ** power * law.pdf(z), -np.inf, np.inf, epsrel=1e-12)[0]


E = np.array([-3.0, -1.0, 0.0, 0.5, 2.0])


def test_news_impact_of_gjr_garch_holds_earlier_terms_at_their_expectation(leveraged):
    r = (GJRGARCH(2, 1) + SkewT()).fit(leveraged)
    t = r.params
    law = r.innovations
    kappa = integrate.quad(lambda z: z * z * law.pdf(z), -np.inf, 0.0, epsrel=1e-12)[0]
    for given, v in ((None, r.unconditional_variance), (2.0, 2.0)):
        expected = (t["omega"] + (t["alpha[1]"] + t["gamma[1]"] * (E < 0)) * E**2
                    + (t["alpha[2]"] + kappa * t["gamma[2]"] + t["beta[1]"]) * v)
        np.testing.assert_allclose(r.news_impact(E, variance=given), expected, rtol=1e-12)


def test_news_impact_of_egarch_and_aparch(leveraged):
    r = (EGARCH() + SkewT()).fit(leveraged)
    t = r.params
    v = np.exp(t["omega"] / (1 - t["beta[1]"]))
    z = E / np.sqrt(v)
    m = _absolute_moment(r.innovations)
    expected = np.exp(t["omega"] + t["alpha[1]"] * (np.abs(z) - m) + t["gamma[1]"] * z + t["beta[1]"] * np.log(v))
    np.testing.assert_allclose(r.news_impact(E), expected, rtol=1e-10)

    r = (APARCH() + StudentT()).fit(leveraged)
    t = r.params
    d = t["delta"]
    level = t["omega"] / (1 - t["alpha[1]"] * _absolute_moment(r.innovations, d, t["gamma[1]"]) - t["beta[1]"])
    expected = (t["omega"] + t["alpha[1]"] * (np.abs(E) - t["gamma[1]"] * E) ** d + t["beta[1]"] * level) ** (2 / d)
    np.testing.assert_allclose(r.news_impact(E), expected, rtol=1e-9)


def test_news_impact_needs_a_stationary_level_or_a_given_variance(leveraged):
    fixed = GARCH(stationarity="integrated").fix(leveraged, [0.05, 0.1, 0.9])
    with pytest.raises(ValueError, match="pass the lagged variance"):
        fixed.news_impact(1.0)
    assert fixed.news_impact(1.0, variance=2.0) == pytest.approx(0.05 + 0.1 + 0.9 * 2.0, rel=1e-14)
    series = GARCH().fit(leveraged).news_impact(pd.Series([-1.0, 1.0], index=["down", "up"]))
    assert list(series.index) == ["down", "up"] and series["down"] == series["up"]
