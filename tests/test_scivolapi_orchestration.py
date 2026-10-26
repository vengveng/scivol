"""scivolapi under the orchestration layers: per-row terms, OOS losses, auto-selection, batches."""

from __future__ import annotations

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
pd = pytest.importorskip("pandas")
from scivol import ARMA, GARCH, GJRGARCH, Normal, StudentT  # noqa: E402
from scivol._evaluation import oos_losses  # noqa: E402
from scivol.result import EstimationResult  # noqa: E402
from tests.test_scivolapi_mean import mean_point, mean_spec, model_for, volatility_point  # noqa: E402


def simulate(seed, n=1200, ar=0.3):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    h, e = 1.0, 0.0
    for t in range(n):
        h = 0.05 + (0.05 + 0.10 * (e < 0)) * e * e + 0.85 * h
        e = np.sqrt(h) * rng.standard_t(6) * np.sqrt(4 / 6)
        y[t] = 0.1 + ar * (y[t - 1] if t else 0.0) + e
    return y


@pytest.fixture(scope="module")
def returns():
    return simulate(3)


@pytest.mark.parametrize("volatility,density", [("gjr", "skew_t"), ("egarch", "nig"), ("aparch", "ged")])
def test_row_terms_sum_to_the_likelihood(returns, volatility, density):
    mean = mean_spec(ar=1, ma=1)
    model = model_for(mean, volatility, density)
    theta = np.array(mean_point(mean) + volatility_point(volatility, 1, 1, density))
    jet = api.Jet(model, returns)
    out = jet.theta(theta, api.VALUE | api.TERMS)
    assert out["terms"].shape == (len(returns) - jet.first_scored,)
    assert np.sum(out["terms"]) == pytest.approx(out["value"], rel=1e-13)
    np.testing.assert_allclose(jet.z(model.to_z(theta), api.TERMS)["terms"], out["terms"], rtol=1e-12)


def test_oos_losses_are_the_likelihood_terms(returns):
    result = (GJRGARCH(1, 1) + StudentT()).fit(returns, hold_back=10)
    assert isinstance(result, EstimationResult)
    n = len(returns)
    logscore = oos_losses(result, returns, full_x=None, start=10, stop=n, loss="logscore")
    assert np.sum(logscore) == pytest.approx(-result.log_likelihood, rel=1e-12)
    tail = oos_losses(result, returns, full_x=None, start=n - 100, stop=n, loss="qlike")
    e, h = np.asarray(result.resid)[-100:], np.asarray(result.sigma2)[-100:]
    np.testing.assert_allclose(tail, np.log(h) + e * e / h, rtol=1e-12)
    with pytest.raises(ValueError, match="after the rows"):
        oos_losses(result, returns, full_x=None, start=5, stop=n, loss="logscore")


def test_batches_rolling_windows_and_frames():
    index = pd.date_range("2000", periods=900)
    frame = pd.DataFrame({f"s{i}": simulate(10 + i, 900) for i in range(3)}, index=index)
    spec = GARCH(1, 1) + Normal()
    batch = spec.fit_many(frame, n_jobs=1)
    assert set(batch) == {"s0", "s1", "s2"} and not batch.failures
    rolling = spec.fit_rolling(frame.iloc[:, :2], window=600, step=150, n_jobs=1)
    assert len(rolling) == 3 and all(not w.batch.failures for w in rolling)
    for w in rolling:  # warm starts reach the same optimum as cold fits
        cold = spec.fit(frame["s0"].iloc[w.start:w.stop])
        assert w.batch["s0"].log_likelihood == pytest.approx(cold.log_likelihood, abs=1e-6)
    multi = (GARCH(1, 1) + StudentT()).fit(frame, n_jobs=1, show_progress=False)
    assert all(isinstance(r, EstimationResult) for r in multi.values())
    assert multi["s1"].sigma2.index[0] == index[0]


def test_hold_back_rows_supply_lags_and_fix_the_scored_sample():
    index = pd.date_range("2000", periods=600)
    frame = pd.DataFrame({"a": simulate(1, 600), "b": simulate(2, 600)}, index=index)
    batch = (ARMA(2, 0) + GARCH(1, 1) + Normal()).fit_many(frame, n_jobs=1, hold_back=5)
    assert batch["a"].sigma2.index[0] == index[5] and len(batch["a"].sigma2) == 595
    y = frame["a"].to_numpy()
    zero = (GARCH(1, 1) + Normal()).fit(y, hold_back=5)
    ar = (ARMA(2, 0) + GARCH(1, 1) + Normal()).fit(y, hold_back=5)
    assert zero.fit_info.likelihood_n_obs == ar.fit_info.likelihood_n_obs == 595  # comparable criteria
    short = (ARMA(2, 0) + GARCH(1, 1) + Normal()).fit(y)
    assert short.fit_info.likelihood_n_obs == 598  # without hold_back, scoring starts after the lags
