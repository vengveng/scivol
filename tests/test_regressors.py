"""Regressors: their names, where `x` goes, and how fitted models take their columns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scivol import ARMA, GARCH, HAR, Normal, Search, StudentT, InformationCriterion
from scivol import stats

N = 1500


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(41)
    x = pd.DataFrame({"noise": rng.standard_normal(N), "vix": rng.standard_normal(N)})
    spec = ARMA(1, 1) + GARCH(1, 1)
    truth = {"const": 0.05, "ar[1]": 0.3, "noise": 0.0, "vix": 0.4, "ma[1]": 0.2,
             "omega": 0.05, "alpha[1]": 0.08, "beta[1]": 0.88}
    y = spec.simulate(N, truth, x=x, burn=0, seed=5).data
    return y, x


def test_columns_name_the_coefficients(sample):
    y, x = sample
    spec = ARMA(1, 1) + GARCH(1, 1)
    named = spec.fit(y, x=x)
    assert list(named.params.index[:5]) == ["const", "ar[1]", "noise", "vix", "ma[1]"]
    unnamed = spec.fit(y, x=x.to_numpy())
    assert list(unnamed.params.index[2:4]) == ["x[1]", "x[2]"]
    np.testing.assert_allclose(unnamed.params.to_numpy(), named.params.to_numpy(), rtol=1e-12)
    series = (ARMA(1) + GARCH(1, 1)).fit(y, x=x["vix"])
    assert "vix" in series.params.index
    assert "Regressors:  noise, vix (mean)" in str(named.summary())
    assert "Regressors" not in str(spec.fit(y).summary())


def test_a_bare_x_goes_to_the_mean_and_a_mapping_names_the_layer(sample):
    y, x = sample
    spec = ARMA(1) + GARCH(1, 1)
    bare, mapped = spec.fit(y, x=x), spec.fit(y, x={"mean": x})
    np.testing.assert_array_equal(bare.params.to_numpy(), mapped.params.to_numpy())
    with pytest.raises(ValueError, match=r"GARCH\(1,1\) takes no regressors"):
        spec.fit(y, x={"volatility": x})
    with pytest.raises(ValueError, match="unknown layers"):
        spec.fit(y, x={"means": x})
    with pytest.raises(ValueError, match="no component that takes regressors"):
        GARCH(1, 1).fit(y, x=x)


def test_regressor_names_must_not_shadow_parameters(sample):
    y, x = sample
    with pytest.raises(ValueError, match="'omega' is also the name of another parameter"):
        (ARMA(1) + GARCH(1, 1)).fit(y, x=x.rename(columns={"vix": "omega"}))


def test_forecasts_match_frames_by_name_and_arrays_by_position(sample):
    y, x = sample
    result = (ARMA(1, 1) + GARCH(1, 1) + StudentT()).fit(y, x=x)
    future = x.iloc[:4]
    expected = result.forecast(4, x=future).mean
    np.testing.assert_array_equal(result.forecast(4, x=future[["vix", "noise"]]).mean, expected)
    np.testing.assert_array_equal(result.forecast(4, x=future.to_numpy()).mean, expected)
    np.testing.assert_array_equal(result.forecast(4, x={"mean": future}).mean, expected)
    with pytest.raises(ValueError, match="x has no column noise"):
        result.forecast(4, x=future[["vix"]].assign(other=1.0))
    with pytest.raises(ValueError, match=r"need the regressors noise, vix"):
        result.forecast(4)
    with pytest.raises(ValueError, match="the model has no regressors"):
        (ARMA(1) + GARCH(1, 1)).fit(y).forecast(4, x=future)


def test_har_takes_ma_terms_and_regressors(sample):
    y, x = sample
    result = (HAR([1, 5], q=1) + GARCH(1, 1)).fit(y, x=x)
    assert list(result.params.index[:6]) == ["const", "har[1]", "har[5]", "noise", "vix", "ma[1]"]
    assert result.success
    assert result.params["vix"] == pytest.approx(0.4, abs=0.08)


def test_simulation_reads_parameters_by_regressor_name(sample):
    _, x = sample
    spec = ARMA(1) + Normal()
    params = {"const": 0.0, "ar[1]": 0.5, "noise": 0.0, "vix": 1.0, "sigma2": 1e-12}
    path = spec.simulate(N, params, x=x, burn=0, seed=1)
    implied = np.zeros(N)
    implied[0] = path.data[0]
    for t in range(1, N):
        implied[t] = 0.5 * path.data[t - 1] + x["vix"].iloc[t]
    np.testing.assert_allclose(path.data[1:], implied[1:], atol=1e-4)


def test_general_to_specific_drops_any_regressor_by_name(sample):
    y, x = sample
    search = Search(mean=[ARMA.up_to(1, 1)], volatility=[GARCH.up_to(1, 1)], density=[Normal()])
    for data in (x, x.to_numpy()):
        selection = search.fit(y, x=data)
        names = list(selection.selected.params.index)
        kept = "vix" if data is x else "x[2]"
        dropped = "noise" if data is x else "x[1]"
        assert kept in names and dropped not in names
        # The selected model takes its regressor from the full data by name.
        refit = selection.selected_model.fit(y, x=data)
        np.testing.assert_allclose(refit.params.to_numpy(), selection.selected.params.to_numpy(), rtol=1e-6)
        forecast = selection.selected.forecast(2, x=data[:2] if data is not x else data.iloc[:2])
        assert np.all(np.isfinite(forecast.mean))


def test_information_criteria_rank_regressor_subsets(sample):
    y, x = sample
    selection = Search(mean=[ARMA.up_to(1)], volatility=[GARCH(1, 1)]).fit(
        y, x=x, procedure=InformationCriterion("bic", adequacy=False))
    assert "vix" in selection.selected.params.index and "noise" not in selection.selected.params.index


def test_rolling_windows_take_their_own_rows_of_every_form(sample):
    y, x = sample
    spec = ARMA(1) + GARCH(1, 1)
    frames = spec.fit_rolling(y, window=1200, step=300, x=x)
    mapped = spec.fit_rolling(y, window=1200, step=300, x={"mean": x})
    for a, b in zip(frames, mapped):
        assert list(a.result.params.index) == list(b.result.params.index)
        np.testing.assert_array_equal(a.result.params.to_numpy(), b.result.params.to_numpy())
    last = spec.fit(y[300:], x=x.iloc[300:])
    np.testing.assert_allclose(frames[-1].result.params.to_numpy(), last.params.to_numpy(), rtol=1e-5)


def test_likelihood_ratio_between_regressor_subsets(sample):
    y, x = sample
    spec = ARMA(1) + GARCH(1, 1)
    full, reduced = spec.fit(y, x=x), spec.fit(y, x=x[["vix"]])
    test = stats.likelihood_ratio(reduced, full)
    assert test.statistic == pytest.approx(2 * (full.log_likelihood - reduced.log_likelihood), rel=1e-6)
    with pytest.raises(ValueError, match="not a restriction"):
        stats.likelihood_ratio(spec.fit(y, x=x[["noise"]].rename(columns={"noise": "other"})), full)


def test_bootstrap_refits_keep_the_regressor_names(sample):
    y, x = sample
    result = (ARMA(1) + GARCH(1, 1)).fit(y, x=x)
    boot = result.bootstrap(replications=3, m=600, seed=2, n_jobs=1)
    assert boot.failures == 0 and boot.draws.shape == (3, len(result.params))
    assert boot.param_names == tuple(result.params.index)
