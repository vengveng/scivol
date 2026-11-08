"""Fits with some parameters held at given values: fit(fixed=...)."""
from __future__ import annotations

import numpy as np
import pytest

from scivol import ARMA, GARCH, GJRGARCH, StudentT, stats

NAN = np.nan


@pytest.fixture(scope="module")
def returns():
    return (GARCH(2, 1) + StudentT()).simulate(3000, [0.05, 0.08, 0.0, 0.88, 0.15], seed=4).data


def test_a_lag_fixed_at_zero_is_the_smaller_model(returns):
    fixed = (GARCH(2, 1) + StudentT()).fit(returns, fixed={"alpha[2]": 0.0})
    smaller = (GARCH(1, 1) + StudentT()).fit(returns)
    assert fixed.success and fixed.fixed == {"alpha[2]": 0.0}
    estimated = fixed.params.drop("alpha[2]")
    np.testing.assert_allclose(estimated.to_numpy(), smaller.params.to_numpy(), rtol=1e-6)
    assert fixed.log_likelihood == pytest.approx(smaller.log_likelihood, rel=1e-12)
    assert fixed.n_estimated == 4 and fixed.bic == pytest.approx(smaller.bic, rel=1e-12)
    # The fixed zero is part of the model, not a binding bound: ordinary standard errors.
    assert fixed.fit_info.inference_status == "available"
    np.testing.assert_allclose(fixed.std_errors.drop("alpha[2]").to_numpy(), smaller.std_errors.to_numpy(), rtol=1e-4)
    assert np.isnan(fixed.std_errors["alpha[2]"])


def test_inference_is_the_free_block_of_the_information(returns):
    result = (GARCH(1, 1) + StudentT()).fit(returns, fixed={"eta": 0.2})
    free = [n for n in result.params.index if n != "eta"]
    hessian = result.hessian.loc[free, free].to_numpy()
    np.testing.assert_allclose(result.std_errors[free].to_numpy(), np.sqrt(np.diag(np.linalg.inv(hessian))),
                               rtol=1e-8)
    assert np.all(result.cov_matrix.loc["eta"].to_numpy() == 0.0)
    # A local maximum over the free entries, with the fixed one at its value exactly.
    assert result.params["eta"] == 0.2
    gradient = np.asarray(result._native_fit.gradient)
    assert np.max(np.abs(gradient[:3])) < 1e-4 * len(returns) and abs(gradient[3]) > 1.0


def test_omega_fixed_at_zero_gives_an_ewma(returns):
    result = GARCH(1, 1, stationarity="integrated").fit(returns, fixed={"omega": 0.0})
    assert result.success and result.params["omega"] == 0.0
    assert result.params["alpha[1]"] + result.params["beta[1]"] == pytest.approx(1.0, abs=1e-10)
    summary = str(result.summary())
    assert "fixed" in summary and "In model domain:                    Yes" in summary
    # The model can be evaluated there too.
    state = GARCH(1, 1, stationarity="integrated").fix(returns, result.params)
    assert state.log_likelihood == pytest.approx(result.log_likelihood, rel=1e-12)


@pytest.mark.parametrize("fixed,error", [
    ({"nu": 5}, "unknown parameters nu"),
    ({"eta": NAN}, "must be finite"),
    ({"omega": -1.0}, "omega is fixed at -1, below its bound omega >= 0"),
    ({"alpha[1]": 0.5, "beta[1]": 0.6}, "no point of the model's domain"),
    ({"omega": 0.1, "alpha[1]": 0.1, "beta[1]": 0.8, "eta": 0.1}, "every parameter is fixed"),
])
def test_declarations_that_cannot_be_fitted_raise(returns, fixed, error):
    with pytest.raises(ValueError, match=error):
        (GARCH(1, 1) + StudentT()).fit(returns, fixed=fixed)


def test_fixed_fits_are_solved_in_theta(returns):
    spec = GARCH(1, 1) + StudentT()
    assert spec.fit(returns, fixed={"eta": 0.2}).fit_info.optimization_space.startswith("theta")
    with pytest.raises(ValueError, match="theta coordinates"):
        spec.fit(returns, fixed={"eta": 0.2}, coordinates="z")


def test_qmle_fixes_mean_and_variance_entries_of_density_free_recursions(returns):
    spec = ARMA(1) + GJRGARCH(1, 1) + StudentT()
    result = spec.fit(returns, method="qmle", fixed={"ar[1]": 0.0})
    assert result.success and result.params["ar[1]"] == 0.0
    errors = result.std_errors
    assert np.isnan(errors["ar[1]"]) and np.isfinite(errors.drop("ar[1]")).all()
    with pytest.raises(ValueError, match="cannot fix them"):
        spec.fit(returns, method="qmle", fixed={"eta": 0.2})


def test_rolling_fits_and_the_bootstrap_keep_the_fixed_values(returns):
    spec = GARCH(1, 1) + StudentT()
    windows = spec.fit_rolling(returns, window=2000, step=500, fixed={"eta": 0.2})
    assert all(w.result.params["eta"] == 0.2 and w.result.fixed == {"eta": 0.2} for w in windows)
    boot = spec.fit(returns, fixed={"eta": 0.2}).bootstrap(replications=3, m=800, seed=1, n_jobs=1)
    assert boot.failures == 0 and np.all(boot.draws[:, 3] == 0.2)


def test_likelihood_ratio_tests_the_fixed_values(returns):
    larger = (GARCH(2, 1) + StudentT()).fit(returns)
    by_fixing = stats.likelihood_ratio((GARCH(2, 1) + StudentT()).fit(returns, fixed={"alpha[2]": 0.0}), larger)
    by_dropping = stats.likelihood_ratio((GARCH(1, 1) + StudentT()).fit(returns), larger)
    assert by_fixing.statistic == pytest.approx(by_dropping.statistic, abs=1e-6)
    assert by_fixing.details["law"] == by_dropping.details["law"] == "chi_bar_square"

    free = (GARCH(1, 1) + StudentT()).fit(returns)
    held = (GARCH(1, 1) + StudentT()).fit(returns, fixed={"eta": 0.15})
    test = stats.likelihood_ratio(held, free)
    assert test.details["law"] == "chi_square" and test.df == 1
    assert test.statistic == pytest.approx(2 * (free.log_likelihood - held.log_likelihood), rel=1e-6)
    with pytest.raises(ValueError, match="estimate all its parameters"):
        stats.likelihood_ratio(held, held)
    ewma = GARCH(1, 1, stationarity="integrated").fit(returns, fixed={"omega": 0.0})
    assert stats.likelihood_ratio(ewma, GARCH(1, 1, stationarity="integrated").fit(returns)).status == "unsupported"


def test_a_fixed_mean_coefficient_through_the_kink_faces_is_the_smaller_model():
    from scivol import EGARCH

    y = (ARMA(1) + EGARCH(1, 1) + StudentT()).simulate(2500, [0.0, 0.1, -0.05, 0.1, -0.06, 0.95, 0.12], seed=9).data
    fixed = (ARMA(1) + EGARCH(1, 1) + StudentT()).fit(y, fixed={"const": 0.0})
    smaller = (ARMA(1, constant=False) + EGARCH(1, 1) + StudentT()).fit(y)
    assert fixed.success and smaller.success
    assert fixed.log_likelihood == pytest.approx(smaller.log_likelihood, rel=1e-12)
    np.testing.assert_allclose(fixed.params.drop("const").to_numpy(), smaller.params.to_numpy(), rtol=1e-5, atol=1e-7)
