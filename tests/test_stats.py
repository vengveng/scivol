"""The public tests (scivol.stats), named results and innovation distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from scivol import ARX, DCC, GARCH, GJRGARCH, SkewT, StudentT, stats


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return (GARCH(1, 1) + SkewT()).simulate(3000, [0.05, 0.08, 0.9, 0.15, -0.2], seed=4).data


@pytest.fixture(scope="module")
def fits(data):
    return {"normal": GARCH(1, 1).fit(data), "student": (GARCH(1, 1) + StudentT()).fit(data),
            "skew": (GARCH(1, 1) + SkewT()).fit(data)}


# Named results

def test_estimates_and_inference_are_named(fits):
    r = fits["skew"]
    names = ["omega", "alpha[1]", "beta[1]", "eta", "lam"]
    for values in (r.params, r.std_errors, r.std_errors_robust, r.tvalues, r.pvalues):
        assert isinstance(values, pd.Series) and list(values.index) == names
    assert isinstance(r.params["lam"], float)
    assert list(r.cov_matrix.columns) == names and list(r.hessian.index) == names
    table = r.table()
    np.testing.assert_allclose(table["t_value"], r.params / r.std_errors, rtol=1e-12)
    assert table.loc["beta[1]", "p_value"] == 0.0  # underflows a double


def test_a_named_series_passes_back_in_by_name(data, fits):
    r = fits["normal"]
    shuffled = r.params.iloc[::-1]
    assert GARCH(1, 1).loglikelihood(data, shuffled) == pytest.approx(r.log_likelihood, rel=1e-12)


# Tests of series

def test_jarque_bera_matches_scipy(data):
    test = stats.jarque_bera(data[:300])
    reference = sps.jarque_bera(data[:300])
    assert test.statistic == pytest.approx(reference.statistic, rel=1e-12)
    assert test.p_value == pytest.approx(reference.pvalue, rel=1e-10)
    assert test.skewness == pytest.approx(sps.skew(data[:300]), rel=1e-12)
    assert test.kurtosis == pytest.approx(sps.kurtosis(data[:300], fisher=False), rel=1e-12)


def test_series_tests_return_one_result_type(data):
    for test in (stats.ljung_box(data, 10), stats.arch_lm(data, 4), stats.dgt(sps.norm.cdf(data / data.std())),
                 stats.berkowitz(sps.norm.cdf(data / data.std())), stats.sign_bias(data)):
        assert isinstance(test, stats.TestResult) and test.status in ("ok", "probability_underflow")
        assert test.rejects(1.0) and np.isfinite(test.log_p_value)
    assert stats.ljung_box(data, 10).box_pierce.df == 10
    with pytest.raises(AttributeError):
        stats.arch_lm(data, 4).not_a_detail


def test_missing_values_are_refused():
    with pytest.raises(ValueError, match="dropna"):
        stats.ljung_box(pd.Series([0.1, np.nan, 0.3, 0.2] * 10), 2)


def test_model_confidence_set_labels_the_columns():
    rng = np.random.default_rng(5)
    losses = pd.DataFrame({"a": rng.chisquare(1, 800), "b": 3 * rng.chisquare(1, 800)})
    mcs = stats.model_confidence_set(losses, seed=1, draws=499)
    assert mcs.included == ("a",) and mcs.eliminated == ("b",)
    assert list(mcs.p_values.index) == ["a", "b"]


# Tests of fitted models

def test_wald_matches_the_quadratic_form(fits):
    r = fits["skew"]
    test = stats.wald(r, ["alpha[1]", "lam"], [0.1, 0.0], robust=True)
    d = r.params[["alpha[1]", "lam"]].to_numpy() - [0.1, 0.0]
    v = r.cov_robust.loc[["alpha[1]", "lam"], ["alpha[1]", "lam"]].to_numpy()
    assert test.statistic == pytest.approx(d @ np.linalg.solve(v, d), rel=1e-10) and test.df == 2


def test_wald_refuses_a_value_on_a_bound(fits):
    assert stats.wald(fits["student"], "eta").status == "at_boundary"
    with pytest.raises(ValueError, match="unknown"):
        stats.wald(fits["student"], "nu")


def test_likelihood_ratio_uses_the_restrictions_law(fits):
    n, t, s = fits["normal"], fits["student"], fits["skew"]
    symmetry = stats.likelihood_ratio(t, s)
    assert symmetry.law == "chi_square" and symmetry.df == 1
    assert symmetry.statistic == pytest.approx(2 * (s.log_likelihood - t.log_likelihood), rel=1e-9)
    normality = stats.likelihood_ratio(n, t)
    assert normality.law == "chi_bar_square" and normality.weights == pytest.approx((0.5, 0.5))
    assert normality.p_value == pytest.approx(0.5 * sps.chi2.sf(normality.statistic, 1), rel=1e-8)
    both = stats.likelihood_ratio(n, s)  # a chain of two restrictions
    assert both.statistic == pytest.approx(2 * (s.log_likelihood - n.log_likelihood), rel=1e-9)
    with pytest.raises(ValueError, match="not a restriction"):
        stats.likelihood_ratio(s, n)


def test_likelihood_ratio_compares_mean_lags_on_one_sample(data, fits):
    ar = (ARX(1) + GARCH(1, 1)).fit(data)
    test = stats.likelihood_ratio(fits["normal"], ar)
    assert test.df == 2 and test.restricted_log_likelihood != fits["normal"].log_likelihood
    assert test.statistic == pytest.approx(2 * (test.log_likelihood - test.restricted_log_likelihood), rel=1e-9)


def test_model_tests_take_a_result(fits):
    r = fits["skew"]
    d = r.diagnostics()
    assert stats.li_mak(r, 10).statistic == pytest.approx(d.li_mak.statistic, rel=1e-12)
    assert stats.absolute_li_mak(r, 5).df == 5 and stats.absolute_sign_bias(r).df == 3
    np.testing.assert_allclose(d.pit, r.pit, rtol=1e-12)


# Distributions

def test_distribution_from_a_fit_and_from_shapes(fits):
    r = fits["skew"]
    law = r.innovations
    direct = SkewT().distribution(eta=r.params["eta"], lam=r.params["lam"])
    assert law.skewness == pytest.approx(direct.skewness, rel=1e-14)
    z = np.linspace(-3, 3, 7)
    np.testing.assert_allclose(law.ppf(law.cdf(z)), z, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(law.cdf(z) + law.sf(z), 1.0, rtol=1e-14)
    assert isinstance(law.pdf(pd.Series(z)), pd.Series) and isinstance(law.cdf(0.0), float)
    assert law.skewness < 0 and law.mean == 0.0 and law.variance == 1.0


def test_divergent_moments():
    t = StudentT().distribution(eta=0.3)
    assert t.kurtosis == np.inf and t.moment(3) == 0.0
    assert np.isnan(StudentT().distribution(eta=0.4).moment(3))
    assert StudentT().distribution(eta=0.2).kurtosis == pytest.approx(9.0, rel=1e-12)


def test_distribution_shapes_are_validated():
    with pytest.raises(TypeError, match="takes eta, lam"):
        SkewT().distribution(nu=5)
    with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
        StudentT().distribution(eta=0.5)


def test_dcc_takes_distributions_as_marginals():
    eps = np.random.default_rng(17).standard_t(8, size=(800, 3)) * np.sqrt(6 / 8)
    result = DCC(distribution="studentt").fit_from_residuals(eps, marginals=[StudentT().distribution(eta=1 / 8)] * 3)
    assert list(result.params.index) == ["a[1]", "b[1]", "eta"]
    assert result.persistence == pytest.approx(result.params["a[1]"] + result.params["b[1]"])
    assert list(result.std_errors.index) == list(result.params.index)
    assert result.cov_matrix is None or list(result.cov_matrix.index) == list(result.params.index)
