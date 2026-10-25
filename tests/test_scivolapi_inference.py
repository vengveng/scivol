"""scivolapi inference: per-row scores, OPG and sandwich covariance, QMLE."""

from __future__ import annotations

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
from tests.test_scivolapi_mean import (  # noqa: E402
    assert_close,
    mean_point,
    mean_spec,
    model_for,
    oracle,
    oracle_scores,
    volatility_point,
)

FULL = api.VALUE | api.GRADIENT | api.HESSIAN


@pytest.fixture(scope="module")
def data():
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


@pytest.fixture(scope="module")
def returns():
    """AR(1)-GJR(1,1) returns with skewed, fat-tailed innovations."""
    rng = np.random.default_rng(5)
    n = 3000
    raw = rng.standard_t(7, n) + 0.4 * (rng.exponential(1.0, n) - 1.0)
    z = (raw - raw.mean()) / raw.std()
    y = np.zeros(n)
    h, e = 1.0, 0.0
    for t in range(n):
        h = 0.05 + (0.04 + 0.10 * (e < 0)) * e * e + 0.85 * h
        e = np.sqrt(h) * z[t]
        y[t] = 0.05 + 0.2 * (y[t - 1] if t else 0.0) + e
    return y


SCORE_CASES = [  # (mean, volatility, density, p, q)
    (dict(ar=1, ma=1), "garch", "normal", 1, 1),
    (dict(ar=2, ma=1), "gjr", "student_t", 1, 1),
    (dict(ar=1, exog=2), "egarch", "skew_t", 1, 1),
    (dict(har=(1, 5)), "aparch", "ged", 1, 1),
    (dict(constant=False), "garch", "nig", 1, 2),
]


@pytest.mark.parametrize("mean,volatility,density,p,q", SCORE_CASES)
def test_row_scores_match_the_oracle_and_sum_to_the_gradient(data, mean, volatility, density, p, q):
    y, x = data
    mean = mean_spec(**mean)
    model = model_for(mean, volatility, density, p, q)
    theta = np.array(mean_point(mean) + volatility_point(volatility, p, q, density))
    jet = api.Jet(model, y, np.ascontiguousarray(x[:, : mean.exog]) if mean.exog else None)
    out = jet.theta(theta, api.GRADIENT | api.SCORES)
    assert out["status"] == "ok"
    assert out["scores"].shape == (len(y) - jet.first_scored, model.dimension)
    assert_close(out["scores"], oracle_scores(theta, y, x, mean, volatility, p, q, density), 1e-11)
    np.testing.assert_allclose(out["scores"].sum(axis=0), out["gradient"], rtol=1e-12, atol=1e-10)
    alone = jet.theta(theta, api.SCORES)  # scores alone run the first-order pass
    np.testing.assert_array_equal(alone["scores"], out["scores"])


def test_scores_are_theta_coordinates_only(data):
    y, _ = data
    model = model_for(mean_spec(constant=False))
    jet = api.Jet(model, y)
    z = model.to_z(np.array(volatility_point("garch", 1, 1, "normal")))
    assert jet.z(z, api.VALUE | api.SCORES)["status"] == "invalid_argument"


def gaussian_fit(y, mean=None, volatility="gjr", density="normal", estimator="mle"):
    model = model_for(mean or mean_spec(constant=False), volatility, density)
    options = api.FitOptions()
    options.estimator = getattr(api.Estimator, estimator)
    return model, api.fit(model, y, options)


def test_mle_reports_the_opg_and_its_sandwich(returns):
    model, r = gaussian_fit(returns, mean_spec(ar=1), "gjr", "student_t")
    assert r.success and r.inference == api.Inference.available and r.estimator == api.Estimator.mle
    out = api.Jet(model, returns).theta(np.array(r.theta), FULL | api.SCORES)
    s = out["scores"]
    np.testing.assert_allclose(r.opg, s.T @ s, rtol=1e-12)
    inverse = np.linalg.inv(out["hessian"])
    np.testing.assert_allclose(r.covariance, inverse, rtol=1e-8)
    np.testing.assert_allclose(r.robust_covariance, inverse @ (s.T @ s) @ inverse, rtol=1e-8)
    np.testing.assert_allclose(r.robust_std_errors, np.sqrt(np.diag(r.robust_covariance)), rtol=1e-14)


def test_gaussian_qmle_is_the_normal_fit_with_the_sandwich(returns):
    _, mle = gaussian_fit(returns, mean_spec(ar=1), "gjr")
    _, qmle = gaussian_fit(returns, mean_spec(ar=1), "gjr", estimator="qmle")
    assert qmle.estimator == api.Estimator.qmle and qmle.success
    np.testing.assert_array_equal(qmle.theta, mle.theta)
    np.testing.assert_allclose(qmle.covariance, mle.robust_covariance, rtol=1e-14)
    np.testing.assert_allclose(qmle.std_errors, mle.robust_std_errors, rtol=1e-14)
    # Skewed, fat tails: the sandwich differs from the inverse information.
    assert np.max(np.abs(qmle.std_errors / mle.std_errors - 1.0)) > 0.05


@pytest.mark.parametrize("volatility,density", [("gjr", "skew_t"), ("aparch", "student_t"), ("garch", "ged")])
def test_staged_qmle_and_its_two_step_covariance(returns, volatility, density):
    mean = mean_spec(ar=1)
    gaussian_model, gaussian = gaussian_fit(returns, mean, volatility)
    model, r = gaussian_fit(returns, mean, volatility, density, estimator="qmle")
    k1, d = gaussian_model.dimension, model.dimension
    assert r.estimator == api.Estimator.qmle and r.success, r.warnings
    assert r.inference == api.Inference.available
    assert [p.name for p in r.phases][-2:] == ["theta:slsqp:shapes", "theta:newton:shapes"]
    # Stage one is the Gaussian fit; stage two maximizes the model's likelihood in the shapes.
    np.testing.assert_array_equal(r.theta[:k1], gaussian.theta)
    out = api.Jet(model, returns).theta(np.array(r.theta), FULL | api.SCORES)
    assert r.log_likelihood == pytest.approx(-out["value"], rel=1e-14)
    assert np.max(np.abs(out["gradient"][k1:])) < 1e-5 * len(returns)

    # Two-step sandwich assembled independently from the jets.
    s1 = api.Jet(gaussian_model, returns).theta(np.array(gaussian.theta), api.SCORES)["scores"]
    psi = np.hstack([s1, out["scores"][:, k1:]])
    a = np.zeros((d, d))
    a[:k1, :k1] = np.array(gaussian.hessian)
    a[k1:, :] = out["hessian"][k1:, :]
    inverse = np.linalg.inv(a)
    expected = inverse @ (psi.T @ psi) @ inverse.T
    np.testing.assert_allclose(r.opg, psi.T @ psi, rtol=1e-12)
    np.testing.assert_allclose(r.covariance, expected, rtol=1e-7, atol=1e-14)
    np.testing.assert_allclose(r.std_errors, np.sqrt(np.diag(expected)), rtol=1e-7)
    np.testing.assert_array_equal(r.robust_covariance, r.covariance)
    # The first stage's uncertainty reaches the shapes: the shape SEs exceed the
    # naive inverse of the shape block alone.
    naive = np.sqrt(np.diag(np.linalg.inv(out["hessian"][k1:, k1:])))
    assert np.all(np.array(r.std_errors[k1:]) > 0.0) and not np.allclose(r.std_errors[k1:], naive, rtol=1e-3)


def test_staged_qmle_is_not_offered_for_egarch(returns):
    with pytest.raises(ValueError, match="EGARCH"):
        gaussian_fit(returns, None, "egarch", "student_t", estimator="qmle")
    _, r = gaussian_fit(returns, None, "egarch", estimator="qmle")
    assert r.estimator == api.Estimator.qmle and r.success


def test_public_qmle_fit(returns):
    import io
    from contextlib import redirect_stdout

    from scivol import ARMA, GJRGARCH, SkewT

    mle = (ARMA(1, 0) + GJRGARCH(1, 1) + SkewT()).fit(returns)
    qmle = (ARMA(1, 0) + GJRGARCH(1, 1) + SkewT()).fit(returns, method="qmle")
    _, native = gaussian_fit(returns, mean_spec(ar=1), "gjr", "skew_t", estimator="qmle")
    assert mle.method == "MLE" and qmle.method == "QMLE"
    np.testing.assert_allclose(qmle.params, native.theta, rtol=1e-10)
    np.testing.assert_allclose(qmle.std_errors, native.std_errors, rtol=1e-6)
    np.testing.assert_allclose(qmle.std_errors_robust, qmle.std_errors, rtol=1e-14)
    assert mle.std_errors_robust is not None and mle.opg.shape == (8, 8)
    assert "shapes" in qmle.fit_info.solver
    with redirect_stdout(io.StringIO()):
        qmle.summary()
