"""Normalized likelihoods must agree with the reported conditional distribution."""

import numpy as np
import pytest

from scivol import ARX, HARX, GARCH, GJRGARCH, EGARCH, Normal


@pytest.mark.parametrize("mean_factory", (lambda: ARX(1), lambda: HARX((1, 3))))
@pytest.mark.parametrize("vol_factory", (GARCH, GJRGARCH, EGARCH))
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_joint_normal_likelihood_is_normalized(mean_factory, vol_factory, p, q):
    rng = np.random.default_rng(6906)
    data = rng.normal(size=180)
    x = rng.normal(size=(len(data), 1))
    mean = mean_factory()
    mean.set_n_exog(1)
    volatility = vol_factory(p, q)
    spec = mean + volatility + Normal()
    mean_parameters = np.full(mean.n_params, 0.04)
    if vol_factory is EGARCH:
        vol_parameters = np.r_[
            -0.1, np.full(p, 0.08 / p), np.full(p, -0.02 / p), np.full(q, 0.8 / q)
        ]
    elif vol_factory is GJRGARCH:
        vol_parameters = np.r_[
            0.1, np.full(p, 0.08 / p), np.full(p, 0.04 / p), np.full(q, 0.8 / q)
        ]
    else:
        vol_parameters = np.r_[0.1, np.full(p, 0.08 / p), np.full(q, 0.8 / q)]
    theta = np.r_[mean_parameters, vol_parameters]
    state = spec.filter(data, theta, x=x)
    expected = -0.5 * np.sum(
        np.log(2.0 * np.pi) + np.log(state.sigma2) + state.residuals**2 / state.sigma2
    )
    assert spec.loglikelihood(data, theta, x=x) == pytest.approx(expected, rel=2e-13)
    fixed = spec.fix(data, theta, x=x)
    assert fixed.log_likelihood == pytest.approx(expected, rel=2e-13)


@pytest.mark.parametrize("vol_factory", (GARCH, GJRGARCH, EGARCH))
def test_joint_normal_fit_reports_normalized_information_criteria(vol_factory):
    rng = np.random.default_rng(591)
    data = rng.normal(size=300)
    x = rng.normal(size=(len(data), 1))
    result = (ARX(1) + vol_factory(1, 1) + Normal()).fit(data, x=x)
    expected = -0.5 * np.sum(
        np.log(2.0 * np.pi) + np.log(result.sigma2) + result.resid**2 / result.sigma2
    )
    assert result.log_likelihood == pytest.approx(expected, rel=2e-12)
    hold_back = result.fit_info.effective_hold_back
    np.testing.assert_array_equal(result.data, data[hold_back:])
    np.testing.assert_allclose(result.filter().sigma2, result.sigma2, rtol=2e-12)
    k = len(result.params)
    assert result.aic == pytest.approx(-2.0 * expected + 2.0 * k, rel=2e-12)
    assert result.bic == pytest.approx(
        -2.0 * expected + np.log(result.n_obs) * k, rel=2e-12
    )
