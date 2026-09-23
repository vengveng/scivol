"""Standalone ARMA profile variance uses the kernel's scored sample."""

import numpy as np
import pytest

from scivol import ARMA, Normal, GED
from scivol._evaluation import _compute_arma_resid_sigma2, _filter_arma
from scivol._kernels.arma_normal import _build


@pytest.mark.parametrize("p,q", [(0, 0), (1, 1), (3, 0), (0, 3), (2, 3)])
def test_profile_variance_agrees_across_fit_and_filter(p, q):
    data = np.random.default_rng(930).normal(size=250)
    result = _build(p, q).fit(data)
    spec = ARMA(p, q) + Normal()
    state = _filter_arma(
        spec, data, result.params, compute_score=False, compute_hessian=False
    )
    _, residuals, variance = _compute_arma_resid_sigma2(ARMA(p, q), data, result.params)
    scored = residuals[max(p, q, 1) :]
    expected = scored @ scored / len(scored)
    np.testing.assert_allclose(result.sigma2, expected, rtol=1e-13)
    np.testing.assert_allclose(state.sigma2, expected, rtol=1e-13)
    np.testing.assert_allclose(variance, expected, rtol=1e-13)
    np.testing.assert_allclose(result._resid, residuals, rtol=1e-13)
    expected_nll = 0.5 * len(scored) * (np.log(2 * np.pi * expected) + 1)
    np.testing.assert_allclose(-result.log_likelihood, expected_nll, rtol=1e-13)
    np.testing.assert_allclose(-state.log_likelihood, expected_nll, rtol=1e-13)


@pytest.mark.parametrize("density", [Normal, GED])
@pytest.mark.parametrize("log_mode", [False, True])
def test_legacy_arma_reporting_matches_canonical_additive_inference(density, log_mode):
    from scivol._evaluation import filter_spec
    from scivol._kernels import arma_normal, arma_ged

    data = np.random.default_rng(392).standard_t(6, size=300)
    module = arma_normal if density is Normal else arma_ged
    result = module._build(2, 1).fit(data, log_mode=log_mode)
    state = filter_spec(
        ARMA(2, 1) + density(),
        data,
        result.params,
        compute_score=True,
        compute_hessian=True,
    )
    np.testing.assert_allclose(result.log_likelihood, state.log_likelihood, rtol=1e-12)
    np.testing.assert_allclose(result.hessian, -state.hessian, rtol=2e-6, atol=2e-6)
    assert result.fit_info.likelihood_n_obs == 298
    if result.cov_matrix is not None:
        np.testing.assert_allclose(
            result.cov_matrix @ result.hessian, np.eye(len(result.params)), atol=1e-10
        )
    else:
        assert np.linalg.eigvalsh(result.hessian).min() <= 0


def test_profile_variance_requires_a_nonempty_scored_sample():
    data = np.ones(3)
    theta = np.array([0.0, 0.1, 0.1, 0.1, 0.1])
    spec = ARMA(3, 1) + Normal()
    with pytest.raises(ValueError, match="conditioning lags"):
        _build(3, 1).fit(data)
    with pytest.raises(ValueError, match="conditioning lags"):
        _filter_arma(spec, data, theta, compute_score=False, compute_hessian=False)
    with pytest.raises(ValueError, match="conditioning lags"):
        _compute_arma_resid_sigma2(ARMA(3, 1), data, theta)
