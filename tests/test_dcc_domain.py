"""DCC normalization is the declared correlation map, without a diagonal floor."""

import numpy as np
import pytest

from scivol import _dcc_kernels as kernels
from scivol._devtools import ad_oracle


@pytest.mark.parametrize("student", [False, True])
@pytest.mark.parametrize("orders", [(1, 1), (2, 1)])
@pytest.mark.parametrize("scale", [1., 2e-7, 1e-30])
def test_positive_diagonal_below_old_floor_matches_ad(student, orders, scale):
    pytest.importorskip("jax")
    p, q = orders
    eps = np.random.default_rng(812).normal(size=(8, 2)) * scale
    qbar = eps.T @ eps / len(eps)
    theta = np.r_[np.full(p, .1/p), np.full(q, .7/q), [8.] if student else []]
    prefix = "dcc_studentt" if student else "dcc"
    expected = getattr(ad_oracle, prefix + "_value_grad_hess")(theta, eps, qbar, p, q)
    actual = getattr(kernels, prefix + "_nll_grad_hess")(theta, eps, qbar, p, q, return_scores=True)
    for value, reference in zip(actual[:3], expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=3e-8, atol=2e-11)
    np.testing.assert_allclose(-actual[3].mean(axis=0), actual[1], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(getattr(kernels, prefix + "_nll")(theta, eps, qbar, p, q),
                               actual[0], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("student", [False, True])
@pytest.mark.parametrize("orders", [(1, 1), (2, 1)])
@pytest.mark.parametrize("diagonal", [0., -1., np.nan, np.inf])
def test_invalid_correlation_is_not_a_finite_likelihood(student, orders, diagonal):
    p, q = orders
    eps = np.ones((8, 2))
    qbar = np.diag([diagonal, 1.])
    theta = np.r_[np.full(p, .1/p), np.full(q, .7/q), [8.] if student else []]
    prefix = "dcc_studentt" if student else "dcc"
    assert np.isinf(getattr(kernels, prefix + "_nll")(theta, eps, qbar, p, q))
    actual = getattr(kernels, prefix + "_nll_grad_hess")(theta, eps, qbar, p, q, return_scores=True)
    assert np.isinf(actual[0])
    for derivative in actual[1:]:
        assert np.isnan(derivative).all()
