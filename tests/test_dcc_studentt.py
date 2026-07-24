"""Student-t DCC kernel, oracle, and estimation tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from scivol import DCC, _core
from scivol._dcc_kernels import (
    compute_qbar,
    dcc_nll,
    dcc_studentt_nll,
    dcc_studentt_nll_grad,
    dcc_studentt_nll_grad_hess,
)
from scivol._devtools.ad_oracle import dcc_studentt_value_grad_hess


HAS_JAX = importlib.util.find_spec("jax") is not None


@pytest.fixture(scope="module")
def oracle_data():
    rng = np.random.default_rng(812)
    eps = rng.standard_t(8.0, size=(160, 3)) * np.sqrt(6.0 / 8.0)
    return eps, compute_qbar(eps)


@pytest.mark.skipif(not HAS_JAX, reason="jax is required for AD-oracle derivative checks")
@pytest.mark.parametrize(
    "p,q,theta",
    [
        (1, 1, np.array([0.05, 0.90, 8.0])),
        (2, 1, np.array([0.03, 0.02, 0.88, 11.0])),
        (1, 2, np.array([0.04, 0.45, 0.45, 6.5])),
        (2, 2, np.array([0.02, 0.01, 0.48, 0.45, 15.0])),
    ],
)
def test_studentt_c_derivatives_match_ad_oracle(oracle_data, p, q, theta):
    eps, qbar = oracle_data
    value_c, grad_c, hess_c = dcc_studentt_nll_grad_hess(theta, eps, qbar, p, q)
    value_ad, grad_ad, hess_ad = dcc_studentt_value_grad_hess(theta, eps, qbar, p, q)

    np.testing.assert_allclose(value_c, value_ad, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(grad_c, grad_ad, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(hess_c, hess_ad, rtol=2e-7, atol=2e-9)


def test_studentt_11_matches_generic_c_entry_points(oracle_data):
    eps, qbar = oracle_data
    theta = np.ascontiguousarray([0.05, 0.90, 8.0], dtype=np.float64)
    eps_c = np.ascontiguousarray(eps, dtype=np.float64)
    qbar_c = np.ascontiguousarray(qbar.ravel(), dtype=np.float64)
    t_count, n_series = eps.shape

    value_11, grad_11, hess_11 = dcc_studentt_nll_grad_hess(
        theta, eps, qbar, 1, 1
    )
    value_pq = _core._dcc_nll_pq_studentt(
        theta.ctypes.data,
        eps_c.ctypes.data,
        qbar_c.ctypes.data,
        t_count,
        n_series,
        1,
        1,
    )
    grad_pq = np.empty(3, dtype=np.float64)
    hess_pq = np.empty(9, dtype=np.float64)
    nll_pq = np.empty(1, dtype=np.float64)
    _core._dcc_nll_grad_hess_pq_studentt(
        theta.ctypes.data,
        eps_c.ctypes.data,
        qbar_c.ctypes.data,
        grad_pq.ctypes.data,
        hess_pq.ctypes.data,
        nll_pq.ctypes.data,
        0,
        t_count,
        n_series,
        1,
        1,
    )

    np.testing.assert_allclose(value_11, value_pq, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(value_11, nll_pq[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(grad_11, grad_pq, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        hess_11, hess_pq.reshape(3, 3), rtol=2e-15, atol=2e-15
    )


def test_studentt_scores_sum_to_gradient(oracle_data):
    eps, qbar = oracle_data
    theta = np.array([0.05, 0.90, 8.0])
    _, grad, scores = dcc_studentt_nll_grad(
        theta, eps, qbar, 1, 1, return_scores=True
    )
    np.testing.assert_allclose(grad, -scores.mean(axis=0), rtol=1e-11, atol=1e-12)


def test_studentt_copula_converges_to_gaussian_likelihood(oracle_data):
    eps, qbar = oracle_data
    dynamics = np.array([0.05, 0.90])
    gaussian = dcc_nll(dynamics, eps, qbar, 1, 1)
    studentt = dcc_studentt_nll(
        np.array([dynamics[0], dynamics[1], 1_000_000.0]),
        eps,
        qbar,
        1,
        1,
    )
    np.testing.assert_allclose(studentt, gaussian, rtol=2e-5, atol=2e-7)


def test_studentt_dcc_warm_start_and_summary(oracle_data):
    eps, _ = oracle_data
    model = DCC(1, 1, distribution="studentt")
    first = model.fit_from_residuals(eps, compute_se=False, maxiter=100)
    second = model.fit_from_residuals(
        eps, initial_result=first, compute_se=True, maxiter=5
    )

    assert second.warm_started
    assert second.theta.shape == (3,)
    assert second.std_errors.shape == (3,)
    assert "Student-t copula correlation MLE" in second.summary()
    assert "nu" in second.summary()


def test_studentt_warm_start_rejects_gaussian_result(oracle_data):
    eps, _ = oracle_data
    gaussian = DCC(1, 1).fit_from_residuals(eps, compute_se=False)
    with pytest.raises(ValueError, match="expected studentt"):
        DCC(1, 1, "studentt").fit_from_residuals(
            eps, initial_result=gaussian, compute_se=False
        )
