"""Compiled concentrated-likelihood controls, including production fast-math."""

import numpy as np
import pytest

from scivol import _core


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("residual", [1e-150, 1e-20, 1., 1e150, 1e154])
def test_concentrated_normal_curvature_is_ratio_first(fixed, residual):
    width = 3 if fixed else 1
    theta, y = np.zeros(width), np.array([0., residual])
    path, seed = np.empty(2), np.zeros(1)
    gradient, hessian = np.empty(width), np.empty((width, width))
    ptr = lambda x: x.ctypes.data
    if fixed:
        value = _core._arma_nll_grad_11_normal(
            ptr(theta), ptr(y), ptr(path), ptr(gradient), 2)
        _core._arma_hess_11_normal(ptr(theta), ptr(y), ptr(path), ptr(hessian), 2)
        plain = _core._arma_nll_11_normal(ptr(theta), ptr(y), ptr(path), 2)
    else:
        value = _core._arma_nll_grad_pq_normal(
            ptr(theta), ptr(y), ptr(path), ptr(seed), ptr(gradient), 2, 0, 0)
        _core._arma_hess_pq_normal(
            ptr(theta), ptr(y), ptr(path), ptr(seed), ptr(hessian), 2, 0, 0)
        plain = _core._arma_nll_pq_normal(
            ptr(theta), ptr(y), ptr(path), ptr(seed), 2, 0, 0)
    assert value == plain
    assert value == pytest.approx(np.log(abs(residual))+.5, abs=1e-12)
    assert gradient[0]*residual == pytest.approx(-1., rel=2e-14)
    assert hessian[0, 0]*residual*residual == pytest.approx(-1., rel=2e-14)
    assert np.isfinite(hessian).all()


def test_zero_rss_has_no_finite_concentrated_estimate():
    theta, data, path, seed = np.zeros(1), np.zeros(2), np.empty(2), np.zeros(1)
    gradient, hessian = np.empty(1), np.empty(1)
    p = lambda x: x.ctypes.data
    value = _core._arma_nll_grad_pq_normal(
        p(theta), p(data), p(path), p(seed), p(gradient), 2, 0, 0)
    _core._arma_hess_pq_normal(p(theta), p(data), p(path), p(seed), p(hessian), 2, 0, 0)
    assert np.isposinf(value)
    assert np.isnan(gradient).all()
    assert np.isnan(hessian).all()
