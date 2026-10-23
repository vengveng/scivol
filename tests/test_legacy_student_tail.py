"""Complete legacy likelihoods retain shape derivatives near the Gaussian limit."""

import numpy as np
import pytest

from scivol import _core


def evaluate_kernel(family, skew, orders, nu):
    p, q = orders
    y = np.array([.5, -.8, 1.2, -.3, .7, -1.1, .2, .9])
    theta = np.r_[.1, np.full(p, .08 / p),
                  np.full(p, .04 / p) if family != "garch" else [],
                  np.full(q, .6 / q), [1.7] if family == "aparch" else [],
                  nu, [.2] if skew else []]
    variance = np.full(len(y), np.mean(y * y))
    gradient = np.empty(len(theta))
    hessian = np.empty((len(theta), len(theta)))
    if family == "aparch":
        args = (theta.ctypes.data, y.ctypes.data, variance.ctypes.data)
        value = _core._aparch_nll_pq(*args, len(y), p, q, 2 if skew else 1, np.nan)
        _core._aparch_nll_grad_pq(*args, gradient.ctypes.data, len(y), p, q, 2 if skew else 1, np.nan)
        _core._aparch_nll_hess_pq(*args, hessian.ctypes.data, len(y), p, q, 2 if skew else 1, np.nan)
    else:
        density = "skewt" if skew else "studentt"
        data = y if skew or family != "garch" else y*y
        args = (theta.ctypes.data, data.ctypes.data, variance.ctypes.data)
        getattr(_core, f"_{family}_ll_grad_pq_{density}")(*args, gradient.ctypes.data, len(y), p, q)
        getattr(_core, f"_{family}_ll_hess_pq_{density}")(*args, hessian.ctypes.data, len(y), p, q)
        if skew:
            value = -_core._skewt_ll(y.ctypes.data, variance.ctypes.data, len(y), nu, .2)
        else:
            value = getattr(_core, f"_{family}_ll_pq_studentt")(*args, len(y), p, q)
    return y, variance, value, gradient, hessian


def reference_likelihood(mp, y, variance, nu, skew):
    c = mp.loggamma((nu+1)/2)-mp.loggamma(nu/2)-mp.log(mp.pi*(nu-2))/2
    lam = mp.mpf(.2) if skew else mp.mpf(0)
    a = 4*lam*mp.exp(c)*(nu-2)/(nu-1)
    b = mp.sqrt(1+3*lam*lam-a*a)
    value = mp.mpf(0)
    for residual, h in zip(y, variance, strict=True):
        h = mp.mpf(float(h))
        u = b*mp.mpf(float(residual))/mp.sqrt(h)+a
        v = u/(1+mp.sign(u)*lam)
        value += mp.log(h)/2-c-mp.log(b)+(nu+1)*mp.log1p(v*v/(nu-2))/2
    return value


@pytest.mark.parametrize("family", ["garch", "gjr_garch", "aparch"])
@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("orders", [(1, 1), (3, 2)])
@pytest.mark.parametrize("nu", [8., 128., 1e10, 1e15])
def test_complete_shape_jet_against_multiprecision(family, skew, orders, nu):
    mp = pytest.importorskip("mpmath")
    y, variance, value, gradient, hessian = evaluate_kernel(family, skew, orders, nu)
    index = -2 if skew else -1
    with mp.workdps(100):
        objective = lambda degrees: reference_likelihood(mp, y, variance, degrees, skew)
        expected = [float(objective(mp.mpf(nu))), float(mp.diff(objective, mp.mpf(nu))),
                    float(mp.diff(objective, mp.mpf(nu), 2))]
    np.testing.assert_allclose(value, expected[0], rtol=0, atol=2e-11)
    np.testing.assert_allclose([gradient[index], hessian[index, index]], expected[1:],
                               rtol=3e-8 if nu < 128 else 2e-11, atol=0)


@pytest.mark.parametrize("nu", [8., 128., 1e10, 1e15])
def test_fused_fixed_skew_path_agrees_with_general(nu):
    y, _, value, gradient, _ = evaluate_kernel("garch", True, (1, 1), nu)
    theta = np.array([.1, .08, .6, nu, .2])
    fixed_gradient = np.empty(5)
    fixed_value = _core._garch_ll_grad_11_skewt(theta.ctypes.data, y.ctypes.data,
                                             fixed_gradient.ctypes.data, len(y))
    np.testing.assert_allclose(fixed_value, value, rtol=0, atol=2e-11)
    np.testing.assert_allclose(fixed_gradient, gradient, rtol=3e-11, atol=0)
