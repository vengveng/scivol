"""Unconditional model moments against independent stationary representations."""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy import stats

from scivol import APARCH, EGARCH, GARCH, Normal, StudentT
from scivol import _scivolapi as api
from tests.test_scivolapi_mean import mean_spec, model_for


@pytest.mark.parametrize("mean,point,expected", [
    (dict(constant=False), [], 0.0),
    ({}, [.4], .4),
    (dict(ar=2, ma=1), [.4, 1.2, -.4, .6], 2.0),
    (dict(har=(1, 5)), [.4, .2, .3], .8),
    (dict(ar=1, har=(5,)), [.4, .2, .3], .8),
    (dict(ar=2), [.4, .5, .6], np.nan),
    (dict(exog=1), [.4, .2], np.nan),
])
def test_unconditional_mean(mean, point, expected):
    model = model_for(mean_spec(**mean), "constant", p=0, q=0)
    theta = np.array(point + [.7])
    actual = model.unconditional_mean(theta)
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == pytest.approx(expected, rel=2e-14)


def test_gjr_density_weight_and_aparch_delta_two_equivalence():
    mean = mean_spec(constant=False)
    for density, shape in [("normal", []), ("skew_t", [.12, -.3])]:
        gjr = model_for(mean, "gjr", density)
        aparch = model_for(mean, "aparch", density)
        alpha, gamma, beta, omega = .08, .3, .7, .1
        # (|e|-gamma*e)^2 = (1-gamma)^2 e^2
        #                       + 4*gamma e^2 I(e<0).
        gjr_theta = np.array([omega, alpha*(1-gamma)**2, 4*alpha*gamma, beta] + shape)
        aparch_theta = np.array([omega, alpha, gamma, beta, 2.] + shape)
        expected = omega / (1 - gjr.persistence(gjr_theta))
        assert aparch.unconditional_variance(aparch_theta) == pytest.approx(expected, rel=3e-13)
        assert gjr.unconditional_variance(gjr_theta) == pytest.approx(expected, rel=3e-13)
        aparch_theta[4] = np.nextafter(2., 1.)
        assert np.isnan(aparch.unconditional_variance(aparch_theta))


def test_constant_divergent_and_invalid_variances():
    mean = mean_spec(constant=False)
    constant = model_for(mean, "constant", p=0, q=0)
    assert constant.unconditional_variance(np.array([.7])) == .7
    garch = model_for(mean)
    assert np.isinf(garch.unconditional_variance(np.array([.1, .2, .8])))
    assert np.isnan(garch.unconditional_variance(np.array([-.1, .2, .7])))
    with pytest.raises(ValueError, match="length"):
        garch.unconditional_variance(np.array([.1, .2]))


@pytest.mark.parametrize("alpha,gamma,beta", [
    ([0.], [0.], [0.]),
    ([.15], [-.08], [0.]),
    ([.15], [-.08], [.9]),
    ([.1], [.04], [-.8]),
    ([.06, .04], [-.02, .01], [.3, .4]),
    ([.06, .04], [-.02, .01], [1.2, -.4]),
    ([.06], [-.02], [-1.2, -.4]),
    ([.15], [-.08], [.99]),
])
def test_egarch_normal_infinite_product_against_mpmath(alpha, gamma, beta):
    model = model_for(mean_spec(constant=False), "egarch", p=len(alpha), q=len(beta))
    omega = -.02
    theta = np.array([omega] + alpha + gamma + beta)
    with mp.workdps(50):
        m = mp.sqrt(2/mp.pi)
        aa, bb, betas = [mp.mpf(x) for x in alpha], [mp.mpf(x) for x in gamma], [mp.mpf(x) for x in beta]
        a, b = [], []
        logarithm = mp.mpf(omega)/(1-sum(betas))
        for k in range(5000):
            a.append((aa[k] if k < len(aa) else 0) + sum(v*a[k-j-1] for j, v in enumerate(betas) if j < k))
            b.append((bb[k] if k < len(bb) else 0) + sum(v*b[k-j-1] for j, v in enumerate(betas) if j < k))
            right, left = a[k]+b[k], a[k]-b[k]
            moment = (mp.exp(right**2/2)*mp.erfc(-right/mp.sqrt(2))
                      + mp.exp(left**2/2)*mp.erfc(-left/mp.sqrt(2)))/2
            logarithm += mp.log(moment)-a[k]*m
        expected = float(mp.exp(logarithm))
    actual = model.unconditional_variance(theta)
    assert actual == pytest.approx(expected, rel=3e-11)
    if any(alpha) or any(gamma):
        assert actual > np.exp(omega/(1-sum(beta)))


def test_egarch_laplace_exponential_moments():
    model = model_for(mean_spec(constant=False), "egarch", "ged")
    theta = np.array([-.02, .15, -.08, .9, 1.])
    weights = .9**np.arange(1000)
    a, b = .15*weights, -.08*weights
    moment = .5/(1-(a+b)/np.sqrt(2)) + .5/(1-(a-b)/np.sqrt(2))
    expected = np.exp(-.02/(1-.9) + np.sum(np.log(moment)-a/np.sqrt(2)))
    assert model.unconditional_variance(theta) == pytest.approx(expected, rel=2e-11)


@pytest.mark.parametrize("density,shape", [("student_t", [.1]), ("skew_t", [.1, -.3]), ("ged", [.8])])
def test_egarch_infinite_exponential_moments(density, shape):
    model = model_for(mean_spec(constant=False), "egarch", density)
    assert np.isinf(model.unconditional_variance(np.array([-.02, .15, -.08, .9] + shape)))
    assert np.isinf(model.unconditional_variance(np.array([-.02, 0., -.08, .9] + shape)))
    # Negative beta makes a later filtered magnitude coefficient positive.
    assert np.isinf(model.unconditional_variance(np.array([-.02, -.15, 0., -.9] + shape)))
    # A deterministic log-variance recursion has no tail-moment restriction.
    assert model.unconditional_variance(np.array([-.02, 0., 0., .9] + shape)) == pytest.approx(np.exp(-.2))


def test_egarch_negative_student_shocks_have_finite_variance():
    model = model_for(mean_spec(constant=False), "egarch", "student_t")
    theta = np.array([-.02, -.15, .04, .5, .1])
    nu, scale = 10., np.sqrt(.8)
    m = 2*quad(lambda z: z*stats.t.pdf(z/scale, nu)/scale, 0, np.inf, epsabs=1e-12)[0]
    logarithm = -.02/(1-.5)
    for weight in .5**np.arange(45):
        moment = quad(lambda z: np.exp(-.15*weight*abs(z)+.04*weight*z)*stats.t.pdf(z/scale, nu)/scale,
                      -np.inf, np.inf, epsabs=1e-12)[0]
        logarithm += np.log(moment)+.15*weight*m
    assert model.unconditional_variance(theta) == pytest.approx(np.exp(logarithm), rel=2e-10)


@pytest.mark.parametrize("spec", [GARCH()+Normal(), EGARCH()+Normal(), APARCH()+Normal(), EGARCH()+StudentT()])
def test_python_result_and_summary_use_model_moments(spec, capsys):
    result = spec.fit(np.random.default_rng(935).normal(size=350))
    model, _, _ = result._native()
    expected = model.unconditional_variance(result.params)
    assert result.unconditional_variance == pytest.approx(expected, nan_ok=True)
    assert result.garch_params.unconditional_variance == pytest.approx(expected, nan_ok=True)
    result.summary()
    output = capsys.readouterr().out
    assert "Unconditional Variance:" in output
    assert "Geometric Variance" not in output
