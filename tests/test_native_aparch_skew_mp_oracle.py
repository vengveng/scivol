"""Independent full-theta MP oracle checks before using it at difficult fits."""

import numpy as np
import pytest

from test_native_aparch_subset_curvature import unscaled_theta_oracle, jet_outputs


@pytest.mark.parametrize("skew", [0., -.25, .3])
def test_skew_mp_oracle_matches_independent_joint_ad(skew):
    from scivol._devtools.ad_oracle import arma_aparch_value_grad_hess

    orders = (1, 1, 1, 1)
    data = np.random.default_rng(712).normal(size=9)
    theta = np.array([.04, .12, -.08, .1, .09, .16, .7, 1.4, 7., skew])
    high = unscaled_theta_oracle(orders, theta, data, .8, digits=100, density="skewt")
    expected = arma_aparch_value_grad_hess(theta, data, *orders, "skewt", initial_variance=.8)
    value, gradient, hessian = jet_outputs(high)
    assert np.isfinite([*value, *gradient, *hessian.ravel()]).all()
    np.testing.assert_allclose(value[0], expected[0], rtol=0, atol=2e-11)
    np.testing.assert_allclose(gradient, expected[1], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(hessian, expected[2], rtol=2e-8, atol=2e-9)
    if skew == 0.:
        reduced = jet_outputs(unscaled_theta_oracle(orders, theta[:-1], data, .8,
            digits=100, density="studentt"))
        np.testing.assert_array_equal(value, reduced[0])
        np.testing.assert_array_equal(gradient[:-1], reduced[1])
        np.testing.assert_array_equal(hessian[:-1, :-1], reduced[2])
