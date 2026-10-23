"""Unskipped production range gates for the staged selected consumer."""

import numpy as np

from .fixtures.garch_cases import REQUESTS, case as zero_case
from .fixtures.mean_cases import linear_case
from .native_api import NativeProblem


def test_public_affine_seed_must_survive_density_contraction():
    # Physical L_b=-x*(y-b*x)/v=-1e-200. Internally e=1e-150,
    # v=1e-300, X=1 and B_b=1e-350. The seed is not a double,
    # but its contracted gradient is. No absolute tolerance can accept zero.
    case = linear_case("normal",1,intercept=False,parameters=[0,1],y=[1],design=[[1e-200]])
    case["internal_response_scale"] = 1e150
    case["internal_design_scales"] = np.asarray([1e-200])
    with NativeProblem(case) as full, full.restricted(case["parameters"],[0]) as reduced:
        for request in (2,3):
            actual = reduced.evaluate(case["parameters"],request)
            assert actual["valid_mask"] == request
            assert actual["g"][0] < 0
            np.testing.assert_allclose(actual["g"][0]/-1e-200,1,rtol=3e-13,atol=0)


def test_actual_garch_dimensional_seed_must_survive_contraction():
    # B_omega=1/r^2=1e-400. This is a full recursive consumer gate,
    # not a synthetic test of B/C actions.
    case = zero_case("garch","normal",parameters=[1e100,.1,.3],y=[0]*4,h0=1e100)
    case["internal_response_scale"] = 1e200
    variance, first, gradient, hessian = 1e100, 0, 0, 0
    for _ in range(1,4):
        variance = 1e100+.3*variance
        first = 1+.3*first
        gradient += .5*first/variance
        hessian -= .5*(first/variance)**2
    with NativeProblem(case) as full, full.restricted(case["parameters"],[0]) as reduced:
        for request in REQUESTS:
            actual = reduced.evaluate(case["parameters"],request)
            assert actual["valid_mask"] == request
            if request & 2:
                assert actual["g"][0] > 0
                np.testing.assert_allclose(actual["g"][0]/gradient,1,rtol=3e-12,atol=0)
            if request & 4:
                assert actual["h"][0,0] < 0
                np.testing.assert_allclose(actual["h"][0,0]/hessian,1,rtol=3e-12,atol=0)
