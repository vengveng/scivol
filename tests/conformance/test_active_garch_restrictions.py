"""Actual GARCH zero/joint public restrictions. Source-only until chief's slot."""

import numpy as np
import pytest

from .fixtures.garch_cases import DENSITIES, REQUESTS, case as zero_case
from .fixtures.joint_cases import case as joint_case, mean_width
from .native_api import NativeProblem
from .oracles.garch_restricted import restricted_jet
from .test_active_mean_restrictions import compare


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("family", ("garch", "gjr"))
@pytest.mark.parametrize("kind", ("zero_fixed", "zero_runtime", "arma_fixed", "arma_runtime", "linear_fixed", "linear_wide"))
@pytest.mark.parametrize("response", (1.0, 3.0))
def test_actual_garch_selected_physical_objective(density, family, kind, response):
    if kind.startswith("zero"):
        case = zero_case(family,density,1 if kind == "zero_fixed" else 3,1 if kind == "zero_fixed" else 2)
        mean = 0
    elif kind.startswith("arma"):
        case = joint_case("arma_"+family,density,1 if kind == "arma_fixed" else 3,1 if kind == "arma_fixed" else 2,
                          1 if kind == "arma_fixed" else 3,1 if kind == "arma_fixed" else 2)
        mean = mean_width(case)
    else:
        columns = 3 if kind == "linear_fixed" else 17
        case = joint_case("linear_"+family,density,p_vol=1 if kind == "linear_fixed" else 3,q_vol=1 if kind == "linear_fixed" else 2,
                          columns=columns,intercept=True)
        case["internal_design_scales"] = np.ones(columns)
        mean = mean_width(case)
    case["internal_response_scale"] = response
    w = len(case["parameters"])
    with NativeProblem(case) as full:
        for indices in ([], [mean], [w-1], [w-1,mean,0], list(reversed(range(w)))):
            indices = list(dict.fromkeys(indices))
            reference = restricted_jet(case,indices)
            with full.restricted(case["parameters"],indices) as reduced:
                for request in REQUESTS:
                    compare(reduced.evaluate(case["parameters"],request),reference,request)


@pytest.mark.parametrize("response", (1.0, 3.0))
def test_fixed_mean_laplace_kink_does_not_block_garch_or_shape_axes(response):
    case = joint_case("linear_garch","ged",columns=0,parameters=[0,.2,.1,.55,1],y=[0]*4,h0=1.1)
    case["internal_response_scale"] = response
    indices = [4,1,3]
    expected = restricted_jet(case,indices)
    with NativeProblem(case) as full, full.restricted(case["parameters"],indices) as reduced:
        for request in REQUESTS:
            compare(reduced.evaluate(case["parameters"],request),expected,request)
        assert full.evaluate(case["parameters"],7)["valid_mask"] & 6 == 0


def test_h_only_mean_does_not_inherit_inactive_variance_range_failure():
    case = joint_case("linear_garch","normal",columns=0,parameters=[0,.4,0,.3],y=[1e200]*4,h0=1.1)
    variance, expected = 1.1, 1/1.1
    for _ in range(1,4):
        variance = .4+.3*variance
        expected += 1/variance
    with NativeProblem(case) as full, full.restricted(case["parameters"],[0]) as reduced:
        actual = reduced.evaluate(case["parameters"],4)
        assert actual["valid_mask"] == 4
        np.testing.assert_allclose(actual["h"],[[expected]],rtol=3e-13)
        assert actual["f"] == actual["sentinel"] and np.all(actual["g"] == actual["sentinel"])


def test_conditioned_variance_has_no_omega_sensitivity():
    case = zero_case("garch","normal",parameters=[.4,.1,.3],y=[0]*4,h0=1e-200)
    variance, first, expected = 1e-200, 0, 0
    for _ in range(1,4):
        variance = .4+.3*variance
        first = 1+.3*first
        expected -= .5*(first/variance)**2
    with NativeProblem(case) as full, full.restricted(case["parameters"],[0]) as reduced:
        actual = reduced.evaluate(case["parameters"],4)
        assert actual["valid_mask"] == 4
        np.testing.assert_allclose(actual["h"],[[expected]],rtol=3e-13)


def test_recursive_cancellation_uncertainty_is_not_nondifferentiability():
    # Retained original fixture: alpha=beta=0 gives h=omega=2 on scored row 4.
    # The actual restricted scalar is log(2)+m*m after exact fixed cancellation.
    case = joint_case("arma_garch","ged",0,2,1,1,intercept=False,
                      parameters=[0,.25,2,0,0,1],y=[0,0,1,0,.25],h0=2,first=4)
    with NativeProblem(case) as full, full.restricted(case["parameters"],[0]) as reduced:
        for request in REQUESTS:
            compare(reduced.evaluate(case["parameters"],request),
                    (np.log(2.0),np.array([0.0]),np.array([[2.0]])),request)


def test_actual_affine_kink_keeps_nondifferentiability_status():
    case = joint_case("linear_garch","ged",columns=0,parameters=[0,2,0,0,1],y=[0]*4,h0=2)
    with NativeProblem(case) as full, full.restricted(case["parameters"],[0]) as reduced:
        for request in REQUESTS:
            actual = reduced.evaluate(case["parameters"],request)
            assert actual["valid_mask"] == request & 1
            for bit, slot in ((2,1),(4,2)):
                if request & bit:
                    assert actual["reasons"][slot] == 6  # nondifferentiable


def test_actual_gjr_fixed_mean_removes_shock_kink_only_from_inactive_path():
    case = joint_case("linear_gjr", "normal", columns=0,
                      parameters=[0, .2, .1, .04, .55], y=[0]*4, h0=1.1)
    indices = [3, 1, 4]
    reference = restricted_jet(case, indices)
    with NativeProblem(case) as full:
        with full.restricted(case["parameters"], indices) as reduced:
            for request in REQUESTS:
                compare(reduced.evaluate(case["parameters"], request), reference, request)
        with full.restricted(case["parameters"], [0]) as moving_mean:
            for request in REQUESTS:
                actual = moving_mean.evaluate(case["parameters"], request)
                assert actual["valid_mask"] == request & 3
                if request & 4:
                    assert actual["reasons"][2] == 6  # proven unequal one-sided H


def test_gjr_shock_kink_can_cancel_in_the_actual_scored_density():
    # h(0)=1 and e(0)=1, so the Gaussian variance slope is exactly zero.
    # The one-sided h'' jump is multiplied by that zero. L is C2 here,
    # with L_m=-1 and L_mm=1, although the variance state itself is not C2.
    # This remains an unskipped composition-proof gate, not a claimed pass.
    case = joint_case("linear_gjr", "normal", columns=0,
                      parameters=[0, .4, .1, .04, .3], y=[0, 1], h0=2, first=1)
    with NativeProblem(case) as full, full.restricted(case["parameters"], [0]) as reduced:
        for request in REQUESTS:
            actual = reduced.evaluate(case["parameters"], request)
            assert actual["valid_mask"] == request
            if request & 2:
                np.testing.assert_allclose(actual["g"], [-1], rtol=3e-13, atol=0)
            if request & 4:
                np.testing.assert_allclose(actual["h"], [[1]], rtol=3e-13, atol=0)
