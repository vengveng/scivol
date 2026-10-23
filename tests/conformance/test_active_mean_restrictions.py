"""Source-only qualification matrix until the chief runs the accepted bridge."""

import itertools

import numpy as np
import pytest

from .fixtures.mean_cases import arma_case, linear_case, mean_count
from .fixtures.garch_cases import DENSITIES, REQUESTS
from .native_api import NativeProblem
from .oracles.constant_scale_restricted import restricted_jet


def compare(actual, expected, request):
    assert actual["status"] == 0 and actual["valid_mask"] == request
    for bit, key, value in zip((1,2,4), ("f","g","h"), expected):
        if request & bit:
            assert np.isfinite(actual[key]).all()
            if bit == 1:
                assert abs(actual[key]-value) <= 3e-9
            else:
                np.testing.assert_allclose(actual[key], value, atol=3e-7 if bit == 2 else 3e-6,
                                           rtol=4e-6 if bit == 2 else 1e-5)
        else:
            assert np.all(actual[key] == actual["sentinel"])


@pytest.mark.parametrize("density", DENSITIES)
def test_every_narrow_subset_against_restricted_scalar_ad(density):
    case = linear_case(density, 0, intercept=True)
    w = len(case["parameters"])
    with NativeProblem(case) as full:
        for bits in itertools.product((False,True), repeat=w):
            indices = [i for i, free in enumerate(bits) if free][::-1]
            reference = restricted_jet(case, indices)
            with full.restricted(case["parameters"], indices) as restricted:
                for request in REQUESTS:
                    compare(restricted.evaluate(case["parameters"], request), reference, request)


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("layout", ("linear_fixed", "linear_wide", "arma_fixed", "arma_runtime"))
def test_fixed_runtime_and_wider_layouts(density, layout):
    if layout.startswith("linear"):
        case = linear_case(density, 3 if layout == "linear_fixed" else 17, intercept=True)
    else:
        case = arma_case(density, 1 if layout == "arma_fixed" else 3, 1 if layout == "arma_fixed" else 2)
    k, w = mean_count(case), len(case["parameters"])
    subsets = [[], [0], [k], list(range(k+1,w)), [w-1,0,k], list(reversed(range(w)))]
    with NativeProblem(case) as full:
        for indices in subsets:
            indices = list(dict.fromkeys(indices))
            expected = restricted_jet(case, indices)
            with full.restricted(case["parameters"], indices) as restricted:
                for request in REQUESTS:
                    compare(restricted.evaluate(case["parameters"], request), expected, request)


@pytest.mark.parametrize("indices", ([1], [2], [2,1], []))
def test_fixed_mean_laplace_centre_all_remaining_subsets(indices):
    case = linear_case("ged",0,intercept=True,parameters=[0,2,1],y=[0,0,0])
    expected = restricted_jet(case,indices)
    with NativeProblem(case) as full, full.restricted(case["parameters"],indices) as restricted:
        for request in REQUESTS:
            compare(restricted.evaluate(case["parameters"],request),expected,request)
        actual = full.evaluate(case["parameters"],7)
        assert actual["valid_mask"] == 1


def test_free_ma_with_identically_zero_history_is_not_a_mean_kink():
    case = arma_case("ged",0,2,intercept=False,parameters=[.1,-.05,2,1],y=[0]*8)
    indices = [3,0,2,1]
    expected = restricted_jet(case,indices)
    with NativeProblem(case) as full, full.restricted(case["parameters"],indices) as restricted:
        for request in REQUESTS:
            compare(restricted.evaluate(case["parameters"],request),expected,request)


def test_fixed_intercept_kink_does_not_block_other_active_mean_columns():
    case = linear_case("ged",1,intercept=True,parameters=[0,.1,2,1],
                       design=[[0],[1],[2]],y=[0,1,-.5])
    indices = [3,1,2]
    expected = restricted_jet(case,indices)
    with NativeProblem(case) as full, full.restricted(case["parameters"],indices) as restricted:
        for request in REQUESTS:
            compare(restricted.evaluate(case["parameters"],request),expected,request)
        assert full.evaluate(case["parameters"],7)["valid_mask"] == 1


@pytest.mark.parametrize("response", (1.0,2.0,.125))
def test_removable_recursive_composition(response):
    # Only row 3 is scored. With m=ma[1], e1=1, e2=-m, e3=m*m.
    # At GED nu=1,v=2 the exact restricted scalar is log(2)+m*m.
    # Exact identity, not differentiation of abs at zero or smoothing.
    case = arma_case("ged",0,1,intercept=False,parameters=[0,2,1],
                     y=[0,1,0,0],first=3)
    case["internal_response_scale"] = response
    with NativeProblem(case) as full, full.restricted(case["parameters"],[0]) as restricted:
        for request in REQUESTS:
            compare(restricted.evaluate(case["parameters"],request),
                    (np.log(2.0),np.array([0.0]),np.array([[2.0]])),request)


def test_h_only_does_not_inherit_unrequested_value_or_gradient_range_failure():
    case = linear_case("normal",0,intercept=True,parameters=[0,1e-20],y=[1e300]*3)
    with NativeProblem(case) as full, full.restricted(case["parameters"],[0]) as restricted:
        actual = restricted.evaluate(case["parameters"],4)
        assert actual["status"] == 0 and actual["valid_mask"] == 4
        np.testing.assert_allclose(actual["h"], [[3e20]], rtol=3e-14)
        assert actual["f"] == actual["sentinel"] and np.all(actual["g"] == actual["sentinel"])


def test_mixed_product_has_gradient_but_no_hessian():
    # e1=-a, e2=m*a. This is an actual ARMA restriction, not a seed-only test.
    case = arma_case("ged",1,1,intercept=False,parameters=[0,0,2,1],y=[1,0,0],first=2)
    with NativeProblem(case) as full, full.restricted(case["parameters"],[1,0]) as restricted:
        for request in REQUESTS:
            actual = restricted.evaluate(case["parameters"],request)
            assert actual["valid_mask"] == request & 3
            if request & 2:
                np.testing.assert_array_equal(actual["g"], [0,0])
            if request & 4:
                assert actual["reasons"][2] == 6  # Public Status::nondifferentiable.


def test_unknown_cancellation_is_not_a_claim_of_nondifferentiability():
    # Retained original fixture. Exact fixed-term cancellation now proves m*m;
    # the complete restricted NLL is log(2)+m*m, not a Peano-only claim.
    case = arma_case("ged",0,2,intercept=False,parameters=[0,.25,2,1],
                     y=[0,0,1,0,.25],first=4)
    with NativeProblem(case) as full, full.restricted(case["parameters"],[0]) as restricted:
        for request in REQUESTS:
            compare(restricted.evaluate(case["parameters"],request),
                    (np.log(2.0),np.array([0.0]),np.array([[2.0]])),request)


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("layout", ("linear_fixed", "linear_wide", "arma_fixed", "arma_runtime"))
@pytest.mark.parametrize("response", (2.0, .125))
def test_selected_unit_composition_against_physical_scalar_ad(density, layout, response):
    if layout.startswith("linear"):
        width = 3 if layout == "linear_fixed" else 17
        case = linear_case(density,width,intercept=True)
        case["internal_design_scales"] = .5 + .125*np.arange(width)
    else:
        case = arma_case(density,1 if layout == "arma_fixed" else 3,1 if layout == "arma_fixed" else 2)
    case["internal_response_scale"] = response
    k, w = mean_count(case), len(case["parameters"])
    with NativeProblem(case) as full:
        for indices in ([], [k], [0], [w-1,0,k], list(reversed(range(w)))):
            indices = list(dict.fromkeys(indices))
            # The independent oracle uses the unchanged physical data/theta;
            # it neither applies B/C nor imports the native unit owner.
            expected = restricted_jet(case,indices)
            with full.restricted(case["parameters"],indices) as restricted:
                for request in REQUESTS:
                    compare(restricted.evaluate(case["parameters"],request),expected,request)
