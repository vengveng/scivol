"""Unexecuted independent full-scalar identities and actual public restrictions."""

import numpy as np
import pytest

from .fixtures.garch_cases import REQUESTS
from .fixtures.mean_cases import arma_case
from .fixtures.joint_cases import case as joint_case
from .native_api import NativeProblem
from .oracles.garch import jax_modules, prepare_logpdf, scalar_jet
from .test_active_mean_restrictions import compare


def exact_laplace_square(point):
    _, jnp, _ = jax_modules()
    variance, m = point
    return .5*jnp.log(2*variance)+jnp.sqrt(2/variance)*m*m


def test_independent_ad_of_complete_simplified_restricted_scalar():
    # e[q]=s, e[q+1]=-s*m, e[q+2]=s*(.25+m*m-.25), s=+/-1.
    # At fixed GED nu=1 this eliminates abs by identity on a neighborhood.
    expected = scalar_jet(exact_laplace_square, [2.,0.])
    assert abs(expected[0]-np.log(2.)) <= 3e-13
    np.testing.assert_allclose(expected[1], [.25,0], rtol=3e-13, atol=0)
    np.testing.assert_allclose(expected[2], [[-.125,0],[0,2]], rtol=3e-13, atol=0)


@pytest.mark.parametrize("family", ("mean","garch","gjr"))
@pytest.mark.parametrize("q", (2,5))
@pytest.mark.parametrize("sign", (1.,-1.))
@pytest.mark.parametrize("response", (1.,2.,.125))
def test_fixed_cancellation_complete_restriction_and_active_scale(family,q,sign,response):
    y = [0.]*q+[sign,0.,sign*.25]
    mean = [0.,.25]+[0.]*(q-2)
    if family == "mean":
        case = arma_case("ged",0,q,intercept=False,parameters=mean+[2.,1.],y=y,first=q+2)
    else:
        volatility = [2.,0.,0.] if family == "garch" else [2.,0.,0.,0.]
        case = joint_case("arma_"+family,"ged",0,q,1,1,intercept=False,
                          parameters=mean+volatility+[1.],y=y,h0=2.,first=q+2)
    case["internal_response_scale"] = response
    expected = scalar_jet(exact_laplace_square,[2.,0.])
    # omega is the exact scored variance for these fixed alpha/beta/gamma axes.
    with NativeProblem(case) as full, full.restricted(case["parameters"],[q,0]) as reduced:
        for request in REQUESTS:
            compare(reduced.evaluate(case["parameters"],request),expected,request)


@pytest.mark.parametrize("family", ("mean","garch"))
def test_unresolved_shifted_square_remains_honestly_unsupported(family):
    # e2=1, e3=1-m, e4=.5-m*(1-m)-.25=(m-.5)^2.
    # The bounded certificate does not carry the required shifted variable proof.
    if family == "mean":
        case = arma_case("ged",0,2,intercept=False,parameters=[.5,.25,2,1],
                         y=[0,0,1,1,.5],first=4)
    else:
        case = joint_case("arma_garch","ged",0,2,1,1,intercept=False,
                          parameters=[.5,.25,2,0,0,1],y=[0,0,1,1,.5],h0=2,first=4)
    with NativeProblem(case) as full, full.restricted(case["parameters"],[0]) as reduced:
        for request in REQUESTS:
            actual = reduced.evaluate(case["parameters"],request)
            assert actual["valid_mask"] == request & 1
            for bit,slot in ((2,1),(4,2)):
                if request & bit:
                    assert actual["reasons"][slot] == 3
            if not request & 1:
                assert actual["f"] == actual["sentinel"]


def test_garch_nonconstant_variance_composes_the_same_certified_square():
    case = joint_case("arma_garch","ged",0,2,1,1,intercept=False,
                      parameters=[0,.25,2,.125,0,1],y=[0,0,1,0,.25],h0=2,first=4)
    _, jnp, _ = jax_modules()
    def objective(point):
        omega,m = point
        variance = omega+.125*m*m
        return .5*jnp.log(2*variance)+jnp.sqrt(2/variance)*m*m
    expected = scalar_jet(objective,[2.,0.])
    np.testing.assert_allclose(expected[1],[.25,0],rtol=3e-13,atol=0)
    np.testing.assert_allclose(expected[2],[[-.125,0],[0,2.0625]],rtol=3e-13,atol=0)
    with NativeProblem(case) as full, full.restricted(case["parameters"],[2,0]) as reduced:
        for request in REQUESTS:
            compare(reduced.evaluate(case["parameters"],request),expected,request)


def test_cubic_mixed_absolute_peano_jet_is_not_a_classical_hessian_certificate():
    # e1=-a, e2=m*a, e3=-m*m*a. Its absolute value is |a|*m*m.
    # G(0,0)=0, but the gradient is undefined at nearby a=0,m!=0.
    case = arma_case("ged",1,1,intercept=False,parameters=[0,0,2,1],
                     y=[1,0,0,0],first=3)
    with NativeProblem(case) as full, full.restricted(case["parameters"],[1,0]) as reduced:
        for request in REQUESTS:
            actual = reduced.evaluate(case["parameters"],request)
            assert actual["valid_mask"] == request & 3
            if request & 2:
                np.testing.assert_array_equal(actual["g"],[0,0])
            if request & 4:
                assert actual["reasons"][2] == 3  # Insufficient evidence, not a fabricated kink.


def test_univariate_cubic_absolute_has_matching_classical_one_sided_jets():
    # |x|^3 really is C2: its gradient 3*x*|x| has derivative zero at zero.
    # The coarser order/parity certificate cannot separate it from x^2*|y|.
    for side in (-1.,1.):
        jet = scalar_jet(lambda point: side*point[0]**3,[0.])
        assert jet[0] == 0
        np.testing.assert_array_equal(jet[1],[0.])
        np.testing.assert_array_equal(jet[2],[[0.]])


def test_gjr_scored_scalar_one_sided_ad_jets_match_classical_derivatives():
    # Same retained two-row fixture as test_gjr_shock_kink_can_cancel... .
    # h=1+c*m*m, e=1-m, c=alpha or alpha+gamma. The full L has L'=-1,L''=1.
    _, jnp, _ = jax_modules()
    logpdf = prepare_logpdf(jnp.asarray([]),"normal")
    jets = []
    for coefficient in (.1,.1+.04):
        def objective(point):
            m = point[0]
            variance = 1+coefficient*m*m
            return .5*jnp.log(variance)-logpdf((1-m)/jnp.sqrt(variance))
        jets.append(scalar_jet(objective,[0.]))
    for jet in jets:
        assert abs(jet[0]-(.5*np.log(2*np.pi)+.5)) <= 3e-13
        np.testing.assert_allclose(jet[1],[-1.],rtol=3e-13,atol=0)
        np.testing.assert_allclose(jet[2],[[1.]],rtol=3e-13,atol=0)
    # This analytic/AD control does not claim the native GJR jump gate is solved.
