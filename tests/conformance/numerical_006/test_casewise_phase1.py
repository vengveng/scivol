"""Retained density and conditioning obligations from mixed-phase Python nodes."""

import mpmath as mp
import numpy as np
import pytest

from native_api import evaluate
from .test_generalized_hyperbolic import SHAPE,Y,_surface_theta
from .oracle_ad import gh_surface_value_grad_hess
from .transport import local_density,surface
from .test_vg_retained_value_accuracy import scalar_vg_nll


@pytest.mark.parametrize("residual",[1e-8,-1e-8,0.,1e-200])
def test_vg_finite_value_obligation_of_python_array_case(residual):
    with mp.workdps(80):
        expected=float(scalar_vg_nll(mp.mpf(residual),mp.mpf(1),mp.mpf(0),mp.mpf(50)))
    actual=local_density(7,[residual,1.,0.,50.],1)[0][0]
    np.testing.assert_allclose(-actual,-expected,rtol=0,atol=3e-13)


@pytest.mark.parametrize("backend",["local-theta","native-F","native-FG","native-FGH"])
def test_variance_gamma_lower_shape_near_center_mathematics(backend):
    params=np.array([0.,1.5000000000000018])
    values=np.array([5.7735026918962544e-6])
    def exact_nll(residual,variance):
        with mp.workdps(70):
            shape=mp.mpf(float(params[1]))
            z=mp.mpf(float(residual))/mp.sqrt(float(variance))
            centered=mp.sqrt(shape)*z
            order=shape-mp.mpf(".5")
            logpdf=(mp.log(shape)/2+mp.log(2)-mp.log(2*mp.pi)/2-mp.loggamma(shape)
                +order*(mp.log(centered)-mp.log(2)/2)+mp.log(mp.besselk(order,mp.sqrt(2)*centered)))
            return float(mp.log(float(variance))/2-logpdf)
    expected=exact_nll(values[0],1.)
    if backend=="local-theta":
        actual=local_density(7,[values[0],1.,*params],1)[0][0]
    else:
        data=np.repeat(values,2)
        h0=float(np.mean(data**2))
        mask={"native-F":1,"native-FG":3,"native-FGH":7}[backend]
        case=dict(model="linear_garch",density="variance_gamma",p=0,q=0,p_ar=0,q_ma=0,p_vol=1,q_vol=1,
            y=data,design=np.empty((2,0)),intercept=True,h0=h0,first=0,stop=2,hold_back=0,
            transformed=False,parameters=np.array([0.,1.,0.,0.,*params]))
        result=evaluate(case,mask)
        assert result["status"]==0 and result["valid_mask"]==mask
        actual=result["f"]
        expected+=exact_nll(values[0],h0)
    assert actual==pytest.approx(expected,rel=3e-10,abs=3e-10)


@pytest.mark.parametrize("fixed_lambda",[-.5,1.],ids=["NIG","Hyperbolic"])
def test_fixed_lambda_density_math_is_not_rephased_with_fit_metadata(fixed_lambda):
    theta,p,q=_surface_theta("none","garch")
    reduced=theta[:-1]
    complete=np.r_[reduced,fixed_lambda]
    actual=surface(reduced,Y,"none","garch",p,q,fixed_lambda)
    general=surface(complete,Y,"none","garch",p,q,None)
    expected=gh_surface_value_grad_hess(reduced,Y,volatility_kind="garch",p_vol=p,q_vol=q,
        fixed_lambda=fixed_lambda,initial_variance=float(np.mean(Y*Y)))
    for left,right in ((actual,expected),((general[0],general[1][:-1],general[2][:-1,:-1]),expected)):
        np.testing.assert_allclose(left[0],right[0],rtol=3e-9,atol=3e-9)
        np.testing.assert_allclose(left[1],right[1],rtol=3e-7,atol=3e-8)
        np.testing.assert_allclose(left[2],right[2],rtol=3e-6,atol=3e-7)
