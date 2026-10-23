"""Complete legacy likelihoods retain shape derivatives near the Gaussian limit."""

import numpy as np
import pytest

pass


def evaluate_kernel(family, skew, orders, nu):
    from native_api import evaluate
    from .oracle_legacy_paths import variance_path
    p,q=orders
    y=np.array([.5,-.8,1.2,-.3,.7,-1.1,.2,.9])
    theta=np.r_[.1,np.full(p,.08/p),np.full(p,.04/p) if family!="garch" else [],
        np.full(q,.6/q),[1.7] if family=="aparch" else [],nu,[.2] if skew else []]
    h0=float(np.mean(y*y))
    case=dict(model="gjr" if family=="gjr_garch" else family,density="skewt" if skew else "studentt",
        p=p,q=q,y=y,h0=h0,parameters=theta,transformed=False,first=0,stop=len(y),hold_back=0)
    actual=evaluate(case,7)
    assert actual["status"]==0 and actual["valid_mask"]==7
    variance=variance_path(y,theta,p,q,family,h0)
    return y,variance,actual["f"],actual["g"],actual["h"]



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
    mp = __import__("mpmath")
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
    from .transport import forced_skew11
    y=np.array([.5,-.8,1.2,-.3,.7,-1.1,.2,.9])
    theta=np.array([.1,.08,.6,nu,.2])
    value,gradient=forced_skew11(theta,y,False)
    fixed_value,fixed_gradient=forced_skew11(theta,y,True)
    np.testing.assert_allclose(fixed_value,value,rtol=0,atol=2e-11)
    np.testing.assert_allclose(fixed_gradient,gradient,rtol=3e-11,atol=0)

