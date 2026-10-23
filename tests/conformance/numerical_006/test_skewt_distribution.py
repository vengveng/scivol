from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

pass
pass


@pytest.mark.parametrize("lam", [-0.50, -0.15, 0.15, 0.50])
def test_hansen_skewt_is_zero_mean_and_unit_variance(lam: float) -> None:
    nu = 8.0

    mass = quad(lambda z: _hansen_skewt_pdf_scalar(z, nu, lam), -np.inf, np.inf)[0]
    mean = quad(lambda z: z * _hansen_skewt_pdf_scalar(z, nu, lam), -np.inf, np.inf)[0]
    second_moment = quad(
        lambda z: z * z * _hansen_skewt_pdf_scalar(z, nu, lam),
        -np.inf,
        np.inf,
    )[0]

    assert mass == pytest.approx(1.0, abs=2e-8)
    assert mean == pytest.approx(0.0, abs=2e-8)
    assert second_moment == pytest.approx(1.0, abs=2e-8)






def test_c_skewt_likelihood_matches_standardized_python_density():
    from .transport import local_density
    resid=np.array([-.70,-.15,.05,.40,1.20])
    sigma2=np.array([.80,1.10,.95,1.30,1.05])
    nu,lam=8.,-.25
    z=resid/np.sqrt(sigma2)
    expected=sum(np.log(_hansen_skewt_pdf_scalar(float(value),nu,lam)) for value in z)-.5*np.log(sigma2).sum()
    actual=-sum(local_density(2,[e,h,nu,lam],1)[0][0] for e,h in zip(resid,sigma2))
    assert actual==pytest.approx(expected,rel=2e-10,abs=2e-10)


def _hansen_skewt_pdf_scalar(z,nu,lam):
    from scipy.special import gammaln
    c=np.exp(gammaln((nu+1)/2)-gammaln(nu/2))/np.sqrt(np.pi*(nu-2))
    a=4*lam*c*(nu-2)/(nu-1)
    b=np.sqrt(1+3*lam*lam-a*a)
    u=b*z+a
    return b*c*(1+(u/(1-lam if u<0 else 1+lam))**2/(nu-2))**(-(nu+1)/2)
