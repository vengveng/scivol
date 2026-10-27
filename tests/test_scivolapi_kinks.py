"""Residual-face jets, one-sided AD references, and nonsmooth qualification."""

from __future__ import annotations

import numpy as np
import pytest

from scivol import ARX, EGARCH, GARCH, Normal, _scivolapi as api
from scivol._fit import native_spec
from tests._oracle import _jax_modules


def egarch_branch(y, first, constant, ar, ma, p, q, active, signs, residual_paths=False):
    jax, jnp, _ = _jax_modules()
    m = int(constant)+ar+ma
    seed = float(np.mean(y*y))

    def objective(theta, absolute):
        omega = theta[m]
        alpha = theta[m+1:m+1+p]
        gamma = theta[m+1+p:m+1+2*p]
        beta = theta[m+1+2*p:]
        residuals, log_variances = [], []
        total = 0.0
        for row in range(first, len(y)):
            t = row-first
            e = jnp.asarray(y[row])
            if constant:
                e = e-theta[0]
            for j in range(ar):
                e = e-theta[int(constant)+j]*y[row-j-1]
            for j in range(min(ma,t)):
                e = e-theta[int(constant)+ar+j]*residuals[t-j-1]
            log_h = jnp.log(seed)
            if t:
                log_h = omega
                for j in range(min(p,t)):
                    k = t-j-1
                    lag_e = residuals[k]
                    if k+first in active:
                        index = active.index(k+first)
                        abs_e = absolute[index]+signs[index]*lag_e
                    else:
                        abs_e = jnp.abs(lag_e)
                    log_h = log_h+alpha[j]*(abs_e*jnp.exp(-log_variances[k]/2)-jnp.sqrt(2/jnp.pi))
                    log_h = log_h+gamma[j]*lag_e*jnp.exp(-log_variances[k]/2)
                for j in range(q):
                    log_h = log_h+beta[j]*(log_variances[t-j-1] if j<t else jnp.log(seed))
            residuals.append(e)
            log_variances.append(log_h)
            total = total+.5*(jnp.log(2*jnp.pi)+log_h+e*e*jnp.exp(-log_h))
        return jnp.stack([residuals[row-first] for row in active]) if residual_paths else total

    return objective


@pytest.mark.parametrize("constant,ar,ma,p,q,active", [
    (True,0,0,1,1,[7]),
    (False,1,0,1,1,[7]),
    (True,1,0,2,1,[7,13]),
    (True,1,1,1,2,[7]),
])
def test_egarch_residual_faces_match_one_sided_jax(constant, ar, ma, p, q, active):
    jax,jnp,_ = _jax_modules()
    spec=api.ModelSpec()
    spec.mean.constant, spec.mean.ar, spec.mean.ma=constant,ar,ma
    spec.volatility=api.VolatilityKind.egarch
    spec.p,spec.q=p,q
    model=api.Model(spec)
    mean=([.125] if constant else [])+([.25] if ar else [])+([.125] if ma else [])
    theta=np.array(mean+[-.125]+[.125/p]*p+[-.0625/p]*p+[.75/q]*q)
    y=np.random.default_rng(9).integers(-12,12,24).astype(float)/8
    first=ar
    # Match the declared residual recursion using dyadic inputs, including MA.
    residuals=[]
    for row in range(first,len(y)):
        mu=(theta[0] if constant else 0)+(.25*y[row-1] if ar else 0)
        if ma and residuals:
            mu+=.125*residuals[-1]
        if row in active:
            y[row]=mu
        residuals.append(y[row]-mu)
    jet=api.Jet(model,y)
    out=jet.theta(theta,api.VALUE|api.GRADIENT|api.HESSIAN|api.KINKS)
    assert out['kink_rows']==active
    assert out['kink_supported']
    assert not out['valid'] & (api.GRADIENT|api.HESSIAN)
    absolute=jnp.zeros(len(active))
    center=egarch_branch(y,first,constant,ar,ma,p,q,active,[0]*len(active))
    np.testing.assert_allclose(out['value'],center(theta,absolute),rtol=2e-14)
    np.testing.assert_allclose(out['smooth_gradient'],jax.grad(center,0)(theta,absolute),rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(out['smooth_hessian'],jax.hessian(center,0)(theta,absolute),rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(out['kink_weights'],jax.grad(center,1)(theta,absolute),rtol=2e-12,atol=2e-12)
    residual=egarch_branch(y,first,constant,ar,ma,p,q,active,[0]*len(active),residual_paths=True)
    np.testing.assert_allclose(out['kink_normals'],jax.jacfwd(residual,0)(theta,absolute),rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(out['kink_curvatures'],jax.jacfwd(jax.jacrev(residual,0),0)(theta,absolute),
                               rtol=2e-13,atol=2e-13)
    for j in range(len(active)):
        for sign in (-1,1):
            signs=[0]*len(active)
            signs[j]=sign
            branch=egarch_branch(y,first,constant,ar,ma,p,q,active,signs)
            expected=out['smooth_gradient']+sign*out['kink_weights'][j]*out['kink_normals'][j]
            np.testing.assert_allclose(expected,jax.grad(branch,0)(theta,absolute),rtol=2e-12,atol=2e-12)
    ordinary=jet.theta(theta,api.VALUE|api.PATHS)
    assert ordinary['value']==out['value']
    sparse=jet.theta(theta,api.KINKS)
    np.testing.assert_array_equal(sparse['smooth_gradient'],out['smooth_gradient'])
    np.testing.assert_array_equal(sparse['smooth_hessian'],out['smooth_hessian'])
    z=model.to_z(theta)
    assert not jet.z(z,api.GRADIENT)['valid'] & api.GRADIENT
    assert jet.z(z,api.KINKS)['status']=='invalid_argument'


def test_roundoff_rule_is_not_an_identification_or_smoothing_width():
    spec=ARX(0)+EGARCH()+Normal()
    model=api.Model(native_spec(spec))
    y=np.array([.125,.8,-.4,1.2,-.7])
    theta=np.array([.125,-.1,.15,-.04,.8])
    jet=api.Jet(model,y)
    assert jet.theta(theta,api.KINKS)['kink_rows']==[0]
    theta[0]+=1e-10
    out=jet.theta(theta,api.VALUE|api.GRADIENT|api.HESSIAN|api.KINKS)
    assert out['status']=='ok'
    assert out['kink_rows']==[]
    ordinary=jet.theta(theta,7)
    for key in ('value','gradient','hessian'):
        np.testing.assert_array_equal(out[key],ordinary[key])


@pytest.mark.parametrize('family', ['aparch','ged'])
def test_subunit_power_cusps_are_not_given_a_finite_clarke_certificate(family):
    jax,jnp,_=_jax_modules()
    spec=api.ModelSpec()
    spec.mean.constant=True
    spec.volatility=getattr(api.VolatilityKind,family if family=='aparch' else 'constant')
    spec.p,spec.q=(1,1) if family=='aparch' else (0,0)
    if family=='ged':
        spec.density=api.DensityKind.ged
    theta=np.array([.125,.1,.1,.1,.7,.7] if family=='aparch' else [.125,1.,.7])
    y=np.array([.125,.8,-.4,1.2,-.7])
    out=api.Jet(api.Model(spec),y).theta(theta,api.VALUE|api.GRADIENT|api.HESSIAN|api.KINKS)
    assert out['valid'] & api.VALUE
    assert not out['valid'] & (api.GRADIENT|api.HESSIAN)
    assert out['kink_rows']==[0]
    assert not out['kink_supported']
    assert np.isnan(out['kink_weights'][0])
    options=api.FitOptions()
    options.start=theta.tolist()
    options.coordinates=api.Coordinates.theta
    options.maximum_iterations=5
    fitted=api.fit(api.Model(spec),y,options)
    assert fitted.estimate_available and fitted.feasible
    assert not fitted.success and fitted.kink_point
    assert fitted.termination==api.Termination.stalled
    assert not len(fitted.gradient) and not len(fitted.std_errors)
    assert any('non-Lipschitz' in warning for warning in fitted.warnings)
    # Check the actual joint likelihood on both open branches. No derivative
    # at zero is manufactured by JAX's convention for abs'(0).
    from tests.test_scivolapi_mean import row_terms
    objective=row_terms(y,np.empty((len(y),0)),spec.mean,family if family=='aparch' else 'constant',
                        spec.p,spec.q,'normal' if family=='aparch' else 'ged')
    gradient=jax.grad(lambda par:jnp.sum(objective(par)))
    for sign in (-1.,1.):
        slopes=[]
        for h in (1e-6,1e-10,1e-14):
            point=theta.copy()
            point[0]+=sign*h
            native=api.Jet(api.Model(spec),y).theta(point,api.GRADIENT)['gradient']
            oracle=np.asarray(gradient(point))
            np.testing.assert_allclose(native,oracle,rtol=3e-10,atol=3e-10)
            slopes.append(abs(oracle[0]))
        assert slopes[2]>100*slopes[0]


def test_unused_last_residual_and_zero_arch_coefficient_are_not_kinks():
    model=api.Model(native_spec(ARX(0)+EGARCH()+Normal()))
    y=np.array([.8,-.4,1.2,-.7,.125])
    theta=np.array([.125,-.1,.15,-.04,.8])
    assert api.Jet(model,y).theta(theta,api.KINKS)['kink_rows']==[]
    theta[2]=0
    y[0]=.125
    assert api.Jet(model,y).theta(theta,7|api.KINKS)['status']=='ok'
