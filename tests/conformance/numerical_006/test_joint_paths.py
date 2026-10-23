"""Exact old Student tail and rational residual cases via canonical core paths."""

from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from .historical_design import _har_history_means
from .historical_path_fixtures import repaired_sample_start
from .path_transport import SENTINEL, assert_valid, evaluate
from .test_student_tail_value_accuracy import RETAINED_THETA, retained_linear_sample


def full_har_design(data, exogenous):
    return np.ascontiguousarray(np.column_stack((np.ones(len(data)),
        _har_history_means(data,1),_har_history_means(data,3),exogenous)))


def declared_case(family, density, data, design, p, q, theta):
    data=np.ascontiguousarray(data,dtype=np.float64)
    h0=float(np.mean(data**2))
    return dict(family=family,density=density,data=data,design=np.ascontiguousarray(design),
        p=p,q=q,theta=np.ascontiguousarray(theta),h0=h0,
        # The old linear startup skips unavailable persistence and shock lags.
        implicit_log_variance=0.,conditioning="retained_omitted_history",
        first=0,stop=len(data),scored_rows=tuple(range(len(data))))


def retained_tail_case(family, skew, nu):
    rng=np.random.default_rng(2931)
    raw=rng.normal(size=31)
    external=rng.normal(size=(len(raw),2))
    # Public HARX physical hold-back precedes construction of the retained design.
    data,external=np.ascontiguousarray(raw[3:]),np.ascontiguousarray(external[3:])
    design=full_har_design(data,external)
    if family==2:
        vol=[-.1,.03,.02,.01,-.02,.01,.02,.4,.3]
    elif family==1:
        vol=[.1,.03,.02,.01,.02,.01,.02,.4,.3]
    else:
        vol=[.1,.03,.02,.01,.4,.3]
    lam=-.038898954 if skew else 0.
    theta=np.r_[np.full(design.shape[1],.04),vol,[nu,lam] if skew else [nu]]
    return declared_case(family,2 if skew else 1,data,design,3,2,theta)


def mp_log_likelihood(result, nu, skew):
    with mp.workdps(80):
        v,l=mp.mpf(nu),mp.mpf(-.038898954 if skew else 0.)
        c=mp.loggamma((v+1)/2)-mp.loggamma(v/2)-mp.log(mp.pi*(v-2))/2
        a=4*l*mp.exp(c)*(v-2)/(v-1)
        b=mp.sqrt(1+3*l*l-a*a)
        total=mp.mpf(0)
        for residual,variance in zip(result["residuals"],result["sigma2"],strict=True):
            h=mp.mpf(float(variance))
            u=b*mp.mpf(float(residual))/mp.sqrt(h)+a
            adjusted=u/(1+l*(1 if u>=0 else -1))
            total+=c+mp.log(b)-mp.log(h)/2-(v+1)/2*mp.log1p(adjusted**2/(v-2))
        return float(total)


def check_projection_and_backend(case, mask, full):
    with_paths=evaluate(case,mask)
    without_paths=evaluate(case,mask,paths=False)
    checked=evaluate(case,mask,paths=False,lane=1)
    for result,paths in ((with_paths,True),(without_paths,False),(checked,False)):
        assert_valid(result,mask,paths=paths)
    for bit,name in ((1,"f"),(2,"g"),(4,"h")):
        if mask&bit:
            np.testing.assert_array_equal(with_paths[name],without_paths[name])
            np.testing.assert_allclose(with_paths[name],checked[name],rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(with_paths[name],full[name],rtol=2e-12,atol=2e-12)
    np.testing.assert_array_equal(with_paths["residuals"],full["residuals"])
    np.testing.assert_allclose(with_paths["sigma2"],full["sigma2"],rtol=2e-12,atol=0)
    return with_paths


@pytest.mark.parametrize("family",[0,1,2],ids=["GARCH","GJRGARCH","EGARCH"])
@pytest.mark.parametrize("skew",[False,True])
@pytest.mark.parametrize("nu",[128.,5.82483924e7,1e10])
def test_joint_legacy_student_tail_likelihood_high_precision(family, skew, nu):
    case=retained_tail_case(family,skew,nu)
    full=evaluate(case,7)
    assert_valid(full,7,paths=True)
    assert full["sigma2"][0]==case["h0"]
    target=mp_log_likelihood(full,nu,skew)
    assert -full["f"]==pytest.approx(target,rel=2e-13,abs=2e-12)
    for mask in range(1,8):
        actual=check_projection_and_backend(case,mask,full)
        if mask&1:
            assert -actual["f"]==pytest.approx(mp_log_likelihood(actual,nu,skew),rel=2e-13,abs=2e-12)


@pytest.mark.parametrize("family",[0,1,2,3])
def test_other_joint_families_share_the_repaired_linear_residual(family):
    original=retained_linear_sample()
    # The old direct JointMeanModel constructor does not trim this retained sample again.
    data=original.data
    design=full_har_design(data,np.ascontiguousarray(original.features[:,-3:]))
    theta=repaired_sample_start(family,data,design.shape[1],2,4)
    theta[:design.shape[1]]=RETAINED_THETA[:design.shape[1]]
    case=declared_case(family,1,data,design,2,4,theta)
    exact=[float(Fraction(float(y))-sum((Fraction(float(x))*Fraction(float(a))
        for x,a in zip(row,theta[:design.shape[1]])),Fraction(0))) for y,row in zip(data,design)]
    full=evaluate(case,7)
    assert_valid(full,7,paths=True)
    for mask in range(1,8):
        actual=check_projection_and_backend(case,mask,full)
        residuals=actual["residuals"]
        np.testing.assert_allclose(residuals,exact,rtol=3*np.finfo(float).eps,atol=0)


def test_egarch_explicit_log_shift_is_distinct_from_retained_omitted_history():
    historical=retained_tail_case(2,False,128.)
    shifted=dict(historical,implicit_log_variance=float(np.log(historical["h0"])),
        conditioning="explicit_binary64_log_shift_negative_control")
    retained=evaluate(historical,7)
    explicit=evaluate(shifted,7)
    assert_valid(retained,7,paths=True)
    assert_valid(explicit,7,paths=True)
    assert retained["sigma2"][0]==explicit["sigma2"][0]==historical["h0"]
    assert np.max(np.abs(retained["sigma2"][1:]-explicit["sigma2"][1:]))>1e-8
    for case,full in ((historical,retained),(shifted,explicit)):
        for mask in range(1,8):
            check_projection_and_backend(case,mask,full)


@pytest.mark.parametrize("family",[0,1,2,3])
def test_short_path_arena_rejects_before_candidate_writes(family):
    original=retained_linear_sample()
    theta=repaired_sample_start(family,original.data,original.columns,2,4)
    theta[:original.columns]=RETAINED_THETA[:original.columns]
    case=declared_case(family,1,original.data,original.features,2,4,theta)
    actual=evaluate(case,7,path_capacity=len(original.data)-1)
    assert actual["status"]!=0 and actual["valid_mask"]==0
    for name in ("f","g","h","residuals","sigma2"):
        array=np.asarray(actual[name])
        np.testing.assert_array_equal(array.view(np.uint64),np.full_like(array,SENTINEL).view(np.uint64))
