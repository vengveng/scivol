"""Exact-input VG tail values, MP forward AD, and sparse retained-point requests."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess

import mpmath as mp
import numpy as np
import pytest

from .oracle_mp import Jet
from .oracle_mp import exp_jet, jet_outputs, log_jet, loggamma_jet

ROOT=Path(__file__).resolve().parents[3]
NATIVE_SOURCE = r"""
#include "local_density_materialization.hpp"
extern "C" int tail_local(unsigned mask,const double* p,double* f,double* g,double* h) {
    return numerical_006::local_density::request<vlk::experiment::generated::VarianceGammaDensity>(mask,p,f,g,h);
}
"""


@pytest.fixture(scope="module")
def vg_bridge(tmp_path_factory):
    existing=os.environ.get("SCIVOL_VG_VALUE_BRIDGE")
    if existing:
        library=ctypes.CDLL(existing)
    else:
        compiler=shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            pytest.fail("A C++20 compiler is required")
        headers=Path(os.environ.get("SCIVOL_VG_HEADER_ROOT",ROOT))
        directory=tmp_path_factory.mktemp("vg_value")
        source,image=directory/"local.cpp",directory/"local.so"
        source.write_text(NATIVE_SOURCE)
        command=[compiler,"-std=c++20","-O3","-ffp-contract=fast","-funroll-loops",
            "-fno-exceptions","-fno-rtti","-fPIC","-shared",
            "-I"+str(ROOT/"tests/bridges/numerical_006"),"-I"+str(headers/"core/include"),
            "-I"+str(headers/"core/src"),"-I"+str(headers/"core/src/math"),
            str(source),str(headers/"core/src/math/gh_math_translation.cpp"),"-o",str(image)]
        subprocess.run(command,check=True,capture_output=True,text=True,timeout=110)
        library=ctypes.CDLL(str(image))
    library.tail_local.argtypes=[ctypes.c_uint]+[ctypes.c_void_p]*4
    library.tail_local.restype=ctypes.c_int
    return library


def local_request(library,point,mask):
    point=np.ascontiguousarray(point,dtype=np.float64)
    outputs=np.full(1,np.nan),np.full(4,np.nan),np.full((4,4),np.nan)
    status=library.tail_local(mask,point.ctypes.data,*(a.ctypes.data for a in outputs))
    assert status==0,status
    for bit,array in zip((1,2,4),outputs):
        assert np.isfinite(array).all() if mask&bit else np.isnan(array).all()
    return outputs


def scalar_vg_nll(e,h,rho,lam):
    z=e/mp.sqrt(h)
    theta=mp.sqrt(2)*rho/mp.sqrt(1-rho*rho)
    scale=mp.sqrt(lam*(1+theta*theta))
    y=scale*z+theta*lam
    a=mp.sqrt(2+theta*theta)
    order=lam-mp.mpf(".5")
    normalizer=mp.log(2*scale)-mp.log(2*mp.pi)/2-mp.loggamma(lam)
    if y==0:
        kernel=(order-1)*mp.log(2)+mp.loggamma(order)-2*order*mp.log(a)
    else:
        kernel=theta*y+order*mp.log(abs(y)/a)+mp.log(mp.besselk(order,a*abs(y)))
    return mp.log(h)/2-normalizer-kernel


@lru_cache(maxsize=12)
def bessel_rule(count,digits):
    with mp.workdps(digits):
        nodes,weights=mp.gauss_quadrature(count,"legendre")
        return tuple((left+2*(nodes[i]+1),2*weights[i])
            for left in range(0,20,4) for i in range(count))


def log_bessel_jet(order,argument,count,digits):
    # DLMF 10.32.9, with the six differentiated integrals before AD composition.
    scale=mp.sqrt(argument.value)
    sums=[mp.mpf(0)]*6
    for node,weight in bessel_rule(count,digits):
        t=node/scale
        excess=2*mp.sinh(t/2)**2
        nt=order.value*t
        kernel=weight*mp.exp(-argument.value*excess)*mp.cosh(nt)
        slope=t*mp.tanh(nt)
        for i,term in enumerate((1,slope,excess,t*t,slope*excess,excess*excess)):
            sums[i]+=kernel*term
    _,dn,dx,dnn,dnx,dxx=[v/sums[0] for v in sums]
    value=-argument.value-mp.log(scale)+mp.log(sums[0])
    gx=-1-dx
    hnn,hnx,hxx=dnn-dn*dn,-dnx+dn*dx,dxx-dx*dx
    width=len(order.gradient)
    return Jet(value,tuple(dn*a+gx*b for a,b in zip(order.gradient,argument.gradient)),tuple(
        dn*order.hessian[i*width+j]+gx*argument.hessian[i*width+j]
        +hnn*order.gradient[i]*order.gradient[j]
        +hnx*(order.gradient[i]*argument.gradient[j]+argument.gradient[i]*order.gradient[j])
        +hxx*argument.gradient[i]*argument.gradient[j]
        for i in range(width) for j in range(width)))


def vg_density_jet(e,h,rho,lam,*,count,digits):
    theta=exp_jet(-log_jet(1-rho*rho)/2)*mp.sqrt(2)*rho
    scale=exp_jet(log_jet(lam*(1+theta*theta))/2)
    z=e*exp_jet(-log_jet(h)/2)
    y=scale*z+theta*lam
    magnitude=y if y.value>0 else -y
    a=exp_jet(log_jet(2+theta*theta)/2)
    order=lam-mp.mpf(".5")
    normalizer=log_jet(scale)+mp.log(2)-mp.log(2*mp.pi)/2-loggamma_jet(lam)
    if y.value==0:
        center=(order-1)*mp.log(2)+loggamma_jet(order)-2*order*log_jet(a)
        curvature=a*a/(2*(1-order))
        return log_jet(h)/2-normalizer-theta*y-center-curvature*y*y/2
    return log_jet(h)/2-normalizer-theta*y-order*(log_jet(magnitude)-log_jet(a))-log_bessel_jet(order,a*magnitude,count,digits)


def local_oracle(point,digits,count):
    with mp.workdps(digits):
        variables=[Jet.variable(float(v),4,i) for i,v in enumerate(point)]
        result=vg_density_jet(*variables,count=count,digits=digits)
        scalar=scalar_vg_nll(*(mp.mpf(float(v)) for v in point))
        assert abs(result.value-scalar)<mp.mpf("1e-24")
        return jet_outputs(result)


@pytest.mark.parametrize("rho",[-np.nextafter(.995,0.),np.nextafter(.995,0.)])
def test_retained_tail_local_all_masks_against_independent_mp_ad(vg_bridge,rho):
    point=np.array([.36197374594952486,.6897939882481383,rho,50.])
    low=local_oracle(point,80,32)
    high=local_oracle(point,120,48)
    for a,b in zip(low,high):
        np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-24)
    full=local_request(vg_bridge,point,7)
    for mask in range(1,8):
        actual=local_request(vg_bridge,point,mask)
        for bit,a,target,projection in zip((1,2,4),actual,high,full):
            if mask&bit:
                np.testing.assert_allclose(a,target,rtol=0 if bit==1 else 3e-10,atol=2e-11)
                np.testing.assert_allclose(a,projection,rtol=3e-10,atol=2e-11)


@pytest.mark.parametrize("point",[
    [.27,.9,.12,2.],
    [0.,1.,0.,1.5000000000000018],
    [0.,1.,0.,50.],
    [89.,1.,0.,2.],
    [91.,1.,0.,2.],
])
def test_vg_product_origin_and_scaled_tail_share_exact_density(vg_bridge,point):
    low=local_oracle(point,80,48)
    high=local_oracle(point,120,64)
    for a,b in zip(low,high):
        np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-24)
    full=local_request(vg_bridge,point,7)
    for mask in range(1,8):
        for bit,a,target,projection in zip((1,2,4),local_request(vg_bridge,point,mask),high,full):
            if mask&bit:
                np.testing.assert_allclose(a,target,rtol=0 if bit==1 else 3e-10,atol=2e-11)
                np.testing.assert_allclose(a,projection,rtol=3e-10,atol=2e-11)


RETAINED_THETA=np.array([
    .034791556550068825,.09448998515685914,-.1578829425824645,.1666270409379095,
    .08158503505473741,.07448731108437091,-.04281813535254858,.8327087232844275,
    .014159438811793051,.11180119472039933,.029816367416422334,-.9999999999999999,
    .13121146888582044,.9999999999999999,4.3716469888015535e-108,
    3.849549272061785e-19,.17469279412032182,-.9949999999999999,50.])


def retained_sample(*, fixed=False):
    return make_retained_sample(fixed=fixed)



def exact_joint_paths(model,theta):
    point=[mp.mpf(float(v)) for v in theta]
    m,p,q=model.columns,model.p_vol,model.q_vol
    delta=point[model.shape_start-1]
    es=[mp.mpf(float(y))-mp.fsum(mp.mpf(float(x))*a for x,a in zip(row,point[:m]))
        for y,row in zip(model.data,model.features)]
    hs=[mp.mpf(float(model.h0))]
    powers=[hs[0]**(delta/2)]
    for t in range(1,len(es)):
        power=point[m]+mp.fsum(point[m+1+j]*(abs(es[t-1-j])-point[m+1+p+j]*es[t-1-j])**delta
            for j in range(min(t,p)))
        power+=mp.fsum(point[m+1+2*p+j]*powers[t-1-j] for j in range(min(t,q)))
        powers.append(power)
        hs.append(power**(2/delta))
    return es,hs


def test_retained_full_harx_aparch_exact_additive_value():
    model=retained_sample()
    references=[]
    for digits in (80,120):
        with mp.workdps(digits):
            es,hs=exact_joint_paths(model,RETAINED_THETA)
            shape=[mp.mpf(float(v)) for v in RETAINED_THETA[-2:]]
            references.append(float(mp.fsum(scalar_vg_nll(e,h,*shape) for e,h in zip(es,hs))))
    assert references[0]==references[1]
    assert references[1]==101.77866403511115
    actual=model.evaluate(RETAINED_THETA,1)[0]
    assert np.isfinite(actual)
    assert abs(actual-references[1])<=2e-11


def retained_joint_oracle(theta=RETAINED_THETA, *, digits, count, parameters=None,
                          model=None, return_jet=False):
    """Differentiate the declared joint recurrence and independent VG integral."""
    model = retained_sample() if model is None else model
    with mp.workdps(digits):
        point = ([Jet.variable(float(value), model.size, index)
                  for index, value in enumerate(theta)] if parameters is None else
                 [Jet(mp.mpf(float(value)), parameter.gradient, parameter.hessian)
                  for value, parameter in zip(theta, parameters, strict=True)])
        m, p, q = model.columns, model.p_vol, model.q_vol
        delta = point[model.shape_start - 1]
        residuals = [Jet.constant(float(y), model.size)
                     - sum((mp.mpf(float(x)) * coefficient
                            for x, coefficient in zip(row, point[:m])),
                           Jet.constant(0, model.size))
                     for y, row in zip(model.data, model.features)]
        initial_variance = Jet.constant(float(model.h0), model.size)
        powers = [exp_jet(delta * mp.log(mp.mpf(float(model.h0))) / 2)]
        total = vg_density_jet(residuals[0], initial_variance, *point[-2:],
                               count=count, digits=digits)
        for t in range(1, len(residuals)):
            power = point[m]
            for j in range(min(t, p)):
                residual = residuals[t - 1 - j]
                assert residual.value != 0
                shock = ((residual if residual.value > 0 else -residual)
                         - point[m + 1 + p + j] * residual)
                power += point[m + 1 + j] * exp_jet(delta * log_jet(shock))
            for j in range(min(t, q)):
                power += point[m + 1 + 2 * p + j] * powers[t - 1 - j]
            powers.append(power)
            variance = exp_jet(2 * log_jet(power) / delta)
            total += vg_density_jet(residuals[t], variance, *point[-2:],
                                    count=count, digits=digits)
        return total if return_jet else jet_outputs(total)


@pytest.fixture(scope="module")
def retained_joint_ad():
    low = retained_joint_oracle(digits=80, count=32)
    high = retained_joint_oracle(digits=120, count=48)
    for coarse, refined in zip(low, high):
        np.testing.assert_allclose(coarse, refined, rtol=1e-14, atol=1e-24)
    return high


@pytest.mark.parametrize("fixed", [False, True])
def test_retained_full_joint_vg_all_masks_match_mp_ad(retained_joint_ad, fixed):
    model = retained_sample(fixed=fixed)
    full = model.evaluate(RETAINED_THETA, 7)
    for mask in range(1, 8):
        actual = model.evaluate(RETAINED_THETA, mask)
        for bit, output, reference, projection in zip((1, 2, 4), actual,
                                                     retained_joint_ad, full):
            if mask & bit:
                assert np.isfinite(output).all()
                np.testing.assert_allclose(output, reference,
                    rtol=0 if bit == 1 else 3e-10, atol=2e-11)
                np.testing.assert_allclose(output, projection,
                    rtol=0 if bit == 1 else 3e-10, atol=2e-11)
            else:
                assert output is None


from .transport import retained_sample as make_retained_sample
