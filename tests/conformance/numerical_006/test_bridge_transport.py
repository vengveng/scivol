"""Dimensional local transport and explicit-intercept admission regressions."""

import ctypes

import mpmath as mp
import numpy as np
import pytest

from .oracle_mp import Jet, jet_outputs, log_jet, loggamma_jet
from .transport import numerical_library


SENTINEL=-987654321.125


def local_request(density,point,mask):
    point=np.ascontiguousarray(point,dtype=np.float64)
    width=len(point)
    outputs=[np.full(1,SENTINEL),np.full(width,SENTINEL),np.full((width,width),SENTINEL)]
    status=numerical_library().numerical_density(density,mask,point.ctypes.data,
        *(a.ctypes.data for a in outputs))
    return status,outputs


def independent_student_local(point):
    with mp.workdps(100):
        residual,variance,nu=[Jet.variable(float(v),3,i) for i,v in enumerate(point)]
        constant=loggamma_jet(nu/2)-loggamma_jet((nu+1)/2)+log_jet((nu-2)*mp.pi)/2
        objective=constant+log_jet(variance)/2+(nu+1)*log_jet(1+residual*residual/(variance*(nu-2)))/2
        return jet_outputs(objective)


@pytest.mark.parametrize("exponent",[-510,510])
def test_normalized_local_materialization_keeps_dimensional_partials(exponent):
    variance=np.ldexp(1.,exponent)
    point=np.array([.75*np.sqrt(variance),variance,7.])
    expected=independent_student_local(point)
    for mask in range(1,8):
        status,outputs=local_request(1,point,mask)
        assert status==0
        for bit,actual,target in zip((1,2,4),outputs,expected):
            if mask&bit:
                np.testing.assert_allclose(actual,target,rtol=4e-12,atol=0)
            else:
                np.testing.assert_array_equal(actual,np.full_like(actual,SENTINEL))


@pytest.mark.parametrize("mask",range(1,8))
def test_nonrepresentable_local_requested_curvature_fails_without_writes(mask):
    variance=np.ldexp(1.,-600)
    status,outputs=local_request(1,[.75*np.sqrt(variance),variance,7.],mask)
    if mask&4:
        assert status!=0
        for output in outputs:
            np.testing.assert_array_equal(output,np.full_like(output,SENTINEL))
    else:
        assert status==0


@pytest.mark.parametrize("bad",[0.,2.,np.nextafter(1.,2.),np.nan,np.inf])
@pytest.mark.parametrize("row",[0,1,2])
def test_joint_transport_rejects_nonunit_explicit_intercept_before_writes(bad,row):
    library=numerical_library()
    data=np.array([.2,-.3,.4])
    design=np.column_stack((np.ones(3),np.array([.1,.2,.3])))
    design[row,0]=bad
    theta=np.array([.01,.02,.1,.05,0.,.3,2.,10.])
    outputs=[np.full(1,SENTINEL),np.full(len(theta),SENTINEL),np.full((len(theta),len(theta)),SENTINEL)]
    valid=ctypes.c_uint(999)
    status=library.numerical_joint(3,1,2,1,1,False,data.ctypes.data,len(data),design.ctypes.data,
        1.,theta.ctypes.data,len(theta),7,False,*(a.ctypes.data for a in outputs),
        ctypes.byref(valid),None,None)
    assert status!=0 and valid.value==0
    for output in outputs:
        np.testing.assert_array_equal(output,np.full_like(output,SENTINEL))
