"""Source-review regressions. Requires the chief's revised private probe."""
from dataclasses import replace
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import pytest

from test_mean_coordinates_006 import (
    SENTINEL, Spec, assert_oracle, call, coordinate_probe, fixture, oracle,
)
from oracles.mean_coordinates_006 import Jet


def helper_fixture(mask):
    spec = Spec(0,0,0,0,1,1,1,mask,0,0)
    actual = SimpleNamespace(status=0,
        f=np.full((1,1),SENTINEL),g=np.full((1,1,1),SENTINEL),
        h=np.full((1,1,1),SENTINEL))
    for bit,name,value in ((1,"f",1.),(2,"g",0.),(4,"h",0.)):
        if mask & bit:
            getattr(actual,name).fill(value)
    return spec,actual,[Jet.constant(1.,1)]


@pytest.mark.parametrize("mask",[1,2,4])
@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize("side",["actual","target","both"])
def test_oracle_helper_rejects_nonfinite(mask,bad,side):
    spec,actual,expected = helper_fixture(mask)
    name = {1:"f",2:"g",4:"h"}[mask]
    if side in ("actual","both"):
        getattr(actual,name).fill(bad)
    if side in ("target","both"):
        if mask == 1:
            expected[0] = replace(expected[0],value=mp.mpf(float(bad)))
        elif mask == 2:
            expected[0] = replace(expected[0],gradient=(mp.mpf(float(bad)),))
        else:
            expected[0] = replace(expected[0],hessian=(mp.mpf(float(bad)),))
    with pytest.raises(AssertionError,match="nonfinite requested"):
        assert_oracle(actual,expected,spec,mask)


@pytest.mark.parametrize("mask",range(1,8))
def test_oracle_helper_accepts_finite_requested_projection(mask):
    spec,actual,expected = helper_fixture(mask)
    assert_oracle(actual,expected,spec,mask)


def test_oracle_helper_ignores_unrequested_target_range():
    spec,actual,expected = helper_fixture(4)
    expected[0] = replace(expected[0],gradient=(mp.inf,))
    assert_oracle(actual,expected,spec,4)


@pytest.mark.parametrize("kind",[0,1])
@pytest.mark.parametrize("fixed",[0,1])
def test_poisoned_arena_preserves_complete_map(coordinate_probe,kind,fixed):
    spec,origin,z,data,anchors,selected = fixture(kind=kind,k=1 if kind == 0 else 2)
    spec.fixed = fixed
    expected = oracle(spec,origin,z,data,anchors,selected)
    for mask in range(1,8):
        spec.mask = mask
        baseline = call(coordinate_probe,spec,origin,z,data,anchors,selected,poison=0)
        assert_oracle(baseline,expected,spec,mask)
        for pattern in (0xA5,0x5A,0xFF):
            actual = call(coordinate_probe,spec,origin,z,data,anchors,selected,poison=pattern)
            assert actual.status == 0
            for name in ("f","g","h"):
                np.testing.assert_array_equal(getattr(actual,name),getattr(baseline,name))


@pytest.mark.parametrize("fixed",[0,1])
def test_h_only_does_not_require_binary64_jacobian(coordinate_probe,fixed):
    # e_1 = y_1 - mu - phi*x. At the zero origin, phi = -(mu+rho)/x.
    spec = Spec(1,0,1,2,2,1,1,7,fixed,0)
    x = 2.**-1024
    origin = np.array([0.,0.,1.,1.])
    z = origin.copy()
    data = np.array([x,0.])
    anchors, selected = [1],[1]
    expected = oracle(spec,origin,z,data,anchors,selected)
    with mp.workdps(100):
        assert expected[1].gradient[0] == -mp.power(2,1024)
        assert expected[1].gradient[1] == -mp.power(2,1024)
    assert all(value == 0 for value in expected[1].hessian)
    for mask in range(1,8):
        spec.mask = mask
        actual = call(coordinate_probe,spec,origin,z,data,anchors,selected,poison=0xFF)
        if mask & 2:
            assert actual.status == -3
            for name in ("f","g","h"):
                np.testing.assert_array_equal(getattr(actual,name),SENTINEL)
        else:
            assert_oracle(actual,expected,spec,mask)


@pytest.mark.parametrize("fixed",[0,1])
def test_finite_jacobian_does_not_admit_unrepresentable_hessian(coordinate_probe,fixed):
    # At pivot 2, B=1-psi=2^-53 and mu=B(origin)*mu(origin)/B-rho/B.
    spec = Spec(0,1,1,2,3,1,1,7,fixed,0)
    origin = np.array([2.**960,np.nextafter(1.,0.),1.,1.])
    z = origin.copy()
    z[0] = 0.
    data = np.zeros(3)
    anchors, selected = [2],[0]
    expected = oracle(spec,origin,z,data,anchors,selected)
    with mp.workdps(100):
        assert expected[0].gradient[1] == mp.power(2,1013)
        assert expected[0].hessian[1*spec.width+1] == mp.power(2,1067)
    for mask in range(1,8):
        spec.mask = mask
        actual = call(coordinate_probe,spec,origin,z,data,anchors,selected,poison=0x5A)
        if mask & 4:
            assert actual.status == -3
            for name in ("f","g","h"):
                np.testing.assert_array_equal(getattr(actual,name),SENTINEL)
        else:
            assert_oracle(actual,expected,spec,mask)
