"""New-only original-coordinate AD checks for internal ARMA coordinate maps.

The chief supplies a privately compiled probe. Missing evidence is a failure,
not a skip, and this module never builds or imports the installed extension.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import pytest

from oracles.mean_coordinates_006 import Jet, inverse_lag_polynomial

SENTINEL = 918273.125
PTR = ctypes.c_void_p
SIZE = ctypes.c_size_t


class Spec(ctypes.Structure):
    _fields_ = [(name, SIZE) for name in
                ("p", "q", "start", "tail", "n", "k", "kind", "mask", "fixed", "mutation")]

    @property
    def width(self):
        return 1+self.p+self.q+self.tail

    @property
    def packed(self):
        return self.width*(self.width+1)//2


@pytest.fixture(scope="module")
def coordinate_probe():
    path = os.environ.get("SCIVOL_COORDINATE_006_PROBE")
    assert path and Path(path).is_file(), "Chief must build and pin SCIVOL_COORDINATE_006_PROBE"
    lib = ctypes.CDLL(str(Path(path).resolve()))
    lib.coordinate_006_workspace.argtypes = [ctypes.POINTER(Spec)]
    lib.coordinate_006_workspace.restype = SIZE
    lib.coordinate_006_run.argtypes = [ctypes.POINTER(Spec), PTR, PTR, PTR, PTR, PTR,
                                       PTR, SIZE, SIZE, PTR, PTR, PTR]
    lib.coordinate_006_run.restype = ctypes.c_int
    return lib


def call(lib, spec, origin, z, data, anchors, selected, *, short=0, missing=0, poison=0xA5):
    z = np.ascontiguousarray(z, dtype=np.float64).reshape(-1,spec.width)
    origin, data = [np.ascontiguousarray(x, dtype=np.float64) for x in (origin,data)]
    anchors, selected = [np.ascontiguousarray(x, dtype=np.uintp) for x in (anchors,selected)]
    required = lib.coordinate_006_workspace(ctypes.byref(spec))
    arena = np.full(required+128, poison, dtype=np.uint8)
    offset = (-arena.ctypes.data)%64+64
    outputs = [np.full(size+2,SENTINEL) for size in
               (len(z)*spec.width, len(z)*spec.width**2, len(z)*spec.width*spec.packed)]
    before = [x.copy() for x in (origin,z,data,anchors,selected)]
    status = lib.coordinate_006_run(ctypes.byref(spec),origin.ctypes.data,z.ctypes.data,
        data.ctypes.data,anchors.ctypes.data,selected.ctypes.data,
        arena.ctypes.data+offset,max(0,required-short),len(z),
        *(None if missing & bit else x[1:].ctypes.data
          for bit,x in zip((1,2,4),outputs)))
    for value, old in zip((origin,z,data,anchors,selected),before):
        np.testing.assert_array_equal(value,old)
    np.testing.assert_array_equal(arena[:offset],poison)
    np.testing.assert_array_equal(arena[offset+required:],poison)
    for value in outputs:
        np.testing.assert_array_equal(value[[0,-1]],SENTINEL)
    return SimpleNamespace(status=status, required=required,
        f=outputs[0][1:-1].reshape(-1,spec.width),
        g=outputs[1][1:-1].reshape(-1,spec.width,spec.width),
        h=outputs[2][1:-1].reshape(-1,spec.width,spec.packed))


def filtered_design(spec, coordinates, data, time):
    """Independent finite convolution, not the native residual recurrence."""
    weights = inverse_lag_polynomial(coordinates[1+spec.p:1+spec.p+spec.q],
                                     time-spec.start+1)
    a = sum((weights[time-u]*mp.mpf(float(data[u]))
             for u in range(spec.start,time+1)),0)
    columns = [sum(weights,0)]
    for lag in range(1,spec.p+1):
        columns.append(sum((weights[time-u]*mp.mpf(float(data[u-lag]))
                            for u in range(spec.start,time+1)),0))
    return a, columns


def solve_by_elimination(matrix, rhs):
    # Oracle solves the AD-valued system directly, not differentiated inverse equations.
    a = [list(row)+[value] for row,value in zip(matrix,rhs)]
    for column in range(len(a)):
        pivot = max(range(column,len(a)),
                    key=lambda r: abs(getattr(a[r][column],"value",a[r][column])))
        a[column],a[pivot] = a[pivot],a[column]
        diagonal = a[column][column]
        a[column] = [v/diagonal for v in a[column]]
        for row in range(len(a)):
            if row != column:
                factor = a[row][column]
                a[row] = [v-factor*w for v,w in zip(a[row],a[column])]
    return [row[-1] for row in a]


def oracle(spec, origin, z, data, anchors, selected):
    with mp.workdps(100):
        width = spec.width
        variables = [Jet.variable(float(x),width,i) for i,x in enumerate(z)]
        original = [mp.mpf(float(x)) for x in origin]
        targets = []
        for time in anchors:
            a,b = filtered_design(spec,original,data,int(time))
            targets.append(a-sum((c*v for c,v in zip(b,original)),0))
        rows = [filtered_design(spec,variables,data,int(t)) for t in anchors]
        if spec.kind == 0:
            a0,b0 = filtered_design(spec,original,data,int(anchors[0]))
            error = -targets[0]/b0[0]
            intercept = (rows[0][0]-sum((rows[0][1][i]*variables[i]
                for i in range(1,1+spec.p)),0)-variables[0])/rows[0][1][0]+error
            variables[0] = intercept
        else:
            matrix, rhs = [], []
            for (a,b),target,r in zip(rows,targets,range(spec.k)):
                matrix.append([b[c] for c in selected])
                rhs.append(a-target-variables[selected[r]]
                    -sum((b[i]*variables[i] for i in range(1+spec.p)
                          if i not in selected),0))
            mapped = solve_by_elimination(matrix,rhs)
            for index,value in zip(selected,mapped):
                variables[index] = value
        return variables


def fixture(p=3,q=2,start=3,kind=1,k=2):
    spec = Spec(p,q,start,2,43,k,kind,7,0,0)
    rng = np.random.default_rng(91003+17*p+q)
    data = rng.normal(.1,.9,spec.n)
    origin = np.r_[.07,rng.uniform(-.12,.12,p),rng.uniform(-.1,.1,q),1.3,7.2]
    selected = list(range(k))
    anchors = [start+2+7*i for i in range(k)][::-1]
    z = origin.copy()
    z[selected] = np.linspace(-.13,.17,k)
    z[1+p:1+p+q] += .018
    return spec,origin,z,data,anchors,selected


def assert_oracle(actual, expected, spec, mask):
    assert actual.status == 0
    for bit,name in ((1,"f"),(2,"g"),(4,"h")):
        value = getattr(actual,name)[0]
        if mask & bit:
            if bit == 1:
                target = np.array([float(x.value) for x in expected])
            elif bit == 2:
                target = np.array([[float(v) for v in x.gradient] for x in expected])
            else:
                upper = np.triu_indices(spec.width)
                target = np.array([np.array([float(v) for v in x.hessian])
                    .reshape(spec.width,spec.width)[upper] for x in expected])
            assert np.isfinite(value).all(), f"nonfinite requested {name} output"
            assert np.isfinite(target).all(), f"nonfinite requested {name} oracle target"
            np.testing.assert_allclose(value,target,rtol=3e-12,atol=3e-12,equal_nan=False)
        else:
            np.testing.assert_array_equal(value,SENTINEL)


@pytest.mark.parametrize("case",[(0,0,1,0,1),(4,0,4,1,3),
                               (3,2,3,0,1),(3,2,3,1,2),(7,5,7,1,3)])
def test_off_origin_complete_curvature_and_masks(coordinate_probe,case):
    spec,origin,z,data,anchors,selected = fixture(*case)
    expected = oracle(spec,origin,z,data,anchors,selected)
    full = call(coordinate_probe,spec,origin,z,data,anchors,selected)
    for fixed in (0,1):
        spec.fixed = fixed
        for mask in range(1,8):
            spec.mask = mask
            actual = call(coordinate_probe,spec,origin,z,data,anchors,selected)
            assert_oracle(actual,expected,spec,mask)
            for bit,name in ((1,"f"),(2,"g"),(4,"h")):
                if mask & bit:
                    np.testing.assert_array_equal(getattr(actual,name),getattr(full,name))


def test_arbitrary_orders_and_nonconsecutive_selected_columns(coordinate_probe):
    spec,origin,z,data,anchors,_ = fixture(8,6,9,1,3)
    selected = [7,0,3]
    z = origin.copy()
    z[selected] = [.02,-.07,.12]
    assert_oracle(call(coordinate_probe,spec,origin,z,data,anchors,selected),
                  oracle(spec,origin,z,data,anchors,selected),spec,7)


@pytest.mark.parametrize("kind",[0,1])
def test_original_z_survives_rounded_theta_plateau(coordinate_probe,kind):
    spec,origin,z,data,anchors,selected = fixture(0,1,1,kind,1)
    spec.n = 3
    data = np.full(3,2.**60)
    origin = np.array([2.**60,.25,1.3,7.2])
    anchors = [2]
    z = origin.copy()
    z[0] = 1.
    second = z.copy()
    second[0] = 2.
    first_result = call(coordinate_probe,spec,origin,z,data,anchors,selected)
    second_result = call(coordinate_probe,spec,origin,second,data,anchors,selected)
    assert_oracle(first_result,oracle(spec,origin,z,data,anchors,selected),spec,7)
    assert_oracle(second_result,oracle(spec,origin,second,data,anchors,selected),spec,7)
    assert first_result.f[0,0] == second_result.f[0,0]
    # At fixed rounded intercept, d2(mu)/d(psi)^2 still depends on original rho.
    assert not np.array_equal(first_result.h,second_result.h)


@pytest.mark.parametrize("kind",[0,1])
def test_long_horizon_near_boundary_and_origin_reuse(coordinate_probe,kind):
    spec,origin,z,data,anchors,selected = fixture(0,1,1,kind,1)
    spec.n = 401
    data = np.cos(np.arange(spec.n)*.073)
    origin[1] = -.997
    z[1] = -.998
    anchors = [399]
    expected = oracle(spec,origin,z,data,anchors,selected)
    trial = z.copy()
    trial[0] += .2
    trial[1] -= .0001
    actual = call(coordinate_probe,spec,origin,np.array([z,trial,z]),data,anchors,selected)
    assert_oracle(actual,expected,spec,7)
    np.testing.assert_array_equal(actual.f[0],actual.f[2])
    np.testing.assert_array_equal(actual.g[0],actual.g[2])
    np.testing.assert_array_equal(actual.h[0],actual.h[2])
    middle = SimpleNamespace(status=actual.status,f=actual.f[1:2],
                             g=actual.g[1:2],h=actual.h[1:2])
    assert_oracle(middle,oracle(spec,origin,trial,data,anchors,selected),spec,7)


@pytest.mark.parametrize("kind",[0,1])
def test_oracle_zero_ma_closed_form(kind):
    spec,origin,z,data,anchors,selected = fixture(0,0,1,kind,1)
    result = oracle(spec,origin,z,data,anchors,selected)
    with mp.workdps(100):
        assert result[0].value == mp.mpf(float(origin[0]))-mp.mpf(float(z[0]))
    assert result[0].gradient[0] == -1
    assert all(value == 0 for value in result[0].hessian)


@pytest.mark.parametrize("mask",range(1,8))
@pytest.mark.parametrize("failure",["short","rank","range","duplicate","nonfinite","null","bad-layout"])
def test_failures_preserve_all_outputs(coordinate_probe,mask,failure):
    spec,origin,z,data,anchors,selected = fixture()
    spec.mask = mask
    short = missing = 0
    if failure == "short": short = 1
    if failure == "rank": data[:] = 1.
    if failure == "range":
        spec,origin,z,data,anchors,selected = fixture(0,0,1,1,1)
        spec.mask = mask
        origin[0] = -np.finfo(np.float64).max
        z[0] = 0.
        data[:] = np.finfo(np.float64).max
    if failure == "duplicate": anchors[1] = anchors[0]
    if failure == "nonfinite": z[-1] = np.nan
    if failure == "null": missing = mask & -mask
    if failure == "bad-layout": spec.mutation = 2
    actual = call(coordinate_probe,spec,origin,z,data,anchors,selected,short=short,missing=missing)
    assert actual.status == (-4 if failure == "rank" else -3 if failure == "range" else -1)
    for name in ("f","g","h"):
        np.testing.assert_array_equal(getattr(actual,name),SENTINEL)


def test_fixed_metadata_and_request_minimal_storage(coordinate_probe):
    spec,origin,z,data,anchors,selected = fixture()
    ordinary = call(coordinate_probe,spec,origin,z,data,anchors,selected)
    spec.fixed = 1
    spec.mutation = 1
    poisoned = call(coordinate_probe,spec,origin,z,data,anchors,selected)
    assert poisoned.status == 0
    for name in ("f","g","h"):
        np.testing.assert_array_equal(getattr(ordinary,name),getattr(poisoned,name))
    sizes = []
    for mask in (1,2,4):
        spec.mask = mask
        sizes.append(coordinate_probe.coordinate_006_workspace(ctypes.byref(spec)))
    assert 0 < sizes[0] < sizes[1] < sizes[2]
