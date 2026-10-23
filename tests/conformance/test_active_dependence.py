"""Restricted fixed-input DCC objectives; never project an ambient AD jet.

Uses the existing independent scalar oracle and probe transport. Collection does
not load JAX or a native library. Run only in an authorized serial AD slot.
"""

from copy import deepcopy
import ctypes
from functools import lru_cache

import numpy as np
import pytest

from fixtures.dependence_cases import LAWS, REQUESTS, case, expected_dispatch, history_case, shape_case
from oracles.dependence import jax_modules, theta_objective
from test_dependence_mathematics import (
    NativeDependence, _Buffers, _Config, _DOUBLE, _Input, _Output, _Requirements,
    _SIZE, _Sample, _U32, _Validity, assert_close, assert_untouched, bridge,
)


@lru_cache(maxsize=1)
def selected_bridge():
    library = bridge()
    axes = [ctypes.POINTER(_SIZE), _SIZE]
    library.scivolapi_dependence_probe_selected_requirements.argtypes = [
        ctypes.POINTER(_Config), _U32, _U32, *axes, ctypes.POINTER(_Requirements)]
    library.scivolapi_dependence_probe_selected_evaluate.argtypes = [
        ctypes.POINTER(_Config), ctypes.POINTER(_Sample), _Input, _U32, _U32, _U32,
        *axes, ctypes.c_void_p, _SIZE, ctypes.POINTER(_Buffers), ctypes.POINTER(_Validity)]
    library.scivolapi_dependence_probe_selected_requirements.restype = _U32
    library.scivolapi_dependence_probe_selected_evaluate.restype = _U32
    return library


class SelectedDependence(NativeDependence):
    def __init__(self, declared, axes):
        super().__init__(declared)
        self.library = selected_bridge()
        self.axes = np.ascontiguousarray(axes, dtype=np.uintp)
        self.axis_pointer = self.axes.ctypes.data_as(ctypes.POINTER(_SIZE))

    def requirements(self, request, *, coordinates=0):
        needed = _Requirements()
        status = self.library.scivolapi_dependence_probe_selected_requirements(
            ctypes.byref(self.config), coordinates, request, self.axis_pointer,
            self.axes.size, ctypes.byref(needed))
        return status, needed

    def evaluate(self, request=7, *, prevalidated=False, bytes_delta=0):
        status, needed = self.requirements(request)
        assert status == 0 and needed.parameters == self.axes.size
        width = self.axes.size
        point = np.ascontiguousarray(self.case["parameters"], dtype=np.float64)
        sentinel = -987654321.125
        lengths = (1, width, width * width)
        storage = [np.full(length + 3, sentinel) for length in lengths]
        buffers = _Buffers(*[_Output(value.ctypes.data_as(_DOUBLE), value.size) for value in storage])
        capacity = needed.bytes + bytes_delta
        assert capacity >= 0
        arena = np.full(capacity + needed.alignment + 64, 165, dtype=np.uint8)
        start = 32 + (-(arena.ctypes.data + 32) % needed.alignment)
        validity = _Validity()
        status = self.library.scivolapi_dependence_probe_selected_evaluate(
            ctypes.byref(self.config), ctypes.byref(self.sample),
            _Input(point.ctypes.data_as(_DOUBLE), point.size), 0, request, prevalidated,
            self.axis_pointer, width, ctypes.c_void_p(arena.ctypes.data + start), capacity,
            ctypes.byref(buffers), ctypes.byref(validity))
        assert status == validity.status
        np.testing.assert_array_equal(arena[:start], 165)
        np.testing.assert_array_equal(arena[start + capacity:], 165)
        for value, length in zip(storage, lengths, strict=True):
            np.testing.assert_array_equal(value[length:], sentinel)
        return dict(f=storage[0][0], g=storage[1][:width], h=storage[2][:width * width].reshape(width, width),
                    status=status, valid_mask=validity.valid_mask, requested=validity.requested,
                    sentinel=sentinel, reasons=np.array([validity.f_reason, validity.g_reason, validity.h_reason]),
                    dispatch=needed.dispatch, workspace_bytes=needed.bytes)


def restricted_jet(declared, axes):
    jax, jnp, _ = jax_modules()
    base = jnp.asarray(declared["parameters"], dtype=jnp.float64)
    indices = jnp.asarray(axes, dtype=jnp.int32)
    scalar = theta_objective(declared, declared["parameters"])

    def objective(active):
        return scalar(base.at[indices].set(active))

    active = base[indices]
    return (float(objective(active)), np.asarray(jax.grad(objective)(active)),
            np.asarray(jax.jacfwd(jax.grad(objective))(active)))


def check_masks(declared, axes, expected):
    consumer = SelectedDependence(declared, axes)
    for prevalidated in (False, True):
        for mask in REQUESTS:
            actual = consumer.evaluate(mask, prevalidated=prevalidated)
            assert actual["status"] == 0 and actual["valid_mask"] == mask
            assert actual["requested"] == mask and actual["dispatch"] == expected_dispatch(declared)
            assert_untouched(actual, mask)
            assert_close(actual, expected, mask)
    return consumer


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("p,q,n", ((1, 1, 2), (3, 2, 3), (2, 3, 5), (1, 1, 7)))
def test_restricted_objective_all_masks_and_declared_dispatch(law, p, q, n):
    declared = history_case(law, p, q, n)
    width = len(declared["parameters"])
    selections = ((), (0,), (width - 1,), (width - 1, 0), tuple(reversed(range(width))))
    for axes in selections:
        expected = restricted_jet(declared, axes)
        for fixed in (False, True):
            fixed_case = deepcopy(declared)
            fixed_case["fixed"] = fixed
            consumer = check_masks(fixed_case, axes, expected)
            assert consumer.evaluate(7, bytes_delta=-1)["valid_mask"] == 0


@pytest.mark.parametrize("nu", (np.nextafter(2.0, np.inf), 128.0, 1e15))
@pytest.mark.parametrize("axes", ((1, 0), (2,), (1, 2, 0)))
def test_fixed_and_active_shape_range(nu, axes):
    declared = shape_case(nu)
    check_masks(declared, axes, restricted_jet(declared, axes))


@pytest.mark.parametrize("fixed", (False, True))
def test_finite_selected_hessian_without_ambient_hessian(fixed):
    # a is fixed at zero; target == seed and t=0 give Q=target for every b.
    # Thus the restricted objective is identically zero, independent of b.
    declared = case("normal", fixed=fixed, residuals=[[0.1, 0.2]],
                    target=np.eye(2) * 1e-200, seed=np.eye(2) * 1e-200,
                    residual_history=[[1.0, 1.0]], parameters=[0.0, 0.5])
    ambient = NativeDependence(declared).evaluate(7)
    assert ambient["valid_mask"] == 3
    assert ambient["reasons"][2] != 0
    for axes in ((), (1,)):
        expected = (0.0, np.zeros(len(axes)), np.zeros((len(axes), len(axes))))
        check_masks(declared, axes, expected)


def test_permutation_shape_not_last_and_reduced_arena():
    declared = case("studentt", p=3, q=2, series=4)
    first = SelectedDependence(declared, (5, 3, 0))
    second = SelectedDependence(declared, (0, 5, 3))
    a, b = first.evaluate(), second.evaluate()
    assert a["valid_mask"] == b["valid_mask"] == 7
    permutation = [2, 0, 1]
    np.testing.assert_allclose(b["g"], a["g"][permutation], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(b["h"], a["h"][np.ix_(permutation, permutation)], rtol=2e-12, atol=2e-12)
    assert_close(a, restricted_jet(declared, (5, 3, 0)), 7)
    scalar = SelectedDependence(declared, (5,)).requirements(7)[1]
    value = first.requirements(1)[1]
    ambient = NativeDependence(declared).requirements(7)[1]
    assert scalar.bytes == value.bytes < first.requirements(7)[1].bytes < ambient.bytes
