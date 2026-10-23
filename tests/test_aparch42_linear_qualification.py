"""Qualification of an unapplied original-coordinate linear adapter patch."""

import ctypes
import inspect

import numpy as np
import pytest

import scivol._joint_mean_fit as joint
from scivol import APARCH, ARX, HARX

def candidate_phase():
    source = inspect.getsource(joint._solve_joint_phase)
    if "optimizer_coordinates=" in source:
        return joint._solve_joint_phase
    old = "    return OptimizeResult(\n        x=theta,\n"
    new = (old + "        optimizer_coordinates=(point.copy()\n"
           "            if transformed and selected == \"returned\" else None),\n")
    assert source.count(old) == 1
    namespace = vars(joint).copy()
    exec(compile(source.replace(old, new), __file__, "exec"), namespace)
    return namespace["_solve_joint_phase"]


@pytest.mark.parametrize("density", (0, 1, 2))
@pytest.mark.parametrize("har,p,q,exog", ((False, 1, 1, 2), (True, 3, 2, 4), (False, 4, 3, 5)))
@pytest.mark.parametrize("curvature", (False, True))
def test_proposed_adapter_retains_native_unscaled_z(monkeypatch, density, har, p, q, exog, curvature):
    rng = np.random.default_rng(147)
    data, x = rng.normal(size=24), rng.normal(size=(24, exog))
    model = joint.JointMeanModel((HARX([1, 3]) if har else ARX(1)) + APARCH(p, q)
                                + joint.DENSITIES[density](), data, x)
    initial = model.initialize()
    original = joint._native_core._linear_joint_fit
    returned = []

    def capture(*args):
        valid = original(*args)
        point = np.ctypeslib.as_array(ctypes.cast(args[2], ctypes.POINTER(ctypes.c_double)),
                                     shape=(model.size,)).copy()
        scales = np.ctypeslib.as_array(ctypes.cast(args[8], ctypes.POINTER(ctypes.c_double)),
                                      shape=(model.size,)).copy()
        returned.append((point, scales))
        return valid

    monkeypatch.setattr(joint._native_core, "_linear_joint_fit", capture)
    result = candidate_phase()(model, initial, transformed=True, accuracy=1e-12,
                               maxiter=2, use_curvature=curvature)
    assert len(returned) == 1
    if result.selected == "returned":
        np.testing.assert_array_equal(result.optimizer_coordinates, returned[0][0])
        assert not np.shares_memory(result.optimizer_coordinates, returned[0][0])
        assert abs(model.evaluate(result.optimizer_coordinates, 1, transformed=True)[0]-result.fun) < 2e-11
        if curvature:
            assert np.any(returned[0][1] != 1)
    else:
        assert result.optimizer_coordinates is None


def test_proposed_adapter_does_not_inverse_rounded_skew_coordinate(monkeypatch):
    rng = np.random.default_rng(148)
    data, x = rng.normal(size=12), rng.normal(size=(12, 2))
    model = joint.JointMeanModel(ARX(1) + APARCH() + joint.DENSITIES[2](), data, x)
    z = model.inverse(model.initialize())
    z[-1] = 15.0
    from scivol._native_fit import _coordinates_to_theta

    initial = np.r_[z[:model.columns], _coordinates_to_theta(model.vol, z[model.columns:])]
    reconstructed = model.inverse(initial)
    assert reconstructed[-1] != z[-1]

    def same_mapped_point(*args):
        def output(index, length, dtype=ctypes.c_double):
            return np.ctypeslib.as_array(ctypes.cast(args[index], ctypes.POINTER(dtype)), shape=(length,))

        f, g = model.evaluate(initial, 3)[:2]
        output(2, model.size)[:] = z
        output(11, model.size)[:] = initial
        output(12, model.size)[:] = g
        output(13, 1)[:] = f
        output(14, 6, ctypes.c_int64)[:] = [0, 1, 1, 1, 0, 2]
        return True

    monkeypatch.setattr(joint._native_core, "_linear_joint_fit", same_mapped_point)
    result = candidate_phase()(model, initial, transformed=True, accuracy=1e-12,
                               maxiter=1, use_curvature=False)
    assert result.selected == "returned"
    np.testing.assert_array_equal(result.x, initial)
    np.testing.assert_array_equal(result.optimizer_coordinates, z)
    np.testing.assert_array_equal(model.inverse(result.x), reconstructed)
