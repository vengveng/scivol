"""Original-coordinate preservation in joint linear-model solver phases."""

import ctypes

import numpy as np
import pytest

import scivol._joint_mean_fit as joint
from scivol import APARCH, ARX, HARX
from scivol._native_fit import _coordinates_to_theta

candidate = joint

def prepare_model(density, har=False, p=1, q=1, exog=2):
    rng = np.random.default_rng(148)
    data, x = rng.normal(size=24), rng.normal(size=(24, exog))
    model = joint.JointMeanModel((HARX([1, 3]) if har else ARX(1)) + APARCH(p, q)
                                + joint.DENSITIES[density](), data, x)
    z = model.inverse(model.initialize())
    if density == 2:
        z[-1] = 15.0
    theta = np.r_[z[:model.columns], _coordinates_to_theta(model.vol, z[model.columns:])]
    return model, theta, z


def native_array(args, index, size, dtype=ctypes.c_double):
    return np.ctypeslib.as_array(ctypes.cast(args[index], ctypes.POINTER(dtype)), shape=(size,))


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("layout", ((False, 1, 1, 2), (True, 3, 2, 4), (False, 4, 3, 5)))
@pytest.mark.parametrize("curvature", (False, True))
def test_original_z_enters_native_phase_without_inverse(monkeypatch, density, layout, curvature):
    model, theta, z = prepare_model(density, *layout)
    original = joint._native_core._linear_joint_fit
    incoming, returned = [], []

    def forbidden_inverse(_):
        raise AssertionError("The supplied original z must not be reconstructed.")

    def capture(*args):
        incoming.append((native_array(args, 2, model.size)*native_array(args, 8, model.size)).copy())
        valid = original(*args)
        returned.append(native_array(args, 2, model.size).copy())
        return valid

    monkeypatch.setattr(model, "inverse", forbidden_inverse)
    monkeypatch.setattr(joint._native_core, "_linear_joint_fit", capture)
    saved = z.copy()
    result = candidate._solve_joint_phase(model, theta, transformed=True, accuracy=1e-12,
        maxiter=2, use_curvature=curvature, initial_coordinates=z)
    np.testing.assert_array_equal(z, saved)
    np.testing.assert_array_equal(incoming[0], z)
    expected = returned[0] if result.selected == "returned" else z
    np.testing.assert_array_equal(result.optimizer_coordinates, expected)
    assert not np.shares_memory(result.optimizer_coordinates, z)
    mapped = np.r_[expected[:model.columns], _coordinates_to_theta(model.vol, expected[model.columns:])]
    np.testing.assert_array_equal(mapped, result.x)


@pytest.mark.parametrize("bad", ("width", "nonfinite", "theta-mismatch", "theta-space"))
def test_invalid_original_coordinates_rejected_before_native(monkeypatch, bad):
    model, theta, z = prepare_model(2)
    transformed = bad != "theta-space"
    if bad == "width":
        z = z[:-1]
    elif bad == "nonfinite":
        z[-1] = np.nan
    elif bad == "theta-mismatch":
        theta = theta.copy()
        theta[0] += 0.01
    calls = []
    monkeypatch.setattr(joint._native_core, "_linear_joint_fit", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="coordinates"):
        candidate._solve_joint_phase(model, theta, transformed=transformed, accuracy=1e-12,
            maxiter=1, use_curvature=False, initial_coordinates=z)
    assert not calls


def test_input_skew_plateau_uses_original_z_for_curvature(monkeypatch):
    model, theta, z = prepare_model(2)
    reconstructed = model.inverse(theta)
    assert reconstructed[-1] != z[-1]
    requests, original_evaluate = [], model.evaluate

    def evaluate(point, request=7, **kwargs):
        requests.append((np.array(point), request, kwargs.get("transformed", False)))
        return original_evaluate(point, request, **kwargs)

    def same_point(*args):
        native_array(args, 2, model.size)[:] = z
        native_array(args, 11, model.size)[:] = theta
        native_array(args, 12, model.size)[:] = original_evaluate(theta, 2)[1]
        native_array(args, 13, 1)[:] = original_evaluate(theta, 1)[0]
        native_array(args, 14, 6, ctypes.c_int64)[:] = [0, 1, 1, 1, 0, 2]
        return True

    monkeypatch.setattr(model, "evaluate", evaluate)
    monkeypatch.setattr(joint._native_core, "_linear_joint_fit", same_point)
    result = candidate._solve_joint_phase(model, theta, transformed=True, accuracy=1e-12,
        maxiter=1, initial_coordinates=z)
    curvature = [p for p, request, transformed in requests if request == 4 and transformed]
    assert len(curvature) == 1
    np.testing.assert_array_equal(curvature[0], z)
    np.testing.assert_array_equal(result.optimizer_coordinates, z)


def test_initial_fallback_retains_incoming_original_coordinates(monkeypatch):
    model, theta, z = prepare_model(2)
    def failed_return(*args):
        native_array(args, 2, model.size)[:] = 100.0
        native_array(args, 11, model.size)[:] = np.nan
        native_array(args, 14, 6, ctypes.c_int64)[:] = [-1, 1, 1, 1, 0, 2]
        return False
    monkeypatch.setattr(joint._native_core, "_linear_joint_fit", failed_return)
    result = candidate._solve_joint_phase(model, theta, transformed=True, accuracy=1e-12,
        maxiter=1, use_curvature=False, initial_coordinates=z)
    assert result.selected == "initial" and not result.success
    np.testing.assert_array_equal(result.x, theta)
    np.testing.assert_array_equal(result.optimizer_coordinates, z)


def test_same_chart_phase_handoff_keeps_native_coordinate(monkeypatch):
    model, theta, z = prepare_model(2)
    original = candidate._solve_joint_phase
    captures = []

    def phase(*args, **kwargs):
        result = original(*args, **kwargs)
        captures.append((None if kwargs.get("initial_coordinates") is None else
            kwargs["initial_coordinates"].copy(), result))
        return result

    monkeypatch.setattr(candidate, "_solve_joint_phase", phase)
    result = candidate.solve_joint(model, theta, transformed=True, accuracy=1e-12,
        maxiter=8, initial_coordinates=z, allow_chart_refinement=False)
    assert len(captures) >= 2
    np.testing.assert_array_equal(captures[0][0], z)
    for previous, following in zip(captures, captures[1:]):
        np.testing.assert_array_equal(following[0], previous[1].optimizer_coordinates)
    assert result.nit <= 8
