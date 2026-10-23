"""Future compiled-image guards. These tests never compile or modify an image."""

import ctypes
import numpy as np
import pytest


@pytest.fixture(scope="module")
def enabled_core():
    from test_aparch42_phase_accounting import enabled_integrated_core
    return enabled_integrated_core.__wrapped__()


@pytest.mark.parametrize("selector", (10, 11, 12))
@pytest.mark.parametrize("family", (0, 1, 2))
def test_linear_accounting_rejects_other_families_before_pointer_use(enabled_core, selector, family):
    descriptor = (family, 0, 4, 1, 1, 16, True)
    with pytest.raises(ValueError, match="require APARCH"):
        enabled_core._request_accounting_call(selector, (descriptor,), 0)


@pytest.mark.parametrize("selector", (10, 11, 12))
def test_linear_accounting_retains_original_parser_errors(enabled_core, selector):
    with pytest.raises(TypeError):
        enabled_core._request_accounting_call(selector, ((3, 0, 4, 1, 1, 16, True),), 0)


def test_linear_boundary_rejection_is_not_a_physical_jet(enabled_core):
    descriptor = (3, 0, 4, 1, 1, 16, True)
    arguments = (descriptor, 0, 0, 0, 1., 1, False, 0, 0, 0, 0, 0, 0, 0)
    result, record = enabled_core._request_accounting_call(10, arguments, 0)
    assert result is False and not any(record["physical"])


@pytest.mark.parametrize("mask", range(1, 8))
@pytest.mark.parametrize("transformed", (False, True))
def test_linear_jet_existing_physical_and_preparation_hooks(enabled_core, mask, transformed):
    descriptor, domain = (3, 0, 4, 1, 1, 16, True), (0, 0, 0, 0)
    y = np.array([.8, -.4, 1.1, .3, -1.2, .7, .6, -.2, 1.4, -.9, .5, .2, -.6, .9, .4, -.7])
    x = np.ascontiguousarray(np.c_[np.ones(16), np.linspace(-1., 1., 16), y[::-1], y**2])
    theta = np.array([.025]*4 + [.15, .04, .12, .45, 1.7])
    size = enabled_core._linear_joint_sizes(descriptor, mask, transformed, domain)[0]
    raw = np.empty(size + 16, dtype=np.uint8)
    work = raw[(-raw.ctypes.data) % 16:][:size]
    parameters = theta.copy()
    if transformed:
        assert enabled_core._linear_joint_inverse(descriptor, theta.ctypes.data,
            parameters.ctypes.data, work.ctypes.data, work.nbytes)
    outputs = [np.full(1, 271828.125), np.full(9, 271828.125), np.full(81, 271828.125)]
    addresses = [output.ctypes.data if mask & bit else 0 for output, bit in zip(outputs, (1, 2, 4))]
    arguments = (descriptor, parameters.ctypes.data, y.ctypes.data, x.ctypes.data,
                 float(np.mean(y*y)), mask, transformed, work.ctypes.data, work.nbytes,
                 *addresses, 0, 0)
    plain = enabled_core._linear_joint_prevalidated(*arguments)
    expected = [output.tobytes() for output in outputs]
    result, record = enabled_core._request_accounting_call(10, arguments, 0)
    assert result is plain is True
    assert [output.tobytes() for output in outputs] == expected
    physical = np.asarray(record["physical"]).reshape(5, 2, 8)
    assert physical[0, int(transformed), mask] == physical.sum() == 1
    assert not any(record["physical_failures"])
    preparations = np.asarray(record["preparation"]).reshape(5, 7, 3)
    order = 2 if mask & 4 else 1 if mask & 2 else 0
    assert preparations[0, 1, order] == 1
    assert preparations[0, 2, order] == int(transformed)
    # Zero in the unhooked categories is deliberately not evidence of zero work.
    assert not any(record["cache_hits"])


def test_linear_fit_and_refine_hooks_preserve_existing_outputs(enabled_core):
    from unittest.mock import patch
    import scivol._joint_mean_fit as joint
    import scivol._active_face_fit as active
    from scivol import APARCH, ARX, Normal

    records = {}
    class Capture:
        def __getattr__(self, name):
            entry = getattr(enabled_core, name)
            if name not in ("_linear_joint_fit", "_linear_joint_refine"):
                return entry
            def call(*args):
                original = ctypes.string_at(args[2], 9*8)
                layout = ((2, 9*8), (11, 9*8), (12, 9*8), (13, 8), (14, 6*8), (21, 3*8)) \
                    if name == "_linear_joint_fit" else ((2, 9*8), (3, 9*8), (12, 6*8), (13, 16*ctypes.sizeof(ctypes.c_size_t)))
                plain = entry(*args)
                expected = [ctypes.string_at(args[index], size) for index, size in layout]
                ctypes.memmove(args[2], original, len(original))
                selector = 11 if name == "_linear_joint_fit" else 12
                result, record = enabled_core._request_accounting_call(selector, args, selector-10)
                assert result == plain
                assert [ctypes.string_at(args[index], size) for index, size in layout] == expected
                existing = np.ctypeslib.as_array((ctypes.c_int64*3).from_address(args[21])) \
                    if selector == 11 else np.ctypeslib.as_array((ctypes.c_size_t*8).from_address(args[13]))
                assert sum(record["physical"]) == int(existing.sum())
                records[name] = record
                return plain
            return call

    rng = np.random.default_rng(94028)
    x, y = rng.normal(size=(16, 3)), rng.normal(size=16)
    with patch.object(joint, "_native_core", Capture()), patch.object(active, "_native_core", Capture()):
        model = joint.JointMeanModel(ARX()+APARCH()+Normal(), y, x)
        initial = np.array([.025]*4 + [.15, .04, .12, .45, 1.7])
        phase = joint._solve_joint_phase(model, initial, transformed=False, accuracy=1e-12, maxiter=2)
        assert any(records["_linear_joint_fit"]["physical"])
        assert any(records["_linear_joint_fit"]["solver_requests"])
        active.native_refinement(model, phase.x, backtracks=1)
        assert any(records["_linear_joint_refine"]["physical"])
        assert not any(records["_linear_joint_refine"]["solver_requests"])
