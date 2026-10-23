"""Independent full joint derivatives when a raw APARCH moment overflows."""

import numpy as np

from scivol import APARCH, ARX, GeneralizedHyperbolic, _native_core
from scivol._joint_mean_fit import JointMeanModel
from scivol._native_fit import _aparch_persistence_evidence
from test_aparch42_gh_complete_stress import (
    DATA, G_ATOL, G_RTOL, H_ATOL, H_RTOL, NLL_ATOL, complete_oracle,
    jet_outputs, mp,
)
from test_aparch42_persistence_evidence import GH_OVERFLOW


def _joint_request(model, theta, mask):
    outputs = (np.full(1, np.nan), np.full(model.size, np.nan),
               np.full((model.size, model.size), np.nan))
    workspace = model.workspace(mask)
    assert _native_core._linear_joint_jet(
        model.descriptor, theta.ctypes.data, model.data.ctypes.data,
        model.features.ctypes.data, model.h0, mask, False,
        workspace.ctypes.data, workspace.nbytes,
        *(output.ctypes.data for output in outputs), 0, 0,
    )
    for bit, output in zip((1, 2, 4), outputs):
        assert np.isfinite(output).all() if mask & bit else np.isnan(output).all()
    return outputs


def _refined_joint_oracle(model, theta):
    low, high = [complete_oracle(theta, model.data, None, "gh", h0=model.h0,
        digits=digits, inner=nodes, cutoff=cutoff, design=model.features,
        first_scored=model.first)
        for digits, nodes, cutoff in ((60, 32, 20), (80, 40, 24))]
    with mp.workdps(80):
        assert abs(low.value-high.value) < mp.mpf("1e-20")
        for left, right in ((low.gradient, high.gradient), (low.hessian, high.hessian)):
            assert max(abs(a-b)/(1+abs(b)) for a, b in zip(left, right, strict=True)) < mp.mpf("1e-20")
    expected = jet_outputs(high)
    assert all(np.isfinite(output).all() for output in expected)
    assert expected[1].shape == (12,) and expected[2].shape == (12, 12)
    assert np.max(np.abs(expected[1])) > 1e60
    assert np.max(np.abs(expected[2])) > 1e120
    assert np.max(np.abs(expected[2][:4, 5])) > 1e60
    return expected


def test_weighted_overflow_full_joint_derivatives_and_all_sparse_requests():
    x = np.c_[np.sin(np.arange(DATA.size)), np.cos(np.arange(DATA.size))]
    models = [JointMeanModel(ARX(1)+APARCH(1, 1)+GeneralizedHyperbolic(), DATA, x, fixed=fixed)
              for fixed in (True, False)]
    model = models[0]
    assert model.columns == 4 and model.first == 0 and model.size == 12
    theta = np.ascontiguousarray(np.r_[np.zeros(model.columns), GH_OVERFLOW])
    shape, raw = np.ascontiguousarray(theta[-3:]), np.full(1, 12345.0)
    assert _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data, 4, theta[6], theta[8], 1, raw.ctypes.data, 0, 0,
    ) != 0
    assert raw[0] == 12345.0
    persistence, slack = _aparch_persistence_evidence(model.vol, theta[model.columns:])
    assert 0.0 < persistence < 1.0 and slack > 0.0
    expected = _refined_joint_oracle(model, theta)
    actual = []
    for model in models:
        saved = (theta.copy(), model.data.copy(), model.features.copy())
        matrix = {}
        for mask in range(1, 8):
            outputs = matrix[mask] = _joint_request(model, theta, mask)
            for bit, output, reference, relative, absolute in zip(
                    (1, 2, 4), outputs, expected, (0.0, G_RTOL, H_RTOL),
                    (NLL_ATOL, G_ATOL, H_ATOL)):
                if mask & bit:
                    np.testing.assert_allclose(output, reference, rtol=relative, atol=absolute)
        for before, after in zip(saved, (theta, model.data, model.features)):
            np.testing.assert_array_equal(after, before)
        actual.append(matrix)
    for mask in range(1, 8):
        for index, bit in enumerate((1, 2, 4)):
            if mask & bit:
                for matrix in actual:
                    np.testing.assert_allclose(matrix[mask][index], matrix[7][index],
                        rtol=0.0 if bit == 1 else 2e-10, atol=NLL_ATOL if bit == 1 else 2e-10)
                np.testing.assert_allclose(actual[0][mask][index], actual[1][mask][index],
                    rtol=0.0 if bit == 1 else 2e-10, atol=NLL_ATOL if bit == 1 else 2e-10)
