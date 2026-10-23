"""Weighted APARCH range evidence through the built extension's entry points."""

import numpy as np
import pytest

from scivol import _native_core
from test_jet_math_aparch_range import assert_requested, complete_oracle, sample
from test_native_aparch import _evaluate, _forced_runtime


def _linear_request(spec, theta, data, design, h0, mask, fixed):
    plan = (3, spec.density, spec.columns, spec.p, spec.q, data.size, fixed)
    size = _native_core._linear_joint_sizes(plan, mask, False, (0, 0, 0, 0))[0]
    workspace = np.empty(size, dtype=np.uint8)
    outputs = (np.full(1, np.nan), np.full(theta.size, np.nan),
               np.full((theta.size, theta.size), np.nan))
    assert _native_core._linear_joint_jet(
        plan, theta.ctypes.data, data.ctypes.data, design.ctypes.data, h0,
        mask, False, workspace.ctypes.data, workspace.nbytes,
        *(array.ctypes.data for array in outputs), 0, 0,
    )
    return outputs


@pytest.mark.parametrize('mode,wide', [(0, False), (0, True), (1, False), (1, True), (2, True)])
@pytest.mark.parametrize('density', [0, 1])
def test_weighted_range_public_entry_matches_complete_mp_ad(mode, wide, density):
    spec, theta, data, design, h0 = sample(mode, wide, density)
    expected, _ = complete_oracle(spec, theta, data, design, h0, 150)
    options = dict(orders=(spec.ar, spec.ma, spec.p, spec.q) if mode == 1 else None,
                   volatility_orders=(spec.p, spec.q), initial_variance=h0)
    for mask in range(1, 8):
        if mode == 2:
            results = [_linear_request(spec, theta, data, design, h0, mask, fixed)
                       for fixed in (False, True)]
        else:
            status, valid, *outputs = _evaluate(theta, data, density, mask, **options)
            assert (status, valid) == (0, mask)
            results = [outputs, _forced_runtime(theta, data, density, request=mask, **options)]
        for outputs in results:
            assert_requested(outputs, expected, mask)
            for bit, output in zip((1, 2, 4), outputs):
                if not mask & bit:
                    assert np.isnan(output).all()
