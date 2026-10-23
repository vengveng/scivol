"""A rounded coupled proposal must not hide an available joint descent step."""

import numpy as np
import pytest

from benchmark_native_aparch import _simulate_aparch
from test_aparch42_integrated_phase_guards import PhaseCall, arena, guarded, integrated_core


THETA = np.array([.024979433829317617, -.21606035862591647,
    .3481627801756052, .022191677815429436, .06255033238257515,
    -.11324406440245911, .9175335046141988, .05248486277820603])
INCOMING_NLL = 581.2457629387867


def canonical_outputs(core, theta, data, h0, mask):
    size = core._arma_aparch_workspace_bytes(1, 1, 1, 1, 0, 0, mask)
    scratch = arena(size)
    outputs = [guarded(count) for count in (1, theta.size, theta.size**2)]
    pointers = [view.ctypes.data if mask & bit else 0
                for bit, (_, view) in zip((1, 2, 4), outputs)]
    status = core._arma_aparch_jet(theta.ctypes.data, theta.size,
        data.ctypes.data, data.size, h0, 1, 1, 1, 1, 0, 0, mask,
        scratch[1].ctypes.data, size, *pointers)
    assert status == (0, mask)
    PhaseCall(core, "", [], [], outputs, [scratch]).assert_canaries()
    return [view.copy() for _, view in outputs]


@pytest.mark.parametrize("runtime", (False, True))
def test_free_chart_recovers_descent_after_coupled_step_rounding(integrated_core, runtime):
    core = integrated_core
    theta = THETA.copy()
    data = _simulate_aparch(joint=True, observations=500)
    from scivol._native_fit import _initial_variance
    h0 = _initial_variance(data)
    assert h0 == .7077813774569074
    initial = canonical_outputs(core, theta, data, h0, 1)[0][0]
    assert abs(initial - INCOMING_NLL) <= 2e-11

    size = core._arma_aparch_multi_phase_workspace_bytes(1, 1, 1, 1, 0, 2, runtime, 1)
    scratch = arena(size)
    outputs = [guarded(2, np.uintp), guarded(2, np.uintp)]
    outputs += [guarded(theta.size), guarded(theta.size), guarded(16), guarded(12 + 2*theta.size)]
    args = [theta.ctypes.data, theta.size, data.ctypes.data, data.size, h0,
            1, 1, 1, 1, 0, 2, 1, scratch[1].ctypes.data, size]
    for _, view in outputs:
        args.extend((view.ctypes.data, view.size))
    args.extend((runtime, 1))
    call = PhaseCall(core, "_arma_aparch_multi_phase_run", args, [theta, data], outputs, [scratch])
    inputs = theta.tobytes(), data.tobytes()
    assert call.invoke() == 0
    call.assert_canaries()
    assert inputs == (theta.tobytes(), data.tobytes())
    continuation, best, summary, trace = [view for _, view in outputs[2:]]
    assert summary[3] == summary[4] == 1
    assert summary[1] < initial - 1e-4
    assert trace[4] >= theta.size + 2  # The fallback, not a coupled or ULP proposal.
    np.testing.assert_array_equal(best[:3], theta[:3])
    np.testing.assert_array_equal(best, continuation)
    assert best[-1] > theta[-1]

    full = canonical_outputs(core, best, data, h0, 7)
    assert abs(full[0][0] - summary[1]) <= 2e-11
    for mask in range(1, 8):
        actual = canonical_outputs(core, best, data, h0, mask)
        for bit, value, reference in zip((1, 2, 4), actual, full):
            if mask & bit:
                np.testing.assert_allclose(value, reference, rtol=0 if bit == 1 else 2e-10,
                                           atol=2e-11 if bit == 1 else 3e-9)
    # A successful proposal is not a stationarity or inference certificate.
    assert np.max(abs(full[1])) > 1e10
