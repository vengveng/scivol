"""APARCH 4.2 range blocker: smooth supported GH/NIG moment H is unavailable.

This is deliberately a failing release-gate regression on core019, not an xfail
or a qualification claim. Run against the frozen image through the local shard
loader. Ordinary all-mask AD coverage remains in test_native_aparch.py.
"""

import numpy as np
import pytest

from scivol import _native_core


@pytest.mark.parametrize("density,shape", [(5, [-.994, 24.999999]),
                                          (4, [-.994, 24.999999, -.5])])
@pytest.mark.parametrize("request", range(1, 8))
def test_supported_gh_nig_power_moment_has_requested_derivative_order(density, shape, request):
    shape = np.asarray(shape, dtype=np.float64)
    width = shape.size+2
    value = np.full(1, np.nan)
    gradient = np.full(width, np.nan)
    hessian = np.full((width, width), np.nan)
    status = _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data, density, .15, 1.7, request,
        value.ctypes.data, gradient.ctypes.data, hessian.ctypes.data,
    )
    assert status == 0, (
        "Supported smooth point rejected. core019 exhausts quadrature depth on rho-rho curvature",
        density, request, status,
    )
    for bit, output in ((1, value), (2, gradient), (4, hessian)):
        if request & bit:
            assert np.isfinite(output).all()
        else:
            assert np.isnan(output).all()
