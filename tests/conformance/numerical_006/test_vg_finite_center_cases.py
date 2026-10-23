"""Exact historical finite centre inputs; no Python-array API claim."""
import mpmath as mp
import numpy as np
import pytest
from .transport import local_density
from .test_vg_retained_value_accuracy import scalar_vg_nll

@pytest.mark.parametrize("residual", (1e-8, -1e-8, 0., 1e-200, 5.7735026918962544e-6))
@pytest.mark.parametrize("shape", (1.5000000000000018, 50.))
def test_historical_finite_centres(residual, shape):
    with mp.workdps(90):
        target = float(scalar_vg_nll(mp.mpf(residual), mp.mpf(1), mp.mpf(0), mp.mpf(shape)))
    for mask in (1, 3, 7):
        actual = local_density(7, [residual, 1., 0., shape], mask)[0][0]
        assert np.isfinite(actual)
        assert abs(actual - target) <= 3e-13
