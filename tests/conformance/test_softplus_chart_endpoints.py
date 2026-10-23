"""Focused complete-jet AD checks, scheduled by chief against an explicit bridge.

Subnormal weighted-map plateaus have independent native fixtures. These complete
likelihood cases keep the original-coordinate slope normal, avoiding JAX's
subnormal flushing. Theta derivatives are anchored to actual mapped doubles.
"""

import mpmath as mp
import numpy as np
import pytest

from fixtures.garch_cases import chart_case as zero_case
from fixtures.joint_cases import case as joint_case, chart_case as joint_chart, mean_width
from oracles.garch import full_jet as zero_jet
from oracles.charts import pullback as zero_pullback
from oracles.joint import full_jet as joint_jet, pullback as joint_pullback


@pytest.mark.parametrize("model", ("garch", "gjr", "arma_garch", "arma_gjr", "linear_garch", "linear_gjr"))
@pytest.mark.parametrize("p,q,coordinate", ((1, 1, -700.0), (3, 2, -700.0), (1, 1, 50.0)))
def test_omega_original_z_complete_jet(native, model, p, q, coordinate):
    joint = "_" in model
    if joint:
        declared = joint_chart(joint_case(model, "normal", p_vol=p, q_vol=q))
        omega = mean_width(declared)
        theta_jet, pullback = joint_jet, joint_pullback
    else:
        declared = zero_case(model, "normal", p, q)
        omega = 0
        theta_jet, pullback = zero_jet, zero_pullback
    declared["parameters"][omega] = coordinate
    declared["original_z"] = declared["parameters"].copy()
    full = native(declared, 7)
    assert full["status"] == 0 and full["valid_mask"] == 7, full.get("reasons")
    theta_reference = theta_jet(declared, theta=full["theta"])
    expected = pullback(declared, theta_reference)
    with mp.workdps(90):
        z = mp.mpf(coordinate)
        slope = float(1 / (1 + mp.exp(-z)))
    assert slope > 0
    for mask in range(1, 8):
        actual = full if mask == 7 else native(declared, mask)
        assert actual["status"] == 0 and actual["valid_mask"] == mask, actual.get("reasons")
        np.testing.assert_array_equal(actual["theta"], full["theta"])
        np.testing.assert_array_equal(actual["scored_rows"], declared["scored_rows"])
        for bit, name, reference in zip((1, 2, 4), ("f", "g", "h"), expected):
            output = np.asarray(actual[name])
            if mask & bit:
                assert np.isfinite(output).all() and np.isfinite(reference).all()
                np.testing.assert_allclose(output, reference, rtol=0 if bit == 1 else 3e-7,
                                           atol=2e-9 if bit == 1 else 3e-7)
            else:
                sentinel = np.full_like(output, actual["sentinel"])
                np.testing.assert_array_equal(output.view(np.uint64), sentinel.view(np.uint64))
        # Ordinary absolute tolerances would hide an erased omega row.
        if mask & 2:
            np.testing.assert_allclose(actual["g"][omega] / slope, expected[1][omega] / slope,
                                       rtol=3e-7, atol=3e-7)
        if mask & 4:
            np.testing.assert_allclose(actual["h"][omega, :] / slope, expected[2][omega, :] / slope,
                                       rtol=4e-6, atol=4e-6)
            np.testing.assert_allclose(actual["h"][:, omega] / slope, expected[2][:, omega] / slope,
                                       rtol=4e-6, atol=4e-6)
