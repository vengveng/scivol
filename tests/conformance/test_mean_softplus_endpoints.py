"""Original-z mean-scale regressions; chief schedules the explicit native bridge.

Complete likelihood derivatives come from the independent JAX theta oracle at
actual mapped doubles. Decimal supplies the scale-map derivatives: differentiating
JAX softplus directly can erase positive-tail curvature after its slope rounds to
one. Subnormal weighted-map plateaus are covered by native supplied-jet fixtures,
not by a JAX likelihood with overflowing theta derivatives.
"""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from fixtures.mean_cases import arma_case, chart_case, linear_case, mean_count
from oracles.garch import jax_modules
from oracles.mean import coordinate_map, full_jet


def scale_derivatives(coordinate):
    with localcontext() as context:
        context.prec = 100
        tail = (-Decimal.from_float(float(coordinate))).exp()
        denominator = 1 + tail
        return float(1 / denominator), float(tail / denominator**2)


def anchored_pullback(case, theta_jet):
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(case["original_z"])
    mapping = coordinate_map(case)
    jacobian = np.array(jax.jacfwd(mapping)(point))
    curvature = np.array(jax.jacfwd(jax.jacfwd(mapping))(point))
    scale = mean_count(case)
    first, second = scale_derivatives(case["original_z"][scale])
    jacobian[scale, :] = 0
    jacobian[scale, scale] = first
    curvature[scale, :, :] = 0
    curvature[scale, scale, scale] = second
    f, g, h = theta_jet
    return f, jacobian.T @ g, jacobian.T @ h @ jacobian + np.einsum("i,ijk->jk", g, curvature)


@pytest.mark.parametrize("layout", ("arma11", "arma32", "linear4", "linear9"))
@pytest.mark.parametrize("coordinate", (-36.0, 50.0))
def test_mean_scale_original_z(native, layout, coordinate):
    if layout.startswith("arma"):
        p, q = (1, 1) if layout == "arma11" else (3, 2)
        declared = chart_case(arma_case(p=p, q=q, first=6, stop=11))
    else:
        columns = 4 if layout == "linear4" else 9
        declared = chart_case(linear_case(columns=columns, intercept=False, first=4, stop=11))
    scale = mean_count(declared)
    declared["parameters"][scale] = coordinate
    # Keep standardized residuals ordinary and the theta Hessian representable.
    if coordinate < 0:
        root_variance = np.sqrt(np.logaddexp(0.0, coordinate))
        declared["y"] *= root_variance
        if declared["model"] == "arma":
            declared["parameters"][0] *= root_variance
        else:
            declared["parameters"][:scale] *= root_variance
    declared["original_z"] = declared["parameters"].copy()
    full = native(declared, 7)
    assert full["status"] == 0 and full["valid_mask"] == 7, full.get("reasons")
    expected = anchored_pullback(declared, full_jet(declared, theta=full["theta"]))
    slope, _ = scale_derivatives(coordinate)
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
        if mask & 2:
            np.testing.assert_allclose(actual["g"][scale] / slope, expected[1][scale] / slope,
                                       rtol=3e-7, atol=3e-7)
        if mask & 4:
            np.testing.assert_allclose(actual["h"][scale, :] / slope, expected[2][scale, :] / slope,
                                       rtol=4e-6, atol=4e-6)
            np.testing.assert_allclose(actual["h"][:, scale] / slope, expected[2][:, scale] / slope,
                                       rtol=4e-6, atol=4e-6)
