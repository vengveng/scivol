from __future__ import annotations

import numpy as np
import pytest

from test_native_aparch import _evaluate, _forced_runtime, _map, _native_core


GED_CASES = {
    "standalone11": (None, (1, 1), 1),
    "standalone32": (None, (3, 2), 0),
    "arma1111": ((1, 1, 1, 1), (1, 1), 1),
    "arma2243": ((2, 2, 4, 3), (4, 3), 0),
}


def ged_point(case: str) -> tuple[np.ndarray, np.ndarray, dict, int]:
    orders, volatility_orders, expected_dispatch = GED_CASES[case]
    p, q = volatility_orders
    mean = [] if orders is None else [0.03, *([0.06] * orders[0]), *([-0.04] * orders[1])]
    theta = np.r_[mean, 0.08, np.full(p, 0.07 / p), np.linspace(0.15, -0.10, p),
                  np.full(q, 0.72 / q), 1.7, 1.6]
    data = np.random.default_rng(421203).normal(size=12)
    layout = dict(orders=orders, volatility_orders=volatility_orders, initial_variance=0.73)
    return theta, data, layout, expected_dispatch


@pytest.mark.parametrize("case", GED_CASES)
@pytest.mark.parametrize("transformed", [False, True])
def test_ged_aparch_all_masks_and_dispatch(case: str, transformed: bool) -> None:
    theta, data, layout, dispatch = ged_point(case)
    map_layout = {key: layout[key] for key in ("orders", "volatility_orders")}
    point = _map(theta, 3, **map_layout) if transformed else theta
    orders = layout["orders"]
    actual_dispatch = (_native_core._aparch_dispatch_kind(*layout["volatility_orders"], 3, transformed)
                       if orders is None else _native_core._arma_aparch_dispatch_kind(*orders, 3, transformed))
    assert actual_dispatch == dispatch
    full = _evaluate(point, data, 3, transformed=transformed, **layout)
    assert full[:2] == (0, 7)
    for mask in range(1, 8):
        sparse = _evaluate(point, data, 3, mask, transformed=transformed, **layout)
        runtime = _forced_runtime(point, data, 3, request=mask, transformed=transformed, **layout)
        assert sparse[:2] == (0, mask)
        for bit, result, general, target in zip((1, 2, 4), sparse[2:], runtime, full[2:]):
            if mask & bit:
                assert np.isfinite(result).all() and np.isfinite(general).all()
                if bit == 1:
                    assert abs(result - target) <= 2e-11
                    assert abs(general - target) <= 2e-11
                else:
                    np.testing.assert_allclose(result, target, rtol=2e-11, atol=2e-11)
                    np.testing.assert_allclose(general, target, rtol=2e-11, atol=2e-11)
            else:
                assert np.isnan(result).all() and np.isnan(general).all()


@pytest.mark.parametrize("case", GED_CASES)
def test_ged_two_reduces_to_normal_in_common_parameters(case: str) -> None:
    theta, data, layout, _ = ged_point(case)
    theta[-1] = 2.0
    ged = _evaluate(theta, data, 3, **layout)
    normal = _evaluate(np.ascontiguousarray(theta[:-1]), data, 0, **layout)
    assert ged[:2] == normal[:2] == (0, 7)
    assert abs(ged[2] - normal[2]) <= 2e-11
    np.testing.assert_allclose(ged[3][:-1], normal[3], rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(ged[4][:-1, :-1], normal[4], rtol=2e-11, atol=2e-11)
