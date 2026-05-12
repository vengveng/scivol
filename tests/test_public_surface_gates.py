from __future__ import annotations

import numpy as np
import pytest

import scivol
import scivol.components as components
from scivol.components.vol import EGARCH


def test_top_level_public_api_exports_egarch() -> None:
    assert "EGARCH" in scivol.__all__
    assert hasattr(scivol, "EGARCH")


def test_components_public_api_exports_egarch() -> None:
    assert "EGARCH" in components.__all__
    assert hasattr(components, "EGARCH")


def test_egarch_component_accepts_generic_orders() -> None:
    spec = EGARCH(2, 1)
    assert spec.p == 2
    assert spec.q == 1


@pytest.mark.parametrize("density", [scivol.SkewT(), scivol.GED()])
def test_egarch_extended_density_surfaces_are_shipped(density: object) -> None:
    spec = EGARCH(2, 1) + density
    result = spec.fit(np.random.default_rng(0).standard_normal(300), solver="slsqp", log_mode=True)

    assert result.params is not None
    assert np.all(np.isfinite(result.params))


@pytest.mark.parametrize("density", [scivol.Normal(), scivol.StudentT(), scivol.SkewT(), scivol.GED()])
def test_arma_egarch_family_surfaces_are_shipped(density: object) -> None:
    spec = scivol.ARMA(1, 1) + scivol.EGARCH(2, 1) + density
    result = spec.fit(np.random.default_rng(2).standard_normal(300), solver="slsqp", log_mode=True)

    assert result.params is not None
    assert np.all(np.isfinite(result.params))


@pytest.mark.parametrize(
    "density",
    [scivol.Normal(), scivol.StudentT(), scivol.SkewT(), scivol.GED()],
)
def test_arma_gjr_garch_family_surfaces_are_shipped(density: object) -> None:
    spec = scivol.ARMA(1, 1) + scivol.GJRGARCH(1, 1) + density
    result = spec.fit(np.random.default_rng(1).standard_normal(300), solver="slsqp", log_mode=True)

    assert result.params is not None
    assert np.all(np.isfinite(result.params))


@pytest.mark.parametrize(
    ("spec", "x"),
    [
        (scivol.ARX(1) + scivol.Normal(), np.zeros((240, 1), dtype=np.float64)),
        (scivol.HARX((1, 5)) + scivol.Normal(), np.zeros((240, 1), dtype=np.float64)),
    ],
)
def test_meanx_normal_standalone_surfaces_are_shipped(spec: object, x: np.ndarray) -> None:
    result = spec.fit(np.random.default_rng(7).standard_normal(x.shape[0]), x=x, solver="slsqp", log_mode=True)

    assert result.params is not None
    assert np.all(np.isfinite(result.params))


@pytest.mark.parametrize(
    ("spec", "x"),
    [
        (scivol.ARX(1) + scivol.StudentT(), np.zeros((240, 1), dtype=np.float64)),
        (scivol.HARX((1, 5)) + scivol.SkewT(), np.zeros((240, 1), dtype=np.float64)),
        (scivol.ARX(1) + scivol.GED(), np.zeros((240, 1), dtype=np.float64)),
    ],
)
def test_meanx_extended_standalone_surfaces_are_shipped(spec: object, x: np.ndarray) -> None:
    result = spec.fit(np.random.default_rng(8).standard_normal(x.shape[0]), x=x, solver="slsqp", log_mode=True)

    assert result.params is not None
    assert np.all(np.isfinite(result.params))


@pytest.mark.parametrize(
    ("spec", "x"),
    [
        (scivol.ARX(1) + scivol.GJRGARCH(1, 1) + scivol.Normal(), np.zeros((240, 1), dtype=np.float64)),
        (scivol.ARX(1) + scivol.GJRGARCH(1, 1) + scivol.StudentT(), np.zeros((240, 1), dtype=np.float64)),
        (scivol.HARX((1, 5)) + scivol.GJRGARCH(1, 1) + scivol.SkewT(), np.zeros((240, 1), dtype=np.float64)),
        (scivol.ARX(1) + scivol.GJRGARCH(1, 1) + scivol.GED(), np.zeros((240, 1), dtype=np.float64)),
    ],
)
def test_meanx_gjr_garch_family_surfaces_are_shipped(spec: object, x: np.ndarray) -> None:
    result = spec.fit(np.random.default_rng(4).standard_normal(x.shape[0]), x=x, solver="slsqp", log_mode=True)

    assert result.params is not None
    assert np.all(np.isfinite(result.params))


@pytest.mark.parametrize(
    ("spec", "x"),
    [
        (scivol.ARX(1) + scivol.EGARCH(1, 1) + scivol.Normal(), np.zeros((240, 1), dtype=np.float64)),
        (scivol.ARX(1) + scivol.EGARCH(1, 1) + scivol.StudentT(), np.zeros((240, 1), dtype=np.float64)),
        (scivol.HARX((1, 5)) + scivol.EGARCH(2, 1) + scivol.SkewT(), np.zeros((240, 1), dtype=np.float64)),
        (scivol.ARX(1) + scivol.EGARCH(2, 1) + scivol.GED(), np.zeros((240, 1), dtype=np.float64)),
    ],
)
def test_meanx_egarch_family_surfaces_are_shipped(spec: object, x: np.ndarray) -> None:
    result = spec.fit(np.random.default_rng(9).standard_normal(x.shape[0]), x=x, solver="slsqp", log_mode=True)

    assert result.params is not None
    assert np.all(np.isfinite(result.params))


def test_gjr_garch_ged_surface_is_shipped() -> None:
    spec = scivol.GJRGARCH(1, 1) + scivol.GED()
    result = spec.fit(np.random.default_rng(5).standard_normal(320), solver="slsqp", log_mode=True)

    assert result.params is not None
    assert np.all(np.isfinite(result.params))
