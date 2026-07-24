from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from scivol import _core
from scivol._evaluation import (
    _hansen_skewt_cdf,
    _hansen_skewt_pdf_scalar,
    _hansen_skewt_ppf,
)


@pytest.mark.parametrize("lam", [-0.50, -0.15, 0.15, 0.50])
def test_hansen_skewt_is_zero_mean_and_unit_variance(lam: float) -> None:
    nu = 8.0

    mass = quad(lambda z: _hansen_skewt_pdf_scalar(z, nu, lam), -np.inf, np.inf)[0]
    mean = quad(lambda z: z * _hansen_skewt_pdf_scalar(z, nu, lam), -np.inf, np.inf)[0]
    second_moment = quad(
        lambda z: z * z * _hansen_skewt_pdf_scalar(z, nu, lam),
        -np.inf,
        np.inf,
    )[0]

    assert mass == pytest.approx(1.0, abs=2e-8)
    assert mean == pytest.approx(0.0, abs=2e-8)
    assert second_moment == pytest.approx(1.0, abs=2e-8)


@pytest.mark.parametrize("lam", [-0.50, -0.15, 0.0, 0.15, 0.50])
def test_hansen_skewt_cdf_and_ppf_are_inverses(lam: float) -> None:
    levels = np.array([1e-5, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0 - 1e-5])
    quantiles = _hansen_skewt_ppf(levels, 8.0, lam)

    np.testing.assert_allclose(_hansen_skewt_cdf(quantiles, 8.0, lam), levels, atol=2e-12)


def test_c_skewt_likelihood_matches_standardized_python_density() -> None:
    resid = np.array([-0.70, -0.15, 0.05, 0.40, 1.20], dtype=np.float64)
    sigma2 = np.array([0.80, 1.10, 0.95, 1.30, 1.05], dtype=np.float64)
    nu = 8.0
    lam = -0.25
    z = resid / np.sqrt(sigma2)
    expected = sum(
        np.log(_hansen_skewt_pdf_scalar(float(value), nu, lam))
        for value in z
    ) - 0.5 * np.log(sigma2).sum()

    actual = _core._skewt_ll(
        resid.ctypes.data,
        sigma2.ctypes.data,
        resid.size,
        nu,
        lam,
    )

    assert actual == pytest.approx(expected, rel=2e-10, abs=2e-10)
