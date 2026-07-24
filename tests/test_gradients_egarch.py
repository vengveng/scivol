"""
Derivative validation for the shipped EGARCH C kernels.

Uses an independent AD oracle rather than finite differences:
- EGARCH(p,q) + Normal: gradient and Hessian
- EGARCH(p,q) + Student-t: gradient and Hessian
- EGARCH(p,q) + Skew-t: gradient and Hessian
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

import scivol._core as _c
from scivol._devtools.ad_oracle import egarch_logspace_value_grad_hess, egarch_value_grad_hess
from scivol._evaluation import _egarch_abs_moment, _hansen_skewt_ppf
from scivol._kernels.transforms import (
    log_hessian_egarch,
    pack_egarch,
    pack_egarch_ged,
    pack_egarch_skewt,
    pack_egarch_studentt,
    unpack_egarch,
    unpack_egarch_ged,
    unpack_egarch_skewt,
    unpack_egarch_studentt,
)


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _make_egarch_data(
    n: int = 700,
    seed: int = 42,
    dist: str = "normal",
    p: int = 1,
    q: int = 1,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    omega = -0.45
    alpha = np.full(p, 0.12 / p, dtype=np.float64)
    gamma = np.zeros(p, dtype=np.float64)
    gamma[0] = -0.06
    beta = np.full(q, 0.95 / q, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    if dist == "normal":
        z = rng.standard_normal(n)
        abs_moment = float(np.sqrt(2.0 / np.pi))
    elif dist == "studentt":
        nu = 8.0
        z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
        abs_moment = _egarch_abs_moment("StudentT", nu, None)
    elif dist == "skewt":
        nu, lam = 8.0, -0.15
        z = _hansen_skewt_ppf(rng.uniform(size=n), nu, lam)
        abs_moment = _egarch_abs_moment("SkewT", nu, lam)
    else:
        raise ValueError(dist)

    sigma2[0] = np.exp(omega / (1.0 - float(np.sum(beta))))
    y[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        logh_t = omega
        for i in range(p):
            if t > i:
                z_lag = z[t - 1 - i]
                logh_t += alpha[i] * (abs(z_lag) - abs_moment) + gamma[i] * z_lag
        for j in range(q):
            if t > j:
                logh_t += beta[j] * np.log(sigma2[t - 1 - j])
        sigma2[t] = np.exp(logh_t)
        y[t] = np.sqrt(sigma2[t]) * z[t]
    return np.ascontiguousarray(y, dtype=np.float64)


@pytest.fixture(scope="module")
def resid_normal() -> NDArray[np.float64]:
    return _make_egarch_data(dist="normal")


@pytest.fixture(scope="module")
def resid_studentt() -> NDArray[np.float64]:
    return _make_egarch_data(dist="studentt")


@pytest.fixture(scope="module")
def resid_skewt() -> NDArray[np.float64]:
    return _make_egarch_data(dist="skewt")


def _sigma_seed(resid: NDArray[np.float64]) -> NDArray[np.float64]:
    sigma2 = np.zeros(len(resid), dtype=np.float64)
    sigma2[0] = np.mean(resid ** 2)
    return sigma2


def _c_egarch_grad_hess(
    params: NDArray[np.float64],
    resid: NDArray[np.float64],
    dist: str,
    p: int = 1,
    q: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    params = np.ascontiguousarray(params, dtype=np.float64)
    resid = np.ascontiguousarray(resid, dtype=np.float64)
    sigma_g = _sigma_seed(resid)
    sigma_h = _sigma_seed(resid)

    if dist == "normal":
        grad_fn = _c._egarch_ll_grad_11_normal if (p, q) == (1, 1) else _c._egarch_ll_grad_pq_normal
        hess_fn = _c._egarch_ll_hess_11_normal if (p, q) == (1, 1) else _c._egarch_ll_hess_pq_normal
    elif dist == "studentt":
        grad_fn = _c._egarch_ll_grad_11_studentt if (p, q) == (1, 1) else _c._egarch_ll_grad_pq_studentt
        hess_fn = _c._egarch_ll_hess_11_studentt if (p, q) == (1, 1) else _c._egarch_ll_hess_pq_studentt
    elif dist == "skewt":
        grad_fn = _c._egarch_ll_grad_11_skewt if (p, q) == (1, 1) else _c._egarch_ll_grad_pq_skewt
        hess_fn = _c._egarch_ll_hess_11_skewt if (p, q) == (1, 1) else _c._egarch_ll_hess_pq_skewt
    elif dist == "ged":
        grad_fn = _c._egarch_ll_grad_11_ged if (p, q) == (1, 1) else _c._egarch_ll_grad_pq_ged
        hess_fn = _c._egarch_ll_hess_11_ged if (p, q) == (1, 1) else _c._egarch_ll_hess_pq_ged
    else:
        raise ValueError(dist)

    grad = np.zeros(len(params), dtype=np.float64)
    hess = np.zeros(len(params) * len(params), dtype=np.float64)
    if (p, q) == (1, 1):
        grad_fn(_as_cptr(params), _as_cptr(resid), _as_cptr(sigma_g), _as_cptr(grad), len(resid))
        hess_fn(_as_cptr(params), _as_cptr(resid), _as_cptr(sigma_h), _as_cptr(hess), len(resid))
    else:
        grad_fn(_as_cptr(params), _as_cptr(resid), _as_cptr(sigma_g), _as_cptr(grad), len(resid), p, q)
        hess_fn(_as_cptr(params), _as_cptr(resid), _as_cptr(sigma_h), _as_cptr(hess), len(resid), p, q)
    return grad, hess.reshape(len(params), len(params))


def _c_egarch_value(
    params: NDArray[np.float64],
    resid: NDArray[np.float64],
    dist: str,
    p: int = 1,
    q: int = 1,
) -> float:
    params = np.ascontiguousarray(params, dtype=np.float64)
    resid = np.ascontiguousarray(resid, dtype=np.float64)
    sigma = _sigma_seed(resid)
    if dist == "normal":
        fn = _c._egarch_ll_11_normal if (p, q) == (1, 1) else _c._egarch_ll_pq_normal
    elif dist == "studentt":
        fn = _c._egarch_ll_11_studentt if (p, q) == (1, 1) else _c._egarch_ll_pq_studentt
    elif dist == "skewt":
        fn = _c._egarch_ll_11_skewt if (p, q) == (1, 1) else _c._egarch_ll_pq_skewt
    elif dist == "ged":
        fn = _c._egarch_ll_11_ged if (p, q) == (1, 1) else _c._egarch_ll_pq_ged
    else:
        raise ValueError(dist)
    if (p, q) == (1, 1):
        return float(fn(_as_cptr(params), _as_cptr(resid), _as_cptr(sigma), len(resid)))
    return float(fn(_as_cptr(params), _as_cptr(resid), _as_cptr(sigma), len(resid), p, q))


@pytest.fixture(scope="module")
def resid_normal_pq() -> NDArray[np.float64]:
    return _make_egarch_data(dist="normal", p=2, q=1)


@pytest.fixture(scope="module")
def resid_studentt_pq() -> NDArray[np.float64]:
    return _make_egarch_data(dist="studentt", p=2, q=1)


@pytest.fixture(scope="module")
def resid_skewt_pq() -> NDArray[np.float64]:
    return _make_egarch_data(dist="skewt", p=2, q=1)


class TestEGARCHNormalAD:
    @pytest.mark.parametrize(
        "params",
        [
            np.array([-0.45, 0.12, -0.06, 0.95], dtype=np.float64),
            np.array([-0.30, 0.08, 0.03, 0.90], dtype=np.float64),
        ],
    )
    def test_gradient_and_hessian_match_ad(
        self,
        resid_normal: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> None:
        grad_c, hess_c = _c_egarch_grad_hess(params, resid_normal, "normal")
        _, grad_ref, hess_ref = egarch_value_grad_hess(params, resid_normal, dist="normal")
        np.testing.assert_allclose(grad_c, grad_ref, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(hess_c, hess_ref, rtol=1e-6, atol=1e-7)

    def test_hessian_symmetry(self, resid_normal: NDArray[np.float64]) -> None:
        _, hess_c = _c_egarch_grad_hess(np.array([-0.45, 0.12, -0.06, 0.95]), resid_normal, "normal")
        np.testing.assert_allclose(hess_c, hess_c.T, rtol=1e-10, atol=1e-12)


class TestEGARCHStudentTAD:
    @pytest.mark.parametrize("nu", [4.0, 8.0, 20.0])
    def test_gradient_and_hessian_match_ad(self, resid_studentt: NDArray[np.float64], nu: float) -> None:
        params = np.array([-0.45, 0.12, -0.06, 0.95, nu], dtype=np.float64)
        grad_c, hess_c = _c_egarch_grad_hess(params, resid_studentt, "studentt")
        _, grad_ref, hess_ref = egarch_value_grad_hess(params, resid_studentt, dist="studentt")
        np.testing.assert_allclose(grad_c, grad_ref, rtol=2e-4, atol=5e-6)
        np.testing.assert_allclose(hess_c, hess_ref, rtol=3e-4, atol=3e-5)

    def test_hessian_symmetry(self, resid_studentt: NDArray[np.float64]) -> None:
        _, hess_c = _c_egarch_grad_hess(np.array([-0.45, 0.12, -0.06, 0.95, 8.0]), resid_studentt, "studentt")
        np.testing.assert_allclose(hess_c, hess_c.T, rtol=1e-10, atol=1e-12)


class TestEGARCHPQNormalAD:
    def test_gradient_and_hessian_match_ad(self, resid_normal_pq: NDArray[np.float64]) -> None:
        params = np.array([-0.45, 0.08, 0.04, -0.06, 0.02, 0.95], dtype=np.float64)
        grad_c, hess_c = _c_egarch_grad_hess(params, resid_normal_pq, "normal", p=2, q=1)
        _, grad_ref, hess_ref = egarch_value_grad_hess(params, resid_normal_pq, dist="normal", p=2, q=1)
        np.testing.assert_allclose(grad_c, grad_ref, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(hess_c, hess_ref, rtol=2e-5, atol=2e-6)


class TestEGARCHPQStudentTAD:
    def test_gradient_and_hessian_match_ad(self, resid_studentt_pq: NDArray[np.float64]) -> None:
        params = np.array([-0.45, 0.08, 0.04, -0.06, 0.02, 0.95, 8.0], dtype=np.float64)
        grad_c, hess_c = _c_egarch_grad_hess(params, resid_studentt_pq, "studentt", p=2, q=1)
        _, grad_ref, hess_ref = egarch_value_grad_hess(params, resid_studentt_pq, dist="studentt", p=2, q=1)
        np.testing.assert_allclose(grad_c, grad_ref, rtol=5e-4, atol=1e-5)
        np.testing.assert_allclose(hess_c, hess_ref, rtol=6e-4, atol=3e-5)


class TestEGARCHSkewTAD:
    def test_gradient_matches_ad(self, resid_skewt: NDArray[np.float64]) -> None:
        params = np.array([-0.45, 0.12, -0.06, 0.95, 8.0, -0.15], dtype=np.float64)
        grad_c, _ = _c_egarch_grad_hess(params, resid_skewt, "skewt")
        _, grad_ref, _ = egarch_value_grad_hess(params, resid_skewt, dist="skewt")
        np.testing.assert_allclose(grad_c, grad_ref, rtol=3e-2, atol=2e-3)

    def test_full_hessian_matches_ad(self, resid_skewt: NDArray[np.float64]) -> None:
        params = np.array([-0.45, 0.12, -0.06, 0.95, 8.0, -0.15], dtype=np.float64)
        _, hess_c = _c_egarch_grad_hess(params, resid_skewt, "skewt")
        _, _, hess_ref = egarch_value_grad_hess(params, resid_skewt, dist="skewt")
        np.testing.assert_allclose(hess_c, hess_ref, rtol=3e-2, atol=5e-2)

    def test_hessian_symmetry(self, resid_skewt: NDArray[np.float64]) -> None:
        _, hess_c = _c_egarch_grad_hess(np.array([-0.45, 0.12, -0.06, 0.95, 8.0, -0.15]), resid_skewt, "skewt")
        np.testing.assert_allclose(hess_c, hess_c.T, rtol=1e-10, atol=1e-12)


class TestEGARCHPQSkewTAD:
    def test_gradient_matches_ad(self, resid_skewt_pq: NDArray[np.float64]) -> None:
        params = np.array([-0.45, 0.08, 0.04, -0.06, 0.02, 0.95, 8.0, -0.15], dtype=np.float64)
        grad_c, _ = _c_egarch_grad_hess(params, resid_skewt_pq, "skewt", p=2, q=1)
        _, grad_ref, _ = egarch_value_grad_hess(params, resid_skewt_pq, dist="skewt", p=2, q=1)
        np.testing.assert_allclose(grad_c, grad_ref, rtol=6e-2, atol=3e-3)

    def test_full_hessian_matches_ad(self, resid_skewt_pq: NDArray[np.float64]) -> None:
        params = np.array([-0.45, 0.08, 0.04, -0.06, 0.02, 0.95, 8.0, -0.15], dtype=np.float64)
        _, hess_c = _c_egarch_grad_hess(params, resid_skewt_pq, "skewt", p=2, q=1)
        _, _, hess_ref = egarch_value_grad_hess(params, resid_skewt_pq, dist="skewt", p=2, q=1)
        np.testing.assert_allclose(hess_c, hess_ref, rtol=3e-2, atol=6e-2)

    def test_hessian_symmetry(self, resid_skewt_pq: NDArray[np.float64]) -> None:
        params = np.array([-0.45, 0.08, 0.04, -0.06, 0.02, 0.95, 8.0, -0.15], dtype=np.float64)
        _, hess_c = _c_egarch_grad_hess(params, resid_skewt_pq, "skewt", p=2, q=1)
        np.testing.assert_allclose(hess_c, hess_c.T, rtol=1e-10, atol=1e-12)


class TestEGARCHGEDAD:
    def test_ged_11_gradient_and_hessian_match_ad(self, resid_normal: NDArray[np.float64]) -> None:
        params = np.array([-0.45, 0.12, -0.06, 0.95, 1.6], dtype=np.float64)
        value_c = _c_egarch_value(params, resid_normal, "ged")
        grad_c, hess_c = _c_egarch_grad_hess(params, resid_normal, "ged")
        value_ref, grad_ref, hess_ref = egarch_value_grad_hess(params, resid_normal, dist="ged")

        np.testing.assert_allclose(value_c, value_ref, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(grad_c, grad_ref, rtol=5e-5, atol=8e-6)
        np.testing.assert_allclose(hess_c, hess_ref, rtol=1e-3, atol=2e-2)

    def test_ged_pq_gradient_and_hessian_match_ad(self, resid_normal_pq: NDArray[np.float64]) -> None:
        params = np.array([-0.45, 0.08, 0.04, -0.06, 0.02, 0.95, 1.7], dtype=np.float64)
        value_c = _c_egarch_value(params, resid_normal_pq, "ged", p=2, q=1)
        grad_c, hess_c = _c_egarch_grad_hess(params, resid_normal_pq, "ged", p=2, q=1)
        value_ref, grad_ref, hess_ref = egarch_value_grad_hess(params, resid_normal_pq, dist="ged", p=2, q=1)

        np.testing.assert_allclose(value_c, value_ref, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(grad_c, grad_ref, rtol=7e-5, atol=1e-5)
        np.testing.assert_allclose(hess_c, hess_ref, rtol=2e-3, atol=3e-2)


@pytest.mark.parametrize(
    ("dist", "params", "unpack_fn", "pack_fn", "resid_fixture"),
    [
        (
            "normal",
            np.array([-0.45, 0.08, 0.04, -0.06, 0.02, 0.95], dtype=np.float64),
            unpack_egarch,
            pack_egarch,
            "normal",
        ),
        (
            "studentt",
            np.array([-0.45, 0.08, 0.04, -0.06, 0.02, 0.95, 8.0], dtype=np.float64),
            unpack_egarch_studentt,
            pack_egarch_studentt,
            "studentt",
        ),
        (
            "ged",
            np.array([-0.45, 0.08, 0.04, -0.06, 0.02, 0.95, 1.7], dtype=np.float64),
            unpack_egarch_ged,
            pack_egarch_ged,
            "normal",
        ),
        (
            "skewt",
            np.array([-0.45, 0.08, 0.04, -0.06, 0.02, 0.95, 8.0, -0.15], dtype=np.float64),
            unpack_egarch_skewt,
            pack_egarch_skewt,
            "skewt",
        ),
    ],
)
def test_egarch_pq_logspace_grad_hess_match_ad(
    request,
    dist: str,
    params: NDArray[np.float64],
    unpack_fn,
    pack_fn,
    resid_fixture: str,
) -> None:
    resid = request.getfixturevalue(f"resid_{resid_fixture}_pq")
    z = unpack_fn(params, 2, 1)
    theta = pack_fn(z, 2, 1)
    sigma_grad = _sigma_seed(resid)

    log_value_fn = getattr(_c, f"_log_egarch_ll_pq_{dist}")
    log_grad_fn = getattr(_c, f"_log_egarch_ll_grad_pq_{dist}")
    value_c = float(log_value_fn(_as_cptr(z), _as_cptr(resid), _as_cptr(_sigma_seed(resid)), resid.size, 2, 1))
    grad_z = np.zeros_like(z)
    log_grad_fn(_as_cptr(z), _as_cptr(resid), _as_cptr(sigma_grad), _as_cptr(grad_z), resid.size, 2, 1)
    grad_theta, hess_theta = _c_egarch_grad_hess(theta, resid, dist, p=2, q=1)
    hess_z = log_hessian_egarch(theta, grad_theta, hess_theta, 2, 1, dist=dist)
    value_ref, grad_ref, hess_ref = egarch_logspace_value_grad_hess(z, resid, dist=dist, p=2, q=1)

    value_rtol = 5e-6 if dist != "skewt" else 2e-5
    grad_rtol = 1e-4 if dist != "skewt" else 6e-3
    hess_rtol = 3e-3 if dist != "skewt" else 3e-2
    hess_atol = 3e-2 if dist != "skewt" else 6e-2
    np.testing.assert_allclose(value_c, value_ref, rtol=value_rtol, atol=5e-6)
    np.testing.assert_allclose(grad_z, grad_ref, rtol=grad_rtol, atol=1e-5)
    np.testing.assert_allclose(hess_z, hess_ref, rtol=hess_rtol, atol=hess_atol)
