"""
Gradient and Hessian validation for GJR-GARCH C implementations.

Uses an independent AD oracle rather than finite differences:
- GJR-GARCH(1,1) + Normal: gradient and Hessian
- GJR-GARCH(1,1) + Student-t: gradient and Hessian
- GJR-GARCH(p,q) + Normal/Student-t: gradient and Hessian
- GJR-GARCH(1,1) + Normal: OPG matrix
- Log-space transform Jacobians
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

import volkit._core as _c
from volkit._devtools.ad_oracle import _jax_modules, gjr_garch_value_grad_hess


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _make_gjr_data(n: int = 800, seed: int = 42) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    omega, alpha, gamma, beta = 1e-6, 0.05, 0.04, 0.90
    sigma2 = np.zeros(n)
    y = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - 0.5 * gamma - beta)
    y[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        e = y[t - 1]
        ind = 1.0 if e < 0 else 0.0
        sigma2[t] = omega + alpha * e**2 + gamma * ind * e**2 + beta * sigma2[t - 1]
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return np.ascontiguousarray(y, dtype=np.float64)


@pytest.fixture(scope="module")
def resid() -> NDArray[np.float64]:
    return _make_gjr_data()


def _sigma_seed(resid: NDArray[np.float64], max_lag: int = 1) -> NDArray[np.float64]:
    sigma2 = np.zeros(len(resid), dtype=np.float64)
    sigma2[:max_lag] = np.mean(resid**2)
    return sigma2


def _c_gjr_grad_hess(
    params: NDArray[np.float64],
    resid: NDArray[np.float64],
    p: int,
    q: int,
    dist: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    params = np.ascontiguousarray(params, dtype=np.float64)
    resid = np.ascontiguousarray(resid, dtype=np.float64)
    max_lag = max(p, q)
    sigma_g = _sigma_seed(resid, max_lag)
    sigma_h = _sigma_seed(resid, max_lag)

    if dist == "normal":
        if p == 1 and q == 1:
            grad_fn = _c._gjr_garch_ll_grad_11_normal
            hess_fn = _c._gjr_garch_ll_hess_11_normal
            extra: tuple[int, ...] = ()
        else:
            grad_fn = _c._gjr_garch_ll_grad_pq_normal
            hess_fn = _c._gjr_garch_ll_hess_pq_normal
            extra = (p, q)
    elif dist == "studentt":
        if p == 1 and q == 1:
            grad_fn = _c._gjr_garch_ll_grad_11_studentt
            hess_fn = _c._gjr_garch_ll_hess_11_studentt
            extra = ()
        else:
            grad_fn = _c._gjr_garch_ll_grad_pq_studentt
            hess_fn = _c._gjr_garch_ll_hess_pq_studentt
            extra = (p, q)
    else:
        raise ValueError(f"Unsupported distribution: {dist}")

    grad = np.zeros(len(params), dtype=np.float64)
    hess = np.zeros(len(params) * len(params), dtype=np.float64)
    grad_fn(_as_cptr(params), _as_cptr(resid), _as_cptr(sigma_g), _as_cptr(grad), len(resid), *extra)
    hess_fn(_as_cptr(params), _as_cptr(resid), _as_cptr(sigma_h), _as_cptr(hess), len(resid), *extra)
    return grad, hess.reshape(len(params), len(params))


class TestGJRGARCHNormalAD:
    @pytest.mark.parametrize(
        "params",
        [
            np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64),
            np.array([1e-4, 0.08, 0.06, 0.85], dtype=np.float64),
        ],
    )
    def test_gradient_and_hessian_match_ad(self, resid: NDArray[np.float64], params: NDArray[np.float64]) -> None:
        grad_c, hess_c = _c_gjr_grad_hess(params, resid, 1, 1, "normal")
        _, grad_ref, hess_ref = gjr_garch_value_grad_hess(params, resid, 1, 1, dist="normal")
        np.testing.assert_allclose(grad_c, grad_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(hess_c, hess_ref, rtol=1e-8, atol=1e-9)

    def test_hessian_symmetry(self, resid: NDArray[np.float64]) -> None:
        _, hess_c = _c_gjr_grad_hess(np.array([1e-6, 0.05, 0.04, 0.90]), resid, 1, 1, "normal")
        np.testing.assert_allclose(hess_c, hess_c.T, rtol=1e-10, atol=1e-12)


class TestGJRGARCHStudentTAD:
    @pytest.mark.parametrize("nu", [4.0, 8.0, 30.0])
    def test_gradient_matches_ad(self, resid: NDArray[np.float64], nu: float) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90, nu], dtype=np.float64)
        grad_c, _ = _c_gjr_grad_hess(params, resid, 1, 1, "studentt")
        _, grad_ref, _ = gjr_garch_value_grad_hess(params, resid, 1, 1, dist="studentt")
        np.testing.assert_allclose(grad_c[:4], grad_ref[:4], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(grad_c[4], grad_ref[4], rtol=5e-2, atol=0.1)

    def test_hessian_matches_ad_on_garch_block(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90, 8.0], dtype=np.float64)
        _, hess_c = _c_gjr_grad_hess(params, resid, 1, 1, "studentt")
        _, _, hess_ref = gjr_garch_value_grad_hess(params, resid, 1, 1, dist="studentt")
        np.testing.assert_allclose(hess_c[:4, :4], hess_ref[:4, :4], rtol=1e-5, atol=1e-6)

    def test_hessian_symmetry(self, resid: NDArray[np.float64]) -> None:
        _, hess_c = _c_gjr_grad_hess(np.array([1e-6, 0.05, 0.04, 0.90, 8.0]), resid, 1, 1, "studentt")
        np.testing.assert_allclose(hess_c, hess_c.T, rtol=1e-10, atol=1e-12)


class TestGJRGARCHpqAD:
    @pytest.mark.parametrize("p,q", [(1, 1), (2, 1), (1, 2), (2, 2)])
    def test_normal_gradient_and_hessian_match_ad(self, resid: NDArray[np.float64], p: int, q: int) -> None:
        k = 1 + 2 * p + q
        params = np.zeros(k, dtype=np.float64)
        params[0] = 1e-4
        params[1:1 + p] = 0.05 / p
        params[1 + p:1 + 2 * p] = 0.04 / p
        params[1 + 2 * p:] = 0.85 / q
        grad_c, hess_c = _c_gjr_grad_hess(params, resid, p, q, "normal")
        _, grad_ref, hess_ref = gjr_garch_value_grad_hess(params, resid, p, q, dist="normal")
        np.testing.assert_allclose(grad_c, grad_ref, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(hess_c, hess_ref, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("p,q", [(1, 1), (2, 1), (1, 2), (2, 2)])
    def test_studentt_gradient_and_garch_hessian_block_match_ad(self, resid: NDArray[np.float64], p: int, q: int) -> None:
        k_garch = 1 + 2 * p + q
        params = np.zeros(k_garch + 1, dtype=np.float64)
        params[0] = 1e-4
        params[1:1 + p] = 0.05 / p
        params[1 + p:1 + 2 * p] = 0.04 / p
        params[1 + 2 * p:k_garch] = 0.85 / q
        params[k_garch] = 8.0
        grad_c, hess_c = _c_gjr_grad_hess(params, resid, p, q, "studentt")
        _, grad_ref, hess_ref = gjr_garch_value_grad_hess(params, resid, p, q, dist="studentt")
        np.testing.assert_allclose(grad_c[:k_garch], grad_ref[:k_garch], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(grad_c[k_garch], grad_ref[k_garch], rtol=5e-2, atol=0.1)
        np.testing.assert_allclose(hess_c[:k_garch, :k_garch], hess_ref[:k_garch, :k_garch], rtol=1e-5, atol=1e-5)


class TestGJRGARCHpqConsistency:
    def test_normal_gradient_pq_matches_11(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        grad_11, _ = _c_gjr_grad_hess(params, resid, 1, 1, "normal")
        sigma2 = _sigma_seed(resid)
        grad_pq = np.zeros(4, dtype=np.float64)
        _c._gjr_garch_ll_grad_pq_normal(
            _as_cptr(params), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(grad_pq), len(resid), 1, 1
        )
        np.testing.assert_allclose(grad_pq, grad_11, rtol=1e-10, atol=1e-12)

    def test_normal_hessian_pq_matches_11(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        _, hess_11 = _c_gjr_grad_hess(params, resid, 1, 1, "normal")
        sigma2 = _sigma_seed(resid)
        hess_pq = np.zeros(16, dtype=np.float64)
        _c._gjr_garch_ll_hess_pq_normal(
            _as_cptr(params), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess_pq), len(resid), 1, 1
        )
        hess_pq = hess_pq.reshape(4, 4)
        np.testing.assert_allclose(hess_pq, hess_11, rtol=1e-10, atol=1e-12)

    def test_studentt_gradient_pq_matches_11(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90, 8.0], dtype=np.float64)
        grad_11, _ = _c_gjr_grad_hess(params, resid, 1, 1, "studentt")
        sigma2 = _sigma_seed(resid)
        grad_pq = np.zeros(5, dtype=np.float64)
        _c._gjr_garch_ll_grad_pq_studentt(
            _as_cptr(params), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(grad_pq), len(resid), 1, 1
        )
        np.testing.assert_allclose(grad_pq, grad_11, rtol=1e-10, atol=1e-12)

    def test_studentt_hessian_pq_matches_11(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90, 8.0], dtype=np.float64)
        _, hess_11 = _c_gjr_grad_hess(params, resid, 1, 1, "studentt")
        sigma2 = _sigma_seed(resid)
        hess_pq = np.zeros(25, dtype=np.float64)
        _c._gjr_garch_ll_hess_pq_studentt(
            _as_cptr(params), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess_pq), len(resid), 1, 1
        )
        hess_pq = hess_pq.reshape(5, 5)
        np.testing.assert_allclose(hess_pq, hess_11, rtol=1e-10, atol=1e-12)


class TestGJRGARCHOPG:
    def test_opg_positive_semidefinite(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        n = len(resid)
        sigma2 = _sigma_seed(resid)

        _c._gjr_garch_variance_11(_as_cptr(params), _as_cptr(resid), _as_cptr(sigma2), n)
        opg = np.zeros(16, dtype=np.float64)
        hess = np.zeros(16, dtype=np.float64)
        _c._gjr_garch_opg_hess_11(
            _as_cptr(params), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(opg), _as_cptr(hess), n
        )
        opg = opg.reshape(4, 4)
        np.testing.assert_allclose(opg, opg.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(opg)
        assert np.all(eigvals >= -1e-10), f"OPG has negative eigenvalue: {eigvals.min()}"

    def test_hess_matches_likelihood_hess_sign_pattern(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        n = len(resid)
        sigma2 = _sigma_seed(resid)
        _c._gjr_garch_variance_11(_as_cptr(params), _as_cptr(resid), _as_cptr(sigma2), n)
        opg = np.zeros(16, dtype=np.float64)
        hess_opg = np.zeros(16, dtype=np.float64)
        _c._gjr_garch_opg_hess_11(
            _as_cptr(params), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(opg), _as_cptr(hess_opg), n
        )
        _, hess_ll = _c_gjr_grad_hess(params, resid, 1, 1, "normal")
        hess_opg = hess_opg.reshape(4, 4)
        assert np.all(np.sign(np.diag(hess_opg)) == np.sign(np.diag(hess_ll)))


class TestGJRGARCHLogTransforms:
    def test_pack_unpack_roundtrip(self) -> None:
        from volkit._kernels.transforms import pack_gjr_garch, unpack_gjr_garch

        theta = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        z = unpack_gjr_garch(theta, 1, 1)
        theta_recovered = pack_gjr_garch(z, 1, 1)
        np.testing.assert_allclose(theta, theta_recovered, rtol=1e-10)

    def test_pack_c_matches_python(self) -> None:
        from volkit._kernels.transforms import pack_gjr_garch, unpack_gjr_garch

        theta_orig = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        z = unpack_gjr_garch(theta_orig, 1, 1)
        theta_py = pack_gjr_garch(z, 1, 1)
        theta_c = np.zeros(4, dtype=np.float64)
        _c._pack_gjr_garch_11(_as_cptr(z), _as_cptr(theta_c))
        np.testing.assert_allclose(theta_py, theta_c, rtol=1e-12)

    def test_jacobian_c_matches_python(self) -> None:
        from volkit._kernels.transforms import jacobian_gjr_garch

        theta = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        j_py = jacobian_gjr_garch(theta, 1, 1)
        j_c = np.zeros(16, dtype=np.float64)
        _c._jacobian_gjr_garch_11(_as_cptr(theta), _as_cptr(j_c))
        np.testing.assert_allclose(j_c.reshape(4, 4), j_py, rtol=1e-10, atol=1e-12)

    def test_jacobian_matches_ad(self) -> None:
        _, jnp, _ = _jax_modules()
        jax = _jax_modules()[0]
        theta = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        z = np.array([np.log(theta[0]), np.log(theta[1] / 0.01), np.log(theta[2] / 0.01), np.log(theta[3] / 0.01)])

        def pack_jax(z_j):
            omega = jnp.exp(z_j[0])
            params = jax.nn.softmax(jnp.array([z_j[1], z_j[2], z_j[3], 0.0], dtype=z_j.dtype))[:3]
            return jnp.concatenate([jnp.array([omega], dtype=z_j.dtype), params])

        j_ad = np.asarray(jax.jacfwd(pack_jax)(jnp.asarray(z, dtype=jnp.float64)), dtype=np.float64)
        j_c = np.zeros(16, dtype=np.float64)
        _c._jacobian_gjr_garch_11(_as_cptr(theta), _as_cptr(j_c))
        np.testing.assert_allclose(j_c.reshape(4, 4), j_ad, rtol=1e-8, atol=1e-10)

    def test_stationarity_constraint(self) -> None:
        from volkit._kernels.transforms import pack_gjr_garch

        rng = np.random.default_rng(42)
        for _ in range(100):
            z = rng.standard_normal(4)
            z[0] = rng.standard_normal() * 5
            theta = pack_gjr_garch(z, 1, 1)
            assert theta[0] > 0
            assert np.all(theta[1:] > 0)
            assert theta[1:].sum() < 1.0

    def test_studentt_pack_unpack(self) -> None:
        from volkit._kernels.transforms import pack_gjr_garch_studentt, unpack_gjr_garch_studentt

        theta = np.array([1e-6, 0.05, 0.04, 0.90, 8.0], dtype=np.float64)
        z = unpack_gjr_garch_studentt(theta, 1, 1)
        theta_recovered = pack_gjr_garch_studentt(z, 1, 1)
        np.testing.assert_allclose(theta, theta_recovered, rtol=1e-8)

    def test_skewt_pack_unpack(self) -> None:
        from volkit._kernels.transforms import pack_gjr_garch_skewt, unpack_gjr_garch_skewt

        theta = np.array([1e-6, 0.05, 0.04, 0.90, 8.0, -0.15], dtype=np.float64)
        z = unpack_gjr_garch_skewt(theta, 1, 1)
        theta_recovered = pack_gjr_garch_skewt(z, 1, 1)
        np.testing.assert_allclose(theta, theta_recovered, rtol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
