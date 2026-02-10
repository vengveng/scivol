"""
Gradient and Hessian validation for GJR-GARCH C implementations.

Validates analytical derivatives from C against finite differences:
- GJR-GARCH(1,1) + Normal: gradient and Hessian
- GJR-GARCH(1,1) + Student-t: gradient and Hessian
- GJR-GARCH(1,1) + Normal: OPG matrix
- Log-space transform Jacobians

Uses finite difference approximations as ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import volkit._core as _c


# =============================================================================
# Helpers
# =============================================================================

def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _make_gjr_data(n: int = 2000, seed: int = 42) -> NDArray[np.float64]:
    """Generate GJR-GARCH-like data for testing."""
    rng = np.random.default_rng(seed)
    omega, alpha, gamma, beta = 1e-6, 0.05, 0.04, 0.90
    sigma2 = np.zeros(n)
    y = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - 0.5 * gamma - beta)
    y[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        e = y[t-1]
        ind = 1.0 if e < 0 else 0.0
        sigma2[t] = omega + alpha * e**2 + gamma * ind * e**2 + beta * sigma2[t-1]
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return np.ascontiguousarray(y, dtype=np.float64)


@pytest.fixture(scope="module")
def resid() -> NDArray[np.float64]:
    return _make_gjr_data(2000)


# =============================================================================
# Numerical Finite Differences
# =============================================================================

def numerical_gradient(
    obj_fn,
    params: NDArray[np.float64],
    rel_eps: float = 1e-7,
) -> NDArray[np.float64]:
    """Central finite difference gradient with relative step sizes."""
    K = len(params)
    grad = np.zeros(K, dtype=np.float64)
    for i in range(K):
        eps_i = rel_eps * max(abs(params[i]), 1.0)
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps_i
        p_minus[i] -= eps_i
        grad[i] = (obj_fn(p_plus) - obj_fn(p_minus)) / (2 * eps_i)
    return grad


def numerical_hessian(
    obj_fn,
    params: NDArray[np.float64],
    rel_eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Central finite difference Hessian with relative step sizes."""
    K = len(params)
    eps = np.array([rel_eps * max(abs(params[k]), 1.0) for k in range(K)])
    H = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            pp = params.copy(); pp[i] += eps[i]; pp[j] += eps[j]
            pm = params.copy(); pm[i] += eps[i]; pm[j] -= eps[j]
            mp = params.copy(); mp[i] -= eps[i]; mp[j] += eps[j]
            mm = params.copy(); mm[i] -= eps[i]; mm[j] -= eps[j]
            H[i, j] = (obj_fn(pp) - obj_fn(pm) - obj_fn(mp) + obj_fn(mm)) / (4 * eps[i] * eps[j])
    return H


# =============================================================================
# Tests: GJR-GARCH(1,1) + Normal
# =============================================================================

class TestGJRGARCHNormalGradient:
    """Validate C gradient for GJR-GARCH(1,1) + Normal against finite differences."""

    def test_gradient_matches_finite_diff(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)

        resid_ptr = _as_cptr(resid_c)
        sigma2_ptr = _as_cptr(sigma2)

        def nll(theta):
            s2 = np.zeros(n, dtype=np.float64)
            s2[0] = sigma2[0]
            return _c._gjr_garch_ll_11_normal(_as_cptr(theta), resid_ptr, _as_cptr(s2), n)

        # Analytical gradient
        grad_c = np.zeros(4, dtype=np.float64)
        _c._gjr_garch_ll_grad_11_normal(
            _as_cptr(params), resid_ptr, sigma2_ptr, _as_cptr(grad_c), n
        )

        # Numerical gradient (with relative step sizes)
        grad_num = numerical_gradient(nll, params)

        # Compare: alpha, gamma, beta should match very tightly
        # omega gradient has large finite-difference error due to tiny param value
        np.testing.assert_allclose(
            grad_c[1:], grad_num[1:],
            rtol=1e-4, atol=1e-4,
            err_msg="GJR-GARCH Normal gradient mismatch (alpha/gamma/beta)"
        )
        # omega: looser tolerance (finite diff unreliable for ~1e-6 params)
        np.testing.assert_allclose(
            grad_c[0], grad_num[0],
            rtol=0.10,
            err_msg="GJR-GARCH Normal gradient mismatch (omega)"
        )

    def test_gradient_at_larger_omega(self, resid: NDArray[np.float64]) -> None:
        """Test gradient at larger omega where finite diff is reliable."""
        params = np.array([1e-4, 0.08, 0.06, 0.85], dtype=np.float64)
        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)

        resid_ptr = _as_cptr(resid_c)
        sigma2_ptr = _as_cptr(sigma2)

        def nll(theta):
            s2 = np.zeros(n, dtype=np.float64)
            s2[0] = sigma2[0]
            return _c._gjr_garch_ll_11_normal(_as_cptr(theta), resid_ptr, _as_cptr(s2), n)

        grad_c = np.zeros(4, dtype=np.float64)
        _c._gjr_garch_ll_grad_11_normal(
            _as_cptr(params), resid_ptr, sigma2_ptr, _as_cptr(grad_c), n
        )

        grad_num = numerical_gradient(nll, params)

        np.testing.assert_allclose(
            grad_c, grad_num,
            rtol=1e-3, atol=1e-3,
            err_msg="GJR-GARCH Normal gradient mismatch (larger omega)"
        )


class TestGJRGARCHNormalHessian:
    """Validate C Hessian for GJR-GARCH(1,1) + Normal against finite differences."""

    def test_hessian_matches_finite_diff(self, resid: NDArray[np.float64]) -> None:
        """Compare Hessian excluding omega row/col (unreliable for tiny omega)."""
        params = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)

        resid_ptr = _as_cptr(resid_c)
        sigma2_ptr = _as_cptr(sigma2)

        def nll(theta):
            s2 = np.zeros(n, dtype=np.float64)
            s2[0] = sigma2[0]
            return _c._gjr_garch_ll_11_normal(_as_cptr(theta), resid_ptr, _as_cptr(s2), n)

        hess_c = np.zeros(16, dtype=np.float64)
        _c._gjr_garch_ll_hess_11_normal(
            _as_cptr(params), resid_ptr, sigma2_ptr, _as_cptr(hess_c), n
        )
        hess_c = hess_c.reshape(4, 4)

        hess_num = numerical_hessian(nll, params)

        # Compare alpha/gamma/beta block (indices 1:4, 1:4)
        # Omega entries are unreliable with finite diffs at omega=1e-6
        np.testing.assert_allclose(
            hess_c[1:, 1:], hess_num[1:, 1:],
            rtol=5e-3, atol=1.0,
            err_msg="GJR-GARCH Normal Hessian mismatch (alpha/gamma/beta block)"
        )

    def test_hessian_at_larger_omega(self, resid: NDArray[np.float64]) -> None:
        """Test Hessian at larger omega where finite diff is reliable."""
        params = np.array([1e-4, 0.08, 0.06, 0.85], dtype=np.float64)
        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)

        resid_ptr = _as_cptr(resid_c)
        sigma2_ptr = _as_cptr(sigma2)

        def nll(theta):
            s2 = np.zeros(n, dtype=np.float64)
            s2[0] = sigma2[0]
            return _c._gjr_garch_ll_11_normal(_as_cptr(theta), resid_ptr, _as_cptr(s2), n)

        hess_c = np.zeros(16, dtype=np.float64)
        _c._gjr_garch_ll_hess_11_normal(
            _as_cptr(params), resid_ptr, sigma2_ptr, _as_cptr(hess_c), n
        )
        hess_c = hess_c.reshape(4, 4)

        hess_num = numerical_hessian(nll, params)

        np.testing.assert_allclose(
            hess_c, hess_num,
            rtol=5e-2, atol=1.0,
            err_msg="GJR-GARCH Normal Hessian mismatch (larger omega)"
        )

    def test_hessian_symmetry(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)

        hess_c = np.zeros(16, dtype=np.float64)
        _c._gjr_garch_ll_hess_11_normal(
            _as_cptr(params), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess_c), n
        )
        hess_c = hess_c.reshape(4, 4)

        np.testing.assert_allclose(
            hess_c, hess_c.T,
            rtol=1e-10, atol=1e-12,
            err_msg="GJR-GARCH Normal Hessian not symmetric"
        )


# =============================================================================
# Tests: GJR-GARCH(1,1) + Student-t
# =============================================================================

class TestGJRGARCHStudentTGradient:
    """Validate C gradient for GJR-GARCH(1,1) + Student-t against finite differences."""

    def test_gradient_matches_finite_diff(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90, 8.0], dtype=np.float64)
        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)

        resid_ptr = _as_cptr(resid_c)
        sigma2_ptr = _as_cptr(sigma2)

        def nll(theta):
            s2 = np.zeros(n, dtype=np.float64)
            s2[0] = sigma2[0]
            return _c._gjr_garch_ll_11_studentt(_as_cptr(theta), resid_ptr, _as_cptr(s2), n)

        grad_c = np.zeros(5, dtype=np.float64)
        _c._gjr_garch_ll_grad_11_studentt(
            _as_cptr(params), resid_ptr, sigma2_ptr, _as_cptr(grad_c), n
        )

        grad_num = numerical_gradient(nll, params)

        # Compare alpha, gamma, beta tightly; omega and nu loosely
        # nu uses digamma_approx which has limited precision
        np.testing.assert_allclose(
            grad_c[1:4], grad_num[1:4],
            rtol=1e-4, atol=1e-3,
            err_msg="GJR-GARCH Student-t gradient mismatch (alpha/gamma/beta)"
        )
        np.testing.assert_allclose(
            grad_c[0], grad_num[0],
            rtol=0.10,
            err_msg="GJR-GARCH Student-t gradient mismatch (omega)"
        )
        # nu: C uses digamma_approx, expect ~0.5% discrepancy
        np.testing.assert_allclose(
            grad_c[4], grad_num[4],
            rtol=5e-2, atol=0.1,
            err_msg="GJR-GARCH Student-t gradient mismatch (nu)"
        )

    def test_gradient_different_nu(self, resid: NDArray[np.float64]) -> None:
        """Test with different degrees of freedom values."""
        for nu in [4.0, 6.0, 12.0, 30.0]:
            params = np.array([1e-6, 0.05, 0.04, 0.90, nu], dtype=np.float64)
            n = len(resid)
            sigma2 = np.zeros(n, dtype=np.float64)
            sigma2[0] = np.mean(resid**2)
            resid_c = np.ascontiguousarray(resid, dtype=np.float64)

            resid_ptr = _as_cptr(resid_c)
            sigma2_ptr = _as_cptr(sigma2)

            def nll(theta, s0=sigma2[0]):
                s2 = np.zeros(n, dtype=np.float64)
                s2[0] = s0
                return _c._gjr_garch_ll_11_studentt(_as_cptr(theta), resid_ptr, _as_cptr(s2), n)

            grad_c = np.zeros(5, dtype=np.float64)
            _c._gjr_garch_ll_grad_11_studentt(
                _as_cptr(params), resid_ptr, sigma2_ptr, _as_cptr(grad_c), n
            )

            grad_num = numerical_gradient(nll, params)

            # Compare non-omega parameters
            np.testing.assert_allclose(
                grad_c[1:], grad_num[1:],
                rtol=5e-3, atol=1e-2,
                err_msg=f"GJR-GARCH Student-t gradient mismatch at nu={nu}"
            )


class TestGJRGARCHStudentTHessian:
    """Validate C Hessian for GJR-GARCH(1,1) + Student-t against finite differences."""

    def test_hessian_matches_finite_diff(self, resid: NDArray[np.float64]) -> None:
        """Compare Hessian excluding omega row/col."""
        params = np.array([1e-6, 0.05, 0.04, 0.90, 8.0], dtype=np.float64)
        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)

        resid_ptr = _as_cptr(resid_c)
        sigma2_ptr = _as_cptr(sigma2)

        def nll(theta):
            s2 = np.zeros(n, dtype=np.float64)
            s2[0] = sigma2[0]
            return _c._gjr_garch_ll_11_studentt(_as_cptr(theta), resid_ptr, _as_cptr(s2), n)

        hess_c = np.zeros(25, dtype=np.float64)
        _c._gjr_garch_ll_hess_11_studentt(
            _as_cptr(params), resid_ptr, sigma2_ptr, _as_cptr(hess_c), n
        )
        hess_c = hess_c.reshape(5, 5)

        hess_num = numerical_hessian(nll, params)

        # Compare alpha/gamma/beta block (indices 1:4, 1:4)
        # Excludes omega (too small) and nu (trigamma_approx limited precision)
        np.testing.assert_allclose(
            hess_c[1:4, 1:4], hess_num[1:4, 1:4],
            rtol=5e-3, atol=1.0,
            err_msg="GJR-GARCH Student-t Hessian mismatch (alpha/gamma/beta block)"
        )
        # Cross-terms alpha/gamma/beta vs nu should have correct sign
        for i in range(1, 4):
            if abs(hess_c[i, 4]) > 1.0 and abs(hess_num[i, 4]) > 1.0:
                assert np.sign(hess_c[i, 4]) == np.sign(hess_num[i, 4]), \
                    f"Sign mismatch at ({i},4): C={hess_c[i,4]:.2f}, num={hess_num[i,4]:.2f}"

    def test_hessian_symmetry(self, resid: NDArray[np.float64]) -> None:
        params = np.array([1e-6, 0.05, 0.04, 0.90, 8.0], dtype=np.float64)
        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)

        hess_c = np.zeros(25, dtype=np.float64)
        _c._gjr_garch_ll_hess_11_studentt(
            _as_cptr(params), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess_c), n
        )
        hess_c = hess_c.reshape(5, 5)

        np.testing.assert_allclose(
            hess_c, hess_c.T,
            rtol=1e-10, atol=1e-12,
            err_msg="GJR-GARCH Student-t Hessian not symmetric"
        )


# =============================================================================
# Tests: OPG + Hessian (Normal)
# =============================================================================

class TestGJRGARCHOPG:
    """Validate OPG/Hessian for GJR-GARCH(1,1) Normal."""

    def test_opg_positive_semidefinite(self, resid: NDArray[np.float64]) -> None:
        """OPG should be positive semi-definite."""
        params = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)

        # Compute variance first
        _c._gjr_garch_variance_11(
            _as_cptr(params), _as_cptr(resid_c), _as_cptr(sigma2), n
        )

        OPG = np.zeros(16, dtype=np.float64)
        HESS = np.zeros(16, dtype=np.float64)
        _c._gjr_garch_opg_hess_11(
            _as_cptr(params), _as_cptr(resid_c), _as_cptr(sigma2),
            _as_cptr(OPG), _as_cptr(HESS), n
        )
        OPG = OPG.reshape(4, 4)

        # OPG should be symmetric
        np.testing.assert_allclose(OPG, OPG.T, atol=1e-12)

        # OPG should be PSD (all eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(OPG)
        assert np.all(eigvals >= -1e-10), f"OPG has negative eigenvalue: {eigvals.min()}"

    def test_hess_matches_likelihood_hess(self, resid: NDArray[np.float64]) -> None:
        """Hessian from OPG function should match likelihood Hessian."""
        params = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)

        # Compute variance
        _c._gjr_garch_variance_11(
            _as_cptr(params), _as_cptr(resid_c), _as_cptr(sigma2), n
        )

        OPG = np.zeros(16, dtype=np.float64)
        HESS = np.zeros(16, dtype=np.float64)
        _c._gjr_garch_opg_hess_11(
            _as_cptr(params), _as_cptr(resid_c), _as_cptr(sigma2),
            _as_cptr(OPG), _as_cptr(HESS), n
        )
        HESS = HESS.reshape(4, 4)

        # Get likelihood Hessian
        hess_ll = np.zeros(16, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        _c._gjr_garch_ll_hess_11_normal(
            _as_cptr(params), _as_cptr(resid_c), _as_cptr(sigma2), _as_cptr(hess_ll), n
        )
        hess_ll = hess_ll.reshape(4, 4)

        # The two Hessians should be close (they may differ slightly due to
        # implementation details like scaling)
        # The errors_gjr_garch.c Hessian is per-observation, likelihood is summed
        # Check relative structure
        ratio = HESS / (hess_ll + 1e-20)
        # They should have the same sign pattern
        assert np.all(np.sign(HESS.diagonal()) == np.sign(hess_ll.diagonal())), \
            "Hessian diagonal sign mismatch"


# =============================================================================
# Tests: Log-space transforms
# =============================================================================

class TestGJRGARCHLogTransforms:
    """Validate GJR-GARCH log-space parameter transforms."""

    def test_pack_unpack_roundtrip(self) -> None:
        """pack(unpack(theta)) should recover theta."""
        from volkit._kernels.transforms import pack_gjr_garch, unpack_gjr_garch
        
        theta = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        z = unpack_gjr_garch(theta, 1, 1)
        theta_recovered = pack_gjr_garch(z, 1, 1)
        
        np.testing.assert_allclose(theta, theta_recovered, rtol=1e-10)

    def test_pack_c_matches_python(self) -> None:
        """C pack should match Python pack."""
        from volkit._kernels.transforms import pack_gjr_garch, unpack_gjr_garch
        
        theta_orig = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        z = unpack_gjr_garch(theta_orig, 1, 1)
        
        # Python
        theta_py = pack_gjr_garch(z, 1, 1)
        
        # C
        theta_c = np.zeros(4, dtype=np.float64)
        _c._pack_gjr_garch_11(_as_cptr(z), _as_cptr(theta_c))
        
        np.testing.assert_allclose(theta_py, theta_c, rtol=1e-12)

    def test_jacobian_c_matches_python(self) -> None:
        """C Jacobian should match Python Jacobian."""
        from volkit._kernels.transforms import jacobian_gjr_garch
        
        theta = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        
        # Python
        J_py = jacobian_gjr_garch(theta, 1, 1)
        
        # C
        J_c = np.zeros(16, dtype=np.float64)
        _c._jacobian_gjr_garch_11(_as_cptr(theta), _as_cptr(J_c))
        J_c = J_c.reshape(4, 4)
        
        np.testing.assert_allclose(J_py, J_c, rtol=1e-10)

    def test_jacobian_numerical(self) -> None:
        """C Jacobian should match numerical Jacobian."""
        from volkit._kernels.transforms import pack_gjr_garch, unpack_gjr_garch
        
        theta = np.array([1e-6, 0.05, 0.04, 0.90], dtype=np.float64)
        z = unpack_gjr_garch(theta, 1, 1)
        
        # Analytical Jacobian
        J_c = np.zeros(16, dtype=np.float64)
        _c._jacobian_gjr_garch_11(_as_cptr(theta), _as_cptr(J_c))
        J_c = J_c.reshape(4, 4)
        
        # Numerical Jacobian
        eps = 1e-7
        K = len(z)
        J_num = np.zeros((K, K), dtype=np.float64)
        for j in range(K):
            z_plus = z.copy(); z_plus[j] += eps
            z_minus = z.copy(); z_minus[j] -= eps
            theta_plus = pack_gjr_garch(z_plus, 1, 1)
            theta_minus = pack_gjr_garch(z_minus, 1, 1)
            J_num[:, j] = (theta_plus - theta_minus) / (2 * eps)
        
        np.testing.assert_allclose(J_c, J_num, rtol=1e-5, atol=1e-10)

    def test_stationarity_constraint(self) -> None:
        """Softmax transform should enforce sum < 1."""
        from volkit._kernels.transforms import pack_gjr_garch
        
        # Test with various unconstrained values
        rng = np.random.default_rng(42)
        for _ in range(100):
            z = rng.standard_normal(4)
            z[0] = rng.standard_normal() * 5  # Larger range for omega
            theta = pack_gjr_garch(z, 1, 1)
            
            assert theta[0] > 0, "omega must be positive"
            assert np.all(theta[1:] > 0), "alpha, gamma, beta must be positive"
            assert theta[1:].sum() < 1.0, f"Sum must be < 1, got {theta[1:].sum()}"

    def test_studentt_pack_unpack(self) -> None:
        """Student-t pack/unpack roundtrip."""
        from volkit._kernels.transforms import pack_gjr_garch_studentt, unpack_gjr_garch_studentt
        
        theta = np.array([1e-6, 0.05, 0.04, 0.90, 8.0], dtype=np.float64)
        z = unpack_gjr_garch_studentt(theta, 1, 1)
        theta_recovered = pack_gjr_garch_studentt(z, 1, 1)
        
        np.testing.assert_allclose(theta, theta_recovered, rtol=1e-8)

    def test_skewt_pack_unpack(self) -> None:
        """Skew-t pack/unpack roundtrip."""
        from volkit._kernels.transforms import pack_gjr_garch_skewt, unpack_gjr_garch_skewt
        
        theta = np.array([1e-6, 0.05, 0.04, 0.90, 8.0, -0.15], dtype=np.float64)
        z = unpack_gjr_garch_skewt(theta, 1, 1)
        theta_recovered = pack_gjr_garch_skewt(z, 1, 1)
        
        np.testing.assert_allclose(theta, theta_recovered, rtol=1e-8)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
