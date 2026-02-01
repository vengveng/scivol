"""
Numerical Hessian Computation
=============================

Provides robust numerical Hessian estimation for GARCH models.

Two approaches are available:
1. Direct finite differences in θ-space (simple, but can hit boundaries)
2. Reparameterized finite differences in unconstrained z-space (robust)

The reparameterized approach avoids boundary issues by transforming to an
unconstrained space where finite differences are always valid.

Transformations:
    ω = exp(z_ω)                          # positivity
    [α, β] = softmax([z_α, z_β, 0])[:2]   # joint softmax for stationarity
    ν = 2 + exp(z_ν)                      # ν > 2 for finite variance
    λ = tanh(z_λ)                         # λ ∈ (-1, 1)

Chain rule for covariance:
    If θ = T(z), J = ∂θ/∂z, and H_z is the Hessian in z-space, then:
    Cov(θ) = J @ inv(H_z) @ J^T
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


# =============================================================================
# CONSTANTS
# =============================================================================

LOG_CLIP_MIN = -700.0
LOG_CLIP_MAX = 700.0


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RobustHessianResult:
    """Result of robust Hessian computation."""
    # In unconstrained z-space
    hessian_z: NDArray[np.float64]
    cov_z: Optional[NDArray[np.float64]]
    
    # In original θ-space (transformed via chain rule)
    hessian_theta: Optional[NDArray[np.float64]]
    cov_theta: Optional[NDArray[np.float64]]
    std_errors: Optional[NDArray[np.float64]]
    
    # Jacobian of transformation
    jacobian: NDArray[np.float64]
    
    # Diagnostics
    success: bool
    message: str


# =============================================================================
# BASIC NUMERICAL HESSIAN (θ-space)
# =============================================================================

def compute_numerical_hessian(
    objective: Callable[[NDArray[np.float64]], float],
    params: NDArray[np.float64],
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """
    Compute Hessian matrix via central finite differences in θ-space.
    
    WARNING: This can produce invalid results near parameter boundaries
    because finite differences may step outside the valid region.
    For boundary-safe computation, use the reparameterized versions.
    
    Parameters
    ----------
    objective : callable
        Objective function (negative log-likelihood)
    params : array
        Parameter vector at which to evaluate
    eps : float
        Step size for finite differences
    
    Returns
    -------
    hess : array (k x k)
        Hessian matrix
    """
    k = len(params)
    hess = np.zeros((k, k))
    
    for i in range(k):
        for j in range(i, k):
            # f(x + e_i + e_j)
            params_pp = params.copy()
            params_pp[i] += eps
            params_pp[j] += eps
            fpp = objective(params_pp)
            
            # f(x + e_i - e_j)
            params_pm = params.copy()
            params_pm[i] += eps
            params_pm[j] -= eps
            fpm = objective(params_pm)
            
            # f(x - e_i + e_j)
            params_mp = params.copy()
            params_mp[i] -= eps
            params_mp[j] += eps
            fmp = objective(params_mp)
            
            # f(x - e_i - e_j)
            params_mm = params.copy()
            params_mm[i] -= eps
            params_mm[j] -= eps
            fmm = objective(params_mm)
            
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
            hess[j, i] = hess[i, j]
    
    return hess


def compute_hessian_unconstrained(
    objective_z: Callable[[NDArray[np.float64]], float],
    z: NDArray[np.float64],
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """
    Compute Hessian via central finite differences in unconstrained z-space.
    
    Since z is unconstrained, finite differences never hit boundaries.
    This is the core function used by the robust Hessian estimators.
    
    Parameters
    ----------
    objective_z : callable
        Objective function in z-space: f(z) -> scalar
    z : array
        Point at which to evaluate Hessian
    eps : float
        Step size for finite differences
    
    Returns
    -------
    H : array (k x k)
        Hessian matrix ∂²f/∂z_i∂z_j
    """
    k = len(z)
    H = np.zeros((k, k), dtype=np.float64)
    
    for i in range(k):
        for j in range(i, k):
            z_pp = z.copy()
            z_pp[i] += eps
            z_pp[j] += eps
            
            z_pm = z.copy()
            z_pm[i] += eps
            z_pm[j] -= eps
            
            z_mp = z.copy()
            z_mp[i] -= eps
            z_mp[j] += eps
            
            z_mm = z.copy()
            z_mm[i] -= eps
            z_mm[j] -= eps
            
            H[i, j] = (objective_z(z_pp) - objective_z(z_pm) 
                       - objective_z(z_mp) + objective_z(z_mm)) / (4 * eps * eps)
            H[j, i] = H[i, j]
    
    return H


# =============================================================================
# NORMAL GARCH TRANSFORMATIONS (3 parameters: omega, alpha, beta)
# =============================================================================

def normal_theta_to_z(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform Normal GARCH parameters from constrained θ to unconstrained z.
    
    θ = [omega, alpha, beta]
    z = [z_omega, z_alpha, z_beta]
    """
    omega, alpha, beta = theta[0], theta[1], theta[2]
    
    z_omega = np.log(omega)
    
    remainder = 1.0 - alpha - beta
    remainder = max(remainder, 1e-10)
    z_alpha = np.log(alpha / remainder)
    z_beta = np.log(beta / remainder)
    
    return np.array([z_omega, z_alpha, z_beta])


def normal_z_to_theta(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform Normal GARCH parameters from unconstrained z to constrained θ."""
    z_omega, z_alpha, z_beta = z[0], z[1], z[2]
    
    omega = np.exp(np.clip(z_omega, LOG_CLIP_MIN, LOG_CLIP_MAX))
    
    z_all = np.array([z_alpha, z_beta, 0.0])
    log_denom = logsumexp(z_all)
    alpha = np.exp(z_alpha - log_denom)
    beta = np.exp(z_beta - log_denom)
    
    return np.array([omega, alpha, beta])


def normal_jacobian(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Jacobian J = ∂θ/∂z for Normal GARCH transformation."""
    theta = normal_z_to_theta(z)
    omega, alpha, beta = theta[0], theta[1], theta[2]
    
    J = np.zeros((3, 3), dtype=np.float64)
    
    J[0, 0] = omega
    J[1, 1] = alpha * (1.0 - alpha)
    J[1, 2] = -alpha * beta
    J[2, 1] = -beta * alpha
    J[2, 2] = beta * (1.0 - beta)
    
    return J


# =============================================================================
# STUDENT-T GARCH TRANSFORMATIONS (4 parameters: omega, alpha, beta, nu)
# =============================================================================

def studentt_theta_to_z(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform Student-t GARCH parameters from constrained θ to unconstrained z.
    
    θ = [omega, alpha, beta, nu]
    z = [z_omega, z_alpha, z_beta, z_nu]
    
    Transformations:
        omega = exp(z_omega)
        [alpha, beta] = softmax([z_alpha, z_beta, 0])[:2]
        nu = 2 + exp(z_nu)
    """
    omega, alpha, beta, nu = theta[0], theta[1], theta[2], theta[3]
    
    z_omega = np.log(omega)
    
    remainder = 1.0 - alpha - beta
    remainder = max(remainder, 1e-10)
    z_alpha = np.log(alpha / remainder)
    z_beta = np.log(beta / remainder)
    
    z_nu = np.log(nu - 2.0)
    
    return np.array([z_omega, z_alpha, z_beta, z_nu])


def studentt_z_to_theta(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform Student-t GARCH parameters from unconstrained z to constrained θ."""
    z_omega, z_alpha, z_beta, z_nu = z[0], z[1], z[2], z[3]
    
    omega = np.exp(np.clip(z_omega, LOG_CLIP_MIN, LOG_CLIP_MAX))
    
    z_all = np.array([z_alpha, z_beta, 0.0])
    log_denom = logsumexp(z_all)
    alpha = np.exp(z_alpha - log_denom)
    beta = np.exp(z_beta - log_denom)
    
    nu = 2.0 + np.exp(np.clip(z_nu, LOG_CLIP_MIN, LOG_CLIP_MAX))
    
    return np.array([omega, alpha, beta, nu])


def studentt_jacobian(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Jacobian J = ∂θ/∂z for Student-t transformation."""
    theta = studentt_z_to_theta(z)
    omega, alpha, beta, nu = theta[0], theta[1], theta[2], theta[3]
    
    J = np.zeros((4, 4), dtype=np.float64)
    
    J[0, 0] = omega
    J[1, 1] = alpha * (1.0 - alpha)
    J[1, 2] = -alpha * beta
    J[2, 1] = -beta * alpha
    J[2, 2] = beta * (1.0 - beta)
    J[3, 3] = nu - 2.0
    
    return J


# =============================================================================
# SKEW-T GARCH TRANSFORMATIONS (5 parameters: omega, alpha, beta, nu, lambda)
# =============================================================================

def skewt_theta_to_z(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform Skew-t GARCH parameters from constrained θ to unconstrained z.
    
    θ = [omega, alpha, beta, nu, lambda]
    z = [z_omega, z_alpha, z_beta, z_nu, z_lambda]
    
    Transformations:
        omega = exp(z_omega)
        [alpha, beta] = softmax([z_alpha, z_beta, 0])[:2]
        nu = 2 + exp(z_nu)
        lambda = tanh(z_lambda)
    """
    omega, alpha, beta, nu, lam = theta[0], theta[1], theta[2], theta[3], theta[4]
    
    z_omega = np.log(omega)
    
    remainder = 1.0 - alpha - beta
    remainder = max(remainder, 1e-10)
    z_alpha = np.log(alpha / remainder)
    z_beta = np.log(beta / remainder)
    
    z_nu = np.log(nu - 2.0)
    
    lam_clipped = np.clip(lam, -0.999, 0.999)
    z_lambda = np.arctanh(lam_clipped)
    
    return np.array([z_omega, z_alpha, z_beta, z_nu, z_lambda])


def skewt_z_to_theta(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform Skew-t GARCH parameters from unconstrained z to constrained θ."""
    z_omega, z_alpha, z_beta, z_nu, z_lambda = z[0], z[1], z[2], z[3], z[4]
    
    omega = np.exp(np.clip(z_omega, LOG_CLIP_MIN, LOG_CLIP_MAX))
    
    z_all = np.array([z_alpha, z_beta, 0.0])
    log_denom = logsumexp(z_all)
    alpha = np.exp(z_alpha - log_denom)
    beta = np.exp(z_beta - log_denom)
    
    nu = 2.0 + np.exp(np.clip(z_nu, LOG_CLIP_MIN, LOG_CLIP_MAX))
    
    lam = np.tanh(z_lambda)
    
    return np.array([omega, alpha, beta, nu, lam])


def skewt_jacobian(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Jacobian J = ∂θ/∂z for Skew-t transformation."""
    theta = skewt_z_to_theta(z)
    omega, alpha, beta, nu, lam = theta[0], theta[1], theta[2], theta[3], theta[4]
    
    J = np.zeros((5, 5), dtype=np.float64)
    
    J[0, 0] = omega
    J[1, 1] = alpha * (1.0 - alpha)
    J[1, 2] = -alpha * beta
    J[2, 1] = -beta * alpha
    J[2, 2] = beta * (1.0 - beta)
    J[3, 3] = nu - 2.0
    J[4, 4] = 1.0 - lam ** 2
    
    return J


# =============================================================================
# ROBUST HESSIAN COMPUTATION (via reparameterization)
# =============================================================================

def _compute_robust_hessian(
    objective_theta: Callable[[NDArray[np.float64]], float],
    theta_hat: NDArray[np.float64],
    theta_to_z: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    z_to_theta: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    eps: float = 1e-5,
) -> RobustHessianResult:
    """
    Internal function to compute robust Hessian via reparameterization.
    
    Parameters
    ----------
    objective_theta : callable
        Negative log-likelihood in θ-space: f(θ) -> scalar
    theta_hat : array
        MLE estimates in original parameter space
    theta_to_z : callable
        Transform θ -> z (unconstrained)
    z_to_theta : callable
        Transform z -> θ (constrained)
    jacobian_fn : callable
        Compute Jacobian J = ∂θ/∂z at a point z
    eps : float
        Step size for finite differences
    
    Returns
    -------
    RobustHessianResult
    """
    # Step 1: Transform to unconstrained space
    try:
        z_hat = theta_to_z(theta_hat)
    except Exception as e:
        return RobustHessianResult(
            hessian_z=np.array([]),
            cov_z=None,
            hessian_theta=None,
            cov_theta=None,
            std_errors=None,
            jacobian=np.array([]),
            success=False,
            message=f"Failed to transform to z-space: {e}",
        )
    
    # Step 2: Define objective in z-space
    def objective_z(z: NDArray[np.float64]) -> float:
        theta = z_to_theta(z)
        return objective_theta(theta)
    
    # Step 3: Compute Hessian in z-space
    try:
        H_z = compute_hessian_unconstrained(objective_z, z_hat, eps=eps)
    except Exception as e:
        return RobustHessianResult(
            hessian_z=np.array([]),
            cov_z=None,
            hessian_theta=None,
            cov_theta=None,
            std_errors=None,
            jacobian=np.array([]),
            success=False,
            message=f"Failed to compute Hessian in z-space: {e}",
        )
    
    # Step 4: Compute covariance in z-space
    try:
        cov_z = np.linalg.inv(H_z)
    except np.linalg.LinAlgError:
        return RobustHessianResult(
            hessian_z=H_z,
            cov_z=None,
            hessian_theta=None,
            cov_theta=None,
            std_errors=None,
            jacobian=np.array([]),
            success=False,
            message="Hessian in z-space is singular",
        )
    
    # Step 5: Compute Jacobian J = ∂θ/∂z
    J = jacobian_fn(z_hat)
    
    # Step 6: Transform covariance to θ-space: Cov(θ) = J @ Cov(z) @ J^T
    cov_theta = J @ cov_z @ J.T
    
    # Step 7: Extract standard errors
    diag_cov = np.diag(cov_theta)
    if np.all(diag_cov > 0):
        std_errors = np.sqrt(diag_cov)
    elif np.any(diag_cov > 0):
        std_errors = np.where(diag_cov > 0, np.sqrt(diag_cov), np.nan)
    else:
        std_errors = None
    
    # Transform Hessian to θ-space: H_θ = J^{-T} @ H_z @ J^{-1}
    try:
        J_inv = np.linalg.inv(J)
        H_theta = J_inv.T @ H_z @ J_inv
    except np.linalg.LinAlgError:
        H_theta = None
    
    return RobustHessianResult(
        hessian_z=H_z,
        cov_z=cov_z,
        hessian_theta=H_theta,
        cov_theta=cov_theta,
        std_errors=std_errors,
        jacobian=J,
        success=std_errors is not None,
        message="Success" if std_errors is not None else "Covariance has negative diagonal",
    )


def compute_robust_hessian_normal(
    objective_theta: Callable[[NDArray[np.float64]], float],
    theta_hat: NDArray[np.float64],
    eps: float = 1e-5,
) -> RobustHessianResult:
    """
    Compute robust Hessian for Normal GARCH model.
    
    Parameters
    ----------
    objective_theta : callable
        Negative log-likelihood in θ-space: f(θ) -> scalar
        where θ = [omega, alpha, beta]
    theta_hat : array [omega, alpha, beta]
        MLE estimates
    eps : float
        Step size for finite differences
    
    Returns
    -------
    RobustHessianResult
    """
    return _compute_robust_hessian(
        objective_theta=objective_theta,
        theta_hat=theta_hat,
        theta_to_z=normal_theta_to_z,
        z_to_theta=normal_z_to_theta,
        jacobian_fn=normal_jacobian,
        eps=eps,
    )


def compute_robust_hessian_studentt(
    objective_theta: Callable[[NDArray[np.float64]], float],
    theta_hat: NDArray[np.float64],
    eps: float = 1e-5,
) -> RobustHessianResult:
    """
    Compute robust Hessian for Student-t GARCH model.
    
    Parameters
    ----------
    objective_theta : callable
        Negative log-likelihood in θ-space: f(θ) -> scalar
        where θ = [omega, alpha, beta, nu]
    theta_hat : array [omega, alpha, beta, nu]
        MLE estimates
    eps : float
        Step size for finite differences
    
    Returns
    -------
    RobustHessianResult
    """
    return _compute_robust_hessian(
        objective_theta=objective_theta,
        theta_hat=theta_hat,
        theta_to_z=studentt_theta_to_z,
        z_to_theta=studentt_z_to_theta,
        jacobian_fn=studentt_jacobian,
        eps=eps,
    )


def compute_robust_hessian_skewt(
    objective_theta: Callable[[NDArray[np.float64]], float],
    theta_hat: NDArray[np.float64],
    eps: float = 1e-5,
) -> RobustHessianResult:
    """
    Compute robust Hessian for Skew-t GARCH model.
    
    Parameters
    ----------
    objective_theta : callable
        Negative log-likelihood in θ-space: f(θ) -> scalar
        where θ = [omega, alpha, beta, nu, lambda]
    theta_hat : array [omega, alpha, beta, nu, lambda]
        MLE estimates
    eps : float
        Step size for finite differences
    
    Returns
    -------
    RobustHessianResult
    """
    return _compute_robust_hessian(
        objective_theta=objective_theta,
        theta_hat=theta_hat,
        theta_to_z=skewt_theta_to_z,
        z_to_theta=skewt_z_to_theta,
        jacobian_fn=skewt_jacobian,
        eps=eps,
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Numerical Hessian Computations")
    print("=" * 70)
    
    # Test 1: Transformation roundtrips
    print("\n--- Test 1: Transformation Roundtrips ---")
    
    theta_n = np.array([1e-5, 0.1, 0.85])
    z_n = normal_theta_to_z(theta_n)
    theta_n_rec = normal_z_to_theta(z_n)
    print(f"Normal: max error = {np.max(np.abs(theta_n - theta_n_rec)):.2e}")
    
    theta_t = np.array([1e-5, 0.1, 0.85, 8.0])
    z_t = studentt_theta_to_z(theta_t)
    theta_t_rec = studentt_z_to_theta(z_t)
    print(f"Student-t: max error = {np.max(np.abs(theta_t - theta_t_rec)):.2e}")
    
    theta_s = np.array([1e-5, 0.1, 0.85, 8.0, -0.1])
    z_s = skewt_theta_to_z(theta_s)
    theta_s_rec = skewt_z_to_theta(z_s)
    print(f"Skew-t: max error = {np.max(np.abs(theta_s - theta_s_rec)):.2e}")
    
    # Test 2: Jacobian numerical verification
    print("\n--- Test 2: Jacobian Verification ---")
    
    eps = 1e-6
    for name, theta, to_z, from_z, jac_fn in [
        ("Normal", theta_n, normal_theta_to_z, normal_z_to_theta, normal_jacobian),
        ("Student-t", theta_t, studentt_theta_to_z, studentt_z_to_theta, studentt_jacobian),
        ("Skew-t", theta_s, skewt_theta_to_z, skewt_z_to_theta, skewt_jacobian),
    ]:
        z = to_z(theta)
        J_analytic = jac_fn(z)
        
        k = len(z)
        J_numeric = np.zeros((k, k))
        for j in range(k):
            z_plus = z.copy()
            z_plus[j] += eps
            z_minus = z.copy()
            z_minus[j] -= eps
            J_numeric[:, j] = (from_z(z_plus) - from_z(z_minus)) / (2 * eps)
        
        max_diff = np.max(np.abs(J_analytic - J_numeric))
        print(f"{name}: Jacobian max diff = {max_diff:.2e}")
    
    # Test 3: Hessian computation
    print("\n--- Test 3: Robust Hessian (quadratic objective) ---")
    
    def test_obj(theta):
        target = np.array([1e-5, 0.1, 0.85, 8.0])
        return 0.5 * np.sum((theta - target) ** 2)
    
    result = compute_robust_hessian_studentt(test_obj, theta_t, eps=1e-5)
    print(f"Success: {result.success}")
    if result.std_errors is not None:
        print(f"SEs: {result.std_errors}")
    
    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)
