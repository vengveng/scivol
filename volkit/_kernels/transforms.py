"""
Parameter transformations for unconstrained optimization.

Transforms constrained GARCH parameters to unconstrained space:
- ω = exp(z_ω)                    ensures ω > 0
- (α, β, r) = softmax(z_α, z_β, 0)  ensures α,β > 0 and α + β < 1
- ν = 2 + exp(z_ν)                ensures ν > 2 (Student-t)
- λ = tanh(z_λ)                   ensures λ ∈ (-1, 1) (SkewT)

The Jacobian J = ∂θ/∂z is used to transform:
- Gradients: ∇_z = Jᵀ ∇_θ
- Covariances: Var(θ) = J Var(z) Jᵀ

This module provides both Python implementations and C-accelerated versions.
The C versions are faster but require pre-allocated output buffers.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from .. import _core


def _as_cptr(arr: NDArray[np.float64]) -> int:
    """Convert NumPy array to C pointer (as integer address)."""
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


# =============================================================================
# GARCH PARAMETER TRANSFORMS (shared across all distributions)
# =============================================================================

def pack_garch(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """
    Transform unconstrained z to constrained GARCH parameters θ.
    
    z = [z_omega, z_alpha_1, ..., z_alpha_p, z_beta_1, ..., z_beta_q]
    θ = [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
    
    Transformations:
        omega = exp(z_omega)
        (alpha_1, ..., alpha_p, beta_1, ..., beta_q, r) = softmax(z_alpha, z_beta, 0)
    
    This ensures:
        omega > 0
        alpha_i > 0, beta_j > 0
        sum(alpha) + sum(beta) < 1  (stationarity)
    """
    z_omega = z[0]
    z_alpha = z[1:1+p]
    z_beta = z[1+p:1+p+q]
    
    # Omega: simple exp transform
    omega = np.exp(np.clip(z_omega, -700.0, 700.0))
    
    # Joint softmax for alpha and beta with slack variable r
    # softmax([z_alpha, z_beta, 0]) gives [alpha, beta, r] where r = 1 - sum(alpha) - sum(beta)
    z_joint = np.concatenate([z_alpha, z_beta, [0.0]])
    lse_joint = logsumexp(z_joint)
    
    # Compute alpha and beta from joint softmax
    alpha = np.exp(z_alpha - lse_joint)
    beta = np.exp(z_beta - lse_joint)
    
    return np.concatenate([[omega], alpha, beta])


def unpack_garch(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """
    Transform constrained GARCH parameters θ to unconstrained z.
    
    This is the inverse of pack_garch().
    """
    omega = theta[0]
    alpha = theta[1:1+p]
    beta = theta[1+p:1+p+q]
    
    z_omega = np.log(omega)
    
    # Inverse joint softmax: z_i = log(theta_i) - log(r) 
    # where r = 1 - sum(alpha) - sum(beta)
    r = 1.0 - alpha.sum() - beta.sum()
    r = max(r, 1e-10)  # Numerical safety
    
    z_alpha = np.log(np.maximum(alpha, 1e-10)) - np.log(r)
    z_beta = np.log(np.maximum(beta, 1e-10)) - np.log(r)
    
    return np.concatenate([[z_omega], z_alpha, z_beta])


def jacobian_garch(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """
    Compute Jacobian J = ∂θ/∂z for GARCH parameters.
    
    For joint softmax over [z_alpha, z_beta, 0]:
        ∂alpha_i/∂z_alpha_j = alpha_i * (δ_ij - alpha_j)
        ∂alpha_i/∂z_beta_k = -alpha_i * beta_k
        ∂beta_i/∂z_alpha_j = -beta_i * alpha_j
        ∂beta_i/∂z_beta_k = beta_i * (δ_ik - beta_k)
    
    Returns K×K matrix where K = 1 + p + q.
    """
    K = 1 + p + q
    omega = theta[0]
    alpha = theta[1:1+p]
    beta = theta[1+p:1+p+q]
    
    J = np.zeros((K, K), dtype=np.float64)
    
    # ∂omega/∂z_omega = omega
    J[0, 0] = omega
    
    # Joint softmax Jacobian
    # ∂alpha_i/∂z_alpha_j = alpha_i * (δ_ij - alpha_j)
    J_alpha_alpha = np.diag(alpha) - np.outer(alpha, alpha)
    J[1:1+p, 1:1+p] = J_alpha_alpha
    
    # ∂alpha_i/∂z_beta_k = -alpha_i * beta_k
    J_alpha_beta = -np.outer(alpha, beta)
    J[1:1+p, 1+p:] = J_alpha_beta
    
    # ∂beta_i/∂z_alpha_j = -beta_i * alpha_j
    J_beta_alpha = -np.outer(beta, alpha)
    J[1+p:, 1:1+p] = J_beta_alpha
    
    # ∂beta_i/∂z_beta_k = beta_i * (δ_ik - beta_k)
    J_beta_beta = np.diag(beta) - np.outer(beta, beta)
    J[1+p:, 1+p:] = J_beta_beta
    
    return J


# =============================================================================
# SOFTPLUS HELPERS (for nu transform - gives nicer optimization landscape)
# =============================================================================

SOFTPLUS_THRESHOLD = 20.0  # For numerical stability


def softplus(x: float) -> float:
    """Numerically stable softplus: log(1 + exp(x))."""
    if x > SOFTPLUS_THRESHOLD:
        return x  # Avoid overflow
    return np.log1p(np.exp(x))


def softplus_inv(y: float) -> float:
    """Inverse softplus: log(exp(y) - 1)."""
    if y > SOFTPLUS_THRESHOLD:
        return y  # Approximate inverse for large y
    return np.log(np.expm1(y))


def softplus_deriv(x: float) -> float:
    """Derivative of softplus: sigmoid(x) = 1 / (1 + exp(-x))."""
    if x > SOFTPLUS_THRESHOLD:
        return 1.0
    if x < -SOFTPLUS_THRESHOLD:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# STUDENT-T PARAMETER TRANSFORMS
# =============================================================================

def pack_studentt(z_nu: float) -> float:
    """Transform unconstrained z_nu to constrained nu > 2 using softplus."""
    return 2.0 + softplus(z_nu)


def unpack_studentt(nu: float) -> float:
    """Transform constrained nu to unconstrained z_nu."""
    return softplus_inv(nu - 2.0)


def jacobian_studentt(nu: float) -> float:
    """Compute ∂nu/∂z_nu = softplus'(z_nu) = sigmoid(z_nu)."""
    # We need z_nu = softplus_inv(nu - 2)
    z_nu = softplus_inv(nu - 2.0)
    return softplus_deriv(z_nu)


# =============================================================================
# SKEW-T PARAMETER TRANSFORMS
# =============================================================================

def pack_skewt(z_nu: float, z_lam: float) -> tuple[float, float]:
    """
    Transform unconstrained (z_nu, z_lam) to constrained (nu, lambda).
    
    nu = 2 + softplus(z_nu)    ensures nu > 2
    lambda = tanh(z_lam)       ensures lambda ∈ (-1, 1)
    """
    nu = 2.0 + softplus(z_nu)
    lam = np.tanh(z_lam)
    return nu, lam


def unpack_skewt(nu: float, lam: float) -> tuple[float, float]:
    """Transform constrained (nu, lambda) to unconstrained (z_nu, z_lam)."""
    z_nu = softplus_inv(nu - 2.0)
    z_lam = np.arctanh(np.clip(lam, -0.999, 0.999))
    return z_nu, z_lam


def jacobian_skewt(nu: float, lam: float) -> NDArray[np.float64]:
    """
    Compute Jacobian J = ∂(nu, lambda)/∂(z_nu, z_lam).
    
    Returns 2×2 diagonal matrix:
        ∂nu/∂z_nu = softplus'(z_nu) = sigmoid(z_nu)
        ∂lambda/∂z_lam = 1 - lambda²  (derivative of tanh)
    """
    z_nu = softplus_inv(nu - 2.0)
    J = np.zeros((2, 2), dtype=np.float64)
    J[0, 0] = softplus_deriv(z_nu)
    J[1, 1] = 1.0 - lam * lam  # sech²(z_lam) = 1 - tanh²(z_lam)
    return J


# =============================================================================
# COMBINED TRANSFORMS FOR FULL PARAMETER VECTORS
# =============================================================================

def pack_garch_studentt(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """
    Transform unconstrained z to constrained θ for GARCH + Student-t.
    
    z = [z_omega, z_alpha..., z_beta..., z_nu]
    θ = [omega, alpha..., beta..., nu]
    """
    n_garch = 1 + p + q
    
    theta_garch = pack_garch(z[:n_garch], p, q)
    nu = pack_studentt(z[n_garch])
    
    return np.concatenate([theta_garch, [nu]])


def unpack_garch_studentt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for GARCH + Student-t."""
    n_garch = 1 + p + q
    
    z_garch = unpack_garch(theta[:n_garch], p, q)
    z_nu = unpack_studentt(theta[n_garch])
    
    return np.concatenate([z_garch, [z_nu]])


def jacobian_garch_studentt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Compute full Jacobian for GARCH + Student-t."""
    n_garch = 1 + p + q
    K = n_garch + 1
    
    J = np.zeros((K, K), dtype=np.float64)
    
    # GARCH block
    J[:n_garch, :n_garch] = jacobian_garch(theta[:n_garch], p, q)
    
    # nu
    J[n_garch, n_garch] = jacobian_studentt(theta[n_garch])
    
    return J


def pack_garch_skewt(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """
    Transform unconstrained z to constrained θ for GARCH + SkewT.
    
    z = [z_omega, z_alpha..., z_beta..., z_nu, z_lam]
    θ = [omega, alpha..., beta..., nu, lambda]
    """
    n_garch = 1 + p + q
    
    theta_garch = pack_garch(z[:n_garch], p, q)
    nu, lam = pack_skewt(z[n_garch], z[n_garch + 1])
    
    return np.concatenate([theta_garch, [nu, lam]])


def unpack_garch_skewt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for GARCH + SkewT."""
    n_garch = 1 + p + q
    
    z_garch = unpack_garch(theta[:n_garch], p, q)
    z_nu, z_lam = unpack_skewt(theta[n_garch], theta[n_garch + 1])
    
    return np.concatenate([z_garch, [z_nu, z_lam]])


def jacobian_garch_skewt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Compute full Jacobian for GARCH + SkewT."""
    n_garch = 1 + p + q
    K = n_garch + 2
    
    J = np.zeros((K, K), dtype=np.float64)
    
    # GARCH block
    J[:n_garch, :n_garch] = jacobian_garch(theta[:n_garch], p, q)
    
    # Distribution parameters
    J_dist = jacobian_skewt(theta[n_garch], theta[n_garch + 1])
    J[n_garch:, n_garch:] = J_dist
    
    return J


# =============================================================================
# GRADIENT AND HESSIAN TRANSFORMS
# =============================================================================

def transform_gradient(grad_theta: NDArray[np.float64], J: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform gradient from θ-space to z-space.
    
    ∇_z L̃(z) = Jᵀ ∇_θ L(θ)
    """
    return J.T @ grad_theta


def transform_covariance(cov_z: NDArray[np.float64], J: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform covariance from z-space to θ-space.
    
    Var(θ) = J Var(z) Jᵀ
    """
    return J @ cov_z @ J.T


def compute_se_via_logspace(
    theta_hat: NDArray[np.float64],
    nll_theta: callable,
    unpack_fn: callable,
    jacobian_fn: callable,
    pack_fn: callable,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """
    Compute standard errors via numerical Hessian in unconstrained (log) space.
    
    This approach avoids boundary issues when computing finite differences:
    1. Transform θ_hat → z_hat (unconstrained space)
    2. Compute numerical Hessian H_z in z-space (safe - no boundaries)
    3. Compute Cov_z = inv(H_z)
    4. Transform back: Cov_θ = J @ Cov_z @ J^T
    
    Parameters
    ----------
    theta_hat : array
        Optimal parameters in constrained space
    nll_theta : callable
        Negative log-likelihood function: nll_theta(theta) -> float
    unpack_fn : callable
        Transform θ → z: unpack_fn(theta) -> z
    jacobian_fn : callable
        Compute Jacobian: jacobian_fn(theta) -> J where J = ∂θ/∂z
    pack_fn : callable
        Transform z → θ: pack_fn(z) -> theta
    
    Returns
    -------
    hessian_theta : array or None
        Hessian in theta-space (transformed from z-space)
    cov_theta : array or None
        Covariance matrix in theta-space
    """
    K = len(theta_hat)
    
    # Transform to unconstrained space
    z_hat = unpack_fn(theta_hat)
    
    # Define NLL in z-space
    def nll_z(z: NDArray[np.float64]) -> float:
        theta = pack_fn(z)
        return nll_theta(theta)
    
    # Pre-allocate buffers for finite difference computation (avoids 4×K² allocations)
    z_pp = np.empty(K, dtype=np.float64)
    z_pm = np.empty(K, dtype=np.float64)
    z_mp = np.empty(K, dtype=np.float64)
    z_mm = np.empty(K, dtype=np.float64)
    
    # Pre-compute step sizes for each parameter
    eps = np.array([1e-5 * max(abs(z_hat[k]), 1.0) for k in range(K)], dtype=np.float64)
    
    # Compute numerical Hessian in z-space using relative step sizes
    # Step size 1e-5 is optimal for second derivative finite differences:
    # - Smaller values (1e-7, 1e-8) cause round-off error amplification
    # - Larger values (1e-3, 1e-4) cause truncation error
    # In unconstrained space, no boundary issues, so 1e-5 is reliable
    H_z = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        eps_i = eps[i]
        for j in range(K):
            eps_j = eps[j]
            
            # Reuse pre-allocated buffers instead of calling z.copy()
            z_pp[:] = z_hat; z_pp[i] += eps_i; z_pp[j] += eps_j
            z_pm[:] = z_hat; z_pm[i] += eps_i; z_pm[j] -= eps_j
            z_mp[:] = z_hat; z_mp[i] -= eps_i; z_mp[j] += eps_j
            z_mm[:] = z_hat; z_mm[i] -= eps_i; z_mm[j] -= eps_j
            
            H_z[i, j] = (nll_z(z_pp) - nll_z(z_pm) - nll_z(z_mp) + nll_z(z_mm)) / (4 * eps_i * eps_j)
    
    # Compute covariance in z-space
    try:
        cov_z = np.linalg.inv(H_z)
    except np.linalg.LinAlgError:
        return None, None
    
    # Compute Jacobian at theta_hat: J = ∂θ/∂z
    J = jacobian_fn(theta_hat)
    
    # Transform to theta-space: Cov_θ = J @ Cov_z @ J^T
    cov_theta = J @ cov_z @ J.T
    
    # Transform Hessian (for completeness): H_θ = inv(J) @ H_z @ inv(J)^T ≈ inv(Cov_θ)
    # More directly: H_θ = inv(Cov_θ) but we compute from H_z for consistency
    try:
        hessian_theta = np.linalg.inv(cov_theta)
    except np.linalg.LinAlgError:
        hessian_theta = None
    
    return hessian_theta, cov_theta


# =============================================================================
# C-ACCELERATED VERSIONS (use pre-allocated buffers)
# =============================================================================

def pack_garch_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    """
    C-accelerated pack_garch (modifies theta_out in-place).
    
    theta_out must be pre-allocated with shape (1 + p + q,).
    """
    if p == 1 and q == 1:
        _core._pack_garch_11(_as_cptr(z), _as_cptr(theta_out))
    else:
        _core._pack_garch_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


def pack_garch_studentt_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    """
    C-accelerated pack_garch_studentt (modifies theta_out in-place).
    
    theta_out must be pre-allocated with shape (2 + p + q,).
    """
    if p == 1 and q == 1:
        _core._pack_garch_studentt_11(_as_cptr(z), _as_cptr(theta_out))
    else:
        _core._pack_garch_studentt_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


def pack_garch_skewt_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    """
    C-accelerated pack_garch_skewt (modifies theta_out in-place).
    
    theta_out must be pre-allocated with shape (3 + p + q,).
    """
    if p == 1 and q == 1:
        _core._pack_garch_skewt_11(_as_cptr(z), _as_cptr(theta_out))
    else:
        _core._pack_garch_skewt_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


def jacobian_garch_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    """
    C-accelerated jacobian_garch (modifies J_out in-place).
    
    J_out must be pre-allocated with shape (K, K) where K = 1 + p + q.
    """
    if p == 1 and q == 1:
        _core._jacobian_garch_11(_as_cptr(theta), _as_cptr(J_out))
    else:
        _core._jacobian_garch_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


def jacobian_garch_studentt_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    """
    C-accelerated jacobian_garch_studentt (modifies J_out in-place).
    
    J_out must be pre-allocated with shape (K, K) where K = 2 + p + q.
    """
    if p == 1 and q == 1:
        _core._jacobian_garch_studentt_11(_as_cptr(theta), _as_cptr(J_out))
    else:
        _core._jacobian_garch_studentt_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


def jacobian_garch_skewt_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    """
    C-accelerated jacobian_garch_skewt (modifies J_out in-place).
    
    J_out must be pre-allocated with shape (K, K) where K = 3 + p + q.
    """
    if p == 1 and q == 1:
        _core._jacobian_garch_skewt_11(_as_cptr(theta), _as_cptr(J_out))
    else:
        _core._jacobian_garch_skewt_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


def transform_grad_c(
    grad_theta: NDArray[np.float64],
    J: NDArray[np.float64],
    grad_z_out: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal"
) -> None:
    """
    C-accelerated gradient transform: grad_z = J^T @ grad_theta (modifies grad_z_out in-place).
    
    dist should be "normal", "studentt", or "skewt".
    """
    K = grad_theta.shape[0]
    
    if p == 1 and q == 1:
        if dist == "normal":
            _core._transform_grad_11_normal(_as_cptr(grad_theta), _as_cptr(J), _as_cptr(grad_z_out))
        elif dist == "studentt":
            _core._transform_grad_11_studentt(_as_cptr(grad_theta), _as_cptr(J), _as_cptr(grad_z_out))
        elif dist == "skewt":
            _core._transform_grad_11_skewt(_as_cptr(grad_theta), _as_cptr(J), _as_cptr(grad_z_out))
        else:
            raise ValueError(f"Unknown distribution: {dist}")
    else:
        _core._transform_grad_pq(_as_cptr(grad_theta), _as_cptr(J), _as_cptr(grad_z_out), K)


# =============================================================================
# ARMA-GARCH PARAMETER TRANSFORMS
# =============================================================================

def pack_arma_garch_normal(
    z: NDArray[np.float64], 
    p_ar: int, 
    q_ma: int, 
    P_arch: int, 
    Q_garch: int
) -> NDArray[np.float64]:
    """
    Transform unconstrained z to constrained θ for ARMA(p,q)+GARCH(P,Q)+Normal.
    
    z = [z_c, z_phi_1..p, z_theta_1..q, z_omega, z_alpha_1..P, z_beta_1..Q]
    θ = [c, phi_1..p, theta_1..q, omega, alpha_1..P, beta_1..Q]
    
    Transformations:
        c = z_c (unbounded)
        phi_i = 0.99 * tanh(z_phi_i)  ensures |phi_i| < 0.99
        theta_j = 0.99 * tanh(z_theta_j)  ensures |theta_j| < 0.99
        omega = exp(z_omega)  ensures omega > 0
        (alpha, beta, r) = softmax([z_alpha, z_beta, 0])  ensures α+β < 1
    """
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    
    # Mean parameters
    c = z[0]  # Unbounded
    phi = 0.99 * np.tanh(z[1:1+p_ar]) if p_ar > 0 else np.array([])
    theta = 0.99 * np.tanh(z[1+p_ar:n_mean]) if q_ma > 0 else np.array([])
    
    # GARCH parameters (same as pure GARCH)
    z_garch = z[n_mean:n_mean+n_vol]
    theta_garch = pack_garch(z_garch, P_arch, Q_garch)
    
    return np.concatenate([[c], phi, theta, theta_garch])


def unpack_arma_garch_normal(
    theta: NDArray[np.float64], 
    p_ar: int, 
    q_ma: int, 
    P_arch: int, 
    Q_garch: int
) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for ARMA+GARCH+Normal."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    
    # Mean parameters
    z_c = theta[0]
    z_phi = np.arctanh(np.clip(theta[1:1+p_ar] / 0.99, -0.999, 0.999)) if p_ar > 0 else np.array([])
    z_theta = np.arctanh(np.clip(theta[1+p_ar:n_mean] / 0.99, -0.999, 0.999)) if q_ma > 0 else np.array([])
    
    # GARCH parameters
    z_garch = unpack_garch(theta[n_mean:n_mean+n_vol], P_arch, Q_garch)
    
    return np.concatenate([[z_c], z_phi, z_theta, z_garch])


def jacobian_arma_garch_normal(
    theta: NDArray[np.float64], 
    p_ar: int, 
    q_ma: int, 
    P_arch: int, 
    Q_garch: int
) -> NDArray[np.float64]:
    """Compute Jacobian J = ∂θ/∂z for ARMA+GARCH+Normal."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    K = n_mean + n_vol
    
    J = np.zeros((K, K), dtype=np.float64)
    
    # c: identity
    J[0, 0] = 1.0
    
    # phi: 0.99 * tanh(z) -> ∂phi/∂z = 0.99 * (1 - tanh²(z)) = 0.99 * (1 - (phi/0.99)²)
    for i in range(p_ar):
        phi_i = theta[1 + i]
        J[1 + i, 1 + i] = 0.99 * (1.0 - (phi_i / 0.99) ** 2)
    
    # theta: same as phi
    for j in range(q_ma):
        theta_j = theta[1 + p_ar + j]
        J[1 + p_ar + j, 1 + p_ar + j] = 0.99 * (1.0 - (theta_j / 0.99) ** 2)
    
    # GARCH block
    J_garch = jacobian_garch(theta[n_mean:n_mean+n_vol], P_arch, Q_garch)
    J[n_mean:, n_mean:] = J_garch
    
    return J


def pack_arma_garch_studentt(
    z: NDArray[np.float64], 
    p_ar: int, 
    q_ma: int, 
    P_arch: int, 
    Q_garch: int
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for ARMA+GARCH+StudentT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    
    # ARMA+GARCH params
    theta_base = pack_arma_garch_normal(z[:n_mean+n_vol], p_ar, q_ma, P_arch, Q_garch)
    
    # nu parameter
    nu = pack_studentt(z[n_mean + n_vol])
    
    return np.concatenate([theta_base, [nu]])


def unpack_arma_garch_studentt(
    theta: NDArray[np.float64], 
    p_ar: int, 
    q_ma: int, 
    P_arch: int, 
    Q_garch: int
) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for ARMA+GARCH+StudentT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    
    z_base = unpack_arma_garch_normal(theta[:n_mean+n_vol], p_ar, q_ma, P_arch, Q_garch)
    z_nu = unpack_studentt(theta[n_mean + n_vol])
    
    return np.concatenate([z_base, [z_nu]])


def jacobian_arma_garch_studentt(
    theta: NDArray[np.float64], 
    p_ar: int, 
    q_ma: int, 
    P_arch: int, 
    Q_garch: int
) -> NDArray[np.float64]:
    """Compute Jacobian for ARMA+GARCH+StudentT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    K = n_mean + n_vol + 1
    
    J = np.zeros((K, K), dtype=np.float64)
    
    # ARMA+GARCH block
    J[:n_mean+n_vol, :n_mean+n_vol] = jacobian_arma_garch_normal(
        theta[:n_mean+n_vol], p_ar, q_ma, P_arch, Q_garch
    )
    
    # nu
    J[K-1, K-1] = jacobian_studentt(theta[n_mean + n_vol])
    
    return J


def pack_arma_garch_skewt(
    z: NDArray[np.float64], 
    p_ar: int, 
    q_ma: int, 
    P_arch: int, 
    Q_garch: int
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for ARMA+GARCH+SkewT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    
    # ARMA+GARCH params
    theta_base = pack_arma_garch_normal(z[:n_mean+n_vol], p_ar, q_ma, P_arch, Q_garch)
    
    # nu and lam parameters
    nu, lam = pack_skewt(z[n_mean + n_vol], z[n_mean + n_vol + 1])
    
    return np.concatenate([theta_base, [nu, lam]])


def unpack_arma_garch_skewt(
    theta: NDArray[np.float64], 
    p_ar: int, 
    q_ma: int, 
    P_arch: int, 
    Q_garch: int
) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for ARMA+GARCH+SkewT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    
    z_base = unpack_arma_garch_normal(theta[:n_mean+n_vol], p_ar, q_ma, P_arch, Q_garch)
    z_nu, z_lam = unpack_skewt(theta[n_mean + n_vol], theta[n_mean + n_vol + 1])
    
    return np.concatenate([z_base, [z_nu, z_lam]])


def jacobian_arma_garch_skewt(
    theta: NDArray[np.float64], 
    p_ar: int, 
    q_ma: int, 
    P_arch: int, 
    Q_garch: int
) -> NDArray[np.float64]:
    """Compute Jacobian for ARMA+GARCH+SkewT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    K = n_mean + n_vol + 2
    
    J = np.zeros((K, K), dtype=np.float64)
    
    # ARMA+GARCH block
    J[:n_mean+n_vol, :n_mean+n_vol] = jacobian_arma_garch_normal(
        theta[:n_mean+n_vol], p_ar, q_ma, P_arch, Q_garch
    )
    
    # nu and lam
    J_dist = jacobian_skewt(theta[n_mean + n_vol], theta[n_mean + n_vol + 1])
    J[n_mean+n_vol:, n_mean+n_vol:] = J_dist
    
    return J
