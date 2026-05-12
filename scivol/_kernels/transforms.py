"""
Parameter transformations for unconstrained optimization.

Transforms constrained GARCH parameters to unconstrained space:
- ω = softplus(z_ω)               ensures ω > 0
- (α, β, r) = softmax(z_α, z_β, 0)  ensures α,β > 0 and α + β < 1
- ν = 2 + softplus(z_ν)           ensures ν > 2 (Student-t)
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

GED_NU_MIN = 1.01


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
        omega = softplus(z_omega)
        (alpha_1, ..., alpha_p, beta_1, ..., beta_q, r) = softmax(z_alpha, z_beta, 0)
    
    This ensures:
        omega > 0
        alpha_i > 0, beta_j > 0
        sum(alpha) + sum(beta) < 1  (stationarity)
    """
    z_omega = z[0]
    z_alpha = z[1:1+p]
    z_beta = z[1+p:1+p+q]
    
    omega = softplus(z_omega)
    
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
    
    z_omega = softplus_inv(omega)
    
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
    
    # ∂omega/∂z_omega for omega = softplus(z_omega)
    J[0, 0] = _softplus_derivative_from_positive(omega)
    
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
POSITIVE_FLOOR = np.finfo(np.float64).tiny


def softplus(x: float) -> float:
    """Numerically stable softplus: log(1 + exp(x))."""
    if x > SOFTPLUS_THRESHOLD:
        return x  # Avoid overflow
    return np.log1p(np.exp(x))


def softplus_inv(y: float) -> float:
    """Inverse softplus: log(exp(y) - 1)."""
    y = max(float(y), POSITIVE_FLOOR)
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


def _softplus_derivative_from_positive(value: float) -> float:
    """Derivative of value = softplus(z) expressed in terms of positive θ = value."""
    value = max(float(value), 0.0)
    if value > SOFTPLUS_THRESHOLD:
        return 1.0
    return -np.expm1(-value)


def _softplus_second_derivative_from_positive(value: float) -> float:
    """Second derivative of value = softplus(z) expressed in terms of positive θ = value."""
    first = _softplus_derivative_from_positive(value)
    return first * (1.0 - first)


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
    return _softplus_derivative_from_positive(nu - 2.0)


# =============================================================================
# GED PARAMETER TRANSFORMS
# =============================================================================

def pack_ged(z_nu: float) -> float:
    """Transform unconstrained z_nu to constrained nu > 1.01 using softplus."""
    return GED_NU_MIN + softplus(z_nu)


def unpack_ged(nu: float) -> float:
    """Transform constrained nu to unconstrained z_nu."""
    return softplus_inv(nu - GED_NU_MIN)


def jacobian_ged(nu: float) -> float:
    """Compute ∂nu/∂z_nu for the GED shape parameter."""
    return _softplus_derivative_from_positive(nu - GED_NU_MIN)


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
    J = np.zeros((2, 2), dtype=np.float64)
    J[0, 0] = _softplus_derivative_from_positive(nu - 2.0)
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


def pack_garch_ged(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """
    Transform unconstrained z to constrained theta for GARCH + GED.

    z = [z_omega, z_alpha..., z_beta..., z_nu]
    theta = [omega, alpha..., beta..., nu]
    """
    n_garch = 1 + p + q

    theta_garch = pack_garch(z[:n_garch], p, q)
    nu = pack_ged(z[n_garch])

    return np.concatenate([theta_garch, [nu]])


def unpack_garch_ged(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform constrained theta to unconstrained z for GARCH + GED."""
    n_garch = 1 + p + q

    z_garch = unpack_garch(theta[:n_garch], p, q)
    z_nu = unpack_ged(theta[n_garch])

    return np.concatenate([z_garch, [z_nu]])


def jacobian_garch_ged(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Compute full Jacobian for GARCH + GED."""
    n_garch = 1 + p + q
    K = n_garch + 1

    J = np.zeros((K, K), dtype=np.float64)
    J[:n_garch, :n_garch] = jacobian_garch(theta[:n_garch], p, q)
    J[n_garch, n_garch] = jacobian_ged(theta[n_garch])
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


def pack_arma_normal(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for ARMA(p,q)+Normal."""
    theta = np.empty(1 + p + q, dtype=np.float64)
    theta[0] = z[0]
    if p > 0:
        theta[1:1 + p] = 0.99 * np.tanh(z[1:1 + p])
    if q > 0:
        theta[1 + p:] = 0.99 * np.tanh(z[1 + p:])
    return theta


def unpack_arma_normal(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for ARMA(p,q)+Normal."""
    z = np.empty(1 + p + q, dtype=np.float64)
    z[0] = theta[0]
    if p > 0:
        z[1:1 + p] = np.arctanh(np.clip(theta[1:1 + p] / 0.99, -0.999, 0.999))
    if q > 0:
        z[1 + p:] = np.arctanh(np.clip(theta[1 + p:] / 0.99, -0.999, 0.999))
    return z


def jacobian_arma_normal(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Compute Jacobian J = ∂θ/∂z for ARMA(p,q)+Normal."""
    K = 1 + p + q
    J = np.eye(K, dtype=np.float64)
    scale = 0.99
    for idx in range(1, K):
        ratio = theta[idx] / scale
        J[idx, idx] = scale * (1.0 - ratio * ratio)
    return J


def _softmax_second_derivatives(params: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Second derivatives for active softmax outputs.

    Returns tensor T with shape (m, m, m) where
    T[i, j, k] = ∂² params[i] / ∂z_j ∂z_k.
    """
    m = params.shape[0]
    tensor = np.zeros((m, m, m), dtype=np.float64)
    for i in range(m):
        p_i = params[i]
        for j in range(m):
            delta_ij = 1.0 if i == j else 0.0
            p_j = params[j]
            for k in range(m):
                delta_ik = 1.0 if i == k else 0.0
                delta_jk = 1.0 if j == k else 0.0
                p_k = params[k]
                tensor[i, j, k] = p_i * (
                    (delta_ik - p_k) * (delta_ij - p_j)
                    - p_j * (delta_jk - p_k)
                )
    return tensor


def _softplus_second_derivative(nu: float) -> float:
    """Second derivative of nu = 2 + softplus(z_nu)."""
    return _softplus_second_derivative_from_positive(nu - 2.0)


def _tanh_scaled_second_derivative(value: float, scale: float = 0.99) -> float:
    """Second derivative of value = scale * tanh(z) expressed in θ-space."""
    ratio = value / scale
    return -2.0 * value * (1.0 - ratio * ratio)


def _tanh_second_derivative(value: float) -> float:
    """Second derivative of value = tanh(z) expressed in θ-space."""
    return -2.0 * value * (1.0 - value * value)


def transform_hessian(
    hess_theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    J: NDArray[np.float64],
    second_derivatives: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Transform Hessian from θ-space to z-space.

    H_z = Jᵀ H_θ J + Σ_i grad_θ[i] * ∂²θ_i/∂z∂zᵀ
    """
    hess_z = J.T @ hess_theta @ J + np.tensordot(grad_theta, second_derivatives, axes=(0, 0))
    return 0.5 * (hess_z + hess_z.T)


def second_derivatives_arma_normal(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Second-derivative tensor for the ARMA(p,q)+Normal log transform."""
    K = 1 + p + q
    tensor = np.zeros((K, K, K), dtype=np.float64)
    for idx in range(1, K):
        tensor[idx, idx, idx] = _tanh_scaled_second_derivative(theta[idx])
    return tensor


def log_hessian_arma_normal(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p: int,
    q: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA(p,q)+Normal."""
    J = jacobian_arma_normal(theta, p, q)
    second = second_derivatives_arma_normal(theta, p, q)
    return transform_hessian(hess_theta, grad_theta, J, second)


def pack_arma_ged(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for ARMA(p,q)+GED."""
    theta_base = pack_arma_normal(z[:1 + p + q], p, q)
    sigma2 = softplus(z[1 + p + q])
    nu = pack_ged(z[2 + p + q])
    return np.concatenate([theta_base, [sigma2, nu]])


def unpack_arma_ged(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for ARMA(p,q)+GED."""
    z_base = unpack_arma_normal(theta[:1 + p + q], p, q)
    z_sigma2 = softplus_inv(theta[1 + p + q])
    z_nu = unpack_ged(theta[2 + p + q])
    return np.concatenate([z_base, [z_sigma2, z_nu]])


def jacobian_arma_ged(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Compute Jacobian J = ∂θ/∂z for ARMA(p,q)+GED."""
    K_base = 1 + p + q
    K = K_base + 2
    J = np.zeros((K, K), dtype=np.float64)
    J[:K_base, :K_base] = jacobian_arma_normal(theta[:K_base], p, q)
    J[K_base, K_base] = _softplus_derivative_from_positive(theta[K_base])
    J[K_base + 1, K_base + 1] = jacobian_ged(theta[K_base + 1])
    return J


def second_derivatives_arma_ged(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Second-derivative tensor for the ARMA(p,q)+GED log transform."""
    K_base = 1 + p + q
    K = K_base + 2
    tensor = np.zeros((K, K, K), dtype=np.float64)
    tensor[:K_base, :K_base, :K_base] = second_derivatives_arma_normal(theta[:K_base], p, q)
    tensor[K_base, K_base, K_base] = _softplus_second_derivative_from_positive(theta[K_base])
    tensor[K_base + 1, K_base + 1, K_base + 1] = _softplus_second_derivative_from_positive(theta[K_base + 1] - GED_NU_MIN)
    return tensor


def log_hessian_arma_ged(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p: int,
    q: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA(p,q)+GED."""
    J = jacobian_arma_ged(theta, p, q)
    second = second_derivatives_arma_ged(theta, p, q)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_arma_garch_normal(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA(p,q)+GARCH(P,Q)+Normal log transforms."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    K = n_mean + n_vol
    tensor = np.zeros((K, K, K), dtype=np.float64)

    for idx in range(1, n_mean):
        tensor[idx, idx, idx] = _tanh_scaled_second_derivative(theta[idx])

    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_garch(
        theta[n_mean:], P_arch, Q_garch, dist="normal"
    )
    return tensor


def log_hessian_arma_garch_normal(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA(p,q)+GARCH(P,Q)+Normal."""
    J = jacobian_arma_garch_normal(theta, p_ar, q_ma, P_arch, Q_garch)
    second = second_derivatives_arma_garch_normal(theta, p_ar, q_ma, P_arch, Q_garch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_arma_garch_studentt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA(p,q)+GARCH(P,Q)+Student-t log transforms."""
    n_mean = 1 + p_ar + q_ma
    K = n_mean + 1 + P_arch + Q_garch + 1
    tensor = np.zeros((K, K, K), dtype=np.float64)

    for idx in range(1, n_mean):
        tensor[idx, idx, idx] = _tanh_scaled_second_derivative(theta[idx])

    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_garch(
        theta[n_mean:], P_arch, Q_garch, dist="studentt"
    )
    return tensor


def log_hessian_arma_garch_studentt(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA(p,q)+GARCH(P,Q)+Student-t."""
    J = jacobian_arma_garch_studentt(theta, p_ar, q_ma, P_arch, Q_garch)
    second = second_derivatives_arma_garch_studentt(theta, p_ar, q_ma, P_arch, Q_garch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_arma_garch_ged(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA(p,q)+GARCH(P,Q)+GED log transforms."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    K = n_mean + n_vol + 1
    tensor = np.zeros((K, K, K), dtype=np.float64)
    tensor[:n_mean + n_vol, :n_mean + n_vol, :n_mean + n_vol] = second_derivatives_arma_garch_normal(
        theta[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_garch
    )
    tensor[K - 1, K - 1, K - 1] = _softplus_second_derivative_from_positive(theta[K - 1] - GED_NU_MIN)
    return tensor


def log_hessian_arma_garch_ged(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA(p,q)+GARCH(P,Q)+GED."""
    J = jacobian_arma_garch_ged(theta, p_ar, q_ma, P_arch, Q_garch)
    second = second_derivatives_arma_garch_ged(theta, p_ar, q_ma, P_arch, Q_garch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_arma_egarch_normal(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA(p,q)+EGARCH(P,Q)+Normal log transforms."""
    n_mean = 1 + p_ar + q_ma
    K = n_mean + 1 + 2 * P_arch + Q_egarch
    tensor = np.zeros((K, K, K), dtype=np.float64)

    for idx in range(1, n_mean):
        tensor[idx, idx, idx] = _tanh_scaled_second_derivative(theta[idx])

    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_egarch(
        theta[n_mean:], P_arch, Q_egarch, dist="normal"
    )
    return tensor


def log_hessian_arma_egarch_normal(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA(p,q)+EGARCH(P,Q)+Normal."""
    J = jacobian_arma_egarch_normal(theta, p_ar, q_ma, P_arch, Q_egarch)
    second = second_derivatives_arma_egarch_normal(theta, p_ar, q_ma, P_arch, Q_egarch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_arma_egarch_studentt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA(p,q)+EGARCH(P,Q)+Student-t log transforms."""
    n_mean = 1 + p_ar + q_ma
    K = n_mean + 1 + 2 * P_arch + Q_egarch + 1
    tensor = np.zeros((K, K, K), dtype=np.float64)

    for idx in range(1, n_mean):
        tensor[idx, idx, idx] = _tanh_scaled_second_derivative(theta[idx])

    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_egarch(
        theta[n_mean:], P_arch, Q_egarch, dist="studentt"
    )
    return tensor


def log_hessian_arma_egarch_studentt(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA(p,q)+EGARCH(P,Q)+Student-t."""
    J = jacobian_arma_egarch_studentt(theta, p_ar, q_ma, P_arch, Q_egarch)
    second = second_derivatives_arma_egarch_studentt(theta, p_ar, q_ma, P_arch, Q_egarch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_arma_garch_skewt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA(p,q)+GARCH(P,Q)+Skew-t log transforms."""
    n_mean = 1 + p_ar + q_ma
    K = n_mean + 1 + P_arch + Q_garch + 2
    tensor = np.zeros((K, K, K), dtype=np.float64)

    for idx in range(1, n_mean):
        tensor[idx, idx, idx] = _tanh_scaled_second_derivative(theta[idx])

    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_garch(
        theta[n_mean:], P_arch, Q_garch, dist="skewt"
    )
    return tensor


def log_hessian_arma_garch_skewt(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA(p,q)+GARCH(P,Q)+Skew-t."""
    J = jacobian_arma_garch_skewt(theta, p_ar, q_ma, P_arch, Q_garch)
    second = second_derivatives_arma_garch_skewt(theta, p_ar, q_ma, P_arch, Q_garch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_garch(
    theta: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> NDArray[np.float64]:
    """Second-derivative tensor for GARCH(p,q) log transforms."""
    K = 1 + p + q + (1 if dist in {"studentt", "ged"} else 2 if dist == "skewt" else 0)
    tensor = np.zeros((K, K, K), dtype=np.float64)
    tensor[0, 0, 0] = _softplus_second_derivative_from_positive(theta[0])

    m = p + q
    if m > 0:
        tensor[1:1 + m, 1:1 + m, 1:1 + m] = _softmax_second_derivatives(theta[1:1 + m])

    if dist == "studentt":
        tensor[K - 1, K - 1, K - 1] = _softplus_second_derivative(theta[K - 1])
    elif dist == "ged":
        tensor[K - 1, K - 1, K - 1] = _softplus_second_derivative_from_positive(theta[K - 1] - GED_NU_MIN)
    elif dist == "skewt":
        tensor[K - 2, K - 2, K - 2] = _softplus_second_derivative(theta[K - 2])
        tensor[K - 1, K - 1, K - 1] = _tanh_second_derivative(theta[K - 1])

    return tensor


def log_hessian_garch(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for GARCH log-mode transforms."""
    if dist == "normal":
        J = jacobian_garch(theta, p, q)
    elif dist == "studentt":
        J = jacobian_garch_studentt(theta, p, q)
    elif dist == "ged":
        J = jacobian_garch_ged(theta, p, q)
    elif dist == "skewt":
        J = jacobian_garch_skewt(theta, p, q)
    else:
        raise ValueError(f"Unknown distribution: {dist}")
    second = second_derivatives_garch(theta, p, q, dist)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_gjr_garch(
    theta: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> NDArray[np.float64]:
    """Second-derivative tensor for GJR-GARCH(p,q) log transforms."""
    m = 2 * p + q
    K = 1 + m + (1 if dist in {"studentt", "ged"} else 2 if dist == "skewt" else 0)
    tensor = np.zeros((K, K, K), dtype=np.float64)
    tensor[0, 0, 0] = _softplus_second_derivative_from_positive(theta[0])

    if m > 0:
        tensor[1:1 + m, 1:1 + m, 1:1 + m] = _softmax_second_derivatives(theta[1:1 + m])

    if dist == "studentt":
        tensor[K - 1, K - 1, K - 1] = _softplus_second_derivative(theta[K - 1])
    elif dist == "ged":
        tensor[K - 1, K - 1, K - 1] = _softplus_second_derivative_from_positive(theta[K - 1] - GED_NU_MIN)
    elif dist == "skewt":
        tensor[K - 2, K - 2, K - 2] = _softplus_second_derivative(theta[K - 2])
        tensor[K - 1, K - 1, K - 1] = _tanh_second_derivative(theta[K - 1])

    return tensor


def log_hessian_gjr_garch(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for GJR-GARCH log-mode transforms."""
    if dist == "normal":
        J = jacobian_gjr_garch(theta, p, q)
    elif dist == "studentt":
        J = jacobian_gjr_garch_studentt(theta, p, q)
    elif dist == "ged":
        J = jacobian_gjr_garch_ged(theta, p, q)
    elif dist == "skewt":
        J = jacobian_gjr_garch_skewt(theta, p, q)
    else:
        raise ValueError(f"Unknown distribution: {dist}")
    second = second_derivatives_gjr_garch(theta, p, q, dist)
    return transform_hessian(hess_theta, grad_theta, J, second)


def compute_se_via_logspace(
    theta_hat: NDArray[np.float64],
    nll_theta: callable,
    unpack_fn: callable,
    jacobian_fn: callable,
    pack_fn: callable,
    hess_z_fn: callable | None = None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """
    Compute runtime standard errors via a Hessian in unconstrained (log) space.
    
    This runtime fallback avoids boundary issues by working in unconstrained space:
    1. Transform θ_hat → z_hat (unconstrained space)
    2. Compute Hessian H_z in z-space (analytical when available, otherwise numerical)
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
    hess_z_fn : callable, optional
        Analytical Hessian in z-space: hess_z_fn(z_hat) -> H_z. If omitted,
        a numerical Hessian in z-space is used.
    
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
    
    if hess_z_fn is not None:
        H_z = np.asarray(hess_z_fn(z_hat), dtype=np.float64)
    else:
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
        H_z = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            eps_i = eps[i]
            for j in range(K):
                eps_j = eps[j]

                z_pp[:] = z_hat; z_pp[i] += eps_i; z_pp[j] += eps_j
                z_pm[:] = z_hat; z_pm[i] += eps_i; z_pm[j] -= eps_j
                z_mp[:] = z_hat; z_mp[i] -= eps_i; z_mp[j] += eps_j
                z_mm[:] = z_hat; z_mm[i] -= eps_i; z_mm[j] -= eps_j

                H_z[i, j] = (nll_z(z_pp) - nll_z(z_pm) - nll_z(z_mp) + nll_z(z_mm)) / (4 * eps_i * eps_j)
    H_z = 0.5 * (H_z + H_z.T)
    
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


def pack_garch_ged_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    """
    C-accelerated pack_garch_ged (modifies theta_out in-place).

    theta_out must be pre-allocated with shape (2 + p + q,).
    """
    if p == 1 and q == 1:
        _core._pack_garch_ged_11(_as_cptr(z), _as_cptr(theta_out))
    else:
        _core._pack_garch_ged_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


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


def jacobian_garch_ged_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    """
    C-accelerated jacobian_garch_ged (modifies J_out in-place).

    J_out must be pre-allocated with shape (K, K) where K = 2 + p + q.
    """
    if p == 1 and q == 1:
        _core._jacobian_garch_ged_11(_as_cptr(theta), _as_cptr(J_out))
    else:
        _core._jacobian_garch_ged_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


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
# GJR-GARCH PARAMETER TRANSFORMS
# =============================================================================

def pack_gjr_garch(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """
    Transform unconstrained z to constrained GJR-GARCH parameters θ.
    
    z = [z_omega, z_alpha_1..p, z_gamma_1..p, z_beta_1..q]
    θ = [omega, alpha_1..p, gamma_1..p, beta_1..q]
    
    Uses omega = softplus(z_omega) and a 4-class softmax
    (alpha, gamma, beta, slack=0) ensuring α+γ+β < 1.
    """
    K = 1 + 2 * p + q
    z_omega = z[0]
    z_softmax = z[1:K]
    
    omega = softplus(z_omega)
    
    z_joint = np.concatenate([z_softmax, [0.0]])
    lse_joint = logsumexp(z_joint)
    
    params = np.exp(z_softmax - lse_joint)
    
    return np.concatenate([[omega], params])


def unpack_gjr_garch(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform constrained GJR-GARCH θ to unconstrained z."""
    K = 1 + 2 * p + q
    omega = theta[0]
    params = theta[1:K]
    
    z_omega = softplus_inv(omega)
    
    r = 1.0 - params.sum()
    r = max(r, 1e-10)
    
    z_params = np.log(np.maximum(params, 1e-10)) - np.log(r)
    
    return np.concatenate([[z_omega], z_params])


def jacobian_gjr_garch(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Compute Jacobian J = ∂θ/∂z for GJR-GARCH parameters."""
    K = 1 + 2 * p + q
    omega = theta[0]
    params = theta[1:K]
    
    J = np.zeros((K, K), dtype=np.float64)
    J[0, 0] = _softplus_derivative_from_positive(omega)
    
    # Softmax Jacobian for [alpha, gamma, beta]
    for i in range(K - 1):
        for j in range(K - 1):
            delta = 1.0 if i == j else 0.0
            J[1 + i, 1 + j] = params[i] * (delta - params[j])
    
    return J


def pack_gjr_garch_studentt(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for GJR-GARCH + Student-t."""
    n_gjr = 1 + 2 * p + q
    theta_gjr = pack_gjr_garch(z[:n_gjr], p, q)
    nu = pack_studentt(z[n_gjr])
    return np.concatenate([theta_gjr, [nu]])


def unpack_gjr_garch_studentt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for GJR-GARCH + Student-t."""
    n_gjr = 1 + 2 * p + q
    z_gjr = unpack_gjr_garch(theta[:n_gjr], p, q)
    z_nu = unpack_studentt(theta[n_gjr])
    return np.concatenate([z_gjr, [z_nu]])


def jacobian_gjr_garch_studentt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Compute full Jacobian for GJR-GARCH + Student-t."""
    n_gjr = 1 + 2 * p + q
    K = n_gjr + 1
    J = np.zeros((K, K), dtype=np.float64)
    J[:n_gjr, :n_gjr] = jacobian_gjr_garch(theta[:n_gjr], p, q)
    J[n_gjr, n_gjr] = jacobian_studentt(theta[n_gjr])
    return J


def pack_gjr_garch_ged(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained theta for GJR-GARCH + GED."""
    n_gjr = 1 + 2 * p + q
    theta_gjr = pack_gjr_garch(z[:n_gjr], p, q)
    nu = pack_ged(z[n_gjr])
    return np.concatenate([theta_gjr, [nu]])


def unpack_gjr_garch_ged(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform constrained theta to unconstrained z for GJR-GARCH + GED."""
    n_gjr = 1 + 2 * p + q
    z_gjr = unpack_gjr_garch(theta[:n_gjr], p, q)
    z_nu = unpack_ged(theta[n_gjr])
    return np.concatenate([z_gjr, [z_nu]])


def jacobian_gjr_garch_ged(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Compute full Jacobian for GJR-GARCH + GED."""
    n_gjr = 1 + 2 * p + q
    K = n_gjr + 1
    J = np.zeros((K, K), dtype=np.float64)
    J[:n_gjr, :n_gjr] = jacobian_gjr_garch(theta[:n_gjr], p, q)
    J[n_gjr, n_gjr] = jacobian_ged(theta[n_gjr])
    return J


def pack_gjr_garch_skewt(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for GJR-GARCH + SkewT."""
    n_gjr = 1 + 2 * p + q
    theta_gjr = pack_gjr_garch(z[:n_gjr], p, q)
    nu, lam = pack_skewt(z[n_gjr], z[n_gjr + 1])
    return np.concatenate([theta_gjr, [nu, lam]])


def unpack_gjr_garch_skewt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for GJR-GARCH + SkewT."""
    n_gjr = 1 + 2 * p + q
    z_gjr = unpack_gjr_garch(theta[:n_gjr], p, q)
    z_nu, z_lam = unpack_skewt(theta[n_gjr], theta[n_gjr + 1])
    return np.concatenate([z_gjr, [z_nu, z_lam]])


def jacobian_gjr_garch_skewt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Compute full Jacobian for GJR-GARCH + SkewT."""
    n_gjr = 1 + 2 * p + q
    K = n_gjr + 2
    J = np.zeros((K, K), dtype=np.float64)
    J[:n_gjr, :n_gjr] = jacobian_gjr_garch(theta[:n_gjr], p, q)
    J_dist = jacobian_skewt(theta[n_gjr], theta[n_gjr + 1])
    J[n_gjr:, n_gjr:] = J_dist
    return J


# C-accelerated GJR-GARCH versions

def pack_gjr_garch_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    """C-accelerated pack_gjr_garch (modifies theta_out in-place)."""
    if p == 1 and q == 1:
        _core._pack_gjr_garch_11(_as_cptr(z), _as_cptr(theta_out))
    else:
        _core._pack_gjr_garch_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


def pack_gjr_garch_studentt_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    """C-accelerated pack for GJR-GARCH + Student-t."""
    if p == 1 and q == 1:
        _core._pack_gjr_garch_studentt_11(_as_cptr(z), _as_cptr(theta_out))
    else:
        _core._pack_gjr_garch_studentt_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


def pack_gjr_garch_skewt_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    """C-accelerated pack for GJR-GARCH + SkewT."""
    if p == 1 and q == 1:
        _core._pack_gjr_garch_skewt_11(_as_cptr(z), _as_cptr(theta_out))
    else:
        _core._pack_gjr_garch_skewt_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


def jacobian_gjr_garch_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    """C-accelerated Jacobian for GJR-GARCH."""
    if p == 1 and q == 1:
        _core._jacobian_gjr_garch_11(_as_cptr(theta), _as_cptr(J_out))
    else:
        _core._jacobian_gjr_garch_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


def jacobian_gjr_garch_studentt_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    """C-accelerated Jacobian for GJR-GARCH + Student-t."""
    if p == 1 and q == 1:
        _core._jacobian_gjr_garch_studentt_11(_as_cptr(theta), _as_cptr(J_out))
    else:
        _core._jacobian_gjr_garch_studentt_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


def jacobian_gjr_garch_skewt_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    """C-accelerated Jacobian for GJR-GARCH + SkewT."""
    if p == 1 and q == 1:
        _core._jacobian_gjr_garch_skewt_11(_as_cptr(theta), _as_cptr(J_out))
    else:
        _core._jacobian_gjr_garch_skewt_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


def transform_grad_gjr_c(
    grad_theta: NDArray[np.float64],
    J: NDArray[np.float64],
    grad_z_out: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal"
) -> None:
    """C-accelerated gradient transform for GJR-GARCH: grad_z = J^T @ grad_theta."""
    K = grad_theta.shape[0]
    
    if p == 1 and q == 1:
        if dist == "normal":
            _core._transform_grad_gjr_11_normal(_as_cptr(grad_theta), _as_cptr(J), _as_cptr(grad_z_out))
        elif dist == "studentt":
            _core._transform_grad_gjr_11_studentt(_as_cptr(grad_theta), _as_cptr(J), _as_cptr(grad_z_out))
        elif dist == "skewt":
            _core._transform_grad_gjr_11_skewt(_as_cptr(grad_theta), _as_cptr(J), _as_cptr(grad_z_out))
        else:
            raise ValueError(f"Unknown distribution: {dist}")
    else:
        _core._transform_grad_pq(_as_cptr(grad_theta), _as_cptr(J), _as_cptr(grad_z_out), K)


# =============================================================================
# EGARCH(1,1) PARAMETER TRANSFORMS
# =============================================================================

def pack_egarch(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained theta for EGARCH(p,q)."""
    z = np.asarray(z, dtype=np.float64)
    K = 1 + 2 * p + q
    theta = np.empty(K, dtype=np.float64)
    theta[0] = z[0]
    theta[1 : 1 + p] = z[1 : 1 + p]
    theta[1 + p : 1 + 2 * p] = z[1 + p : 1 + 2 * p]
    theta[1 + 2 * p :] = np.tanh(z[1 + 2 * p : K])
    return theta


def unpack_egarch(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Transform constrained theta to unconstrained z for EGARCH(p,q)."""
    theta = np.asarray(theta, dtype=np.float64)
    K = 1 + 2 * p + q
    z = np.empty(K, dtype=np.float64)
    z[0] = theta[0]
    z[1 : 1 + p] = theta[1 : 1 + p]
    z[1 + p : 1 + 2 * p] = theta[1 + p : 1 + 2 * p]
    z[1 + 2 * p :] = np.arctanh(np.clip(theta[1 + 2 * p : K], -0.999999, 0.999999))
    return z


def jacobian_egarch(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """Compute the Jacobian for EGARCH(p,q)."""
    theta = np.asarray(theta, dtype=np.float64)
    K = 1 + 2 * p + q
    J = np.zeros((K, K), dtype=np.float64)
    J[0, 0] = 1.0
    for i in range(p):
        J[1 + i, 1 + i] = 1.0
        J[1 + p + i, 1 + p + i] = 1.0
    beta_base = 1 + 2 * p
    for j in range(q):
        idx = beta_base + j
        J[idx, idx] = 1.0 - float(theta[idx]) * float(theta[idx])
    return J


def pack_egarch_studentt(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    K_vol = 1 + 2 * p + q
    theta_egarch = pack_egarch(z[:K_vol], p, q)
    nu = pack_studentt(z[K_vol])
    return np.concatenate([theta_egarch, [nu]])


def unpack_egarch_studentt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    K_vol = 1 + 2 * p + q
    z_egarch = unpack_egarch(theta[:K_vol], p, q)
    return np.concatenate([z_egarch, [unpack_studentt(theta[K_vol])]])


def jacobian_egarch_studentt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    K_vol = 1 + 2 * p + q
    J = np.zeros((K_vol + 1, K_vol + 1), dtype=np.float64)
    J[:K_vol, :K_vol] = jacobian_egarch(theta[:K_vol], p, q)
    J[K_vol, K_vol] = jacobian_studentt(theta[K_vol])
    return J


def pack_egarch_ged(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    K_vol = 1 + 2 * p + q
    theta_egarch = pack_egarch(z[:K_vol], p, q)
    nu = pack_ged(z[K_vol])
    return np.concatenate([theta_egarch, [nu]])


def unpack_egarch_ged(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    K_vol = 1 + 2 * p + q
    z_egarch = unpack_egarch(theta[:K_vol], p, q)
    return np.concatenate([z_egarch, [unpack_ged(theta[K_vol])]])


def jacobian_egarch_ged(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    K_vol = 1 + 2 * p + q
    J = np.zeros((K_vol + 1, K_vol + 1), dtype=np.float64)
    J[:K_vol, :K_vol] = jacobian_egarch(theta[:K_vol], p, q)
    J[K_vol, K_vol] = jacobian_ged(theta[K_vol])
    return J


def pack_egarch_skewt(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    K_vol = 1 + 2 * p + q
    theta_egarch = pack_egarch(z[:K_vol], p, q)
    nu, lam = pack_skewt(z[K_vol], z[K_vol + 1])
    return np.concatenate([theta_egarch, [nu, lam]])


def unpack_egarch_skewt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    K_vol = 1 + 2 * p + q
    z_egarch = unpack_egarch(theta[:K_vol], p, q)
    z_nu, z_lam = unpack_skewt(theta[K_vol], theta[K_vol + 1])
    return np.concatenate([z_egarch, [z_nu, z_lam]])


def jacobian_egarch_skewt(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    K_vol = 1 + 2 * p + q
    J = np.zeros((K_vol + 2, K_vol + 2), dtype=np.float64)
    J[:K_vol, :K_vol] = jacobian_egarch(theta[:K_vol], p, q)
    J[K_vol:, K_vol:] = jacobian_skewt(theta[K_vol], theta[K_vol + 1])
    return J


def second_derivatives_egarch(
    theta: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> NDArray[np.float64]:
    """Second-derivative tensor for EGARCH log transforms."""
    K = 1 + 2 * p + q + (1 if dist in {"studentt", "ged"} else 2 if dist == "skewt" else 0)
    tensor = np.zeros((K, K, K), dtype=np.float64)
    beta_base = 1 + 2 * p
    for j in range(q):
        idx = beta_base + j
        tensor[idx, idx, idx] = _tanh_second_derivative(theta[idx])
    if dist == "studentt":
        tensor[K - 1, K - 1, K - 1] = _softplus_second_derivative(theta[K - 1])
    elif dist == "ged":
        tensor[K - 1, K - 1, K - 1] = _softplus_second_derivative_from_positive(theta[K - 1] - GED_NU_MIN)
    elif dist == "skewt":
        tensor[K - 2, K - 2, K - 2] = _softplus_second_derivative(theta[K - 2])
        tensor[K - 1, K - 1, K - 1] = _tanh_second_derivative(theta[K - 1])
    return tensor


def log_hessian_egarch(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for EGARCH transforms."""
    if dist == "normal":
        J = jacobian_egarch(theta, p, q)
    elif dist == "studentt":
        J = jacobian_egarch_studentt(theta, p, q)
    elif dist == "ged":
        J = jacobian_egarch_ged(theta, p, q)
    elif dist == "skewt":
        J = jacobian_egarch_skewt(theta, p, q)
    else:
        raise ValueError(f"Unknown distribution: {dist}")
    second = second_derivatives_egarch(theta, p, q, dist)
    return transform_hessian(hess_theta, grad_theta, J, second)


def pack_egarch_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    if (p, q) == (1, 1):
        _core._pack_egarch_11(_as_cptr(z), _as_cptr(theta_out))
        return
    _core._pack_egarch_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


def pack_egarch_studentt_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    if (p, q) == (1, 1):
        _core._pack_egarch_studentt_11(_as_cptr(z), _as_cptr(theta_out))
        return
    _core._pack_egarch_studentt_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


def pack_egarch_ged_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    if (p, q) == (1, 1):
        _core._pack_egarch_ged_11(_as_cptr(z), _as_cptr(theta_out))
        return
    _core._pack_egarch_ged_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


def pack_egarch_skewt_c(z: NDArray[np.float64], theta_out: NDArray[np.float64], p: int, q: int) -> None:
    if (p, q) == (1, 1):
        _core._pack_egarch_skewt_11(_as_cptr(z), _as_cptr(theta_out))
        return
    _core._pack_egarch_skewt_pq(_as_cptr(z), _as_cptr(theta_out), p, q)


def jacobian_egarch_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    if (p, q) == (1, 1):
        _core._jacobian_egarch_11(_as_cptr(theta), _as_cptr(J_out))
        return
    _core._jacobian_egarch_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


def jacobian_egarch_studentt_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    if (p, q) == (1, 1):
        _core._jacobian_egarch_studentt_11(_as_cptr(theta), _as_cptr(J_out))
        return
    _core._jacobian_egarch_studentt_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


def jacobian_egarch_ged_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    if (p, q) == (1, 1):
        _core._jacobian_egarch_ged_11(_as_cptr(theta), _as_cptr(J_out))
        return
    _core._jacobian_egarch_ged_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


def jacobian_egarch_skewt_c(theta: NDArray[np.float64], J_out: NDArray[np.float64], p: int, q: int) -> None:
    if (p, q) == (1, 1):
        _core._jacobian_egarch_skewt_11(_as_cptr(theta), _as_cptr(J_out))
        return
    _core._jacobian_egarch_skewt_pq(_as_cptr(theta), _as_cptr(J_out), p, q)


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
        omega = softplus(z_omega)  ensures omega > 0
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


def pack_arma_garch_ged(
    z: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for ARMA+GARCH+GED."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    theta_base = pack_arma_garch_normal(z[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_garch)
    nu = pack_ged(z[n_mean + n_vol])
    return np.concatenate([theta_base, [nu]])


def unpack_arma_garch_ged(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for ARMA+GARCH+GED."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    z_base = unpack_arma_garch_normal(theta[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_garch)
    z_nu = unpack_ged(theta[n_mean + n_vol])
    return np.concatenate([z_base, [z_nu]])


def jacobian_arma_garch_ged(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Compute Jacobian for ARMA+GARCH+GED."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    K = n_mean + n_vol + 1
    J = np.zeros((K, K), dtype=np.float64)
    J[:n_mean + n_vol, :n_mean + n_vol] = jacobian_arma_garch_normal(
        theta[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_garch
    )
    J[K - 1, K - 1] = jacobian_ged(theta[n_mean + n_vol])
    return J


def pack_arma_egarch_normal(
    z: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for ARMA+EGARCH+Normal."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch

    theta = np.empty(n_mean + n_vol, dtype=np.float64)
    theta[0] = z[0]
    if p_ar > 0:
        theta[1:1 + p_ar] = 0.99 * np.tanh(z[1:1 + p_ar])
    if q_ma > 0:
        theta[1 + p_ar:n_mean] = 0.99 * np.tanh(z[1 + p_ar:n_mean])

    z_egarch = z[n_mean:n_mean + n_vol]
    theta[n_mean:n_mean + n_vol] = pack_egarch(z_egarch, P_arch, Q_egarch)
    return theta


def unpack_arma_egarch_normal(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for ARMA+EGARCH+Normal."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch

    z = np.empty(n_mean + n_vol, dtype=np.float64)
    z[0] = theta[0]
    if p_ar > 0:
        z[1:1 + p_ar] = np.arctanh(np.clip(theta[1:1 + p_ar] / 0.99, -0.999, 0.999))
    if q_ma > 0:
        z[1 + p_ar:n_mean] = np.arctanh(np.clip(theta[1 + p_ar:n_mean] / 0.99, -0.999, 0.999))

    z[n_mean:n_mean + n_vol] = unpack_egarch(theta[n_mean:n_mean + n_vol], P_arch, Q_egarch)
    return z


def jacobian_arma_egarch_normal(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Compute Jacobian J = ∂θ/∂z for ARMA+EGARCH+Normal."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch
    K = n_mean + n_vol

    J = np.zeros((K, K), dtype=np.float64)
    J[0, 0] = 1.0
    for idx in range(1, n_mean):
        ratio = theta[idx] / 0.99
        J[idx, idx] = 0.99 * (1.0 - ratio * ratio)
    J[n_mean:, n_mean:] = jacobian_egarch(theta[n_mean:n_mean + n_vol], P_arch, Q_egarch)
    return J


def pack_arma_egarch_studentt(
    z: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for ARMA+EGARCH+StudentT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch
    theta_base = pack_arma_egarch_normal(z[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_egarch)
    nu = pack_studentt(z[n_mean + n_vol])
    return np.concatenate([theta_base, [nu]])


def unpack_arma_egarch_studentt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for ARMA+EGARCH+StudentT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch
    z_base = unpack_arma_egarch_normal(theta[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_egarch)
    z_nu = unpack_studentt(theta[n_mean + n_vol])
    return np.concatenate([z_base, [z_nu]])


def jacobian_arma_egarch_studentt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Compute Jacobian for ARMA+EGARCH+StudentT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch
    K = n_mean + n_vol + 1

    J = np.zeros((K, K), dtype=np.float64)
    J[:n_mean + n_vol, :n_mean + n_vol] = jacobian_arma_egarch_normal(
        theta[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_egarch
    )
    J[K - 1, K - 1] = jacobian_studentt(theta[n_mean + n_vol])
    return J


def second_derivatives_arma_egarch_ged(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA(p,q)+EGARCH(P,Q)+GED log transforms."""
    n_mean = 1 + p_ar + q_ma
    K = n_mean + 1 + 2 * P_arch + Q_egarch + 1
    tensor = np.zeros((K, K, K), dtype=np.float64)

    for idx in range(1, n_mean):
        tensor[idx, idx, idx] = _tanh_scaled_second_derivative(theta[idx])

    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_egarch(
        theta[n_mean:], P_arch, Q_egarch, dist="ged"
    )
    return tensor


def log_hessian_arma_egarch_ged(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA(p,q)+EGARCH(P,Q)+GED."""
    J = jacobian_arma_egarch_ged(theta, p_ar, q_ma, P_arch, Q_egarch)
    second = second_derivatives_arma_egarch_ged(theta, p_ar, q_ma, P_arch, Q_egarch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def pack_arma_egarch_ged(
    z: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for ARMA+EGARCH+GED."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch
    theta_base = pack_arma_egarch_normal(z[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_egarch)
    nu = pack_ged(z[n_mean + n_vol])
    return np.concatenate([theta_base, [nu]])


def unpack_arma_egarch_ged(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for ARMA+EGARCH+GED."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch
    z_base = unpack_arma_egarch_normal(theta[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_egarch)
    z_nu = unpack_ged(theta[n_mean + n_vol])
    return np.concatenate([z_base, [z_nu]])


def jacobian_arma_egarch_ged(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Compute Jacobian for ARMA+EGARCH+GED."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch
    K = n_mean + n_vol + 1

    J = np.zeros((K, K), dtype=np.float64)
    J[:n_mean + n_vol, :n_mean + n_vol] = jacobian_arma_egarch_normal(
        theta[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_egarch
    )
    J[K - 1, K - 1] = jacobian_ged(theta[n_mean + n_vol])
    return J


def second_derivatives_arma_egarch_skewt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA(p,q)+EGARCH(P,Q)+SkewT log transforms."""
    n_mean = 1 + p_ar + q_ma
    K = n_mean + 1 + 2 * P_arch + Q_egarch + 2
    tensor = np.zeros((K, K, K), dtype=np.float64)

    for idx in range(1, n_mean):
        tensor[idx, idx, idx] = _tanh_scaled_second_derivative(theta[idx])

    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_egarch(
        theta[n_mean:], P_arch, Q_egarch, dist="skewt"
    )
    return tensor


def log_hessian_arma_egarch_skewt(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA(p,q)+EGARCH(P,Q)+SkewT."""
    J = jacobian_arma_egarch_skewt(theta, p_ar, q_ma, P_arch, Q_egarch)
    second = second_derivatives_arma_egarch_skewt(theta, p_ar, q_ma, P_arch, Q_egarch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def pack_arma_egarch_skewt(
    z: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained θ for ARMA+EGARCH+SkewT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch
    theta_base = pack_arma_egarch_normal(z[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_egarch)
    nu, lam = pack_skewt(z[n_mean + n_vol], z[n_mean + n_vol + 1])
    return np.concatenate([theta_base, [nu, lam]])


def unpack_arma_egarch_skewt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Transform constrained θ to unconstrained z for ARMA+EGARCH+SkewT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch
    z_base = unpack_arma_egarch_normal(theta[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_egarch)
    z_nu, z_lam = unpack_skewt(theta[n_mean + n_vol], theta[n_mean + n_vol + 1])
    return np.concatenate([z_base, [z_nu, z_lam]])


def jacobian_arma_egarch_skewt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_egarch: int,
) -> NDArray[np.float64]:
    """Compute Jacobian for ARMA+EGARCH+SkewT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_egarch
    K = n_mean + n_vol + 2

    J = np.zeros((K, K), dtype=np.float64)
    J[:n_mean + n_vol, :n_mean + n_vol] = jacobian_arma_egarch_normal(
        theta[:n_mean + n_vol], p_ar, q_ma, P_arch, Q_egarch
    )
    J[n_mean + n_vol:, n_mean + n_vol:] = jacobian_skewt(
        theta[n_mean + n_vol], theta[n_mean + n_vol + 1]
    )
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


# =============================================================================
# ARMA-GJR-GARCH PARAMETER TRANSFORMS
# =============================================================================

def pack_arma_gjr_garch_normal(
    z: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained theta for ARMA+GJR-GARCH+Normal."""
    n_mean = 1 + p_ar + q_ma
    theta_mean = pack_arma_normal(z[:n_mean], p_ar, q_ma)
    theta_vol = pack_gjr_garch(z[n_mean:], P_arch, Q_garch)
    return np.concatenate([theta_mean, theta_vol])


def unpack_arma_gjr_garch_normal(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Transform constrained theta to unconstrained z for ARMA+GJR-GARCH+Normal."""
    n_mean = 1 + p_ar + q_ma
    z_mean = unpack_arma_normal(theta[:n_mean], p_ar, q_ma)
    z_vol = unpack_gjr_garch(theta[n_mean:], P_arch, Q_garch)
    return np.concatenate([z_mean, z_vol])


def jacobian_arma_gjr_garch_normal(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Compute Jacobian for ARMA+GJR-GARCH+Normal."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_garch
    K = n_mean + n_vol
    J = np.zeros((K, K), dtype=np.float64)
    J[:n_mean, :n_mean] = jacobian_arma_normal(theta[:n_mean], p_ar, q_ma)
    J[n_mean:, n_mean:] = jacobian_gjr_garch(theta[n_mean:], P_arch, Q_garch)
    return J


def pack_arma_gjr_garch_studentt(
    z: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained theta for ARMA+GJR-GARCH+StudentT."""
    n_mean = 1 + p_ar + q_ma
    theta_mean = pack_arma_normal(z[:n_mean], p_ar, q_ma)
    theta_vol = pack_gjr_garch_studentt(z[n_mean:], P_arch, Q_garch)
    return np.concatenate([theta_mean, theta_vol])


def unpack_arma_gjr_garch_studentt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Transform constrained theta to unconstrained z for ARMA+GJR-GARCH+StudentT."""
    n_mean = 1 + p_ar + q_ma
    z_mean = unpack_arma_normal(theta[:n_mean], p_ar, q_ma)
    z_vol = unpack_gjr_garch_studentt(theta[n_mean:], P_arch, Q_garch)
    return np.concatenate([z_mean, z_vol])


def jacobian_arma_gjr_garch_studentt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Compute Jacobian for ARMA+GJR-GARCH+StudentT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_garch + 1
    K = n_mean + n_vol
    J = np.zeros((K, K), dtype=np.float64)
    J[:n_mean, :n_mean] = jacobian_arma_normal(theta[:n_mean], p_ar, q_ma)
    J[n_mean:, n_mean:] = jacobian_gjr_garch_studentt(theta[n_mean:], P_arch, Q_garch)
    return J


def pack_arma_gjr_garch_ged(
    z: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained theta for ARMA+GJR-GARCH+GED."""
    n_mean = 1 + p_ar + q_ma
    theta_mean = pack_arma_normal(z[:n_mean], p_ar, q_ma)
    theta_vol = pack_gjr_garch_ged(z[n_mean:], P_arch, Q_garch)
    return np.concatenate([theta_mean, theta_vol])


def unpack_arma_gjr_garch_ged(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Transform constrained theta to unconstrained z for ARMA+GJR-GARCH+GED."""
    n_mean = 1 + p_ar + q_ma
    z_mean = unpack_arma_normal(theta[:n_mean], p_ar, q_ma)
    z_vol = unpack_gjr_garch_ged(theta[n_mean:], P_arch, Q_garch)
    return np.concatenate([z_mean, z_vol])


def jacobian_arma_gjr_garch_ged(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Compute Jacobian for ARMA+GJR-GARCH+GED."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_garch + 1
    K = n_mean + n_vol
    J = np.zeros((K, K), dtype=np.float64)
    J[:n_mean, :n_mean] = jacobian_arma_normal(theta[:n_mean], p_ar, q_ma)
    J[n_mean:, n_mean:] = jacobian_gjr_garch_ged(theta[n_mean:], P_arch, Q_garch)
    return J


def pack_arma_gjr_garch_skewt(
    z: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Transform unconstrained z to constrained theta for ARMA+GJR-GARCH+SkewT."""
    n_mean = 1 + p_ar + q_ma
    theta_mean = pack_arma_normal(z[:n_mean], p_ar, q_ma)
    theta_vol = pack_gjr_garch_skewt(z[n_mean:], P_arch, Q_garch)
    return np.concatenate([theta_mean, theta_vol])


def unpack_arma_gjr_garch_skewt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Transform constrained theta to unconstrained z for ARMA+GJR-GARCH+SkewT."""
    n_mean = 1 + p_ar + q_ma
    z_mean = unpack_arma_normal(theta[:n_mean], p_ar, q_ma)
    z_vol = unpack_gjr_garch_skewt(theta[n_mean:], P_arch, Q_garch)
    return np.concatenate([z_mean, z_vol])


def jacobian_arma_gjr_garch_skewt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Compute Jacobian for ARMA+GJR-GARCH+SkewT."""
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_garch + 2
    K = n_mean + n_vol
    J = np.zeros((K, K), dtype=np.float64)
    J[:n_mean, :n_mean] = jacobian_arma_normal(theta[:n_mean], p_ar, q_ma)
    J[n_mean:, n_mean:] = jacobian_gjr_garch_skewt(theta[n_mean:], P_arch, Q_garch)
    return J


def second_derivatives_arma_gjr_garch_normal(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA+GJR-GARCH+Normal log transforms."""
    n_mean = 1 + p_ar + q_ma
    K = n_mean + 1 + 2 * P_arch + Q_garch
    tensor = np.zeros((K, K, K), dtype=np.float64)
    tensor[:n_mean, :n_mean, :n_mean] = second_derivatives_arma_normal(theta[:n_mean], p_ar, q_ma)
    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_gjr_garch(
        theta[n_mean:], P_arch, Q_garch, dist="normal"
    )
    return tensor


def log_hessian_arma_gjr_garch_normal(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA+GJR-GARCH+Normal."""
    J = jacobian_arma_gjr_garch_normal(theta, p_ar, q_ma, P_arch, Q_garch)
    second = second_derivatives_arma_gjr_garch_normal(theta, p_ar, q_ma, P_arch, Q_garch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_arma_gjr_garch_studentt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA+GJR-GARCH+StudentT log transforms."""
    n_mean = 1 + p_ar + q_ma
    K = n_mean + 1 + 2 * P_arch + Q_garch + 1
    tensor = np.zeros((K, K, K), dtype=np.float64)
    tensor[:n_mean, :n_mean, :n_mean] = second_derivatives_arma_normal(theta[:n_mean], p_ar, q_ma)
    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_gjr_garch(
        theta[n_mean:], P_arch, Q_garch, dist="studentt"
    )
    return tensor


def log_hessian_arma_gjr_garch_studentt(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA+GJR-GARCH+StudentT."""
    J = jacobian_arma_gjr_garch_studentt(theta, p_ar, q_ma, P_arch, Q_garch)
    second = second_derivatives_arma_gjr_garch_studentt(theta, p_ar, q_ma, P_arch, Q_garch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_arma_gjr_garch_ged(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA+GJR-GARCH+GED log transforms."""
    n_mean = 1 + p_ar + q_ma
    K = n_mean + 1 + 2 * P_arch + Q_garch + 1
    tensor = np.zeros((K, K, K), dtype=np.float64)
    tensor[:n_mean, :n_mean, :n_mean] = second_derivatives_arma_normal(theta[:n_mean], p_ar, q_ma)
    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_gjr_garch(
        theta[n_mean:], P_arch, Q_garch, dist="ged"
    )
    return tensor


def log_hessian_arma_gjr_garch_ged(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA+GJR-GARCH+GED."""
    J = jacobian_arma_gjr_garch_ged(theta, p_ar, q_ma, P_arch, Q_garch)
    second = second_derivatives_arma_gjr_garch_ged(theta, p_ar, q_ma, P_arch, Q_garch)
    return transform_hessian(hess_theta, grad_theta, J, second)


def second_derivatives_arma_gjr_garch_skewt(
    theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Second-derivative tensor for ARMA+GJR-GARCH+SkewT log transforms."""
    n_mean = 1 + p_ar + q_ma
    K = n_mean + 1 + 2 * P_arch + Q_garch + 2
    tensor = np.zeros((K, K, K), dtype=np.float64)
    tensor[:n_mean, :n_mean, :n_mean] = second_derivatives_arma_normal(theta[:n_mean], p_ar, q_ma)
    tensor[n_mean:, n_mean:, n_mean:] = second_derivatives_gjr_garch(
        theta[n_mean:], P_arch, Q_garch, dist="skewt"
    )
    return tensor


def log_hessian_arma_gjr_garch_skewt(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
) -> NDArray[np.float64]:
    """Analytical z-space Hessian for ARMA+GJR-GARCH+SkewT."""
    J = jacobian_arma_gjr_garch_skewt(theta, p_ar, q_ma, P_arch, Q_garch)
    second = second_derivatives_arma_gjr_garch_skewt(theta, p_ar, q_ma, P_arch, Q_garch)
    return transform_hessian(hess_theta, grad_theta, J, second)
