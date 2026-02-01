"""
Likelihood Functions for GARCH(1,1) Estimation
===============================================

This module provides:
- Log-likelihood functions
- Analytical gradients
- Analytical Hessians

For GARCH(1,1) models with:
- Normal innovations
- Student-t innovations
- Skewed Student-t innovations

All functions follow the convention of returning the NEGATIVE log-likelihood
and its derivatives (for minimization).

References:
    Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
    Journal of Econometrics, 31, 307-327.
    
    Fiorentini, G., Calzolari, G., & Panattoni, L. (1996). Analytic Derivatives and 
    the Computation of GARCH Estimates. Journal of Applied Econometrics, 11(4), 399-417.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, psi, polygamma

# Optional Numba JIT compilation for performance
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else args[0]

# =============================================================================
# CONSTANTS
# =============================================================================

VARIANCE_FLOOR = 1e-12  # Minimum variance to prevent division by zero


# =============================================================================
# SIMPLE LOG-LIKELIHOOD FUNCTIONS (for use with fit_garch_mle)
# =============================================================================
# These have signature: loglik(resid, sigma2) -> float
# They return the LOG-LIKELIHOOD (not negative), as fit_garch_mle negates internally.

def normal_loglik(resid: NDArray[np.float64], sigma2: NDArray[np.float64]) -> float:
    """
    Normal log-likelihood given residuals and conditional variances.
    
    ℓ = -0.5 · Σ[log(2π) + log(σ²_t) + ε²_t/σ²_t]
      = -0.5 · n·log(2π) - 0.5 · Σ[log(σ²_t) + ε²_t/σ²_t]
    
    Parameters
    ----------
    resid : array
        Residuals ε_t
    sigma2 : array
        Conditional variances σ²_t
    
    Returns
    -------
    float
        Log-likelihood value
    """
    n = len(resid)
    return -0.5 * (n * np.log(2 * np.pi) + np.sum(np.log(sigma2) + resid**2 / sigma2))


def studentt_loglik(
    resid: NDArray[np.float64], 
    sigma2: NDArray[np.float64], 
    nu: float = 8.0
) -> float:
    """
    Student-t log-likelihood given residuals and conditional variances.
    
    ℓ = n·[Γ((ν+1)/2) - Γ(ν/2) - 0.5·log(π(ν-2))]
        - 0.5·Σ[log(σ²_t) + (ν+1)·log(1 + ε²_t/(σ²_t·(ν-2)))]
    
    Parameters
    ----------
    resid : array
        Residuals ε_t
    sigma2 : array
        Conditional variances σ²_t
    nu : float
        Degrees of freedom (must be > 2)
    
    Returns
    -------
    float
        Log-likelihood value
    """
    n = len(resid)
    z2 = resid**2 / sigma2
    const = gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2))
    return n * const - 0.5 * np.sum(np.log(sigma2) + (nu + 1) * np.log(1 + z2 / (nu - 2)))


def make_studentt_loglik(nu: float) -> Callable[[NDArray[np.float64], NDArray[np.float64]], float]:
    """
    Factory function to create a Student-t log-likelihood with fixed nu.
    
    Usage:
        loglik_fn = make_studentt_loglik(nu=6.0)
        result = fit_garch_mle(resid, loglik_fn=loglik_fn)
    """
    def loglik(resid: NDArray[np.float64], sigma2: NDArray[np.float64]) -> float:
        return studentt_loglik(resid, sigma2, nu)
    return loglik


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class LikelihoodResult:
    """Container for likelihood evaluation results."""
    nll: float                              # Negative log-likelihood
    sigma2: NDArray[np.float64]             # Conditional variances


@dataclass 
class GradientResult:
    """Container for gradient evaluation results."""
    nll: float                              # Negative log-likelihood
    grad: NDArray[np.float64]               # Gradient vector
    sigma2: NDArray[np.float64]             # Conditional variances


@dataclass
class HessianResult:
    """Container for Hessian evaluation results."""
    nll: float                              # Negative log-likelihood
    grad: NDArray[np.float64]               # Gradient vector
    hess: NDArray[np.float64]               # Hessian matrix
    sigma2: NDArray[np.float64]             # Conditional variances


@dataclass
class RobustSEResult:
    """Container for robust standard error computation (QMLE)."""
    nll: float                              # Negative log-likelihood
    grad: NDArray[np.float64]               # Gradient vector
    hess: NDArray[np.float64]               # Hessian matrix
    opg: NDArray[np.float64]                # Outer Product of Gradients
    sigma2: NDArray[np.float64]             # Conditional variances
    cov_mle: NDArray[np.float64]            # MLE covariance (H^{-1})
    cov_robust: NDArray[np.float64]         # Robust/sandwich covariance (H^{-1} OPG H^{-1})
    se_mle: NDArray[np.float64]             # MLE standard errors
    se_robust: NDArray[np.float64]          # Robust standard errors


# =============================================================================
# GARCH(1,1) VARIANCE RECURSION
# =============================================================================

@njit(cache=True)
def _garch11_variance_core(
    omega: float,
    alpha: float,
    beta: float,
    resid2: NDArray[np.float64],
    sigma2_init: float,
    variance_floor: float,
) -> NDArray[np.float64]:
    """Numba-accelerated GARCH(1,1) variance recursion."""
    n = len(resid2)
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = max(sigma2_init, variance_floor)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1]
        if sigma2[t] < variance_floor:
            sigma2[t] = variance_floor
    
    return sigma2


def garch11_variance(
    omega: float,
    alpha: float,
    beta: float,
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> NDArray[np.float64]:
    """
    Compute GARCH(1,1) conditional variances.
    
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    
    Uses Numba JIT compilation if available.
    
    Parameters
    ----------
    omega : float
        Intercept
    alpha : float
        ARCH coefficient
    beta : float
        GARCH coefficient
    resid2 : array
        Squared residuals
    sigma2_init : float
        Initial variance (typically sample variance)
    
    Returns
    -------
    sigma2 : array
        Conditional variances
    """
    return _garch11_variance_core(omega, alpha, beta, resid2, sigma2_init, VARIANCE_FLOOR)


# =============================================================================
# NORMAL DISTRIBUTION
# =============================================================================

def normal_nll(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
) -> float:
    """
    Normal negative log-likelihood (excluding constant).
    
    -ℓ = 0.5 * Σ[log(σ²_t) + ε²_t/σ²_t]
    
    Note: The constant -n/2 * log(2π) is omitted as it doesn't affect optimization.
    """
    return 0.5 * np.sum(np.log(sigma2) + resid**2 / sigma2)


def normal_loglik(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
) -> float:
    """
    Normal log-likelihood (for reporting, includes constant).
    
    ℓ = -n/2 * log(2π) - 0.5 * Σ[log(σ²_t) + ε²_t/σ²_t]
    """
    n = len(resid)
    const = -0.5 * n * np.log(2 * np.pi)
    return const - 0.5 * np.sum(np.log(sigma2) + resid**2 / sigma2)


# =============================================================================
# GARCH(1,1) + NORMAL: LIKELIHOOD, GRADIENT, HESSIAN
# =============================================================================

def garch11_normal_nll(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> LikelihoodResult:
    """
    GARCH(1,1) + Normal negative log-likelihood.
    
    Parameters
    ----------
    params : array [omega, alpha, beta]
    resid2 : array of squared residuals
    sigma2_init : float, initial variance
    
    Returns
    -------
    LikelihoodResult with nll and sigma2
    """
    omega, alpha, beta = params[0], params[1], params[2]
    
    n = len(resid2)
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = max(sigma2_init, VARIANCE_FLOOR)
    
    nll = np.log(sigma2[0]) + resid2[0] / sigma2[0]
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], VARIANCE_FLOOR)
        nll += np.log(sigma2[t]) + resid2[t] / sigma2[t]
    
    return LikelihoodResult(nll=0.5 * nll, sigma2=sigma2)


def garch11_normal_gradient(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> GradientResult:
    """
    GARCH(1,1) + Normal: negative log-likelihood and analytical gradient.
    
    Gradient is computed using the recursive formula:
        ∂σ²_t/∂ω = 1 + β·∂σ²_{t-1}/∂ω
        ∂σ²_t/∂α = ε²_{t-1} + β·∂σ²_{t-1}/∂α
        ∂σ²_t/∂β = σ²_{t-1} + β·∂σ²_{t-1}/∂β
    
    Parameters
    ----------
    params : array [omega, alpha, beta]
    resid2 : array of squared residuals
    sigma2_init : float, initial variance
    
    Returns
    -------
    GradientResult with nll, grad, sigma2
    """
    omega, alpha, beta = params[0], params[1], params[2]
    
    n = len(resid2)
    sigma2 = np.empty(n, dtype=np.float64)
    grad = np.zeros(3, dtype=np.float64)
    
    # Derivatives of σ²_t w.r.t. parameters
    d_prev = np.zeros(3, dtype=np.float64)  # [∂σ²/∂ω, ∂σ²/∂α, ∂σ²/∂β]
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 0: Initial observation (σ²_0 is fixed, so derivatives are zero)
    # ─────────────────────────────────────────────────────────────────────────
    sigma2[0] = max(sigma2_init, VARIANCE_FLOOR)
    
    inv_s2 = 1.0 / sigma2[0]
    res_os = resid2[0] * inv_s2
    c_grad = 0.5 * (1.0 - res_os) * inv_s2  # ∂(-ℓ_t)/∂σ²_t
    
    nll = np.log(sigma2[0]) + res_os
    
    grad[0] += c_grad * d_prev[0]
    grad[1] += c_grad * d_prev[1]
    grad[2] += c_grad * d_prev[2]
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 1 to n-1
    # ─────────────────────────────────────────────────────────────────────────
    for t in range(1, n):
        # 1. Variance recursion
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], VARIANCE_FLOOR)
        
        # 2. Derivative recursion
        d_curr = np.empty(3, dtype=np.float64)
        d_curr[0] = 1.0 + beta * d_prev[0]           # ∂σ²_t/∂ω
        d_curr[1] = resid2[t-1] + beta * d_prev[1]   # ∂σ²_t/∂α
        d_curr[2] = sigma2[t-1] + beta * d_prev[2]   # ∂σ²_t/∂β
        
        # 3. Gradient contribution
        inv_s2 = 1.0 / sigma2[t]
        res_os = resid2[t] * inv_s2
        c_grad = 0.5 * (1.0 - res_os) * inv_s2
        
        nll += np.log(sigma2[t]) + res_os
        
        grad[0] += c_grad * d_curr[0]
        grad[1] += c_grad * d_curr[1]
        grad[2] += c_grad * d_curr[2]
        
        # 4. Update for next iteration
        d_prev = d_curr
    
    return GradientResult(nll=0.5 * nll, grad=grad, sigma2=sigma2)


def garch11_normal_hessian(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> HessianResult:
    """
    GARCH(1,1) + Normal: negative log-likelihood, gradient, and analytical Hessian.
    
    The Hessian uses the recursive formulas for both first and second derivatives:
        ∂²σ²_t/∂θ_i∂θ_j = β·∂²σ²_{t-1}/∂θ_i∂θ_j + [indicator terms]
    
    Parameters
    ----------
    params : array [omega, alpha, beta]
    resid2 : array of squared residuals
    sigma2_init : float, initial variance
    
    Returns
    -------
    HessianResult with nll, grad, hess, sigma2
    """
    omega, alpha, beta = params[0], params[1], params[2]
    
    n = len(resid2)
    sigma2 = np.empty(n, dtype=np.float64)
    grad = np.zeros(3, dtype=np.float64)
    hess = np.zeros((3, 3), dtype=np.float64)
    
    # First derivatives: [∂σ²/∂ω, ∂σ²/∂α, ∂σ²/∂β]
    d_prev = np.zeros(3, dtype=np.float64)
    
    # Second derivatives (upper triangle, stored as):
    # [∂²σ²/∂ω², ∂²σ²/∂ω∂α, ∂²σ²/∂ω∂β, ∂²σ²/∂α², ∂²σ²/∂α∂β, ∂²σ²/∂β²]
    d2_prev = np.zeros(6, dtype=np.float64)
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 0
    # ─────────────────────────────────────────────────────────────────────────
    sigma2[0] = max(sigma2_init, VARIANCE_FLOOR)
    
    inv_s2 = 1.0 / sigma2[0]
    res_os = resid2[0] * inv_s2
    
    nll = np.log(sigma2[0]) + res_os
    
    # Coefficients for gradient and Hessian (for NEGATIVE log-likelihood)
    c_grad = -0.5 * (res_os - 1.0) * inv_s2
    c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os)
    
    # Hessian contribution at t=0 (all zeros since d_prev = 0)
    idx = 0
    for i in range(3):
        for j in range(i, 3):
            contrib = c_hess * d_prev[i] * d_prev[j] + c_grad * d2_prev[idx]
            hess[i, j] += contrib
            if j > i:
                hess[j, i] += contrib
            idx += 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 1 to n-1
    # ─────────────────────────────────────────────────────────────────────────
    for t in range(1, n):
        # 1. Variance recursion
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], VARIANCE_FLOOR)
        
        # 2. First derivative recursion
        d_curr = np.empty(3, dtype=np.float64)
        d_curr[0] = 1.0 + beta * d_prev[0]           # ∂σ²/∂ω
        d_curr[1] = resid2[t-1] + beta * d_prev[1]   # ∂σ²/∂α
        d_curr[2] = sigma2[t-1] + beta * d_prev[2]   # ∂σ²/∂β
        
        # 3. Second derivative recursion
        d2_curr = np.empty(6, dtype=np.float64)
        d2_curr[0] = beta * d2_prev[0]                         # ∂²σ²/∂ω²
        d2_curr[1] = beta * d2_prev[1]                         # ∂²σ²/∂ω∂α
        d2_curr[2] = d_prev[0] + beta * d2_prev[2]             # ∂²σ²/∂ω∂β
        d2_curr[3] = beta * d2_prev[3]                         # ∂²σ²/∂α²
        d2_curr[4] = d_prev[1] + beta * d2_prev[4]             # ∂²σ²/∂α∂β
        d2_curr[5] = 2.0 * d_prev[2] + beta * d2_prev[5]       # ∂²σ²/∂β²
        
        # 4. Scalars
        inv_s2 = 1.0 / sigma2[t]
        res_os = resid2[t] * inv_s2
        
        nll += np.log(sigma2[t]) + res_os
        
        # Coefficients (for NEGATIVE log-likelihood)
        c_grad = -0.5 * (res_os - 1.0) * inv_s2
        c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os)
        
        # 5. Gradient contribution (for POSITIVE gradient used in standard convention)
        grad[0] += 0.5 * (1.0 - res_os) * inv_s2 * d_curr[0]
        grad[1] += 0.5 * (1.0 - res_os) * inv_s2 * d_curr[1]
        grad[2] += 0.5 * (1.0 - res_os) * inv_s2 * d_curr[2]
        
        # 6. Hessian contribution
        idx = 0
        for i in range(3):
            for j in range(i, 3):
                contrib = c_hess * d_curr[i] * d_curr[j] + c_grad * d2_curr[idx]
                hess[i, j] += contrib
                if j > i:
                    hess[j, i] += contrib
                idx += 1
        
        # 7. Update for next iteration
        d_prev = d_curr
        d2_prev = d2_curr
    
    return HessianResult(nll=0.5 * nll, grad=grad, hess=hess, sigma2=sigma2)


def garch11_normal_robust_se(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> RobustSEResult:
    """
    GARCH(1,1) + Normal: Compute robust (sandwich) standard errors for QMLE.
    
    Computes:
    - Hessian H
    - OPG = Σ_t (g_t · g_t') where g_t is the score at observation t
    - MLE covariance: H^{-1}
    - Robust covariance: H^{-1} · OPG · H^{-1}
    
    Parameters
    ----------
    params : array [omega, alpha, beta]
    resid2 : array of squared residuals
    sigma2_init : float, initial variance
    
    Returns
    -------
    RobustSEResult with all covariance matrices and standard errors
    """
    omega, alpha, beta = params[0], params[1], params[2]
    
    n = len(resid2)
    K = 3  # number of parameters
    sigma2 = np.empty(n, dtype=np.float64)
    grad = np.zeros(K, dtype=np.float64)
    hess = np.zeros((K, K), dtype=np.float64)
    opg = np.zeros((K, K), dtype=np.float64)
    
    # First derivatives
    d_prev = np.zeros(K, dtype=np.float64)
    
    # Second derivatives (upper triangle)
    d2_prev = np.zeros(6, dtype=np.float64)
    
    nll = 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 0
    # ─────────────────────────────────────────────────────────────────────────
    sigma2[0] = max(sigma2_init, VARIANCE_FLOOR)
    
    inv_s2 = 1.0 / sigma2[0]
    res_os = resid2[0] * inv_s2
    
    nll += np.log(sigma2[0]) + res_os
    
    # Gradient coefficient for this observation
    c_grad_t = 0.5 * (1.0 - res_os) * inv_s2
    
    # Per-observation gradient (for OPG)
    g_t = np.array([c_grad_t * d_prev[0], c_grad_t * d_prev[1], c_grad_t * d_prev[2]])
    
    # Accumulate gradient
    grad += g_t
    
    # Accumulate OPG: outer product of per-observation gradient
    opg += np.outer(g_t, g_t)
    
    # Hessian coefficients
    c_grad_hess = -0.5 * (res_os - 1.0) * inv_s2
    c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os)
    
    # Hessian contribution
    idx = 0
    for i in range(K):
        for j in range(i, K):
            contrib = c_hess * d_prev[i] * d_prev[j] + c_grad_hess * d2_prev[idx]
            hess[i, j] += contrib
            if j > i:
                hess[j, i] += contrib
            idx += 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 1 to n-1
    # ─────────────────────────────────────────────────────────────────────────
    for t in range(1, n):
        # 1. Variance recursion
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], VARIANCE_FLOOR)
        
        # 2. First derivative recursion
        d_curr = np.empty(K, dtype=np.float64)
        d_curr[0] = 1.0 + beta * d_prev[0]
        d_curr[1] = resid2[t-1] + beta * d_prev[1]
        d_curr[2] = sigma2[t-1] + beta * d_prev[2]
        
        # 3. Second derivative recursion
        d2_curr = np.empty(6, dtype=np.float64)
        d2_curr[0] = beta * d2_prev[0]
        d2_curr[1] = beta * d2_prev[1]
        d2_curr[2] = d_prev[0] + beta * d2_prev[2]
        d2_curr[3] = beta * d2_prev[3]
        d2_curr[4] = d_prev[1] + beta * d2_prev[4]
        d2_curr[5] = 2.0 * d_prev[2] + beta * d2_prev[5]
        
        # 4. Scalars
        inv_s2 = 1.0 / sigma2[t]
        res_os = resid2[t] * inv_s2
        
        nll += np.log(sigma2[t]) + res_os
        
        # 5. Per-observation gradient (for OPG)
        c_grad_t = 0.5 * (1.0 - res_os) * inv_s2
        g_t = np.array([c_grad_t * d_curr[0], c_grad_t * d_curr[1], c_grad_t * d_curr[2]])
        
        # Accumulate gradient
        grad += g_t
        
        # Accumulate OPG
        opg += np.outer(g_t, g_t)
        
        # 6. Hessian contribution
        c_grad_hess = -0.5 * (res_os - 1.0) * inv_s2
        c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os)
        
        idx = 0
        for i in range(K):
            for j in range(i, K):
                contrib = c_hess * d_curr[i] * d_curr[j] + c_grad_hess * d2_curr[idx]
                hess[i, j] += contrib
                if j > i:
                    hess[j, i] += contrib
                idx += 1
        
        # 7. Update
        d_prev = d_curr
        d2_prev = d2_curr
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute covariance matrices
    # ─────────────────────────────────────────────────────────────────────────
    
    # MLE covariance: H^{-1} (Hessian is for negative log-likelihood)
    try:
        hess_inv = np.linalg.inv(hess)
        cov_mle = hess_inv
    except np.linalg.LinAlgError:
        cov_mle = np.full((K, K), np.nan)
        hess_inv = cov_mle
    
    # Robust (sandwich) covariance: H^{-1} · OPG · H^{-1}
    try:
        cov_robust = hess_inv @ opg @ hess_inv
    except:
        cov_robust = np.full((K, K), np.nan)
    
    # Standard errors
    se_mle = np.sqrt(np.diag(cov_mle)) if not np.any(np.isnan(cov_mle)) else np.full(K, np.nan)
    se_robust = np.sqrt(np.diag(cov_robust)) if not np.any(np.isnan(cov_robust)) else np.full(K, np.nan)
    
    return RobustSEResult(
        nll=0.5 * nll,
        grad=grad,
        hess=hess,
        opg=opg,
        sigma2=sigma2,
        cov_mle=cov_mle,
        cov_robust=cov_robust,
        se_mle=se_mle,
        se_robust=se_robust,
    )


# =============================================================================
# STUDENT-T DISTRIBUTION
# =============================================================================

def studentt_nll(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    nu: float,
) -> float:
    """
    Student-t negative log-likelihood.
    
    -ℓ = -n·[Γ((ν+1)/2) - Γ(ν/2) - 0.5·log(π(ν-2))]
         + 0.5·Σ[log(σ²_t) + (ν+1)·log(1 + ε²_t/(σ²_t·(ν-2)))]
    """
    n = len(resid)
    nu_m2 = nu - 2.0
    
    # Constant term
    const = n * (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * nu_m2))
    
    # Sum terms
    z2 = resid**2 / sigma2
    sum_log_sigma2 = np.sum(np.log(sigma2))
    sum_log_tail = np.sum(np.log1p(z2 / nu_m2))
    
    return 0.5 * (sum_log_sigma2 + (nu + 1) * sum_log_tail) - const


def studentt_loglik(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    nu: float,
) -> float:
    """Student-t log-likelihood (for reporting)."""
    return -studentt_nll(resid, sigma2, nu)


# =============================================================================
# GARCH(1,1) + STUDENT-T: LIKELIHOOD, GRADIENT, HESSIAN
# =============================================================================

def garch11_studentt_nll(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> LikelihoodResult:
    """
    GARCH(1,1) + Student-t negative log-likelihood.
    
    Parameters
    ----------
    params : array [omega, alpha, beta, nu]
    resid2 : array of squared residuals
    sigma2_init : float, initial variance
    
    Returns
    -------
    LikelihoodResult with nll and sigma2
    """
    omega, alpha, beta, nu = params[0], params[1], params[2], params[3]
    
    n = len(resid2)
    nu_m2 = nu - 2.0
    inv_nu_m2 = 1.0 / nu_m2
    
    # Constant term
    const = n * (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * nu_m2))
    
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = max(sigma2_init, VARIANCE_FLOOR)
    
    r2_os2 = resid2[0] / sigma2[0]
    var1 = np.log(sigma2[0])
    var2 = np.log1p(r2_os2 * inv_nu_m2)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], VARIANCE_FLOOR)
        
        r2_os2 = resid2[t] / sigma2[t]
        var1 += np.log(sigma2[t])
        var2 += np.log1p(r2_os2 * inv_nu_m2)
    
    nll = 0.5 * (var1 + (nu + 1) * var2) - const
    
    return LikelihoodResult(nll=nll, sigma2=sigma2)


def garch11_studentt_gradient(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> GradientResult:
    """
    GARCH(1,1) + Student-t: negative log-likelihood and analytical gradient.
    
    Parameters
    ----------
    params : array [omega, alpha, beta, nu]
    resid2 : array of squared residuals
    sigma2_init : float, initial variance
    
    Returns
    -------
    GradientResult with nll, grad, sigma2
    """
    omega, alpha, beta, nu = params[0], params[1], params[2], params[3]
    
    n = len(resid2)
    inv_nu_m2 = 1.0 / (nu - 2.0)
    
    # Digamma functions for ν gradient
    digamma_half_nu_plus_1 = psi(0.5 * (nu + 1.0))
    digamma_half_nu = psi(0.5 * nu)
    
    sigma2 = np.empty(n, dtype=np.float64)
    grad = np.zeros(4, dtype=np.float64)  # [ω, α, β, ν]
    
    # Derivatives of σ² w.r.t. parameters
    d_prev = np.zeros(3, dtype=np.float64)
    
    # For computing NLL
    const = n * (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi / inv_nu_m2))
    var1 = 0.0
    var2 = 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 0
    # ─────────────────────────────────────────────────────────────────────────
    sigma2[0] = max(sigma2_init, VARIANCE_FLOOR)
    
    inv_s2 = 1.0 / sigma2[0]
    res_os2 = resid2[0] * inv_s2
    one_plus_tail = 1.0 + res_os2 * inv_nu_m2
    
    var1 += np.log(sigma2[0])
    var2 += np.log(one_plus_tail)
    
    # Gradient w.r.t. σ² (for GARCH params)
    S_var = 0.5 * inv_s2 - 0.5 * (nu + 1.0) * resid2[0] * inv_nu_m2 * inv_s2 * inv_s2 / one_plus_tail
    
    grad[0] += S_var * d_prev[0]
    grad[1] += S_var * d_prev[1]
    grad[2] += S_var * d_prev[2]
    
    # Gradient w.r.t. ν
    g_nu = (-0.5 * digamma_half_nu_plus_1 + 0.5 * digamma_half_nu + 0.5 * inv_nu_m2 
            + 0.5 * np.log(one_plus_tail) 
            - 0.5 * (nu + 1.0) * resid2[0] * inv_nu_m2 * inv_nu_m2 * inv_s2 / one_plus_tail)
    grad[3] += g_nu
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 1 to n-1
    # ─────────────────────────────────────────────────────────────────────────
    for t in range(1, n):
        # 1. Variance recursion
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], VARIANCE_FLOOR)
        
        # 2. Derivative recursion
        d_curr = np.empty(3, dtype=np.float64)
        d_curr[0] = 1.0 + beta * d_prev[0]
        d_curr[1] = resid2[t-1] + beta * d_prev[1]
        d_curr[2] = sigma2[t-1] + beta * d_prev[2]
        
        # 3. Scalars
        inv_s2 = 1.0 / sigma2[t]
        res_os2 = resid2[t] * inv_s2
        one_plus_tail = 1.0 + res_os2 * inv_nu_m2
        
        var1 += np.log(sigma2[t])
        var2 += np.log(one_plus_tail)
        
        # 4. Gradient contribution for GARCH params
        S_var = 0.5 * inv_s2 - 0.5 * (nu + 1.0) * resid2[t] * inv_nu_m2 * inv_s2 * inv_s2 / one_plus_tail
        
        grad[0] += S_var * d_curr[0]
        grad[1] += S_var * d_curr[1]
        grad[2] += S_var * d_curr[2]
        
        # 5. Gradient contribution for ν
        g_nu = (-0.5 * digamma_half_nu_plus_1 + 0.5 * digamma_half_nu + 0.5 * inv_nu_m2
                + 0.5 * np.log(one_plus_tail)
                - 0.5 * (nu + 1.0) * resid2[t] * inv_nu_m2 * inv_nu_m2 * inv_s2 / one_plus_tail)
        grad[3] += g_nu
        
        # 6. Update
        d_prev = d_curr
    
    nll = 0.5 * (var1 + (nu + 1) * var2) - const
    
    return GradientResult(nll=nll, grad=grad, sigma2=sigma2)


def garch11_studentt_hessian(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> HessianResult:
    """
    GARCH(1,1) + Student-t: negative log-likelihood, gradient, and analytical Hessian.
    
    Parameters
    ----------
    params : array [omega, alpha, beta, nu]
    resid2 : array of squared residuals
    sigma2_init : float, initial variance
    
    Returns
    -------
    HessianResult with nll, grad, hess, sigma2
    """
    omega, alpha, beta, nu = params[0], params[1], params[2], params[3]
    
    n = len(resid2)
    inv_nu_m2 = 1.0 / (nu - 2.0)
    inv_nu_m2_2 = inv_nu_m2 * inv_nu_m2
    inv_nu_m2_3 = inv_nu_m2_2 * inv_nu_m2
    
    # Digamma and trigamma functions
    digamma_half_nu_plus_1 = psi(0.5 * (nu + 1.0))
    digamma_half_nu = psi(0.5 * nu)
    trigamma_half_nu_plus_1 = polygamma(1, 0.5 * (nu + 1.0))
    trigamma_half_nu = polygamma(1, 0.5 * nu)
    
    K = 4  # [ω, α, β, ν]
    sigma2 = np.empty(n, dtype=np.float64)
    grad = np.zeros(K, dtype=np.float64)
    hess = np.zeros((K, K), dtype=np.float64)
    
    # First derivatives of σ²
    d_prev = np.zeros(3, dtype=np.float64)
    
    # Second derivatives of σ² (3x3 matrix stored as array)
    C_prev = np.zeros((3, 3), dtype=np.float64)
    
    # For NLL
    const = n * (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi / inv_nu_m2))
    var1 = 0.0
    var2 = 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 0
    # ─────────────────────────────────────────────────────────────────────────
    sigma2[0] = max(sigma2_init, VARIANCE_FLOOR)
    
    inv_s2 = 1.0 / sigma2[0]
    res_os2 = resid2[0] * inv_s2
    one_plus_tail = 1.0 + res_os2 * inv_nu_m2
    
    var1 += np.log(sigma2[0])
    var2 += np.log(one_plus_tail)
    
    # Scalars for gradient/Hessian
    S_var = 0.5 * inv_s2 - 0.5 * (nu + 1.0) * resid2[0] * inv_nu_m2 * inv_s2 * inv_s2 / one_plus_tail
    
    H_var = (-0.5 * inv_s2 * inv_s2 
             + (nu + 1.0) * resid2[0] * inv_nu_m2 * inv_s2**3 / one_plus_tail
             - 0.5 * (nu + 1.0) * resid2[0]**2 * inv_nu_m2_2 * inv_s2**4 / (one_plus_tail**2))
    
    zi = res_os2 * inv_nu_m2
    dS_dnu = 0.5 * res_os2 * inv_s2 * inv_nu_m2_2 / (one_plus_tail**2) * (3.0 * one_plus_tail - (nu + 1.0) * zi)
    
    H_nu_nu = (-0.25 * trigamma_half_nu_plus_1 + 0.25 * trigamma_half_nu
               - 0.5 * inv_nu_m2_2
               - res_os2 * inv_nu_m2_2 / (2.0 * one_plus_tail)
               - 0.5 * (nu + 1.0) * res_os2**2 * inv_nu_m2_3 / (one_plus_tail**2))
    
    # Hessian contributions
    for i in range(3):
        for j in range(3):
            hess[i, j] += H_var * d_prev[i] * d_prev[j] + S_var * C_prev[i, j]
        hess[i, 3] += dS_dnu * d_prev[i]
        hess[3, i] += dS_dnu * d_prev[i]
    hess[3, 3] += H_nu_nu
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 1 to n-1
    # ─────────────────────────────────────────────────────────────────────────
    for t in range(1, n):
        # 1. Variance recursion
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], VARIANCE_FLOOR)
        
        # 2. First derivative recursion
        d_curr = np.empty(3, dtype=np.float64)
        d_curr[0] = 1.0 + beta * d_prev[0]
        d_curr[1] = resid2[t-1] + beta * d_prev[1]
        d_curr[2] = sigma2[t-1] + beta * d_prev[2]
        
        # 3. Second derivative recursion
        C_curr = np.empty((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                C_curr[i, j] = beta * C_prev[i, j]
                if i == 2:
                    C_curr[i, j] += d_prev[j]
                if j == 2:
                    C_curr[i, j] += d_prev[i]
        
        # 4. Scalars
        inv_s2 = 1.0 / sigma2[t]
        res_os2 = resid2[t] * inv_s2
        one_plus_tail = 1.0 + res_os2 * inv_nu_m2
        
        var1 += np.log(sigma2[t])
        var2 += np.log(one_plus_tail)
        
        S_var = 0.5 * inv_s2 - 0.5 * (nu + 1.0) * resid2[t] * inv_nu_m2 * inv_s2 * inv_s2 / one_plus_tail
        
        H_var = (-0.5 * inv_s2 * inv_s2
                 + (nu + 1.0) * resid2[t] * inv_nu_m2 * inv_s2**3 / one_plus_tail
                 - 0.5 * (nu + 1.0) * resid2[t]**2 * inv_nu_m2_2 * inv_s2**4 / (one_plus_tail**2))
        
        zi = res_os2 * inv_nu_m2
        dS_dnu = 0.5 * res_os2 * inv_s2 * inv_nu_m2_2 / (one_plus_tail**2) * (3.0 * one_plus_tail - (nu + 1.0) * zi)
        
        H_nu_nu = (-0.25 * trigamma_half_nu_plus_1 + 0.25 * trigamma_half_nu
                   - 0.5 * inv_nu_m2_2
                   - res_os2 * inv_nu_m2_2 / (2.0 * one_plus_tail)
                   - 0.5 * (nu + 1.0) * res_os2**2 * inv_nu_m2_3 / (one_plus_tail**2))
        
        # 5. Gradient contributions
        grad[0] += S_var * d_curr[0]
        grad[1] += S_var * d_curr[1]
        grad[2] += S_var * d_curr[2]
        
        g_nu = (-0.5 * digamma_half_nu_plus_1 + 0.5 * digamma_half_nu + 0.5 * inv_nu_m2
                + 0.5 * np.log(one_plus_tail)
                - 0.5 * (nu + 1.0) * resid2[t] * inv_nu_m2_2 * inv_s2 / one_plus_tail)
        grad[3] += g_nu
        
        # 6. Hessian contributions
        for i in range(3):
            for j in range(3):
                hess[i, j] += H_var * d_curr[i] * d_curr[j] + S_var * C_curr[i, j]
            hess[i, 3] += dS_dnu * d_curr[i]
            hess[3, i] += dS_dnu * d_curr[i]
        hess[3, 3] += H_nu_nu
        
        # 7. Update
        d_prev = d_curr
        C_prev = C_curr
    
    nll = 0.5 * (var1 + (nu + 1) * var2) - const
    
    return HessianResult(nll=nll, grad=grad, hess=hess, sigma2=sigma2)


def garch11_studentt_robust_se(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> RobustSEResult:
    """
    GARCH(1,1) + Student-t: Compute robust (sandwich) standard errors for QMLE.
    
    Parameters
    ----------
    params : array [omega, alpha, beta, nu]
    resid2 : array of squared residuals
    sigma2_init : float, initial variance
    
    Returns
    -------
    RobustSEResult with all covariance matrices and standard errors
    """
    omega, alpha, beta, nu = params[0], params[1], params[2], params[3]
    
    n = len(resid2)
    K = 4  # [ω, α, β, ν]
    inv_nu_m2 = 1.0 / (nu - 2.0)
    inv_nu_m2_2 = inv_nu_m2 * inv_nu_m2
    inv_nu_m2_3 = inv_nu_m2_2 * inv_nu_m2
    
    # Digamma and trigamma
    digamma_half_nu_plus_1 = psi(0.5 * (nu + 1.0))
    digamma_half_nu = psi(0.5 * nu)
    trigamma_half_nu_plus_1 = polygamma(1, 0.5 * (nu + 1.0))
    trigamma_half_nu = polygamma(1, 0.5 * nu)
    
    sigma2 = np.empty(n, dtype=np.float64)
    grad = np.zeros(K, dtype=np.float64)
    hess = np.zeros((K, K), dtype=np.float64)
    opg = np.zeros((K, K), dtype=np.float64)
    
    # First derivatives of σ²
    d_prev = np.zeros(3, dtype=np.float64)
    
    # Second derivatives of σ² (3x3)
    C_prev = np.zeros((3, 3), dtype=np.float64)
    
    # For NLL
    const = n * (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi / inv_nu_m2))
    var1 = 0.0
    var2 = 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 0
    # ─────────────────────────────────────────────────────────────────────────
    sigma2[0] = max(sigma2_init, VARIANCE_FLOOR)
    
    inv_s2 = 1.0 / sigma2[0]
    res_os2 = resid2[0] * inv_s2
    one_plus_tail = 1.0 + res_os2 * inv_nu_m2
    
    var1 += np.log(sigma2[0])
    var2 += np.log(one_plus_tail)
    
    # Gradient scalars
    S_var = 0.5 * inv_s2 - 0.5 * (nu + 1.0) * resid2[0] * inv_nu_m2 * inv_s2 * inv_s2 / one_plus_tail
    g_nu = (-0.5 * digamma_half_nu_plus_1 + 0.5 * digamma_half_nu + 0.5 * inv_nu_m2
            + 0.5 * np.log(one_plus_tail)
            - 0.5 * (nu + 1.0) * resid2[0] * inv_nu_m2_2 * inv_s2 / one_plus_tail)
    
    # Per-observation gradient
    g_t = np.array([S_var * d_prev[0], S_var * d_prev[1], S_var * d_prev[2], g_nu])
    grad += g_t
    opg += np.outer(g_t, g_t)
    
    # Hessian scalars
    H_var = (-0.5 * inv_s2 * inv_s2
             + (nu + 1.0) * resid2[0] * inv_nu_m2 * inv_s2**3 / one_plus_tail
             - 0.5 * (nu + 1.0) * resid2[0]**2 * inv_nu_m2_2 * inv_s2**4 / (one_plus_tail**2))
    
    zi = res_os2 * inv_nu_m2
    dS_dnu = 0.5 * res_os2 * inv_s2 * inv_nu_m2_2 / (one_plus_tail**2) * (3.0 * one_plus_tail - (nu + 1.0) * zi)
    
    H_nu_nu = (-0.25 * trigamma_half_nu_plus_1 + 0.25 * trigamma_half_nu
               - 0.5 * inv_nu_m2_2
               - res_os2 * inv_nu_m2_2 / (2.0 * one_plus_tail)
               - 0.5 * (nu + 1.0) * res_os2**2 * inv_nu_m2_3 / (one_plus_tail**2))
    
    # Hessian contributions
    for i in range(3):
        for j in range(3):
            hess[i, j] += H_var * d_prev[i] * d_prev[j] + S_var * C_prev[i, j]
        hess[i, 3] += dS_dnu * d_prev[i]
        hess[3, i] += dS_dnu * d_prev[i]
    hess[3, 3] += H_nu_nu
    
    # ─────────────────────────────────────────────────────────────────────────
    # t = 1 to n-1
    # ─────────────────────────────────────────────────────────────────────────
    for t in range(1, n):
        # 1. Variance recursion
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], VARIANCE_FLOOR)
        
        # 2. First derivative recursion
        d_curr = np.empty(3, dtype=np.float64)
        d_curr[0] = 1.0 + beta * d_prev[0]
        d_curr[1] = resid2[t-1] + beta * d_prev[1]
        d_curr[2] = sigma2[t-1] + beta * d_prev[2]
        
        # 3. Second derivative recursion
        C_curr = np.empty((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                C_curr[i, j] = beta * C_prev[i, j]
                if i == 2:
                    C_curr[i, j] += d_prev[j]
                if j == 2:
                    C_curr[i, j] += d_prev[i]
        
        # 4. Scalars
        inv_s2 = 1.0 / sigma2[t]
        res_os2 = resid2[t] * inv_s2
        one_plus_tail = 1.0 + res_os2 * inv_nu_m2
        
        var1 += np.log(sigma2[t])
        var2 += np.log(one_plus_tail)
        
        # 5. Per-observation gradient
        S_var = 0.5 * inv_s2 - 0.5 * (nu + 1.0) * resid2[t] * inv_nu_m2 * inv_s2 * inv_s2 / one_plus_tail
        g_nu = (-0.5 * digamma_half_nu_plus_1 + 0.5 * digamma_half_nu + 0.5 * inv_nu_m2
                + 0.5 * np.log(one_plus_tail)
                - 0.5 * (nu + 1.0) * resid2[t] * inv_nu_m2_2 * inv_s2 / one_plus_tail)
        
        g_t = np.array([S_var * d_curr[0], S_var * d_curr[1], S_var * d_curr[2], g_nu])
        grad += g_t
        opg += np.outer(g_t, g_t)
        
        # 6. Hessian scalars
        H_var = (-0.5 * inv_s2 * inv_s2
                 + (nu + 1.0) * resid2[t] * inv_nu_m2 * inv_s2**3 / one_plus_tail
                 - 0.5 * (nu + 1.0) * resid2[t]**2 * inv_nu_m2_2 * inv_s2**4 / (one_plus_tail**2))
        
        zi = res_os2 * inv_nu_m2
        dS_dnu = 0.5 * res_os2 * inv_s2 * inv_nu_m2_2 / (one_plus_tail**2) * (3.0 * one_plus_tail - (nu + 1.0) * zi)
        
        H_nu_nu = (-0.25 * trigamma_half_nu_plus_1 + 0.25 * trigamma_half_nu
                   - 0.5 * inv_nu_m2_2
                   - res_os2 * inv_nu_m2_2 / (2.0 * one_plus_tail)
                   - 0.5 * (nu + 1.0) * res_os2**2 * inv_nu_m2_3 / (one_plus_tail**2))
        
        # 7. Hessian contributions
        for i in range(3):
            for j in range(3):
                hess[i, j] += H_var * d_curr[i] * d_curr[j] + S_var * C_curr[i, j]
            hess[i, 3] += dS_dnu * d_curr[i]
            hess[3, i] += dS_dnu * d_curr[i]
        hess[3, 3] += H_nu_nu
        
        # 8. Update
        d_prev = d_curr
        C_prev = C_curr
    
    nll = 0.5 * (var1 + (nu + 1) * var2) - const
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute covariance matrices
    # ─────────────────────────────────────────────────────────────────────────
    
    try:
        hess_inv = np.linalg.inv(hess)
        cov_mle = hess_inv
    except np.linalg.LinAlgError:
        cov_mle = np.full((K, K), np.nan)
        hess_inv = cov_mle
    
    try:
        cov_robust = hess_inv @ opg @ hess_inv
    except:
        cov_robust = np.full((K, K), np.nan)
    
    se_mle = np.sqrt(np.diag(cov_mle)) if not np.any(np.isnan(cov_mle)) else np.full(K, np.nan)
    se_robust = np.sqrt(np.diag(cov_robust)) if not np.any(np.isnan(cov_robust)) else np.full(K, np.nan)
    
    return RobustSEResult(
        nll=nll,
        grad=grad,
        hess=hess,
        opg=opg,
        sigma2=sigma2,
        cov_mle=cov_mle,
        cov_robust=cov_robust,
        se_mle=se_mle,
        se_robust=se_robust,
    )


# =============================================================================
# SKEWED STUDENT-T (Hansen, 1994)
# =============================================================================

def skewt_loglik(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    nu: float,
    lam: float,
) -> float:
    """
    Skewed Student-t log-likelihood (Hansen, 1994 parameterization).
    
    Parameters
    ----------
    resid : array of residuals
    sigma2 : array of conditional variances
    nu : degrees of freedom (> 2)
    lam : asymmetry parameter (-1, 1)
    
    Returns
    -------
    loglik : float
    """
    # Hansen (1994) constants
    c = gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2))
    a = 4 * lam * np.exp(c) * (nu - 2) / (nu - 1)
    b = np.sqrt(1 + 3 * lam**2 - a**2)
    
    # Standardized residuals
    z = resid / np.sqrt(sigma2)
    
    # Adjust for asymmetry
    z_adj = (b * z + a) / (1 - lam * np.sign(b * z + a))
    
    # Log-likelihood
    ll = (len(resid) * c 
          - 0.5 * np.sum(np.log(sigma2))
          + np.sum(np.log(b))
          - 0.5 * (nu + 1) * np.sum(np.log(1 + z_adj**2 / (nu - 2))))
    
    return ll


def skewt_nll(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    nu: float,
    lam: float,
) -> float:
    """Skewed Student-t negative log-likelihood."""
    return -skewt_loglik(resid, sigma2, nu, lam)


# =============================================================================
# WRAPPER FUNCTIONS FOR OPTIMIZER
# =============================================================================

def make_normal_objective(resid2: NDArray[np.float64], sigma2_init: float):
    """Create objective function for GARCH(1,1) + Normal."""
    
    def objective(params: NDArray[np.float64]) -> float:
        result = garch11_normal_nll(params, resid2, sigma2_init)
        return result.nll
    
    def gradient(params: NDArray[np.float64]) -> NDArray[np.float64]:
        result = garch11_normal_gradient(params, resid2, sigma2_init)
        return result.grad
    
    def hessian(params: NDArray[np.float64]) -> NDArray[np.float64]:
        result = garch11_normal_hessian(params, resid2, sigma2_init)
        return result.hess
    
    return objective, gradient, hessian


def make_studentt_objective(resid2: NDArray[np.float64], sigma2_init: float):
    """Create objective function for GARCH(1,1) + Student-t."""
    
    def objective(params: NDArray[np.float64]) -> float:
        result = garch11_studentt_nll(params, resid2, sigma2_init)
        return result.nll
    
    def gradient(params: NDArray[np.float64]) -> NDArray[np.float64]:
        result = garch11_studentt_gradient(params, resid2, sigma2_init)
        return result.grad
    
    def hessian(params: NDArray[np.float64]) -> NDArray[np.float64]:
        result = garch11_studentt_hessian(params, resid2, sigma2_init)
        return result.hess
    
    return objective, gradient, hessian


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing analytical derivatives...")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate GARCH(1,1) process
    n = 500
    omega_true, alpha_true, beta_true = 0.01, 0.1, 0.85
    
    sigma2_true = np.zeros(n)
    resid = np.zeros(n)
    sigma2_true[0] = omega_true / (1 - alpha_true - beta_true)
    
    for t in range(1, n):
        sigma2_true[t] = omega_true + alpha_true * resid[t-1]**2 + beta_true * sigma2_true[t-1]
        resid[t] = np.sqrt(sigma2_true[t]) * np.random.standard_normal()
    
    resid2 = resid**2
    sigma2_init = np.mean(resid2)
    
    # Test Normal
    print("\n1. GARCH(1,1) + Normal")
    print("-" * 40)
    
    params_normal = np.array([omega_true, alpha_true, beta_true])
    
    # Analytical gradient
    result_grad = garch11_normal_gradient(params_normal, resid2, sigma2_init)
    
    # Numerical gradient
    eps = 1e-6
    grad_num = np.zeros(3)
    for i in range(3):
        p_plus = params_normal.copy()
        p_minus = params_normal.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        
        nll_plus = garch11_normal_nll(p_plus, resid2, sigma2_init).nll
        nll_minus = garch11_normal_nll(p_minus, resid2, sigma2_init).nll
        grad_num[i] = (nll_plus - nll_minus) / (2 * eps)
    
    print(f"Analytical gradient: {result_grad.grad}")
    print(f"Numerical gradient:  {grad_num}")
    print(f"Max difference:      {np.max(np.abs(result_grad.grad - grad_num)):.2e}")
    
    # Analytical Hessian
    result_hess = garch11_normal_hessian(params_normal, resid2, sigma2_init)
    
    # Numerical Hessian
    hess_num = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            p_pp = params_normal.copy()
            p_pm = params_normal.copy()
            p_mp = params_normal.copy()
            p_mm = params_normal.copy()
            
            p_pp[i] += eps; p_pp[j] += eps
            p_pm[i] += eps; p_pm[j] -= eps
            p_mp[i] -= eps; p_mp[j] += eps
            p_mm[i] -= eps; p_mm[j] -= eps
            
            hess_num[i, j] = (garch11_normal_nll(p_pp, resid2, sigma2_init).nll
                             - garch11_normal_nll(p_pm, resid2, sigma2_init).nll
                             - garch11_normal_nll(p_mp, resid2, sigma2_init).nll
                             + garch11_normal_nll(p_mm, resid2, sigma2_init).nll) / (4 * eps**2)
    
    print(f"\nAnalytical Hessian:\n{result_hess.hess}")
    print(f"\nNumerical Hessian:\n{hess_num}")
    print(f"Max difference: {np.max(np.abs(result_hess.hess - hess_num)):.2e}")
    
    # Test Student-t
    print("\n\n2. GARCH(1,1) + Student-t")
    print("-" * 40)
    
    nu_true = 8.0
    params_t = np.array([omega_true, alpha_true, beta_true, nu_true])
    
    # Analytical gradient
    result_grad_t = garch11_studentt_gradient(params_t, resid2, sigma2_init)
    
    # Numerical gradient
    grad_num_t = np.zeros(4)
    for i in range(4):
        p_plus = params_t.copy()
        p_minus = params_t.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        
        nll_plus = garch11_studentt_nll(p_plus, resid2, sigma2_init).nll
        nll_minus = garch11_studentt_nll(p_minus, resid2, sigma2_init).nll
        grad_num_t[i] = (nll_plus - nll_minus) / (2 * eps)
    
    print(f"Analytical gradient: {result_grad_t.grad}")
    print(f"Numerical gradient:  {grad_num_t}")
    print(f"Max difference:      {np.max(np.abs(result_grad_t.grad - grad_num_t)):.2e}")
    
    print("\n" + "=" * 60)
    print("Tests completed!")

