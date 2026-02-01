"""
ARMA(p,q)-GARCH(P,Q) Estimator with Analytical Gradients and Hessians
======================================================================

Reference implementation for C kernel development.

Model:
------
Mean equation (ARMA):
    μ_t = c + Σᵢ φᵢ y_{t-i} + Σⱼ θⱼ e_{t-j}
    e_t = y_t - μ_t

Variance equation (GARCH):  
    h_t = ω + Σᵢ αᵢ e²_{t-i} + Σⱼ βⱼ h_{t-j}

Likelihood (Normal):
    ℓ_t = -½ log(h_t) - ½ e_t²/h_t

Key insight for derivatives:
- For ARMA (linear in params for AR, nonlinear for MA due to e_{t-j} recursion)
- AR terms: ∂e_t/∂φᵢ = -y_{t-i} + Σⱼ θⱼ ∂e_{t-j}/∂φᵢ
- MA terms: ∂e_t/∂θⱼ = -e_{t-j} + Σₖ θₖ ∂e_{t-k}/∂θⱼ
- GARCH coupling: ∂h_t/∂(ARMA params) comes through ∂e²/∂θ = 2e·∂e/∂θ

Special cases:
- ARMA(1,1)-GARCH(1,1): Optimized implementation
- ARMA(p,q)-GARCH(P,Q): General implementation

Author: volkit development
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import time
from scipy.optimize import minimize

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: identity decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) is False else decorator(args[0]) if args else decorator


# =============================================================================
# CONSTANTS AND HELPERS
# =============================================================================

# Minimum variance floor to prevent numerical issues
H_FLOOR = 1e-12

# Log of constants
LOG_2PI = np.log(2 * np.pi)


# =============================================================================
# DISTRIBUTION LIKELIHOODS (Per-observation)
# =============================================================================

@njit(cache=True)
def _normal_nll(e: float, h: float) -> float:
    """Normal NLL: 0.5*log(2π) + 0.5*log(h) + 0.5*e²/h"""
    return 0.5 * (LOG_2PI + np.log(h) + e * e / h)


@njit(cache=True)
def _normal_nll_grad(e: float, h: float) -> Tuple[float, float]:
    """
    Gradients of Normal NLL w.r.t. e and h.
    Returns (∂NLL/∂e, ∂NLL/∂h)
    """
    ell_e = e / h
    ell_h = 0.5 * (1.0 / h - e * e / (h * h))
    return ell_e, ell_h


@njit(cache=True)
def _normal_nll_hess(e: float, h: float) -> Tuple[float, float, float]:
    """
    Second derivatives of Normal NLL.
    Returns (∂²NLL/∂e², ∂²NLL/∂h², ∂²NLL/∂e∂h)
    """
    h2 = h * h
    h3 = h2 * h
    e2 = e * e
    
    ell_ee = 1.0 / h
    ell_hh = 0.5 * (-1.0 / h2 + 2.0 * e2 / h3)
    ell_eh = -e / h2
    
    return ell_ee, ell_hh, ell_eh


@njit(cache=True)
def _lgamma(x: float) -> float:
    """Log-gamma function using Stirling's approximation for large x."""
    if x <= 0:
        return np.inf
    if x < 10:
        # Use recursion: log(Γ(x)) = log(Γ(x+1)) - log(x)
        result = 0.0
        while x < 10:
            result -= np.log(x)
            x += 1.0
        return result + _lgamma(x)
    # Stirling's approximation for x >= 10
    return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi) + 1.0 / (12.0 * x)


@njit(cache=True)
def _digamma(x: float) -> float:
    """Digamma function (ψ(x) = d/dx log(Γ(x)))."""
    if x <= 0:
        return np.nan
    result = 0.0
    # Use recursion to get x > 6
    while x < 6:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic expansion for large x
    x2 = x * x
    return result + np.log(x) - 0.5 / x - 1.0 / (12.0 * x2) + 1.0 / (120.0 * x2 * x2)


@njit(cache=True)
def _studentt_nll(e: float, h: float, nu: float) -> float:
    """
    Student-t NLL (scaled by h).
    
    z = e / sqrt(h)
    NLL = -log(Γ((ν+1)/2)) + log(Γ(ν/2)) + 0.5*log(νπ) + 0.5*log(h) + ((ν+1)/2)*log(1 + z²/ν)
    """
    if nu <= 2.0:
        return 1e10
    
    z2 = e * e / h
    
    # Log-likelihood terms
    const = _lgamma(0.5 * (nu + 1)) - _lgamma(0.5 * nu) - 0.5 * np.log(nu * np.pi)
    
    nll = -const + 0.5 * np.log(h) + 0.5 * (nu + 1) * np.log(1.0 + z2 / nu)
    
    return nll


@njit(cache=True)
def _studentt_nll_grad(e: float, h: float, nu: float) -> Tuple[float, float, float]:
    """
    Gradients of Student-t NLL w.r.t. e, h, and ν.
    Returns (∂NLL/∂e, ∂NLL/∂h, ∂NLL/∂ν)
    """
    z2 = e * e / h
    factor = 1.0 + z2 / nu
    
    # ∂NLL/∂e = (ν+1) * e / (h * (ν + z²))
    ell_e = (nu + 1) * e / (h * nu * factor)
    
    # ∂NLL/∂h = 0.5/h - 0.5*(ν+1)*z²/(h²*(ν + z²)/ν) = 0.5/h - 0.5*(ν+1)*e²/(h²*(ν + e²/h))
    ell_h = 0.5 / h - 0.5 * (nu + 1) * z2 / (h * nu * factor)
    
    # ∂NLL/∂ν = 0.5*(ψ((ν+1)/2) - ψ(ν/2) - 1/ν - log(1 + z²/ν) + (ν+1)*z²/(ν²*(1 + z²/ν)))
    psi_term = 0.5 * (_digamma(0.5 * (nu + 1)) - _digamma(0.5 * nu))
    ell_nu = -psi_term + 0.5 / nu + 0.5 * np.log(factor) - 0.5 * (nu + 1) * z2 / (nu * nu * factor)
    
    return ell_e, ell_h, ell_nu


@njit(cache=True)
def _studentt_nll_hess_eh(e: float, h: float, nu: float) -> Tuple[float, float, float]:
    """
    Second derivatives of Student-t NLL w.r.t. (e, h).
    Returns (∂²NLL/∂e², ∂²NLL/∂h², ∂²NLL/∂e∂h)
    
    Note: We skip ν second derivatives for simplicity (use BFGS for ν).
    """
    z2 = e * e / h
    factor = 1.0 + z2 / nu
    factor2 = factor * factor
    
    # ∂²NLL/∂e² = (ν+1)/h * (1/(ν·factor) - 2e²/(h·ν²·factor²))
    #           = (ν+1)/(h·ν·factor) * (1 - 2z²/(ν·factor))
    ell_ee = (nu + 1) / (h * nu * factor) * (1.0 - 2.0 * z2 / (nu * factor))
    
    # ∂²NLL/∂h²
    term1 = -0.5 / (h * h)
    term2 = (nu + 1) * z2 / (h * h * nu * factor)
    term3 = -(nu + 1) * z2 * z2 / (h * h * nu * nu * factor2)
    ell_hh = term1 + term2 + term3
    
    # ∂²NLL/∂e∂h = -(ν+1)*e/(h²·ν·factor) + (ν+1)*e*z²/(h²·ν²·factor²)
    ell_eh = -(nu + 1) * e / (h * h * nu * factor) * (1.0 - z2 / (nu * factor))
    
    return ell_ee, ell_hh, ell_eh


@njit(cache=True)
def _skewt_nll(e: float, h: float, nu: float, lam: float) -> float:
    """
    Hansen's Skew-t NLL.
    
    Parameters:
        e: residual
        h: variance
        nu: degrees of freedom (> 2)
        lam: skewness parameter in (-1, 1)
    """
    if nu <= 2.0 or abs(lam) >= 1.0:
        return 1e10
    
    # Constants
    a = 4.0 * lam * ((nu - 2.0) / (nu - 1.0)) * _lgamma_ratio(nu)
    b2 = 1.0 + 3.0 * lam * lam - a * a
    b = np.sqrt(b2)
    
    # Standardized residual
    z = e / np.sqrt(h)
    
    # Adjusted residual
    if z < -a / b:
        # Left tail
        zstar = b * z + a
        scale = 1.0 - lam
    else:
        # Right tail
        zstar = b * z + a
        scale = 1.0 + lam
    
    zstar_adj = zstar / scale
    z2_adj = zstar_adj * zstar_adj
    
    # Log-likelihood
    const = np.log(b) - np.log(scale) + _lgamma(0.5 * (nu + 1)) - _lgamma(0.5 * nu) - 0.5 * np.log((nu - 2) * np.pi)
    
    nll = -const + 0.5 * np.log(h) + 0.5 * (nu + 1) * np.log(1.0 + z2_adj / (nu - 2))
    
    return nll


@njit(cache=True)
def _lgamma_ratio(nu: float) -> float:
    """Compute Γ((ν+1)/2) / (√((ν-2)/π) * Γ(ν/2))"""
    return np.exp(_lgamma(0.5 * (nu + 1)) - _lgamma(0.5 * nu) - 0.5 * np.log((nu - 2) / np.pi))


# =============================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS (ARMA(1,1)-GARCH(1,1) NORMAL)
# =============================================================================


@njit(cache=True)
def _forward_recursion_11(
    y: np.ndarray,
    c: float,
    phi: float, 
    theta_ma: float,
    omega: float,
    alpha: float,
    beta: float,
    h0: float,
    e0: float,
    resid: np.ndarray,
    sigma2: np.ndarray,
    de: np.ndarray,
    dh: np.ndarray,
    d2e: np.ndarray,
    d2h: np.ndarray,
) -> Tuple[float, bool]:
    """
    Numba-accelerated forward recursion for ARMA(1,1)-GARCH(1,1).
    
    Initialization convention:
    - e_0 = 0 (given, conditioned on)
    - h_0 = mean(y²) (given, conditioned on)
    - LL starts at t=1 (first obs with proper y_{t-1} available)
    
    At t=1: e_1 = y_1 - c - φ·y_0 - θ·e_0
    
    Returns (nll, valid) where nll is scaled by (n-1) not n.
    """
    n = len(y)
    K = 6  # Number of parameters
    n_eff = n - 1  # Effective sample size (LL starts at t=1)
    
    # Parameter indices
    I_c = 0
    I_phi = 1
    I_theta = 2
    I_omega = 3
    I_alpha = 4
    I_beta = 5
    
    # =========================================================================
    # Initialize t=0 (conditioning values, not in likelihood)
    # =========================================================================
    # e_0 is given as input (typically 0)
    # h_0 is given as input (typically mean(y²))
    resid[0] = e0  # Store e_0 for use at t=1
    sigma2[0] = h0
    
    # Sensitivities at t=0 are all zero (conditioning values don't depend on params)
    for k in range(K):
        de[0, k] = 0.0
        dh[0, k] = 0.0
        for l in range(K):
            d2e[0, k, l] = 0.0
            d2h[0, k, l] = 0.0
    
    # Check validity
    if sigma2[0] < H_FLOOR:
        return 1e10, False
    
    # Accumulators (will start summing from t=1)
    sum_log_h = 0.0
    sum_e2_over_h = 0.0
    
    # =========================================================================
    # Forward recursion t=1,...,n-1
    # =========================================================================
    for t in range(1, n):
        e_prev = resid[t - 1]
        h_prev = sigma2[t - 1]
        e2_prev = e_prev * e_prev
        
        # ARMA residual: e_t = y_t - c - φ·y_{t-1} - θ·e_{t-1}
        resid[t] = y[t] - c - phi * y[t - 1] - theta_ma * e_prev
        
        # GARCH variance: h_t = ω + α·e²_{t-1} + β·h_{t-1}
        sigma2[t] = omega + alpha * e2_prev + beta * h_prev
        
        # Guard against numerical issues
        if sigma2[t] < H_FLOOR or not np.isfinite(sigma2[t]):
            return 1e10, False
        
        # =====================================================
        # ∂e_t/∂θ sensitivities (first derivatives)
        # =====================================================
        de[t, I_c] = -1.0 - theta_ma * de[t - 1, I_c]
        de[t, I_phi] = -y[t - 1] - theta_ma * de[t - 1, I_phi]
        de[t, I_theta] = -e_prev - theta_ma * de[t - 1, I_theta]
        de[t, I_omega] = -theta_ma * de[t - 1, I_omega]
        de[t, I_alpha] = -theta_ma * de[t - 1, I_alpha]
        de[t, I_beta] = -theta_ma * de[t - 1, I_beta]
        
        # =====================================================
        # ∂²e_t/∂θ∂θ' sensitivities (second derivatives)
        # =====================================================
        # Initialize with recursion through θ
        for k in range(K):
            for l in range(K):
                d2e[t, k, l] = -theta_ma * d2e[t - 1, k, l]
        
        # Add the derivative-of-derivative terms for theta
        d2e[t, I_theta, I_theta] += -2.0 * de[t - 1, I_theta]
        d2e[t, I_c, I_theta] += -de[t - 1, I_c]
        d2e[t, I_theta, I_c] += -de[t - 1, I_c]
        d2e[t, I_phi, I_theta] += -de[t - 1, I_phi]
        d2e[t, I_theta, I_phi] += -de[t - 1, I_phi]
        
        # =====================================================
        # ∂(e²)/∂θ = 2·e·∂e/∂θ
        # =====================================================
        de2_prev_c = 2.0 * e_prev * de[t - 1, I_c]
        de2_prev_phi = 2.0 * e_prev * de[t - 1, I_phi]
        de2_prev_theta = 2.0 * e_prev * de[t - 1, I_theta]
        de2_prev_omega = 2.0 * e_prev * de[t - 1, I_omega]
        de2_prev_alpha = 2.0 * e_prev * de[t - 1, I_alpha]
        de2_prev_beta = 2.0 * e_prev * de[t - 1, I_beta]
        
        # =====================================================
        # ∂h_t/∂θ sensitivities
        # =====================================================
        dh[t, I_c] = alpha * de2_prev_c + beta * dh[t - 1, I_c]
        dh[t, I_phi] = alpha * de2_prev_phi + beta * dh[t - 1, I_phi]
        dh[t, I_theta] = alpha * de2_prev_theta + beta * dh[t - 1, I_theta]
        dh[t, I_omega] = 1.0 + alpha * de2_prev_omega + beta * dh[t - 1, I_omega]
        dh[t, I_alpha] = e2_prev + alpha * de2_prev_alpha + beta * dh[t - 1, I_alpha]
        dh[t, I_beta] = h_prev + alpha * de2_prev_beta + beta * dh[t - 1, I_beta]
        
        # =====================================================
        # ∂²h_t/∂θ∂θ' (second sensitivities for Hessian)
        # =====================================================
        for k in range(K):
            for l in range(K):
                d2e2_kl = 2.0 * (de[t - 1, k] * de[t - 1, l] + e_prev * d2e[t - 1, k, l])
                d2h[t, k, l] = alpha * d2e2_kl + beta * d2h[t - 1, k, l]
        
        # Indicator terms for α
        d2h[t, I_alpha, I_c] += de2_prev_c
        d2h[t, I_c, I_alpha] += de2_prev_c
        d2h[t, I_alpha, I_phi] += de2_prev_phi
        d2h[t, I_phi, I_alpha] += de2_prev_phi
        d2h[t, I_alpha, I_theta] += de2_prev_theta
        d2h[t, I_theta, I_alpha] += de2_prev_theta
        d2h[t, I_alpha, I_omega] += de2_prev_omega
        d2h[t, I_omega, I_alpha] += de2_prev_omega
        d2h[t, I_alpha, I_alpha] += 2.0 * de2_prev_alpha
        d2h[t, I_alpha, I_beta] += de2_prev_beta
        d2h[t, I_beta, I_alpha] += de2_prev_beta
        
        # Indicator terms for β
        d2h[t, I_beta, I_c] += dh[t - 1, I_c]
        d2h[t, I_c, I_beta] += dh[t - 1, I_c]
        d2h[t, I_beta, I_phi] += dh[t - 1, I_phi]
        d2h[t, I_phi, I_beta] += dh[t - 1, I_phi]
        d2h[t, I_beta, I_theta] += dh[t - 1, I_theta]
        d2h[t, I_theta, I_beta] += dh[t - 1, I_theta]
        d2h[t, I_beta, I_omega] += dh[t - 1, I_omega]
        d2h[t, I_omega, I_beta] += dh[t - 1, I_omega]
        d2h[t, I_beta, I_alpha] += dh[t - 1, I_alpha]
        d2h[t, I_alpha, I_beta] += dh[t - 1, I_alpha]
        d2h[t, I_beta, I_beta] += 2.0 * dh[t - 1, I_beta]
        
        # Accumulate log-likelihood (starts at t=1)
        sum_log_h += np.log(sigma2[t])
        sum_e2_over_h += resid[t] * resid[t] / sigma2[t]
    
    # Negative log-likelihood (scaled by effective sample size)
    nll = 0.5 * (sum_log_h + sum_e2_over_h) / n_eff
    
    return nll, True


@njit(cache=True)
def _compute_gradient_11(
    resid: np.ndarray,
    sigma2: np.ndarray,
    de: np.ndarray,
    dh: np.ndarray,
) -> np.ndarray:
    """
    Numba-accelerated gradient computation.
    
    Gradient: ∂NLL/∂θ = (1/(n-1)) Σ_{t=1}^{n-1} [ (e/h)·∂e/∂θ + ½(1/h - e²/h²)·∂h/∂θ ]
    
    Note: t=0 is skipped (conditioning values).
    """
    n = len(resid)
    n_eff = n - 1
    K = 6
    grad = np.zeros(K)
    
    # Sum from t=1 (skip t=0 which is conditioning)
    for t in range(1, n):
        e = resid[t]
        h = sigma2[t]
        
        # Scalar partials
        ell_e = e / h
        ell_h = 0.5 * (1.0 / h - e * e / (h * h))
        
        for k in range(K):
            grad[k] += ell_e * de[t, k] + ell_h * dh[t, k]
    
    return grad / n_eff


@njit(cache=True)
def _compute_hessian_11(
    resid: np.ndarray,
    sigma2: np.ndarray,
    de: np.ndarray,
    dh: np.ndarray,
    d2e: np.ndarray,
    d2h: np.ndarray,
) -> np.ndarray:
    """
    Numba-accelerated Hessian computation.
    
    Includes the ℓ_e·d²e term for exact second derivatives.
    Note: t=0 is skipped (conditioning values).
    """
    n = len(resid)
    n_eff = n - 1
    K = 6
    hess = np.zeros((K, K))
    
    # Sum from t=1 (skip t=0 which is conditioning)
    for t in range(1, n):
        e = resid[t]
        h = sigma2[t]
        h2 = h * h
        h3 = h2 * h
        e2 = e * e
        
        # Scalar first and second partials for NLL
        ell_e = e / h  # ∂NLL/∂e
        ell_ee = 1.0 / h
        ell_hh = 0.5 * (-1.0 / h2 + 2.0 * e2 / h3)
        ell_eh = -e / h2
        ell_h = 0.5 * (1.0 / h - e2 / h2)
        
        # Accumulate Hessian
        for k in range(K):
            for l in range(K):
                hess[k, l] += (ell_ee * de[t, k] * de[t, l]
                             + ell_hh * dh[t, k] * dh[t, l]
                             + ell_eh * (de[t, k] * dh[t, l] + dh[t, k] * de[t, l])
                             + ell_e * d2e[t, k, l]
                             + ell_h * d2h[t, k, l])
    
    return hess / n_eff


# =============================================================================
# RESULT CONTAINER
# =============================================================================

@dataclass
class ARMAGARCHResult:
    """Container for ARMA-GARCH estimation results."""
    # Parameters
    c: float                    # Constant
    phi: NDArray[np.float64]    # AR coefficients
    theta: NDArray[np.float64]  # MA coefficients  
    omega: float                # GARCH constant
    alpha: NDArray[np.float64]  # ARCH coefficients
    beta: NDArray[np.float64]   # GARCH coefficients
    
    # Fit statistics
    log_likelihood: float
    n_obs: int
    n_params: int
    
    # Optimization
    converged: bool
    n_iter: int
    time_elapsed: float
    solver: str
    
    # Intermediate results
    resid: NDArray[np.float64]      # Residuals e_t
    sigma2: NDArray[np.float64]     # Conditional variances h_t
    
    # Standard errors (if computed)
    std_errors: Optional[NDArray[np.float64]] = None
    hessian: Optional[NDArray[np.float64]] = None
    
    @property
    def aic(self) -> float:
        return -2 * self.log_likelihood + 2 * self.n_params
    
    @property
    def bic(self) -> float:
        return -2 * self.log_likelihood + np.log(self.n_obs) * self.n_params
    
    @property
    def persistence(self) -> float:
        return np.sum(self.alpha) + np.sum(self.beta)


# =============================================================================
# ARMA(1,1)-GARCH(1,1) SPECIALIZED IMPLEMENTATION
# =============================================================================

class ARMA11GARCH11Normal:
    """
    ARMA(1,1)-GARCH(1,1) with Normal innovations.
    
    Specialized implementation for the common case.
    
    Parameters: θ = [c, φ, θ_ma, ω, α, β]  (6 params)
    
    Model:
        μ_t = c + φ·y_{t-1} + θ_ma·e_{t-1}
        e_t = y_t - μ_t
        h_t = ω + α·e²_{t-1} + β·h_{t-1}
        y_t | F_{t-1} ~ N(μ_t, h_t)
    """
    
    def __init__(self, y: NDArray[np.float64]):
        """Initialize with data."""
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.n = len(y)
        self.K = 6  # Number of parameters
        
        # Pre-allocate working arrays
        self._resid = np.zeros(self.n, dtype=np.float64)
        self._sigma2 = np.zeros(self.n, dtype=np.float64)
        
        # Pre-allocate sensitivity arrays
        self._de = np.zeros((self.n, self.K), dtype=np.float64)  # ∂e_t/∂θ
        self._dh = np.zeros((self.n, self.K), dtype=np.float64)  # ∂h_t/∂θ
        self._d2e = np.zeros((self.n, self.K, self.K), dtype=np.float64)  # ∂²e_t/∂θ∂θ'
        self._d2h = np.zeros((self.n, self.K, self.K), dtype=np.float64)  # ∂²h_t/∂θ∂θ'
        
        # Initial values (conditioning)
        self._e0 = 0.0  # e_0 = 0 (conditioned on)
        self._h0 = np.mean(y ** 2)  # h_0 = mean(y²) (conditioned on)
    
    def _compute_states(self, params: NDArray[np.float64]) -> Tuple[float, bool]:
        """
        Compute residuals and variances, return negative log-likelihood.
        
        Uses numba-accelerated core function for performance.
        """
        c, phi, theta_ma, omega, alpha, beta = params
        
        # Validate parameters
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10, False
        if not np.isfinite(c) or not np.isfinite(phi) or not np.isfinite(theta_ma):
            return 1e10, False
        
        # Call numba-accelerated forward recursion
        nll, valid = _forward_recursion_11(
            self.y, c, phi, theta_ma, omega, alpha, beta,
            self._h0, self._e0,
            self._resid, self._sigma2, self._de, self._dh, self._d2e, self._d2h
        )
        
        return nll, valid
    
    def objective(self, params: NDArray[np.float64]) -> float:
        """Negative log-likelihood (for minimization)."""
        nll, valid = self._compute_states(params)
        return nll if valid else 1e10
    
    def gradient(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Analytical gradient of negative log-likelihood.
        
        Uses numba-accelerated computation.
        """
        # Ensure states are computed
        nll, valid = self._compute_states(params)
        if not valid:
            return np.full(self.K, np.nan, dtype=np.float64)
        
        return _compute_gradient_11(self._resid, self._sigma2, self._de, self._dh)
    
    def hessian(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Analytical Hessian of negative log-likelihood.
        
        Uses numba-accelerated computation.
        """
        # Ensure states are computed
        nll, valid = self._compute_states(params)
        if not valid:
            return np.full((self.K, self.K), np.nan, dtype=np.float64)
        
        return _compute_hessian_11(
            self._resid, self._sigma2, self._de, self._dh, self._d2e, self._d2h
        )
    
    def fit(
        self, 
        solver: str = "trust-constr",
        verbose: bool = False,
        x0: Optional[NDArray[np.float64]] = None,
    ) -> ARMAGARCHResult:
        """
        Fit the ARMA(1,1)-GARCH(1,1) model.
        
        Parameters
        ----------
        solver : str
            'trust-constr' (uses gradient + Hessian)
            'nelder-mead' (gradient-free)
            'slsqp' (uses gradient only)
        verbose : bool
            Print optimization progress
        x0 : array, optional
            Initial parameter values [c, φ, θ_ma, ω, α, β]
        
        Returns
        -------
        ARMAGARCHResult
        """
        t_start = time.perf_counter()
        
        # Default starting values
        # Note: The likelihood surface has saddle points. Starting from (phi=0, theta=0)
        # tends to work well for most solvers.
        if x0 is None:
            var_y = np.var(self.y)
            x0 = np.array([
                np.mean(self.y),  # c
                0.0,              # φ (start at zero)
                0.0,              # θ_ma (start at zero)
                var_y * 0.05,     # ω
                0.1,              # α
                0.85,             # β
            ], dtype=np.float64)
        
        # Bounds
        bounds = [
            (-np.inf, np.inf),  # c
            (-0.99, 0.99),      # φ (stationarity)
            (-0.99, 0.99),      # θ_ma (invertibility)
            (1e-10, np.inf),    # ω > 0
            (1e-10, 0.999),     # α > 0
            (1e-10, 0.999),     # β > 0
        ]
        
        # Stationarity constraint for GARCH: α + β < 1
        def garch_constraint(x):
            return 0.999 - x[4] - x[5]
        
        constraints = {'type': 'ineq', 'fun': garch_constraint}
        
        if solver.lower() == "trust-constr":
            from scipy.optimize import LinearConstraint
            
            # Trust-constr with analytical derivatives
            # Use LinearConstraint for alpha + beta <= 0.999 (better than NonlinearConstraint)
            linear_constraint = LinearConstraint(
                [[0, 0, 0, 0, 1, 1]],  # alpha + beta
                -np.inf,               # lower bound
                0.999                  # upper bound
            )
            
            res = minimize(
                self.objective,
                x0,
                method='trust-constr',
                jac=self.gradient,
                hess=self.hessian,
                bounds=bounds,
                constraints=linear_constraint,
                options={'disp': verbose, 'maxiter': 500, 'gtol': 1e-8},
            )
        
        elif solver.lower() == "slsqp-nograd":
            # SLSQP without gradients (for comparison)
            # Uses finite-difference gradients internally
            res = minimize(
                self.objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'disp': verbose,
                    'maxiter': 1000,
                },
            )
        
        elif solver.lower() == "slsqp":
            res = minimize(
                self.objective,
                x0,
                method='SLSQP',
                jac=self.gradient,
                bounds=bounds,
                constraints=constraints,
                options={'disp': verbose, 'maxiter': 2000, 'ftol': 1e-15},
            )
        
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        t_elapsed = time.perf_counter() - t_start
        
        # Extract final parameters
        c, phi, theta_ma, omega, alpha, beta = res.x
        
        # Compute final states
        self._compute_states(res.x)
        
        # Compute Hessian for standard errors
        try:
            hess = self.hessian(res.x)
            hess_inv = np.linalg.inv(hess)
            std_errors = np.sqrt(np.diag(hess_inv) / self.n)
        except:
            hess = None
            std_errors = None
        
        return ARMAGARCHResult(
            c=c,
            phi=np.array([phi]),
            theta=np.array([theta_ma]),
            omega=omega,
            alpha=np.array([alpha]),
            beta=np.array([beta]),
            log_likelihood=-res.fun * (self.n - 1) - 0.5 * (self.n - 1) * np.log(2 * np.pi),  # Full LL with constant
            n_obs=self.n,
            n_params=self.K,
            converged=res.success,
            n_iter=res.nit if hasattr(res, 'nit') else res.nfev,
            time_elapsed=t_elapsed,
            solver=solver,
            resid=self._resid.copy(),
            sigma2=self._sigma2.copy(),
            std_errors=std_errors,
            hessian=hess,
        )


# =============================================================================
# NUMERICAL DERIVATIVE VERIFICATION
# =============================================================================

def verify_gradient_numerically(
    model: ARMA11GARCH11Normal,
    params: NDArray[np.float64],
    eps: float = 1e-7,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Verify analytical gradient against finite differences.
    
    Uses relative step size for better accuracy across different scales.
    
    Returns (analytical, numerical, relative_error)
    """
    grad_ana = model.gradient(params)
    grad_num = np.zeros_like(params)
    
    for i in range(len(params)):
        # Use relative step size
        step = eps * max(abs(params[i]), 1e-8)
        
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += step
        p_minus[i] -= step
        grad_num[i] = (model.objective(p_plus) - model.objective(p_minus)) / (2 * step)
    
    rel_err = np.abs(grad_ana - grad_num) / (np.abs(grad_ana) + 1e-10)
    
    return grad_ana, grad_num, rel_err


def verify_hessian_numerically(
    model: ARMA11GARCH11Normal,
    params: NDArray[np.float64],
    eps: float = 1e-5,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Verify analytical Hessian against finite differences (via gradient).
    
    More accurate than double finite-difference of objective.
    
    Returns (analytical, numerical, relative_error)
    """
    hess_ana = model.hessian(params)
    K = len(params)
    hess_num = np.zeros((K, K), dtype=np.float64)
    
    # Compute Hessian via finite differences of gradient
    for i in range(K):
        step = eps * max(abs(params[i]), 1e-8)
        
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += step
        p_minus[i] -= step
        
        grad_plus = model.gradient(p_plus)
        grad_minus = model.gradient(p_minus)
        
        hess_num[:, i] = (grad_plus - grad_minus) / (2 * step)
    
    # Symmetrize
    hess_num = 0.5 * (hess_num + hess_num.T)
    
    rel_err = np.abs(hess_ana - hess_num) / (np.abs(hess_ana) + 1e-10)
    
    return hess_ana, hess_num, rel_err


# =============================================================================
# ARMA(1,1) MODEL (NO GARCH - CONSTANT VARIANCE)
# =============================================================================

@njit(cache=True)
def _arma11_forward(
    y: np.ndarray,
    c: float,
    phi: float,
    theta_ma: float,
    sigma2: float,
    e0: float,
    resid: np.ndarray,
    de: np.ndarray,
    d2e: np.ndarray,
) -> Tuple[float, bool]:
    """
    Forward recursion for ARMA(1,1) with constant variance.
    
    Parameters: [c, phi, theta, sigma2] (K=4)
    
    Initialization:
    - e_0 = 0 (conditioned on)
    - LL starts at t=1
    """
    n = len(y)
    n_eff = n - 1  # Effective sample size
    K = 4
    
    I_c = 0
    I_phi = 1
    I_theta = 2
    I_sigma2 = 3
    
    if sigma2 <= H_FLOOR:
        return 1e10, False
    
    # Initialize t=0 (conditioning values)
    resid[0] = e0  # Store e_0 for use at t=1
    
    for k in range(K):
        de[0, k] = 0.0
        for l in range(K):
            d2e[0, k, l] = 0.0
    
    sum_e2 = 0.0  # Start sum at 0 (will sum from t=1)
    
    for t in range(1, n):
        e_prev = resid[t - 1]
        
        # e_t = y_t - c - φ·y_{t-1} - θ·e_{t-1}
        resid[t] = y[t] - c - phi * y[t - 1] - theta_ma * e_prev
        
        # First derivatives
        de[t, I_c] = -1.0 - theta_ma * de[t - 1, I_c]
        de[t, I_phi] = -y[t - 1] - theta_ma * de[t - 1, I_phi]
        de[t, I_theta] = -e_prev - theta_ma * de[t - 1, I_theta]
        de[t, I_sigma2] = 0.0
        
        # Second derivatives (only theta-related are non-zero)
        for k in range(K):
            for l in range(K):
                d2e[t, k, l] = -theta_ma * d2e[t - 1, k, l]
        
        d2e[t, I_theta, I_theta] += -2.0 * de[t - 1, I_theta]
        d2e[t, I_c, I_theta] += -de[t - 1, I_c]
        d2e[t, I_theta, I_c] += -de[t - 1, I_c]
        d2e[t, I_phi, I_theta] += -de[t - 1, I_phi]
        d2e[t, I_theta, I_phi] += -de[t - 1, I_phi]
        
        sum_e2 += resid[t] * resid[t]
    
    # NLL/(n-1) = 0.5*log(σ²) + 0.5*Σe²/((n-1)*σ²)
    nll = 0.5 * np.log(sigma2) + 0.5 * sum_e2 / (n_eff * sigma2)
    
    return nll, True


@njit(cache=True)
def _arma11_gradient(
    resid: np.ndarray,
    de: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    """Gradient for ARMA(1,1) with constant variance. Skips t=0."""
    n = len(resid)
    n_eff = n - 1
    K = 4
    grad = np.zeros(K)
    
    I_sigma2 = 3
    
    sum_e2 = 0.0
    # Sum from t=1 (skip t=0)
    for t in range(1, n):
        e = resid[t]
        sum_e2 += e * e
        
        # ∂NLL/∂θ_k = (1/σ²)·e·∂e/∂θ_k (for mean params)
        for k in range(3):  # c, phi, theta only
            grad[k] += e * de[t, k] / sigma2
    
    # ∂NLL/∂σ² = (n-1)/(2σ²) - Σe²/(2σ⁴)
    grad[I_sigma2] = 0.5 / sigma2 - 0.5 * sum_e2 / (n_eff * sigma2 * sigma2)
    
    # Scale by 1/(n-1)
    for k in range(3):
        grad[k] /= n_eff
    
    return grad


@njit(cache=True)
def _arma11_hessian(
    resid: np.ndarray,
    de: np.ndarray,
    d2e: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    """Hessian for ARMA(1,1) with constant variance. Skips t=0."""
    n = len(resid)
    n_eff = n - 1
    K = 4
    hess = np.zeros((K, K))
    
    I_sigma2 = 3
    
    sum_e2 = 0.0
    sum_e_de = np.zeros(3)
    
    # Sum from t=1 (skip t=0)
    for t in range(1, n):
        e = resid[t]
        sum_e2 += e * e
        
        for k in range(3):
            sum_e_de[k] += e * de[t, k]
        
        # ∂²NLL/∂θ_k∂θ_l = (1/σ²)·[(∂e/∂θ_k)(∂e/∂θ_l) + e·∂²e/∂θ_k∂θ_l]
        for k in range(3):
            for l in range(3):
                hess[k, l] += (de[t, k] * de[t, l] + e * d2e[t, k, l]) / sigma2
    
    # Scale mean-param block by 1/(n-1)
    for k in range(3):
        for l in range(3):
            hess[k, l] /= n_eff
    
    # ∂²NLL/∂σ²² = -1/(2σ⁴) + Σe²/(σ⁶)
    hess[I_sigma2, I_sigma2] = -0.5 / (sigma2 * sigma2) + sum_e2 / (n_eff * sigma2**3)
    
    # ∂²NLL/∂θ_k∂σ² = -e·(∂e/∂θ_k)/σ⁴
    for k in range(3):
        val = -sum_e_de[k] / (n_eff * sigma2 * sigma2)
        hess[k, I_sigma2] = val
        hess[I_sigma2, k] = val
    
    return hess


@dataclass
class ARMA11Result:
    """Result of ARMA(1,1) estimation."""
    c: float
    phi: float
    theta: float
    sigma2: float
    log_likelihood: float
    aic: float
    bic: float
    n_obs: int
    converged: bool
    time_elapsed: float
    resid: Optional[NDArray[np.float64]] = None


class ARMA11Normal:
    """
    ARMA(1,1) model with Normal innovations and constant variance.
    
    Model:
        y_t = c + φ·y_{t-1} + θ·e_{t-1} + e_t
        e_t ~ N(0, σ²)
    
    Parameters: [c, φ, θ, σ²]
    """
    
    def __init__(self, y: NDArray[np.float64]):
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.n = len(y)
        self.K = 4
        
        self._resid = np.zeros(self.n, dtype=np.float64)
        self._de = np.zeros((self.n, self.K), dtype=np.float64)
        self._d2e = np.zeros((self.n, self.K, self.K), dtype=np.float64)
        self._e0 = 0.0
    
    def _compute_states(self, params: NDArray[np.float64]) -> Tuple[float, bool]:
        c, phi, theta_ma, sigma2 = params
        return _arma11_forward(
            self.y, c, phi, theta_ma, sigma2, self._e0,
            self._resid, self._de, self._d2e
        )
    
    def objective(self, params: NDArray[np.float64]) -> float:
        nll, valid = self._compute_states(params)
        return nll if valid else 1e10
    
    def gradient(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        nll, valid = self._compute_states(params)
        if not valid:
            return np.full(self.K, np.nan)
        return _arma11_gradient(self._resid, self._de, params[3])
    
    def hessian(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        nll, valid = self._compute_states(params)
        if not valid:
            return np.full((self.K, self.K), np.nan)
        return _arma11_hessian(self._resid, self._de, self._d2e, params[3])
    
    def fit(
        self,
        x0: Optional[NDArray[np.float64]] = None,
        solver: str = "slsqp",
        verbose: bool = True,
    ) -> ARMA11Result:
        """Fit ARMA(1,1) model."""
        
        if x0 is None:
            # Default starting values
            var_y = np.var(self.y)
            x0 = np.array([np.mean(self.y), 0.0, 0.0, var_y])
        
        bounds = [
            (None, None),      # c
            (-0.999, 0.999),   # phi
            (-0.999, 0.999),   # theta
            (1e-10, None),     # sigma2
        ]
        
        t0 = time.perf_counter()
        
        if solver == "slsqp":
            res = minimize(
                self.objective, x0, method="SLSQP",
                jac=self.gradient, bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-12}
            )
        elif solver == "trust-constr":
            from scipy.optimize import Bounds
            lb = np.array([-np.inf, -0.999, -0.999, 1e-10])
            ub = np.array([np.inf, 0.999, 0.999, np.inf])
            res = minimize(
                self.objective, x0, method="trust-constr",
                jac=self.gradient, hess=self.hessian,
                bounds=Bounds(lb, ub),
                options={"maxiter": 500, "gtol": 1e-8}
            )
        else:
            res = minimize(
                self.objective, x0, method="SLSQP",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-12}
            )
        
        t_elapsed = time.perf_counter() - t0
        
        c, phi, theta, sigma2 = res.x
        
        # Compute final residuals
        self._compute_states(res.x)
        
        n_eff = self.n - 1  # Effective sample size (t=0 is conditioning)
        ll = -res.fun * n_eff - 0.5 * n_eff * np.log(2 * np.pi)  # Full LL with constant
        
        aic = 2 * self.K - 2 * ll
        bic = self.K * np.log(n_eff) - 2 * ll
        
        return ARMA11Result(
            c=c, phi=phi, theta=theta, sigma2=sigma2,
            log_likelihood=ll, aic=aic, bic=bic,
            n_obs=self.n, converged=res.success,
            time_elapsed=t_elapsed,
            resid=self._resid.copy()
        )


# =============================================================================
# GARCH(1,1) MODEL (NO ARMA - FOR SEQUENTIAL ESTIMATION)
# =============================================================================

@njit(cache=True)
def _garch11_forward(
    eps2: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    h0: float,
    sigma2: np.ndarray,
    dh: np.ndarray,
    d2h: np.ndarray,
) -> Tuple[float, bool]:
    """
    Forward recursion for GARCH(1,1) given squared residuals.
    
    Initialization:
    - h_0 = mean(eps²) (conditioned on)
    - LL starts at t=1
    """
    n = len(eps2)
    n_eff = n - 1  # Effective sample size
    K = 3  # omega, alpha, beta
    
    I_omega = 0
    I_alpha = 1
    I_beta = 2
    
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10, False
    
    # Initialize t=0 (conditioning)
    sigma2[0] = h0
    dh[0, :] = 0.0
    d2h[0, :, :] = 0.0
    
    if sigma2[0] < H_FLOOR:
        return 1e10, False
    
    # Start sums at 0 (will sum from t=1)
    sum_log_h = 0.0
    sum_e2_over_h = 0.0
    
    for t in range(1, n):
        e2_prev = eps2[t - 1]
        h_prev = sigma2[t - 1]
        
        sigma2[t] = omega + alpha * e2_prev + beta * h_prev
        
        if sigma2[t] < H_FLOOR or not np.isfinite(sigma2[t]):
            return 1e10, False
        
        # First derivatives
        dh[t, I_omega] = 1.0 + beta * dh[t - 1, I_omega]
        dh[t, I_alpha] = e2_prev + beta * dh[t - 1, I_alpha]
        dh[t, I_beta] = h_prev + beta * dh[t - 1, I_beta]
        
        # Second derivatives
        for k in range(K):
            for l in range(K):
                d2h[t, k, l] = beta * d2h[t - 1, k, l]
        
        # Indicator terms for beta
        d2h[t, I_beta, I_omega] += dh[t - 1, I_omega]
        d2h[t, I_omega, I_beta] += dh[t - 1, I_omega]
        d2h[t, I_beta, I_alpha] += dh[t - 1, I_alpha]
        d2h[t, I_alpha, I_beta] += dh[t - 1, I_alpha]
        d2h[t, I_beta, I_beta] += 2.0 * dh[t - 1, I_beta]
        
        sum_log_h += np.log(sigma2[t])
        sum_e2_over_h += eps2[t] / sigma2[t]
    
    nll = 0.5 * (sum_log_h + sum_e2_over_h) / n_eff
    return nll, True


@njit(cache=True)
def _garch11_gradient(
    eps2: np.ndarray,
    sigma2: np.ndarray,
    dh: np.ndarray,
) -> np.ndarray:
    """Gradient for GARCH(1,1). Skips t=0."""
    n = len(eps2)
    n_eff = n - 1
    K = 3
    grad = np.zeros(K)
    
    # Sum from t=1 (skip t=0)
    for t in range(1, n):
        e2 = eps2[t]
        h = sigma2[t]
        
        ell_h = 0.5 * (1.0 / h - e2 / (h * h))
        
        for k in range(K):
            grad[k] += ell_h * dh[t, k]
    
    return grad / n_eff


@njit(cache=True)
def _garch11_hessian(
    eps2: np.ndarray,
    sigma2: np.ndarray,
    dh: np.ndarray,
    d2h: np.ndarray,
) -> np.ndarray:
    """Hessian for GARCH(1,1). Skips t=0."""
    n = len(eps2)
    n_eff = n - 1
    K = 3
    hess = np.zeros((K, K))
    
    # Sum from t=1 (skip t=0)
    for t in range(1, n):
        e2 = eps2[t]
        h = sigma2[t]
        h2 = h * h
        h3 = h2 * h
        
        ell_h = 0.5 * (1.0 / h - e2 / h2)
        ell_hh = 0.5 * (-1.0 / h2 + 2.0 * e2 / h3)
        
        for k in range(K):
            for l in range(K):
                hess[k, l] += ell_hh * dh[t, k] * dh[t, l] + ell_h * d2h[t, k, l]
    
    return hess / n_eff


@dataclass
class GARCH11Result:
    """Result of GARCH(1,1) estimation."""
    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    aic: float
    bic: float
    n_obs: int
    converged: bool
    time_elapsed: float
    sigma2: Optional[NDArray[np.float64]] = None


class GARCH11Normal:
    """
    GARCH(1,1) model given pre-computed residuals.
    
    Model:
        h_t = ω + α·e²_{t-1} + β·h_{t-1}
        e_t ~ N(0, h_t)
    
    Parameters: [ω, α, β]
    """
    
    def __init__(self, eps: NDArray[np.float64]):
        self.eps = np.ascontiguousarray(eps, dtype=np.float64)
        self.eps2 = self.eps ** 2
        self.n = len(eps)
        self.K = 3
        
        self._sigma2 = np.zeros(self.n, dtype=np.float64)
        self._dh = np.zeros((self.n, self.K), dtype=np.float64)
        self._d2h = np.zeros((self.n, self.K, self.K), dtype=np.float64)
        self._h0 = np.mean(self.eps2)
    
    def _compute_states(self, params: NDArray[np.float64]) -> Tuple[float, bool]:
        omega, alpha, beta = params
        return _garch11_forward(
            self.eps2, omega, alpha, beta, self._h0,
            self._sigma2, self._dh, self._d2h
        )
    
    def objective(self, params: NDArray[np.float64]) -> float:
        nll, valid = self._compute_states(params)
        return nll if valid else 1e10
    
    def gradient(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        nll, valid = self._compute_states(params)
        if not valid:
            return np.full(self.K, np.nan)
        return _garch11_gradient(self.eps2, self._sigma2, self._dh)
    
    def hessian(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        nll, valid = self._compute_states(params)
        if not valid:
            return np.full((self.K, self.K), np.nan)
        return _garch11_hessian(self.eps2, self._sigma2, self._dh, self._d2h)
    
    def fit(
        self,
        x0: Optional[NDArray[np.float64]] = None,
        solver: str = "slsqp",
        verbose: bool = True,
    ) -> GARCH11Result:
        """Fit GARCH(1,1) model."""
        
        if x0 is None:
            var_eps = np.var(self.eps)
            x0 = np.array([var_eps * 0.02, 0.05, 0.90])
        
        bounds = [
            (1e-10, None),     # omega
            (1e-10, 0.999),    # alpha
            (1e-10, 0.999),    # beta
        ]
        
        from scipy.optimize import LinearConstraint
        constraint = LinearConstraint([[0, 1, 1]], -np.inf, 0.999)
        
        t0 = time.perf_counter()
        
        if solver == "slsqp":
            res = minimize(
                self.objective, x0, method="SLSQP",
                jac=self.gradient, bounds=bounds,
                constraints={"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2]},
                options={"maxiter": 2000, "ftol": 1e-12}
            )
        elif solver == "trust-constr":
            from scipy.optimize import Bounds
            lb = np.array([1e-10, 1e-10, 1e-10])
            ub = np.array([np.inf, 0.999, 0.999])
            res = minimize(
                self.objective, x0, method="trust-constr",
                jac=self.gradient, hess=self.hessian,
                bounds=Bounds(lb, ub), constraints=constraint,
                options={"maxiter": 500, "gtol": 1e-8}
            )
        else:
            res = minimize(
                self.objective, x0, method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2]},
                options={"maxiter": 2000, "ftol": 1e-12}
            )
        
        t_elapsed = time.perf_counter() - t0
        
        omega, alpha, beta = res.x
        
        self._compute_states(res.x)
        
        n_eff = self.n - 1  # Effective sample size (t=0 is conditioning)
        ll = -res.fun * n_eff - 0.5 * n_eff * np.log(2 * np.pi)  # Full LL with constant
        aic = 2 * self.K - 2 * ll
        bic = self.K * np.log(n_eff) - 2 * ll
        
        return GARCH11Result(
            omega=omega, alpha=alpha, beta=beta,
            log_likelihood=ll, aic=aic, bic=bic,
            n_obs=self.n, converged=res.success,
            time_elapsed=t_elapsed,
            sigma2=self._sigma2.copy()
        )


# =============================================================================
# LOG-TRANSFORMED (UNCONSTRAINED) PARAMETRIZATIONS
# =============================================================================

@njit(cache=True)
def _softmax_garch(eta_alpha: float, eta_beta: float, margin: float = 0.001) -> Tuple[float, float]:
    """
    Softmax transform for GARCH α, β ensuring α > 0, β > 0, α + β < 1-margin.
    
    α = (1-margin) * exp(η_α) / (1 + exp(η_α) + exp(η_β))
    β = (1-margin) * exp(η_β) / (1 + exp(η_α) + exp(η_β))
    """
    # Numerical stability: shift by max
    max_eta = max(0.0, eta_alpha, eta_beta)
    exp_a = np.exp(eta_alpha - max_eta)
    exp_b = np.exp(eta_beta - max_eta)
    exp_0 = np.exp(-max_eta)
    
    denom = exp_0 + exp_a + exp_b
    scale = 1.0 - margin
    
    alpha = scale * exp_a / denom
    beta = scale * exp_b / denom
    
    return alpha, beta


@njit(cache=True)
def _softmax_garch_jac(eta_alpha: float, eta_beta: float, margin: float = 0.001) -> Tuple[float, float, float, float]:
    """
    Jacobian of softmax transform.
    
    Returns (∂α/∂η_α, ∂α/∂η_β, ∂β/∂η_α, ∂β/∂η_β).
    """
    alpha, beta = _softmax_garch(eta_alpha, eta_beta, margin)
    scale = 1.0 - margin
    
    # ∂α/∂η_α = α * (1 - α/scale)
    # ∂α/∂η_β = -α * β / scale
    da_da = alpha * (1.0 - alpha / scale)
    da_db = -alpha * beta / scale
    db_da = -alpha * beta / scale
    db_db = beta * (1.0 - beta / scale)
    
    return da_da, da_db, db_da, db_db


class ARMA11GARCH11NormalLog:
    """
    ARMA(1,1)-GARCH(1,1) with log-transformed (unconstrained) parameters.
    
    Transformations:
        c       = c* (no transform)
        φ       = tanh(φ*)
        θ       = tanh(θ*)
        ω       = exp(log_ω)
        (α, β)  = softmax(η_α, η_β) with margin
    
    Unconstrained params: [c*, φ*, θ*, log_ω, η_α, η_β]
    """
    
    def __init__(self, y: NDArray[np.float64]):
        self._base = ARMA11GARCH11Normal(y)
        self.n = self._base.n
        self.K = 6
    
    def _transform(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform unconstrained to constrained params."""
        c_star, phi_star, theta_star, log_omega, eta_alpha, eta_beta = u
        
        c = c_star
        phi = np.tanh(phi_star)
        theta = np.tanh(theta_star)
        omega = np.exp(log_omega)
        alpha, beta = _softmax_garch(eta_alpha, eta_beta)
        
        return np.array([c, phi, theta, omega, alpha, beta])
    
    def _inverse_transform(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constrained to unconstrained params."""
        c, phi, theta, omega, alpha, beta = p
        
        c_star = c
        phi_star = np.arctanh(np.clip(phi, -0.999, 0.999))
        theta_star = np.arctanh(np.clip(theta, -0.999, 0.999))
        log_omega = np.log(omega)
        
        # Inverse softmax: η_α = log(α) - log(1-α-β), η_β = log(β) - log(1-α-β)
        margin = 0.001
        remainder = 1.0 - margin - alpha - beta
        if remainder < 1e-10:
            remainder = 1e-10
        eta_alpha = np.log(alpha / (1 - margin)) - np.log(remainder / (1 - margin))
        eta_beta = np.log(beta / (1 - margin)) - np.log(remainder / (1 - margin))
        
        return np.array([c_star, phi_star, theta_star, log_omega, eta_alpha, eta_beta])
    
    def objective(self, u: NDArray[np.float64]) -> float:
        """NLL in unconstrained space."""
        p = self._transform(u)
        return self._base.objective(p)
    
    def gradient(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gradient in unconstrained space using chain rule."""
        p = self._transform(u)
        grad_p = self._base.gradient(p)
        
        # Jacobian of transform
        c_star, phi_star, theta_star, log_omega, eta_alpha, eta_beta = u
        
        # ∂φ/∂φ* = 1 - tanh²(φ*)
        phi = np.tanh(phi_star)
        dphi = 1.0 - phi * phi
        
        # ∂θ/∂θ* = 1 - tanh²(θ*)
        theta = np.tanh(theta_star)
        dtheta = 1.0 - theta * theta
        
        # ∂ω/∂log_ω = ω
        omega = np.exp(log_omega)
        
        # Softmax Jacobian
        da_da, da_db, db_da, db_db = _softmax_garch_jac(eta_alpha, eta_beta)
        
        # Chain rule: ∂L/∂u = J^T @ ∂L/∂p
        grad_u = np.zeros(6)
        grad_u[0] = grad_p[0]  # c
        grad_u[1] = grad_p[1] * dphi  # phi
        grad_u[2] = grad_p[2] * dtheta  # theta
        grad_u[3] = grad_p[3] * omega  # omega
        grad_u[4] = grad_p[4] * da_da + grad_p[5] * db_da  # eta_alpha
        grad_u[5] = grad_p[4] * da_db + grad_p[5] * db_db  # eta_beta
        
        return grad_u
    
    def fit(
        self,
        x0: Optional[NDArray[np.float64]] = None,
        solver: str = "bfgs",
        verbose: bool = True,
    ) -> ARMAGARCHResult:
        """Fit using unconstrained optimization."""
        
        if x0 is None:
            # Default in constrained space, then transform
            var_y = np.var(self._base.y)
            p0 = np.array([np.mean(self._base.y), 0.0, 0.0, var_y * 0.02, 0.1, 0.85])
            u0 = self._inverse_transform(p0)
        else:
            u0 = x0
        
        t0 = time.perf_counter()
        
        if solver == "bfgs":
            res = minimize(
                self.objective, u0, method="BFGS",
                jac=self.gradient,
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        elif solver == "l-bfgs-b":
            res = minimize(
                self.objective, u0, method="L-BFGS-B",
                jac=self.gradient,
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        elif solver == "bfgs-nograd":
            res = minimize(
                self.objective, u0, method="BFGS",
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        else:
            res = minimize(
                self.objective, u0, method="BFGS",
                jac=self.gradient,
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        
        t_elapsed = time.perf_counter() - t0
        
        # Transform back to constrained
        p = self._transform(res.x)
        c, phi, theta_ma, omega, alpha, beta = p
        
        n_eff = self.n - 1
        ll = -res.fun * n_eff - 0.5 * n_eff * np.log(2 * np.pi)
        
        # Compute final states for sigma2 and resid
        self._base._compute_states(p)
        
        return ARMAGARCHResult(
            c=c,
            phi=np.array([phi]),
            theta=np.array([theta_ma]),
            omega=omega,
            alpha=np.array([alpha]),
            beta=np.array([beta]),
            log_likelihood=ll,
            n_obs=self.n,
            n_params=self.K,
            converged=res.success,
            n_iter=res.nit if hasattr(res, 'nit') else res.nfev,
            time_elapsed=t_elapsed,
            solver=solver,
            resid=self._base._resid.copy(),
            sigma2=self._base._sigma2.copy(),
            std_errors=None,
            hessian=None,
        )


class ARMA11NormalLog:
    """
    ARMA(1,1) with log-transformed (unconstrained) parameters.
    
    Transformations:
        c       = c* (no transform)
        φ       = tanh(φ*)
        θ       = tanh(θ*)
        σ²      = exp(log_σ²)
    
    Unconstrained params: [c*, φ*, θ*, log_σ²]
    """
    
    def __init__(self, y: NDArray[np.float64]):
        self._base = ARMA11Normal(y)
        self.n = self._base.n
        self.K = 4
    
    def _transform(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform unconstrained to constrained params."""
        c_star, phi_star, theta_star, log_sigma2 = u
        
        c = c_star
        phi = np.tanh(phi_star)
        theta = np.tanh(theta_star)
        sigma2 = np.exp(log_sigma2)
        
        return np.array([c, phi, theta, sigma2])
    
    def _inverse_transform(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constrained to unconstrained params."""
        c, phi, theta, sigma2 = p
        
        c_star = c
        phi_star = np.arctanh(np.clip(phi, -0.999, 0.999))
        theta_star = np.arctanh(np.clip(theta, -0.999, 0.999))
        log_sigma2 = np.log(sigma2)
        
        return np.array([c_star, phi_star, theta_star, log_sigma2])
    
    def objective(self, u: NDArray[np.float64]) -> float:
        """NLL in unconstrained space."""
        p = self._transform(u)
        return self._base.objective(p)
    
    def gradient(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gradient in unconstrained space using chain rule."""
        p = self._transform(u)
        grad_p = self._base.gradient(p)
        
        c_star, phi_star, theta_star, log_sigma2 = u
        
        phi = np.tanh(phi_star)
        dphi = 1.0 - phi * phi
        
        theta = np.tanh(theta_star)
        dtheta = 1.0 - theta * theta
        
        sigma2 = np.exp(log_sigma2)
        
        grad_u = np.zeros(4)
        grad_u[0] = grad_p[0]  # c
        grad_u[1] = grad_p[1] * dphi  # phi
        grad_u[2] = grad_p[2] * dtheta  # theta
        grad_u[3] = grad_p[3] * sigma2  # sigma2
        
        return grad_u
    
    def fit(
        self,
        x0: Optional[NDArray[np.float64]] = None,
        solver: str = "bfgs",
        verbose: bool = True,
    ) -> ARMA11Result:
        """Fit using unconstrained optimization."""
        
        if x0 is None:
            var_y = np.var(self._base.y)
            p0 = np.array([np.mean(self._base.y), 0.0, 0.0, var_y])
            u0 = self._inverse_transform(p0)
        else:
            u0 = x0
        
        t0 = time.perf_counter()
        
        if solver == "bfgs":
            res = minimize(
                self.objective, u0, method="BFGS",
                jac=self.gradient,
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        elif solver == "bfgs-nograd":
            res = minimize(
                self.objective, u0, method="BFGS",
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        else:
            res = minimize(
                self.objective, u0, method="BFGS",
                jac=self.gradient,
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        
        t_elapsed = time.perf_counter() - t0
        
        p = self._transform(res.x)
        c, phi, theta, sigma2 = p
        
        self._base._compute_states(p)
        
        n_eff = self.n - 1
        ll = -res.fun * n_eff - 0.5 * n_eff * np.log(2 * np.pi)
        aic = 2 * self.K - 2 * ll
        bic = self.K * np.log(n_eff) - 2 * ll
        
        return ARMA11Result(
            c=c, phi=phi, theta=theta, sigma2=sigma2,
            log_likelihood=ll, aic=aic, bic=bic,
            n_obs=self.n, converged=res.success,
            time_elapsed=t_elapsed,
            resid=self._base._resid.copy()
        )


# =============================================================================
# ARMA(1,1)-GARCH(1,1) WITH STUDENT-T
# =============================================================================

@njit(cache=True)
def _forward_recursion_11_studentt(
    y: np.ndarray,
    c: float,
    phi: float,
    theta_ma: float,
    omega: float,
    alpha: float,
    beta: float,
    nu: float,
    h0: float,
    e0: float,
    resid: np.ndarray,
    sigma2: np.ndarray,
) -> Tuple[float, bool]:
    """Forward recursion for ARMA(1,1)-GARCH(1,1) with Student-t."""
    n = len(y)
    n_eff = n - 1
    
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1 or nu <= 2:
        return 1e10, False
    
    resid[0] = e0
    sigma2[0] = h0
    
    if sigma2[0] < H_FLOOR:
        return 1e10, False
    
    sum_nll = 0.0
    
    for t in range(1, n):
        e_prev = resid[t - 1]
        h_prev = sigma2[t - 1]
        
        resid[t] = y[t] - c - phi * y[t - 1] - theta_ma * e_prev
        sigma2[t] = omega + alpha * e_prev * e_prev + beta * h_prev
        
        if sigma2[t] < H_FLOOR or not np.isfinite(sigma2[t]):
            return 1e10, False
        
        sum_nll += _studentt_nll(resid[t], sigma2[t], nu)
    
    return sum_nll / n_eff, True


class ARMA11GARCH11StudentT:
    """
    ARMA(1,1)-GARCH(1,1) with Student-t innovations.
    
    Parameters: [c, φ, θ, ω, α, β, ν]
    """
    
    def __init__(self, y: NDArray[np.float64]):
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.n = len(y)
        self.K = 7  # 6 ARMA-GARCH params + nu
        
        self._resid = np.zeros(self.n, dtype=np.float64)
        self._sigma2 = np.zeros(self.n, dtype=np.float64)
        
        self._e0 = 0.0
        self._h0 = np.mean(y ** 2)
    
    def objective(self, params: NDArray[np.float64]) -> float:
        c, phi, theta_ma, omega, alpha, beta, nu = params
        nll, valid = _forward_recursion_11_studentt(
            self.y, c, phi, theta_ma, omega, alpha, beta, nu,
            self._h0, self._e0, self._resid, self._sigma2
        )
        return nll if valid else 1e10
    
    def gradient(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Numerical gradient (analytical is complex for Student-t)."""
        eps = 1e-7
        grad = np.zeros(self.K)
        f0 = self.objective(params)
        
        for k in range(self.K):
            p_plus = params.copy()
            h = eps * max(1.0, abs(params[k]))
            p_plus[k] += h
            grad[k] = (self.objective(p_plus) - f0) / h
        
        return grad
    
    def fit(
        self,
        x0: Optional[NDArray[np.float64]] = None,
        solver: str = "slsqp",
        verbose: bool = True,
    ) -> 'ARMAGARCHStudentTResult':
        
        if x0 is None:
            var_y = np.var(self.y)
            x0 = np.array([np.mean(self.y), 0.0, 0.0, var_y * 0.02, 0.1, 0.85, 8.0])
        
        bounds = [
            (None, None),      # c
            (-0.999, 0.999),   # phi
            (-0.999, 0.999),   # theta
            (1e-10, None),     # omega
            (1e-10, 0.999),    # alpha
            (1e-10, 0.999),    # beta
            (2.01, 100.0),     # nu
        ]
        
        from scipy.optimize import LinearConstraint
        constraint = LinearConstraint([[0, 0, 0, 0, 1, 1, 0]], -np.inf, 0.999)
        
        t0 = time.perf_counter()
        
        if solver == "slsqp":
            res = minimize(
                self.objective, x0, method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": lambda x: 0.999 - x[4] - x[5]},
                options={"maxiter": 2000, "ftol": 1e-12}
            )
        else:
            from scipy.optimize import Bounds
            lb = np.array([-np.inf, -0.999, -0.999, 1e-10, 1e-10, 1e-10, 2.01])
            ub = np.array([np.inf, 0.999, 0.999, np.inf, 0.999, 0.999, 100.0])
            res = minimize(
                self.objective, x0, method="L-BFGS-B",
                bounds=Bounds(lb, ub),
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        
        t_elapsed = time.perf_counter() - t0
        
        c, phi, theta_ma, omega, alpha, beta, nu = res.x
        
        self.objective(res.x)  # Compute final states
        
        n_eff = self.n - 1
        ll = -res.fun * n_eff
        
        return ARMAGARCHStudentTResult(
            c=c, phi=phi, theta=theta_ma,
            omega=omega, alpha=alpha, beta=beta, nu=nu,
            log_likelihood=ll, n_obs=self.n,
            converged=res.success, time_elapsed=t_elapsed,
            resid=self._resid.copy(), sigma2=self._sigma2.copy()
        )


@dataclass
class ARMAGARCHStudentTResult:
    """Result for ARMA-GARCH with Student-t."""
    c: float
    phi: float
    theta: float
    omega: float
    alpha: float
    beta: float
    nu: float
    log_likelihood: float
    n_obs: int
    converged: bool
    time_elapsed: float
    resid: NDArray[np.float64]
    sigma2: NDArray[np.float64]
    
    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


# =============================================================================
# ARMA(1,1)-GARCH(1,1) WITH HANSEN SKEW-T
# =============================================================================

@njit(cache=True)
def _forward_recursion_11_skewt(
    y: np.ndarray,
    c: float,
    phi: float,
    theta_ma: float,
    omega: float,
    alpha: float,
    beta: float,
    nu: float,
    lam: float,
    h0: float,
    e0: float,
    resid: np.ndarray,
    sigma2: np.ndarray,
) -> Tuple[float, bool]:
    """Forward recursion for ARMA(1,1)-GARCH(1,1) with Hansen Skew-t."""
    n = len(y)
    n_eff = n - 1
    
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10, False
    if nu <= 2 or abs(lam) >= 1:
        return 1e10, False
    
    resid[0] = e0
    sigma2[0] = h0
    
    if sigma2[0] < H_FLOOR:
        return 1e10, False
    
    sum_nll = 0.0
    
    for t in range(1, n):
        e_prev = resid[t - 1]
        h_prev = sigma2[t - 1]
        
        resid[t] = y[t] - c - phi * y[t - 1] - theta_ma * e_prev
        sigma2[t] = omega + alpha * e_prev * e_prev + beta * h_prev
        
        if sigma2[t] < H_FLOOR or not np.isfinite(sigma2[t]):
            return 1e10, False
        
        sum_nll += _skewt_nll(resid[t], sigma2[t], nu, lam)
    
    return sum_nll / n_eff, True


class ARMA11GARCH11SkewT:
    """
    ARMA(1,1)-GARCH(1,1) with Hansen Skew-t innovations.
    
    Parameters: [c, φ, θ, ω, α, β, ν, λ]
    """
    
    def __init__(self, y: NDArray[np.float64]):
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.n = len(y)
        self.K = 8  # 6 ARMA-GARCH params + nu + lambda
        
        self._resid = np.zeros(self.n, dtype=np.float64)
        self._sigma2 = np.zeros(self.n, dtype=np.float64)
        
        self._e0 = 0.0
        self._h0 = np.mean(y ** 2)
    
    def objective(self, params: NDArray[np.float64]) -> float:
        c, phi, theta_ma, omega, alpha, beta, nu, lam = params
        nll, valid = _forward_recursion_11_skewt(
            self.y, c, phi, theta_ma, omega, alpha, beta, nu, lam,
            self._h0, self._e0, self._resid, self._sigma2
        )
        return nll if valid else 1e10
    
    def fit(
        self,
        x0: Optional[NDArray[np.float64]] = None,
        solver: str = "slsqp",
        verbose: bool = True,
    ) -> 'ARMAGARCHSkewTResult':
        
        if x0 is None:
            var_y = np.var(self.y)
            x0 = np.array([np.mean(self.y), 0.0, 0.0, var_y * 0.02, 0.1, 0.85, 8.0, 0.0])
        
        bounds = [
            (None, None),      # c
            (-0.999, 0.999),   # phi
            (-0.999, 0.999),   # theta
            (1e-10, None),     # omega
            (1e-10, 0.999),    # alpha
            (1e-10, 0.999),    # beta
            (2.01, 100.0),     # nu
            (-0.999, 0.999),   # lambda
        ]
        
        t0 = time.perf_counter()
        
        res = minimize(
            self.objective, x0, method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": lambda x: 0.999 - x[4] - x[5]},
            options={"maxiter": 2000, "ftol": 1e-12}
        )
        
        t_elapsed = time.perf_counter() - t0
        
        c, phi, theta_ma, omega, alpha, beta, nu, lam = res.x
        
        self.objective(res.x)
        
        n_eff = self.n - 1
        ll = -res.fun * n_eff
        
        return ARMAGARCHSkewTResult(
            c=c, phi=phi, theta=theta_ma,
            omega=omega, alpha=alpha, beta=beta, nu=nu, lam=lam,
            log_likelihood=ll, n_obs=self.n,
            converged=res.success, time_elapsed=t_elapsed,
            resid=self._resid.copy(), sigma2=self._sigma2.copy()
        )


@dataclass
class ARMAGARCHSkewTResult:
    """Result for ARMA-GARCH with Skew-t."""
    c: float
    phi: float
    theta: float
    omega: float
    alpha: float
    beta: float
    nu: float
    lam: float
    log_likelihood: float
    n_obs: int
    converged: bool
    time_elapsed: float
    resid: NDArray[np.float64]
    sigma2: NDArray[np.float64]
    
    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


# =============================================================================
# GENERAL ARMA(p,q)-GARCH(P,Q) IMPLEMENTATION
# =============================================================================

@njit(cache=True)
def _arma_garch_forward_pq(
    y: np.ndarray,
    c: float,
    phi: np.ndarray,      # AR coefficients [φ₁, ..., φₚ]
    theta: np.ndarray,    # MA coefficients [θ₁, ..., θ_q]
    omega: float,
    alpha: np.ndarray,    # ARCH coefficients [α₁, ..., αₚ]
    beta: np.ndarray,     # GARCH coefficients [β₁, ..., β_Q]
    h0: float,
    e0: np.ndarray,       # Initial residuals [e₀, e₋₁, ...]
    h0_vec: np.ndarray,   # Initial variances [h₀, h₋₁, ...]
    resid: np.ndarray,
    sigma2: np.ndarray,
    dist: int,            # 0=Normal, 1=Student-t, 2=Skew-t
    nu: float,
    lam: float,
) -> Tuple[float, bool]:
    """
    Forward recursion for ARMA(p,q)-GARCH(P,Q).
    
    Mean: y_t = c + Σᵢ φᵢ y_{t-i} + Σⱼ θⱼ e_{t-j} + e_t
    Var:  h_t = ω + Σᵢ αᵢ e²_{t-i} + Σⱼ βⱼ h_{t-j}
    """
    n = len(y)
    p_ar = len(phi)
    q_ma = len(theta)
    P_arch = len(alpha)
    Q_garch = len(beta)
    
    max_lag = max(p_ar, q_ma, P_arch, Q_garch)
    n_eff = n - max_lag
    
    if n_eff <= 0:
        return 1e10, False
    
    if omega <= 0:
        return 1e10, False
    
    # Check GARCH stationarity
    alpha_sum = 0.0
    for i in range(P_arch):
        if alpha[i] < 0:
            return 1e10, False
        alpha_sum += alpha[i]
    
    beta_sum = 0.0
    for j in range(Q_garch):
        if beta[j] < 0:
            return 1e10, False
        beta_sum += beta[j]
    
    if alpha_sum + beta_sum >= 1.0:
        return 1e10, False
    
    # Initialize pre-sample
    for i in range(max_lag):
        resid[i] = e0[i] if i < len(e0) else 0.0
        sigma2[i] = h0_vec[i] if i < len(h0_vec) else h0
    
    sum_nll = 0.0
    
    for t in range(max_lag, n):
        # ARMA residual
        mu_t = c
        for i in range(p_ar):
            if t - 1 - i >= 0:
                mu_t += phi[i] * y[t - 1 - i]
        for j in range(q_ma):
            if t - 1 - j >= 0:
                mu_t += theta[j] * resid[t - 1 - j]
        
        resid[t] = y[t] - mu_t
        
        # GARCH variance
        h_t = omega
        for i in range(P_arch):
            if t - 1 - i >= 0:
                e_lag = resid[t - 1 - i]
                h_t += alpha[i] * e_lag * e_lag
        for j in range(Q_garch):
            if t - 1 - j >= 0:
                h_t += beta[j] * sigma2[t - 1 - j]
        
        sigma2[t] = h_t
        
        if h_t < H_FLOOR or not np.isfinite(h_t):
            return 1e10, False
        
        # Log-likelihood
        if dist == 0:  # Normal
            sum_nll += _normal_nll(resid[t], h_t)
        elif dist == 1:  # Student-t
            sum_nll += _studentt_nll(resid[t], h_t, nu)
        else:  # Skew-t
            sum_nll += _skewt_nll(resid[t], h_t, nu, lam)
    
    return sum_nll / n_eff, True


class ARMApqGARCHPQ:
    """
    General ARMA(p,q)-GARCH(P,Q) model.
    
    Distributions: 'normal', 'studentt', 'skewt'
    
    Parameters depend on distribution:
        Normal:   [c, φ₁..φₚ, θ₁..θ_q, ω, α₁..αₚ, β₁..β_Q]
        Student-t: + [ν]
        Skew-t:    + [ν, λ]
    """
    
    def __init__(
        self,
        y: NDArray[np.float64],
        p: int = 1,
        q: int = 1,
        P: int = 1,
        Q: int = 1,
        dist: str = "normal",
    ):
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.n = len(y)
        self.p = p  # AR order
        self.q = q  # MA order
        self.P = P  # ARCH order
        self.Q = Q  # GARCH order
        self.dist = dist
        
        # Distribution code: 0=Normal, 1=Student-t, 2=Skew-t
        self._dist_code = {"normal": 0, "studentt": 1, "skewt": 2}[dist]
        
        # Number of parameters
        self.K = 1 + p + q + 1 + P + Q  # c + AR + MA + omega + ARCH + GARCH
        if dist == "studentt":
            self.K += 1  # nu
        elif dist == "skewt":
            self.K += 2  # nu, lambda
        
        self._resid = np.zeros(self.n, dtype=np.float64)
        self._sigma2 = np.zeros(self.n, dtype=np.float64)
        
        self._h0 = np.mean(y ** 2)
        self._max_lag = max(p, q, P, Q)
        self._e0 = np.zeros(self._max_lag, dtype=np.float64)
        self._h0_vec = np.full(self._max_lag, self._h0, dtype=np.float64)
    
    def _unpack_params(self, params: NDArray[np.float64]):
        """Unpack flat parameter array."""
        idx = 0
        c = params[idx]; idx += 1
        phi = params[idx:idx+self.p]; idx += self.p
        theta = params[idx:idx+self.q]; idx += self.q
        omega = params[idx]; idx += 1
        alpha = params[idx:idx+self.P]; idx += self.P
        beta = params[idx:idx+self.Q]; idx += self.Q
        
        nu = 10.0
        lam = 0.0
        if self.dist == "studentt":
            nu = params[idx]; idx += 1
        elif self.dist == "skewt":
            nu = params[idx]; idx += 1
            lam = params[idx]; idx += 1
        
        return c, phi, theta, omega, alpha, beta, nu, lam
    
    def objective(self, params: NDArray[np.float64]) -> float:
        c, phi, theta, omega, alpha, beta, nu, lam = self._unpack_params(params)
        
        nll, valid = _arma_garch_forward_pq(
            self.y, c, phi, theta, omega, alpha, beta,
            self._h0, self._e0, self._h0_vec,
            self._resid, self._sigma2,
            self._dist_code, nu, lam
        )
        return nll if valid else 1e10
    
    def fit(
        self,
        x0: Optional[NDArray[np.float64]] = None,
        solver: str = "slsqp",
        verbose: bool = True,
    ) -> 'ARMApqGARCHPQResult':
        
        if x0 is None:
            var_y = np.var(self.y)
            x0 = [np.mean(self.y)]  # c
            x0.extend([0.0] * self.p)  # phi
            x0.extend([0.0] * self.q)  # theta
            x0.append(var_y * 0.02)   # omega
            x0.extend([0.05 / self.P] * self.P)  # alpha (split evenly)
            x0.extend([0.90 / self.Q] * self.Q)  # beta (split evenly)
            if self.dist == "studentt":
                x0.append(8.0)
            elif self.dist == "skewt":
                x0.extend([8.0, 0.0])
            x0 = np.array(x0)
        
        # Build bounds
        bounds = [(None, None)]  # c
        bounds.extend([(-0.999, 0.999)] * self.p)  # phi
        bounds.extend([(-0.999, 0.999)] * self.q)  # theta
        bounds.append((1e-10, None))  # omega
        bounds.extend([(1e-10, 0.999)] * self.P)  # alpha
        bounds.extend([(1e-10, 0.999)] * self.Q)  # beta
        if self.dist == "studentt":
            bounds.append((2.01, 100.0))  # nu
        elif self.dist == "skewt":
            bounds.append((2.01, 100.0))  # nu
            bounds.append((-0.999, 0.999))  # lambda
        
        # GARCH constraint: sum(alpha) + sum(beta) < 1
        def garch_constraint(x):
            _, _, _, _, alpha, beta, _, _ = self._unpack_params(x)
            return 0.999 - np.sum(alpha) - np.sum(beta)
        
        t0 = time.perf_counter()
        
        res = minimize(
            self.objective, x0, method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": garch_constraint},
            options={"maxiter": 2000, "ftol": 1e-12}
        )
        
        t_elapsed = time.perf_counter() - t0
        
        c, phi, theta, omega, alpha, beta, nu, lam = self._unpack_params(res.x)
        
        self.objective(res.x)
        
        n_eff = self.n - self._max_lag
        ll = -res.fun * n_eff
        
        return ARMApqGARCHPQResult(
            c=c, phi=phi.copy(), theta=theta.copy(),
            omega=omega, alpha=alpha.copy(), beta=beta.copy(),
            nu=nu if self.dist != "normal" else None,
            lam=lam if self.dist == "skewt" else None,
            dist=self.dist,
            log_likelihood=ll, n_obs=self.n, n_eff=n_eff,
            p=self.p, q=self.q, P=self.P, Q=self.Q,
            converged=res.success, time_elapsed=t_elapsed,
            resid=self._resid.copy(), sigma2=self._sigma2.copy()
        )


@dataclass
class ARMApqGARCHPQResult:
    """Result for general ARMA(p,q)-GARCH(P,Q)."""
    c: float
    phi: NDArray[np.float64]
    theta: NDArray[np.float64]
    omega: float
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    nu: Optional[float]
    lam: Optional[float]
    dist: str
    log_likelihood: float
    n_obs: int
    n_eff: int
    p: int
    q: int
    P: int
    Q: int
    converged: bool
    time_elapsed: float
    resid: NDArray[np.float64]
    sigma2: NDArray[np.float64]
    
    @property
    def persistence(self) -> float:
        return np.sum(self.alpha) + np.sum(self.beta)
    
    @property
    def aic(self) -> float:
        k = 1 + self.p + self.q + 1 + self.P + self.Q
        if self.nu is not None:
            k += 1
        if self.lam is not None:
            k += 1
        return 2 * k - 2 * self.log_likelihood
    
    @property
    def bic(self) -> float:
        k = 1 + self.p + self.q + 1 + self.P + self.Q
        if self.nu is not None:
            k += 1
        if self.lam is not None:
            k += 1
        return k * np.log(self.n_eff) - 2 * self.log_likelihood


# =============================================================================
# LOG-TRANSFORMED (UNCONSTRAINED) PARAMETRIZATIONS
# =============================================================================

class GARCH11NormalLog:
    """
    GARCH(1,1) with log-transformed (unconstrained) parameters.
    
    Transformations:
        ω       = exp(log_ω)
        (α, β)  = softmax(η_α, η_β)
    
    Unconstrained params: [log_ω, η_α, η_β]
    """
    
    def __init__(self, eps: NDArray[np.float64]):
        self._base = GARCH11Normal(eps)
        self.n = self._base.n
        self.K = 3
    
    def _transform(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform unconstrained to constrained params."""
        log_omega, eta_alpha, eta_beta = u
        
        omega = np.exp(log_omega)
        alpha, beta = _softmax_garch(eta_alpha, eta_beta)
        
        return np.array([omega, alpha, beta])
    
    def _inverse_transform(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constrained to unconstrained params."""
        omega, alpha, beta = p
        
        log_omega = np.log(omega)
        
        margin = 0.001
        remainder = 1.0 - margin - alpha - beta
        if remainder < 1e-10:
            remainder = 1e-10
        eta_alpha = np.log(alpha / (1 - margin)) - np.log(remainder / (1 - margin))
        eta_beta = np.log(beta / (1 - margin)) - np.log(remainder / (1 - margin))
        
        return np.array([log_omega, eta_alpha, eta_beta])
    
    def objective(self, u: NDArray[np.float64]) -> float:
        """NLL in unconstrained space."""
        p = self._transform(u)
        return self._base.objective(p)
    
    def gradient(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gradient in unconstrained space using chain rule."""
        p = self._transform(u)
        grad_p = self._base.gradient(p)
        
        log_omega, eta_alpha, eta_beta = u
        
        omega = np.exp(log_omega)
        da_da, da_db, db_da, db_db = _softmax_garch_jac(eta_alpha, eta_beta)
        
        grad_u = np.zeros(3)
        grad_u[0] = grad_p[0] * omega  # omega
        grad_u[1] = grad_p[1] * da_da + grad_p[2] * db_da  # eta_alpha
        grad_u[2] = grad_p[1] * da_db + grad_p[2] * db_db  # eta_beta
        
        return grad_u
    
    def fit(
        self,
        x0: Optional[NDArray[np.float64]] = None,
        solver: str = "bfgs",
        verbose: bool = True,
    ) -> GARCH11Result:
        """Fit using unconstrained optimization."""
        
        if x0 is None:
            var_eps = np.var(self._base.eps)
            p0 = np.array([var_eps * 0.02, 0.1, 0.85])
            u0 = self._inverse_transform(p0)
        else:
            u0 = x0
        
        t0 = time.perf_counter()
        
        if solver == "bfgs":
            res = minimize(
                self.objective, u0, method="BFGS",
                jac=self.gradient,
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        elif solver == "bfgs-nograd":
            res = minimize(
                self.objective, u0, method="BFGS",
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        else:
            res = minimize(
                self.objective, u0, method="BFGS",
                jac=self.gradient,
                options={"maxiter": 2000, "gtol": 1e-8}
            )
        
        t_elapsed = time.perf_counter() - t0
        
        p = self._transform(res.x)
        omega, alpha, beta = p
        
        self._base._compute_states(p)
        
        n_eff = self.n - 1
        ll = -res.fun * n_eff - 0.5 * n_eff * np.log(2 * np.pi)
        aic = 2 * self.K - 2 * ll
        bic = self.K * np.log(n_eff) - 2 * ll
        
        return GARCH11Result(
            omega=omega, alpha=alpha, beta=beta,
            log_likelihood=ll, aic=aic, bic=bic,
            n_obs=self.n, converged=res.success,
            time_elapsed=t_elapsed,
            sigma2=self._base._sigma2.copy()
        )


def verify_log_gradient(model, u: NDArray[np.float64], eps: float = 1e-7) -> Tuple[NDArray, NDArray, NDArray]:
    """Verify gradient of log-transformed model via finite differences."""
    grad_ana = model.gradient(u)
    
    K = len(u)
    grad_num = np.zeros(K)
    
    for k in range(K):
        u_plus = u.copy()
        u_minus = u.copy()
        h = eps * max(1.0, abs(u[k]))
        u_plus[k] += h
        u_minus[k] -= h
        grad_num[k] = (model.objective(u_plus) - model.objective(u_minus)) / (2 * h)
    
    # Relative error
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = np.abs(grad_ana - grad_num) / (np.abs(grad_num) + 1e-12)
        rel_err = np.where(np.isfinite(rel_err), rel_err, 0.0)
    
    return grad_ana, grad_num, rel_err


# =============================================================================
# MAIN: TEST AND COMPARE
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 80)
    print("ARMA(1,1)-GARCH(1,1) ESTIMATOR TEST")
    print("=" * 80)
    
    # Load test data
    print("\nLoading data...")
    data_raw = pd.read_csv("data/DATA_HW1.csv", skiprows=1, parse_dates=["DATE"], dayfirst=True)
    data = data_raw.rename(columns={
        "S&PCOMP(RI)": "stock",
        "SPUHYBD(RI)": "cbond",
    }).set_index("DATE")[["stock", "cbond"]]
    
    lr = np.log1p(data.pct_change(fill_method=None))
    lr = lr.dropna()  # Remove NaN from first row
    mask_zero = (lr == 0).any(axis=1)
    lr = lr[~mask_zero]
    y = np.asarray(lr['stock'].values, dtype=np.float64)
    
    print(f"Data: {len(y)} observations")
    
    # Create model
    print("\nCreating ARMA(1,1)-GARCH(1,1) model...")
    model = ARMA11GARCH11Normal(y)
    
    # Test parameters
    var_y = np.var(y)
    print(f"Sample variance: {var_y:.6e}")
    print(f"Sample mean: {np.mean(y):.6e}")
    
    test_params = np.array([
        0.0,                    # c (close to zero for returns)
        0.05,                   # φ
        0.05,                   # θ_ma
        var_y * 0.05,           # ω
        0.10,                   # α
        0.80,                   # β (make sure α + β < 1)
    ], dtype=np.float64)
    
    print(f"Test params: {test_params}")
    print(f"α + β = {test_params[4] + test_params[5]:.3f}")
    
    # =========================================================================
    # VERIFY GRADIENT
    # =========================================================================
    print("\n" + "-" * 80)
    print("GRADIENT VERIFICATION")
    print("-" * 80)
    
    grad_ana, grad_num, grad_err = verify_gradient_numerically(model, test_params)
    
    param_names = ['c', 'φ', 'θ_ma', 'ω', 'α', 'β']
    print(f"\n{'Param':>8}  {'Analytical':>14}  {'Numerical':>14}  {'Rel Err':>12}  Status")
    print(f"{'-'*8}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*6}")
    
    for i, name in enumerate(param_names):
        status = "✓" if grad_err[i] < 1e-4 else "✗"
        print(f"{name:>8}  {grad_ana[i]:>14.6e}  {grad_num[i]:>14.6e}  {grad_err[i]:>12.2e}  {status}")
    
    # =========================================================================
    # VERIFY HESSIAN
    # =========================================================================
    print("\n" + "-" * 80)
    print("HESSIAN VERIFICATION")
    print("-" * 80)
    
    hess_ana, hess_num, hess_err = verify_hessian_numerically(model, test_params)
    
    print("\nHessian relative errors (max by row):")
    for i, name in enumerate(param_names):
        max_err = np.max(hess_err[i, :])
        status = "✓" if max_err < 1e-3 else "✗"
        print(f"  {name}: max_err = {max_err:.2e} {status}")
    
    print(f"\nOverall max Hessian error: {np.max(hess_err):.2e}")
    
    # =========================================================================
    # FIT WITH DIFFERENT SOLVERS
    # =========================================================================
    print("\n" + "-" * 80)
    print("FITTING WITH DIFFERENT SOLVERS")
    print("-" * 80)
    
    results = {}
    
    for solver in ["slsqp", "slsqp-nograd"]:
        print(f"\nFitting with {solver}...")
        try:
            res = model.fit(solver=solver, verbose=False)
            results[solver] = res
            print(f"  Converged: {res.converged}")
            print(f"  Log-lik: {res.log_likelihood:.4f}")
            print(f"  Time: {res.time_elapsed*1000:.1f} ms")
            print(f"  Iterations: {res.n_iter}")
        except Exception as e:
            print(f"  FAILED: {e}")
    
    # =========================================================================
    # COMPARE RESULTS
    # =========================================================================
    print("\n" + "-" * 80)
    print("PARAMETER COMPARISON")
    print("-" * 80)
    
    print(f"\n{'Param':>8}", end="")
    for solver in results:
        print(f"  {solver:>14}", end="")
    print()
    print("-" * (8 + 16 * len(results)))
    
    for i, name in enumerate(param_names):
        print(f"{name:>8}", end="")
        for solver, res in results.items():
            val = [res.c, res.phi[0], res.theta[0], res.omega, res.alpha[0], res.beta[0]][i]
            print(f"  {val:>14.6e}", end="")
        print()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    
    # Find best result by log-likelihood
    best_solver = max(results, key=lambda s: results[s].log_likelihood)
    res = results[best_solver]
    
    print(f"\nBest result ({best_solver}):")
    print(f"  c      = {res.c:.6f}")
    print(f"  φ      = {res.phi[0]:.6f}")
    print(f"  θ_ma   = {res.theta[0]:.6f}")
    print(f"  ω      = {res.omega:.6e}")
    print(f"  α      = {res.alpha[0]:.6f}")
    print(f"  β      = {res.beta[0]:.6f}")
    print(f"  α + β  = {res.persistence:.6f}")
    print(f"  AIC    = {res.aic:.2f}")
    print(f"  BIC    = {res.bic:.2f}")
    
    # Speed comparison
    if "slsqp" in results and "slsqp-nograd" in results:
        speedup = results["slsqp-nograd"].time_elapsed / results["slsqp"].time_elapsed
        print(f"\n  Speedup from analytical gradient: {speedup:.1f}x")
    
    # =========================================================================
    # LINKED VS UNLINKED ESTIMATION COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("LINKED vs UNLINKED ESTIMATION COMPARISON")
    print("=" * 80)
    
    # LINKED: Joint ARMA(1,1)-GARCH(1,1) estimation
    print("\n1. LINKED (Joint ARMA-GARCH):")
    print("-" * 40)
    linked_model = ARMA11GARCH11Normal(y)
    linked_res = linked_model.fit(solver="slsqp", verbose=False)
    
    print(f"   c      = {linked_res.c:.6f}")
    print(f"   φ      = {linked_res.phi[0]:.6f}")
    print(f"   θ_ma   = {linked_res.theta[0]:.6f}")
    print(f"   ω      = {linked_res.omega:.6e}")
    print(f"   α      = {linked_res.alpha[0]:.6f}")
    print(f"   β      = {linked_res.beta[0]:.6f}")
    print(f"   LL     = {linked_res.log_likelihood:.2f}")
    print(f"   Time   = {linked_res.time_elapsed*1000:.0f}ms")
    
    # UNLINKED: First ARMA(1,1), then GARCH(1,1) on residuals
    print("\n2. UNLINKED (Sequential ARMA → GARCH):")
    print("-" * 40)
    
    # Step 1: Fit ARMA(1,1)
    t0 = time.perf_counter()
    arma_model = ARMA11Normal(y)
    arma_res = arma_model.fit(solver="slsqp", verbose=False)
    
    print(f"   ARMA(1,1) step:")
    print(f"     c      = {arma_res.c:.6f}")
    print(f"     φ      = {arma_res.phi:.6f}")
    print(f"     θ_ma   = {arma_res.theta:.6f}")
    print(f"     σ²     = {arma_res.sigma2:.6e}")
    print(f"     LL     = {arma_res.log_likelihood:.2f}")
    
    # Step 2: Fit GARCH(1,1) on ARMA residuals
    garch_model = GARCH11Normal(arma_res.resid)
    garch_res = garch_model.fit(solver="slsqp", verbose=False)
    t_unlinked = time.perf_counter() - t0
    
    print(f"\n   GARCH(1,1) step (on ARMA residuals):")
    print(f"     ω      = {garch_res.omega:.6e}")
    print(f"     α      = {garch_res.alpha:.6f}")
    print(f"     β      = {garch_res.beta:.6f}")
    print(f"     LL     = {garch_res.log_likelihood:.2f}")
    
    # Combined log-likelihood for unlinked
    # Note: ARMA LL uses constant σ², GARCH LL uses time-varying h_t
    # The proper comparison is GARCH LL since both model variance
    unlinked_ll = garch_res.log_likelihood
    
    print(f"\n   Combined (GARCH LL): {unlinked_ll:.2f}")
    print(f"   Time   = {t_unlinked*1000:.0f}ms")
    
    # COMPARISON
    print("\n" + "-" * 40)
    print("COMPARISON:")
    print("-" * 40)
    print(f"{'':>20} {'Linked':>12} {'Unlinked':>12} {'Diff':>12}")
    print("-" * 56)
    print(f"{'φ':>20} {linked_res.phi[0]:>12.4f} {arma_res.phi:>12.4f} {linked_res.phi[0]-arma_res.phi:>12.4f}")
    print(f"{'θ_ma':>20} {linked_res.theta[0]:>12.4f} {arma_res.theta:>12.4f} {linked_res.theta[0]-arma_res.theta:>12.4f}")
    print(f"{'ω':>20} {linked_res.omega:>12.2e} {garch_res.omega:>12.2e} {linked_res.omega-garch_res.omega:>12.2e}")
    print(f"{'α':>20} {linked_res.alpha[0]:>12.4f} {garch_res.alpha:>12.4f} {linked_res.alpha[0]-garch_res.alpha:>12.4f}")
    print(f"{'β':>20} {linked_res.beta[0]:>12.4f} {garch_res.beta:>12.4f} {linked_res.beta[0]-garch_res.beta:>12.4f}")
    print(f"{'Log-Likelihood':>20} {linked_res.log_likelihood:>12.2f} {unlinked_ll:>12.2f} {linked_res.log_likelihood-unlinked_ll:>12.2f}")
    
    ll_diff = linked_res.log_likelihood - unlinked_ll
    if ll_diff > 0:
        print(f"\n   → Linked estimation achieves {ll_diff:.2f} higher LL")
        print(f"   → This demonstrates the value of joint optimization")
    else:
        print(f"\n   → Unlinked estimation achieves {-ll_diff:.2f} higher LL")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
