"""
GARCH Estimation Framework - Pure Python Implementation
========================================================

A single-file GARCH estimator extracted from the volkit library.
All optimization settings, bounds, and constants are defined at the top.
Likelihood functions are imported from the likelihoods module.

Usage:
    from garch_estimator import fit_garch
    
    # Simple usage - everything is handled automatically
    result = fit_garch(residuals, dist="normal")
    result = fit_garch(residuals, dist="studentt")
    result = fit_garch(residuals, dist="normal", method="qmle")  # robust SEs
    
    # With log-space optimization (Normal only)
    result = fit_garch(residuals, solver="trust-exact", use_logspace=True)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize
from scipy.special import logsumexp

# Optional Numba JIT compilation for performance
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else args[0]

# Import likelihood functions from likelihoods module
from likelihoods import (
    # Simple log-likelihood functions
    normal_loglik,          # (resid, sigma2) -> float
    studentt_loglik,        # (resid, sigma2, nu) -> float
    skewt_loglik,           # (resid, sigma2, nu, lam) -> float
    # GARCH(1,1) + Normal: analytical derivatives
    garch11_normal_gradient,
    garch11_normal_hessian,
    garch11_normal_robust_se,
    # GARCH(1,1) + Student-t: analytical derivatives
    garch11_studentt_gradient,
    garch11_studentt_hessian,
    garch11_studentt_robust_se,
    # Note: Skew-t uses numerical derivatives (no analytical available)
)

# Import numerical Hessian computation (reparameterized, boundary-safe)
from numerical_hessians import (
    compute_numerical_hessian,
    compute_hessian_unconstrained,
    compute_robust_hessian_normal,
    compute_robust_hessian_studentt,
    compute_robust_hessian_skewt,
)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Parameter Bounds
# ─────────────────────────────────────────────────────────────────────────────

OMEGA_BOUNDS = (1e-8, 1.0)           # Intercept term
ALPHA_BOUNDS = (1e-8, 0.99)          # ARCH coefficient(s)
BETA_BOUNDS  = (1e-8, 0.99)          # GARCH coefficient(s)
NU_BOUNDS    = (2.01, 100.0)         # Student-t degrees of freedom (must be > 2 for finite variance)
LAMBDA_BOUNDS = (-0.99, 0.99)        # Skew-t asymmetry parameter

# ─────────────────────────────────────────────────────────────────────────────
# Initial Parameter Values
# ─────────────────────────────────────────────────────────────────────────────

OMEGA_INIT = 1e-8                    # Will be scaled by sample variance
PERSISTENCE_TARGET = 0.90            # Target α + β for initialization
BETA_SHARE = 0.80                    # Fraction of persistence allocated to β
NU_INIT = 10.0                       # Initial degrees of freedom for Student-t
LAMBDA_INIT = 0.0                    # Initial skewness (symmetric)

# ─────────────────────────────────────────────────────────────────────────────
# Stationarity Constraint
# ─────────────────────────────────────────────────────────────────────────────

STATIONARITY_LOWER = 1e-12           # Lower bound for α + β
STATIONARITY_UPPER = 1.0 - 1e-8      # Upper bound for α + β (strict inequality)
STATIONARITY_PENALTY = 1e10          # Penalty for violating stationarity in unconstrained opt

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer Settings: Nelder-Mead (derivative-free)
# ─────────────────────────────────────────────────────────────────────────────

NELDER_MEAD_CONFIG = {
    "method": "Nelder-Mead",
    "tol": 1e-12,
    "options": {
        "maxfev": 50_000,
        "maxiter": 5_000,
        "xatol": 1e-8,
        "fatol": 1e-12,
        "adaptive": True,
        "disp": False,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer Settings: SLSQP (gradient-based with constraints)
# ─────────────────────────────────────────────────────────────────────────────

SLSQP_CONFIG = {
    "method": "SLSQP",
    "tol": 1e-12,
    "options": {
        "maxiter": 5_000,
        "ftol": 1e-16,
        "disp": False,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer Settings: Trust-Region Constrained (second-order, most robust)
# ─────────────────────────────────────────────────────────────────────────────

TRUST_CONSTR_CONFIG = {
    "method": "trust-constr",
    "tol": 1e-12,
    "options": {
        "maxiter": 5_000,
        "xtol": 1e-8,
        "gtol": 1e-8,
        "initial_tr_radius": 0.1,    # Will be adjusted based on (p,q)
        "disp": False,
        "verbose": 0,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer Settings: Trust-Exact (for unconstrained log-space optimization)
# ─────────────────────────────────────────────────────────────────────────────

TRUST_EXACT_CONFIG = {
    "method": "trust-exact",
    "tol": 1e-12,
    "options": {
        "maxiter": 5_000,
        "disp": False,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Numerical Stability
# ─────────────────────────────────────────────────────────────────────────────

LOG_CLIP_MIN = -700.0                # Prevent exp underflow
LOG_CLIP_MAX = 700.0                 # Prevent exp overflow
VARIANCE_FLOOR = 1e-12               # Minimum variance to prevent division by zero
OBJECTIVE_SCALE = 2.0                # Scaling factor for log-space optimization


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GARCHParams:
    """Container for GARCH(p,q) parameters."""
    omega: float
    alpha: NDArray[np.float64]  # length p
    beta: NDArray[np.float64]   # length q
    
    @property
    def persistence(self) -> float:
        """Sum of alpha and beta coefficients."""
        return float(np.sum(self.alpha) + np.sum(self.beta))
    
    @property
    def is_stationary(self) -> bool:
        """Check if persistence < 1."""
        return self.persistence < 1.0
    
    @property
    def unconditional_variance(self) -> float:
        """Long-run variance (only valid if stationary)."""
        if not self.is_stationary:
            return np.inf
        return self.omega / (1.0 - self.persistence)
    
    def to_array(self) -> NDArray[np.float64]:
        """Flatten to 1D array [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]."""
        return np.concatenate([[self.omega], self.alpha, self.beta])
    
    @classmethod
    def from_array(cls, arr: NDArray[np.float64], p: int, q: int) -> "GARCHParams":
        """Construct from flat array."""
        return cls(
            omega=arr[0],
            alpha=arr[1:1+p],
            beta=arr[1+p:1+p+q],
        )


@dataclass
class DistributionParams:
    """Container for distribution shape parameters."""
    nu: Optional[float] = None       # Student-t degrees of freedom
    lam: Optional[float] = None      # Skew-t asymmetry (lambda)
    
    def to_array(self) -> NDArray[np.float64]:
        """Flatten to 1D array."""
        parts = []
        if self.nu is not None:
            parts.append(self.nu)
        if self.lam is not None:
            parts.append(self.lam)
        return np.array(parts) if parts else np.array([])


@dataclass
class EstimationResult:
    """Container for estimation results."""
    garch_params: GARCHParams
    dist_params: DistributionParams
    log_likelihood: float
    n_obs: int
    converged: bool
    n_iter: int
    message: str
    
    # Estimation method
    method: str = "MLE"
    
    # Covariance / standard errors (computed post-estimation)
    hessian: Optional[NDArray[np.float64]] = None
    cov_matrix: Optional[NDArray[np.float64]] = None
    std_errors: Optional[NDArray[np.float64]] = None
    
    # Robust covariance / standard errors (for QMLE)
    opg: Optional[NDArray[np.float64]] = None               # Outer Product of Gradients
    cov_robust: Optional[NDArray[np.float64]] = None        # Sandwich covariance
    std_errors_robust: Optional[NDArray[np.float64]] = None # Robust standard errors
    
    # Conditional variances and standardized residuals
    sigma2: Optional[NDArray[np.float64]] = None
    std_resid: Optional[NDArray[np.float64]] = None
    
    # Timing
    time_elapsed: Optional[float] = None  # Estimation time in seconds
    
    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        k = len(self.garch_params.to_array()) + len(self.dist_params.to_array())
        return 2 * k - 2 * self.log_likelihood
    
    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        k = len(self.garch_params.to_array()) + len(self.dist_params.to_array())
        return k * np.log(self.n_obs) - 2 * self.log_likelihood


# =============================================================================
# VARIANCE COMPUTATION (PURE PYTHON)
# =============================================================================

def garch_variance(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    p: int,
    q: int,
) -> NDArray[np.float64]:
    """
    Compute GARCH(p,q) conditional variances.
    
    Parameters
    ----------
    params : array [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
    resid2 : array of squared residuals (length n)
    p : ARCH order (number of alpha coefficients)
    q : GARCH order (number of beta coefficients)
    
    Returns
    -------
    sigma2 : array of conditional variances (length n)
    """
    n = len(resid2)
    sigma2 = np.empty(n, dtype=np.float64)
    
    omega = params[0]
    alpha = params[1:1+p]
    beta = params[1+p:1+p+q]
    
    # Initialize with sample variance
    sigma2[0] = max(np.mean(resid2), VARIANCE_FLOOR)
    
    for t in range(1, n):
        sigma2[t] = omega
        
        # ARCH terms: α_i * ε²_{t-i}
        for i in range(p):
            if t - i - 1 >= 0:
                sigma2[t] += alpha[i] * resid2[t - i - 1]
        
        # GARCH terms: β_j * σ²_{t-j}
        for j in range(q):
            if t - j - 1 >= 0:
                sigma2[t] += beta[j] * sigma2[t - j - 1]
        
        # Floor to prevent numerical issues
        sigma2[t] = max(sigma2[t], VARIANCE_FLOOR)
    
    return sigma2


@njit(cache=True)
def _garch_variance_11_core(
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


def garch_variance_11(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Optimized GARCH(1,1) variance computation.
    
    Faster than the general version for the common (1,1) case.
    Uses Numba JIT compilation if available.
    """
    omega, alpha, beta = params[0], params[1], params[2]
    sigma2_init = np.mean(resid2)
    return _garch_variance_11_core(omega, alpha, beta, resid2, sigma2_init, VARIANCE_FLOOR)


# =============================================================================
# INITIAL PARAMETER GENERATION
# =============================================================================

def default_start_garch(
    resid: NDArray[np.float64],
    p: int = 1,
    q: int = 1,
) -> NDArray[np.float64]:
    """
    Generate sensible starting values for GARCH(p,q) parameters.
    
    Strategy:
    - omega: Small fraction of sample variance
    - alpha: Split (1-BETA_SHARE) * PERSISTENCE_TARGET evenly across p lags
    - beta: Split BETA_SHARE * PERSISTENCE_TARGET evenly across q lags
    """
    sample_var = np.var(resid)
    
    omega = max(OMEGA_INIT, 0.05 * sample_var * (1 - PERSISTENCE_TARGET))
    
    total_alpha = PERSISTENCE_TARGET * (1 - BETA_SHARE)
    total_beta = PERSISTENCE_TARGET * BETA_SHARE
    
    alpha = np.full(p, total_alpha / p) if p > 0 else np.array([])
    beta = np.full(q, total_beta / q) if q > 0 else np.array([])
    
    return np.concatenate([[omega], alpha, beta])


def default_start_studentt() -> NDArray[np.float64]:
    """Initial value for Student-t degrees of freedom."""
    return np.array([NU_INIT])


def default_start_skewt() -> NDArray[np.float64]:
    """Initial values for Skew-t parameters [nu, lambda]."""
    return np.array([NU_INIT, LAMBDA_INIT])


# =============================================================================
# BOUNDS GENERATION
# =============================================================================

def get_bounds_garch(p: int, q: int) -> List[Tuple[float, float]]:
    """Generate parameter bounds for GARCH(p,q)."""
    return (
        [OMEGA_BOUNDS] +
        [ALPHA_BOUNDS] * p +
        [BETA_BOUNDS] * q
    )


def get_bounds_studentt() -> List[Tuple[float, float]]:
    """Parameter bounds for Student-t."""
    return [NU_BOUNDS]


def get_bounds_skewt() -> List[Tuple[float, float]]:
    """Parameter bounds for Skew-t."""
    return [NU_BOUNDS, LAMBDA_BOUNDS]


# =============================================================================
# STATIONARITY CONSTRAINT
# =============================================================================

def build_stationarity_constraint(p: int, q: int) -> LinearConstraint:
    """
    Build linear constraint: α_1 + ... + α_p + β_1 + ... + β_q < 1
    
    Matrix form: A @ x where A = [0, 1, 1, ..., 1] (zeros for omega and shape params)
    """
    # [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
    A = np.array([[0.0] + [1.0] * p + [1.0] * q])
    return LinearConstraint(A, lb=STATIONARITY_LOWER, ub=STATIONARITY_UPPER)


def build_stationarity_constraint_with_shape(
    p: int,
    q: int,
    n_shape: int,
) -> LinearConstraint:
    """
    Build stationarity constraint for joint optimization with shape parameters.
    
    Shape parameters (nu, lambda) are not included in the constraint.
    """
    # [omega, alpha..., beta..., nu, lambda, ...]
    A = np.array([[0.0] + [1.0] * p + [1.0] * q + [0.0] * n_shape])
    return LinearConstraint(A, lb=STATIONARITY_LOWER, ub=STATIONARITY_UPPER)


# =============================================================================
# SOFTPLUS FUNCTIONS (for Student-t/Skew-t nu transformation)
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
# HANSEN'S λ_MAX FOR SKEW-T (ensures b² > 0)
# =============================================================================

def hansen_lambda_max(nu: float, safety: float = 0.999) -> float:
    """
    Compute maximum admissible |λ| for Hansen's skew-t given ν.
    
    Hansen's skew-t requires b² > 0 where:
        c = Γ((ν+1)/2) / [√(π(ν-2)) Γ(ν/2)]
        a = 4λc(ν-2)/(ν-1)
        b² = 1 + 3λ² - a²
    
    We use a slightly smaller value (safety < 1) to stay in the interior.
    """
    from scipy.special import gammaln
    
    if nu <= 2:
        return 0.0
    
    c = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi * (nu - 2))
    k = 4 * c * (nu - 2) / (nu - 1)
    k2 = k * k
    
    if k2 <= 3:
        return safety * 1.0
    else:
        lam_max = 1.0 / np.sqrt(k2 - 3)
        return safety * min(lam_max, 1.0)


# =============================================================================
# LOG-SPACE TRANSFORMATIONS - NORMAL (for unconstrained optimization)
# =============================================================================

def pack_params_logspace(z: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """
    Transform unconstrained parameters z to constrained θ.
    
    Uses softmax-like transformation to enforce:
    - omega > 0 via exp
    - alpha_i > 0, beta_j > 0
    - sum(alpha) + sum(beta) < 1 (STATIONARITY) via JOINT softmax
    
    The joint softmax ensures persistence < 1:
        [alpha, beta] = exp([z_alpha, z_beta]) / (1 + sum(exp([z_alpha, z_beta])))
    
    This is numerically stable using logsumexp.
    """
    z_omega = z[0]
    z_alpha = z[1:1+p]
    z_beta = z[1+p:1+p+q]
    
    # omega = exp(z_omega), clipped for stability
    omega = np.exp(np.clip(z_omega, LOG_CLIP_MIN, LOG_CLIP_MAX))
    
    # JOINT softmax over alpha AND beta to enforce stationarity
    # This ensures sum(alpha) + sum(beta) < 1
    if p > 0 or q > 0:
        z_persistence = np.concatenate([z_alpha, z_beta])
        lse_joint = logsumexp(z_persistence)
        log_denom = np.logaddexp(0.0, lse_joint)  # log(1 + sum(exp(z)))
        
        if p > 0:
            alpha = np.exp(z_alpha - log_denom)
        else:
            alpha = np.array([])
        
        if q > 0:
            beta = np.exp(z_beta - log_denom)
        else:
            beta = np.array([])
    else:
        alpha = np.array([])
        beta = np.array([])
    
    return np.concatenate([[omega], alpha, beta])


def unpack_params_logspace(theta: NDArray[np.float64], p: int, q: int) -> NDArray[np.float64]:
    """
    Transform constrained θ to unconstrained z.
    
    Inverse of pack_params_logspace (joint softmax).
    
    For joint softmax: param_i = exp(z_i) / (1 + sum(exp(z)))
    Inverting: z_i = log(param_i) - log(1 - sum(params))
    where sum(params) = sum(alpha) + sum(beta) = persistence
    """
    omega = theta[0]
    alpha = theta[1:1+p]
    beta = theta[1+p:1+p+q]
    
    z_omega = np.log(omega)
    
    # Joint inverse: z_i = log(param_i) - log(1 - persistence)
    persistence = np.sum(alpha) + np.sum(beta)
    log_remainder = np.log1p(-persistence)  # log(1 - persistence)
    
    if p > 0:
        z_alpha = np.log(alpha) - log_remainder
    else:
        z_alpha = np.array([])
    
    if q > 0:
        z_beta = np.log(beta) - log_remainder
    else:
        z_beta = np.array([])
    
    return np.concatenate([[z_omega], z_alpha, z_beta])


# =============================================================================
# LOG-SPACE TRANSFORMATIONS - STUDENT-T (4 params: omega, alpha, beta, nu)
# =============================================================================

def studentt_z_to_theta(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform Student-t GARCH parameters from unconstrained z to constrained θ.
    
    z = [z_ω, z_α, z_β, z_ν]
    θ = [ω, α, β, ν]
    
    Transformations:
        ω = exp(z_ω)
        [α, β] = softmax([z_α, z_β, 0])[:2]
        ν = 2 + softplus(z_ν)
    """
    z_omega, z_alpha, z_beta, z_nu = z[0], z[1], z[2], z[3]
    
    omega = np.exp(np.clip(z_omega, LOG_CLIP_MIN, LOG_CLIP_MAX))
    
    z_all = np.array([z_alpha, z_beta, 0.0])
    log_denom = logsumexp(z_all)
    alpha = np.exp(z_alpha - log_denom)
    beta = np.exp(z_beta - log_denom)
    
    nu = 2.0 + softplus(z_nu)
    
    return np.array([omega, alpha, beta, nu])


def studentt_theta_to_z(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse transformation: θ → z for Student-t."""
    omega, alpha, beta, nu = theta[0], theta[1], theta[2], theta[3]
    
    z_omega = np.log(omega)
    
    remainder = max(1.0 - alpha - beta, 1e-10)
    z_alpha = np.log(alpha / remainder)
    z_beta = np.log(beta / remainder)
    
    z_nu = softplus_inv(nu - 2.0)
    
    return np.array([z_omega, z_alpha, z_beta, z_nu])


def studentt_jacobian(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Jacobian J = ∂θ/∂z for Student-t transformation."""
    theta = studentt_z_to_theta(z)
    omega, alpha, beta = theta[0], theta[1], theta[2]
    
    J = np.zeros((4, 4), dtype=np.float64)
    
    J[0, 0] = omega
    J[1, 1] = alpha * (1.0 - alpha)
    J[1, 2] = -alpha * beta
    J[2, 1] = -beta * alpha
    J[2, 2] = beta * (1.0 - beta)
    J[3, 3] = softplus_deriv(z[3])
    
    return J


# =============================================================================
# LOG-SPACE TRANSFORMATIONS - SKEW-T (5 params: omega, alpha, beta, nu, lambda)
# =============================================================================

def skewt_z_to_theta(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform Skew-t GARCH parameters from unconstrained z to constrained θ.
    
    z = [z_ω, z_α, z_β, z_ν, z_λ]
    θ = [ω, α, β, ν, λ]
    
    Transformations:
        ω = exp(z_ω)
        [α, β] = softmax([z_α, z_β, 0])[:2]
        ν = 2 + softplus(z_ν)
        λ = λ_max(ν) * tanh(z_λ)
    """
    z_omega, z_alpha, z_beta, z_nu, z_lambda = z[0], z[1], z[2], z[3], z[4]
    
    omega = np.exp(np.clip(z_omega, LOG_CLIP_MIN, LOG_CLIP_MAX))
    
    z_all = np.array([z_alpha, z_beta, 0.0])
    log_denom = logsumexp(z_all)
    alpha = np.exp(z_alpha - log_denom)
    beta = np.exp(z_beta - log_denom)
    
    nu = 2.0 + softplus(z_nu)
    
    lam_max = hansen_lambda_max(nu)
    lam = lam_max * np.tanh(z_lambda)
    
    return np.array([omega, alpha, beta, nu, lam])


def skewt_theta_to_z(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse transformation: θ → z for Skew-t."""
    omega, alpha, beta, nu, lam = theta[0], theta[1], theta[2], theta[3], theta[4]
    
    z_omega = np.log(omega)
    
    remainder = max(1.0 - alpha - beta, 1e-10)
    z_alpha = np.log(alpha / remainder)
    z_beta = np.log(beta / remainder)
    
    z_nu = softplus_inv(nu - 2.0)
    
    lam_max = hansen_lambda_max(nu)
    lam_scaled = np.clip(lam / lam_max, -0.999, 0.999)
    z_lambda = np.arctanh(lam_scaled)
    
    return np.array([z_omega, z_alpha, z_beta, z_nu, z_lambda])


def skewt_jacobian(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute Jacobian J = ∂θ/∂z for Skew-t transformation.
    
    Note: This ignores the dependence of λ_max on ν (small effect).
    """
    theta = skewt_z_to_theta(z)
    omega, alpha, beta, nu, lam = theta[0], theta[1], theta[2], theta[3], theta[4]
    
    J = np.zeros((5, 5), dtype=np.float64)
    
    J[0, 0] = omega
    J[1, 1] = alpha * (1.0 - alpha)
    J[1, 2] = -alpha * beta
    J[2, 1] = -beta * alpha
    J[2, 2] = beta * (1.0 - beta)
    J[3, 3] = softplus_deriv(z[3])
    
    lam_max = hansen_lambda_max(nu)
    tanh_z = np.tanh(z[4])
    J[4, 4] = lam_max * (1.0 - tanh_z ** 2)
    
    return J


# =============================================================================
# LOG-SPACE CHAIN RULE TRANSFORMATIONS (for Normal)
# =============================================================================

def transform_gradient_logspace(
    grad_theta: NDArray[np.float64],
    theta: NDArray[np.float64],
    p: int,
    q: int,
) -> NDArray[np.float64]:
    """
    Transform gradient from θ-space to z-space via chain rule.
    
    g_z = J^T · g_θ
    
    where J is the Jacobian ∂θ/∂z of the JOINT softmax transformation.
    
    For the joint softmax transformation:
        omega = exp(z_omega)
        [alpha, beta] = exp([z_alpha, z_beta]) / (1 + Σ exp([z_alpha, z_beta]))
    
    The Jacobian entries for the persistence parameters are:
        ∂param_i/∂z_j = param_i · (δ_ij - param_j)
    
    where param = [alpha, beta] and the sum runs over ALL persistence params.
    
    Parameters
    ----------
    grad_theta : array
        Gradient in θ-space [∂L/∂omega, ∂L/∂alpha, ∂L/∂beta]
    theta : array
        Parameters in θ-space [omega, alpha, beta]
    p, q : int
        GARCH orders
    
    Returns
    -------
    grad_z : array
        Gradient in z-space
    """
    omega = theta[0]
    alpha = theta[1:1+p]
    beta = theta[1+p:1+p+q]
    
    g_omega = grad_theta[0]
    g_alpha = grad_theta[1:1+p]
    g_beta = grad_theta[1+p:1+p+q]
    
    K = 1 + p + q
    grad_z = np.empty(K, dtype=np.float64)
    
    # omega: ∂L/∂z_omega = omega · ∂L/∂omega
    grad_z[0] = omega * g_omega
    
    # For JOINT softmax over [alpha, beta]:
    # g_z_i = param_i · (g_i - Σ_k param_k · g_k)
    # where the sum is over ALL persistence params (alpha AND beta)
    if p > 0 or q > 0:
        # Concatenate all persistence params and their gradients
        params = np.concatenate([alpha, beta])
        g_params = np.concatenate([g_alpha, g_beta])
        
        # Weighted sum over ALL persistence parameters
        s_total = np.dot(params, g_params)  # Σ param_k · g_k
        
        # Transform
        grad_z_persistence = params * (g_params - s_total)
        
        # Split back
        if p > 0:
            grad_z[1:1+p] = grad_z_persistence[:p]
        if q > 0:
            grad_z[1+p:1+p+q] = grad_z_persistence[p:]
    
    return grad_z


def transform_hessian_logspace(
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    theta: NDArray[np.float64],
    p: int,
    q: int,
) -> NDArray[np.float64]:
    """
    Transform Hessian from θ-space to z-space via chain rule.
    
    H_z = J^T · H_θ · J + Σ_k g_θ_k · ∂²θ_k/∂z∂z
    
    where J is the Jacobian of the JOINT softmax transformation and the second
    term accounts for the curvature of the transformation itself.
    
    For joint softmax over [alpha, beta], there are cross-terms between
    alpha and beta in both the Jacobian and the second derivatives.
    
    Parameters
    ----------
    grad_theta : array
        Gradient in θ-space
    hess_theta : array (K x K)
        Hessian in θ-space
    theta : array
        Parameters in θ-space
    p, q : int
        GARCH orders
    
    Returns
    -------
    hess_z : array (K x K)
        Hessian in z-space
    """
    omega = theta[0]
    alpha = theta[1:1+p]
    beta = theta[1+p:1+p+q]
    
    K = 1 + p + q
    n_persist = p + q  # number of persistence parameters
    
    # Concatenate persistence parameters
    params = np.concatenate([alpha, beta])
    g_persist = np.concatenate([grad_theta[1:1+p], grad_theta[1+p:1+p+q]])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build Jacobian J = ∂θ/∂z
    # ─────────────────────────────────────────────────────────────────────────
    J = np.zeros((K, K), dtype=np.float64)
    
    # omega: ∂omega/∂z_omega = omega
    J[0, 0] = omega
    
    # Joint softmax block: ∂param_i/∂z_j = param_i · (δ_ij - param_j)
    # This is a FULL (p+q) x (p+q) block with cross-terms between alpha and beta
    if n_persist > 0:
        J_persist = np.diag(params) - np.outer(params, params)
        J[1:, 1:] = J_persist
    
    # ─────────────────────────────────────────────────────────────────────────
    # First term: H1 = J^T · H_θ · J
    # ─────────────────────────────────────────────────────────────────────────
    H1 = J.T @ hess_theta @ J
    
    # ─────────────────────────────────────────────────────────────────────────
    # Second term: H2 = Σ_k g_θ_k · ∂²θ_k/∂z∂z
    # ─────────────────────────────────────────────────────────────────────────
    H2 = np.zeros((K, K), dtype=np.float64)
    
    # omega: ∂²omega/∂z_omega² = omega (since omega = exp(z_omega))
    H2[0, 0] = grad_theta[0] * omega
    
    # Joint softmax second derivatives (with cross-terms between alpha and beta):
    # ∂²param_i/∂z_j∂z_k = param_i · [(δ_ij - param_j)(δ_ik - param_k) - param_k·(δ_jk - param_j)]
    if n_persist > 0:
        for i in range(n_persist):
            g_i = g_persist[i]
            p_i = params[i]
            for j in range(n_persist):
                p_j = params[j]
                for k in range(n_persist):
                    p_k = params[k]
                    delta_ij = 1.0 if i == j else 0.0
                    delta_ik = 1.0 if i == k else 0.0
                    delta_jk = 1.0 if j == k else 0.0
                    # Second derivative of joint softmax
                    d2 = p_i * ((delta_ij - p_j) * (delta_ik - p_k) 
                                - p_k * (delta_jk - p_j))
                    H2[1 + j, 1 + k] += g_i * d2
    
    return H1 + H2


# =============================================================================
# CORE OPTIMIZATION PIPELINES
# =============================================================================

def fit_garch_mle(
    resid: NDArray[np.float64],
    loglik_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    grad_fn: Optional[Callable[[NDArray[np.float64], NDArray[np.float64], float], "GradientResult"]] = None,
    hess_fn: Optional[Callable[[NDArray[np.float64], NDArray[np.float64], float], "HessianResult"]] = None,
    p: int = 1,
    q: int = 1,
    solver: Literal["nelder-mead", "slsqp", "trust-constr", "trust-exact"] = "nelder-mead",
    use_logspace: bool = False,
    verbose: bool = False,
) -> EstimationResult:
    """
    Fit GARCH(p,q) model with Normal innovations via MLE.
    
    Parameters
    ----------
    resid : array
        Residual series (e.g., demeaned returns or AR residuals)
    loglik_fn : callable
        Log-likelihood function: loglik_fn(resid, sigma2) -> float
        Should return the log-likelihood (not negative!)
    grad_fn : callable, optional
        Gradient function: grad_fn(params, resid2, sigma2_init) -> GradientResult
        If provided, used by SLSQP, trust-constr, trust-exact
    hess_fn : callable, optional
        Hessian function: hess_fn(params, resid2, sigma2_init) -> HessianResult
        If provided, used by trust-constr, trust-exact
    p : int
        ARCH order
    q : int
        GARCH order
    solver : str
        Optimization method
    use_logspace : bool
        If True, optimize in unconstrained log-space (recommended for trust-exact)
    verbose : bool
        Print optimization progress
    
    Returns
    -------
    EstimationResult
    """
    start_time = time.perf_counter()
    
    n = len(resid)
    resid2 = resid ** 2
    sigma2_init = float(np.var(resid))
    
    # Select variance function
    var_fn = garch_variance_11 if (p == 1 and q == 1) else lambda θ, r2: garch_variance(θ, r2, p, q)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build gradient wrapper (if provided)
    # ─────────────────────────────────────────────────────────────────────────
    jac = None
    if grad_fn is not None and p == 1 and q == 1:
        def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            result = grad_fn(theta, resid2, sigma2_init)
            return result.grad / n  # Scale to match objective
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build Hessian wrapper (if provided)
    # ─────────────────────────────────────────────────────────────────────────
    hess = None
    if hess_fn is not None and p == 1 and q == 1:
        def hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            result = hess_fn(theta, resid2, sigma2_init)
            return result.hess / n  # Scale to match objective
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRAINED OPTIMIZATION (bounds + linear constraint)
    # ─────────────────────────────────────────────────────────────────────────
    if not use_logspace:
        
        def objective(theta: NDArray[np.float64]) -> float:
            """Negative log-likelihood, scaled by n for numerical stability."""
            sigma2 = var_fn(theta, resid2)
            ll = loglik_fn(resid, sigma2)
            return -ll / n
        
        start = default_start_garch(resid, p, q)
        bounds = get_bounds_garch(p, q)
        constraint = build_stationarity_constraint(p, q)
        
        if solver == "nelder-mead":
            config = NELDER_MEAD_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective, start, bounds=bounds, **config)
            
        elif solver == "slsqp":
            config = SLSQP_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            # SLSQP uses gradient if provided
            res = minimize(objective, start, jac=jac, bounds=bounds, constraints=constraint, **config)
            
        elif solver == "trust-constr":
            config = TRUST_CONSTR_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose, "verbose": 2 if verbose else 0}
            # Adjust trust radius based on problem size
            config["options"]["initial_tr_radius"] = max(1 / (10 ** (p + q + 1)), 1e-6)
            # trust-constr uses gradient and Hessian if provided
            res = minimize(objective, start, jac=jac, hess=hess, bounds=bounds, constraints=constraint, **config)
            
        else:
            raise ValueError(f"Unknown solver '{solver}' for constrained optimization")
        
        theta_hat = res.x
        nll = res.fun * n  # Unscale
        
    # ─────────────────────────────────────────────────────────────────────────
    # UNCONSTRAINED LOG-SPACE OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────────
    else:
        
        def objective_logspace(z: NDArray[np.float64]) -> float:
            theta = pack_params_logspace(z, p, q)
            sigma2 = var_fn(theta, resid2)
            ll = loglik_fn(resid, sigma2)
            return -ll * OBJECTIVE_SCALE / n
        
        # Build log-space gradient wrapper (chain rule transformed)
        jac_logspace = None
        if grad_fn is not None and p == 1 and q == 1:
            def jac_logspace(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta = pack_params_logspace(z, p, q)
                result = grad_fn(theta, resid2, sigma2_init)
                grad_theta = result.grad
                # Transform via chain rule: g_z = J^T · g_θ
                grad_z = transform_gradient_logspace(grad_theta, theta, p, q)
                return grad_z * OBJECTIVE_SCALE / n
        
        # Build log-space Hessian wrapper (chain rule transformed)
        hess_logspace = None
        if hess_fn is not None and p == 1 and q == 1:
            def hess_logspace(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta = pack_params_logspace(z, p, q)
                # Need both gradient and Hessian in θ-space
                hess_result = hess_fn(theta, resid2, sigma2_init)
                grad_theta = hess_result.grad
                hess_theta = hess_result.hess
                # Transform via chain rule: H_z = J^T · H_θ · J + correction
                hess_z = transform_hessian_logspace(grad_theta, hess_theta, theta, p, q)
                return hess_z * OBJECTIVE_SCALE / n
        
        theta0 = default_start_garch(resid, p, q)
        z0 = unpack_params_logspace(theta0, p, q)
        
        if solver == "nelder-mead":
            config = NELDER_MEAD_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective_logspace, z0, **config)
            
        elif solver == "slsqp":
            config = SLSQP_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective_logspace, z0, jac=jac_logspace, **config)
            
        elif solver == "trust-exact":
            config = TRUST_EXACT_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            if hess_logspace is None:
                warnings.warn("trust-exact works best with analytical Hessian. "
                              "Pass hess_fn for faster convergence.")
            res = minimize(objective_logspace, z0, jac=jac_logspace, hess=hess_logspace, **config)
            
        elif solver == "trust-constr":
            config = TRUST_CONSTR_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose, "verbose": 2 if verbose else 0}
            res = minimize(objective_logspace, z0, jac=jac_logspace, hess=hess_logspace, **config)
            
        else:
            raise ValueError(f"Solver '{solver}' not supported for log-space optimization")
        
        theta_hat = pack_params_logspace(res.x, p, q)
        nll = res.fun * n / OBJECTIVE_SCALE  # Unscale
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build result
    # ─────────────────────────────────────────────────────────────────────────
    sigma2_final = var_fn(theta_hat, resid2)
    std_resid = resid / np.sqrt(sigma2_final)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute standard errors from Hessian
    # ─────────────────────────────────────────────────────────────────────────
    hessian_mat = None
    cov_matrix = None
    std_errors = None
    
    # Try analytical Hessian first
    if hess_fn is not None and p == 1 and q == 1:
        try:
            hess_result = hess_fn(theta_hat, resid2, sigma2_init)
            hessian_mat = hess_result.hess
            cov_matrix = np.linalg.inv(hessian_mat)
            diag_cov = np.diag(cov_matrix)
            # Need ALL positive diagonals for valid SEs
            if np.all(diag_cov > 0):
                std_errors = np.sqrt(diag_cov)
            else:
                # Analytical Hessian failed; trigger numerical fallback
                hessian_mat = None
                cov_matrix = None
        except (np.linalg.LinAlgError, ValueError):
            hessian_mat = None
            cov_matrix = None
    
    # Fallback to numerical Hessian if analytical failed or wasn't available
    if std_errors is None:
        try:
            # Build objective for numerical differentiation
            def obj_for_hess(theta):
                sigma2 = var_fn(theta, resid2)
                return -loglik_fn(resid, sigma2) / n
            hessian_mat = compute_numerical_hessian(obj_for_hess, theta_hat)
            cov_matrix = np.linalg.inv(hessian_mat) / n  # Unscale
            diag_cov = np.diag(cov_matrix)
            # Need ALL positive diagonals for valid SEs
            if np.all(diag_cov > 0):
                std_errors = np.sqrt(diag_cov)
            elif np.any(diag_cov > 0):
                # Partial SEs: NaN for negative diagonals
                std_errors = np.where(diag_cov > 0, np.sqrt(diag_cov), np.nan)
        except (np.linalg.LinAlgError, ValueError):
            pass
    
    return EstimationResult(
        garch_params=GARCHParams.from_array(theta_hat, p, q),
        dist_params=DistributionParams(),
        log_likelihood=-nll,
        n_obs=n,
        converged=res.success,
        n_iter=res.nit if hasattr(res, 'nit') else res.nfev,
        message=res.message if hasattr(res, 'message') else str(res.get('message', '')),
        hessian=hessian_mat,
        cov_matrix=cov_matrix,
        std_errors=std_errors,
        sigma2=sigma2_final,
        std_resid=std_resid,
        time_elapsed=time.perf_counter() - start_time,
    )


def fit_garch_studentt_mle(
    resid: NDArray[np.float64],
    loglik_fn: Optional[Callable[[NDArray[np.float64], NDArray[np.float64], float], float]] = None,
    grad_fn: Optional[Callable] = None,
    hess_fn: Optional[Callable] = None,
    p: int = 1,
    q: int = 1,
    solver: Literal["nelder-mead", "slsqp", "trust-constr"] = "nelder-mead",
    use_logspace: bool = False,
    verbose: bool = False,
) -> EstimationResult:
    """
    Fit GARCH(p,q) model with Student-t innovations via MLE.
    
    Parameters
    ----------
    resid : array
        Residual series
    loglik_fn : callable, optional
        Log-likelihood function: loglik_fn(resid, sigma2, nu) -> float
        If None, uses studentt_loglik from likelihoods module.
    grad_fn : callable, optional
        Gradient function for GARCH(1,1)+Student-t
    hess_fn : callable, optional
        Hessian function for GARCH(1,1)+Student-t
    p, q : int
        GARCH orders
    solver : str
        Optimization method
    use_logspace : bool
        If True, optimize in unconstrained log-space (stationarity enforced by construction)
    verbose : bool
        Print progress
    
    Returns
    -------
    EstimationResult
    """
    start_time = time.perf_counter()
    
    n = len(resid)
    resid2 = resid ** 2
    sigma2_init = float(np.var(resid))
    
    # Use default likelihood if not provided
    if loglik_fn is None:
        loglik_fn = studentt_loglik
    
    var_fn = garch_variance_11 if (p == 1 and q == 1) else lambda θ, r2: garch_variance(θ, r2, p, q)
    
    # ─────────────────────────────────────────────────────────────────────────
    # UNCONSTRAINED LOG-SPACE OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────────
    if use_logspace:
        
        def objective_logspace(z: NDArray[np.float64]) -> float:
            theta = studentt_z_to_theta(z)
            omega, alpha, beta, nu = theta[0], theta[1], theta[2], theta[3]
            garch_params = np.array([omega, alpha, beta])
            sigma2 = var_fn(garch_params, resid2)
            ll = loglik_fn(resid, sigma2, nu)
            if not np.isfinite(ll):
                return STATIONARITY_PENALTY
            return -ll / n
        
        # Initial values
        theta0 = np.concatenate([default_start_garch(resid, p, q), default_start_studentt()])
        z0 = studentt_theta_to_z(theta0)
        
        if solver == "nelder-mead":
            config = NELDER_MEAD_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective_logspace, z0, **config)
            
        elif solver == "slsqp":
            config = SLSQP_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective_logspace, z0, **config)
            
        elif solver == "trust-constr":
            config = TRUST_CONSTR_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose, "verbose": 2 if verbose else 0}
            res = minimize(objective_logspace, z0, **config)
            
        else:
            raise ValueError(f"Unknown solver '{solver}' for log-space optimization")
        
        theta_hat = studentt_z_to_theta(res.x)
        garch_hat = np.array([theta_hat[0], theta_hat[1], theta_hat[2]])
        nu_hat = theta_hat[3]
        nll = res.fun * n
        z_hat = res.x  # Save for SE computation
        
    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRAINED OPTIMIZATION (bounds + linear constraint)
    # ─────────────────────────────────────────────────────────────────────────
    else:
        
        def objective(params: NDArray[np.float64]) -> float:
            theta_garch = params[:1+p+q]
            nu = params[1+p+q]
            
            # Stationarity check (penalty method)
            persistence = np.sum(theta_garch[1:])
            if persistence >= STATIONARITY_UPPER:
                return STATIONARITY_PENALTY
            
            sigma2 = var_fn(theta_garch, resid2)
            ll = loglik_fn(resid, sigma2, nu)
            return -ll / n
        
        # Build gradient wrapper (if provided)
        jac = None
        if grad_fn is not None and p == 1 and q == 1:
            def jac(params: NDArray[np.float64]) -> NDArray[np.float64]:
                result = grad_fn(params, resid2, sigma2_init)
                return result.grad / n
        
        # Build Hessian wrapper (if provided)
        hess = None
        if hess_fn is not None and p == 1 and q == 1:
            def hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                result = hess_fn(params, resid2, sigma2_init)
                return result.hess / n
        
        # Initial values
        start_garch = default_start_garch(resid, p, q)
        start_nu = default_start_studentt()
        start = np.concatenate([start_garch, start_nu])
        
        # Bounds
        bounds = get_bounds_garch(p, q) + get_bounds_studentt()
        
        # Constraint (only on GARCH params)
        constraint = build_stationarity_constraint_with_shape(p, q, n_shape=1)
        
        if solver == "nelder-mead":
            config = NELDER_MEAD_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective, start, bounds=bounds, **config)
            
        elif solver == "slsqp":
            config = SLSQP_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective, start, jac=jac, bounds=bounds, constraints=constraint, **config)
            
        elif solver == "trust-constr":
            config = TRUST_CONSTR_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose, "verbose": 2 if verbose else 0}
            res = minimize(objective, start, jac=jac, hess=hess, bounds=bounds, constraints=constraint, **config)
            
        else:
            raise ValueError(f"Unknown solver '{solver}'")
        
        garch_hat = res.x[:1+p+q]
        nu_hat = res.x[1+p+q]
        nll = res.fun * n
        z_hat = None  # Not used for constrained optimization
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build result
    # ─────────────────────────────────────────────────────────────────────────
    sigma2_final = var_fn(garch_hat, resid2)
    std_resid = resid / np.sqrt(sigma2_final)
    
    # Compute standard errors
    hessian_mat = None
    cov_matrix = None
    std_errors = None
    
    if use_logspace and z_hat is not None:
        # Compute SEs via Hessian in z-space + chain rule
        try:
            H_z = compute_hessian_unconstrained(objective_logspace, z_hat, eps=1e-5)
            cov_z = np.linalg.inv(H_z) / n
            J = studentt_jacobian(z_hat)
            cov_matrix = J @ cov_z @ J.T
            diag_cov = np.diag(cov_matrix)
            std_errors = np.where(diag_cov > 0, np.sqrt(diag_cov), np.nan)
        except Exception:
            pass  # Keep std_errors as None
    else:
        # Try analytical Hessian first
        if hess_fn is not None and p == 1 and q == 1:
            try:
                full_params = np.concatenate([garch_hat, [nu_hat]])
                hess_result = hess_fn(full_params, resid2, sigma2_init)
                hessian_mat = hess_result.hess
                cov_matrix = np.linalg.inv(hessian_mat)
                diag_cov = np.diag(cov_matrix)
                if np.all(diag_cov > 0):
                    std_errors = np.sqrt(diag_cov)
                else:
                    hessian_mat = None
                    cov_matrix = None
            except (np.linalg.LinAlgError, ValueError):
                hessian_mat = None
                cov_matrix = None
        
        # Fallback to robust Hessian (reparameterized, boundary-safe)
        if std_errors is None and p == 1 and q == 1:
            try:
                full_params = np.array([garch_hat[0], garch_hat[1], garch_hat[2], nu_hat])
                robust_result = compute_robust_hessian_studentt(objective, full_params, eps=1e-5)
                if robust_result.success:
                    cov_matrix = robust_result.cov_theta / n
                    std_errors = robust_result.std_errors / np.sqrt(n)
                    hessian_mat = robust_result.hessian_theta
            except Exception:
                pass
    
    return EstimationResult(
        garch_params=GARCHParams.from_array(garch_hat, p, q),
        dist_params=DistributionParams(nu=nu_hat),
        log_likelihood=-nll,
        n_obs=n,
        converged=res.success,
        n_iter=res.nit if hasattr(res, 'nit') else res.nfev,
        message=res.message if hasattr(res, 'message') else "",
        hessian=hessian_mat,
        cov_matrix=cov_matrix,
        std_errors=std_errors,
        sigma2=sigma2_final,
        std_resid=std_resid,
        time_elapsed=time.perf_counter() - start_time,
    )


def fit_garch_skewt_mle(
    resid: NDArray[np.float64],
    loglik_fn: Optional[Callable[[NDArray[np.float64], NDArray[np.float64], float, float], float]] = None,
    p: int = 1,
    q: int = 1,
    solver: Literal["nelder-mead", "slsqp", "trust-constr"] = "nelder-mead",
    use_logspace: bool = False,
    verbose: bool = False,
) -> EstimationResult:
    """
    Fit GARCH(p,q) model with Skewed Student-t innovations via MLE.
    
    Uses numerical derivatives (no analytical derivatives available for Skew-t).
    
    Parameters
    ----------
    resid : array
        Residual series
    loglik_fn : callable, optional
        Log-likelihood function: loglik_fn(resid, sigma2, nu, lam) -> float
        If None, uses skewt_loglik from likelihoods module.
    p, q : int
        GARCH orders
    solver : str
        Optimization method
    use_logspace : bool
        If True, optimize in unconstrained log-space (stationarity enforced by construction)
    verbose : bool
        Print progress
    
    Returns
    -------
    EstimationResult
    """
    start_time = time.perf_counter()
    
    n = len(resid)
    resid2 = resid ** 2
    
    # Use default likelihood if not provided
    if loglik_fn is None:
        loglik_fn = skewt_loglik
    
    var_fn = garch_variance_11 if (p == 1 and q == 1) else lambda θ, r2: garch_variance(θ, r2, p, q)
    
    # ─────────────────────────────────────────────────────────────────────────
    # UNCONSTRAINED LOG-SPACE OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────────
    if use_logspace:
        
        def objective_logspace(z: NDArray[np.float64]) -> float:
            theta = skewt_z_to_theta(z)
            omega, alpha, beta, nu, lam = theta[0], theta[1], theta[2], theta[3], theta[4]
            garch_params = np.array([omega, alpha, beta])
            sigma2 = var_fn(garch_params, resid2)
            ll = loglik_fn(resid, sigma2, nu, lam)
            if not np.isfinite(ll):
                return STATIONARITY_PENALTY
            return -ll / n
        
        # Initial values
        theta0 = np.concatenate([default_start_garch(resid, p, q), default_start_skewt()])
        z0 = skewt_theta_to_z(theta0)
        
        if solver == "nelder-mead":
            config = NELDER_MEAD_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective_logspace, z0, **config)
            
        elif solver == "slsqp":
            config = SLSQP_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective_logspace, z0, **config)
            
        elif solver == "trust-constr":
            config = TRUST_CONSTR_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose, "verbose": 2 if verbose else 0}
            res = minimize(objective_logspace, z0, **config)
            
        else:
            raise ValueError(f"Unknown solver '{solver}' for log-space optimization")
        
        theta_hat = skewt_z_to_theta(res.x)
        garch_hat = np.array([theta_hat[0], theta_hat[1], theta_hat[2]])
        nu_hat = theta_hat[3]
        lam_hat = theta_hat[4]
        nll = res.fun * n
        z_hat = res.x  # Save for SE computation
        
    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRAINED OPTIMIZATION (bounds + linear constraint)
    # ─────────────────────────────────────────────────────────────────────────
    else:
        
        def objective(params: NDArray[np.float64]) -> float:
            theta_garch = params[:1+p+q]
            nu = params[1+p+q]
            lam = params[1+p+q+1]
            
            # Stationarity check
            persistence = np.sum(theta_garch[1:])
            if persistence >= STATIONARITY_UPPER:
                return STATIONARITY_PENALTY
            
            sigma2 = var_fn(theta_garch, resid2)
            ll = loglik_fn(resid, sigma2, nu, lam)
            
            if not np.isfinite(ll):
                return STATIONARITY_PENALTY
                
            return -ll / n
        
        # Initial values
        start_garch = default_start_garch(resid, p, q)
        start_shape = default_start_skewt()
        start = np.concatenate([start_garch, start_shape])
        
        # Bounds
        bounds = get_bounds_garch(p, q) + get_bounds_skewt()
        
        # Constraint
        constraint = build_stationarity_constraint_with_shape(p, q, n_shape=2)
        
        if solver == "nelder-mead":
            config = NELDER_MEAD_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective, start, bounds=bounds, **config)
            
        elif solver == "slsqp":
            config = SLSQP_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose}
            res = minimize(objective, start, bounds=bounds, constraints=constraint, **config)
            
        elif solver == "trust-constr":
            config = TRUST_CONSTR_CONFIG.copy()
            config["options"] = {**config["options"], "disp": verbose, "verbose": 2 if verbose else 0}
            res = minimize(objective, start, bounds=bounds, constraints=constraint, **config)
            
        else:
            raise ValueError(f"Unknown solver '{solver}'")
        
        garch_hat = res.x[:1+p+q]
        nu_hat = res.x[1+p+q]
        lam_hat = res.x[1+p+q+1]
        nll = res.fun * n
        z_hat = None  # Not used for constrained optimization
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build result
    # ─────────────────────────────────────────────────────────────────────────
    sigma2_final = var_fn(garch_hat, resid2)
    std_resid = resid / np.sqrt(sigma2_final)
    
    # Compute standard errors
    hessian_mat = None
    cov_matrix = None
    std_errors = None
    
    if use_logspace and z_hat is not None:
        # Compute SEs via Hessian in z-space + chain rule
        try:
            H_z = compute_hessian_unconstrained(objective_logspace, z_hat, eps=1e-5)
            cov_z = np.linalg.inv(H_z) / n
            J = skewt_jacobian(z_hat)
            cov_matrix = J @ cov_z @ J.T
            diag_cov = np.diag(cov_matrix)
            std_errors = np.where(diag_cov > 0, np.sqrt(diag_cov), np.nan)
        except Exception:
            pass  # Keep std_errors as None
    else:
        # Robust Hessian (reparameterized, boundary-safe)
        if p == 1 and q == 1:
            try:
                full_params = np.array([garch_hat[0], garch_hat[1], garch_hat[2], nu_hat, lam_hat])
                robust_result = compute_robust_hessian_skewt(objective, full_params, eps=1e-5)
                if robust_result.success:
                    cov_matrix = robust_result.cov_theta / n
                    std_errors = robust_result.std_errors / np.sqrt(n)
                    hessian_mat = robust_result.hessian_theta
            except Exception:
                pass
    
    return EstimationResult(
        garch_params=GARCHParams.from_array(garch_hat, p, q),
        dist_params=DistributionParams(nu=nu_hat, lam=lam_hat),
        log_likelihood=-nll,
        n_obs=n,
        converged=res.success,
        n_iter=res.nit if hasattr(res, 'nit') else res.nfev,
        message=res.message if hasattr(res, 'message') else "",
        hessian=hessian_mat,
        cov_matrix=cov_matrix,
        std_errors=std_errors,
        sigma2=sigma2_final,
        std_resid=std_resid,
        time_elapsed=time.perf_counter() - start_time,
    )


def fit_garch_qmle(
    resid: NDArray[np.float64],
    loglik_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    robust_se_fn: Optional[Callable] = None,
    grad_fn: Optional[Callable] = None,
    hess_fn: Optional[Callable] = None,
    p: int = 1,
    q: int = 1,
    solver: Literal["nelder-mead", "slsqp", "trust-constr"] = "nelder-mead",
    verbose: bool = False,
) -> EstimationResult:
    """
    Fit GARCH(p,q) model via Quasi-Maximum Likelihood Estimation (QMLE).
    
    Uses Normal log-likelihood for parameter estimation, but computes robust
    (sandwich) standard errors that are valid even when the true distribution
    is non-Normal.
    
    The sandwich covariance estimator is:
        V_robust = H^{-1} · OPG · H^{-1}
    
    where:
        H = Hessian of the negative log-likelihood
        OPG = Outer Product of Gradients = Σ_t (g_t · g_t')
        g_t = score (gradient of -log L) at observation t
    
    Parameters
    ----------
    resid : array
        Residual series
    loglik_fn : callable
        Normal log-likelihood: loglik_fn(resid, sigma2) -> float
    robust_se_fn : callable, optional
        Function that computes robust SEs: robust_se_fn(params, resid2, sigma2_init)
        -> RobustSEResult. If provided, used after optimization to get robust SEs.
        This should be `garch11_normal_robust_se` from likelihoods.py
    p, q : int
        GARCH orders (only p=q=1 supported with robust_se_fn)
    solver : str
        Optimization method
    verbose : bool
        Print progress
    
    Returns
    -------
    EstimationResult
        Contains both MLE standard errors (.se) and robust standard errors (.se_robust)
    """
    start_time = time.perf_counter()
    
    # Step 1: Optimize using Normal likelihood (same as MLE)
    result = fit_garch_mle(
        resid=resid,
        loglik_fn=loglik_fn,
        grad_fn=grad_fn,
        hess_fn=hess_fn,
        p=p,
        q=q,
        solver=solver,
        use_logspace=False,
        verbose=verbose,
    )
    
    # Step 2: Compute robust standard errors if function is provided
    if robust_se_fn is not None:
        if p != 1 or q != 1:
            warnings.warn("Robust SE computation only implemented for GARCH(1,1). "
                          "Returning MLE standard errors.")
        else:
            # Extract GARCH params [omega, alpha, beta]
            garch_params = np.array([
                result.garch_params.omega,
                result.garch_params.alpha[0],
                result.garch_params.beta[0],
            ], dtype=np.float64)
            
            resid2 = resid ** 2
            sigma2_init = float(np.var(resid))
            
            try:
                robust_result = robust_se_fn(garch_params, resid2, sigma2_init)
                
                # Update result with robust SEs
                result.method = "QMLE"
                result.hessian = robust_result.hess
                result.opg = robust_result.opg
                result.cov_matrix = robust_result.cov_mle[:3, :3]
                result.cov_robust = robust_result.cov_robust[:3, :3]
                result.std_errors = robust_result.se_mle[:3]
                result.std_errors_robust = robust_result.se_robust[:3]
                
                if verbose:
                    print("Robust (sandwich) standard errors computed successfully.")
                    print(f"  SE(omega):  MLE={robust_result.se_mle[0]:.6f}, Robust={robust_result.se_robust[0]:.6f}")
                    print(f"  SE(alpha):  MLE={robust_result.se_mle[1]:.6f}, Robust={robust_result.se_robust[1]:.6f}")
                    print(f"  SE(beta):   MLE={robust_result.se_mle[2]:.6f}, Robust={robust_result.se_robust[2]:.6f}")
            except Exception as e:
                warnings.warn(f"Robust SE computation failed: {e}. Returning MLE standard errors.")
    
    # Update time to include robust SE computation
    result.time_elapsed = time.perf_counter() - start_time
    return result


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def fit_garch(
    resid: NDArray[np.float64],
    dist: Literal["normal", "studentt", "skewt"] = "normal",
    p: int = 1,
    q: int = 1,
    method: Literal["mle", "qmle"] = "mle",
    solver: Literal["nelder-mead", "slsqp", "trust-constr", "trust-exact"] = "trust-constr",
    use_logspace: bool = False,
    use_derivatives: bool = True,
    verbose: bool = False,
) -> EstimationResult:
    """
    High-level GARCH estimation interface.
    
    All likelihood functions and analytical derivatives are handled automatically.
    
    Parameters
    ----------
    resid : array
        Residual series (demeaned returns or AR residuals)
    dist : str
        Distribution for innovations: "normal", "studentt", or "skewt"
    p : int
        ARCH order (default 1)
    q : int
        GARCH order (default 1)
    method : str
        "mle" for Maximum Likelihood, "qmle" for Quasi-MLE with robust SEs
    solver : str
        Optimization method:
        - "nelder-mead": derivative-free, robust but slow
        - "slsqp": uses gradient, good for constrained problems
        - "trust-constr": uses gradient + Hessian, recommended
        - "trust-exact": uses gradient + Hessian in log-space (Normal only)
    use_logspace : bool
        If True, optimize in unconstrained log-space. Stationarity is enforced
        by construction via softmax transformation. Standard errors are computed
        via chain rule.
        
        Transformations used:
        - omega = exp(z_omega)
        - [alpha, beta] = softmax([z_alpha, z_beta, 0])[:2]
        - nu = 2 + softplus(z_nu)  (Student-t, Skew-t)
        - lambda = lambda_max(nu) * tanh(z_lambda)  (Skew-t only)
        
        Required for trust-exact solver. Supported for all distributions.
    use_derivatives : bool
        If True (default), use analytical gradient and Hessian for faster convergence.
        If False, use numerical derivatives. Ignored when use_logspace=True.
    verbose : bool
        Print optimization progress
    
    Returns
    -------
    EstimationResult
        Contains estimated parameters, log-likelihood, conditional variances,
        standardized residuals, and standard errors.
    
    Examples
    --------
    >>> result = fit_garch(returns, dist="normal")
    >>> print(f"Persistence: {result.garch_params.persistence:.4f}")
    
    >>> result = fit_garch(returns, dist="studentt")
    >>> print(f"Degrees of freedom: {result.dist_params.nu:.2f}")
    
    >>> result = fit_garch(returns, method="qmle")  # robust standard errors
    >>> print(f"Robust SE(alpha): {result.std_errors_robust[1]:.4f}")
    
    >>> # Log-space optimization (recommended for Student-t and Skew-t)
    >>> result = fit_garch(returns, dist="studentt", solver="trust-constr", use_logspace=True)
    >>> result = fit_garch(returns, dist="skewt", solver="trust-constr", use_logspace=True)
    """
    resid = np.asarray(resid, dtype=np.float64)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Validate inputs
    # ─────────────────────────────────────────────────────────────────────────
    if p != 1 or q != 1:
        if use_derivatives:
            warnings.warn("Analytical derivatives only available for GARCH(1,1). "
                          "Using numerical derivatives.")
            use_derivatives = False
    
    if solver == "trust-exact":
        if dist != "normal":
            warnings.warn(f"trust-exact solver only available for Normal distribution. "
                          f"Using trust-constr for dist='{dist}'.")
            solver = "trust-constr"  # type: ignore[assignment]
        if not use_logspace:
            warnings.warn("trust-exact requires use_logspace=True. Enabling log-space mode.")
            use_logspace = True
    
    # ─────────────────────────────────────────────────────────────────────────
    # Select gradient and Hessian functions
    # ─────────────────────────────────────────────────────────────────────────
    grad_fn = None
    hess_fn = None
    robust_se_fn = None
    
    if use_derivatives and p == 1 and q == 1:
        if dist == "normal":
            grad_fn = garch11_normal_gradient
            hess_fn = garch11_normal_hessian
            robust_se_fn = garch11_normal_robust_se
        elif dist == "studentt":
            grad_fn = garch11_studentt_gradient
            hess_fn = garch11_studentt_hessian
            robust_se_fn = garch11_studentt_robust_se
        # Skew-t uses numerical derivatives (no analytical available)
    
    # ─────────────────────────────────────────────────────────────────────────
    # QMLE: Normal likelihood with robust standard errors
    # ─────────────────────────────────────────────────────────────────────────
    if method == "qmle":
        if solver == "trust-exact":
            raise ValueError("QMLE does not support trust-exact solver. "
                             "Use trust-constr, slsqp, or nelder-mead.")
        
        return fit_garch_qmle(
            resid=resid,
            loglik_fn=normal_loglik,
            robust_se_fn=robust_se_fn if use_derivatives else None,
            grad_fn=grad_fn if use_derivatives else None,
            hess_fn=hess_fn if use_derivatives else None,
            p=p,
            q=q,
            solver=solver,  # type: ignore[arg-type]
            verbose=verbose,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # MLE: Distribution-specific estimation
    # ─────────────────────────────────────────────────────────────────────────
    if dist == "normal":
        return fit_garch_mle(
            resid=resid,
            loglik_fn=normal_loglik,
            grad_fn=grad_fn,
            hess_fn=hess_fn,
            p=p,
            q=q,
            solver=solver,  # type: ignore[arg-type]
            use_logspace=use_logspace,
            verbose=verbose,
        )
    
    elif dist == "studentt":
        return fit_garch_studentt_mle(
            resid=resid,
            loglik_fn=studentt_loglik,
            grad_fn=grad_fn if not use_logspace else None,  # Don't use analytical grads in logspace
            hess_fn=hess_fn if not use_logspace else None,
            p=p,
            q=q,
            solver=solver,  # type: ignore[arg-type]
            use_logspace=use_logspace,
            verbose=verbose,
        )
    
    elif dist == "skewt":
        return fit_garch_skewt_mle(
            resid=resid,
            loglik_fn=None,  # Uses internal skewt likelihood
            p=p,
            q=q,
            solver=solver,  # type: ignore[arg-type]
            use_logspace=use_logspace,
            verbose=verbose,
        )
    
    else:
        raise ValueError(f"Unknown distribution '{dist}'. Must be 'normal', 'studentt', or 'skewt'.")


# =============================================================================
# STANDARD ERROR COMPUTATION (Post-estimation)
# =============================================================================

# Note: compute_numerical_hessian is now imported from numerical_hessians.py


def compute_opg_matrix(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    params: NDArray[np.float64],
    p: int,
    q: int,
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Compute Outer Product of Gradients (OPG) matrix for robust standard errors.
    
    OPG = Σ_t g_t g_t'
    
    where g_t is the gradient of the log-likelihood contribution at time t.
    """
    n = len(resid)
    k = len(params)
    
    # This is a simplified version - full implementation would compute
    # analytic gradients of each observation's contribution
    
    # For now, return identity scaled by n (placeholder)
    warnings.warn(
        "OPG computation not fully implemented - using Hessian-based standard errors",
        UserWarning
    )
    return np.eye(k) * n


def compute_robust_cov(
    hessian: NDArray[np.float64],
    opg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute robust (sandwich) covariance matrix.
    
    V_robust = H^{-1} @ OPG @ H^{-1}
    
    This is consistent even when the likelihood is misspecified (QMLE).
    """
    try:
        H_inv = np.linalg.inv(hessian)
        return H_inv @ opg @ H_inv
    except np.linalg.LinAlgError:
        warnings.warn("Hessian inversion failed - returning None for covariance")
        return None


# =============================================================================
# LIKELIHOOD FUNCTION WRAPPERS (use likelihoods.py for full implementation)
# =============================================================================

def _example_normal_loglik(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
) -> float:
    """
    Normal log-likelihood.
    
    For analytical gradients/Hessians, use likelihoods.py module.
    """
    n = len(resid)
    ll = -0.5 * n * np.log(2 * np.pi)
    ll -= 0.5 * np.sum(np.log(sigma2) + resid**2 / sigma2)
    return ll


def _example_studentt_loglik(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    nu: float,
) -> float:
    """
    Student-t log-likelihood.
    
    For analytical gradients/Hessians, use likelihoods.py module.
    """
    from scipy.special import gammaln
    
    n = len(resid)
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    z2 = resid**2 / sigma2
    ll = n * const - 0.5 * np.sum(np.log(sigma2)) - (nu + 1) / 2 * np.sum(np.log(1 + z2 / (nu - 2)))
    return ll


def _example_skewt_loglik(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    nu: float,
    lam: float,
) -> float:
    """
    Skewed Student-t log-likelihood (Hansen 1994).
    
    For analytical gradients/Hessians, use likelihoods.py module.
    """
    from scipy.special import gammaln
    
    c = gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2))
    a = 4 * lam * np.exp(c) * (nu - 2) / (nu - 1)
    b = np.sqrt(1 + 3 * lam**2 - a**2)
    
    z = resid / np.sqrt(sigma2)
    z_adj = (b * z + a) / (1 - lam * np.sign(b * z + a))
    
    ll = (len(resid) * c 
          - 0.5 * np.sum(np.log(sigma2))
          + np.sum(np.log(b))
          - 0.5 * (nu + 1) * np.sum(np.log(1 + z_adj**2 / (nu - 2))))
    
    return ll


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    
    # Generate GARCH(1,1) process
    n = 1000
    omega_true, alpha_true, beta_true = 0.01, 0.1, 0.85
    
    sigma2 = np.zeros(n)
    resid = np.zeros(n)
    sigma2[0] = omega_true / (1 - alpha_true - beta_true)
    
    for t in range(1, n):
        sigma2[t] = omega_true + alpha_true * resid[t-1]**2 + beta_true * sigma2[t-1]
        resid[t] = np.sqrt(sigma2[t]) * np.random.standard_normal()
    
    # Fit model - new simple interface
    print("Fitting GARCH(1,1) to synthetic data...")
    print(f"True params: omega={omega_true}, alpha={alpha_true}, beta={beta_true}")
    print()
    
    # Test 1: Normal MLE with analytical derivatives (default)
    print("=== Normal MLE (trust-constr + analytical derivatives) ===")
    result = fit_garch(resid, dist="normal", solver="trust-constr")
    print(f"  omega = {result.garch_params.omega:.6f}")
    print(f"  alpha = {result.garch_params.alpha[0]:.6f}")
    print(f"  beta  = {result.garch_params.beta[0]:.6f}")
    print(f"  persistence = {result.garch_params.persistence:.6f}")
    print(f"  Converged: {result.converged}")
    print()
    
    # Test 2: Student-t MLE
    print("=== Student-t MLE ===")
    result_t = fit_garch(resid, dist="studentt", solver="nelder-mead")
    print(f"  omega = {result_t.garch_params.omega:.6f}")
    print(f"  alpha = {result_t.garch_params.alpha[0]:.6f}")
    print(f"  beta  = {result_t.garch_params.beta[0]:.6f}")
    print(f"  nu (df) = {result_t.dist_params.nu:.2f}")
    print()
    
    # Test 3: QMLE with robust SEs
    print("=== QMLE (Normal likelihood + robust SEs) ===")
    result_q = fit_garch(resid, method="qmle", solver="trust-constr")
    print(f"  omega = {result_q.garch_params.omega:.6f}")
    print(f"  alpha = {result_q.garch_params.alpha[0]:.6f}")
    print(f"  beta  = {result_q.garch_params.beta[0]:.6f}")
    if result_q.std_errors_robust is not None:
        print(f"  Robust SEs: {result_q.std_errors_robust}")

