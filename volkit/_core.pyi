# src/volkit/_core.pyi
"""
Stub file for the compiled extension **volkit._core**

Every pointer parameter is typed as `_IntPtr`, a union of:

  • `int`                       ─ raw address, e.g. `array.ctypes.data`
  • `ctypes.c_void_p`           ─ generic void pointer
  • `ctypes._Pointer[Any]`      ─ any typed ctypes pointer

Editors and type-checkers use this file only; at runtime CPython loads the
matching _core.cpython-*.so.
"""

from typing import Any, Union
import ctypes

_IntPtr = Union[int, ctypes.c_void_p, ctypes._Pointer[Any]]
_Size = Union[int, ctypes.c_size_t]

# ── GARCH variance computation ────────────────────────────────────────────

def _garch_variance_pq(
    theta_ptr: _IntPtr,   # GARCH parameters [omega, alpha1, ..., alphap, beta1, ..., betaq]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Output: conditional variances (modified in-place)
    n: _Size,             # Number of observations
    p: _Size,             # GARCH order (number of alpha parameters)
    q: _Size,             # ARCH order (number of beta parameters)
) -> None:
    """Compute GARCH(p,q) conditional variances"""
    ...

def _garch_variance_11(
    theta_ptr: _IntPtr,   # GARCH(1,1) parameters [omega, alpha, beta]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Output: conditional variances (modified in-place)
    n: _Size,             # Number of observations
) -> None:
    """Compute GARCH(1,1) conditional variances (optimized)"""
    ...

# ── GARCH log-likelihood computation ──────────────────────────────────────

def _garch_ll_11_normal(
    theta_ptr: _IntPtr,   # GARCH(1,1) parameters [omega, alpha, beta]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Working array for conditional variances
    n: _Size,             # Number of observations
) -> float:
    """Compute GARCH(1,1) + Normal log-likelihood (optimized)"""
    ...

def _garch_ll_pq_normal(
    theta_ptr: _IntPtr,   # GARCH parameters [omega, alpha1, ..., alphap, beta1, ..., betaq]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Working array for conditional variances
    n: _Size,             # Number of observations
    p: _Size,             # GARCH order (number of alpha parameters)
    q: _Size,             # ARCH order (number of beta parameters)
) -> float:
    """Compute GARCH(p,q) + Normal log-likelihood"""
    ...

# ── Pure likelihood functions ─────────────────────────────────────────────

def _normal_ll(
    sigma2_ptr: _IntPtr,  # Conditional variances
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    n: _Size,             # Number of observations
) -> float:
    """Compute Normal log-likelihood given variances"""
    ...

def _studentt_ll(
    sigma2_ptr: _IntPtr,  # Conditional variances
    r2os2_ptr: _IntPtr,   # Residuals squared over sigma squared (eps2/sigma2)
    n: _Size,             # Number of observations
    nu: float,            # Degrees of freedom parameter
) -> float:
    """Compute Student-t log-likelihood given variances"""
    ...

def _skewt_ll(
    resid_ptr: _IntPtr,   # Residuals (not squared)
    sigma2_ptr: _IntPtr,  # Conditional variances
    n: _Size,             # Number of observations
    nu: float,            # Degrees of freedom parameter (> 2)
    lam: float,           # Asymmetry parameter (-1, 1)
) -> float:
    """
    Hansen (1994) Skewed Student-t log-likelihood.
    
    Parameters
    ----------
    resid_ptr : Pointer to residuals array (not squared)
    sigma2_ptr : Pointer to conditional variances array
    n : Number of observations
    nu : Degrees of freedom (must be > 2)
    lam : Asymmetry parameter (must be in (-1, 1))
    
    Returns
    -------
    ll : Log-likelihood value
    """
    ...

def _skewt_nll(
    resid_ptr: _IntPtr,   # Residuals (not squared)
    sigma2_ptr: _IntPtr,  # Conditional variances
    n: _Size,             # Number of observations
    nu: float,            # Degrees of freedom parameter (> 2)
    lam: float,           # Asymmetry parameter (-1, 1)
) -> float:
    """Hansen (1994) Skewed Student-t negative log-likelihood."""
    ...

def _garch_ll_grad_11_skewt(
    theta_ptr: _IntPtr,   # [omega, alpha, beta, nu, lam]
    y_ptr: _IntPtr,       # Returns data
    grad_ptr: _IntPtr,    # Output: gradient [5] (modified in-place)
    n: _Size,
) -> float:
    """GARCH(1,1) + Skew-t NLL with analytical gradient. Returns NLL."""
    ...

# ── Standard error computation (OPG & Hessian) ────────────────────────────

def _garch_opg_hess_pq(
    params_ptr: _IntPtr,  # GARCH parameters [omega, alpha1, ..., alphap, beta1, ..., betaq]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Conditional variances
    OPG_ptr: _IntPtr,     # Output: Outer Product of Gradients matrix (modified in-place)
    HESS_ptr: _IntPtr,    # Output: Hessian matrix (modified in-place)
    n: _Size,             # Number of observations
    p: _Size,             # GARCH order (number of alpha parameters)
    q: _Size,             # ARCH order (number of beta parameters)
) -> None:
    """Compute GARCH(p,q) OPG and Hessian matrices for robust standard errors (with recursive derivatives)"""
    ...

def _garch_opg_hess_11(
    params_ptr: _IntPtr,  # GARCH(1,1) parameters [omega, alpha, beta]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Conditional variances
    OPG_ptr: _IntPtr,     # Output: Outer Product of Gradients matrix (modified in-place)
    HESS_ptr: _IntPtr,    # Output: Hessian matrix (modified in-place)
    n: _Size,             # Number of observations
) -> None:
    """Compute GARCH(1,1) OPG and Hessian matrices for robust standard errors (with recursive derivatives)"""
    ...

def _garch_ll_grad_hess_pq_normal(
    theta_ptr: _IntPtr,   # GARCH parameters [omega, alpha1, ..., alphap, beta1, ..., betaq]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Working array for conditional variances
    grad_ptr: _IntPtr,    # Output: gradient vector (modified in-place)
    hess_ptr: _IntPtr,    # Output: Hessian matrix (modified in-place)
    nll_ptr: _IntPtr,     # Output: negative log-likelihood (scalar)
    n: _Size,             # Number of observations
    p: _Size,             # GARCH order (number of alpha parameters)
    q: _Size,             # ARCH order (number of beta parameters)
) -> None:
    """Compute GARCH(p,q) + Normal log-likelihood with gradient and Hessian"""
    ...


# void garch_ll_grad_hess_11_normal(
#         const double * __restrict params,  /* [ω, α, β] */
#         const double * __restrict resid2,
#         double       * __restrict sigma2,   /* n        */
#         double       * __restrict grad,     /* 3        */
#         double       * __restrict hess,     /* 3×3      */
#         double       * __restrict nll,      /* scalar   */
#         size_t n)

def _garch_ll_grad_hess_11_normal(
    theta_ptr: _IntPtr,   # GARCH(1,1) parameters [omega, alpha, beta]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Working array for conditional variances
    grad_ptr: _IntPtr,    # Output: gradient vector (modified in-place)
    hess_ptr: _IntPtr,    # Output: Hessian matrix (modified in-place)
    nll_ptr: _IntPtr,     # Output: negative log-likelihood (scalar)
    n: _Size,             # Number of observations
) -> None:
    """Compute GARCH(1,1) + Normal log-likelihood with gradient and Hessian (optimized)"""
    ...


# // __attribute__((visibility("default"), hot, flatten))
# // void garch_ll_grad_11_normal(
# //         const double * __restrict params,   /* [ω, α, β]          */
# //         const double * __restrict resid2,   /* ε_t², length n     */
# //         double       * __restrict sigma2,   /* working buffer n   */
# //         double       * __restrict grad,     /* output length 3    */
# //         size_t n)

# // __attribute__((visibility("default"), hot, flatten))
# // void garch_ll_hess_11_normal(
# //         const double * __restrict params,   /* [ω, α, β]          */
# //         const double * __restrict resid2,   /* ε_t², length n     */
# //         double       * __restrict sigma2,   /* working buffer n   */
# //         double       * __restrict hess,     /* output 3 × 3 row-major */
# //         size_t n)

def _garch_ll_grad_11_normal(
    theta_ptr: _IntPtr,   # GARCH(1,1) parameters [omega, alpha, beta]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Working array for conditional variances
    grad_ptr: _IntPtr,    # Output: gradient vector (modified in-place)
    n: _Size,             # Number of observations
) -> None:
    """Compute GARCH(1,1) + Normal log-likelihood gradient (optimized)"""
    ...

def _garch_ll_hess_11_normal(
    theta_ptr: _IntPtr,   # GARCH(1,1) parameters [omega, alpha, beta]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Working array for conditional variances
    hess_ptr: _IntPtr,    # Output: Hessian matrix (modified in-place)
    n: _Size,             # Number of observations
) -> None:
    """Compute GARCH(1,1) + Normal log-likelihood Hessian (optimized)"""
    ...



# // __attribute__((visibility("default"), hot, flatten))
# // void garch_ll_grad_pq_normal(
# //         const double * __restrict params,
# //         const double * __restrict resid2,
# //         double       * __restrict sigma2,
# //         double       * __restrict grad,
# //         size_t n,
# //         size_t p,
# //         size_t q)

# // __attribute__((visibility("default"), hot, flatten))
# // void garch_ll_hess_pq_normal(
# //         const double * __restrict params,
# //         const double * __restrict resid2,
# //         double       * __restrict sigma2,
# //         double       * __restrict hess,
# //         size_t n,
# //         size_t p,
# //         size_t q)

def _garch_ll_grad_pq_normal(
    theta_ptr: _IntPtr,   # GARCH parameters [omega, alpha1, ..., alphap, beta1, ..., betaq]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Working array for conditional variances
    grad_ptr: _IntPtr,    # Output: gradient vector (modified in-place)
    n: _Size,             # Number of observations
    p: _Size,             # GARCH order (number of alpha parameters)
    q: _Size,             # ARCH order (number of beta parameters)
) -> None:
    """Compute GARCH(p,q) + Normal log-likelihood gradient"""
    ...

def _garch_ll_hess_pq_normal(
    theta_ptr: _IntPtr,   # GARCH parameters [omega, alpha1, ..., alphap, beta1, ..., betaq]
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Working array for conditional variances
    hess_ptr: _IntPtr,    # Output: Hessian matrix (modified in-place)
    n: _Size,             # Number of observations
    p: _Size,             # GARCH order (number of alpha parameters)
    q: _Size,             # ARCH order (number of beta parameters)
) -> None:
    """Compute GARCH(p,q) + Normal log-likelihood Hessian"""
    ...


def _garch_ll_11_studentt(
    theta_ptr: _IntPtr,
    eps2_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    n: _Size,
) -> float: ...

def _garch_ll_pq_studentt(
    theta_ptr: _IntPtr,
    eps2_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    n: _Size,
    p: _Size,
    q: _Size,
) -> float: ...

def _garch_ll_grad_11_studentt(
    theta_ptr: _IntPtr,
    eps2_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    grad_ptr: _IntPtr,
    n: _Size,
) -> None: ...

def _garch_ll_hess_11_studentt(
    theta_ptr: _IntPtr,
    eps2_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    hess_ptr: _IntPtr,
    n: _Size,
) -> None: ...

def _garch_ll_grad_pq_studentt(
    theta_ptr: _IntPtr,
    eps2_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    grad_ptr: _IntPtr,
    n: _Size,
    p: _Size,
    q: _Size,
) -> None: ...

def _garch_ll_hess_pq_studentt(
    theta_ptr: _IntPtr,
    eps2_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    hess_ptr: _IntPtr,
    n: _Size,
    p: _Size,
    q: _Size,
) -> None: ...

# ── Log-space transforms ───────────────────────────────────────────────────

# GARCH(1,1) specialized versions
def _pack_garch_11(z_ptr: _IntPtr, theta_ptr: _IntPtr) -> None:
    """Transform z -> theta for GARCH(1,1). theta is modified in-place."""
    ...

def _pack_garch_studentt_11(z_ptr: _IntPtr, theta_ptr: _IntPtr) -> None:
    """Transform z -> theta for GARCH(1,1)+StudentT. theta is modified in-place."""
    ...

def _pack_garch_skewt_11(z_ptr: _IntPtr, theta_ptr: _IntPtr) -> None:
    """Transform z -> theta for GARCH(1,1)+SkewT. theta is modified in-place."""
    ...

def _jacobian_garch_11(theta_ptr: _IntPtr, J_ptr: _IntPtr) -> None:
    """Compute Jacobian J = d(theta)/d(z) for GARCH(1,1). J is 3x3, row-major."""
    ...

def _jacobian_garch_studentt_11(theta_ptr: _IntPtr, J_ptr: _IntPtr) -> None:
    """Compute Jacobian J = d(theta)/d(z) for GARCH(1,1)+StudentT. J is 4x4, row-major."""
    ...

def _jacobian_garch_skewt_11(theta_ptr: _IntPtr, J_ptr: _IntPtr) -> None:
    """Compute Jacobian J = d(theta)/d(z) for GARCH(1,1)+SkewT. J is 5x5, row-major."""
    ...

def _transform_grad_11_normal(
    grad_theta_ptr: _IntPtr,
    J_ptr: _IntPtr,
    grad_z_ptr: _IntPtr,
) -> None:
    """Compute grad_z = J^T @ grad_theta for K=3. grad_z is modified in-place."""
    ...

def _transform_grad_11_studentt(
    grad_theta_ptr: _IntPtr,
    J_ptr: _IntPtr,
    grad_z_ptr: _IntPtr,
) -> None:
    """Compute grad_z = J^T @ grad_theta for K=4. grad_z is modified in-place."""
    ...

def _transform_grad_11_skewt(
    grad_theta_ptr: _IntPtr,
    J_ptr: _IntPtr,
    grad_z_ptr: _IntPtr,
) -> None:
    """Compute grad_z = J^T @ grad_theta for K=5. grad_z is modified in-place."""
    ...

# General GARCH(p,q) versions
def _pack_garch_pq(z_ptr: _IntPtr, theta_ptr: _IntPtr, p: _Size, q: _Size) -> None:
    """Transform z -> theta for GARCH(p,q). theta is modified in-place."""
    ...

def _pack_garch_studentt_pq(z_ptr: _IntPtr, theta_ptr: _IntPtr, p: _Size, q: _Size) -> None:
    """Transform z -> theta for GARCH(p,q)+StudentT. theta is modified in-place."""
    ...

def _pack_garch_skewt_pq(z_ptr: _IntPtr, theta_ptr: _IntPtr, p: _Size, q: _Size) -> None:
    """Transform z -> theta for GARCH(p,q)+SkewT. theta is modified in-place."""
    ...

def _jacobian_garch_pq(theta_ptr: _IntPtr, J_ptr: _IntPtr, p: _Size, q: _Size) -> None:
    """Compute Jacobian J = d(theta)/d(z) for GARCH(p,q). J is K*K, row-major."""
    ...

def _jacobian_garch_studentt_pq(theta_ptr: _IntPtr, J_ptr: _IntPtr, p: _Size, q: _Size) -> None:
    """Compute Jacobian J = d(theta)/d(z) for GARCH(p,q)+StudentT. J is K*K, row-major."""
    ...

def _jacobian_garch_skewt_pq(theta_ptr: _IntPtr, J_ptr: _IntPtr, p: _Size, q: _Size) -> None:
    """Compute Jacobian J = d(theta)/d(z) for GARCH(p,q)+SkewT. J is K*K, row-major."""
    ...

def _transform_grad_pq(
    grad_theta_ptr: _IntPtr,
    J_ptr: _IntPtr,
    grad_z_ptr: _IntPtr,
    K: _Size,
) -> None:
    """Compute grad_z = J^T @ grad_theta for general K. grad_z is modified in-place."""
    ...

# ── ARMA-GARCH Functions ──────────────────────────────────────────────────

def _arma_garch_nll_11_normal(
    params_ptr: _IntPtr,   # [c, phi, theta, omega, alpha, beta]
    y_ptr: _IntPtr,        # Observations
    resid_ptr: _IntPtr,    # Output: residuals
    sigma2_ptr: _IntPtr,   # Output: variances
    h0: float,             # Initial variance
    n: _Size,              # Number of observations
) -> float:
    """ARMA(1,1)-GARCH(1,1) negative log-likelihood with Normal innovations."""
    ...

def _arma_garch_nll_grad_11_normal(
    params_ptr: _IntPtr,   # [c, phi, theta, omega, alpha, beta]
    y_ptr: _IntPtr,        # Observations
    resid_ptr: _IntPtr,    # Output: residuals
    sigma2_ptr: _IntPtr,   # Output: variances
    grad_ptr: _IntPtr,     # Output: gradient (6 elements)
    h0: float,             # Initial variance
    n: _Size,              # Number of observations
) -> float:
    """ARMA(1,1)-GARCH(1,1) NLL and gradient with Normal innovations."""
    ...

def _arma_garch_nll_11_studentt(
    params_ptr: _IntPtr,   # [c, phi, theta, omega, alpha, beta, nu]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    h0: float,
    n: _Size,
) -> float:
    """ARMA(1,1)-GARCH(1,1) NLL with Student-t innovations."""
    ...

def _arma_garch_nll_grad_11_studentt(
    params_ptr: _IntPtr,   # [c, phi, theta, omega, alpha, beta, nu]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    grad_ptr: _IntPtr,     # Output: gradient (7 elements, modified in-place)
    h0: float,
    n: _Size,
) -> float:
    """ARMA(1,1)-GARCH(1,1) NLL with analytical gradient for Student-t innovations."""
    ...

def _arma_garch_nll_11_skewt(
    params_ptr: _IntPtr,   # [c, phi, theta, omega, alpha, beta, nu, lam]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    h0: float,
    n: _Size,
) -> float:
    """ARMA(1,1)-GARCH(1,1) NLL with Skew-t innovations."""
    ...

def _arma_garch_nll_pq_normal(
    params_ptr: _IntPtr,   # [c, phi_1..phi_p, theta_1..theta_q, omega, alpha_1..alpha_P, beta_1..beta_Q]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    e0_ptr: _IntPtr,       # Initial residuals
    h0_ptr: _IntPtr,       # Initial variances
    n: _Size,
    p_ar: _Size,
    q_ma: _Size,
    P_arch: _Size,
    Q_garch: _Size,
) -> float:
    """General ARMA(p,q)-GARCH(P,Q) NLL with Normal innovations."""
    ...

def _arma_garch_nll_pq_studentt(
    params_ptr: _IntPtr,   # [..., nu]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    e0_ptr: _IntPtr,
    h0_ptr: _IntPtr,
    n: _Size,
    p_ar: _Size,
    q_ma: _Size,
    P_arch: _Size,
    Q_garch: _Size,
) -> float:
    """General ARMA(p,q)-GARCH(P,Q) NLL with Student-t innovations."""
    ...

def _arma_garch_nll_pq_skewt(
    params_ptr: _IntPtr,   # [..., nu, lam]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,
    sigma2_ptr: _IntPtr,
    e0_ptr: _IntPtr,
    h0_ptr: _IntPtr,
    n: _Size,
    p_ar: _Size,
    q_ma: _Size,
    P_arch: _Size,
    Q_garch: _Size,
) -> float:
    """General ARMA(p,q)-GARCH(P,Q) NLL with Skew-t innovations."""
    ...

# =============================================================================
# Pure ARMA (no volatility dynamics) - Concentrated Likelihood
# =============================================================================

def _arma_nll_11_normal(
    params_ptr: _IntPtr,   # [c, phi, theta]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,    # Output: residuals
    n: _Size,
) -> float:
    """ARMA(1,1) NLL with Normal (concentrated likelihood)."""
    ...

def _arma_nll_grad_11_normal(
    params_ptr: _IntPtr,   # [c, phi, theta]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,    # Output: residuals
    grad_ptr: _IntPtr,     # Output: gradient (3 elements)
    n: _Size,
) -> float:
    """ARMA(1,1) NLL with gradient (concentrated likelihood)."""
    ...

def _arma_hess_11_normal(
    params_ptr: _IntPtr,   # [c, phi, theta]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,    # Working array
    hess_ptr: _IntPtr,     # Output: 3x3 Hessian
    n: _Size,
) -> None:
    """ARMA(1,1) Hessian (expected, concentrated likelihood)."""
    ...

def _arma_nll_pq_normal(
    params_ptr: _IntPtr,   # [c, phi_1..phi_p, theta_1..theta_q]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,    # Output: residuals
    e0_ptr: _IntPtr,       # Initial residuals (q elements)
    n: _Size,
    p_ar: _Size,
    q_ma: _Size,
) -> float:
    """ARMA(p,q) NLL with Normal (concentrated likelihood)."""
    ...

def _arma_nll_grad_pq_normal(
    params_ptr: _IntPtr,   # [c, phi_1..phi_p, theta_1..theta_q]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,    # Output: residuals
    e0_ptr: _IntPtr,       # Initial residuals
    grad_ptr: _IntPtr,     # Output: gradient (1+p+q elements)
    n: _Size,
    p_ar: _Size,
    q_ma: _Size,
) -> float:
    """ARMA(p,q) NLL with gradient (concentrated likelihood)."""
    ...

def _arma_hess_pq_normal(
    params_ptr: _IntPtr,   # [c, phi_1..phi_p, theta_1..theta_q]
    y_ptr: _IntPtr,
    resid_ptr: _IntPtr,    # Working array
    e0_ptr: _IntPtr,       # Initial residuals
    hess_ptr: _IntPtr,     # Output: (1+p+q)x(1+p+q) Hessian
    n: _Size,
    p_ar: _Size,
    q_ma: _Size,
) -> None:
    """ARMA(p,q) Hessian (expected, concentrated likelihood)."""
    ...

# Nothing is meant for star-import; keep top-level clean
__all__: list[str] = []