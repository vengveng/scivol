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

# ── Standard error computation (OPG & Hessian) ────────────────────────────

def _garch_opg_hess_pq(
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Conditional variances
    OPG_ptr: _IntPtr,     # Output: Outer Product of Gradients matrix (modified in-place)
    HESS_ptr: _IntPtr,    # Output: Hessian matrix (modified in-place)
    n: _Size,             # Number of observations
    p: _Size,             # GARCH order (number of alpha parameters)
    q: _Size,             # ARCH order (number of beta parameters)
) -> None:
    """Compute GARCH(p,q) OPG and Hessian matrices for robust standard errors"""
    ...

def _garch_opg_hess_11(
    eps2_ptr: _IntPtr,    # Squared residuals/returns
    sigma2_ptr: _IntPtr,  # Conditional variances
    OPG_ptr: _IntPtr,     # Output: Outer Product of Gradients matrix (modified in-place)
    HESS_ptr: _IntPtr,    # Output: Hessian matrix (modified in-place)
    n: _Size,             # Number of observations
) -> None:
    """Compute GARCH(1,1) OPG and Hessian matrices for robust standard errors (optimized)"""
    ...

# Nothing is meant for star-import; keep top-level clean
__all__: list[str] = []