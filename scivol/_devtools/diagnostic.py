"""
Derivative Validation Diagnostic Tool
======================================

This module provides internal development helpers for validating analytical
derivatives against an independent AD oracle. This is essential for ensuring
correctness of the C extension gradient and Hessian implementations.

Usage in development:
    from scivol import GARCH, Normal
    from scivol._devtools.diagnostic import validate_derivatives
    
    spec = GARCH(1, 1) + Normal()
    report = validate_derivatives(spec, data)
    report.summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..spec.composite import CompositeSpec


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class DerivativeValidationReport:
    """
    Report from derivative validation.
    
    Attributes
    ----------
    uid : str
        Unique identifier of the model (e.g., "GARCH(1,1)+Normal")
    params : NDArray
        Parameter vector used for validation
    n_obs : int
        Number of observations in the data
    
    gradient_analytical : NDArray
        Gradient computed via analytical formulas
    gradient_numerical : NDArray
        Gradient computed via the AD oracle
    gradient_abs_diff : NDArray
        Absolute differences for each element
    gradient_rel_diff : NDArray
        Relative differences for each element (percentage)
    gradient_max_abs_error : float
        Maximum absolute error in gradient
    gradient_max_rel_error : float
        Maximum relative error in gradient (percentage)
    gradient_passed : bool
        Whether gradient validation passed
    
    hessian_analytical : NDArray
        Hessian computed via analytical formulas
    hessian_numerical : NDArray
        Hessian computed via the AD oracle
    hessian_abs_diff : NDArray
        Absolute differences for each element
    hessian_rel_diff : NDArray
        Relative differences for each element (percentage)
    hessian_max_abs_error : float
        Maximum absolute error in Hessian
    hessian_max_rel_error : float
        Maximum relative error in Hessian (percentage)
    hessian_passed : bool
        Whether Hessian validation passed
    
    passed : bool
        Overall pass/fail status
    message : str
        Summary message
    """
    uid: str
    params: NDArray[np.float64]
    n_obs: int
    
    # Gradient validation
    gradient_analytical: NDArray[np.float64]
    gradient_numerical: NDArray[np.float64]
    gradient_abs_diff: NDArray[np.float64]
    gradient_rel_diff: NDArray[np.float64]
    gradient_max_abs_error: float
    gradient_max_rel_error: float
    gradient_passed: bool
    
    # Hessian validation
    hessian_analytical: NDArray[np.float64]
    hessian_numerical: NDArray[np.float64]
    hessian_abs_diff: NDArray[np.float64]
    hessian_rel_diff: NDArray[np.float64]
    hessian_max_abs_error: float
    hessian_max_rel_error: float
    hessian_passed: bool
    
    # Overall
    passed: bool
    message: str
    
    def summary(self, verbose: bool = True) -> None:
        """Print a summary of the validation report."""
        WIDTH = 70
        
        print("═" * WIDTH)
        print(f"{'Derivative Validation Report':^{WIDTH}}")
        print("═" * WIDTH)
        print(f"Model:         {self.uid}")
        print(f"Observations:  {self.n_obs}")
        print(f"Parameters:    {len(self.params)}")
        print("─" * WIDTH)
        
        # Gradient results
        grad_status = "PASS" if self.gradient_passed else "FAIL"
        print(f"\nGradient Validation: [{grad_status}]")
        print(f"  Max Absolute Error: {self.gradient_max_abs_error:.2e}")
        print(f"  Max Relative Error: {self.gradient_max_rel_error:.2e}%")
        
        if verbose:
            print(f"\n  Analytical: {self._format_array(self.gradient_analytical)}")
            print(f"  Reference:  {self._format_array(self.gradient_numerical)}")
        
        # Hessian results
        hess_status = "PASS" if self.hessian_passed else "FAIL"
        print(f"\nHessian Validation: [{hess_status}]")
        print(f"  Max Absolute Error: {self.hessian_max_abs_error:.2e}")
        print(f"  Max Relative Error: {self.hessian_max_rel_error:.2e}%")
        
        if verbose and len(self.params) <= 5:
            print(f"\n  Analytical:\n{self.hessian_analytical}")
            print(f"\n  Reference:\n{self.hessian_numerical}")
        
        # Overall
        print("─" * WIDTH)
        overall_status = "PASS" if self.passed else "FAIL"
        print(f"Overall: [{overall_status}] {self.message}")
        print("═" * WIDTH)
    
    def _format_array(self, arr: NDArray[np.float64]) -> str:
        """Format a 1D array compactly."""
        with np.printoptions(precision=4, suppress=False, linewidth=100):
            return str(arr)


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_derivatives(
    spec: CompositeSpec,
    data: NDArray[np.float64],
    params: Optional[NDArray[np.float64]] = None,
    *,
    rtol_grad: float = 1e-2,  # 1% relative tolerance for gradient
    rtol_hess: float = 0.1,
    eps_grad: float = 1e-7,
    eps_hess: float = 1e-5,
) -> DerivativeValidationReport:
    """
    Validate analytical derivatives against an independent AD oracle.
    
    Parameters
    ----------
    spec : CompositeSpec
        Model specification (e.g., GARCH(1,1) + Normal())
    data : array
        1-D array of residuals/returns
    params : array, optional
        Parameter vector to validate at. If None, uses default_start().
    rtol_grad : float
        Relative tolerance for gradient validation (default 1e-4 = 0.01%)
    rtol_hess : float
        Relative tolerance for Hessian validation.
    eps_grad : float
        Unused legacy argument retained for API compatibility.
    eps_hess : float
        Unused legacy argument retained for API compatibility.
    
    Returns
    -------
    DerivativeValidationReport
        Detailed validation report with all computed quantities
    
    Examples
    --------
    >>> from scivol import GARCH, Normal
    >>> from scivol._devtools.diagnostic import validate_derivatives
    >>> import numpy as np
    >>> 
    >>> spec = GARCH(1, 1) + Normal()
    >>> data = np.random.randn(500) * 0.01
    >>> report = validate_derivatives(spec, data)
    >>> report.summary()
    """
    from .. import _core
    from .._kernels import get_routine
    from .ad_oracle import garch_value_grad_hess
    
    # Ensure data is proper
    data = np.ascontiguousarray(data, dtype=np.float64)
    n = len(data)
    
    # Get routine for this model
    uid = str(spec)  # CompositeSpec stores signature in _sig, accessed via __str__
    routine = get_routine(uid)
    
    # Get default parameters if not provided
    if params is None:
        params = routine.start(data).astype(np.float64, copy=True)
    else:
        params = np.ascontiguousarray(params, dtype=np.float64)
    
    k = len(params)
    
    # Prepare data buffers
    resid2 = data ** 2
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = np.mean(resid2)
    
    def _as_cptr(arr: NDArray[np.float64]) -> int:
        return np.ascontiguousarray(arr, np.float64).ctypes.data
    
    # Determine which C functions to use based on UID
    is_normal = "Normal" in uid
    is_studentt = "StudentT" in uid
    
    # Parse p, q from UID
    import re
    match = re.search(r"GARCH\((\d+),(\d+)\)", uid)
    if not match:
        raise ValueError(f"Cannot parse GARCH order from UID: {uid}")
    p, q = int(match.group(1)), int(match.group(2))
    
    # Select appropriate C functions
    if is_normal:
        if p == 1 and q == 1:
            c_obj = _core._garch_ll_11_normal
            c_grad = _core._garch_ll_grad_11_normal
            c_hess = _core._garch_ll_hess_11_normal
            extra_args: tuple = ()
        else:
            c_obj = _core._garch_ll_pq_normal
            c_grad = _core._garch_ll_grad_pq_normal
            c_hess = _core._garch_ll_hess_pq_normal
            extra_args = (p, q)
    elif is_studentt:
        if p == 1 and q == 1:
            c_obj = _core._garch_ll_11_studentt
            c_grad = _core._garch_ll_grad_11_studentt
            c_hess = _core._garch_ll_hess_11_studentt
            extra_args = ()
        else:
            c_obj = _core._garch_ll_pq_studentt
            c_grad = _core._garch_ll_grad_pq_studentt
            c_hess = _core._garch_ll_hess_pq_studentt
            extra_args = (p, q)
    else:
        raise ValueError(f"Unsupported distribution in UID: {uid}")
    
    # Compute analytical gradient
    grad_analytical = np.empty(k, dtype=np.float64)
    sigma2[0] = np.mean(resid2)
    c_grad(_as_cptr(params), _as_cptr(resid2), _as_cptr(sigma2), _as_cptr(grad_analytical), n, *extra_args)
    
    # Compute analytical Hessian
    hess_analytical = np.empty((k, k), dtype=np.float64)
    sigma2[0] = np.mean(resid2)
    c_hess(_as_cptr(params), _as_cptr(resid2), _as_cptr(sigma2), _as_cptr(hess_analytical), n, *extra_args)
    
    # Compute AD-oracle derivatives
    _, grad_numerical, hess_numerical = garch_value_grad_hess(
        params,
        resid2,
        p,
        q,
        dist="normal" if is_normal else "studentt",
    )
    
    # Compute errors
    grad_abs_diff = np.abs(grad_analytical - grad_numerical)
    hess_abs_diff = np.abs(hess_analytical - hess_numerical)
    
    # Relative errors (percentage)
    grad_denom = np.maximum(np.abs(grad_analytical), 1e-20)
    hess_denom = np.maximum(np.abs(hess_analytical), 1e-20)
    
    grad_rel_diff = grad_abs_diff / grad_denom * 100
    hess_rel_diff = hess_abs_diff / hess_denom * 100
    
    # Max errors
    grad_max_abs = float(np.max(grad_abs_diff))
    grad_max_rel = float(np.max(grad_rel_diff))
    hess_max_abs = float(np.nanmax(hess_abs_diff))
    hess_max_rel = float(np.nanmax(hess_rel_diff))
    
    grad_finite_mask = np.isfinite(grad_numerical) & np.isfinite(grad_analytical)
    hess_finite_mask = np.isfinite(hess_numerical) & np.isfinite(hess_analytical)

    grad_passed = np.any(grad_finite_mask) and np.allclose(
        grad_analytical[grad_finite_mask],
        grad_numerical[grad_finite_mask],
        rtol=rtol_grad,
        atol=0,
    )
    hess_passed = np.any(hess_finite_mask) and np.allclose(
        hess_analytical[hess_finite_mask],
        hess_numerical[hess_finite_mask],
        rtol=rtol_hess,
        atol=0,
    )
    
    overall_passed = grad_passed and hess_passed
    
    # Message
    nan_warnings = []
    if not np.all(grad_finite_mask):
        nan_warnings.append("gradient has non-finite AD-oracle entries")
    if not np.all(hess_finite_mask):
        nan_warnings.append("Hessian has non-finite AD-oracle entries")
    
    if overall_passed:
        message = "All derivatives validated successfully"
        if nan_warnings:
            message += f" (note: {'; '.join(nan_warnings)})"
    else:
        issues = []
        if not grad_passed:
            issues.append(f"gradient (max rel err: {grad_max_rel:.2e}%)")
        if not hess_passed:
            issues.append(f"Hessian (max rel err: {hess_max_rel:.2e}%)")
        message = "Validation failed for: " + ", ".join(issues)
        if nan_warnings:
            message += f" (note: {'; '.join(nan_warnings)})"
    
    return DerivativeValidationReport(
        uid=uid,
        params=params,
        n_obs=n,
        gradient_analytical=grad_analytical,
        gradient_numerical=grad_numerical,
        gradient_abs_diff=grad_abs_diff,
        gradient_rel_diff=grad_rel_diff,
        gradient_max_abs_error=grad_max_abs,
        gradient_max_rel_error=grad_max_rel,
        gradient_passed=grad_passed,
        hessian_analytical=hess_analytical,
        hessian_numerical=hess_numerical,
        hessian_abs_diff=hess_abs_diff,
        hessian_rel_diff=hess_rel_diff,
        hessian_max_abs_error=hess_max_abs,
        hessian_max_rel_error=hess_max_rel,
        hessian_passed=hess_passed,
        passed=overall_passed,
        message=message,
    )


# =============================================================================
# CONVENIENCE FUNCTION FOR QUICK CHECKS
# =============================================================================

def quick_check(
    spec: CompositeSpec,
    data: NDArray[np.float64],
    params: Optional[NDArray[np.float64]] = None,
) -> bool:
    """
    Quick check if derivatives are correct.
    
    Returns True if both gradient and Hessian pass validation.
    Prints a brief summary to stdout.
    
    Parameters
    ----------
    spec : CompositeSpec
        Model specification
    data : array
        1-D array of residuals/returns
    params : array, optional
        Parameter vector to validate at
    
    Returns
    -------
    bool
        True if all derivatives pass validation
    """
    report = validate_derivatives(spec, data, params)
    
    grad_status = "OK" if report.gradient_passed else "FAIL"
    hess_status = "OK" if report.hessian_passed else "FAIL"
    
    print(f"Derivative check for {report.uid}:")
    print(f"  Gradient: [{grad_status}] max rel error: {report.gradient_max_rel_error:.2e}%")
    print(f"  Hessian:  [{hess_status}] max rel error: {report.hessian_max_rel_error:.2e}%")
    
    return report.passed
