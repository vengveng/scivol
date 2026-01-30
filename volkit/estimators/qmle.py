"""
QMLE Estimator with Robust (Sandwich) Standard Errors.

Uses Normal log-likelihood for parameter estimation, but computes robust
standard errors that are valid even when the true distribution is non-Normal.

The sandwich covariance estimator is:
    V_robust = H^{-1} @ OPG @ H^{-1}

where:
    H = Hessian of the negative log-likelihood
    OPG = Outer Product of Gradients = sum_t (g_t @ g_t')
    g_t = score (gradient of -log L) at observation t
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

# ── intra-package imports (relative) ──────────────────────────────────
from ..spec import CompositeSpec
from ..components import Component
from ..roles import Role
from .._kernels import get_routine
from .base import Estimator
from .. import _core

if TYPE_CHECKING:
    from ..result import EstimationResult


def _as_cptr(a: NDArray[np.float64]) -> int:
    """Convert numpy array to C pointer (as integer address)."""
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _compute_robust_se_c(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute OPG and Hessian using C extensions.
    
    Returns (opg, hess) matrices.
    """
    n = len(resid2)
    k = 1 + p + q  # GARCH params only (no distribution params for Normal)
    
    opg = np.zeros((k, k), dtype=np.float64)
    hess = np.zeros((k, k), dtype=np.float64)
    
    # Try specialized (1,1) version first
    if p == 1 and q == 1:
        try:
            _core._garch_opg_hess_11(
                _as_cptr(resid2),
                _as_cptr(sigma2),
                _as_cptr(opg),
                _as_cptr(hess),
                n,
            )
            return opg, hess
        except (AttributeError, TypeError):
            pass
    
    # Fall back to general (p,q) version
    try:
        _core._garch_opg_hess_pq(
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(opg),
            _as_cptr(hess),
            n, p, q,
        )
        return opg, hess
    except (AttributeError, TypeError):
        pass
    
    raise RuntimeError("C extension OPG/Hessian computation not available")


def _compute_robust_se_python(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute OPG and Hessian using pure Python (fallback).
    
    Only supports GARCH(1,1) + Normal.
    
    Returns (opg, hess) matrices.
    """
    # Import from likelihoods module
    import sys
    from pathlib import Path
    
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    
    from likelihoods import garch11_normal_robust_se
    
    omega, alpha, beta = params[0], params[1], params[2]
    sigma2_init = float(sigma2[0])
    
    result = garch11_normal_robust_se(params[:3], resid2, sigma2_init)
    
    return result.opg, result.hess


class QMLE(Estimator):
    """
    Quasi-Maximum Likelihood Estimator with robust (sandwich) standard errors.
    
    Uses Normal log-likelihood for parameter estimation, then computes
    robust standard errors via the sandwich covariance estimator.
    
    Usage:
        spec = GARCH(1, 1) + Normal()
        estimator = QMLE()
        result = estimator.fit(spec, data)
        
        # Access robust standard errors
        print(result.std_errors_robust)
    """
    
    def fit(
        self,
        spec: Union[CompositeSpec, Component],
        data: np.ndarray,
        solver: str = "trust",
        verbose: bool = False,
        **kwargs: Any,
    ) -> EstimationResult:
        """
        Fit GARCH model via QMLE with robust standard errors.
        
        Parameters
        ----------
        spec : CompositeSpec or Component
            Model specification (e.g., GARCH(1,1) + Normal())
        data : array
            Residual series (demeaned returns or AR residuals)
        solver : str
            Optimization method: "nelder-mead", "slsqp", "trust"
        verbose : bool
            Print progress
        **kwargs
            Additional arguments passed to the kernel fit function
            
        Returns
        -------
        EstimationResult
            Contains both MLE and robust standard errors
        """
        from ..result import EstimationResult
        
        t_start = time.perf_counter()
        
        # Validate inputs
        spec = self._validate_spec(spec)
        data = self._validate_data(data)
        self._warn_small_sample(spec, data)
        
        # Check for Normal density (QMLE requires Normal likelihood)
        density = None
        for comp in spec.components:
            if comp.role == Role.DENSITY:
                density = comp
                break
        
        if density is None or density.signature != "Normal":
            warnings.warn(
                "QMLE is designed for Normal density. "
                "For Student-t or Skew-t, consider using MLE with the correct distribution."
            )
        
        # Get GARCH component to extract p, q
        vol = None
        for comp in spec.components:
            if comp.role == Role.VOLATILITY:
                vol = comp
                break
        
        if vol is None:
            raise ValueError("Spec must include a volatility component (e.g., GARCH)")
        
        p, q = vol.p, vol.q
        
        # Step 1: Run MLE optimization
        routine = get_routine(str(spec))
        result = routine.fit(data, solver=solver, verbose=verbose, **kwargs)
        
        # Step 2: Compute robust standard errors
        if result.sigma2 is None:
            warnings.warn("Cannot compute robust SEs: sigma2 not available from estimation")
            self._last_result = result
            return result
        
        resid2 = data ** 2
        
        try:
            # Try C extension first
            opg, hess = _compute_robust_se_c(result.params, resid2, result.sigma2, p, q)
        except RuntimeError:
            # Fall back to Python implementation
            if p != 1 or q != 1:
                warnings.warn(
                    f"Robust SE computation only supports GARCH(1,1) in Python fallback. "
                    f"Got GARCH({p},{q}). Returning MLE standard errors only."
                )
                self._last_result = result
                return result
            
            try:
                opg, hess = _compute_robust_se_python(result.params, resid2, result.sigma2)
            except Exception as e:
                warnings.warn(f"Robust SE computation failed: {e}")
                self._last_result = result
                return result
        
        # Step 3: Compute covariance matrices
        try:
            hess_inv = np.linalg.inv(hess)
            cov_mle = hess_inv
            cov_robust = hess_inv @ opg @ hess_inv
        except np.linalg.LinAlgError:
            warnings.warn("Hessian is singular, cannot compute covariance matrix")
            self._last_result = result
            return result
        
        t_elapsed = time.perf_counter() - t_start
        
        # Step 4: Create enhanced result with robust SEs
        # We need to create a new EstimationResult with the additional information
        enhanced_result = EstimationResult(
            spec=result.spec,
            optimization_result=result.optimization_result,
            data=data,
            sigma2=result.sigma2,
            time_elapsed=t_elapsed,
            hessian=hess,
            cov_matrix=cov_mle,
            opg=opg,
            cov_robust=cov_robust,
            method="QMLE",
        )
        
        self._last_result = enhanced_result
        
        if verbose:
            se_mle = enhanced_result.std_errors
            se_robust = enhanced_result.std_errors_robust
            if se_mle is not None and se_robust is not None:
                print("Robust (sandwich) standard errors computed successfully.")
                param_names = ["omega"] + [f"alpha_{i+1}" for i in range(p)] + [f"beta_{j+1}" for j in range(q)]
                for i, name in enumerate(param_names):
                    print(f"  SE({name}):  MLE={se_mle[i]:.6f}, Robust={se_robust[i]:.6f}")
        
        return enhanced_result
