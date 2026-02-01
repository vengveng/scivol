"""
QMLE Estimator with Robust (Sandwich) Standard Errors.

Supports:
- GARCH + Normal: Robust SEs for all parameters
- GARCH + StudentT: Robust SEs for GARCH params, MLE SE for nu (two-step)
- GARCH + SkewT: Robust SEs for GARCH params, MLE SE for nu, lam (two-step)

The sandwich covariance estimator is:
    V_robust = H^{-1} @ OPG @ H^{-1}

where:
    H = Hessian of the negative log-likelihood
    OPG = Outer Product of Gradients = sum_t (g_t @ g_t')
    g_t = score (gradient of -log L) at observation t

For Student-t and Skew-t, we use a two-step procedure:
1. Fit GARCH with Normal LL → robust SEs for [omega, alpha, beta]
2. Fix GARCH, fit shape params → MLE SEs for [nu] or [nu, lam]

The final covariance is block-diagonal (GARCH and shape blocks are independent).
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, Bounds

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
    
    Returns (opg, hess) matrices for GARCH parameters only.
    """
    n = len(resid2)
    k = 1 + p + q  # GARCH params only (no distribution params for Normal)
    
    # Extract GARCH params only (first k elements)
    garch_params = params[:k].copy()
    
    opg = np.zeros((k, k), dtype=np.float64)
    hess = np.zeros((k, k), dtype=np.float64)
    
    # Try specialized (1,1) version first
    if p == 1 and q == 1:
        try:
            _core._garch_opg_hess_11(
                _as_cptr(garch_params),
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
            _as_cptr(garch_params),
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


def _numerical_hessian(
    func,
    x: NDArray[np.float64],
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """
    Compute numerical Hessian using central differences.
    """
    n = len(x)
    H = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            
            H[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * eps * eps)
            H[j, i] = H[i, j]
    
    return H


def _fit_studentt_shape(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    verbose: bool = False,
) -> tuple[float, float, NDArray[np.float64]]:
    """
    Fit Student-t degrees of freedom (nu) with GARCH fixed.
    
    Returns (nu_hat, nll, hessian_nu).
    """
    n = len(resid)
    resid2 = resid ** 2
    
    # Precompute z² = resid²/sigma²
    z2 = resid2 / np.maximum(sigma2, 1e-12)
    z2_ptr = _as_cptr(z2)
    sigma2_ptr = _as_cptr(sigma2)
    
    def nll_nu(nu_arr):
        nu = nu_arr[0]
        if nu <= 2.01 or nu > 100:
            return 1e10
        # Use C function for likelihood
        ll = _core._studentt_ll(sigma2_ptr, z2_ptr, n, nu)
        return -ll / n  # Negative log-likelihood, scaled
    
    # Optimize
    result = minimize(
        nll_nu,
        x0=np.array([8.0]),
        method="L-BFGS-B",
        bounds=[(2.01, 100.0)],
        options={"maxiter": 1000, "disp": verbose}
    )
    
    nu_hat = result.x[0]
    nll = result.fun * n  # Unscale
    
    # Compute Hessian for SE
    hess = _numerical_hessian(nll_nu, result.x, eps=1e-5) * n
    
    return nu_hat, nll, hess


def _fit_skewt_shape(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    verbose: bool = False,
) -> tuple[float, float, float, NDArray[np.float64]]:
    """
    Fit Skew-t shape parameters (nu, lambda) with GARCH fixed.
    
    Returns (nu_hat, lam_hat, nll, hessian_shape).
    """
    n = len(resid)
    resid_ptr = _as_cptr(resid)
    sigma2_ptr = _as_cptr(sigma2)
    
    def nll_shape(params):
        nu, lam = params[0], params[1]
        if nu <= 2.01 or nu > 100:
            return 1e10
        if lam <= -0.99 or lam >= 0.99:
            return 1e10
        # Use C function for likelihood
        ll = _core._skewt_ll(resid_ptr, sigma2_ptr, n, nu, lam)
        if not np.isfinite(ll):
            return 1e10
        return -ll / n  # Negative log-likelihood, scaled
    
    # Optimize
    result = minimize(
        nll_shape,
        x0=np.array([8.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(2.01, 100.0), (-0.99, 0.99)],
        options={"maxiter": 1000, "disp": verbose}
    )
    
    nu_hat = result.x[0]
    lam_hat = result.x[1]
    nll = result.fun * n  # Unscale
    
    # Compute Hessian for SE
    hess = _numerical_hessian(nll_shape, result.x, eps=1e-5) * n
    
    return nu_hat, lam_hat, nll, hess


class QMLE(Estimator):
    """
    Quasi-Maximum Likelihood Estimator with robust (sandwich) standard errors.
    
    Supports three distribution types:
    
    1. **Normal**: Standard QMLE with robust SEs for [omega, alpha, beta]
    
    2. **StudentT**: Two-step procedure
       - Step 1: Fit GARCH with Normal LL → robust SEs for GARCH params
       - Step 2: Fix GARCH, fit nu → MLE SE for nu
    
    3. **SkewT**: Two-step procedure
       - Step 1: Fit GARCH with Normal LL → robust SEs for GARCH params
       - Step 2: Fix GARCH, fit [nu, lam] → MLE SEs for shape params
    
    Usage:
        # Normal
        spec = GARCH(1, 1) + Normal()
        result = QMLE().fit(spec, data)
        
        # Student-t with robust GARCH SEs
        spec = GARCH(1, 1) + StudentT()
        result = QMLE().fit(spec, data)
        
        # Skew-t with robust GARCH SEs
        spec = GARCH(1, 1) + SkewT()
        result = QMLE().fit(spec, data)
    """
    
    def fit(
        self,
        spec: Union[CompositeSpec, Component],
        data: np.ndarray,
        solver: str = "slsqp",
        verbose: bool = False,
        **kwargs: Any,
    ) -> EstimationResult:
        """
        Fit GARCH model via QMLE with robust standard errors.
        
        Parameters
        ----------
        spec : CompositeSpec or Component
            Model specification (e.g., GARCH(1,1) + StudentT())
        data : array
            Residual series (demeaned returns or AR residuals)
        solver : str
            Optimization method for GARCH: "nelder-mead", "slsqp", "trust"
        verbose : bool
            Print progress
        **kwargs
            Additional arguments passed to the kernel fit function
            
        Returns
        -------
        EstimationResult
            Contains both MLE and robust standard errors:
            - std_errors: MLE standard errors (for shape params)
            - std_errors_robust: Robust sandwich SEs (for GARCH params, 
              or full if Normal density)
        """
        from ..result import EstimationResult
        from ..components.density import Normal, StudentT, SkewT
        
        t_start = time.perf_counter()
        
        # Validate inputs
        spec = self._validate_spec(spec)
        data = self._validate_data(data)
        self._warn_small_sample(spec, data)
        
        # Identify density type
        density = None
        for comp in spec.components:
            if comp.role == Role.DENSITY:
                density = comp
                break
        
        density_type = "normal"
        if density is not None:
            sig = density.signature
            if sig == "StudentT":
                density_type = "studentt"
            elif sig == "SkewT":
                density_type = "skewt"
        
        # Get GARCH component
        vol = None
        for comp in spec.components:
            if comp.role == Role.VOLATILITY:
                vol = comp
                break
        
        if vol is None:
            raise ValueError("Spec must include a volatility component (e.g., GARCH)")
        
        p, q = vol.p, vol.q
        n_garch = 1 + p + q
        
        # =====================================================================
        # Step 1: Fit GARCH with Normal likelihood
        # =====================================================================
        
        # Build Normal spec for GARCH fitting
        from ..components.density import Normal as NormalDensity
        from ..components.vol import GARCH
        normal_spec = GARCH(p, q) + NormalDensity()
        
        # Get routine and fit
        routine = get_routine(str(normal_spec))
        garch_result = routine.fit(data, solver=solver, verbose=verbose, **kwargs)
        
        if garch_result.sigma2 is None:
            warnings.warn("Cannot compute robust SEs: sigma2 not available from estimation")
            self._last_result = garch_result
            return garch_result
        
        sigma2 = garch_result.sigma2
        garch_params = garch_result.params[:n_garch]
        
        # =====================================================================
        # Step 2: Compute robust SEs for GARCH parameters
        # =====================================================================
        
        resid2 = data ** 2
        
        try:
            opg_garch, hess_garch = _compute_robust_se_c(garch_params, resid2, sigma2, p, q)
        except RuntimeError:
            if p != 1 or q != 1:
                warnings.warn(
                    f"Robust SE computation only supports GARCH(1,1). "
                    f"Got GARCH({p},{q}). Returning MLE standard errors only."
                )
                self._last_result = garch_result
                return garch_result
            
            try:
                opg_garch, hess_garch = _compute_robust_se_python(garch_params, resid2, sigma2)
            except Exception as e:
                warnings.warn(f"Robust SE computation failed: {e}")
                self._last_result = garch_result
                return garch_result
        
        # Compute GARCH covariance matrices
        # Note: C function returns averaged H and OPG (divided by n)
        # For averaged quantities:
        #   I = n * H_avg, so Cov = I^{-1} = H_avg^{-1} / n
        #   Cov_robust = H_avg^{-1} @ OPG_avg @ H_avg^{-1} / n
        n = len(data)
        try:
            hess_inv_garch = np.linalg.inv(hess_garch)
            cov_mle_garch = hess_inv_garch / n
            cov_robust_garch = hess_inv_garch @ opg_garch @ hess_inv_garch / n
        except np.linalg.LinAlgError:
            warnings.warn("GARCH Hessian is singular, cannot compute covariance matrix")
            self._last_result = garch_result
            return garch_result
        
        # =====================================================================
        # Step 3: Handle distribution-specific fitting
        # =====================================================================
        
        if density_type == "normal":
            # Normal: we're done - just return robust SEs
            t_elapsed = time.perf_counter() - t_start
            
            enhanced_result = EstimationResult(
                spec=normal_spec,
                optimization_result=garch_result.optimization_result,
                data=data,
                sigma2=sigma2,
                time_elapsed=t_elapsed,
                hessian=hess_garch,
                cov_matrix=cov_mle_garch,
                opg=opg_garch,
                cov_robust=cov_robust_garch,
                method="QMLE",
            )
            
            # Update component fitted params
            vol.unpack(garch_params)
            
            self._last_result = enhanced_result
            
            if verbose:
                self._print_robust_se(enhanced_result, p, q)
            
            return enhanced_result
        
        elif density_type == "studentt":
            # Student-t: fit nu with GARCH fixed
            nu_hat, nll_shape, hess_nu = _fit_studentt_shape(data, sigma2, verbose)
            
            # Build full covariance (block diagonal)
            n_params = n_garch + 1  # GARCH + nu
            full_params = np.concatenate([garch_params, [nu_hat]])
            
            # Block diagonal covariance
            cov_full_robust = np.zeros((n_params, n_params), dtype=np.float64)
            cov_full_robust[:n_garch, :n_garch] = cov_robust_garch
            
            try:
                cov_nu = np.linalg.inv(hess_nu)
                cov_full_robust[n_garch:, n_garch:] = cov_nu
            except np.linalg.LinAlgError:
                cov_full_robust[n_garch, n_garch] = np.nan
            
            # MLE covariance (for shape params)
            cov_full_mle = np.zeros((n_params, n_params), dtype=np.float64)
            cov_full_mle[:n_garch, :n_garch] = cov_mle_garch
            cov_full_mle[n_garch:, n_garch:] = cov_full_robust[n_garch:, n_garch:]
            
            # Build full Hessian and OPG (block diagonal)
            hess_full = np.zeros((n_params, n_params), dtype=np.float64)
            hess_full[:n_garch, :n_garch] = hess_garch
            hess_full[n_garch:, n_garch:] = hess_nu
            
            opg_full = np.zeros((n_params, n_params), dtype=np.float64)
            opg_full[:n_garch, :n_garch] = opg_garch
            # Shape params OPG not available, use Hessian as proxy
            opg_full[n_garch:, n_garch:] = hess_nu
            
            t_elapsed = time.perf_counter() - t_start
            
            # Create fake optimization result
            class FakeOptResult:
                def __init__(self, x, fun, success):
                    self.x = x
                    self.fun = fun
                    self.success = success
                    self.nit = 0
                    self.message = "QMLE two-step estimation"
            
            # Compute total log-likelihood (Student-t with fitted params)
            resid_ptr = _as_cptr(data)
            z2 = resid2 / np.maximum(sigma2, 1e-12)
            z2_ptr = _as_cptr(z2)
            sigma2_ptr = _as_cptr(sigma2)
            ll_studentt = _core._studentt_ll(sigma2_ptr, z2_ptr, len(data), nu_hat)
            
            fake_opt = FakeOptResult(full_params, -ll_studentt, True)
            
            enhanced_result = EstimationResult(
                spec=spec,
                optimization_result=fake_opt,
                data=data,
                sigma2=sigma2,
                time_elapsed=t_elapsed,
                hessian=hess_full,
                cov_matrix=cov_full_mle,
                opg=opg_full,
                cov_robust=cov_full_robust,
                method="QMLE",
            )
            
            # Update component fitted params
            vol.unpack(garch_params)
            if density is not None:
                density.fitted_params = {'nu': nu_hat}
            
            self._last_result = enhanced_result
            
            if verbose:
                self._print_robust_se_studentt(enhanced_result, p, q, nu_hat)
            
            return enhanced_result
        
        elif density_type == "skewt":
            # Skew-t: fit [nu, lam] with GARCH fixed
            nu_hat, lam_hat, nll_shape, hess_shape = _fit_skewt_shape(data, sigma2, verbose)
            
            # Build full covariance (block diagonal)
            n_params = n_garch + 2  # GARCH + nu + lam
            full_params = np.concatenate([garch_params, [nu_hat, lam_hat]])
            
            # Block diagonal covariance
            cov_full_robust = np.zeros((n_params, n_params), dtype=np.float64)
            cov_full_robust[:n_garch, :n_garch] = cov_robust_garch
            
            try:
                cov_shape = np.linalg.inv(hess_shape)
                cov_full_robust[n_garch:, n_garch:] = cov_shape
            except np.linalg.LinAlgError:
                cov_full_robust[n_garch:, n_garch:] = np.nan
            
            # MLE covariance
            cov_full_mle = np.zeros((n_params, n_params), dtype=np.float64)
            cov_full_mle[:n_garch, :n_garch] = cov_mle_garch
            cov_full_mle[n_garch:, n_garch:] = cov_full_robust[n_garch:, n_garch:]
            
            # Build full Hessian and OPG (block diagonal)
            hess_full = np.zeros((n_params, n_params), dtype=np.float64)
            hess_full[:n_garch, :n_garch] = hess_garch
            hess_full[n_garch:, n_garch:] = hess_shape
            
            opg_full = np.zeros((n_params, n_params), dtype=np.float64)
            opg_full[:n_garch, :n_garch] = opg_garch
            opg_full[n_garch:, n_garch:] = hess_shape
            
            t_elapsed = time.perf_counter() - t_start
            
            # Create fake optimization result
            class FakeOptResult:
                def __init__(self, x, fun, success):
                    self.x = x
                    self.fun = fun
                    self.success = success
                    self.nit = 0
                    self.message = "QMLE two-step estimation"
            
            # Compute total log-likelihood (Skew-t with fitted params)
            resid_ptr = _as_cptr(data)
            sigma2_ptr = _as_cptr(sigma2)
            ll_skewt = _core._skewt_ll(resid_ptr, sigma2_ptr, len(data), nu_hat, lam_hat)
            
            fake_opt = FakeOptResult(full_params, -ll_skewt, True)
            
            enhanced_result = EstimationResult(
                spec=spec,
                optimization_result=fake_opt,
                data=data,
                sigma2=sigma2,
                time_elapsed=t_elapsed,
                hessian=hess_full,
                cov_matrix=cov_full_mle,
                opg=opg_full,
                cov_robust=cov_full_robust,
                method="QMLE",
            )
            
            # Update component fitted params
            vol.unpack(garch_params)
            if density is not None:
                density.fitted_params = {'nu': nu_hat, 'lam': lam_hat}
            
            self._last_result = enhanced_result
            
            if verbose:
                self._print_robust_se_skewt(enhanced_result, p, q, nu_hat, lam_hat)
            
            return enhanced_result
        
        else:
            raise ValueError(f"Unknown density type: {density_type}")
    
    def _print_robust_se(self, result, p: int, q: int):
        """Print robust SE info for Normal."""
        se_mle = result.std_errors
        se_robust = result.std_errors_robust
        if se_mle is not None and se_robust is not None:
            print("QMLE: Robust (sandwich) standard errors computed successfully.")
            param_names = ["omega"] + [f"alpha_{i+1}" for i in range(p)] + [f"beta_{j+1}" for j in range(q)]
            for i, name in enumerate(param_names):
                if i < len(se_mle) and i < len(se_robust):
                    print(f"  SE({name}):  MLE={se_mle[i]:.6f}, Robust={se_robust[i]:.6f}")
    
    def _print_robust_se_studentt(self, result, p: int, q: int, nu: float):
        """Print robust SE info for Student-t."""
        se_robust = result.std_errors_robust
        if se_robust is not None:
            print("QMLE (Student-t two-step): Robust GARCH SEs, MLE SE for nu.")
            param_names = ["omega"] + [f"alpha_{i+1}" for i in range(p)] + [f"beta_{j+1}" for j in range(q)] + ["nu"]
            for i, name in enumerate(param_names):
                if i < len(se_robust):
                    se_type = "Robust" if i < (1 + p + q) else "MLE"
                    print(f"  SE({name}):  {se_type}={se_robust[i]:.6f}")
            print(f"  nu (df) = {nu:.2f}")
    
    def _print_robust_se_skewt(self, result, p: int, q: int, nu: float, lam: float):
        """Print robust SE info for Skew-t."""
        se_robust = result.std_errors_robust
        if se_robust is not None:
            print("QMLE (Skew-t two-step): Robust GARCH SEs, MLE SEs for nu, lambda.")
            param_names = ["omega"] + [f"alpha_{i+1}" for i in range(p)] + [f"beta_{j+1}" for j in range(q)] + ["nu", "lambda"]
            for i, name in enumerate(param_names):
                if i < len(se_robust):
                    se_type = "Robust" if i < (1 + p + q) else "MLE"
                    print(f"  SE({name}):  {se_type}={se_robust[i]:.6f}")
            print(f"  nu (df) = {nu:.2f}, lambda = {lam:.4f}")
