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
from .._settings import settings

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


def _compute_arma_garch_robust_se(
    params: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    P_arch: int,
    Q_garch: int,
    h0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute OPG and Hessian for ARMA-GARCH+Normal using analytical per-observation
    gradient computation.
    
    Parameters
    ----------
    params : array [c, phi..., theta..., omega, alpha..., beta...]
    y : array of observations
    p_ar, q_ma : ARMA orders
    P_arch, Q_garch : GARCH orders
    h0 : initial variance
    
    Returns (opg, hess) - both are averaged (divided by n_eff)
    """
    n = len(y)
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    K = n_mean + n_vol
    
    # Only support ARMA(1,1)-GARCH(1,1) for now
    if not (p_ar == 1 and q_ma == 1 and P_arch == 1 and Q_garch == 1):
        raise NotImplementedError("ARMA-GARCH QMLE only supports ARMA(1,1)-GARCH(1,1) currently")
    
    # Parse parameters
    c = params[0]
    phi = params[1]
    theta_ma = params[2]
    omega = params[3]
    alpha = params[4]
    beta = params[5]
    
    # Compute residuals and variances
    resid = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    resid[0] = 0.0
    sigma2[0] = h0
    
    for t in range(1, n):
        resid[t] = y[t] - c - phi * y[t-1] - theta_ma * resid[t-1]
        sigma2[t] = omega + alpha * resid[t-1]**2 + beta * sigma2[t-1]
    
    # Compute per-observation gradients using analytical sensitivity recursion
    n_eff = n - 1
    
    # Sensitivity arrays (for mean params only, not omega, alpha, beta)
    de_prev = np.zeros(3, dtype=np.float64)  # [c, phi, theta]
    dh_prev = np.zeros(6, dtype=np.float64)  # [c, phi, theta, omega, alpha, beta]
    
    # Accumulate OPG and Hessian
    OPG = np.zeros((K, K), dtype=np.float64)
    
    for t in range(1, n):
        e_prev = resid[t-1]
        h_prev = sigma2[t-1]
        e2_prev = e_prev ** 2
        e_t = resid[t]
        h_t = sigma2[t]
        
        # Residual sensitivities: de_t/d[c, phi, theta]
        de_curr = np.zeros(3, dtype=np.float64)
        de_curr[0] = -1.0 - theta_ma * de_prev[0]  # c
        de_curr[1] = -y[t-1] - theta_ma * de_prev[1]  # phi
        de_curr[2] = -e_prev - theta_ma * de_prev[2]  # theta
        
        # Squared residual sensitivities: d(e²)/d[c, phi, theta]
        de2_prev = 2.0 * e_prev * de_prev
        
        # Variance sensitivities: dh_t/d[c, phi, theta, omega, alpha, beta]
        dh_curr = np.zeros(6, dtype=np.float64)
        dh_curr[0] = alpha * de2_prev[0] + beta * dh_prev[0]  # c
        dh_curr[1] = alpha * de2_prev[1] + beta * dh_prev[1]  # phi
        dh_curr[2] = alpha * de2_prev[2] + beta * dh_prev[2]  # theta
        dh_curr[3] = 1.0 + beta * dh_prev[3]  # omega
        dh_curr[4] = e2_prev + beta * dh_prev[4]  # alpha
        dh_curr[5] = h_prev + beta * dh_prev[5]  # beta
        
        # Normal log-likelihood gradient contributions:
        # ℓ_t = -0.5 * log(h_t) - 0.5 * e_t² / h_t
        # ∂ℓ_t/∂e_t = -e_t / h_t
        # ∂ℓ_t/∂h_t = -0.5/h_t + 0.5*e_t²/h_t²
        # For NLL, negate:
        dnll_de = e_t / h_t
        dnll_dh = 0.5 / h_t - 0.5 * e_t**2 / h_t**2
        
        # Per-observation gradient (K-vector)
        g_t = np.zeros(K, dtype=np.float64)
        # Mean params (c, phi, theta)
        for k in range(3):
            g_t[k] = dnll_de * de_curr[k] + dnll_dh * dh_curr[k]
        # Variance params (omega, alpha, beta)
        for k in range(3):
            g_t[3 + k] = dnll_dh * dh_curr[3 + k]
        
        # Accumulate OPG
        OPG += np.outer(g_t, g_t)
        
        # Update for next iteration
        de_prev = de_curr
        dh_prev = dh_curr
    
    # Average
    OPG /= n_eff
    
    # Compute Hessian numerically (as analytical Hessian is complex)
    y_ptr = _as_cptr(y)
    def nll_total(theta):
        r = np.zeros(n, dtype=np.float64)
        s = np.zeros(n, dtype=np.float64)
        return _core._arma_garch_nll_11_normal(
            _as_cptr(theta), y_ptr, _as_cptr(r), _as_cptr(s), h0, n
        )
    
    hess = _numerical_hessian(nll_total, params, eps=1e-5)
    
    return OPG, hess


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
        
        P_arch, Q_garch = vol.p, vol.q
        n_vol = 1 + P_arch + Q_garch
        
        # Check for MEAN component (ARMA)
        mean = None
        for comp in spec.components:
            if comp.role == Role.MEAN:
                mean = comp
                break
        
        has_arma = (mean is not None)
        if has_arma:
            p_ar, q_ma = mean.p, mean.q
            n_mean = 1 + p_ar + q_ma
        else:
            p_ar, q_ma = 0, 0
            n_mean = 0
        
        n_core = n_mean + n_vol  # Core params before distribution shape params
        
        # =====================================================================
        # Step 1: Fit with Normal likelihood
        # =====================================================================
        
        # Build Normal spec for fitting
        from ..components.density import Normal as NormalDensity
        from ..components.vol import GARCH
        from ..components.mean import ARMA
        
        if has_arma:
            normal_spec = ARMA(p_ar, q_ma) + GARCH(P_arch, Q_garch) + NormalDensity()
        else:
            normal_spec = GARCH(P_arch, Q_garch) + NormalDensity()
        
        # Get routine and fit
        routine = get_routine(str(normal_spec))
        fit_result = routine.fit(data, solver=solver, verbose=verbose, **kwargs)
        
        if fit_result.sigma2 is None:
            warnings.warn("Cannot compute robust SEs: sigma2 not available from estimation")
            self._last_result = fit_result
            return fit_result
        
        sigma2 = fit_result.sigma2
        core_params = fit_result.params[:n_core]
        
        # =====================================================================
        # Step 2: Compute robust SEs for core parameters
        # =====================================================================
        
        n = len(data)
        h0 = np.mean(data ** 2)
        
        if has_arma:
            # ARMA-GARCH: use specialized function
            try:
                opg_core, hess_core = _compute_arma_garch_robust_se(
                    core_params, data, p_ar, q_ma, P_arch, Q_garch, h0
                )
            except (NotImplementedError, Exception) as e:
                warnings.warn(f"ARMA-GARCH robust SE computation failed: {e}. "
                             "Returning MLE standard errors only.")
                self._last_result = fit_result
                return fit_result
        else:
            # Pure GARCH: use existing implementation
            resid2 = data ** 2
            try:
                opg_core, hess_core = _compute_robust_se_c(core_params, resid2, sigma2, P_arch, Q_garch)
            except RuntimeError:
                if P_arch != 1 or Q_garch != 1:
                    warnings.warn(
                        f"Robust SE computation only supports GARCH(1,1). "
                        f"Got GARCH({P_arch},{Q_garch}). Returning MLE standard errors only."
                    )
                    self._last_result = fit_result
                    return fit_result
                
                try:
                    opg_core, hess_core = _compute_robust_se_python(core_params, resid2, sigma2)
                except Exception as e:
                    warnings.warn(f"Robust SE computation failed: {e}")
                    self._last_result = fit_result
                    return fit_result
        
        # Compute covariance matrices for core parameters
        # Note: Functions return averaged H and OPG (divided by n)
        # For averaged quantities:
        #   I = n * H_avg, so Cov = I^{-1} = H_avg^{-1} / n
        #   Cov_robust = H_avg^{-1} @ OPG_avg @ H_avg^{-1} / n
        try:
            hess_inv_core = np.linalg.inv(hess_core)
            cov_mle_core = hess_inv_core / n
            cov_robust_core = hess_inv_core @ opg_core @ hess_inv_core / n
        except np.linalg.LinAlgError:
            warnings.warn("Hessian is singular, cannot compute covariance matrix")
            self._last_result = fit_result
            return fit_result
        
        # =====================================================================
        # Step 3: Handle distribution-specific fitting
        # =====================================================================
        
        if density_type == "normal":
            # Normal: we're done - just return robust SEs
            t_elapsed = time.perf_counter() - t_start
            
            enhanced_result = EstimationResult(
                spec=normal_spec,
                optimization_result=fit_result.optimization_result,
                data=data,
                sigma2=sigma2,
                time_elapsed=t_elapsed,
                hessian=hess_core,
                cov_matrix=cov_mle_core,
                opg=opg_core,
                cov_robust=cov_robust_core,
                method="QMLE",
            )
            
            # Update component fitted params
            if has_arma:
                mean.unpack(core_params[:n_mean])
                vol.unpack(core_params[n_mean:n_mean + n_vol])
            else:
                vol.unpack(core_params)
            
            self._last_result = enhanced_result
            
            if verbose:
                self._print_robust_se(enhanced_result, P_arch, Q_garch)
            
            return enhanced_result
        
        elif density_type == "studentt":
            # Student-t: fit nu with GARCH/ARMA-GARCH fixed
            # Compute residuals from estimated model
            resid = fit_result.resid if hasattr(fit_result, 'resid') and fit_result.resid is not None else data
            nu_hat, nll_shape, hess_nu = _fit_studentt_shape(resid, sigma2, verbose)
            
            # Build full covariance (block diagonal)
            n_params = n_core + 1  # Core + nu
            full_params = np.concatenate([core_params, [nu_hat]])
            
            # Block diagonal covariance
            cov_full_robust = np.zeros((n_params, n_params), dtype=np.float64)
            cov_full_robust[:n_core, :n_core] = cov_robust_core
            
            try:
                cov_nu = np.linalg.inv(hess_nu)
                cov_full_robust[n_core:, n_core:] = cov_nu
            except np.linalg.LinAlgError:
                cov_full_robust[n_core, n_core] = np.nan
            
            # MLE covariance (for shape params)
            cov_full_mle = np.zeros((n_params, n_params), dtype=np.float64)
            cov_full_mle[:n_core, :n_core] = cov_mle_core
            cov_full_mle[n_core:, n_core:] = cov_full_robust[n_core:, n_core:]
            
            # Build full Hessian and OPG (block diagonal)
            hess_full = np.zeros((n_params, n_params), dtype=np.float64)
            hess_full[:n_core, :n_core] = hess_core
            hess_full[n_core:, n_core:] = hess_nu
            
            opg_full = np.zeros((n_params, n_params), dtype=np.float64)
            opg_full[:n_core, :n_core] = opg_core
            # Shape params OPG not available, use Hessian as proxy
            opg_full[n_core:, n_core:] = hess_nu
            
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
            # Use residuals from fitted model
            resid_data = resid if 'resid' in dir() else data
            resid2 = resid_data ** 2
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
            if has_arma:
                mean.unpack(core_params[:n_mean])
                vol.unpack(core_params[n_mean:n_mean + n_vol])
            else:
                vol.unpack(core_params)
            if density is not None:
                density.fitted_params = {'nu': nu_hat}
            
            self._last_result = enhanced_result
            
            if verbose:
                self._print_robust_se_studentt(enhanced_result, P_arch, Q_garch, nu_hat)
            
            return enhanced_result
        
        elif density_type == "skewt":
            # Skew-t: fit [nu, lam] with GARCH/ARMA-GARCH fixed
            # Compute residuals from estimated model
            resid = fit_result.resid if hasattr(fit_result, 'resid') and fit_result.resid is not None else data
            nu_hat, lam_hat, nll_shape, hess_shape = _fit_skewt_shape(resid, sigma2, verbose)
            
            # Build full covariance (block diagonal)
            n_params = n_core + 2  # Core + nu + lam
            full_params = np.concatenate([core_params, [nu_hat, lam_hat]])
            
            # Block diagonal covariance
            cov_full_robust = np.zeros((n_params, n_params), dtype=np.float64)
            cov_full_robust[:n_core, :n_core] = cov_robust_core
            
            try:
                cov_shape = np.linalg.inv(hess_shape)
                cov_full_robust[n_core:, n_core:] = cov_shape
            except np.linalg.LinAlgError:
                cov_full_robust[n_core:, n_core:] = np.nan
            
            # MLE covariance
            cov_full_mle = np.zeros((n_params, n_params), dtype=np.float64)
            cov_full_mle[:n_core, :n_core] = cov_mle_core
            cov_full_mle[n_core:, n_core:] = cov_full_robust[n_core:, n_core:]
            
            # Build full Hessian and OPG (block diagonal)
            hess_full = np.zeros((n_params, n_params), dtype=np.float64)
            hess_full[:n_core, :n_core] = hess_core
            hess_full[n_core:, n_core:] = hess_shape
            
            opg_full = np.zeros((n_params, n_params), dtype=np.float64)
            opg_full[:n_core, :n_core] = opg_core
            opg_full[n_core:, n_core:] = hess_shape
            
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
            resid_data = resid if 'resid' in dir() else data
            resid_ptr = _as_cptr(resid_data)
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
            if has_arma:
                mean.unpack(core_params[:n_mean])
                vol.unpack(core_params[n_mean:n_mean + n_vol])
            else:
                vol.unpack(core_params)
            if density is not None:
                density.fitted_params = {'nu': nu_hat, 'lam': lam_hat}
            
            self._last_result = enhanced_result
            
            if verbose:
                self._print_robust_se_skewt(enhanced_result, P_arch, Q_garch, nu_hat, lam_hat)
            
            return enhanced_result
        
        else:
            raise ValueError(f"Unknown density type: {density_type}")
    
    def _print_robust_se(self, result, p: int, q: int):
        """Print robust SE info for Normal."""
        _r = settings.names.resolve
        se_mle = result.std_errors
        se_robust = result.std_errors_robust
        if se_mle is not None and se_robust is not None:
            print("QMLE: Robust (sandwich) standard errors computed successfully.")
            param_names = [_r("omega")] + [_r(f"alpha[{i+1}]") for i in range(p)] + [_r(f"beta[{j+1}]") for j in range(q)]
            for i, name in enumerate(param_names):
                if i < len(se_mle) and i < len(se_robust):
                    print(f"  SE({name}):  MLE={se_mle[i]:.6f}, Robust={se_robust[i]:.6f}")
    
    def _print_robust_se_studentt(self, result, p: int, q: int, nu: float):
        """Print robust SE info for Student-t."""
        _r = settings.names.resolve
        se_robust = result.std_errors_robust
        if se_robust is not None:
            nu_name = _r("nu")
            print(f"QMLE (Student-t two-step): Robust GARCH SEs, MLE SE for {nu_name}.")
            param_names = [_r("omega")] + [_r(f"alpha[{i+1}]") for i in range(p)] + [_r(f"beta[{j+1}]") for j in range(q)] + [nu_name]
            for i, name in enumerate(param_names):
                if i < len(se_robust):
                    se_type = "Robust" if i < (1 + p + q) else "MLE"
                    print(f"  SE({name}):  {se_type}={se_robust[i]:.6f}")
            print(f"  {nu_name} (df) = {nu:.2f}")
    
    def _print_robust_se_skewt(self, result, p: int, q: int, nu: float, lam: float):
        """Print robust SE info for Skew-t."""
        _r = settings.names.resolve
        se_robust = result.std_errors_robust
        if se_robust is not None:
            nu_name = _r("nu")
            lam_name = _r("lambda")
            print(f"QMLE (Skew-t two-step): Robust GARCH SEs, MLE SEs for {nu_name}, {lam_name}.")
            param_names = [_r("omega")] + [_r(f"alpha[{i+1}]") for i in range(p)] + [_r(f"beta[{j+1}]") for j in range(q)] + [nu_name, lam_name]
            for i, name in enumerate(param_names):
                if i < len(se_robust):
                    se_type = "Robust" if i < (1 + p + q) else "MLE"
                    print(f"  SE({name}):  {se_type}={se_robust[i]:.6f}")
            print(f"  {nu_name} (df) = {nu:.2f}, {lam_name} = {lam:.4f}")
