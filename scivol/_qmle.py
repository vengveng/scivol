# scivol/_qmle.py
"""
QMLE fitting with robust (sandwich) standard errors.

Supports:
- GARCH + Normal: Robust SEs for all parameters
- GARCH + StudentT: Robust SEs for GARCH params, MLE SE for nu (two-step)
- GARCH + SkewT: Robust SEs for GARCH params, MLE SE for nu, lam (two-step)
- GJR-GARCH variants of all the above

The sandwich covariance estimator is::

    V_robust = H^{-1} @ OPG @ H^{-1}

where H is the Hessian and OPG is the outer product of per-observation
gradients.

For Student-t and Skew-t, a two-step procedure is used:

1. Fit GARCH with Normal LL -> robust SEs for volatility params
2. Fix GARCH, fit shape params -> MLE SEs for [nu] or [nu, lam]

The final covariance is block-diagonal (GARCH and shape blocks independent).
"""
from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .spec import CompositeSpec
from .components.base import Component
from .roles import Role
from ._kernels import get_routine
from . import _core
from ._settings import settings
from ._validation import validate_spec, validate_data, warn_small_sample

if TYPE_CHECKING:
    from .result import EstimationResult


# =====================================================================
# Pointer helper
# =====================================================================

def _as_cptr(a: NDArray[np.float64]) -> int:
    """Convert numpy array to C pointer (as integer address)."""
    return np.ascontiguousarray(a, np.float64).ctypes.data


# =====================================================================
# OPG / Hessian computation (GARCH)
# =====================================================================

def _compute_robust_se_garch_c(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute OPG and Hessian for GARCH using C extensions.

    Returns (opg, hess) matrices for GARCH parameters only.
    """
    n = len(resid2)
    k = 1 + p + q

    garch_params = params[:k].copy()
    opg = np.zeros((k, k), dtype=np.float64)
    hess = np.zeros((k, k), dtype=np.float64)

    # Specialized (1,1) first
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

    # General (p,q)
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

    raise RuntimeError("C extension OPG/Hessian computation not available for GARCH")


# =====================================================================
# OPG / Hessian computation (GJR-GARCH)
# =====================================================================

def _compute_robust_se_gjrgarch_c(
    params: NDArray[np.float64],
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute OPG and Hessian for GJR-GARCH using C extensions.

    GJR-GARCH takes raw residuals (not squared) because the indicator
    I(eps<0) requires the sign.

    Parameters are [omega, alpha_1..p, gamma_1..p, beta_1..q],
    so k = 1 + 2*p + q.

    Returns (opg, hess) matrices for GJR-GARCH parameters only.
    """
    n = len(resid)
    k = 1 + 2 * p + q

    gjr_params = params[:k].copy()
    opg = np.zeros((k, k), dtype=np.float64)
    hess = np.zeros((k, k), dtype=np.float64)

    # Specialized (1,1) first
    if p == 1 and q == 1:
        try:
            _core._gjr_garch_opg_hess_11(
                _as_cptr(gjr_params),
                _as_cptr(resid),
                _as_cptr(sigma2),
                _as_cptr(opg),
                _as_cptr(hess),
                n,
            )
            return opg, hess
        except (AttributeError, TypeError):
            pass

    # General (p,q)
    try:
        _core._gjr_garch_opg_hess_pq(
            _as_cptr(gjr_params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(opg),
            _as_cptr(hess),
            n, p, q,
        )
        return opg, hess
    except (AttributeError, TypeError):
        pass

    raise RuntimeError("C extension OPG/Hessian computation not available for GJR-GARCH")


# =====================================================================
# Python fallback for GARCH(1,1) OPG/Hessian
# =====================================================================

def _compute_robust_se_python(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute OPG and Hessian using pure Python (fallback).
    Only supports GARCH(1,1) + Normal.
    """
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from likelihoods import garch11_normal_robust_se

    result = garch11_normal_robust_se(params[:3], resid2, float(sigma2[0]))
    return result.opg, result.hess


# =====================================================================
# Numerical Hessian
# =====================================================================

def _numerical_hessian(
    func,
    x: NDArray[np.float64],
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute numerical Hessian using central differences."""
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


# =====================================================================
# ARMA-GARCH robust SE (analytical OPG + numerical Hessian)
# =====================================================================

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
    Compute OPG and Hessian for ARMA-GARCH+Normal.

    Only supports ARMA(1,1)-GARCH(1,1) currently.
    Returns (opg, hess) - both averaged (divided by n_eff).
    """
    n = len(y)
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    K = n_mean + n_vol

    if not (p_ar == 1 and q_ma == 1 and P_arch == 1 and Q_garch == 1):
        raise NotImplementedError("ARMA-GARCH QMLE only supports ARMA(1,1)-GARCH(1,1) currently")

    c = params[0]
    phi = params[1]
    theta_ma = params[2]
    omega = params[3]
    alpha = params[4]
    beta = params[5]

    # Forward pass: residuals and variances
    resid = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    resid[0] = 0.0
    sigma2[0] = h0

    for t in range(1, n):
        resid[t] = y[t] - c - phi * y[t - 1] - theta_ma * resid[t - 1]
        sigma2[t] = omega + alpha * resid[t - 1] ** 2 + beta * sigma2[t - 1]

    # Per-observation gradients via sensitivity recursion
    n_eff = n - 1
    de_prev = np.zeros(3, dtype=np.float64)
    dh_prev = np.zeros(6, dtype=np.float64)
    OPG = np.zeros((K, K), dtype=np.float64)

    for t in range(1, n):
        e_prev = resid[t - 1]
        e2_prev = e_prev ** 2
        h_prev = sigma2[t - 1]
        e_t = resid[t]
        h_t = sigma2[t]

        de_curr = np.zeros(3, dtype=np.float64)
        de_curr[0] = -1.0 - theta_ma * de_prev[0]
        de_curr[1] = -y[t - 1] - theta_ma * de_prev[1]
        de_curr[2] = -e_prev - theta_ma * de_prev[2]

        de2_prev = 2.0 * e_prev * de_prev

        dh_curr = np.zeros(6, dtype=np.float64)
        dh_curr[0] = alpha * de2_prev[0] + beta * dh_prev[0]
        dh_curr[1] = alpha * de2_prev[1] + beta * dh_prev[1]
        dh_curr[2] = alpha * de2_prev[2] + beta * dh_prev[2]
        dh_curr[3] = 1.0 + beta * dh_prev[3]
        dh_curr[4] = e2_prev + beta * dh_prev[4]
        dh_curr[5] = h_prev + beta * dh_prev[5]

        dnll_de = e_t / h_t
        dnll_dh = 0.5 / h_t - 0.5 * e_t ** 2 / h_t ** 2

        g_t = np.zeros(K, dtype=np.float64)
        for k in range(3):
            g_t[k] = dnll_de * de_curr[k] + dnll_dh * dh_curr[k]
        for k in range(3):
            g_t[3 + k] = dnll_dh * dh_curr[3 + k]

        OPG += np.outer(g_t, g_t)
        de_prev = de_curr
        dh_prev = dh_curr

    OPG /= n_eff

    # Numerical Hessian
    y_ptr = _as_cptr(y)

    def nll_total(theta):
        r = np.zeros(n, dtype=np.float64)
        s = np.zeros(n, dtype=np.float64)
        return _core._arma_garch_nll_11_normal(
            _as_cptr(theta), y_ptr, _as_cptr(r), _as_cptr(s), h0, n
        )

    hess = _numerical_hessian(nll_total, params, eps=1e-5)
    return OPG, hess


# =====================================================================
# Shape parameter fitting (Student-t, Skew-t)
# =====================================================================

def _fit_studentt_shape(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    verbose: bool = False,
) -> tuple[float, float, NDArray[np.float64]]:
    """Fit Student-t nu with GARCH fixed.  Returns (nu_hat, nll, hessian_nu)."""
    n = len(resid)
    resid2 = resid ** 2
    z2 = resid2 / np.maximum(sigma2, 1e-12)
    z2_ptr = _as_cptr(z2)
    sigma2_ptr = _as_cptr(sigma2)

    def nll_nu(nu_arr):
        nu = nu_arr[0]
        if nu <= 2.01 or nu > 100:
            return 1e10
        ll = _core._studentt_ll(sigma2_ptr, z2_ptr, n, nu)
        return -ll / n

    result = minimize(
        nll_nu, x0=np.array([8.0]), method="L-BFGS-B",
        bounds=[(2.01, 100.0)], options={"maxiter": 1000, "disp": verbose},
    )
    nu_hat = result.x[0]
    nll = result.fun * n
    hess = _numerical_hessian(nll_nu, result.x, eps=1e-5) * n
    return nu_hat, nll, hess


def _fit_skewt_shape(
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    verbose: bool = False,
) -> tuple[float, float, float, NDArray[np.float64]]:
    """Fit Skew-t (nu, lam) with GARCH fixed.  Returns (nu, lam, nll, hessian)."""
    n = len(resid)
    resid_ptr = _as_cptr(resid)
    sigma2_ptr = _as_cptr(sigma2)

    def nll_shape(params):
        nu, lam = params[0], params[1]
        if nu <= 2.01 or nu > 100:
            return 1e10
        if lam <= -0.99 or lam >= 0.99:
            return 1e10
        ll = _core._skewt_ll(resid_ptr, sigma2_ptr, n, nu, lam)
        if not np.isfinite(ll):
            return 1e10
        return -ll / n

    result = minimize(
        nll_shape, x0=np.array([8.0, 0.0]), method="L-BFGS-B",
        bounds=[(2.01, 100.0), (-0.99, 0.99)],
        options={"maxiter": 1000, "disp": verbose},
    )
    nu_hat, lam_hat = result.x[0], result.x[1]
    nll = result.fun * n
    hess = _numerical_hessian(nll_shape, result.x, eps=1e-5) * n
    return nu_hat, lam_hat, nll, hess


# =====================================================================
# Verbose printing helpers
# =====================================================================

def _print_robust_se(result: EstimationResult, p: int, q: int) -> None:
    _r = settings.names.resolve
    se_mle = result.std_errors
    se_robust = result.std_errors_robust
    if se_mle is not None and se_robust is not None:
        print("QMLE: Robust (sandwich) standard errors computed successfully.")
        names = [_r("omega")] + [_r(f"alpha[{i+1}]") for i in range(p)] + [_r(f"beta[{j+1}]") for j in range(q)]
        for i, name in enumerate(names):
            if i < len(se_mle) and i < len(se_robust):
                print(f"  SE({name}):  MLE={se_mle[i]:.6f}, Robust={se_robust[i]:.6f}")


def _print_robust_se_studentt(result: EstimationResult, p: int, q: int, nu: float) -> None:
    _r = settings.names.resolve
    se_robust = result.std_errors_robust
    if se_robust is not None:
        nu_name = _r("nu")
        print(f"QMLE (Student-t two-step): Robust GARCH SEs, MLE SE for {nu_name}.")
        names = [_r("omega")] + [_r(f"alpha[{i+1}]") for i in range(p)] + [_r(f"beta[{j+1}]") for j in range(q)] + [nu_name]
        for i, name in enumerate(names):
            if i < len(se_robust):
                se_type = "Robust" if i < (1 + p + q) else "MLE"
                print(f"  SE({name}):  {se_type}={se_robust[i]:.6f}")
        print(f"  {nu_name} (df) = {nu:.2f}")


def _print_robust_se_skewt(result: EstimationResult, p: int, q: int, nu: float, lam: float) -> None:
    _r = settings.names.resolve
    se_robust = result.std_errors_robust
    if se_robust is not None:
        nu_name = _r("nu")
        lam_name = _r("lambda")
        print(f"QMLE (Skew-t two-step): Robust GARCH SEs, MLE SEs for {nu_name}, {lam_name}.")
        names = [_r("omega")] + [_r(f"alpha[{i+1}]") for i in range(p)] + [_r(f"beta[{j+1}]") for j in range(q)] + [nu_name, lam_name]
        for i, name in enumerate(names):
            if i < len(se_robust):
                se_type = "Robust" if i < (1 + p + q) else "MLE"
                print(f"  SE({name}):  {se_type}={se_robust[i]:.6f}")
        print(f"  {nu_name} (df) = {nu:.2f}, {lam_name} = {lam:.4f}")


# =====================================================================
# Fake optimization result for the two-step path
# =====================================================================

class _FakeOptResult:
    """Minimal optimization-result-like object for two-step QMLE."""
    def __init__(self, x: NDArray[np.float64], fun: float, success: bool) -> None:
        self.x = x
        self.fun = fun
        self.success = success
        self.nit = 0
        self.message = "QMLE two-step estimation"


# =====================================================================
# Main entry point
# =====================================================================

def fit_qmle(
    spec: Union[CompositeSpec, Component],
    data: np.ndarray,
    *,
    solver: str = "slsqp",
    verbose: bool = False,
    **kwargs: Any,
) -> EstimationResult:
    """
    Fit a model via Quasi-Maximum Likelihood with robust sandwich SEs.

    Parameters
    ----------
    spec : CompositeSpec or Component
        Model specification (e.g. ``GARCH(1,1) + StudentT()``).
    data : 1-D array
        Returns or residuals to fit.
    solver : str
        Optimization solver for the Normal-likelihood step.
    verbose : bool
        Print progress.
    **kwargs
        Extra arguments forwarded to the kernel ``routine.fit()``.

    Returns
    -------
    EstimationResult
        Contains both MLE and robust standard errors.
    """
    from .result import EstimationResult
    from .components.density import Normal as NormalDensity, StudentT, SkewT
    from .components.vol import GARCH, GJRGARCH
    from .components.mean import ARMA

    t_start = time.perf_counter()

    # Validate
    spec = validate_spec(spec)
    data = validate_data(data)
    warn_small_sample(spec, data)

    # ── identify components ────────────────────────────────────────
    density = spec.get_component(Role.DENSITY)
    vol = spec.get_component(Role.VOLATILITY)
    mean = spec.get_component(Role.MEAN)

    if vol is None:
        raise ValueError("Spec must include a volatility component (e.g., GARCH)")

    is_gjr = isinstance(vol, GJRGARCH)
    P_arch, Q_garch = vol.p, vol.q
    n_vol = 1 + (2 * P_arch + Q_garch if is_gjr else P_arch + Q_garch)

    density_type = "normal"
    if density is not None:
        sig = density.signature
        if sig == "StudentT":
            density_type = "studentt"
        elif sig == "SkewT":
            density_type = "skewt"

    has_arma = mean is not None
    if has_arma:
        p_ar, q_ma = mean.p, mean.q
        n_mean = 1 + p_ar + q_ma
    else:
        p_ar, q_ma = 0, 0
        n_mean = 0

    n_core = n_mean + n_vol

    # ==================================================================
    # Step 1: Fit with Normal likelihood via the MLE kernel
    # ==================================================================
    vol_cls = GJRGARCH if is_gjr else GARCH
    if has_arma:
        normal_spec = ARMA(p_ar, q_ma) + vol_cls(P_arch, Q_garch) + NormalDensity()
    else:
        normal_spec = vol_cls(P_arch, Q_garch) + NormalDensity()

    routine = get_routine(str(normal_spec))
    fit_result = routine.fit(data, solver=solver, verbose=verbose, **kwargs)

    if fit_result.sigma2 is None:
        warnings.warn("Cannot compute robust SEs: sigma2 not available from estimation")
        return fit_result

    sigma2 = fit_result.sigma2
    core_params = fit_result.params[:n_core]

    # ==================================================================
    # Step 2: Compute robust SEs for core parameters
    # ==================================================================
    n = len(data)
    h0 = np.mean(data ** 2)

    if has_arma:
        try:
            opg_core, hess_core = _compute_arma_garch_robust_se(
                core_params, data, p_ar, q_ma, P_arch, Q_garch, h0,
            )
        except (NotImplementedError, Exception) as e:
            warnings.warn(
                f"ARMA-GARCH robust SE computation failed: {e}. "
                "Returning MLE standard errors only."
            )
            return fit_result
    elif is_gjr:
        # GJR-GARCH: use raw residuals (not squared)
        try:
            opg_core, hess_core = _compute_robust_se_gjrgarch_c(
                core_params, data, sigma2, P_arch, Q_garch,
            )
        except RuntimeError:
            warnings.warn(
                f"Robust SE computation for GJR-GARCH({P_arch},{Q_garch}) "
                "failed. Returning MLE standard errors only."
            )
            return fit_result
    else:
        # Standard GARCH
        resid2 = data ** 2
        try:
            opg_core, hess_core = _compute_robust_se_garch_c(
                core_params, resid2, sigma2, P_arch, Q_garch,
            )
        except RuntimeError:
            if P_arch != 1 or Q_garch != 1:
                warnings.warn(
                    f"Robust SE computation only supports GARCH(1,1). "
                    f"Got GARCH({P_arch},{Q_garch}). Returning MLE standard errors only."
                )
                return fit_result
            try:
                opg_core, hess_core = _compute_robust_se_python(core_params, resid2, sigma2)
            except Exception as e:
                warnings.warn(f"Robust SE computation failed: {e}")
                return fit_result

    # Covariance matrices
    try:
        hess_inv_core = np.linalg.inv(hess_core)
        cov_mle_core = hess_inv_core / n
        cov_robust_core = hess_inv_core @ opg_core @ hess_inv_core / n
    except np.linalg.LinAlgError:
        warnings.warn("Hessian is singular, cannot compute covariance matrix")
        return fit_result

    # ==================================================================
    # Step 3: Distribution-specific handling
    # ==================================================================

    def _update_components(core_params: NDArray[np.float64]) -> None:
        """Unpack fitted params into the original component objects."""
        if has_arma and mean is not None:
            mean.unpack(core_params[:n_mean])
            vol.unpack(core_params[n_mean:n_mean + n_vol])
        else:
            vol.unpack(core_params)

    if density_type == "normal":
        t_elapsed = time.perf_counter() - t_start
        enhanced = EstimationResult(
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
        _update_components(core_params)
        if verbose:
            _print_robust_se(enhanced, P_arch, Q_garch)
        return enhanced

    # ── non-Normal: two-step for shape params ─────────────────────
    resid = (
        fit_result.resid
        if hasattr(fit_result, "resid") and fit_result.resid is not None
        else data
    )

    if density_type == "studentt":
        nu_hat, _, hess_shape = _fit_studentt_shape(resid, sigma2, verbose)
        n_params = n_core + 1
        full_params = np.concatenate([core_params, [nu_hat]])
    elif density_type == "skewt":
        nu_hat, lam_hat, _, hess_shape = _fit_skewt_shape(resid, sigma2, verbose)
        n_params = n_core + 2
        full_params = np.concatenate([core_params, [nu_hat, lam_hat]])
    else:
        raise ValueError(f"Unknown density type: {density_type}")

    # Block-diagonal covariance
    cov_full_robust = np.zeros((n_params, n_params), dtype=np.float64)
    cov_full_robust[:n_core, :n_core] = cov_robust_core
    try:
        cov_shape = np.linalg.inv(hess_shape)
        cov_full_robust[n_core:, n_core:] = cov_shape
    except np.linalg.LinAlgError:
        cov_full_robust[n_core:, n_core:] = np.nan

    cov_full_mle = np.zeros((n_params, n_params), dtype=np.float64)
    cov_full_mle[:n_core, :n_core] = cov_mle_core
    cov_full_mle[n_core:, n_core:] = cov_full_robust[n_core:, n_core:]

    hess_full = np.zeros((n_params, n_params), dtype=np.float64)
    hess_full[:n_core, :n_core] = hess_core
    hess_full[n_core:, n_core:] = hess_shape

    opg_full = np.zeros((n_params, n_params), dtype=np.float64)
    opg_full[:n_core, :n_core] = opg_core
    opg_full[n_core:, n_core:] = hess_shape  # proxy

    # Total log-likelihood under the fitted distribution
    if density_type == "studentt":
        resid2_data = resid ** 2
        z2 = resid2_data / np.maximum(sigma2, 1e-12)
        ll_dist = _core._studentt_ll(_as_cptr(sigma2), _as_cptr(z2), n, nu_hat)
    else:  # skewt
        ll_dist = _core._skewt_ll(_as_cptr(resid), _as_cptr(sigma2), n, nu_hat, lam_hat)

    t_elapsed = time.perf_counter() - t_start
    fake_opt = _FakeOptResult(full_params, -ll_dist, True)

    enhanced = EstimationResult(
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

    _update_components(core_params)
    if density is not None:
        if density_type == "studentt":
            density.fitted_params = {"nu": nu_hat}
        elif density_type == "skewt":
            density.fitted_params = {"nu": nu_hat, "lam": lam_hat}

    if verbose:
        if density_type == "studentt":
            _print_robust_se_studentt(enhanced, P_arch, Q_garch, nu_hat)
        else:
            _print_robust_se_skewt(enhanced, P_arch, Q_garch, nu_hat, lam_hat)

    return enhanced
