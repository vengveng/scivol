"""
volkit_compat.py - Legacy-compatible wrapper for volkit GARCH estimation.

Provides a `fit_garch()` function with an interface similar to the old
garch_estimator.py for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
from numpy.typing import NDArray

# volkit library
from volkit import GARCH, Normal, StudentT, SkewT, MLE, QMLE


@dataclass
class GARCHParams:
    """GARCH parameter container."""
    omega: float
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    
    @classmethod
    def from_array(cls, params: NDArray[np.float64], p: int, q: int) -> "GARCHParams":
        return cls(
            omega=params[0],
            alpha=params[1:1+q],
            beta=params[1+q:1+q+p],
        )


@dataclass
class DistributionParams:
    """Distribution parameter container."""
    nu: Optional[float] = None
    lam: Optional[float] = None


@dataclass
class EstimationResult:
    """Wrapper result compatible with old interface."""
    garch_params: GARCHParams
    dist_params: DistributionParams
    log_likelihood: float
    n_obs: int
    converged: bool
    n_iter: int
    message: str = ""
    hessian: Optional[NDArray] = None
    cov_matrix: Optional[NDArray] = None
    std_errors: Optional[NDArray] = None
    std_errors_robust: Optional[NDArray] = None
    sigma2: Optional[NDArray] = None
    std_resid: Optional[NDArray] = None
    time_elapsed: float = 0.0
    aic: Optional[float] = None
    bic: Optional[float] = None
    method: str = "MLE"


def fit_garch(
    resid: NDArray[np.float64],
    dist: Literal["normal", "studentt", "skewt"] = "normal",
    p: int = 1,
    q: int = 1,
    method: Literal["mle", "qmle"] = "mle",
    solver: str = "slsqp",
    use_logspace: bool = True,
    use_derivatives: bool = True,
    verbose: bool = False,
) -> EstimationResult:
    """
    Legacy-compatible interface to volkit GARCH estimation.
    
    Maps old interface to volkit component-based API.
    
    Parameters
    ----------
    resid : array
        Residual series (demeaned returns or AR residuals)
    dist : str
        Distribution: "normal", "studentt", or "skewt"
    p, q : int
        GARCH orders (default 1, 1)
    method : str
        "mle" or "qmle"
    solver : str
        Optimizer: "nelder-mead", "slsqp", "trust-constr", "trust"
    use_logspace : bool
        If True, optimize in unconstrained log-space
    use_derivatives : bool
        Ignored (volkit uses analytical derivatives when available)
    verbose : bool
        Print progress
    
    Returns
    -------
    EstimationResult
        Contains parameters, log-likelihood, conditional variances, etc.
    """
    # Build spec
    if dist == "normal":
        spec = GARCH(p, q) + Normal()
    elif dist == "studentt":
        spec = GARCH(p, q) + StudentT()
    elif dist == "skewt":
        spec = GARCH(p, q) + SkewT()
    else:
        raise ValueError(f"Unknown distribution: {dist}")
    
    # Map solver names
    solver_map = {
        "nelder-mead": "nelder-mead",
        "slsqp": "slsqp",
        "trust-constr": "trust",
        "trust-exact": "trust",
        "trust": "trust",
    }
    volkit_solver = solver_map.get(solver, "slsqp")
    
    # Fit with volkit
    resid_arr = np.ascontiguousarray(resid, dtype=np.float64)
    
    if method == "qmle":
        estimator = QMLE()
    else:
        estimator = MLE()
    
    result = estimator.fit(spec, resid_arr, solver=volkit_solver, log_mode=use_logspace, verbose=verbose)
    
    # Extract parameters
    params = result.params
    n_garch = 1 + p + q  # omega + alphas + betas
    
    garch_params = GARCHParams(
        omega=params[0],
        alpha=params[1:1+q],
        beta=params[1+q:1+q+p],
    )
    
    dist_params = DistributionParams()
    if dist == "studentt" and len(params) > n_garch:
        dist_params.nu = params[n_garch]
    elif dist == "skewt" and len(params) > n_garch:
        dist_params.nu = params[n_garch]
        if len(params) > n_garch + 1:
            dist_params.lam = params[n_garch + 1]
    
    # Standard errors and covariance
    std_errors = result.std_errors if hasattr(result, 'std_errors') else None
    std_errors_robust = result.std_errors_robust if hasattr(result, 'std_errors_robust') else None
    cov_matrix = result.cov_matrix if hasattr(result, 'cov_matrix') else None
    hessian = result.hessian if hasattr(result, 'hessian') else None
    
    return EstimationResult(
        garch_params=garch_params,
        dist_params=dist_params,
        log_likelihood=result.loglikelihood,
        n_obs=len(resid_arr),
        converged=result.success if hasattr(result, 'success') else True,
        n_iter=result.n_iter if hasattr(result, 'n_iter') else 0,
        message="",
        hessian=hessian,
        cov_matrix=cov_matrix,
        std_errors=std_errors,
        std_errors_robust=std_errors_robust,
        sigma2=result.sigma2 if hasattr(result, 'sigma2') else None,
        std_resid=result.std_resid if hasattr(result, 'std_resid') else None,
        time_elapsed=result.time_elapsed if hasattr(result, 'time_elapsed') else 0.0,
        aic=result.aic if hasattr(result, 'aic') else None,
        bic=result.bic if hasattr(result, 'bic') else None,
        method="QMLE" if method == "qmle" else "MLE",
    )
