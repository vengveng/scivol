"""
Optimizer Benchmark for scivol Models
=====================================

Comprehensive benchmark of optimization methods for shipped GARCH-family and
linked ARMA-volatility models using real financial data. Results guide default
parameter settings in scivol.

This script:
1. Tests all optimizer/mode combinations on multiple asset classes
2. Tracks: convergence, speed, parameter stability, log-likelihood
3. Produces explicit theta-vs-z comparison summaries for setting defaults

Models tested (all via scivol C extensions):
- GARCH(1,1) + Normal/Student-t/Skew-t/GED
- GJR-GARCH(1,1) + Normal/Student-t/Skew-t/GED
- EGARCH(1,1)/(2,1) + Normal/Student-t/Skew-t/GED
- ARMA(1,1)-GARCH(1,1) + Normal/Student-t/Skew-t/GED
- ARMA(1,1)-GJR-GARCH(1,1) + Normal/Student-t/Skew-t/GED
- ARMA(1,1)-EGARCH(1,1) + Normal/Student-t/Skew-t/GED
- ARMA(1,1) + Normal/GED

Keep this file evergreen - run periodically to validate/update defaults.

EGARCH smoke coverage is included for the shipped Normal, StudentT, SkewT, and GED surfaces.
ARMA-EGARCH smoke coverage is included for the shipped Normal, StudentT,
SkewT, and GED surfaces. ARX/HARX smoke coverage now includes standalone
Normal, StudentT, SkewT, and GED fits plus linked GARCH, GJR-GARCH, and
EGARCH rows. The full theta-vs-z policy benchmark currently covers the shipped
EGARCH Normal surfaces `(1,1)` and `(2,1)`.
"""
from __future__ import annotations

import warnings
import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Literal
from joblib import Parallel, delayed
import multiprocessing

# scivol library (all models via C extensions)
from scivol import ARMA, ARX, EGARCH, GARCH, GED, GJRGARCH, HARX, Normal, StudentT, SkewT

# Suppress convergence warnings during benchmark
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning,ignore::UserWarning")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Get maximum available workers
N_WORKERS = multiprocessing.cpu_count()
print(f"Using {N_WORKERS} parallel workers")

DEFAULT_POLICY_SOLVER = "slsqp"
MODEL_DISPLAY_ORDER = ["garch", "gjr_garch", "egarch_11", "egarch_21", "arma_garch", "arma_gjr_garch", "arma_egarch", "arma"]
PARAMETER_COLUMNS = ["omega", "alpha", "gamma", "beta", "c", "phi", "theta", "nu", "lam"]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ar1(series: pd.Series) -> dict:
    """
    Fit AR(1) model via OLS: y_t = c + phi * y_{t-1} + e_t
    
    Returns: {"c": float, "phi": float, "se_phi": float, "resid": pd.Series, "T": int}
    """
    x = series.dropna()
    y = x.iloc[1:]
    x_lag = x.shift(1).iloc[1:]

    # y = c + phi * x_lag + e
    cov = y.cov(x_lag)
    var = x_lag.var()
    phi = cov / var
    c = y.mean() - phi * x_lag.mean()

    eps = y - (c + phi * x_lag)
    T = len(y)
    sse = float((eps**2).sum())
    se_phi = np.sqrt(sse / (T - 2) / float(((x_lag - x_lag.mean())**2).sum()))

    return {
        "c": float(c),
        "phi": float(phi),
        "se_phi": float(se_phi),
        "resid": eps,   # pd.Series indexed by dates t=2..T
        "T": int(T),
    }


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("localdev_benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Data columns
DATA_RENAME = {
    "DATE": "date",
    "S&PCOMP(RI)": "stock",
    "SPUTBIX(RI)": "gbond",
    "SPUHYBD(RI)": "cbond",
    "WILURET(RI)": "resec",
    "RJEFCRT(TR)": "commo",
    "USBINXB": "usdfx",
}

# Asset classes to benchmark (all available)
ASSETS = ["stock", "gbond", "cbond", "resec", "commo", "usdfx"]

# =====================
# GARCH-only configs (scivol)
# =====================
GARCH_DIST_SPECS = {
    "normal": Normal(),
    "studentt": StudentT(),
    "skewt": SkewT(),
    "ged": GED(),
}

GARCH_OPTIMIZER_CONFIGS = [
    {"solver": "nelder-mead", "log_mode": False},
    {"solver": "slsqp", "log_mode": False},
    {"solver": "trust", "log_mode": False},
    {"solver": "nelder-mead", "log_mode": True},
    {"solver": "slsqp", "log_mode": True},
    {"solver": "trust", "log_mode": True},
]

# =====================
# ARMA-GARCH configs (scivol)
# =====================
# Mapping for ARMA-GARCH density components
ARMA_GARCH_DIST_SPECS = {
    "normal": Normal(),
    "studentt": StudentT(),
    "skewt": SkewT(),
    "ged": GED(),
}

ARMA_GARCH_OPTIMIZER_CONFIGS = [
    {"solver": "nelder-mead", "log_mode": False},
    {"solver": "slsqp", "log_mode": False},
    {"solver": "trust", "log_mode": False},
    {"solver": "nelder-mead", "log_mode": True},
    {"solver": "slsqp", "log_mode": True},
    {"solver": "trust", "log_mode": True},
]

# =====================
# ARMA-EGARCH configs (scivol)
# =====================
ARMA_EGARCH_DIST_SPECS = {
    "normal": Normal(),
    "studentt": StudentT(),
    "skewt": SkewT(),
    "ged": GED(),
}

ARMA_EGARCH_OPTIMIZER_CONFIGS = [
    {"solver": "nelder-mead", "log_mode": False},
    {"solver": "slsqp", "log_mode": False},
    {"solver": "trust", "log_mode": False},
    {"solver": "nelder-mead", "log_mode": True},
    {"solver": "slsqp", "log_mode": True},
    {"solver": "trust", "log_mode": True},
]

# =====================
# ARMA-GJR-GARCH configs (scivol)
# =====================
ARMA_GJR_GARCH_DIST_SPECS = {
    "normal": Normal(),
    "studentt": StudentT(),
    "skewt": SkewT(),
    "ged": GED(),
}

ARMA_GJR_GARCH_OPTIMIZER_CONFIGS = [
    {"solver": "nelder-mead", "log_mode": False},
    {"solver": "slsqp", "log_mode": False},
    {"solver": "trust", "log_mode": False},
    {"solver": "nelder-mead", "log_mode": True},
    {"solver": "slsqp", "log_mode": True},
    {"solver": "trust", "log_mode": True},
]

# =====================
# Pure ARMA configs (scivol)
# =====================
# ARMA ships Normal and GED constant-variance surfaces.
ARMA_DIST_SPECS = {
    "normal": Normal(),
    "ged": GED(),
}

ARMA_OPTIMIZER_CONFIGS = [
    {"solver": "nelder-mead", "log_mode": False},
    {"solver": "slsqp", "log_mode": False},
    {"solver": "trust", "log_mode": False},
    {"solver": "nelder-mead", "log_mode": True},
    {"solver": "slsqp", "log_mode": True},
    {"solver": "trust", "log_mode": True},
]

# =====================
# GJR-GARCH configs (scivol)
# =====================
GJR_GARCH_DIST_SPECS = {
    "normal": Normal(),
    "studentt": StudentT(),
    "skewt": SkewT(),
    "ged": GED(),
}

GJR_GARCH_OPTIMIZER_CONFIGS = [
    {"solver": "nelder-mead", "log_mode": False},
    {"solver": "slsqp", "log_mode": False},
    {"solver": "trust", "log_mode": False},
    {"solver": "nelder-mead", "log_mode": True},
    {"solver": "slsqp", "log_mode": True},
    {"solver": "trust", "log_mode": True},
]

# =====================
# EGARCH configs (scivol)
# =====================
EGARCH_DIST_SPECS = {
    "normal": Normal(),
    "studentt": StudentT(),
    "skewt": SkewT(),
    "ged": GED(),
}

EGARCH_ORDER_SPECS = {
    "egarch_11": (1, 1),
    "egarch_21": (2, 1),
}

EGARCH_OPTIMIZER_CONFIGS = [
    {"solver": "nelder-mead", "log_mode": False},
    {"solver": "slsqp", "log_mode": False},
    {"solver": "trust", "log_mode": False},
    {"solver": "nelder-mead", "log_mode": True},
    {"solver": "slsqp", "log_mode": True},
    {"solver": "trust", "log_mode": True},
]

# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    model_type: str  # "garch", "gjr_garch", "egarch_11", "egarch_21", "arma_garch", "arma_gjr_garch", "arma_egarch", "arma"
    asset: str
    dist: str
    solver: str
    log_mode: bool
    
    # Optimization outcome
    converged: bool
    n_iter: int
    time_elapsed: float
    log_likelihood: float
    
    # GARCH Parameters (if applicable)
    omega: Optional[float] = None
    alpha: Optional[float] = None
    gamma: Optional[float] = None
    beta: Optional[float] = None
    persistence: Optional[float] = None
    
    # ARMA Parameters (if applicable)
    c: Optional[float] = None
    phi: Optional[float] = None
    theta: Optional[float] = None
    
    # Distribution params
    nu: Optional[float] = None  # Student-t, Skew-t
    lam: Optional[float] = None  # Skew-t only
    
    # Model selection
    aic: Optional[float] = None
    bic: Optional[float] = None
    error_message: Optional[str] = None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """
    Load and preprocess all asset series.
    
    Returns:
        log_returns: Raw log returns (for ARMA-GARCH models)
        ar1_residuals: AR(1) residuals (for pure GARCH models)
    """
    print("Loading data...")
    data_raw = pd.read_csv("localdev_data/DATA_HW1.csv", skiprows=1, parse_dates=["DATE"], dayfirst=True)
    data_raw = data_raw.rename(columns=DATA_RENAME).set_index("date")
    
    # Keep only the columns we need
    available = [col for col in ASSETS if col in data_raw.columns]
    data_raw = data_raw[available]
    
    # Daily log-returns
    log_returns = np.log1p(data_raw.pct_change(fill_method=None))
    
    # Remove zero-return days
    mask_zero = (log_returns == 0).any(axis=1)
    log_returns = log_returns[~mask_zero]
    print(f"  Removed {mask_zero.sum()} zero-return days")
    
    # AR(1) residuals for each asset (for pure GARCH models)
    ar1_residuals = {}
    log_returns_dict = {}
    
    for asset in available:
        x = log_returns[asset].dropna()
        log_returns_dict[asset] = x
        
        ar1_res = ar1(x)
        ar1_residuals[asset] = ar1_res["resid"]
        print(f"  {asset}: T={len(ar1_residuals[asset])}, AR(1) phi={ar1_res['phi']:.4f}")
    
    return log_returns_dict, ar1_residuals


# =============================================================================
# GARCH BENCHMARK (scivol)
# =============================================================================

def run_garch_benchmark(
    resid: np.ndarray,
    asset: str,
    dist_name: str,
    config: Dict[str, Any],
) -> Optional[BenchmarkResult]:
    """Run single GARCH benchmark using scivol."""
    try:
        # Build model spec
        dist_component = GARCH_DIST_SPECS[dist_name]
        spec = GARCH(1, 1) + dist_component
        
        # Fit using scivol
        result = spec.fit(
            resid,
            solver=config["solver"],
            log_mode=config["log_mode"],
            verbose=False,
        )
        
        # Extract parameters
        params = result.params
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        persistence = alpha + beta
        
        # Distribution params
        nu = params[3] if len(params) > 3 else None
        lam = params[4] if len(params) > 4 else None
        
        converged = result.converged if hasattr(result, 'converged') else True
        n_iter = result.n_iter if hasattr(result, 'n_iter') else 0
        time_elapsed = result.time_elapsed if hasattr(result, 'time_elapsed') else 0.0
        
        return BenchmarkResult(
            model_type="garch",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=converged,
            n_iter=n_iter,
            time_elapsed=time_elapsed,
            log_likelihood=result.log_likelihood,
            omega=omega,
            alpha=alpha,
            beta=beta,
            persistence=persistence,
            nu=nu,
            lam=lam,
            aic=result.aic if hasattr(result, 'aic') else None,
            bic=result.bic if hasattr(result, 'bic') else None,
        )
    except Exception as e:
        return BenchmarkResult(
            model_type="garch",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=False,
            n_iter=0,
            time_elapsed=0.0,
            log_likelihood=np.nan,
            error_message=str(e),
        )


# =============================================================================
# ARMA(1,1)-GARCH(1,1) BENCHMARK (scivol)
# =============================================================================

def run_arma_garch_benchmark(
    y: np.ndarray,
    asset: str,
    dist_name: str,
    config: Dict[str, Any],
) -> Optional[BenchmarkResult]:
    """Run single ARMA(1,1)-GARCH(1,1) benchmark using scivol."""
    try:
        # Build model spec using scivol components
        dist_component = ARMA_GARCH_DIST_SPECS[dist_name]
        spec = ARMA(1, 1) + GARCH(1, 1) + dist_component
        
        # Fit using scivol
        result = spec.fit(
            y,
            solver=config["solver"],
            log_mode=config["log_mode"],
            verbose=False,
        )
        
        # Extract parameters: [c, phi, theta, omega, alpha, beta, (nu), (lam)]
        params = result.params
        c = params[0]
        phi = params[1]
        theta = params[2]
        omega = params[3]
        alpha = params[4]
        beta = params[5]
        persistence = alpha + beta
        
        # Distribution params
        nu = params[6] if len(params) > 6 else None
        lam = params[7] if len(params) > 7 else None
        
        converged = result.converged if hasattr(result, 'converged') else True
        n_iter = result.n_iter if hasattr(result, 'n_iter') else 0
        time_elapsed = result.time_elapsed if hasattr(result, 'time_elapsed') else 0.0
        
        return BenchmarkResult(
            model_type="arma_garch",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=converged,
            n_iter=n_iter,
            time_elapsed=time_elapsed,
            log_likelihood=result.log_likelihood,
            c=c,
            phi=phi,
            theta=theta,
            omega=omega,
            alpha=alpha,
            beta=beta,
            persistence=persistence,
            nu=nu,
            lam=lam,
            aic=result.aic if hasattr(result, 'aic') else None,
            bic=result.bic if hasattr(result, 'bic') else None,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return BenchmarkResult(
            model_type="arma_garch",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=False,
            n_iter=0,
            time_elapsed=0.0,
            log_likelihood=np.nan,
            error_message=str(e),
        )


# =============================================================================
# ARMA(1,1)-GJR-GARCH(1,1) BENCHMARK (scivol)
# =============================================================================

def run_arma_gjr_garch_benchmark(
    y: np.ndarray,
    asset: str,
    dist_name: str,
    config: Dict[str, Any],
) -> Optional[BenchmarkResult]:
    """Run single ARMA(1,1)-GJR-GARCH(1,1) benchmark using scivol."""
    try:
        dist_component = ARMA_GJR_GARCH_DIST_SPECS[dist_name]
        spec = ARMA(1, 1) + GJRGARCH(1, 1) + dist_component

        result = spec.fit(
            y,
            solver=config["solver"],
            log_mode=config["log_mode"],
            verbose=False,
        )

        params = result.params
        c = params[0]
        phi = params[1]
        theta = params[2]
        omega = params[3]
        alpha = params[4]
        gamma = params[5]
        beta = params[6]
        persistence = alpha + 0.5 * gamma + beta

        if dist_name == "skewt" and len(params) > 8:
            nu = params[7]
            lam = params[8]
        elif dist_name in {"studentt", "ged"} and len(params) > 7:
            nu = params[7]
            lam = None
        else:
            nu = None
            lam = None
        lam = params[8] if len(params) > 8 else None

        converged = result.converged if hasattr(result, "converged") else True
        n_iter = result.n_iter if hasattr(result, "n_iter") else 0
        time_elapsed = result.time_elapsed if hasattr(result, "time_elapsed") else 0.0

        return BenchmarkResult(
            model_type="arma_gjr_garch",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=converged,
            n_iter=n_iter,
            time_elapsed=time_elapsed,
            log_likelihood=result.log_likelihood,
            c=c,
            phi=phi,
            theta=theta,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            persistence=persistence,
            nu=nu,
            lam=lam,
            aic=result.aic if hasattr(result, "aic") else None,
            bic=result.bic if hasattr(result, "bic") else None,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return BenchmarkResult(
            model_type="arma_gjr_garch",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=False,
            n_iter=0,
            time_elapsed=0.0,
            log_likelihood=np.nan,
            error_message=str(e),
        )


# =============================================================================
# ARMA(1,1)-EGARCH(1,1) BENCHMARK (scivol)
# =============================================================================

def run_arma_egarch_benchmark(
    y: np.ndarray,
    asset: str,
    dist_name: str,
    config: Dict[str, Any],
) -> Optional[BenchmarkResult]:
    """Run single ARMA(1,1)-EGARCH(1,1) benchmark using scivol."""
    try:
        dist_component = ARMA_EGARCH_DIST_SPECS[dist_name]
        spec = ARMA(1, 1) + EGARCH(1, 1) + dist_component

        result = spec.fit(
            y,
            solver=config["solver"],
            log_mode=config["log_mode"],
            verbose=False,
        )

        params = result.params
        c = params[0]
        phi = params[1]
        theta = params[2]
        omega = params[3]
        alpha = params[4]
        gamma = params[5]
        beta = params[6]
        persistence = beta
        if dist_name == "skewt" and len(params) > 8:
            nu = params[7]
            lam = params[8]
        elif dist_name in {"studentt", "ged"} and len(params) > 7:
            nu = params[7]
            lam = None
        else:
            nu = None
            lam = None

        converged = result.converged if hasattr(result, "converged") else True
        n_iter = result.n_iter if hasattr(result, "n_iter") else 0
        time_elapsed = result.time_elapsed if hasattr(result, "time_elapsed") else 0.0

        return BenchmarkResult(
            model_type="arma_egarch",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=converged,
            n_iter=n_iter,
            time_elapsed=time_elapsed,
            log_likelihood=result.log_likelihood,
            c=c,
            phi=phi,
            theta=theta,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            persistence=persistence,
            nu=nu,
            lam=lam,
            aic=result.aic if hasattr(result, "aic") else None,
            bic=result.bic if hasattr(result, "bic") else None,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return BenchmarkResult(
            model_type="arma_egarch",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=False,
            n_iter=0,
            time_elapsed=0.0,
            log_likelihood=np.nan,
            error_message=str(e),
        )


# =============================================================================
# PURE ARMA BENCHMARK (constant variance)
# =============================================================================

def run_arma_benchmark(
    y: np.ndarray,
    asset: str,
    dist_name: str,
    config: Dict[str, Any],
) -> Optional[BenchmarkResult]:
    """Run single ARMA(1,1) benchmark using scivol (constant variance)."""
    try:
        # Build model spec using scivol components
        dist_component = ARMA_DIST_SPECS[dist_name]
        spec = ARMA(1, 1) + dist_component
        
        # Fit using scivol
        result = spec.fit(
            y,
            solver=config["solver"],
            log_mode=config["log_mode"],
            verbose=False,
        )
        
        # Extract parameters:
        #   Normal -> [c, phi, theta]
        #   GED    -> [c, phi, theta, sigma2, nu]
        params = result.params
        c = params[0]
        phi = params[1]
        theta = params[2]
        nu = params[4] if len(params) > 4 else None
        
        converged = result.converged if hasattr(result, 'converged') else True
        n_iter = result.n_iter if hasattr(result, 'n_iter') else 0
        time_elapsed = result.time_elapsed if hasattr(result, 'time_elapsed') else 0.0
        
        return BenchmarkResult(
            model_type="arma",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=converged,
            n_iter=n_iter,
            time_elapsed=time_elapsed,
            log_likelihood=result.log_likelihood,
            c=c,
            phi=phi,
            theta=theta,
            omega=None,
            alpha=None,
            beta=None,
            persistence=None,
            nu=nu,
            lam=lam,
            aic=result.aic if hasattr(result, 'aic') else None,
            bic=result.bic if hasattr(result, 'bic') else None,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return BenchmarkResult(
            model_type="arma",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=False,
            n_iter=0,
            time_elapsed=0.0,
            log_likelihood=np.nan,
            error_message=str(e),
        )


# =============================================================================
# GJR-GARCH(1,1) BENCHMARK (scivol)
# =============================================================================

def run_gjr_garch_benchmark(
    resid: np.ndarray,
    asset: str,
    dist_name: str,
    config: Dict[str, Any],
) -> Optional[BenchmarkResult]:
    """Run single GJR-GARCH(1,1) benchmark using scivol."""
    try:
        dist_component = GJR_GARCH_DIST_SPECS[dist_name]
        spec = GJRGARCH(1, 1) + dist_component
        
        result = spec.fit(
            resid,
            solver=config["solver"],
            log_mode=config["log_mode"],
            verbose=False,
        )
        
        params = result.params
        omega = params[0]
        alpha = params[1]
        gamma = params[2]
        beta = params[3]
        persistence = alpha + 0.5 * gamma + beta  # Symmetric dist approximation
        
        nu = params[4] if len(params) > 4 else None
        lam = params[5] if len(params) > 5 else None
        
        converged = result.converged if hasattr(result, 'converged') else True
        n_iter = result.n_iter if hasattr(result, 'n_iter') else 0
        time_elapsed = result.time_elapsed if hasattr(result, 'time_elapsed') else 0.0
        
        return BenchmarkResult(
            model_type="gjr_garch",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=converged,
            n_iter=n_iter,
            time_elapsed=time_elapsed,
            log_likelihood=result.log_likelihood,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            persistence=persistence,
            nu=nu,
            lam=lam,
            aic=result.aic if hasattr(result, 'aic') else None,
            bic=result.bic if hasattr(result, 'bic') else None,
        )
    except Exception as e:
        return BenchmarkResult(
            model_type="gjr_garch",
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=False,
            n_iter=0,
            time_elapsed=0.0,
            log_likelihood=np.nan,
            error_message=str(e),
        )


# =============================================================================
# EGARCH(p,q) BENCHMARK (scivol)
# =============================================================================

def run_egarch_benchmark(
    resid: np.ndarray,
    asset: str,
    dist_name: str,
    config: Dict[str, Any],
    *,
    p: int,
    q: int,
    model_type: str,
) -> Optional[BenchmarkResult]:
    """Run single EGARCH benchmark using scivol."""
    try:
        dist_component = EGARCH_DIST_SPECS[dist_name]
        spec = EGARCH(p, q) + dist_component

        result = spec.fit(
            resid,
            solver=config["solver"],
            log_mode=config["log_mode"],
            verbose=False,
        )

        params = result.params
        omega = params[0]
        alpha = params[1]
        gamma = params[1 + p]
        beta = params[1 + 2 * p]
        persistence = beta

        nu = params[1 + 2 * p + q] if len(params) > 1 + 2 * p + q else None
        lam = params[2 + 2 * p + q] if len(params) > 2 + 2 * p + q else None

        converged = result.converged if hasattr(result, "converged") else True
        n_iter = result.n_iter if hasattr(result, "n_iter") else 0
        time_elapsed = result.time_elapsed if hasattr(result, "time_elapsed") else 0.0

        return BenchmarkResult(
            model_type=model_type,
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=converged,
            n_iter=n_iter,
            time_elapsed=time_elapsed,
            log_likelihood=result.log_likelihood,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            persistence=persistence,
            nu=nu,
            lam=lam,
            aic=result.aic if hasattr(result, "aic") else None,
            bic=result.bic if hasattr(result, "bic") else None,
        )
    except Exception as e:
        return BenchmarkResult(
            model_type=model_type,
            asset=asset,
            dist=dist_name,
            solver=config["solver"],
            log_mode=config["log_mode"],
            converged=False,
            n_iter=0,
            time_elapsed=0.0,
            log_likelihood=np.nan,
            error_message=str(e),
        )


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def _run_single_garch_task(resid: np.ndarray, asset: str, dist_name: str, config: Dict[str, Any]) -> Tuple[str, Optional[BenchmarkResult]]:
    """Wrapper for running single GARCH task with descriptive output."""
    mode = "log" if config["log_mode"] else "con"
    config_str = f"{config['solver']}[{mode}]"
    task_desc = f"GARCH/{asset}/{dist_name}: {config_str}"
    result = run_garch_benchmark(resid, asset, dist_name, config)
    return task_desc, result


def _run_single_arma_garch_task(y: np.ndarray, asset: str, dist_name: str, config: Dict[str, Any]) -> Tuple[str, Optional[BenchmarkResult]]:
    """Wrapper for running single ARMA-GARCH task with descriptive output."""
    mode = "log" if config["log_mode"] else "con"
    config_str = f"{config['solver']}[{mode}]"
    task_desc = f"ARMA-GARCH/{asset}/{dist_name}: {config_str}"
    result = run_arma_garch_benchmark(y, asset, dist_name, config)
    return task_desc, result


def _run_single_arma_gjr_garch_task(y: np.ndarray, asset: str, dist_name: str, config: Dict[str, Any]) -> Tuple[str, Optional[BenchmarkResult]]:
    """Wrapper for running single ARMA-GJR-GARCH task with descriptive output."""
    mode = "log" if config["log_mode"] else "con"
    config_str = f"{config['solver']}[{mode}]"
    task_desc = f"ARMA-GJR-GARCH/{asset}/{dist_name}: {config_str}"
    result = run_arma_gjr_garch_benchmark(y, asset, dist_name, config)
    return task_desc, result


def _run_single_arma_egarch_task(y: np.ndarray, asset: str, dist_name: str, config: Dict[str, Any]) -> Tuple[str, Optional[BenchmarkResult]]:
    """Wrapper for running single ARMA-EGARCH task with descriptive output."""
    mode = "log" if config["log_mode"] else "con"
    config_str = f"{config['solver']}[{mode}]"
    task_desc = f"ARMA-EGARCH/{asset}/{dist_name}: {config_str}"
    result = run_arma_egarch_benchmark(y, asset, dist_name, config)
    return task_desc, result


def _run_single_arma_task(y: np.ndarray, asset: str, dist_name: str, config: Dict[str, Any]) -> Tuple[str, Optional[BenchmarkResult]]:
    """Wrapper for running single ARMA task with descriptive output."""
    mode = "log" if config["log_mode"] else "con"
    config_str = f"{config['solver']}[{mode}]"
    task_desc = f"ARMA/{asset}/{dist_name}: {config_str}"
    result = run_arma_benchmark(y, asset, dist_name, config)
    return task_desc, result


def _run_single_gjr_garch_task(resid: np.ndarray, asset: str, dist_name: str, config: Dict[str, Any]) -> Tuple[str, Optional[BenchmarkResult]]:
    """Wrapper for running single GJR-GARCH task with descriptive output."""
    mode = "log" if config["log_mode"] else "con"
    config_str = f"{config['solver']}[{mode}]"
    task_desc = f"GJR-GARCH/{asset}/{dist_name}: {config_str}"
    result = run_gjr_garch_benchmark(resid, asset, dist_name, config)
    return task_desc, result


def _run_single_egarch_task(
    resid: np.ndarray,
    asset: str,
    dist_name: str,
    config: Dict[str, Any],
    *,
    p: int,
    q: int,
    model_type: str,
) -> Tuple[str, Optional[BenchmarkResult]]:
    """Wrapper for running single EGARCH task with descriptive output."""
    mode = "log" if config["log_mode"] else "con"
    config_str = f"{config['solver']}[{mode}]"
    task_desc = f"EGARCH({p},{q})/{asset}/{dist_name}: {config_str}"
    result = run_egarch_benchmark(
        resid,
        asset,
        dist_name,
        config,
        p=p,
        q=q,
        model_type=model_type,
    )
    return task_desc, result


def run_full_benchmark(
    log_returns: Dict[str, pd.Series],
    ar1_residuals: Dict[str, pd.Series],
) -> List[BenchmarkResult]:
    """Run full benchmark across all models, assets, and configs using parallel processing."""
    
    # Build list of all tasks
    tasks = []
    task_types = []
    
    # =====================
    # 1. GARCH(1,1) Models (scivol)
    # =====================
    for asset in ASSETS:
        if asset not in ar1_residuals:
            continue
        resid = np.ascontiguousarray(ar1_residuals[asset], dtype=np.float64)
        
        for dist_name in GARCH_DIST_SPECS.keys():
            for config in GARCH_OPTIMIZER_CONFIGS:
                tasks.append(delayed(_run_single_garch_task)(resid, asset, dist_name, config))
                task_types.append("GARCH")
    
    # =====================
    # 2. ARMA(1,1)-GARCH(1,1) Models (scivol)
    # =====================
    for asset in ASSETS:
        if asset not in log_returns:
            continue
        y = np.ascontiguousarray(log_returns[asset], dtype=np.float64)
        
        for dist_name in ARMA_GARCH_DIST_SPECS.keys():
            for config in ARMA_GARCH_OPTIMIZER_CONFIGS:
                tasks.append(delayed(_run_single_arma_garch_task)(y, asset, dist_name, config))
                task_types.append("ARMA-GARCH")
    
    # =====================
    # 3. ARMA(1,1)-GJR-GARCH(1,1) Models (scivol)
    # =====================
    for asset in ASSETS:
        if asset not in log_returns:
            continue
        y = np.ascontiguousarray(log_returns[asset], dtype=np.float64)

        for dist_name in ARMA_GJR_GARCH_DIST_SPECS.keys():
            for config in ARMA_GJR_GARCH_OPTIMIZER_CONFIGS:
                tasks.append(delayed(_run_single_arma_gjr_garch_task)(y, asset, dist_name, config))
                task_types.append("ARMA-GJR-GARCH")

    # =====================
    # 4. GJR-GARCH(1,1) Models (scivol)
    # =====================
    for asset in ASSETS:
        if asset not in ar1_residuals:
            continue
        resid = np.ascontiguousarray(ar1_residuals[asset], dtype=np.float64)
        
        for dist_name in GJR_GARCH_DIST_SPECS.keys():
            for config in GJR_GARCH_OPTIMIZER_CONFIGS:
                tasks.append(delayed(_run_single_gjr_garch_task)(resid, asset, dist_name, config))
                task_types.append("GJR-GARCH")
    
    # =====================
    # 5. EGARCH Models (scivol)
    # =====================
    for asset in ASSETS:
        if asset not in ar1_residuals:
            continue
        resid = np.ascontiguousarray(ar1_residuals[asset], dtype=np.float64)

        for model_type, (p, q) in EGARCH_ORDER_SPECS.items():
            for dist_name in EGARCH_DIST_SPECS.keys():
                for config in EGARCH_OPTIMIZER_CONFIGS:
                    tasks.append(
                        delayed(_run_single_egarch_task)(
                            resid,
                            asset,
                            dist_name,
                            config,
                            p=p,
                            q=q,
                            model_type=model_type,
                        )
                    )
                    task_types.append(f"EGARCH({p},{q})")

    # =====================
    # 6. ARMA(1,1)-EGARCH(1,1) Models (scivol)
    # =====================
    for asset in ASSETS:
        if asset not in log_returns:
            continue
        y = np.ascontiguousarray(log_returns[asset], dtype=np.float64)

        for dist_name in ARMA_EGARCH_DIST_SPECS.keys():
            for config in ARMA_EGARCH_OPTIMIZER_CONFIGS:
                tasks.append(delayed(_run_single_arma_egarch_task)(y, asset, dist_name, config))
                task_types.append("ARMA-EGARCH")

    # =====================
    # 7. Pure ARMA(1,1) Models (scivol, constant variance)
    # =====================
    for asset in ASSETS:
        if asset not in log_returns:
            continue
        y = np.ascontiguousarray(log_returns[asset], dtype=np.float64)
        
        for dist_name in ARMA_DIST_SPECS.keys():
            for config in ARMA_OPTIMIZER_CONFIGS:
                tasks.append(delayed(_run_single_arma_task)(y, asset, dist_name, config))
                task_types.append("ARMA")
    
    total_tasks = len(tasks)
    print(f"\n{'=' * 60}")
    print(f"Running {total_tasks} benchmark tasks in parallel ({N_WORKERS} workers)")
    print(f"{'=' * 60}\n")
    
    # Count tasks by type
    from collections import Counter
    task_counts = Counter(task_types)
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} tasks")
    print()
    
    # Run all tasks in parallel with progress reporting
    start_time = time.perf_counter()
    results_with_desc = Parallel(n_jobs=N_WORKERS, verbose=10, backend='loky')(tasks)
    elapsed = time.perf_counter() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"Parallel execution completed in {elapsed:.1f}s")
    print(f"{'=' * 60}\n")
    
    # Extract results and print summary
    all_results = []
    n_converged = 0
    n_failed = 0
    
    for task_desc, result in results_with_desc:
        if result is not None:
            all_results.append(result)
            if result.converged:
                n_converged += 1
            else:
                n_failed += 1
    
    print(f"Results: {n_converged} converged, {n_failed} failed, {len(all_results)} total")
    
    return all_results


# =============================================================================
# ANALYSIS & REPORTING
# =============================================================================

def results_to_dataframe(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Convert results to DataFrame."""
    records = []
    for r in results:
        records.append({
            "model_type": r.model_type,
            "asset": r.asset,
            "dist": r.dist,
            "solver": r.solver,
            "log_mode": r.log_mode,
            "converged": r.converged,
            "n_iter": r.n_iter,
            "time_s": r.time_elapsed,
            "log_lik": r.log_likelihood,
            "omega": r.omega,
            "alpha": r.alpha,
            "gamma": r.gamma,
            "beta": r.beta,
            "persistence": r.persistence,
            "c": r.c,
            "phi": r.phi,
            "theta": r.theta,
            "nu": r.nu,
            "lam": r.lam,
            "aic": r.aic,
            "bic": r.bic,
            "error_message": r.error_message,
        })
    return pd.DataFrame(records)


def _mode_label(log_mode: bool) -> str:
    return "z" if log_mode else "theta"


def _choose_mode_recommendation(
    theta_rate: float,
    z_rate: float,
    theta_aic: float,
    z_aic: float,
    theta_time: float,
    z_time: float,
) -> Tuple[str, str]:
    """Choose a mode recommendation from benchmark aggregates."""
    conv_tol = 0.05
    aic_tol = 0.25
    speed_tol = 0.90

    if np.isfinite(theta_rate) and np.isfinite(z_rate):
        if z_rate >= theta_rate + conv_tol:
            return "z", "higher convergence"
        if theta_rate >= z_rate + conv_tol:
            return "theta", "higher convergence"

    if np.isfinite(theta_aic) and np.isfinite(z_aic):
        if z_aic <= theta_aic - aic_tol:
            return "z", "better AIC"
        if theta_aic <= z_aic - aic_tol:
            return "theta", "better AIC"

    if np.isfinite(theta_time) and np.isfinite(z_time):
        if z_time <= theta_time * speed_tol:
            return "z", "faster"
        if theta_time <= z_time * speed_tol:
            return "theta", "faster"

    return "ambiguous", "no decisive edge"


def build_theta_z_policy_table(
    df: pd.DataFrame,
    *,
    solver: str = DEFAULT_POLICY_SOLVER,
) -> pd.DataFrame:
    """Build explicit theta-vs-z comparison table for default-path decisions."""
    rows = []
    solver_df = df[df["solver"] == solver].copy()

    for model_type in MODEL_DISPLAY_ORDER:
        model_df = solver_df[solver_df["model_type"] == model_type]
        if model_df.empty:
            continue

        for dist in sorted(model_df["dist"].dropna().unique()):
            dist_df = model_df[model_df["dist"] == dist]
            theta_df = dist_df[dist_df["log_mode"] == False]
            z_df = dist_df[dist_df["log_mode"] == True]

            if theta_df.empty or z_df.empty:
                continue

            theta_conv = theta_df["converged"].mean()
            z_conv = z_df["converged"].mean()
            theta_time = theta_df.loc[theta_df["converged"], "time_s"].mean()
            z_time = z_df.loc[z_df["converged"], "time_s"].mean()
            theta_aic = theta_df.loc[theta_df["converged"], "aic"].mean()
            z_aic = z_df.loc[z_df["converged"], "aic"].mean()

            theta_conv_params = theta_df.loc[theta_df["converged"], ["asset", *PARAMETER_COLUMNS]]
            z_conv_params = z_df.loc[z_df["converged"], ["asset", *PARAMETER_COLUMNS]]
            paired = theta_conv_params.merge(z_conv_params, on="asset", suffixes=("_theta", "_z"))

            param_delta_l1 = np.nan
            param_delta_max = np.nan
            aic_delta = np.nan
            time_ratio = np.nan

            if not paired.empty:
                diff_cols = {}
                for col in PARAMETER_COLUMNS:
                    theta_col = f"{col}_theta"
                    z_col = f"{col}_z"
                    diff_cols[col] = (paired[z_col] - paired[theta_col]).abs()

                diff_frame = pd.DataFrame(diff_cols).dropna(axis=1, how="all")
                if not diff_frame.empty:
                    param_delta_l1 = diff_frame.fillna(0.0).sum(axis=1).median()
                    param_delta_max = diff_frame.max(axis=1).median()

                paired_metrics = theta_df.loc[theta_df["converged"], ["asset", "aic", "time_s"]].merge(
                    z_df.loc[z_df["converged"], ["asset", "aic", "time_s"]],
                    on="asset",
                    suffixes=("_theta", "_z"),
                )
                if not paired_metrics.empty:
                    aic_delta = (paired_metrics["aic_z"] - paired_metrics["aic_theta"]).mean()
                    theta_mean_time = paired_metrics["time_s_theta"].mean()
                    z_mean_time = paired_metrics["time_s_z"].mean()
                    if np.isfinite(theta_mean_time) and theta_mean_time > 0.0 and np.isfinite(z_mean_time):
                        time_ratio = z_mean_time / theta_mean_time

            recommendation, reason = _choose_mode_recommendation(
                theta_conv,
                z_conv,
                theta_aic,
                z_aic,
                theta_time,
                z_time,
            )

            rows.append(
                {
                    "model_type": model_type,
                    "dist": dist,
                    "policy_solver": solver,
                    "theta_convergence": theta_conv,
                    "z_convergence": z_conv,
                    "theta_time_s": theta_time,
                    "z_time_s": z_time,
                    "theta_aic": theta_aic,
                    "z_aic": z_aic,
                    "paired_assets": len(paired),
                    "mean_aic_delta_z_minus_theta": aic_delta,
                    "median_param_l1_delta": param_delta_l1,
                    "median_param_max_delta": param_delta_max,
                    "time_ratio_z_over_theta": time_ratio,
                    "recommended_mode": recommendation,
                    "recommendation_reason": reason,
                }
            )

    return pd.DataFrame(rows)


def _format_float(value: Any, fmt: str) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return format(float(value), fmt)


def write_theta_z_policy_report(policy_df: pd.DataFrame, output_path: Path) -> None:
    """Write human-readable theta-vs-z benchmark report."""
    lines = [
        "# Theta vs Z policy benchmark",
        "",
        f"Primary decision solver: `{DEFAULT_POLICY_SOLVER}`",
        "",
        "| Model | Dist | Theta conv | Z conv | Theta time | Z time | Z-Theta AIC | Paired assets | Median L1 delta | Recommendation | Why |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for row in policy_df.itertuples(index=False):
        lines.append(
            "| "
            f"{row.model_type} | "
            f"{row.dist} | "
            f"{_format_float(row.theta_convergence, '.1%')} | "
            f"{_format_float(row.z_convergence, '.1%')} | "
            f"{_format_float(row.theta_time_s, '.3f')}s | "
            f"{_format_float(row.z_time_s, '.3f')}s | "
            f"{_format_float(row.mean_aic_delta_z_minus_theta, '.3f')} | "
            f"{row.paired_assets} | "
            f"{_format_float(row.median_param_l1_delta, '.4g')} | "
            f"{row.recommended_mode} | "
            f"{row.recommendation_reason} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_theta_z_summary(policy_df: pd.DataFrame) -> None:
    """Print concise theta-vs-z policy summary."""
    print("\n" + "-" * 80)
    print(f"THETA VS Z POLICY SUMMARY ({DEFAULT_POLICY_SOLVER.upper()})")
    print("-" * 80)

    if policy_df.empty:
        print("No theta-vs-z comparison rows available.")
        return

    for row in policy_df.itertuples(index=False):
        print(
            f"  {row.model_type:12s} + {row.dist:8s}: "
            f"theta_conv={_format_float(row.theta_convergence, '.1%')}, "
            f"z_conv={_format_float(row.z_convergence, '.1%')}, "
            f"theta_time={_format_float(row.theta_time_s, '.3f')}s, "
            f"z_time={_format_float(row.z_time_s, '.3f')}s, "
            f"recommend={row.recommended_mode} ({row.recommendation_reason})"
        )


def print_summary(df: pd.DataFrame):
    """Print summary report."""
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal runs: {len(df)}")
    print(f"Assets tested: {df['asset'].nunique()}")
    print(f"Model types: {df['model_type'].unique().tolist()}")
    
    # Summary by model type
    for model_type in MODEL_DISPLAY_ORDER:
        model_df = df[df["model_type"] == model_type]
        if len(model_df) == 0:
            continue
        
        print("\n" + "-" * 80)
        print(f"{model_type.upper()} MODELS")
        print("-" * 80)
        
        # Group by distribution and config
        groups = model_df.groupby(["dist", "solver", "log_mode"])
        
        for name, group in groups:
            n_conv = group["converged"].sum()
            n_total = len(group)
            rate = n_conv / n_total if n_total > 0 else 0
            avg_time = group[group["converged"]]["time_s"].mean() if n_conv > 0 else np.nan
            
            if isinstance(name, tuple):
                config_str = "/".join(str(x) for x in name)
            else:
                config_str = str(name)
            
            time_str = f"{avg_time:.3f}s" if not np.isnan(avg_time) else "N/A"
            print(f"  {config_str:40s}: {n_conv:2d}/{n_total:2d} ({rate:5.1%}), time={time_str}")
    
    print("\n" + "-" * 80)
    print("DEFAULT-POLICY NOTE")
    print("-" * 80)
    print(
        f"Use the theta-vs-z policy table below for default log_mode decisions "
        f"under solver='{DEFAULT_POLICY_SOLVER}'."
    )


# =============================================================================
# MODEL SMOKE TESTS
# =============================================================================

def verify_all_models() -> Dict[str, bool]:
    """
    Quick smoke test that all model types run and converge with scivol API.
    
    Returns dict mapping model name -> converged
    """
    print("\n" + "=" * 60)
    print("MODEL VERIFICATION (Smoke Tests)")
    print("=" * 60)
    
    np.random.seed(42)
    n = 1000
    y = np.random.randn(n) * 0.01
    x = np.random.randn(n, 1) * 0.1
    
    results = {}
    
    # Test each model type
    model_specs = [
        ("ARMA(1,1)+Normal", ARMA(1, 1) + Normal(), {}),
        ("GARCH(1,1)+Normal", GARCH(1, 1) + Normal(), {}),
        ("GARCH(1,1)+StudentT", GARCH(1, 1) + StudentT(), {}),
        ("GARCH(1,1)+SkewT", GARCH(1, 1) + SkewT(), {}),
        ("GJR-GARCH(1,1)+Normal", GJRGARCH(1, 1) + Normal(), {}),
        ("GJR-GARCH(1,1)+StudentT", GJRGARCH(1, 1) + StudentT(), {}),
        ("GJR-GARCH(1,1)+SkewT", GJRGARCH(1, 1) + SkewT(), {}),
        ("GJR-GARCH(1,1)+GED", GJRGARCH(1, 1) + GED(), {}),
        ("EGARCH(1,1)+Normal", EGARCH(1, 1) + Normal(), {}),
        ("EGARCH(1,1)+StudentT", EGARCH(1, 1) + StudentT(), {}),
        ("EGARCH(1,1)+SkewT", EGARCH(1, 1) + SkewT(), {}),
        ("EGARCH(1,1)+GED", EGARCH(1, 1) + GED(), {}),
        ("EGARCH(2,1)+Normal", EGARCH(2, 1) + Normal(), {}),
        ("EGARCH(2,1)+StudentT", EGARCH(2, 1) + StudentT(), {}),
        ("EGARCH(2,1)+SkewT", EGARCH(2, 1) + SkewT(), {}),
        ("EGARCH(2,1)+GED", EGARCH(2, 1) + GED(), {}),
        ("ARMA(1,1)+GARCH(1,1)+Normal", ARMA(1, 1) + GARCH(1, 1) + Normal(), {}),
        ("ARMA(1,1)+GARCH(1,1)+StudentT", ARMA(1, 1) + GARCH(1, 1) + StudentT(), {}),
        ("ARMA(1,1)+GARCH(1,1)+SkewT", ARMA(1, 1) + GARCH(1, 1) + SkewT(), {}),
        ("ARMA(1,1)+GJR-GARCH(1,1)+Normal", ARMA(1, 1) + GJRGARCH(1, 1) + Normal(), {}),
        ("ARMA(1,1)+GJR-GARCH(1,1)+StudentT", ARMA(1, 1) + GJRGARCH(1, 1) + StudentT(), {}),
        ("ARMA(1,1)+GJR-GARCH(1,1)+SkewT", ARMA(1, 1) + GJRGARCH(1, 1) + SkewT(), {}),
        ("ARMA(1,1)+GJR-GARCH(1,1)+GED", ARMA(1, 1) + GJRGARCH(1, 1) + GED(), {}),
        ("ARMA(1,1)+EGARCH(1,1)+Normal", ARMA(1, 1) + EGARCH(1, 1) + Normal(), {}),
        ("ARMA(1,1)+EGARCH(1,1)+StudentT", ARMA(1, 1) + EGARCH(1, 1) + StudentT(), {}),
        ("ARMA(1,1)+EGARCH(1,1)+SkewT", ARMA(1, 1) + EGARCH(1, 1) + SkewT(), {}),
        ("ARMA(1,1)+EGARCH(1,1)+GED", ARMA(1, 1) + EGARCH(1, 1) + GED(), {}),
        ("ARX(1)+Normal", ARX(1) + Normal(), {"x": x}),
        ("HARX(1,5)+Normal", HARX((1, 5)) + Normal(), {"x": x}),
        ("ARX(1)+StudentT", ARX(1) + StudentT(), {"x": x}),
        ("HARX(1,5)+SkewT", HARX((1, 5)) + SkewT(), {"x": x}),
        ("ARX(1)+GED", ARX(1) + GED(), {"x": x}),
        ("ARX(1)+GARCH(1,1)+StudentT", ARX(1) + GARCH(1, 1) + StudentT(), {"x": x}),
        ("HARX(1,5)+GARCH(1,1)+SkewT", HARX((1, 5)) + GARCH(1, 1) + SkewT(), {"x": x}),
        ("ARX(1)+GARCH(1,1)+GED", ARX(1) + GARCH(1, 1) + GED(), {"x": x}),
        ("ARX(1)+GJR-GARCH(1,1)+StudentT", ARX(1) + GJRGARCH(1, 1) + StudentT(), {"x": x}),
        ("HARX(1,5)+GJR-GARCH(1,1)+SkewT", HARX((1, 5)) + GJRGARCH(1, 1) + SkewT(), {"x": x}),
        ("ARX(1)+GJR-GARCH(1,1)+GED", ARX(1) + GJRGARCH(1, 1) + GED(), {"x": x}),
        ("ARX(1)+EGARCH(1,1)+StudentT", ARX(1) + EGARCH(1, 1) + StudentT(), {"x": x}),
        ("HARX(1,5)+EGARCH(2,1)+SkewT", HARX((1, 5)) + EGARCH(2, 1) + SkewT(), {"x": x}),
        ("ARX(1)+EGARCH(2,1)+GED", ARX(1) + EGARCH(2, 1) + GED(), {"x": x}),
    ]
    
    for name, spec, fit_kwargs in model_specs:
        print(f"\n[{name}] ", end="")
        try:
            result = spec.fit(y, solver="slsqp", verbose=False, **fit_kwargs)
            success = result.converged if hasattr(result, 'converged') else True
            ll = result.log_likelihood
            
            # Basic sanity checks
            is_valid = (
                success and
                np.isfinite(ll) and
                ll > 0 and  # LL should be positive for these small returns
                np.all(np.isfinite(result.params))
            )
            
            results[name] = is_valid
            print(f"{'✓' if is_valid else '✗'} LL={ll:.2f}, success={success}")
            
            if not is_valid:
                print(f"    Params: {result.params}")
        except Exception as e:
            results[name] = False
            print(f"✗ Error: {e}")
    
    # Summary
    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\nModel verification: {n_pass}/{n_total} passed")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete benchmark."""
    
    print("=" * 80)
    print("SCIVOL OPTIMIZER BENCHMARK")
    print("All models use C extensions via scivol")
    print("Models: GARCH, GJR-GARCH, EGARCH, ARMA, ARMA-GARCH, ARMA-EGARCH, ARX/HARX mean-only, ARX/HARX linked")
    print("=" * 80)
    print()
    
    # Verify all models run first
    model_results = verify_all_models()
    
    # Load data
    log_returns, ar1_residuals = load_data()
    
    # Limit to available assets
    available_assets = list(ar1_residuals.keys())
    print(f"\nAssets available: {available_assets}")
    
    # Override global ASSETS to only use available
    global ASSETS
    ASSETS = available_assets
    
    # Run benchmark
    start_time = time.perf_counter()
    results = run_full_benchmark(log_returns, ar1_residuals)
    elapsed = time.perf_counter() - start_time
    
    print(f"\nBenchmark completed in {elapsed:.1f}s")
    
    # Convert to DataFrames
    df = results_to_dataframe(results)
    
    # Print summary
    print_summary(df)

    # Explicit theta-vs-z policy report for default-path decisions
    policy_df = build_theta_z_policy_table(df, solver=DEFAULT_POLICY_SOLVER)
    print_theta_z_summary(policy_df)
    
    # Save results
    df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
    policy_df.to_csv(RESULTS_DIR / "theta_z_policy_summary.csv", index=False)
    write_theta_z_policy_report(policy_df, RESULTS_DIR / "theta_z_policy_summary.md")
    
    print(f"\nResults saved to {RESULTS_DIR}/")
    
    return df


if __name__ == "__main__":
    df = main()
