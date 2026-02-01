"""
Optimizer Benchmark for volkit Models
=====================================

Comprehensive benchmark of optimization methods for GARCH and ARMA-GARCH models
using real financial data. Results guide default parameter settings in volkit.

This script:
1. Tests all optimizer/mode combinations on multiple asset classes
2. Tracks: convergence, speed, parameter stability, log-likelihood
3. Produces summary statistics for setting defaults

Models tested (all via volkit C extensions):
- GARCH(1,1) + Normal/Student-t/Skew-t
- ARMA(1,1)-GARCH(1,1) + Normal/Student-t/Skew-t

Keep this file evergreen - run periodically to validate/update defaults.
"""
from __future__ import annotations

import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Literal

# volkit library (all models via C extensions)
from volkit import ARMA, GARCH, Normal, StudentT, SkewT

# Utilities for diagnostic tests
from utilities import ar1

# Suppress convergence warnings during benchmark
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("benchmark_results")
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
# GARCH-only configs (volkit)
# =====================
GARCH_DIST_SPECS = {
    "normal": Normal(),
    "studentt": StudentT(),
    "skewt": SkewT(),
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
# ARMA-GARCH configs (volkit)
# =====================
# Mapping for ARMA-GARCH density components
ARMA_GARCH_DIST_SPECS = {
    "normal": Normal(),
    "studentt": StudentT(),
    "skewt": SkewT(),
}

ARMA_GARCH_OPTIMIZER_CONFIGS = [
    {"solver": "nelder-mead", "log_mode": False},
    {"solver": "slsqp", "log_mode": False},
    {"solver": "trust", "log_mode": False},
]


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    model_type: str  # "garch", "arma", "arma_garch"
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
    data_raw = pd.read_csv("data/DATA_HW1.csv", skiprows=1, parse_dates=["DATE"], dayfirst=True)
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
# GARCH BENCHMARK (volkit)
# =============================================================================

def run_garch_benchmark(
    resid: np.ndarray,
    asset: str,
    dist_name: str,
    config: Dict[str, Any],
) -> Optional[BenchmarkResult]:
    """Run single GARCH benchmark using volkit."""
    try:
        # Build model spec
        dist_component = GARCH_DIST_SPECS[dist_name]
        spec = GARCH(1, 1) + dist_component
        
        # Fit using volkit
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
        
        converged = result.success if hasattr(result, 'success') else True
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
            log_likelihood=result.loglikelihood,
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
        )


# =============================================================================
# ARMA(1,1)-GARCH(1,1) BENCHMARK (volkit)
# =============================================================================

def run_arma_garch_benchmark(
    y: np.ndarray,
    asset: str,
    dist_name: str,
    config: Dict[str, Any],
) -> Optional[BenchmarkResult]:
    """Run single ARMA(1,1)-GARCH(1,1) benchmark using volkit."""
    try:
        # Build model spec using volkit components
        dist_component = ARMA_GARCH_DIST_SPECS[dist_name]
        spec = ARMA(1, 1) + GARCH(1, 1) + dist_component
        
        # Fit using volkit
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
        
        converged = result.success if hasattr(result, 'success') else True
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
            log_likelihood=result.loglikelihood,
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
        )


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_full_benchmark(
    log_returns: Dict[str, pd.Series],
    ar1_residuals: Dict[str, pd.Series],
) -> List[BenchmarkResult]:
    """Run full benchmark across all models, assets, and configs."""
    
    all_results: List[BenchmarkResult] = []
    
    # Count total configs
    n_garch = len(ASSETS) * len(GARCH_DIST_SPECS) * len(GARCH_OPTIMIZER_CONFIGS)
    n_arma_garch = len(ASSETS) * len(ARMA_GARCH_DIST_SPECS) * len(ARMA_GARCH_OPTIMIZER_CONFIGS)
    total_configs = n_garch + n_arma_garch
    current = 0
    
    # =====================
    # 1. GARCH(1,1) Models (volkit)
    # =====================
    print("\n" + "=" * 60)
    print("GARCH(1,1) Models (volkit)")
    print("=" * 60)
    
    for asset in ASSETS:
        if asset not in ar1_residuals:
            continue
            
        resid = np.ascontiguousarray(ar1_residuals[asset], dtype=np.float64)
        
        for dist_name in GARCH_DIST_SPECS.keys():
            for config in GARCH_OPTIMIZER_CONFIGS:
                current += 1
                mode = "log" if config["log_mode"] else "con"
                config_str = f"{config['solver']}[{mode}]"
                print(f"[{current}/{total_configs}] GARCH/{asset}/{dist_name}: {config_str}...", end=" ", flush=True)
                
                result = run_garch_benchmark(resid, asset, dist_name, config)
                if result is not None:
                    all_results.append(result)
                    status = "✓" if result.converged else "✗"
                    print(f"{status} {result.time_elapsed:.3f}s")
                else:
                    print("FAILED")
    
    # =====================
    # 2. ARMA(1,1)-GARCH(1,1) Models (volkit)
    # =====================
    print("\n" + "=" * 60)
    print("ARMA(1,1)-GARCH(1,1) Models (volkit)")
    print("=" * 60)
    
    for asset in ASSETS:
        if asset not in log_returns:
            continue
            
        y = np.ascontiguousarray(log_returns[asset], dtype=np.float64)
        
        for dist_name in ARMA_GARCH_DIST_SPECS.keys():
            for config in ARMA_GARCH_OPTIMIZER_CONFIGS:
                current += 1
                mode = "log" if config["log_mode"] else "con"
                config_str = f"{config['solver']}[{mode}]"
                print(f"[{current}/{total_configs}] ARMA-GARCH/{asset}/{dist_name}: {config_str}...", end=" ", flush=True)
                
                result = run_arma_garch_benchmark(y, asset, dist_name, config)
                if result is not None:
                    all_results.append(result)
                    status = "✓" if result.converged else "✗"
                    print(f"{status} {result.time_elapsed:.3f}s")
                else:
                    print("FAILED")
    
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
            "beta": r.beta,
            "persistence": r.persistence,
            "c": r.c,
            "phi": r.phi,
            "theta": r.theta,
            "nu": r.nu,
            "lam": r.lam,
            "aic": r.aic,
            "bic": r.bic,
        })
    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame):
    """Print summary report."""
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal runs: {len(df)}")
    print(f"Assets tested: {df['asset'].nunique()}")
    print(f"Model types: {df['model_type'].unique().tolist()}")
    
    # Summary by model type
    for model_type in ["garch", "arma_garch"]:
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
    
    # Recommendations
    print("\n" + "-" * 80)
    print("RECOMMENDED DEFAULTS")
    print("-" * 80)
    
    for model_type in ["garch", "arma_garch"]:
        model_df = df[(df["model_type"] == model_type) & (df["converged"])]
        if len(model_df) == 0:
            continue
        
        # Best config by convergence and speed
        for dist in model_df["dist"].unique():
            dist_df = model_df[model_df["dist"] == dist]
            best = dist_df.groupby(["solver", "log_mode"]).agg({
                "time_s": "mean",
                "converged": "count"
            }).reset_index()
            best = best.sort_values(["converged", "time_s"], ascending=[False, True]).iloc[0]
            print(f"\n{model_type.upper()} + {dist.upper()}:")
            print(f"  solver=\"{best['solver']}\", log_mode={best['log_mode']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete benchmark."""
    
    print("=" * 80)
    print("VOLKIT OPTIMIZER BENCHMARK")
    print("All models use C extensions via volkit")
    print("Models: GARCH(1,1), ARMA(1,1)-GARCH(1,1)")
    print("=" * 80)
    print()
    
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
    
    # Save results
    df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
    
    print(f"\nResults saved to {RESULTS_DIR}/")
    
    return df


if __name__ == "__main__":
    df = main()
