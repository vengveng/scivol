"""
GARCH Solver Comparison for Student-t and Skew-t
=================================================

Quick test to verify we're using the best solver for each distribution.
Results inform the choice in analysis.py but this file is not part of final submission.
"""

import numpy as np
import pandas as pd
from utilities import ar1
from garch_estimator import fit_garch

# =============================================================================
# DATA LOADING
# =============================================================================

data_raw = pd.read_csv("data/DATA_HW1.csv", skiprows=1, parse_dates=["DATE"], dayfirst=True)
data = data_raw.rename(columns={
    "S&PCOMP(RI)": "stock",
    "SPUHYBD(RI)": "cbond",
}).set_index("DATE")[["stock", "cbond"]]

lr = np.log1p(data.pct_change(fill_method=None))
mask_zero = (lr == 0).any(axis=1)
lr = lr[~mask_zero]

# AR(1) residuals
AR1_RESID = {}
for asset in ["stock", "cbond"]:
    ar1_res = ar1(lr[asset])
    AR1_RESID[asset] = np.asarray(ar1_res["resid"], dtype=np.float64)

print(f"Data: {len(lr)} observations after removing {mask_zero.sum()} zero-return days\n")


# =============================================================================
# SOLVER CONFIGURATIONS
# =============================================================================

# Constrained optimization (bounds + linear constraint)
# (per request) do NOT test trust-constr in non-log mode
SOLVERS_CONSTRAINED = ["nelder-mead", "slsqp"]

# Unconstrained log-space optimization (stationarity enforced by construction)
# Note: for Normal, "trust-exact" is supported; for Student-t/Skew-t it's not.
SOLVERS_LOGSPACE = ["nelder-mead", "slsqp", "trust-constr"]

# For Student-t, we can use analytical derivatives
# For Skew-t, only numerical derivatives available


def test_solvers(
    asset: str,
    dist: str,
    solvers: list[str],
    use_derivatives: bool,
    use_logspace: bool,
) -> pd.DataFrame:
    """Test multiple solvers and return comparison DataFrame."""
    eps = AR1_RESID[asset]
    results = []
    
    for solver in solvers:
        try:
            r = fit_garch(
                eps, 
                dist=dist, 
                method="mle", 
                p=1, q=1,
                solver=solver, 
                use_derivatives=use_derivatives,
                use_logspace=use_logspace,
                verbose=False,
            )
            
            results.append({
                "mode": "logspace" if use_logspace else "constrained",
                "solver": solver,
                "converged": r.converged,
                "omega": r.garch_params.omega,
                "alpha": r.garch_params.alpha[0],
                "beta": r.garch_params.beta[0],
                "persistence": r.garch_params.persistence,
                "nu": r.dist_params.nu,
                "lambda": r.dist_params.lam if dist == "skewt" else None,
                "log_lik": r.log_likelihood,
                "n_iter": r.n_iter,
                "time": r.time_elapsed,
                "se_available": r.std_errors is not None,
            })
        except Exception as e:
            results.append({
                "mode": "logspace" if use_logspace else "constrained",
                "solver": solver,
                "converged": False,
                "error": str(e),
            })
    
    return pd.DataFrame(results)


def print_comparison(df: pd.DataFrame, title: str):
    """Print solver comparison table."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    
    # Find best by log-likelihood
    valid = df[df["converged"] == True].copy()
    if len(valid) > 0:
        best_idx = valid["log_lik"].idxmax()
        best_ll = valid.loc[best_idx, "log_lik"]
    else:
        best_ll = None
    
    print(f"\n{'Mode':>11}  {'Solver':>14}  {'Conv':>5}  {'ω':>12}  {'α':>8}  {'β':>8}  "
          f"{'α+β':>8}  {'ν':>6}  {'λ':>8}  {'LogLik':>12}  {'Time':>8}  {'SEs':>5}")
    print(f"{'─'*11}  {'─'*14}  {'─'*5}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*12}  {'─'*8}  {'─'*5}")
    
    for _, row in df.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            print(f"{row['mode']:>11}  {row['solver']:>14}  ERROR: {row['error']}")
            continue
        
        conv = "Yes" if row["converged"] else "No"
        lam_str = f"{row['lambda']:.4f}" if pd.notna(row.get("lambda")) else "—"
        se_str = "Yes" if row.get("se_available") else "No"
        
        # Mark best with asterisk
        marker = " *" if best_ll and abs(row["log_lik"] - best_ll) < 0.01 else "  "
        
        print(f"{row['mode']:>11}  {row['solver']:>14}  {conv:>5}  {row['omega']:>12.2e}  {row['alpha']:>8.4f}  "
              f"{row['beta']:>8.4f}  {row['persistence']:>8.4f}  {row['nu']:>6.2f}  "
              f"{lam_str:>8}  {row['log_lik']:>12.2f}{marker} {row['time']:>7.2f}s  {se_str:>5}")
    
    if best_ll:
        print(f"\n* Best log-likelihood: {best_ll:.2f}")


# =============================================================================
# RUN TESTS
# =============================================================================

print("\n" + "#" * 100)
print("# STUDENT-T GARCH SOLVER COMPARISON")
print("#" * 100)

for asset in ["stock", "cbond"]:
    df_c = test_solvers(asset, "studentt", SOLVERS_CONSTRAINED, use_derivatives=True, use_logspace=False)
    df_l = test_solvers(asset, "studentt", SOLVERS_LOGSPACE, use_derivatives=False, use_logspace=True)
    df = pd.concat([df_c, df_l], ignore_index=True)
    print_comparison(df, f"Student-t GARCH — {asset.upper()}")

print("\n" + "#" * 100)
print("# SKEW-T GARCH SOLVER COMPARISON")
print("#" * 100)

for asset in ["stock", "cbond"]:
    df_c = test_solvers(asset, "skewt", SOLVERS_CONSTRAINED, use_derivatives=False, use_logspace=False)
    df_l = test_solvers(asset, "skewt", SOLVERS_LOGSPACE, use_derivatives=False, use_logspace=True)
    df = pd.concat([df_c, df_l], ignore_index=True)
    print_comparison(df, f"Skew-t GARCH — {asset.upper()}")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 100)
print("""
Student-t:
  - Compare constrained vs log-space; pick the fastest converged solver at the best LL.
  - Constrained: trust-constr + analytical derivatives is usually fastest when it behaves.
  - Log-space: tends to be more stable near the stationarity boundary, but (currently)
    runs without analytical derivatives in this estimator.

Skew-t:
  - Compare constrained vs log-space; no analytical derivatives available.
  - nelder-mead is typically most robust but slow.
  - trust-constr can be fast but sometimes finicky (quasi-Newton warnings).
""")
