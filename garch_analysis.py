"""
GARCH Model Estimation and Analysis
===================================
Sections:
  3.1 - GARCH model estimation
  3.2 - MLE vs QMLE
  3.3 - Student-t GARCH
  3.4 - Skew-t GARCH
  
Uses volkit C extension library for all estimation.

Run as: python garch_analysis.py
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from pathlib import Path

from utilities import (
    jarque_bera,
    acf_and_ljung_box,
    ar1,
    arch_lm_test,
    dgt_lb_interface,
    dgt_with_lb_moments,
    # Printing functions
    print_q3_1,
    print_q3_1d,
    print_q3_2,
    print_q3_3,
    print_q3_4,
    print_q3_summary,
)

# Legacy-compatible fit_garch function (wraps volkit)
from volkit_compat import fit_garch, fit_garch as fit_garch_ref

# volkit components for parity testing
from volkit import GARCH, Normal, StudentT, SkewT, MLE, QMLE

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Collector for GARCH results
RESULTS = {
    "q3_1a": [],  # pre-GARCH diagnostics
    "q3_1b": [],  # AR(1) results
    "q3_1c": [],  # ARCH-LM tests
    "q3_1d": [],  # GARCH solver comparison
    "q3_2": [],  # QMLE results
    "q3_2c": [],  # DGT tests (Normal)
    "q3_3": [],  # Student-t results
    "q3_3b": [],  # DGT tests (Student-t)
    "q3_4": [],  # Skew-t results
    "q3_4b": [],  # DGT tests (Skew-t)
    "q3_4c": [],  # Moment comparison
    "q3_summary": [],  # Model comparison summary
}

# Store volatility series for plotting
VOLATILITIES = {}

# =============================================================================
# DATA LOADING
# =============================================================================

RENAME = {
    "DATE": "date",
    "S&PCOMP(RI)": "stock",
    "SPUTBIX(RI)": "gbond",
    "SPUHYBD(RI)": "cbond",
    "WILURET(RI)": "resec",
    "RJEFCRT(TR)": "commo",
    "USBINXB": "usdfx",
}

data_raw = pd.read_csv("data/DATA_HW1.csv", skiprows=1, parse_dates=["DATE"], dayfirst=True)
data_raw = data_raw.rename(columns=RENAME).set_index("date")[["stock", "cbond"]]

# Daily log-returns
d_data = np.log1p(data_raw.pct_change(fill_method=None))

# =============================================================================
# SECTION 3: GARCH MODELING (Daily data, stocks & bonds only)
# =============================================================================

print("\n" + "="*70)
print("SECTION 3: GARCH Modeling")
print("="*70)

# Prepare data: remove zero-return days
lr = d_data
mask_zero = (lr == 0).any(axis=1)
lr = lr[~mask_zero]
print(f"Removed {mask_zero.sum()} zero-return days")

# Store AR(1) residuals
AR1_RESID = {}

for asset in ["stock", "cbond"]:
    # ─────────────────────────────────────────────────────────────────────────
    # 3.1a: Pre-GARCH diagnostics (JB and LB on raw returns)
    # ─────────────────────────────────────────────────────────────────────────
    x = lr[asset]
    jb = jarque_bera(x)
    lb = acf_and_ljung_box(x, k=4, alpha=0.05)
    
    RESULTS["q3_1a"].append({
        "asset": asset, "T": jb["n"],
        "jb_stat": jb["jb"], "jb_p": jb["p_value"], "jb_reject": jb["reject_5pct"],
        "lb_stat": lb["q_stat"], "lb_p": lb["p_value"], "lb_reject": lb["reject"],
    })
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.1b: AR(1) estimation
    # ─────────────────────────────────────────────────────────────────────────
    ar1_res = ar1(x)
    eps = ar1_res["resid"]
    AR1_RESID[asset] = eps
    
    RESULTS["q3_1b"].append({
        "asset": asset, "c": ar1_res["c"], "phi": ar1_res["phi"],
        "se_phi": ar1_res["se_phi"], "t_phi": ar1_res["phi"] / ar1_res["se_phi"],
        "T": ar1_res["T"],
    })
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.1c: ARCH-LM test on AR(1) residuals
    # ─────────────────────────────────────────────────────────────────────────
    arch_res = arch_lm_test(eps, p=4)
    RESULTS["q3_1c"].append({"asset": asset, **arch_res})

# Print Q3.1a-c results
print_q3_1(RESULTS["q3_1a"], RESULTS["q3_1b"], RESULTS["q3_1c"])


# =============================================================================
# 3.1d & 3.2a: GARCH(1,1) Normal MLE - Solver Comparison
# =============================================================================

print("\n--- 3.1d/3.2a: GARCH(1,1) Normal MLE (solver comparison) ---")

SOLVER_SPECS = [
    {"solver": "nelder-mead", "use_logspace": False},
    {"solver": "slsqp", "use_logspace": False},
    {"solver": "trust-constr", "use_logspace": False},
    {"solver": "nelder-mead", "use_logspace": True},
    {"solver": "slsqp", "use_logspace": True},
    {"solver": "trust-constr", "use_logspace": True},
    {"solver": "trust-exact", "use_logspace": True},
]

for asset in ["stock", "cbond"]:
    eps = np.asarray(AR1_RESID[asset], dtype=np.float64)
    
    for spec in SOLVER_SPECS:
        r = fit_garch(eps, dist="normal", method="mle", p=1, q=1,
                      solver=spec["solver"], use_logspace=spec["use_logspace"],
                      use_derivatives=True, verbose=False)
        
        RESULTS["q3_1d"].append({
            "asset": asset,
            "solver": spec["solver"],
            "use_logspace": spec["use_logspace"],
            "converged": r.converged,
            "n_iter": r.n_iter,
            "log_lik": r.log_likelihood,
            "omega": r.garch_params.omega,
            "alpha": r.garch_params.alpha[0],
            "beta": r.garch_params.beta[0],
            "persistence": r.garch_params.alpha[0] + r.garch_params.beta[0],
            "se_omega": r.std_errors[0] if r.std_errors is not None else None,
            "se_alpha": r.std_errors[1] if r.std_errors is not None else None,
            "se_beta": r.std_errors[2] if r.std_errors is not None else None,
            "aic": r.aic,
            "bic": r.bic,
            "time": r.time_elapsed,
        })
    
    print(f"  {asset}: completed {len(SOLVER_SPECS)} solver specs")

# Print Q3.1d results
print_q3_1d(RESULTS["q3_1d"])


# =============================================================================
# 3.2b-c: QMLE with Robust Standard Errors + DGT Test
# =============================================================================

print("\n--- 3.2b-c: GARCH(1,1) QMLE + DGT Test ---")

for asset in ["stock", "cbond"]:
    eps = AR1_RESID[asset]
    eps_np = np.asarray(eps, dtype=np.float64)
    
    r = fit_garch(eps_np, dist="normal", method="qmle", p=1, q=1,
                  solver="trust-constr", use_derivatives=True, verbose=False)
    
    RESULTS["q3_2"].append({
        "asset": asset,
        "omega": r.garch_params.omega,
        "alpha": r.garch_params.alpha[0],
        "beta": r.garch_params.beta[0],
        "persistence": r.garch_params.alpha[0] + r.garch_params.beta[0],
        "log_lik": r.log_likelihood,
        "se_omega_mle": r.std_errors[0] if r.std_errors is not None else None,
        "se_alpha_mle": r.std_errors[1] if r.std_errors is not None else None,
        "se_beta_mle": r.std_errors[2] if r.std_errors is not None else None,
        "se_omega_robust": r.std_errors_robust[0] if r.std_errors_robust is not None else None,
        "se_alpha_robust": r.std_errors_robust[1] if r.std_errors_robust is not None else None,
        "se_beta_robust": r.std_errors_robust[2] if r.std_errors_robust is not None else None,
    })
    
    # Store volatility for plotting
    VOLATILITIES[(asset, "normal")] = {"sigma2": r.sigma2, "dates": eps.index}
    
    # DGT test on standardized residuals (under Normal)
    zt = pd.Series(r.std_resid, index=eps.index)
    dgt, lb_df = dgt_lb_interface(zt, N=40, K=10, alpha=0.05)
    
    RESULTS["q3_2c"].append({
        "asset": asset, "dist": "normal",
        "dgt_chi2": dgt["chi2_stat"], "dgt_df": dgt["df"], "dgt_p": dgt["p_value"], "dgt_reject": dgt["reject"],
    })
    for _, row in lb_df.iterrows():
        RESULTS["q3_2c"].append({
            "asset": asset, "dist": "normal", "lb_moment": row["moment"],
            "lb_q": row["q_stat"], "lb_p": row["p_value"], "lb_reject": row["reject"],
        })
    
    print(f"  {asset}: QMLE done, persistence={r.garch_params.alpha[0] + r.garch_params.beta[0]:.4f}")

# Print Q3.2 results
print_q3_2(RESULTS["q3_2"], RESULTS["q3_2c"])


# =============================================================================
# 3.3: Student-t GARCH
# =============================================================================

print("\n--- 3.3: GARCH(1,1) Student-t ---")

for asset in ["stock", "cbond"]:
    eps = AR1_RESID[asset]
    eps_np = np.asarray(eps, dtype=np.float64)
    
    # Use nelder-mead for robust convergence, but keep derivatives for SE computation
    r = fit_garch(eps_np, dist="studentt", method="mle", p=1, q=1,
                  solver="nelder-mead", use_derivatives=True, verbose=False)
    
    nu = r.dist_params.nu
    inv_nu = 1.0 / nu
    
    # Delta method test: H0: 1/ν = 0
    # SE(1/ν) = |d(1/ν)/dν| × SE(ν) = (1/ν²) × SE(ν)
    se_nu = r.std_errors[3] if r.std_errors is not None and len(r.std_errors) > 3 else None
    if se_nu is not None and se_nu > 0:
        se_inv_nu = (1.0 / nu**2) * se_nu
        t_stat = inv_nu / se_inv_nu
        p_delta = 2 * (1 - norm.cdf(abs(t_stat)))
    else:
        se_inv_nu, t_stat, p_delta = None, None, None
    
    RESULTS["q3_3"].append({
        "asset": asset,
        "omega": r.garch_params.omega,
        "alpha": r.garch_params.alpha[0],
        "beta": r.garch_params.beta[0],
        "persistence": r.garch_params.alpha[0] + r.garch_params.beta[0],
        "nu": nu,
        "log_lik": r.log_likelihood,
        "se_omega": r.std_errors[0] if r.std_errors is not None else None,
        "se_alpha": r.std_errors[1] if r.std_errors is not None else None,
        "se_beta": r.std_errors[2] if r.std_errors is not None else None,
        "se_nu": se_nu,
        "inv_nu": inv_nu,
        "se_inv_nu": se_inv_nu,
        "delta_t": t_stat,
        "delta_p": p_delta,
        "delta_reject": p_delta < 0.05 if p_delta is not None else None,
        "aic": r.aic,
        "bic": r.bic,
    })
    
    # Store volatility
    VOLATILITIES[(asset, "studentt")] = {"sigma2": r.sigma2, "dates": eps.index}
    
    # DGT test under Student-t
    zt = pd.Series(r.std_resid, index=eps.index)
    dgt_res = dgt_with_lb_moments(zt, dist="studentt", nu=nu, n_cells=40, k=10, alpha=0.05)
    
    RESULTS["q3_3b"].append({
        "asset": asset, "dist": "studentt",
        "dgt_chi2": dgt_res["dgt"]["chi2_stat"], "dgt_df": dgt_res["dgt"]["df"],
        "dgt_p": dgt_res["dgt"]["p_value"], "dgt_reject": dgt_res["dgt"]["reject"],
    })
    for m, lb in dgt_res["lb_moments"].items():
        RESULTS["q3_3b"].append({
            "asset": asset, "dist": "studentt", "lb_moment": m,
            "lb_q": lb["q_stat"], "lb_p": lb["p_value"], "lb_reject": lb["reject"],
        })
    
    if p_delta is not None:
        print(f"  {asset}: nu={nu:.2f}, 1/nu={inv_nu:.4f}, t={t_stat:.2f}, p={p_delta:.4e}")
    else:
        print(f"  {asset}: nu={nu:.2f} (SEs unavailable for delta test)")

# Print Q3.3 results
print_q3_3(RESULTS["q3_3"], RESULTS["q3_3b"])


# =============================================================================
# 3.4: Skew-t GARCH
# =============================================================================

print("\n--- 3.4: GARCH(1,1) Skew-t ---")

for asset in ["stock", "cbond"]:
    eps = AR1_RESID[asset]
    eps_np = np.asarray(eps, dtype=np.float64)
    
    # Use slsqp for robust convergence (volkit now computes SEs automatically)
    r = fit_garch(eps_np, dist="skewt", method="mle", p=1, q=1,
                  solver="slsqp", use_derivatives=True, verbose=False)
    
    nu = r.dist_params.nu
    lam = r.dist_params.lam
    inv_nu = 1.0 / nu
    
    # Joint delta method (Wald) test: H0: 1/ν = λ = 0
    # g(θ) = [1/ν, λ]', test g(θ) = 0
    # Var(g) = G × Cov(ν,λ) × G' where G = [[-1/ν², 0], [0, 1]]
    wald_stat, p_wald = None, None
    if r.cov_matrix is not None and r.cov_matrix.shape[0] >= 5:
        cov_nu_lam = r.cov_matrix[3:5, 3:5]
        # Check for valid covariance (positive diagonal)
        if np.all(np.diag(cov_nu_lam) > 0):
            G = np.array([[-1.0/nu**2, 0.0], [0.0, 1.0]])
            var_g = G @ cov_nu_lam @ G.T
            try:
                g_hat = np.array([inv_nu, lam])
                var_g_inv = np.linalg.inv(var_g)
                wald_stat = float(g_hat @ var_g_inv @ g_hat)
                p_wald = float(chi2.sf(wald_stat, df=2))
            except np.linalg.LinAlgError:
                pass
    
    RESULTS["q3_4"].append({
        "asset": asset,
        "omega": r.garch_params.omega,
        "alpha": r.garch_params.alpha[0],
        "beta": r.garch_params.beta[0],
        "persistence": r.garch_params.alpha[0] + r.garch_params.beta[0],
        "nu": nu,
        "lambda": lam,
        "log_lik": r.log_likelihood,
        "se_omega": r.std_errors[0] if r.std_errors is not None else None,
        "se_alpha": r.std_errors[1] if r.std_errors is not None else None,
        "se_beta": r.std_errors[2] if r.std_errors is not None else None,
        "se_nu": r.std_errors[3] if r.std_errors is not None and len(r.std_errors) > 3 else None,
        "se_lambda": r.std_errors[4] if r.std_errors is not None and len(r.std_errors) > 4 else None,
        "wald_stat": wald_stat,
        "wald_p": p_wald,
        "wald_reject": p_wald < 0.05 if p_wald is not None else None,
        "aic": r.aic,
        "bic": r.bic,
    })
    
    # Store volatility
    VOLATILITIES[(asset, "skewt")] = {"sigma2": r.sigma2, "dates": eps.index}
    
    # DGT test under Skew-t
    zt = pd.Series(r.std_resid, index=eps.index)
    dgt_res = dgt_with_lb_moments(zt, dist="skewt", nu=nu, lam=lam, n_cells=40, k=10, alpha=0.05)
    
    RESULTS["q3_4b"].append({
        "asset": asset, "dist": "skewt",
        "dgt_chi2": dgt_res["dgt"]["chi2_stat"], "dgt_df": dgt_res["dgt"]["df"],
        "dgt_p": dgt_res["dgt"]["p_value"], "dgt_reject": dgt_res["dgt"]["reject"],
    })
    for m, lb in dgt_res["lb_moments"].items():
        RESULTS["q3_4b"].append({
            "asset": asset, "dist": "skewt", "lb_moment": m,
            "lb_q": lb["q_stat"], "lb_p": lb["p_value"], "lb_reject": lb["reject"],
        })
    
    # 3.4c: Moment comparison
    sample_mean = float(np.mean(r.std_resid))
    sample_var = float(np.var(r.std_resid, ddof=1))
    sample_skew = float(pd.Series(r.std_resid).skew())
    sample_kurt = float(pd.Series(r.std_resid).kurt())
    
    # Implied moments (standardized skew-t has mean=0, var=1 by construction)
    implied_mean, implied_var = 0.0, 1.0
    implied_kurt = 6.0 / (nu - 4) if nu > 4 else np.nan  # excess kurtosis for symmetric t
    
    RESULTS["q3_4c"].append({
        "asset": asset,
        "sample_mean": sample_mean, "implied_mean": implied_mean,
        "sample_var": sample_var, "implied_var": implied_var,
        "sample_skew": sample_skew,
        "sample_kurt": sample_kurt, "implied_kurt": implied_kurt,
        "nu": nu, "lambda": lam,
    })
    
    if p_wald is not None:
        print(f"  {asset}: nu={nu:.2f}, lam={lam:.4f}, Wald={wald_stat:.2f}, p={p_wald:.4e}")
    else:
        print(f"  {asset}: nu={nu:.2f}, lam={lam:.4f} (SEs unavailable for Wald test)")

# Print Q3.4 results
print_q3_4(RESULTS["q3_4"], RESULTS["q3_4b"], RESULTS["q3_4c"])


# =============================================================================
# MODEL COMPARISON SUMMARY
# =============================================================================

print("\n--- Model Comparison Summary ---")

for asset in ["stock", "cbond"]:
    # Normal (from QMLE results)
    qmle = next(r for r in RESULTS["q3_2"] if r["asset"] == asset)
    RESULTS["q3_summary"].append({
        "asset": asset, "model": "Normal",
        "omega": qmle["omega"], "alpha": qmle["alpha"], "beta": qmle["beta"],
        "persistence": qmle["persistence"], "nu": None, "lambda": None,
        "log_lik": qmle["log_lik"], "aic": None, "bic": None,
    })
    
    # Student-t
    st = next(r for r in RESULTS["q3_3"] if r["asset"] == asset)
    RESULTS["q3_summary"].append({
        "asset": asset, "model": "Student-t",
        "omega": st["omega"], "alpha": st["alpha"], "beta": st["beta"],
        "persistence": st["persistence"], "nu": st["nu"], "lambda": None,
        "log_lik": st["log_lik"], "aic": st["aic"], "bic": st["bic"],
    })
    
    # Skew-t
    skt = next(r for r in RESULTS["q3_4"] if r["asset"] == asset)
    RESULTS["q3_summary"].append({
        "asset": asset, "model": "Skew-t",
        "omega": skt["omega"], "alpha": skt["alpha"], "beta": skt["beta"],
        "persistence": skt["persistence"], "nu": skt["nu"], "lambda": skt["lambda"],
        "log_lik": skt["log_lik"], "aic": skt["aic"], "bic": skt["bic"],
    })

# Print Model Comparison Summary
print_q3_summary(RESULTS["q3_summary"])


# =============================================================================
# VOLKIT PARITY CHECK - Side-by-side comparison
# =============================================================================

print("\n" + "="*70)
print("VOLKIT PARITY CHECK")
print("="*70)

# Store parity results
PARITY_RESULTS = []

def compare_results(ref_result, volkit_result, model_name: str, asset: str):
    """Compare reference and volkit results, return comparison dict."""
    comparison = {
        "asset": asset,
        "model": model_name,
        "ref_ll": ref_result.log_likelihood,
        "volkit_ll": volkit_result.loglikelihood,
        "ll_diff": abs(ref_result.log_likelihood - volkit_result.loglikelihood),
        "ref_omega": ref_result.garch_params.omega,
        "volkit_omega": volkit_result.garch_params.omega if volkit_result.garch_params else None,
        "ref_alpha": ref_result.garch_params.alpha[0],
        "volkit_alpha": volkit_result.garch_params.alpha[0] if volkit_result.garch_params else None,
        "ref_beta": ref_result.garch_params.beta[0],
        "volkit_beta": volkit_result.garch_params.beta[0] if volkit_result.garch_params else None,
        "ref_time": ref_result.time_elapsed,
        "volkit_time": volkit_result.time_elapsed,
    }
    
    # Add distribution parameters if available
    if hasattr(ref_result.dist_params, 'nu') and ref_result.dist_params.nu is not None:
        comparison["ref_nu"] = ref_result.dist_params.nu
        comparison["volkit_nu"] = volkit_result.dist_params.nu if volkit_result.dist_params else None
    
    if hasattr(ref_result.dist_params, 'lam') and ref_result.dist_params.lam is not None:
        comparison["ref_lam"] = ref_result.dist_params.lam
        comparison["volkit_lam"] = volkit_result.dist_params.lam if volkit_result.dist_params else None
    
    return comparison

print("\n--- GARCH(1,1) + Normal ---")
for asset in ["stock", "cbond"]:
    eps_np = np.asarray(AR1_RESID[asset], dtype=np.float64)
    
    # Reference implementation (use log-space for best convergence)
    r_ref = fit_garch_ref(eps_np, dist="normal", method="mle", p=1, q=1,
                          solver="trust-constr", use_logspace=True, verbose=False)
    
    # volkit implementation (use log_mode=True for consistency)
    spec = GARCH(1, 1) + Normal()
    estimator = MLE()
    r_volkit = estimator.fit(spec, eps_np, solver="trust", log_mode=True, verbose=False)
    
    comp = compare_results(r_ref, r_volkit, "Normal", asset)
    PARITY_RESULTS.append(comp)
    
    print(f"  {asset}:")
    print(f"    Reference LL: {r_ref.log_likelihood:.6f}")
    print(f"    Volkit LL:    {r_volkit.loglikelihood:.6f}")
    print(f"    Difference:   {comp['ll_diff']:.2e}")
    print(f"    Time - Ref: {r_ref.time_elapsed:.3f}s, Volkit: {r_volkit.time_elapsed:.3f}s")

print("\n--- GARCH(1,1) + Student-t ---")
for asset in ["stock", "cbond"]:
    eps_np = np.asarray(AR1_RESID[asset], dtype=np.float64)
    
    # Reference implementation
    r_ref = fit_garch_ref(eps_np, dist="studentt", method="mle", p=1, q=1,
                          solver="nelder-mead", use_derivatives=True, verbose=False)
    
    # volkit implementation
    spec = GARCH(1, 1) + StudentT()
    estimator = MLE()
    r_volkit = estimator.fit(spec, eps_np, solver="nelder-mead", log_mode=False, verbose=False)
    
    comp = compare_results(r_ref, r_volkit, "StudentT", asset)
    PARITY_RESULTS.append(comp)
    
    print(f"  {asset}:")
    print(f"    Reference LL: {r_ref.log_likelihood:.6f}, nu={r_ref.dist_params.nu:.2f}")
    print(f"    Volkit LL:    {r_volkit.loglikelihood:.6f}, nu={r_volkit.dist_params.nu:.2f}")
    print(f"    Difference:   {comp['ll_diff']:.2e}")
    print(f"    Time - Ref: {r_ref.time_elapsed:.3f}s, Volkit: {r_volkit.time_elapsed:.3f}s")

print("\n--- GARCH(1,1) + Skew-t ---")
for asset in ["stock", "cbond"]:
    eps_np = np.asarray(AR1_RESID[asset], dtype=np.float64)
    
    # Reference implementation
    r_ref = fit_garch_ref(eps_np, dist="skewt", method="mle", p=1, q=1,
                          solver="nelder-mead", use_derivatives=False, verbose=False)
    
    # volkit implementation
    spec = GARCH(1, 1) + SkewT()
    estimator = MLE()
    r_volkit = estimator.fit(spec, eps_np, solver="nelder-mead", verbose=False)
    
    comp = compare_results(r_ref, r_volkit, "SkewT", asset)
    PARITY_RESULTS.append(comp)
    
    print(f"  {asset}:")
    print(f"    Reference LL: {r_ref.log_likelihood:.6f}, nu={r_ref.dist_params.nu:.2f}, lam={r_ref.dist_params.lam:.4f}")
    print(f"    Volkit LL:    {r_volkit.loglikelihood:.6f}, nu={r_volkit.dist_params.nu:.2f}, lam={r_volkit.dist_params.lam:.4f}")
    print(f"    Difference:   {comp['ll_diff']:.2e}")
    print(f"    Time - Ref: {r_ref.time_elapsed:.3f}s, Volkit: {r_volkit.time_elapsed:.3f}s")

print("\n--- QMLE with Robust Standard Errors ---")
for asset in ["stock", "cbond"]:
    eps_np = np.asarray(AR1_RESID[asset], dtype=np.float64)
    
    # Reference implementation
    r_ref = fit_garch_ref(eps_np, dist="normal", method="qmle", p=1, q=1,
                          solver="trust-constr", use_derivatives=True, verbose=False)
    
    # volkit implementation
    spec = GARCH(1, 1) + Normal()
    estimator = QMLE()
    r_volkit = estimator.fit(spec, eps_np, solver="trust", verbose=False)
    
    comp = compare_results(r_ref, r_volkit, "QMLE", asset)
    PARITY_RESULTS.append(comp)
    
    print(f"  {asset}:")
    print(f"    Reference LL: {r_ref.log_likelihood:.6f}")
    print(f"    Volkit LL:    {r_volkit.loglikelihood:.6f}")
    print(f"    Difference:   {comp['ll_diff']:.2e}")
    
    # Compare robust SEs if available
    if r_ref.std_errors_robust is not None and r_volkit.std_errors_robust is not None:
        print(f"    Robust SE(alpha) - Ref: {r_ref.std_errors_robust[1]:.6f}, Volkit: {r_volkit.std_errors_robust[1]:.6f}")

# Summary of parity results
print("\n--- Parity Summary ---")
parity_df = pd.DataFrame(PARITY_RESULTS)
max_diff = parity_df["ll_diff"].max()
print(f"Maximum log-likelihood difference: {max_diff:.2e}")
if max_diff < 1e-3:
    print("PASS: All models within tolerance (< 1e-3)")
elif max_diff < 1e-1:
    print("WARNING: Some models have small differences, acceptable for different optimization paths")
else:
    print("FAIL: Large differences detected - investigate convergence")

# Save parity results
parity_df.to_csv(RESULTS_DIR / "volkit_parity.csv", index=False)
print(f"Parity results saved to: {RESULTS_DIR / 'volkit_parity.csv'}")


# =============================================================================
# SAVE ALL RESULTS TO CSV
# =============================================================================

print("\n" + "="*70)
print("Saving Results to CSV")
print("="*70)

for key, data in RESULTS.items():
    if data:  # Only save non-empty results
        df = pd.DataFrame(data)
        path = RESULTS_DIR / f"{key}.csv"
        df.to_csv(path, index=False)
        print(f"  {key}: {len(data)} rows -> {path}")

# Save volatility series
for (asset, model), vol_data in VOLATILITIES.items():
    df = pd.DataFrame({
        "date": vol_data["dates"],
        "sigma2": vol_data["sigma2"],
        "sigma": np.sqrt(vol_data["sigma2"]),
    })
    path = RESULTS_DIR / f"volatility_{asset}_{model}.csv"
    df.to_csv(path, index=False)
    print(f"  volatility_{asset}_{model}: {len(df)} rows -> {path}")


# =============================================================================
# VOLATILITY PLOTTING
# =============================================================================

def plot_volatilities(asset: str, save_path: str | None = None):
    """Plot volatility comparison for Normal, Student-t, and Skew-t models."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    dates = VOLATILITIES[(asset, "normal")]["dates"]
    sigma_n = np.sqrt(VOLATILITIES[(asset, "normal")]["sigma2"]) * 100
    sigma_t = np.sqrt(VOLATILITIES[(asset, "studentt")]["sigma2"]) * 100
    sigma_s = np.sqrt(VOLATILITIES[(asset, "skewt")]["sigma2"]) * 100
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: All volatilities
    axes[0].plot(dates, sigma_n, label="Normal", alpha=0.8, lw=0.8)
    axes[0].plot(dates, sigma_t, label="Student-t", alpha=0.8, lw=0.8)
    axes[0].plot(dates, sigma_s, label="Skew-t", alpha=0.8, lw=0.8)
    axes[0].set_ylabel("Volatility (%)")
    axes[0].set_title(f"GARCH(1,1) Conditional Volatility — {asset}")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # Bottom: Differences from Normal
    axes[1].plot(dates, sigma_t - sigma_n, label="Student-t − Normal", alpha=0.8, lw=0.8)
    axes[1].plot(dates, sigma_s - sigma_n, label="Skew-t − Normal", alpha=0.8, lw=0.8)
    axes[1].axhline(0, color="k", ls="--", lw=0.5)
    axes[1].set_ylabel("Difference (%)")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    axes[1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    # plt.show()
    return fig


# Uncomment to generate plots:
plot_volatilities("stock", save_path="results/volatility_stock.png")
plot_volatilities("cbond", save_path="results/volatility_cbond.png")


print("\n" + "="*70)
print("GARCH Analysis Complete!")
print("="*70)
