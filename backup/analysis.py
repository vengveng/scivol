"""
Sections:
  2.1 - Individual asset diagnostics (daily & weekly)
  2.2 - Portfolio diagnostics
  3.1 - GARCH model estimation
  3.2 - MLE vs QMLE
  3.3 - Student-t GARCH
  3.4 - Skew-t GARCH
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import gammaln
from scipy.stats import norm, chi2

from utilities import (
    extreme_observations,
    wait_years,
    jarque_bera,
    acf_and_ljung_box,
    acf_lb_interface,
    summary_stats,
    ar1,
    arch_lm_test,
    dgt_lb_interface,
    dgt_with_lb_moments,
    # Printing functions
    print_q2_1a,
    print_q2_1c,
    print_q2_1d,
    print_q2_2,
    print_q3_1,
    print_q3_1d,
    print_q3_2,
    print_q3_3,
    print_q3_4,
    print_q3_summary,
)
from garch_estimator import fit_garch

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Collector for all results
RESULTS = {
    "q2_1a": [],  # crashes/booms
    "q2_1b": [],  # extreme return probabilities
    "q2_1c": [],  # Jarque-Bera tests
    "q2_1d_acf": [],  # ACF values
    "q2_1d_lb": [],  # Ljung-Box tests
    "q2_2": [],  # portfolio summary stats
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

data_raw = pd.read_csv("data/DATA_HW1.csv", skiprows=1, parse_dates=["DATE"], dayfirst=True)
dff_rate = pd.read_csv("data/DFF.csv", parse_dates=["date"], index_col="date")

RENAME = {
    "DATE": "date",
    "S&PCOMP(RI)": "stock",
    "SPUTBIX(RI)": "gbond",
    "SPUHYBD(RI)": "cbond",
    "WILURET(RI)": "resec",
    "RJEFCRT(TR)": "commo",
    "USBINXB": "usdfx",
}
ASSETS = [a for a in RENAME.values() if a != "date"]

data_raw = data_raw.rename(columns=RENAME).set_index("date")[ASSETS]
rf_ann_pct = dff_rate.reindex(data_raw.index).ffill()

# Daily data
d_data = pd.concat({
    "px": data_raw,
    "sr": data_raw.pct_change(fill_method=None),
    "lr": np.log1p(data_raw.pct_change(fill_method=None)),
}, axis=1)
d_data.columns.names = ["measure", "asset"]

# Weekly data (Friday close)
w_raw = data_raw.resample("W-FRI", label="right", closed="right").last()
w_data = pd.concat({
    "px": w_raw,
    "sr": w_raw.pct_change(fill_method=None),
    "lr": np.log1p(w_raw.pct_change(fill_method=None)),
}, axis=1)
w_data.columns.names = ["measure", "asset"]

DATA = {"d": d_data, "w": w_data}


# =============================================================================
# SECTION 2.1: INDIVIDUAL ASSET DIAGNOSTICS
# =============================================================================

print("\n" + "="*70)
print("SECTION 2.1: Individual Asset Diagnostics")
print("="*70)

# 2.1a: Crashes and booms for S&P 500
for freq in ["d", "w"]:
    lr_stock = DATA[freq]["lr"]["stock"]
    for row in extreme_observations(lr_stock, k=5):
        RESULTS["q2_1a"].append({
            "freq": freq,
            "type": row["type"],
            "rank": row["rank"],
            "date": row["date"].date().isoformat(),
            "return": row["r"],
        })

# 2.1b: Test extreme returns against normality
for freq in ["d", "w"]:
    lr_df = DATA[freq]["lr"]
    for asset in ASSETS:
        lr = lr_df[asset].dropna()
        mu, sigma, T = lr.mean(), lr.std(ddof=0), len(lr)

        # crashes
        for rank, (dt, val) in enumerate(lr.nsmallest(5).items(), 1):
            z = (val - mu) / sigma
            p = norm.cdf(z) # left-tail
            y50 = wait_years(p, freq)
            RESULTS["q2_1b"].append({
                "freq": freq, "asset": asset, "type": "crash", "rank": rank,
                "date": dt.date().isoformat(),
                "return": val,
                "mu": mu, "sigma": sigma, "z": z,
                "p_value": p,
                "wait_years_50": y50,})

        for rank, (dt, val) in enumerate(lr.nlargest(5).items(), 1):
            z = (val - mu) / sigma
            p = norm.sf(z) # right-tail
            y50 = wait_years(p, freq)
            RESULTS["q2_1b"].append({
                "freq": freq, "asset": asset, "type": "boom", "rank": rank,
                "date": dt.date().isoformat(),
                "return": val,
                "mu": mu, "sigma": sigma, "z": z,
                "p_value": p,
                "wait_years_50": y50,})

# 2.1c: Jarque-Bera tests
for freq in ["d", "w"]:
    for asset in ASSETS:
        jb = jarque_bera(DATA[freq]["lr"][asset], 0.05)
        RESULTS["q2_1c"].append({"freq": freq, "asset": asset, **jb})

# 2.1d: ACF and Ljung-Box tests (returns and squared returns)
for freq in ["d", "w"]:
    for asset in ASSETS:
        r = DATA[freq]["lr"][asset].dropna()
        
        # Returns
        acf_rows, lb_row = acf_lb_interface(r, k=10, alpha=0.05)
        for row in acf_rows:
            RESULTS["q2_1d_acf"].append({"freq": freq, "asset": asset, "series": "r", **row})
        RESULTS["q2_1d_lb"].append({"freq": freq, "asset": asset, "series": "r", **lb_row})
        
        # Squared returns
        acf_rows2, lb_row2 = acf_lb_interface(r**2, k=10, alpha=0.05)
        for row in acf_rows2:
            RESULTS["q2_1d_acf"].append({"freq": freq, "asset": asset, "series": "r2", **row})
        RESULTS["q2_1d_lb"].append({"freq": freq, "asset": asset, "series": "r2", **lb_row2})


# Print Section 2.1 results
print_q2_1a(RESULTS["q2_1a"])
print_q2_1c(RESULTS["q2_1c"])
print_q2_1d(RESULTS["q2_1d_lb"])


# =============================================================================
# SECTION 2.2: PORTFOLIO DIAGNOSTICS
# =============================================================================

print("\n" + "="*70)
print("SECTION 2.2: Portfolio Diagnostics")
print("="*70)

for freq in ["d", "w"]:
    sr_df = DATA[freq]["sr"]
    
    # Equal-weight portfolio
    sr_ptf = sr_df.mean(axis=1)
    RESULTS["q2_2"].append({"freq": freq, "type": "portfolio", "asset": "EW6", **summary_stats(sr_ptf)})
    
    # Individual assets
    for asset in ASSETS:
        RESULTS["q2_2"].append({"freq": freq, "type": "asset", "asset": asset, **summary_stats(sr_df[asset])})

# Print Section 2.2 results
print_q2_2(RESULTS["q2_2"])


# =============================================================================
# SECTION 3: GARCH MODELING (Daily data, stocks & bonds only)
# =============================================================================

print("\n" + "="*70)
print("SECTION 3: GARCH Modeling")
print("="*70)

# Prepare data: remove zero-return days, compute log-returns
lr = DATA["d"]["lr"][["stock", "cbond"]]
mask_zero = (lr == 0).any(axis=1)
lr = lr[~mask_zero]
print(f"Removed {mask_zero.sum()} zero-return days")

rf_log = np.log1p(rf_ann_pct["dff"] / 100 / 360)
rf_log = rf_log.reindex(lr.index).ffill()
lr = lr.sub(rf_log, axis=0)

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
        "t_omega_mle": r.garch_params.omega / r.std_errors[0] if r.std_errors is not None else None,
        "t_alpha_mle": r.garch_params.alpha[0] / r.std_errors[1] if r.std_errors is not None else None,
        "t_beta_mle": r.garch_params.beta[0] / r.std_errors[2] if r.std_errors is not None else None,
        "t_omega_robust": r.garch_params.omega / r.std_errors_robust[0] if r.std_errors_robust is not None else None,
        "t_alpha_robust": r.garch_params.alpha[0] / r.std_errors_robust[1] if r.std_errors_robust is not None else None,
        "t_beta_robust": r.garch_params.beta[0] / r.std_errors_robust[2] if r.std_errors_robust is not None else None,
            
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
    
    # Use nelder-mead for robust convergence (no analytical derivatives for skew-t)
    r = fit_garch(eps_np, dist="skewt", method="mle", p=1, q=1,
                  solver="nelder-mead", use_derivatives=False, verbose=False)
    
    nu = r.dist_params.nu
    lam = r.dist_params.lam
    inv_nu = 1.0 / nu
    
    # Joint delta method (Wald) test: H0: 1/ν = λ = 0
    # g(θ) = [1/ν, λ]', test g(θ) = 0
    # Var(g) = G × Cov(ν,λ) × G' where G = [[-1/ν², 0], [0, 1]]
    wald_stat, p_wald = None, None
    if r.cov_matrix is not None and r.cov_matrix.shape[0] >= 5:
        cov_nu_lam = r.cov_matrix[3:5, 3:5]
        print("========================a")
        print(cov_nu_lam)
        print("========================a")
        # Check for valid covariance (positive diagonal)
        if np.all(np.diag(cov_nu_lam) > 0):
            G = np.array([[-1.0/nu**2, 0.0], [0.0, 1.0]])
            var_g = G @ cov_nu_lam @ G.T
            se_invnu = np.sqrt(var_g[0, 0])
            se_lam   = np.sqrt(var_g[1, 1])
            try:
                g_hat = np.array([inv_nu, lam])
                var_g_inv = np.linalg.inv(var_g)
                wald_stat = float(g_hat @ var_g_inv @ g_hat)
                p_wald = float(chi2.sf(wald_stat, df=2))
                # parameter t-stats
                t_invnu = inv_nu / se_invnu
                t_lam = lam / se_lam
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
        "t_invnu": t_invnu,
        "t_lambda": t_lam,
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
    

    # 3.4c: Moment comparison
    x = np.asarray(r.std_resid, dtype=float)

    sample_mean = float(np.mean(x))
    sample_var  = float(np.var(x, ddof=1))
    sample_skew = float(pd.Series(x).skew())
    sample_kurt = float(pd.Series(x).kurt()) + 3.0  # total kurtosis

    implied_mean, implied_var = 0.0, 1.0
    implied_skew = np.nan
    implied_kurt = np.nan
    # (nu>3 for skewness, nu>4 for kurtosis)
    if nu > 2 and (-1.0 < lam < 1.0):
        logc = gammaln((nu + 1.0) / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0)) - gammaln(nu / 2.0)
        c = float(np.exp(logc))

        a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
        m2 = 1.0 + 3.0 * lam**2
        b2 = m2 - a**2

        if b2 > 0:
            b = float(np.sqrt(b2))
            if nu > 3:
                m3 = 16.0 * c * lam * (1.0 + lam**2) * (nu - 2.0)**2 / ((nu - 1.0) * (nu - 3.0))
                mu3 = (m3 - 3.0 * a * m2 + 2.0 * a**3) / (b**3)   # = skewness since Var=1
                implied_skew = float(mu3)

            if nu > 4:
                if nu <= 3:
                    # shouldn't happen given nu>4
                    m3 = 16.0 * c * lam * (1.0 + lam**2) * (nu - 2.0)**2 / ((nu - 1.0) * (nu - 3.0))
                m4 = 3.0 * (nu - 2.0) / (nu - 4.0) * (1.0 + 10.0 * lam**2 + 5.0 * lam**4)
                mu4 = (m4 - 4.0 * a * m3 + 6.0 * a**2 * m2 - 3.0 * a**4) / (b**4)
                implied_kurt = float(mu4)  

    RESULTS["q3_4c"].append({
        "asset": asset,
        "sample_mean": sample_mean, "implied_mean": implied_mean,
        "sample_var": sample_var, "implied_var": implied_var,
        "sample_skew": sample_skew, "implied_skew": implied_skew,
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
# 3.4d: VOLATILITY PLOTTING (commented out to avoid blocking)
# =============================================================================

def plot_volatilities(save_path: str | None = None):
    """Volatility comparison for Normal, Student-t, and Skew-t models."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    
    assets = [("stock", "Stock"), ("cbond", "Corporate Bond")]
    COLORS = {"normal": "orange", "studentt": "red", "skewt": "navy"}
    
    # Create 3x2 subplot grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    
    for col, (asset_key, asset_name) in enumerate(assets):
        dates = VOLATILITIES[(asset_key, "normal")]["dates"]
        sigma_n = (
            np.sqrt(VOLATILITIES[(asset_key, "normal")]["sigma2"]) * 100
        )
        sigma_t = (
            np.sqrt(VOLATILITIES[(asset_key, "studentt")]["sigma2"]) * 100
        )
        sigma_s = (
            np.sqrt(VOLATILITIES[(asset_key, "skewt")]["sigma2"]) * 100
        )
        
        # Row 0: Volatility levels
        axes[0, col].plot(
            dates,
            sigma_n,
            label="Normal",
            alpha=0.9,
            lw=0.8,
            c=COLORS["normal"],
        )
        axes[0, col].plot(
            dates,
            sigma_t,
            label="Student-t",
            alpha=0.9,
            lw=0.8,
            c=COLORS["studentt"],
        )
        axes[0, col].plot(
            dates,
            sigma_s,
            label="Skew-t",
            alpha=0.9,
            lw=0.8,
            c=COLORS["skewt"],
        )
        axes[0, col].set_title(f"GARCH(1,1) Volatility — {asset_name}")
        axes[0, col].legend(loc="upper left")
        axes[0, col].set_ylim(-0.05, 6)
        axes[0, col].grid(True, alpha=0.3)
        
        # Row 1: Differences from Normal
        axes[1, col].plot(
            dates,
            sigma_t - sigma_n,
            label="Student-t − Normal",
            alpha=0.9,
            lw=0.8,
            c=COLORS["studentt"],
        )
        axes[1, col].plot(
            dates,
            sigma_s - sigma_n,
            label="Skew-t − Normal",
            alpha=0.9,
            lw=0.8,
            c=COLORS["skewt"],
        )
        # axes[1, col].plot(
        #     dates,
        #     sigma_s - sigma_t,
        #     label="Skew-t − Student-t",
        #     alpha=0.9,
        #     lw=0.8,
        #     c="green"
        # )
        
        axes[1, col].axhline(0, color="k", ls="--", lw=0.5)
        axes[1, col].legend(loc="upper left")
        axes[1, col].grid(True, alpha=0.3)

        # Row 2: Difference of differences (Skew-t - Student-t)
        diff = (sigma_s - sigma_t) * 100
        axes[2, col].plot(
            dates,
            diff,
            label="Skew-t − Student-t",
            alpha=0.9,
            lw=0.8,
            c="black",
        )
        axes[2, col].axhline(0, color="k", ls="--", lw=0.5)
        axes[2, col].set_xlabel("Date")
        axes[2, col].legend(loc="upper left")
        axes[2, col].set_ylim(-0.4, 0.8)
        axes[2, col].grid(True, alpha=0.3)
        
        # Set date range and ticks
        axes[2, col].set_xlim(dates[0], dates[-1])
        year_ticks = [
            datetime(year, 1, 1) for year in range(2000, 2026, 5)
        ]
        axes[2, col].set_xticks(year_ticks)
        axes[2, col].xaxis.set_major_formatter(
            mdates.DateFormatter("%Y")
        )
    
    # Share y-axis only for top row (volatility plots)
    axes[0, 1].sharey(axes[0, 0])
    axes[2, 1].sharey(axes[2, 0])
    
    # Y-axis labels
    axes[0, 0].set_ylabel("Volatility (%)")
    axes[1, 0].set_ylabel("Difference (%)")
    axes[1, 1].set_ylabel("Difference (%)")
    axes[2, 0].set_ylabel("Difference (bps)")
    # axes[2, 1].set_ylabel("Difference (bps)")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=450, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig

plot_volatilities(save_path="results/volatility_comparison.png")

print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)

