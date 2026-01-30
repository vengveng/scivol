import pandas as pd
import numpy as np
from scipy.stats import norm, chi2, t
from scipy.special import gammaln

metric_sign = {
    "mean": "pos",
    "skew": "pos",
    "max":  "pos",
    "var":  "neg",
    "kurt": "neg",
    "min":  "neg",
}
metrics = list(metric_sign.keys())

def extreme_observations(series: pd.Series, k: int = 5) -> list[dict]:
    """
    Output: {"type": "crash" | "boom", "rank": int, "date": pd.Timestamp, "r": float}
    """
    x = series.dropna()

    crashes = x.nsmallest(k)
    booms   = x.nlargest(k)

    rows: list[dict] = []

    for rank, (dt, val) in enumerate(crashes.items(), start=1):
        rows.append({"type": "crash", "rank": rank, "date": dt, "r": float(val)})

    for rank, (dt, val) in enumerate(booms.items(), start=1):
        rows.append({"type": "boom", "rank": rank, "date": dt, "r": float(val)})

    return rows

def jarque_bera(series, confidence_level: float = 0.05) -> dict:
    """
    Output: {"n": int, "skew": float, "kurt": float, "jb": float, "p_value": float, "reject_5pct": bool}
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size

    m = x.mean()
    y = x - m
    m2 = np.mean(y**2)
    m3 = np.mean(y**3)
    m4 = np.mean(y**4)

    S = m3 / (m2 ** 1.5)
    K = m4 / (m2 ** 2)
    jb = (n / 6.0) * (S**2 + ((K - 3.0) ** 2) / 4.0)
    p = chi2.sf(jb, df=2)  # survival function = 1 - cdf, more stable

    return {
        "n": n, 
        "skew": S, 
        "kurt": K, 
        "jb": jb, 
        "p_value": p, 
        f"reject_{int(confidence_level*100)}pct": p < confidence_level}

def acf_and_ljung_box(series: pd.Series, k: int = 10, alpha: float = 0.05) -> dict:
    """
    Output: {"T": int, "k": int, "alpha": float, "acf": np.ndarray, "ci": float, "q_stat": float, "p_value": float, "reject": bool}
    """

    x = series.dropna()
    T = x.shape[0]
    acf = np.array([x.autocorr(lag=i) for i in range(1, k + 1)], dtype=float)

    # Autocorrelation CI
    z = norm.ppf(1 - alpha / 2)
    ci = z / np.sqrt(T)

    # LBQ stat
    lags = np.arange(1, k + 1)
    q_stat = T * (T + 2) * np.sum((acf ** 2) / (T - lags))

    p_value = chi2.sf(q_stat, df=k)
    reject = p_value < alpha

    return {
        "T": int(T),
        "k": int(k),
        "alpha": float(alpha),
        "acf": acf,
        "ci": float(ci),
        "q_stat": float(q_stat),
        "p_value": float(p_value),
        "reject": bool(reject),
    }

def acf_lb_interface(series: pd.Series, k: int = 10, alpha: float = 0.05) -> tuple[list[dict], dict]:
    """
    Output: rows_acf: [{"lag": int, "acf": float, "ci_low": float, "ci_high": float, "T": int}], 
            row_lb: {"k": int, "T": int, "q_stat": float, "p_value": float, "reject": bool}
    """
    res = acf_and_ljung_box(series, k=k, alpha=alpha)

    rows_acf = []
    for lag, rho in enumerate(res["acf"], start=1):
        rows_acf.append({
            "lag": lag,
            "acf": float(rho),
            "ci_low": float(-res["ci"]),
            "ci_high": float(res["ci"]),
            "T": res["T"],
        })

    row_lb = {
        "k": res["k"],
        "T": res["T"],
        "q_stat": res["q_stat"],
        "p_value": res["p_value"],
        "reject": res["reject"],
    }

    return rows_acf, row_lb

def summary_stats(series: pd.Series) -> dict:
    x = series.dropna()
    return {
        "mean": float(x.mean()),
        "var": float(x.var(ddof=0)),
        "skew": float(x.skew()),
        "kurt": float(x.kurt()),
        "min": float(x.min()),
        "max": float(x.max()),
        "T": int(x.shape[0]),
    }



def portfolio_rank(df: pd.DataFrame, subset: tuple[str, ...]) -> pd.Series:
    
    """Return portfolio overall_rank by freq for a given subset of metrics."""
    portfolio_mask = (df["series"] == "portfolio") | (df["asset"] == "EW6")

    ranks = pd.DataFrame(index=df.index)
    for m in subset:
        asc = (metric_sign[m] == "neg")  # neg metrics: lower is better -> ascending=True
        ranks[m] = df.groupby("freq")[m].rank(ascending=asc, method="min")

    sums = ranks.sum(axis=1)
    overall_rank = sums.groupby(df["freq"]).rank(ascending=True, method="min").astype(int)

    # one portfolio row per freq
    return overall_rank[portfolio_mask].groupby(df.loc[portfolio_mask, "freq"]).first()

def add_ranks(df: pd.DataFrame, pos_cols: list[str], neg_cols: list[str], freq_col: str = "freq") -> pd.DataFrame:
    out = df.copy()
    rank_cols = pos_cols + neg_cols

    for c in rank_cols:
        asc = c in neg_cols  # neg metrics: lower is better
        out[c] = out.groupby(freq_col)[c].rank(ascending=asc, method="min").astype(int)

    sums = out[rank_cols].sum(axis=1)
    out["overall_rank"] = sums.groupby(out[freq_col]).rank(ascending=True, method="min").astype(int)
    return out

def ar1(series: pd.Series) -> dict:
    """
    Output: {"c": float, "phi": float, "se_phi": float, "resid": pd.Series, "T": int}
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

def arch_lm_test(resid: pd.Series, p: int = 4, alpha: float = 0.05) -> dict:
    """
    Output: {"T": int, "p": int, "lm": float, "p_value": float, "reject": bool}
    """
    e = resid.dropna()
    e2 = e**2

    y = e2.iloc[p:]
    X_lags = pd.concat([e2.shift(i) for i in range(1, p + 1)], axis=1).iloc[p:]
    X_lags.columns = [f"lag{i}" for i in range(1, p + 1)]
    X = np.column_stack([np.ones(len(y)), X_lags.to_numpy()])

    # OLS
    y_np = y.to_numpy()
    beta = np.linalg.lstsq(X, y_np, rcond=None)[0]
    u = y_np - X @ beta

    # R^2
    sst = np.sum((y_np - y_np.mean())**2)
    sse = np.sum(u**2)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0

    T = len(y_np)
    lm = T * r2
    p_value = chi2.sf(lm, df=p)
    reject = p_value < alpha

    return {
        "T": int(T), 
        "p": int(p), 
        "lm": float(lm), 
        "p_value": float(p_value), 
        "reject": bool(reject)}


def print_garch_results(result, asset_name, estimator: str = "MLE"):
    print("\n")
    print(f"\nGARCH(1,1) {estimator} results for {asset_name}:")
    print("Estimated parameters:")
    print(f"  omega      = {result.garch_params.omega:.6f}")
    print(f"  alpha      = {result.garch_params.alpha[0]:.6f}")
    print(f"  beta       = {result.garch_params.beta[0]:.6f}")
    print(f"  alpha+beta = {result.garch_params.persistence:.6f}")
    print(f"  ll         = {result.log_likelihood:.2f}")
    print(f"  AIC        = {result.aic:.2f}")
    print(f"  BIC        = {result.bic:.2f}")
    print(f"  Converged: {result.converged}")




# --- helpers: "standardized" Student-t CDF used by your likelihood (variance = 1) ---
def _std_t_cdf(x: np.ndarray, nu: float) -> np.ndarray:
    # If Y ~ t_nu (standard), Var(Y)=nu/(nu-2). For variance-1 version: X = Y*sqrt((nu-2)/nu)
    # So F_X(x) = F_Y(x * sqrt(nu/(nu-2))).
    s = np.sqrt(nu / (nu - 2.0))
    return t.cdf(x * s, df=nu)

def _hansen_skewt_cdf(z: np.ndarray, nu: float, lam: float) -> np.ndarray:
    # Hansen (1994) skewed-t constants, matching your skewt_loglik()
    c = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2.0))
    a = 4.0 * lam * np.exp(c) * (nu - 2.0) / (nu - 1.0)
    b = np.sqrt(1.0 + 3.0 * lam * lam - a * a)

    bz_a = b * z + a
    # piecewise transform back to symmetric standardized-t variable y
    y = np.where(bz_a < 0.0, bz_a / (1.0 + lam), bz_a / (1.0 - lam))
    return _std_t_cdf(y, nu)

def pit_transform(z: pd.Series, dist: str, *, nu: float | None = None, lam: float | None = None,
                  eps: float = 1e-12) -> pd.Series:
    """
    Probability Integral Transform u_t = F(z_t).
    z must be standardized residuals (Series).
    """
    x = z.dropna()
    zz = x.to_numpy(dtype=float)

    dist = dist.lower()
    if dist == "normal":
        u = norm.cdf(zz)
    elif dist in ("studentt", "student-t", "t"):
        if nu is None:
            raise ValueError("nu is required for Student-t PIT")
        u = _std_t_cdf(zz, float(nu))
    elif dist in ("skewt", "skew-t", "skewed-t"):
        if nu is None or lam is None:
            raise ValueError("nu and lam are required for skew-t PIT")
        u = _hansen_skewt_cdf(zz, float(nu), float(lam))
    else:
        raise ValueError(f"Unknown dist '{dist}'")

    # keep strictly inside (0,1) for downstream moments / numeric stability
    u = np.clip(u, eps, 1.0 - eps)
    return pd.Series(u, index=x.index, name="u")

def dgt_test(u: pd.Series, n_cells: int = 40, alpha: float = 0.05) -> dict:
    """
    DGT "cell" test = Pearson chi-square uniformity test on N equal-probability bins of PIT.
    """
    x = u.to_numpy(dtype=float)
    T = int(x.shape[0])

    edges = np.linspace(0.0, 1.0, n_cells + 1)
    counts, _ = np.histogram(x, bins=edges)

    expected = T / n_cells
    chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
    df = n_cells - 1
    p_value = float(chi2.sf(chi2_stat, df=df))
    reject = bool(p_value < alpha)

    return {
        "T": T,
        "N": int(n_cells),
        "alpha": float(alpha),
        "chi2_stat": chi2_stat,
        "df": int(df),
        "p_value": p_value,
        "reject": reject,
        "counts": counts,   # np.ndarray
        "edges": edges,     # np.ndarray
    }

def dgt_with_lb_moments(
    z: pd.Series,
    dist: str,
    *,
    n_cells: int = 40,
    k: int = 10,
    alpha: float = 0.05,
    nu: float | None = None,
    lam: float | None = None,
    powers: tuple[int, ...] = (1, 2, 3, 4),
) -> dict:
    """
    Returns:
      - u_t PIT series
      - DGT cell test
      - Ljung-Box on PIT moments (centered powers of u-0.5): m_p = (u-0.5)^p
        using your existing acf_and_ljung_box(series, k, alpha).
    """
    u = pit_transform(z, dist, nu=nu, lam=lam)
    dgt = dgt_test(u, n_cells=n_cells, alpha=alpha)

    lb = {}
    u_center = u - 0.5
    for p in powers:
        m = pd.Series((u_center.to_numpy(dtype=float) ** p), index=u.index, name=f"m{p}")
        lb[p] = acf_and_ljung_box(m, k=k, alpha=alpha)

    return {
        "dist": dist,
        "nu": None if nu is None else float(nu),
        "lam": None if lam is None else float(lam),
        "u": u,          # pd.Series
        "dgt": dgt,      # dict
        "lb_moments": lb # dict keyed by power -> your LB dict
    }

def dgt_lb_interface(
    std_resid: pd.Series,
    N: int = 40,
    K: int = 10,
    alpha: float = 0.05,
) -> tuple[dict, pd.DataFrame]:
    """
    Inputs
    ------
    std_resid : pd.Series
        Standardized residuals z_t
    N : int
        Number of DGT cells
    K : int
        Ljung-Box lags for PIT moments
    alpha : float
        Significance level

    Output
    ------
    row_dgt : dict
        {"T","N","chi2_stat","df","p_value","reject"}
    out_lb : pd.DataFrame
        columns: ["moment","T","k","q_stat","p_value","reject"]
        for moments of v_t = u_t - 0.5, m=1..4
    """
    z = std_resid.dropna()
    u = pd.Series(norm.cdf(z.to_numpy(dtype=float)), index=z.index, name="u")  # PIT under Normal
    T = int(u.shape[0])

    # --- DGT cell-count chi-square on uniform(0,1) ---
    edges = np.linspace(0.0, 1.0, N + 1)
    counts, _ = np.histogram(np.clip(u.to_numpy(), 0.0, 1.0), bins=edges)
    expected = T / N

    chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
    df = int(N - 1)
    p_value = float(chi2.sf(chi2_stat, df=df))
    reject = bool(p_value < alpha)

    row_dgt = {
        "T": T,
        "N": int(N),
        "chi2_stat": chi2_stat,
        "df": df,
        "p_value": p_value,
        "reject": reject,
    }

    # --- LB on PIT moments (centered) ---
    v = (u - 0.5)
    rows = []
    for m in (1, 2, 3, 4):
        series_m = (v ** m).rename(f"(u-0.5)^{m}")
        _, row_lb = acf_lb_interface(series_m, k=K, alpha=alpha)  # your existing function
        rows.append({
            "moment": m,
            "T": int(row_lb["T"]),
            "k": int(row_lb["k"]),
            "q_stat": float(row_lb["q_stat"]),
            "p_value": float(row_lb["p_value"]),
            "reject": bool(row_lb["reject"]),
        })

    out_lb = pd.DataFrame(rows)
    return row_dgt, out_lb


# =============================================================================
# RESULT PRINTING FUNCTIONS
# =============================================================================

def _fmt(x, decimals=4, width=10):
    """Format a number for display."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return " " * (width - 2) + "NA"
    if isinstance(x, bool):
        return str(x).rjust(width)
    if isinstance(x, int):
        return str(x).rjust(width)
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.{decimals}e}".rjust(width)
    return f"{x:.{decimals}f}".rjust(width)


def print_q2_1a(results: list[dict]):
    """Print Q2.1a: Extreme observations (crashes & booms)."""
    print("\n" + "─" * 60)
    print("Q2.1a: Top 5 Crashes and Booms (S&P 500)")
    print("─" * 60)
    
    for freq in ["d", "w"]:
        freq_label = "Daily" if freq == "d" else "Weekly"
        print(f"\n{freq_label}:")
        
        crashes = [r for r in results if r["freq"] == freq and r["type"] == "crash"]
        booms = [r for r in results if r["freq"] == freq and r["type"] == "boom"]
        
        print(f"  {'Rank':>4}  {'Crash Date':>12}  {'Return':>10}  │  {'Boom Date':>12}  {'Return':>10}")
        print(f"  {'─'*4}  {'─'*12}  {'─'*10}  │  {'─'*12}  {'─'*10}")
        
        for c, b in zip(sorted(crashes, key=lambda x: x["rank"]), 
                        sorted(booms, key=lambda x: x["rank"])):
            print(f"  {c['rank']:4d}  {c['date']:>12}  {c['return']*100:>9.2f}%  │  "
                  f"{b['date']:>12}  {b['return']*100:>9.2f}%")


def print_q2_1c(results: list[dict]):
    """Print Q2.1c: Jarque-Bera tests."""
    print("\n" + "─" * 70)
    print("Q2.1c: Jarque-Bera Tests for Normality")
    print("─" * 70)
    
    for freq in ["d", "w"]:
        freq_label = "Daily" if freq == "d" else "Weekly"
        print(f"\n{freq_label}:")
        print(f"  {'Asset':>8}  {'Skewness':>10}  {'Kurtosis':>10}  {'JB Stat':>12}  {'p-value':>12}  {'Reject':>8}")
        print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*8}")
        
        for r in [r for r in results if r["freq"] == freq]:
            print(f"  {r['asset']:>8}  {r['skew']:>10.4f}  {r['kurt']:>10.4f}  "
                  f"{r['jb']:>12.2f}  {r['p_value']:>12.2e}  {str(r['reject_5pct']):>8}")


def print_q2_1d(results_lb: list[dict]):
    """Print Q2.1d: Ljung-Box tests summary."""
    print("\n" + "─" * 70)
    print("Q2.1d: Ljung-Box Tests (10 lags)")
    print("─" * 70)
    
    for freq in ["d", "w"]:
        freq_label = "Daily" if freq == "d" else "Weekly"
        print(f"\n{freq_label}:")
        print(f"  {'Asset':>8}  {'Series':>8}  {'Q(10)':>12}  {'p-value':>12}  {'Reject':>8}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}")
        
        for r in [r for r in results_lb if r["freq"] == freq]:
            print(f"  {r['asset']:>8}  {r['series']:>8}  {r['q_stat']:>12.2f}  "
                  f"{r['p_value']:>12.4f}  {str(r['reject']):>8}")


def print_q2_2(results: list[dict]):
    """Print Q2.2: Portfolio summary statistics."""
    print("\n" + "─" * 80)
    print("Q2.2: Summary Statistics")
    print("─" * 80)
    
    for freq in ["d", "w"]:
        freq_label = "Daily" if freq == "d" else "Weekly"
        print(f"\n{freq_label}:")
        print(f"  {'Asset':>8}  {'Mean(%)':>10}  {'Std(%)':>10}  {'Skewness':>10}  {'Kurtosis':>10}  {'Sharpe':>10}")
        print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
        
        for r in [r for r in results if r["freq"] == freq]:
            sharpe = r["mean"] / r["var"]**0.5 if r["var"] > 0 else 0
            print(f"  {r['asset']:>8}  {r['mean']*100:>10.4f}  {r['var']**0.5*100:>10.4f}  "
                  f"{r['skew']:>10.4f}  {r['kurt']:>10.4f}  {sharpe:>10.4f}")


def print_q3_1(results_a: list[dict], results_b: list[dict], results_c: list[dict]):
    """Print Q3.1: Pre-GARCH diagnostics."""
    print("\n" + "─" * 80)
    print("Q3.1: Pre-GARCH Diagnostics")
    print("─" * 80)
    
    print("\n(a) Jarque-Bera & Ljung-Box on Log-Returns:")
    print(f"  {'Asset':>8}  {'T':>6}  {'JB Stat':>12}  {'JB p':>12}  {'LB(4)':>12}  {'LB p':>12}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
    for r in results_a:
        print(f"  {r['asset']:>8}  {r['T']:>6}  {r['jb_stat']:>12.2f}  {r['jb_p']:>12.2e}  "
              f"{r['lb_stat']:>12.2f}  {r['lb_p']:>12.4f}")
    
    print("\n(b) AR(1) Estimation:")
    print(f"  {'Asset':>8}  {'c':>12}  {'φ':>12}  {'SE(φ)':>12}  {'t(φ)':>12}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
    for r in results_b:
        print(f"  {r['asset']:>8}  {r['c']:>12.6f}  {r['phi']:>12.6f}  "
              f"{r['se_phi']:>12.6f}  {r['t_phi']:>12.4f}")
    
    print("\n(c) ARCH-LM Test (4 lags) on AR(1) Residuals:")
    print(f"  {'Asset':>8}  {'LM Stat':>12}  {'p-value':>12}  {'Reject H0':>12}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}")
    for r in results_c:
        print(f"  {r['asset']:>8}  {r['lm']:>12.2f}  {r['p_value']:>12.4e}  {str(r['reject']):>12}")


def print_q3_1d(results: list[dict]):
    """Print Q3.1d: Solver comparison."""
    print("\n" + "─" * 100)
    print("Q3.1d: GARCH(1,1) Normal MLE — Solver Comparison")
    print("─" * 100)
    
    for asset in ["stock", "cbond"]:
        print(f"\n{asset.upper()}:")
        print(f"  {'Solver':>14}  {'LogSp':>5}  {'ω':>12}  {'α':>10}  {'β':>10}  "
              f"{'α+β':>8}  {'LogLik':>12}  {'Time':>8}")
        print(f"  {'─'*14}  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*12}  {'─'*8}")
        
        for r in [r for r in results if r["asset"] == asset]:
            logsp = "Yes" if r["use_logspace"] else "No"
            print(f"  {r['solver']:>14}  {logsp:>5}  {r['omega']:>12.2e}  {r['alpha']:>10.4f}  "
                  f"{r['beta']:>10.4f}  {r['persistence']:>8.4f}  {r['log_lik']:>12.2f}  "
                  f"{r['time']:>7.3f}s")


def print_q3_2(results: list[dict], results_dgt: list[dict]):
    """Print Q3.2: QMLE results with robust SEs."""
    print("\n" + "─" * 90)
    print("Q3.2: GARCH(1,1) Normal QMLE with Robust Standard Errors")
    print("─" * 90)
    
    print(f"\n{'Asset':>8}  {'Param':>8}  {'Estimate':>12}  {'SE(MLE)':>12}  {'SE(Robust)':>12}  {'Diff':>12}")
    print(f"{'─'*8}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
    
    for r in results:
        for param in ["omega", "alpha", "beta"]:
            est = r[param]
            se_mle = r[f"se_{param}_mle"]
            se_rob = r[f"se_{param}_robust"]
            diff = (se_rob - se_mle) if se_mle and se_rob else None
            print(f"{r['asset']:>8}  {param:>8}  {_fmt(est, 6, 12)}  {_fmt(se_mle, 6, 12)}  "
                  f"{_fmt(se_rob, 6, 12)}  {_fmt(diff, 6, 12)}")
        print()
    
    # DGT summary
    print("\nDGT Test (Normal distribution):")
    print(f"  {'Asset':>8}  {'χ²':>12}  {'df':>6}  {'p-value':>12}  {'Reject':>8}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*6}  {'─'*12}  {'─'*8}")
    for r in [r for r in results_dgt if "dgt_chi2" in r]:
        print(f"  {r['asset']:>8}  {r['dgt_chi2']:>12.2f}  {r['dgt_df']:>6}  "
              f"{r['dgt_p']:>12.4e}  {str(r['dgt_reject']):>8}")


def print_q3_3(results: list[dict], results_dgt: list[dict]):
    """Print Q3.3: Student-t GARCH results."""
    print("\n" + "─" * 105)
    print("Q3.3: GARCH(1,1) Student-t MLE")
    print("─" * 105)
    
    print(f"\n{'Asset':>8}  {'ω':>12}  {'α':>10}  {'β':>10}  {'ν':>8}  {'1/ν':>8}  {'SE(1/ν)':>10}  {'LogLik':>12}")
    print(f"{'─'*8}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*12}")
    
    for r in results:
        print(f"{r['asset']:>8}  {r['omega']:>12.2e}  {r['alpha']:>10.4f}  {r['beta']:>10.4f}  "
              f"{r['nu']:>8.2f}  {r['inv_nu']:>8.4f}  {_fmt(r['se_inv_nu'], 4, 10)}  {r['log_lik']:>12.2f}")
    
    print("\nDelta Method Test (H₀: 1/ν = 0):")
    print(f"  {'Asset':>8}  {'t-stat':>12}  {'p-value':>12}  {'Reject H₀':>12}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}")
    for r in results:
        print(f"  {r['asset']:>8}  {_fmt(r['delta_t'], 4, 12)}  {_fmt(r['delta_p'], 4, 12)}  "
              f"{str(r['delta_reject']):>12}")
    
    print("\nDGT Test (Student-t distribution):")
    print(f"  {'Asset':>8}  {'χ²':>12}  {'df':>6}  {'p-value':>12}  {'Reject':>8}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*6}  {'─'*12}  {'─'*8}")
    for r in [r for r in results_dgt if "dgt_chi2" in r]:
        print(f"  {r['asset']:>8}  {r['dgt_chi2']:>12.2f}  {r['dgt_df']:>6}  "
              f"{r['dgt_p']:>12.4e}  {str(r['dgt_reject']):>8}")


def print_q3_4(results: list[dict], results_dgt: list[dict], results_moments: list[dict]):
    """Print Q3.4: Skew-t GARCH results."""
    print("\n" + "─" * 115)
    print("Q3.4: GARCH(1,1) Skew-t MLE")
    print("─" * 115)
    
    print(f"\n{'Asset':>8}  {'ω':>12}  {'α':>10}  {'β':>10}  {'ν':>8}  {'λ':>10}  {'LogLik':>12}  {'AIC':>12}  {'BIC':>12}")
    print(f"{'─'*8}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*12}")
    
    for r in results:
        print(f"{r['asset']:>8}  {r['omega']:>12.2e}  {r['alpha']:>10.4f}  {r['beta']:>10.4f}  "
              f"{r['nu']:>8.2f}  {r['lambda']:>10.4f}  {r['log_lik']:>12.2f}  {r['aic']:>12.2f}  {r['bic']:>12.2f}")
    
    print("\nWald Test (H₀: 1/ν = λ = 0):")
    print(f"  {'Asset':>8}  {'Wald':>12}  {'p-value':>12}  {'Reject H₀':>12}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}")
    for r in results:
        print(f"  {r['asset']:>8}  {_fmt(r['wald_stat'], 2, 12)}  {_fmt(r['wald_p'], 4, 12)}  "
              f"{str(r['wald_reject']):>12}")
    
    print("\nDGT Test (Skew-t distribution):")
    print(f"  {'Asset':>8}  {'χ²':>12}  {'df':>6}  {'p-value':>12}  {'Reject':>8}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*6}  {'─'*12}  {'─'*8}")
    for r in [r for r in results_dgt if "dgt_chi2" in r]:
        print(f"  {r['asset']:>8}  {r['dgt_chi2']:>12.2f}  {r['dgt_df']:>6}  "
              f"{r['dgt_p']:>12.4e}  {str(r['dgt_reject']):>8}")
    
    print("\nMoment Comparison (Standardized Residuals):")
    print(f"  {'Asset':>8}  {'Moment':>10}  {'Sample':>12}  {'Implied':>12}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}")
    for r in results_moments:
        print(f"  {r['asset']:>8}  {'Mean':>10}  {r['sample_mean']:>12.4f}  {r['implied_mean']:>12.4f}")
        print(f"  {r['asset']:>8}  {'Variance':>10}  {r['sample_var']:>12.4f}  {r['implied_var']:>12.4f}")
        print(f"  {r['asset']:>8}  {'Skewness':>10}  {r['sample_skew']:>12.4f}  {'—':>12}")
        print(f"  {r['asset']:>8}  {'Ex.Kurt':>10}  {r['sample_kurt']:>12.4f}  {_fmt(r['implied_kurt'], 4, 12)}")
        print()


def print_q3_summary(results: list[dict]):
    """Print Q3 Model Comparison Summary."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)
    
    for asset in ["stock", "cbond"]:
        print(f"\n{asset.upper()}:")
        print(f"  {'Model':>12}  {'ω':>12}  {'α':>10}  {'β':>10}  {'α+β':>8}  "
              f"{'ν':>8}  {'λ':>10}  {'LogLik':>12}")
        print(f"  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*12}")
        
        for r in [r for r in results if r["asset"] == asset]:
            nu_str = f"{r['nu']:.2f}" if r['nu'] else "—"
            lam_str = f"{r['lambda']:.4f}" if r['lambda'] else "—"
            print(f"  {r['model']:>12}  {r['omega']:>12.2e}  {r['alpha']:>10.4f}  "
                  f"{r['beta']:>10.4f}  {r['persistence']:>8.4f}  {nu_str:>8}  "
                  f"{lam_str:>10}  {r['log_lik']:>12.2f}")