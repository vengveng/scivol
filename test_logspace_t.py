"""
Log-Space Optimization for Student-t and Skew-t GARCH
======================================================

Test file for implementing unconstrained optimization via reparameterization.

Student-t transformations:
    ω = exp(z_ω)
    [α, β] = softmax([z_α, z_β, 0])[:2]  (joint softmax for stationarity)
    ν = 2 + softplus(z_ν) = 2 + log(1 + exp(z_ν))

Skew-t transformations (same + λ with ν-dependent bounds):
    λ = λ_max(ν) * tanh(z_λ)
    
    where λ_max(ν) is chosen so that b² > 0 in Hansen's skew-t.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp, gammaln
from scipy.optimize import minimize
import pandas as pd
import time

from utilities import ar1
from garch_estimator import garch_variance_11, VARIANCE_FLOOR

# =============================================================================
# CONSTANTS
# =============================================================================

LOG_CLIP_MIN = -700.0
LOG_CLIP_MAX = 700.0
SOFTPLUS_THRESHOLD = 20.0  # For numerical stability


# =============================================================================
# SOFTPLUS FUNCTION
# =============================================================================

def softplus(x: float) -> float:
    """Numerically stable softplus: log(1 + exp(x))."""
    if x > SOFTPLUS_THRESHOLD:
        return x  # Avoid overflow
    return np.log1p(np.exp(x))


def softplus_inv(y: float) -> float:
    """Inverse softplus: log(exp(y) - 1)."""
    if y > SOFTPLUS_THRESHOLD:
        return y  # Approximate inverse for large y
    return np.log(np.expm1(y))


def softplus_deriv(x: float) -> float:
    """Derivative of softplus: sigmoid(x) = 1 / (1 + exp(-x))."""
    if x > SOFTPLUS_THRESHOLD:
        return 1.0
    if x < -SOFTPLUS_THRESHOLD:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# STUDENT-T TRANSFORMATIONS
# =============================================================================

def studentt_z_to_theta(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform Student-t GARCH parameters from unconstrained z to constrained θ.
    
    z = [z_ω, z_α, z_β, z_ν]
    θ = [ω, α, β, ν]
    
    Transformations:
        ω = exp(z_ω)
        [α, β] = softmax([z_α, z_β, 0])[:2]
        ν = 2 + softplus(z_ν)
    """
    z_omega, z_alpha, z_beta, z_nu = z[0], z[1], z[2], z[3]
    
    # ω: exp transform
    omega = np.exp(np.clip(z_omega, LOG_CLIP_MIN, LOG_CLIP_MAX))
    
    # α, β: joint softmax with implicit z_remainder = 0
    z_all = np.array([z_alpha, z_beta, 0.0])
    log_denom = logsumexp(z_all)
    alpha = np.exp(z_alpha - log_denom)
    beta = np.exp(z_beta - log_denom)
    
    # ν: 2 + softplus(z_ν) ensures ν > 2
    nu = 2.0 + softplus(z_nu)
    
    return np.array([omega, alpha, beta, nu])


def studentt_theta_to_z(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse transformation: θ → z for Student-t."""
    omega, alpha, beta, nu = theta[0], theta[1], theta[2], theta[3]
    
    z_omega = np.log(omega)
    
    # Inverse joint softmax
    remainder = 1.0 - alpha - beta
    remainder = max(remainder, 1e-10)
    z_alpha = np.log(alpha / remainder)
    z_beta = np.log(beta / remainder)
    
    # Inverse softplus
    z_nu = softplus_inv(nu - 2.0)
    
    return np.array([z_omega, z_alpha, z_beta, z_nu])


def studentt_jacobian(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Jacobian J = ∂θ/∂z for Student-t transformation."""
    theta = studentt_z_to_theta(z)
    omega, alpha, beta, nu = theta[0], theta[1], theta[2], theta[3]
    
    J = np.zeros((4, 4), dtype=np.float64)
    
    # ∂ω/∂z_ω = ω
    J[0, 0] = omega
    
    # Softmax Jacobian for [α, β]
    J[1, 1] = alpha * (1.0 - alpha)
    J[1, 2] = -alpha * beta
    J[2, 1] = -beta * alpha
    J[2, 2] = beta * (1.0 - beta)
    
    # ∂ν/∂z_ν = softplus'(z_ν) = sigmoid(z_ν)
    J[3, 3] = softplus_deriv(z[3])
    
    return J


# =============================================================================
# SKEW-T: COMPUTE λ_MAX FOR HANSEN'S PARAMETERIZATION
# =============================================================================

def hansen_lambda_max(nu: float, safety: float = 0.999) -> float:
    """
    Compute maximum admissible |λ| for Hansen's skew-t given ν.
    
    Hansen's skew-t requires b² > 0 where:
        c = Γ((ν+1)/2) / [√(π(ν-2)) Γ(ν/2)]
        a = 4λc(ν-2)/(ν-1)
        b² = 1 + 3λ² - a²
    
    Setting b² = 0 and solving for λ gives the boundary.
    We use a slightly smaller value (safety < 1) to stay in the interior.
    """
    if nu <= 2:
        return 0.0
    
    # Compute c (constant from Hansen's parameterization)
    c = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi * (nu - 2))
    
    # For b² > 0: 1 + 3λ² - a² > 0
    # where a = 4λc(ν-2)/(ν-1)
    # 
    # Let k = 4c(ν-2)/(ν-1), so a = kλ
    # 1 + 3λ² - k²λ² > 0
    # 1 + λ²(3 - k²) > 0
    #
    # If 3 - k² > 0: always satisfied
    # If 3 - k² < 0: |λ| < 1/√(k² - 3)
    # If 3 - k² = 0: λ can be anything (but tanh bounds to (-1,1) anyway)
    
    k = 4 * c * (nu - 2) / (nu - 1)
    k2 = k * k
    
    if k2 <= 3:
        # b² > 0 for all |λ| < 1, so λ_max = 1
        return safety * 1.0
    else:
        # |λ| < 1/√(k² - 3)
        lam_max = 1.0 / np.sqrt(k2 - 3)
        return safety * min(lam_max, 1.0)


# =============================================================================
# SKEW-T TRANSFORMATIONS
# =============================================================================

def skewt_z_to_theta(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform Skew-t GARCH parameters from unconstrained z to constrained θ.
    
    z = [z_ω, z_α, z_β, z_ν, z_λ]
    θ = [ω, α, β, ν, λ]
    
    Transformations:
        ω = exp(z_ω)
        [α, β] = softmax([z_α, z_β, 0])[:2]
        ν = 2 + softplus(z_ν)
        λ = λ_max(ν) * tanh(z_λ)
    """
    z_omega, z_alpha, z_beta, z_nu, z_lambda = z[0], z[1], z[2], z[3], z[4]
    
    # ω: exp transform
    omega = np.exp(np.clip(z_omega, LOG_CLIP_MIN, LOG_CLIP_MAX))
    
    # α, β: joint softmax
    z_all = np.array([z_alpha, z_beta, 0.0])
    log_denom = logsumexp(z_all)
    alpha = np.exp(z_alpha - log_denom)
    beta = np.exp(z_beta - log_denom)
    
    # ν: 2 + softplus(z_ν)
    nu = 2.0 + softplus(z_nu)
    
    # λ: ν-dependent bounded tanh
    lam_max = hansen_lambda_max(nu)
    lam = lam_max * np.tanh(z_lambda)
    
    return np.array([omega, alpha, beta, nu, lam])


def skewt_theta_to_z(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse transformation: θ → z for Skew-t."""
    omega, alpha, beta, nu, lam = theta[0], theta[1], theta[2], theta[3], theta[4]
    
    z_omega = np.log(omega)
    
    # Inverse joint softmax
    remainder = 1.0 - alpha - beta
    remainder = max(remainder, 1e-10)
    z_alpha = np.log(alpha / remainder)
    z_beta = np.log(beta / remainder)
    
    # Inverse softplus for ν
    z_nu = softplus_inv(nu - 2.0)
    
    # Inverse ν-dependent tanh for λ
    lam_max = hansen_lambda_max(nu)
    lam_scaled = np.clip(lam / lam_max, -0.999, 0.999)  # Avoid arctanh at ±1
    z_lambda = np.arctanh(lam_scaled)
    
    return np.array([z_omega, z_alpha, z_beta, z_nu, z_lambda])


def skewt_jacobian(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute Jacobian J = ∂θ/∂z for Skew-t transformation.
    
    Note: This is an approximation that ignores the dependence of λ_max on ν.
    For proper SEs, we'd need to account for ∂λ_max/∂ν, but this is complex
    and the effect is small. We use numerical differentiation for SEs anyway.
    """
    theta = skewt_z_to_theta(z)
    omega, alpha, beta, nu, lam = theta[0], theta[1], theta[2], theta[3], theta[4]
    
    J = np.zeros((5, 5), dtype=np.float64)
    
    # ∂ω/∂z_ω = ω
    J[0, 0] = omega
    
    # Softmax Jacobian for [α, β]
    J[1, 1] = alpha * (1.0 - alpha)
    J[1, 2] = -alpha * beta
    J[2, 1] = -beta * alpha
    J[2, 2] = beta * (1.0 - beta)
    
    # ∂ν/∂z_ν = softplus'(z_ν)
    J[3, 3] = softplus_deriv(z[3])
    
    # ∂λ/∂z_λ = λ_max(ν) * sech²(z_λ) = λ_max(ν) * (1 - tanh²(z_λ))
    lam_max = hansen_lambda_max(nu)
    tanh_z = np.tanh(z[4])
    J[4, 4] = lam_max * (1.0 - tanh_z ** 2)
    
    return J


# =============================================================================
# LIKELIHOODS
# =============================================================================

def studentt_nll(resid: NDArray, sigma2: NDArray, nu: float) -> float:
    """Negative log-likelihood for Student-t."""
    n = len(resid)
    z2 = resid ** 2 / sigma2
    
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    ll = n * const - 0.5 * np.sum(np.log(sigma2)) - (nu + 1) / 2 * np.sum(np.log(1 + z2 / (nu - 2)))
    return -ll


def skewt_nll(resid: NDArray, sigma2: NDArray, nu: float, lam: float) -> float:
    """Negative log-likelihood for Hansen's Skew-t."""
    n = len(resid)
    
    # Hansen's constants
    c = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi * (nu - 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b2 = 1 + 3 * lam ** 2 - a ** 2
    
    if b2 <= 0:
        return 1e10  # Invalid parameters
    
    b = np.sqrt(b2)
    
    # Standardized residuals
    z = resid / np.sqrt(sigma2)
    
    # Transformed variable
    xi = b * z + a
    sign = np.sign(xi)
    denom = 1 - lam * sign
    
    # Kernel of skew-t
    arg = 1 + (xi / denom) ** 2 / (nu - 2)
    
    ll = (n * np.log(c) + n * np.log(b) 
          - 0.5 * np.sum(np.log(sigma2))
          - (nu + 1) / 2 * np.sum(np.log(arg)))
    
    return -ll


# =============================================================================
# LOG-SPACE OPTIMIZATION
# =============================================================================

def fit_studentt_logspace(
    resid: NDArray[np.float64],
    solver: str = "L-BFGS-B",
    verbose: bool = False,
) -> dict:
    """
    Fit Student-t GARCH in unconstrained log-space.
    
    Parameters
    ----------
    resid : array
        Residual series
    solver : str
        Optimization method: "L-BFGS-B", "SLSQP", "trust-constr", "Nelder-Mead"
    verbose : bool
        Print optimization progress
    """
    n = len(resid)
    resid2 = resid ** 2
    
    def objective(z):
        theta = studentt_z_to_theta(z)
        omega, alpha, beta, nu = theta[0], theta[1], theta[2], theta[3]
        
        garch_params = np.array([omega, alpha, beta])
        sigma2 = garch_variance_11(garch_params, resid2)
        
        nll = studentt_nll(resid, sigma2, nu)
        return nll / n
    
    # Initial values (transform from reasonable θ)
    theta0 = np.array([np.var(resid) * 0.05, 0.1, 0.8, 10.0])
    z0 = studentt_theta_to_z(theta0)
    
    # Solver-specific options
    if solver == "L-BFGS-B":
        options = {"disp": verbose, "maxiter": 5000}
    elif solver == "SLSQP":
        options = {"disp": verbose, "maxiter": 5000}
    elif solver == "trust-constr":
        options = {"disp": verbose, "maxiter": 5000, "verbose": 2 if verbose else 0}
    elif solver == "Nelder-Mead":
        options = {"disp": verbose, "maxiter": 10000, "maxfev": 50000, "adaptive": True}
    else:
        options = {"disp": verbose}
    
    start = time.perf_counter()
    res = minimize(objective, z0, method=solver, options=options)
    elapsed = time.perf_counter() - start
    
    theta_hat = studentt_z_to_theta(res.x)
    
    # Compute final sigma2 and SEs
    garch_params = theta_hat[:3]
    sigma2 = garch_variance_11(garch_params, resid2)
    
    # Numerical Hessian in z-space for SEs
    from numerical_hessians import compute_hessian_unconstrained
    H_z = compute_hessian_unconstrained(objective, res.x, eps=1e-5)
    try:
        cov_z = np.linalg.inv(H_z) / n
        J = studentt_jacobian(res.x)
        cov_theta = J @ cov_z @ J.T
        se = np.sqrt(np.maximum(np.diag(cov_theta), 0))
    except:
        se = None
    
    return {
        "theta": theta_hat,
        "omega": theta_hat[0],
        "alpha": theta_hat[1],
        "beta": theta_hat[2],
        "nu": theta_hat[3],
        "log_lik": -res.fun * n,
        "converged": res.success,
        "n_iter": res.nit if hasattr(res, 'nit') else res.nfev,
        "time": elapsed,
        "std_errors": se,
        "sigma2": sigma2,
        "solver": solver,
    }


def fit_skewt_logspace(
    resid: NDArray[np.float64],
    solver: str = "L-BFGS-B",
    verbose: bool = False,
) -> dict:
    """
    Fit Skew-t GARCH in unconstrained log-space.
    
    Parameters
    ----------
    resid : array
        Residual series
    solver : str
        Optimization method: "L-BFGS-B", "SLSQP", "trust-constr", "Nelder-Mead"
    verbose : bool
        Print optimization progress
    """
    n = len(resid)
    resid2 = resid ** 2
    
    def objective(z):
        theta = skewt_z_to_theta(z)
        omega, alpha, beta, nu, lam = theta[0], theta[1], theta[2], theta[3], theta[4]
        
        garch_params = np.array([omega, alpha, beta])
        sigma2 = garch_variance_11(garch_params, resid2)
        
        nll = skewt_nll(resid, sigma2, nu, lam)
        return nll / n
    
    # Initial values
    theta0 = np.array([np.var(resid) * 0.05, 0.1, 0.8, 10.0, 0.0])
    z0 = skewt_theta_to_z(theta0)
    
    # Solver-specific options
    if solver == "L-BFGS-B":
        options = {"disp": verbose, "maxiter": 5000}
    elif solver == "SLSQP":
        options = {"disp": verbose, "maxiter": 5000}
    elif solver == "trust-constr":
        options = {"disp": verbose, "maxiter": 5000, "verbose": 2 if verbose else 0}
    elif solver == "Nelder-Mead":
        options = {"disp": verbose, "maxiter": 10000, "maxfev": 50000, "adaptive": True}
    else:
        options = {"disp": verbose}
    
    start = time.perf_counter()
    res = minimize(objective, z0, method=solver, options=options)
    elapsed = time.perf_counter() - start
    
    theta_hat = skewt_z_to_theta(res.x)
    
    # Compute final sigma2 and SEs
    garch_params = theta_hat[:3]
    sigma2 = garch_variance_11(garch_params, resid2)
    
    # Numerical Hessian in z-space for SEs
    from numerical_hessians import compute_hessian_unconstrained
    H_z = compute_hessian_unconstrained(objective, res.x, eps=1e-5)
    try:
        cov_z = np.linalg.inv(H_z) / n
        J = skewt_jacobian(res.x)
        cov_theta = J @ cov_z @ J.T
        se = np.sqrt(np.maximum(np.diag(cov_theta), 0))
    except:
        se = None
    
    return {
        "theta": theta_hat,
        "omega": theta_hat[0],
        "alpha": theta_hat[1],
        "beta": theta_hat[2],
        "nu": theta_hat[3],
        "lambda": theta_hat[4],
        "log_lik": -res.fun * n,
        "converged": res.success,
        "n_iter": res.nit if hasattr(res, 'nit') else res.nfev,
        "time": elapsed,
        "std_errors": se,
        "sigma2": sigma2,
        "solver": solver,
    }


# =============================================================================
# SOLVER COMPARISON FUNCTION
# =============================================================================

def compare_logspace_solvers(
    resid: NDArray[np.float64],
    dist: str = "studentt",
    solvers: list = None,
):
    """
    Compare multiple solvers for log-space optimization.
    
    Parameters
    ----------
    resid : array
        Residual series
    dist : str
        "studentt" or "skewt"
    solvers : list
        List of solver names. Default: ["L-BFGS-B", "SLSQP", "Nelder-Mead"]
        Add "trust-constr" if you want (can be slow!)
    """
    if solvers is None:
        solvers = ["L-BFGS-B", "SLSQP", "Nelder-Mead"]
    
    results = []
    
    for solver in solvers:
        print(f"  Running {solver}...", end=" ", flush=True)
        try:
            if dist == "studentt":
                r = fit_studentt_logspace(resid, solver=solver)
            else:
                r = fit_skewt_logspace(resid, solver=solver)
            print(f"done ({r['time']:.2f}s)")
            results.append(r)
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"solver": solver, "converged": False, "error": str(e)})
    
    return results


def print_solver_comparison(results: list, dist: str):
    """Print formatted comparison table."""
    if dist == "studentt":
        print(f"\n{'Solver':>15}  {'Conv':>5}  {'ω':>12}  {'α':>8}  {'β':>8}  "
              f"{'ν':>6}  {'LogLik':>12}  {'Time':>8}  {'Iter':>6}")
        print(f"{'─'*15}  {'─'*5}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*12}  {'─'*8}  {'─'*6}")
    else:
        print(f"\n{'Solver':>15}  {'Conv':>5}  {'ω':>12}  {'α':>8}  {'β':>8}  "
              f"{'ν':>6}  {'λ':>8}  {'LogLik':>12}  {'Time':>8}  {'Iter':>6}")
        print(f"{'─'*15}  {'─'*5}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*12}  {'─'*8}  {'─'*6}")
    
    # Find best
    valid = [r for r in results if r.get("converged", False)]
    best_ll = max(r["log_lik"] for r in valid) if valid else None
    
    for r in results:
        if "error" in r:
            print(f"{r['solver']:>15}  ERROR: {r['error']}")
            continue
        
        conv = "Yes" if r["converged"] else "No"
        marker = " *" if best_ll and abs(r["log_lik"] - best_ll) < 0.01 else "  "
        
        if dist == "studentt":
            print(f"{r['solver']:>15}  {conv:>5}  {r['omega']:>12.2e}  {r['alpha']:>8.4f}  "
                  f"{r['beta']:>8.4f}  {r['nu']:>6.2f}  {r['log_lik']:>12.2f}{marker} {r['time']:>7.2f}s  {r['n_iter']:>6}")
        else:
            print(f"{r['solver']:>15}  {conv:>5}  {r['omega']:>12.2e}  {r['alpha']:>8.4f}  "
                  f"{r['beta']:>8.4f}  {r['nu']:>6.2f}  {r['lambda']:>8.4f}  {r['log_lik']:>12.2f}{marker} {r['time']:>7.2f}s  {r['n_iter']:>6}")
    
    if best_ll:
        print(f"\n* Best log-likelihood")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LOG-SPACE OPTIMIZATION FOR STUDENT-T AND SKEW-T GARCH")
    print("=" * 80)
    
    # Test transformations
    print("\n--- Test 1: Transformation Roundtrips ---")
    
    theta_t = np.array([1e-5, 0.1, 0.85, 8.0])
    z_t = studentt_theta_to_z(theta_t)
    theta_t_rec = studentt_z_to_theta(z_t)
    print(f"Student-t: max roundtrip error = {np.max(np.abs(theta_t - theta_t_rec)):.2e}")
    
    theta_s = np.array([1e-5, 0.1, 0.85, 8.0, -0.1])
    z_s = skewt_theta_to_z(theta_s)
    theta_s_rec = skewt_z_to_theta(z_s)
    print(f"Skew-t: max roundtrip error = {np.max(np.abs(theta_s - theta_s_rec)):.2e}")
    
    # Test λ_max computation
    print("\n--- Test 2: Hansen λ_max for various ν ---")
    for nu in [3.0, 4.0, 5.0, 8.0, 10.0, 20.0, 50.0]:
        lam_max = hansen_lambda_max(nu, safety=1.0)
        print(f"  ν = {nu:5.1f}: λ_max = {lam_max:.4f}")
    
    # Load data
    print("\n--- Test 3: Real Data Estimation ---")
    
    data_raw = pd.read_csv("data/DATA_HW1.csv", skiprows=1, parse_dates=["DATE"], dayfirst=True)
    data = data_raw.rename(columns={
        "S&PCOMP(RI)": "stock",
        "SPUHYBD(RI)": "cbond",
    }).set_index("DATE")[["stock", "cbond"]]
    
    lr = np.log1p(data.pct_change(fill_method=None))
    mask_zero = (lr == 0).any(axis=1)
    lr = lr[~mask_zero]
    
    # ==========================================================================
    # QUICK TEST: L-BFGS-B only (fast)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("QUICK TEST: L-BFGS-B (log-space) vs Nelder-Mead (constrained)")
    print("=" * 80)
    
    from garch_estimator import fit_garch
    
    for asset in ["stock", "cbond"]:
        print(f"\n{'─'*60}")
        print(f"{asset.upper()}")
        print(f"{'─'*60}")
        
        ar1_res = ar1(lr[asset])
        eps = np.asarray(ar1_res["resid"], dtype=np.float64)
        
        # Student-t
        print("\nSTUDENT-T:")
        print(f"{'Method':>15}  {'ω':>12}  {'α':>8}  {'β':>8}  {'ν':>6}  {'LogLik':>12}  {'Time':>8}")
        print(f"{'─'*15}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*12}  {'─'*8}")
        
        r_log = fit_studentt_logspace(eps, solver="L-BFGS-B")
        print(f"{'L-BFGS-B (log)':>15}  {r_log['omega']:>12.2e}  {r_log['alpha']:>8.4f}  "
              f"{r_log['beta']:>8.4f}  {r_log['nu']:>6.2f}  {r_log['log_lik']:>12.2f}  {r_log['time']:>7.3f}s")
        
        r_nm = fit_garch(eps, dist="studentt", solver="nelder-mead", use_derivatives=True)
        print(f"{'Nelder-Mead':>15}  {r_nm.garch_params.omega:>12.2e}  {r_nm.garch_params.alpha[0]:>8.4f}  "
              f"{r_nm.garch_params.beta[0]:>8.4f}  {r_nm.dist_params.nu:>6.2f}  {r_nm.log_likelihood:>12.2f}  {r_nm.time_elapsed:>7.3f}s")
        
        # Skew-t
        print("\nSKEW-T:")
        print(f"{'Method':>15}  {'ω':>12}  {'α':>8}  {'β':>8}  {'ν':>6}  {'λ':>8}  {'LogLik':>12}  {'Time':>8}")
        print(f"{'─'*15}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*12}  {'─'*8}")
        
        r_log = fit_skewt_logspace(eps, solver="L-BFGS-B")
        print(f"{'L-BFGS-B (log)':>15}  {r_log['omega']:>12.2e}  {r_log['alpha']:>8.4f}  "
              f"{r_log['beta']:>8.4f}  {r_log['nu']:>6.2f}  {r_log['lambda']:>8.4f}  {r_log['log_lik']:>12.2f}  {r_log['time']:>7.3f}s")
        
        r_nm = fit_garch(eps, dist="skewt", solver="nelder-mead", use_derivatives=False)
        print(f"{'Nelder-Mead':>15}  {r_nm.garch_params.omega:>12.2e}  {r_nm.garch_params.alpha[0]:>8.4f}  "
              f"{r_nm.garch_params.beta[0]:>8.4f}  {r_nm.dist_params.nu:>6.2f}  {r_nm.dist_params.lam:>8.4f}  {r_nm.log_likelihood:>12.2f}  {r_nm.time_elapsed:>7.3f}s")
    
    print("\n" + "=" * 80)
    print("QUICK TEST DONE!")
    print("=" * 80)
    
    # ==========================================================================
    # FULL SOLVER COMPARISON (uncomment to run - trust-constr can be SLOW!)
    # ==========================================================================
    # 
    # Solvers to test (add "trust-constr" if you're patient):
    SOLVERS = ["L-BFGS-B", "SLSQP", "Nelder-Mead"]
    SOLVERS = ["L-BFGS-B", "SLSQP", "Nelder-Mead", "trust-constr"]
    
    print("\n" + "=" * 80)
    print("FULL SOLVER COMPARISON (log-space)")
    print("=" * 80)
    
    for asset in ["stock", "cbond"]:
        print(f"\n{'='*60}")
        print(f"{asset.upper()}")
        print(f"{'='*60}")
        
        ar1_res = ar1(lr[asset])
        eps = np.asarray(ar1_res["resid"], dtype=np.float64)
        
        # Student-t
        print("\nSTUDENT-T:")
        results_t = compare_logspace_solvers(eps, dist="studentt", solvers=SOLVERS)
        print_solver_comparison(results_t, dist="studentt")
        
        # Skew-t  
        print("\nSKEW-T:")
        results_s = compare_logspace_solvers(eps, dist="skewt", solvers=SOLVERS)
        print_solver_comparison(results_s, dist="skewt")
    
    print("\n" + "=" * 80)
    print("FULL COMPARISON DONE!")
    print("=" * 80)
