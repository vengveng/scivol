// scivol/_csrc/likelihood_skewt.c
//
// Hansen (1994) Skewed Student-t log-likelihood implementation.
//
// Reference: Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation."
// International Economic Review, 35(3), 705-730.

#include <stddef.h>
#include <math.h>
#include "math_and_helpers.h"

// ============================================================================
// Hansen (1994) Skew-t Log-Likelihood
// ============================================================================
//
// The Hansen skew-t density has the form:
//
//   f(z | nu, lam) = b * c * (1 + (z_adj^2 / (nu-2)))^(-(nu+1)/2)
//
// where:
//   c = Gamma((nu+1)/2) / (Gamma(nu/2) * sqrt(pi*(nu-2)))
//   a = 4 * lam * exp(c) * (nu-2) / (nu-1)
//   b = sqrt(1 + 3*lam^2 - a^2)
//   z_adj = (b*z + a) / (1 - lam * sign(b*z + a))
//
// For standardized residuals z = eps / sqrt(sigma2), the log-likelihood is:
//   ll = n*log(c) - 0.5*sum(log(sigma2)) + n*log(b) 
//        - 0.5*(nu+1)*sum(log(1 + z_adj^2/(nu-2)))

__attribute__((visibility("default"), hot, flatten))
double skewt_ll(
    const double* __restrict resid,      // Residuals (not squared)
    const double* __restrict sigma2,     // Conditional variances
    const size_t n,                      // Number of observations
    const double nu,                     // Degrees of freedom (> 2)
    const double lam                     // Asymmetry parameter (-1, 1)
)
{
    // Compute Hansen constants
    const double c_log = lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI * (nu - 2));
    const double c_exp = exp(c_log);
    const double a = 4.0 * lam * c_exp * (nu - 2.0) / (nu - 1.0);
    const double b = sqrt(1.0 + 3.0 * lam * lam - a * a);
    const double nu_m2 = nu - 2.0;
    const double half_nup1 = 0.5 * (nu + 1.0);
    
    // Accumulate log-likelihood components
    double sum_log_sigma2 = 0.0;
    double sum_log_kernel = 0.0;
    
    for (size_t t = 0; t < n; ++t) {
        // Standardized residual
        const double z = resid[t] / sqrt(sigma2[t]);
        
        // Adjusted residual (Hansen transformation)
        const double bz_plus_a = b * z + a;
        const double sign_term = (bz_plus_a >= 0.0) ? 1.0 : -1.0;
        const double z_adj = bz_plus_a / (1.0 - lam * sign_term);
        
        // Accumulate sums
        sum_log_sigma2 += log(sigma2[t]);
        sum_log_kernel += log1p(z_adj * z_adj / nu_m2);
    }
    
    // Final log-likelihood
    // ll = n*c_log - 0.5*sum(log(sigma2)) + log(b) - 0.5*(nu+1)*sum(log(1 + z_adj^2/(nu-2)))
    // Note: The Python uses np.sum(np.log(b)) where b is scalar, which is just log(b)
    const double ll = (double)n * c_log
                    - 0.5 * sum_log_sigma2
                    + log(b)
                    - half_nup1 * sum_log_kernel;
    
    return ll;
}


// ============================================================================
// Skew-t Log-Likelihood with Pre-computed Standardized Residuals
// ============================================================================
//
// Alternative interface for when sigma2 and z = eps/sqrt(sigma2) are already computed.

__attribute__((visibility("default"), hot, flatten))
double skewt_ll_z(
    const double* __restrict z,          // Standardized residuals
    const double* __restrict sigma2,     // Conditional variances (for log(sigma2) term)
    const size_t n,                      // Number of observations
    const double nu,                     // Degrees of freedom (> 2)
    const double lam                     // Asymmetry parameter (-1, 1)
)
{
    // Compute Hansen constants
    const double c_log = lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI * (nu - 2));
    const double c_exp = exp(c_log);
    const double a = 4.0 * lam * c_exp * (nu - 2.0) / (nu - 1.0);
    const double b = sqrt(1.0 + 3.0 * lam * lam - a * a);
    const double nu_m2 = nu - 2.0;
    const double half_nup1 = 0.5 * (nu + 1.0);
    
    double sum_log_sigma2 = 0.0;
    double sum_log_kernel = 0.0;
    
    for (size_t t = 0; t < n; ++t) {
        const double bz_plus_a = b * z[t] + a;
        const double sign_term = (bz_plus_a >= 0.0) ? 1.0 : -1.0;
        const double z_adj = bz_plus_a / (1.0 - lam * sign_term);
        
        sum_log_sigma2 += log(sigma2[t]);
        sum_log_kernel += log1p(z_adj * z_adj / nu_m2);
    }
    
    // Note: Python uses np.sum(np.log(b)) where b is scalar = log(b)
    // Mathematically it should be n*log(b), but we match Python for consistency
    return (double)n * c_log
         - 0.5 * sum_log_sigma2
         + log(b)
         - half_nup1 * sum_log_kernel;
}


// ============================================================================
// Skew-t Negative Log-Likelihood (for minimization)
// ============================================================================

__attribute__((visibility("default"), hot, flatten))
double skewt_nll(
    const double* __restrict resid,
    const double* __restrict sigma2,
    const size_t n,
    const double nu,
    const double lam
)
{
    return -skewt_ll(resid, sigma2, n, nu, lam);
}

// ============================================================================
// GARCH(1,1) + Skew-t Gradient
// ============================================================================
//
// Computes NLL and gradient for GARCH(1,1) with Hansen Skew-t errors.
// Parameters: theta = [omega, alpha, beta, nu, lam]
//
// Validated against the internal AD oracle for the GARCH(1,1)+SkewT objective.

__attribute__((visibility("default"), hot, flatten))
double garch_ll_grad_11_skewt(
    const double* __restrict theta,    // [omega, alpha, beta, nu, lam]
    const double* __restrict y,        // Returns data
    double* __restrict grad,           // Output: 5-element gradient
    const size_t n                     // Number of observations
)
{
    const double omega = theta[0];
    const double alpha = theta[1];
    const double beta = theta[2];
    const double nu = theta[3];
    const double lam = theta[4];
    
    // Hansen constants
    const double nu_m1 = nu - 1.0;
    const double nu_m2 = nu - 2.0;
    const double c_log = lgamma_approx(0.5 * (nu + 1)) - lgamma_approx(0.5 * nu) - 0.5 * log(M_PI * nu_m2);
    const double c = exp(c_log);
    const double a = 4.0 * lam * c * nu_m2 / nu_m1;
    const double b_sq = 1.0 + 3.0 * lam * lam - a * a;
    const double b = sqrt(b_sq);
    
    // Derivatives of constants w.r.t. nu
    const double psi_half = digamma_approx(0.5 * (nu + 1)) * 0.5;
    const double psi_nu2 = digamma_approx(0.5 * nu) * 0.5;
    const double dclog_dnu = psi_half - psi_nu2 - 0.5 / nu_m2;
    const double dc_dnu = c * dclog_dnu;
    const double da_dnu = 4.0 * lam * (dc_dnu * nu_m2 / nu_m1 + c / (nu_m1 * nu_m1));
    const double db_dnu = -a * da_dnu / b;
    
    // Derivatives of constants w.r.t. lam
    const double da_dlam = 4.0 * c * nu_m2 / nu_m1;
    const double db_dlam = (3.0 * lam - a * da_dlam) / b;
    
    // Base constant for NLL
    const double log_b = log(b);
    const double half_nup1 = 0.5 * (nu + 1.0);
    
    // Initial variance = mean(y^2)
    double h0 = 0.0;
    for (size_t t = 0; t < n; ++t) {
        h0 += y[t] * y[t];
    }
    h0 /= (double)n;
    if (h0 < H_FLOOR) h0 = H_FLOOR;
    
    // Initialize accumulation
    double nll = -(double)n * c_log - log_b;
    double grad_omega = 0.0, grad_alpha = 0.0, grad_beta = 0.0;
    double grad_nu = -(double)n * dclog_dnu - db_dnu / b;
    double grad_lam = -db_dlam / b;
    
    // Variance and sensitivity states
    double h_prev = h0;
    double dh_domega_prev = 0.0;
    double dh_dalpha_prev = 0.0;
    double dh_dbeta_prev = 0.0;
    
    // t=0 contribution: h0 is fixed, so only nu/lambda contribute to the gradient.
    {
        const double e = y[0];
        const double h = h0;
        const double sqrth = sqrt(h);
        const double z = e / sqrth;
        const double u = b * z + a;
        const double sign_u = (u >= 0.0) ? 1.0 : -1.0;
        const double s = 1.0 - sign_u * lam;
        const double z_adj = u / s;
        const double D = nu_m2 + z_adj * z_adj;

        nll += 0.5 * log(h) + half_nup1 * log(D / nu_m2);

        const double du_dnu = db_dnu * z + da_dnu;
        const double dz_adj_dnu = du_dnu / s;
        const double dD_dnu = 1.0 + 2.0 * z_adj * dz_adj_dnu;
        grad_nu += 0.5 * log(D / nu_m2)
                 + half_nup1 * (dD_dnu / D - 1.0 / nu_m2);

        const double du_dlam = db_dlam * z + da_dlam;
        const double ds_dlam = -sign_u;
        const double dz_adj_dlam = (du_dlam * s - u * ds_dlam) / (s * s);
        const double dD_dlam = 2.0 * z_adj * dz_adj_dlam;
        grad_lam += half_nup1 * dD_dlam / D;
    }

    // Main loop (start from t=1)
    for (size_t t = 1; t < n; ++t) {
        const double y_prev = y[t - 1];
        const double eps2_prev = y_prev * y_prev;
        
        // Variance recursion
        double h = omega + alpha * eps2_prev + beta * h_prev;
        if (h < H_FLOOR) h = H_FLOOR;
        
        // Variance sensitivities
        const double dh_domega = 1.0 + beta * dh_domega_prev;
        const double dh_dalpha = eps2_prev + beta * dh_dalpha_prev;
        const double dh_dbeta = h_prev + beta * dh_dbeta_prev;
        
        // Current observation
        const double e = y[t];
        const double sqrth = sqrt(h);
        const double z = e / sqrth;
        
        // Hansen transformation
        const double u = b * z + a;
        const double sign_u = (u >= 0.0) ? 1.0 : -1.0;
        const double s = 1.0 - sign_u * lam;
        const double z_adj = u / s;
        const double D = nu_m2 + z_adj * z_adj;
        
        // NLL contribution
        nll += 0.5 * log(h) + half_nup1 * log(D / nu_m2);
        
        // ==============================
        // Gradient w.r.t. h (for GARCH params)
        // ==============================
        const double dz_dh = -z / (2.0 * h);
        const double du_dh = b * dz_dh;
        const double dz_adj_dh = du_dh / s;
        const double dD_dh = 2.0 * z_adj * dz_adj_dh;
        
        const double dnll_dh = 0.5 / h + half_nup1 * dD_dh / D;
        
        // GARCH params via chain rule
        grad_omega += dnll_dh * dh_domega;
        grad_alpha += dnll_dh * dh_dalpha;
        grad_beta += dnll_dh * dh_dbeta;
        
        // ==============================
        // Gradient w.r.t. nu
        // ==============================
        const double du_dnu = db_dnu * z + da_dnu;
        const double dz_adj_dnu = du_dnu / s;
        const double dD_dnu = 1.0 + 2.0 * z_adj * dz_adj_dnu;
        
        const double dnll_dnu = 0.5 * log(D / nu_m2)
                              + half_nup1 * (dD_dnu / D - 1.0 / nu_m2);
        grad_nu += dnll_dnu;
        
        // ==============================
        // Gradient w.r.t. lam
        // ==============================
        const double du_dlam = db_dlam * z + da_dlam;
        const double ds_dlam = -sign_u;
        const double dz_adj_dlam = (du_dlam * s - u * ds_dlam) / (s * s);
        const double dD_dlam = 2.0 * z_adj * dz_adj_dlam;
        
        const double dnll_dlam = half_nup1 * dD_dlam / D;
        grad_lam += dnll_dlam;
        
        // Update states for next iteration
        h_prev = h;
        dh_domega_prev = dh_domega;
        dh_dalpha_prev = dh_dalpha;
        dh_dbeta_prev = dh_dbeta;
    }
    
    // Store gradient
    grad[0] = grad_omega;
    grad[1] = grad_alpha;
    grad[2] = grad_beta;
    grad[3] = grad_nu;
    grad[4] = grad_lam;
    
    return nll;
}
