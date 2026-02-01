// volkit/_csrc/likelihood_skewt.c
//
// Hansen (1994) Skewed Student-t log-likelihood implementation.
//
// Reference: Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation."
// International Economic Review, 35(3), 705-730.

#include <stddef.h>
#include <math.h>

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
