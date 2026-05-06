// scivol/_csrc/transforms_logspace.c
//
// Parameter transformations for unconstrained (log-space) optimization.
//
// Transforms:
//   omega = softplus(z_omega)               ensures omega > 0
//   (alpha, beta, r) = softmax(z_alpha, z_beta, 0)  ensures alpha + beta < 1
//   nu = 2 + softplus(z_nu)                 ensures nu > 2
//   lam = tanh(z_lam)                       ensures -1 < lam < 1

#include <math.h>
#include <stddef.h>
#include "math_and_helpers.h"

#define SOFTPLUS_THRESHOLD 20.0

// Softplus: log(1 + exp(x)) - numerically stable
VLK_FORCE_INLINE double softplus(double x) {
    if (x > SOFTPLUS_THRESHOLD) return x;
    return log1p(exp(x));
}

// Softplus inverse: log(exp(y) - 1)
VLK_FORCE_INLINE double softplus_inv(double y) {
    if (y > SOFTPLUS_THRESHOLD) return y;
    return log(expm1(y));
}

// Softplus derivative: sigmoid(x) = 1 / (1 + exp(-x))
VLK_FORCE_INLINE double softplus_deriv(double x) {
    if (x > SOFTPLUS_THRESHOLD) return 1.0;
    if (x < -SOFTPLUS_THRESHOLD) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

VLK_FORCE_INLINE double softplus_deriv_from_positive(double y) {
    if (y > SOFTPLUS_THRESHOLD) return 1.0;
    if (y <= 0.0) return 0.0;
    return -expm1(-y);
}

// =============================================================================
// GARCH(1,1) SPECIALIZED FUNCTIONS
// =============================================================================

__attribute__((visibility("default"), hot, flatten))
void pack_garch_11(const double *z, double *theta)
{
    // z = [z_omega, z_alpha, z_beta]
    // theta = [omega, alpha, beta]
    
    const double z_omega = z[0];
    const double z_alpha = z[1];
    const double z_beta  = z[2];
    
    // omega = softplus(z_omega)
    theta[0] = softplus(z_omega);
    
    // Joint softmax: (alpha, beta, r) = softmax(z_alpha, z_beta, 0)
    // Using log-sum-exp trick for numerical stability:
    // lse = max(z_alpha, z_beta, 0) + log(exp(z_alpha - max) + exp(z_beta - max) + exp(0 - max))
    const double m = MAX(MAX(z_alpha, z_beta), 0.0);
    const double sum_exp = exp(z_alpha - m) + exp(z_beta - m) + exp(-m);
    const double lse = m + log(sum_exp);
    
    theta[1] = exp(z_alpha - lse);  // alpha
    theta[2] = exp(z_beta - lse);   // beta
}

__attribute__((visibility("default"), hot, flatten))
void pack_garch_studentt_11(const double *z, double *theta)
{
    // z = [z_omega, z_alpha, z_beta, z_nu]
    // theta = [omega, alpha, beta, nu]
    
    // First pack GARCH params
    pack_garch_11(z, theta);
    
    // nu = 2 + softplus(z_nu)
    theta[3] = 2.0 + softplus(z[3]);
}

__attribute__((visibility("default"), hot, flatten))
void pack_garch_skewt_11(const double *z, double *theta)
{
    // z = [z_omega, z_alpha, z_beta, z_nu, z_lam]
    // theta = [omega, alpha, beta, nu, lam]
    
    // First pack GARCH params
    pack_garch_11(z, theta);
    
    // nu = 2 + softplus(z_nu)
    theta[3] = 2.0 + softplus(z[3]);
    
    // lam = tanh(z_lam)
    theta[4] = tanh(z[4]);
}

__attribute__((visibility("default"), hot, flatten))
void jacobian_garch_11(const double *theta, double *J)
{
    // Compute Jacobian J = d(theta)/d(z) for GARCH(1,1)
    // J is 3x3 matrix in row-major order
    //
    // For joint softmax:
    //   d(alpha)/d(z_alpha) = alpha * (1 - alpha)
    //   d(alpha)/d(z_beta)  = -alpha * beta
    //   d(beta)/d(z_alpha)  = -beta * alpha
    //   d(beta)/d(z_beta)   = beta * (1 - beta)
    
    const double omega = theta[0];
    const double alpha = theta[1];
    const double beta  = theta[2];
    
    // Initialize to zero
    dzeros(J, 9);
    
    // J[0,0] = d(omega)/d(z_omega) for omega = softplus(z_omega)
    J[0] = softplus_deriv_from_positive(omega);
    
    // J[1,1] = d(alpha)/d(z_alpha) = alpha * (1 - alpha)
    J[4] = alpha * (1.0 - alpha);
    
    // J[1,2] = d(alpha)/d(z_beta) = -alpha * beta
    J[5] = -alpha * beta;
    
    // J[2,1] = d(beta)/d(z_alpha) = -beta * alpha
    J[7] = -beta * alpha;
    
    // J[2,2] = d(beta)/d(z_beta) = beta * (1 - beta)
    J[8] = beta * (1.0 - beta);
}

__attribute__((visibility("default"), hot, flatten))
void jacobian_garch_studentt_11(const double *theta, double *J)
{
    // Compute Jacobian for GARCH(1,1) + Student-t
    // J is 4x4 matrix in row-major order
    
    const double omega = theta[0];
    const double alpha = theta[1];
    const double beta  = theta[2];
    const double nu    = theta[3];
    
    // Initialize to zero
    dzeros(J, 16);
    
    // GARCH block (same as jacobian_garch_11)
    J[0]  = softplus_deriv_from_positive(omega);  // J[0,0]
    J[5]  = alpha * (1.0 - alpha);    // J[1,1]
    J[6]  = -alpha * beta;            // J[1,2]
    J[9]  = -beta * alpha;            // J[2,1]
    J[10] = beta * (1.0 - beta);      // J[2,2]
    
    // J[3,3] = d(nu)/d(z_nu) = softplus'(z_nu) = sigmoid(z_nu)
    // z_nu = softplus_inv(nu - 2)
    const double z_nu = softplus_inv(nu - 2.0);
    J[15] = softplus_deriv(z_nu);
}

__attribute__((visibility("default"), hot, flatten))
void jacobian_garch_skewt_11(const double *theta, double *J)
{
    // Compute Jacobian for GARCH(1,1) + SkewT
    // J is 5x5 matrix in row-major order
    
    const double omega = theta[0];
    const double alpha = theta[1];
    const double beta  = theta[2];
    const double nu    = theta[3];
    const double lam   = theta[4];
    
    // Initialize to zero
    dzeros(J, 25);
    
    // GARCH block
    J[0]  = softplus_deriv_from_positive(omega);  // J[0,0]
    J[6]  = alpha * (1.0 - alpha);    // J[1,1]
    J[7]  = -alpha * beta;            // J[1,2]
    J[11] = -beta * alpha;            // J[2,1]
    J[12] = beta * (1.0 - beta);      // J[2,2]
    
    // J[3,3] = d(nu)/d(z_nu) = softplus'(z_nu) = sigmoid(z_nu)
    const double z_nu = softplus_inv(nu - 2.0);
    J[18] = softplus_deriv(z_nu);
    
    // J[4,4] = d(lam)/d(z_lam) = 1 - lam^2 = sech^2(z_lam)
    J[24] = 1.0 - lam * lam;
}

__attribute__((visibility("default"), hot, flatten))
void transform_grad_11_normal(const double *grad_theta, const double *J, double *grad_z)
{
    // grad_z = J^T @ grad_theta for K=3
    // J is 3x3 row-major
    
    // J^T @ grad_theta:
    // grad_z[i] = sum_j J[j,i] * grad_theta[j] = sum_j J[j*3 + i] * grad_theta[j]
    
    grad_z[0] = J[0] * grad_theta[0];  // Only J[0,0] is nonzero in column 0
    grad_z[1] = J[4] * grad_theta[1] + J[7] * grad_theta[2];  // J[1,1] and J[2,1]
    grad_z[2] = J[5] * grad_theta[1] + J[8] * grad_theta[2];  // J[1,2] and J[2,2]
}

__attribute__((visibility("default"), hot, flatten))
void transform_grad_11_studentt(const double *grad_theta, const double *J, double *grad_z)
{
    // grad_z = J^T @ grad_theta for K=4
    // J is 4x4 row-major
    
    grad_z[0] = J[0] * grad_theta[0];
    grad_z[1] = J[5] * grad_theta[1] + J[9] * grad_theta[2];
    grad_z[2] = J[6] * grad_theta[1] + J[10] * grad_theta[2];
    grad_z[3] = J[15] * grad_theta[3];
}

__attribute__((visibility("default"), hot, flatten))
void transform_grad_11_skewt(const double *grad_theta, const double *J, double *grad_z)
{
    // grad_z = J^T @ grad_theta for K=5
    // J is 5x5 row-major
    
    grad_z[0] = J[0] * grad_theta[0];
    grad_z[1] = J[6] * grad_theta[1] + J[11] * grad_theta[2];
    grad_z[2] = J[7] * grad_theta[1] + J[12] * grad_theta[2];
    grad_z[3] = J[18] * grad_theta[3];
    grad_z[4] = J[24] * grad_theta[4];
}


// =============================================================================
// GJR-GARCH(1,1) SPECIALIZED FUNCTIONS
// =============================================================================

// Pack: z = [z_omega, z_alpha, z_gamma, z_beta] -> theta = [omega, alpha, gamma, beta]
// Uses 4-class softmax (alpha, gamma, beta, slack=0) to ensure alpha+gamma+beta < 1
__attribute__((visibility("default"), hot, flatten))
void pack_gjr_garch_11(const double *z, double *theta)
{
    const double z_omega = z[0];
    const double z_alpha = z[1];
    const double z_gamma = z[2];
    const double z_beta  = z[3];

    theta[0] = softplus(z_omega);

    const double m = MAX(MAX(MAX(z_alpha, z_gamma), z_beta), 0.0);
    const double sum_exp = exp(z_alpha - m) + exp(z_gamma - m) + exp(z_beta - m) + exp(-m);
    const double lse = m + log(sum_exp);

    theta[1] = exp(z_alpha - lse);  // alpha
    theta[2] = exp(z_gamma - lse);  // gamma
    theta[3] = exp(z_beta - lse);   // beta
}

__attribute__((visibility("default"), hot, flatten))
void pack_gjr_garch_studentt_11(const double *z, double *theta)
{
    pack_gjr_garch_11(z, theta);
    theta[4] = 2.0 + softplus(z[4]);
}

__attribute__((visibility("default"), hot, flatten))
void pack_gjr_garch_skewt_11(const double *z, double *theta)
{
    pack_gjr_garch_11(z, theta);
    theta[4] = 2.0 + softplus(z[4]);
    theta[5] = tanh(z[5]);
}

// Jacobian for GJR-GARCH(1,1) Normal (4x4)
__attribute__((visibility("default"), hot, flatten))
void jacobian_gjr_garch_11(const double *theta, double *J)
{
    const double omega = theta[0];
    const double alpha = theta[1];
    const double gamma = theta[2];
    const double beta  = theta[3];

    dzeros(J, 16);

    J[0]  = softplus_deriv_from_positive(omega);  // J[0,0] = d(omega)/d(z_omega)
    // Softmax Jacobian for (alpha, gamma, beta) in positions [1..3] x [1..3]
    J[5]  = alpha * (1.0 - alpha);        // J[1,1]
    J[6]  = -alpha * gamma;               // J[1,2]
    J[7]  = -alpha * beta;                // J[1,3]
    J[9]  = -gamma * alpha;               // J[2,1]
    J[10] = gamma * (1.0 - gamma);        // J[2,2]
    J[11] = -gamma * beta;                // J[2,3]
    J[13] = -beta * alpha;                // J[3,1]
    J[14] = -beta * gamma;                // J[3,2]
    J[15] = beta * (1.0 - beta);          // J[3,3]
}

// Jacobian for GJR-GARCH(1,1) Student-t (5x5)
__attribute__((visibility("default"), hot, flatten))
void jacobian_gjr_garch_studentt_11(const double *theta, double *J)
{
    const double omega = theta[0];
    const double alpha = theta[1];
    const double gamma = theta[2];
    const double beta  = theta[3];
    const double nu    = theta[4];

    dzeros(J, 25);

    J[0]  = softplus_deriv_from_positive(omega);
    J[6]  = alpha * (1.0 - alpha);
    J[7]  = -alpha * gamma;
    J[8]  = -alpha * beta;
    J[11] = -gamma * alpha;
    J[12] = gamma * (1.0 - gamma);
    J[13] = -gamma * beta;
    J[16] = -beta * alpha;
    J[17] = -beta * gamma;
    J[18] = beta * (1.0 - beta);

    const double z_nu = softplus_inv(nu - 2.0);
    J[24] = softplus_deriv(z_nu);
}

// Jacobian for GJR-GARCH(1,1) Skew-t (6x6)
__attribute__((visibility("default"), hot, flatten))
void jacobian_gjr_garch_skewt_11(const double *theta, double *J)
{
    const double omega = theta[0];
    const double alpha = theta[1];
    const double gamma = theta[2];
    const double beta  = theta[3];
    const double nu    = theta[4];
    const double lam   = theta[5];

    dzeros(J, 36);

    J[0]  = softplus_deriv_from_positive(omega);
    J[7]  = alpha * (1.0 - alpha);
    J[8]  = -alpha * gamma;
    J[9]  = -alpha * beta;
    J[13] = -gamma * alpha;
    J[14] = gamma * (1.0 - gamma);
    J[15] = -gamma * beta;
    J[19] = -beta * alpha;
    J[20] = -beta * gamma;
    J[21] = beta * (1.0 - beta);

    const double z_nu = softplus_inv(nu - 2.0);
    J[28] = softplus_deriv(z_nu);
    J[35] = 1.0 - lam * lam;
}

// Transform gradient for GJR-GARCH(1,1) Normal: grad_z = J^T @ grad_theta (K=4)
__attribute__((visibility("default"), hot, flatten))
void transform_grad_gjr_11_normal(const double *grad_theta, const double *J, double *grad_z)
{
    grad_z[0] = J[0] * grad_theta[0];
    grad_z[1] = J[5]  * grad_theta[1] + J[9]  * grad_theta[2] + J[13] * grad_theta[3];
    grad_z[2] = J[6]  * grad_theta[1] + J[10] * grad_theta[2] + J[14] * grad_theta[3];
    grad_z[3] = J[7]  * grad_theta[1] + J[11] * grad_theta[2] + J[15] * grad_theta[3];
}

// Transform gradient for GJR-GARCH(1,1) Student-t (K=5)
__attribute__((visibility("default"), hot, flatten))
void transform_grad_gjr_11_studentt(const double *grad_theta, const double *J, double *grad_z)
{
    grad_z[0] = J[0] * grad_theta[0];
    grad_z[1] = J[6]  * grad_theta[1] + J[11] * grad_theta[2] + J[16] * grad_theta[3];
    grad_z[2] = J[7]  * grad_theta[1] + J[12] * grad_theta[2] + J[17] * grad_theta[3];
    grad_z[3] = J[8]  * grad_theta[1] + J[13] * grad_theta[2] + J[18] * grad_theta[3];
    grad_z[4] = J[24] * grad_theta[4];
}

// Transform gradient for GJR-GARCH(1,1) Skew-t (K=6)
__attribute__((visibility("default"), hot, flatten))
void transform_grad_gjr_11_skewt(const double *grad_theta, const double *J, double *grad_z)
{
    grad_z[0] = J[0] * grad_theta[0];
    grad_z[1] = J[7]  * grad_theta[1] + J[13] * grad_theta[2] + J[19] * grad_theta[3];
    grad_z[2] = J[8]  * grad_theta[1] + J[14] * grad_theta[2] + J[20] * grad_theta[3];
    grad_z[3] = J[9]  * grad_theta[1] + J[15] * grad_theta[2] + J[21] * grad_theta[3];
    grad_z[4] = J[28] * grad_theta[4];
    grad_z[5] = J[35] * grad_theta[5];
}


// =============================================================================
// GJR-GARCH(p,q) GENERAL FUNCTIONS
// =============================================================================

// Pack: z = [z_omega, z_alpha_1..p, z_gamma_1..p, z_beta_1..q] ->
//        theta = [omega, alpha_1..p, gamma_1..p, beta_1..q]
__attribute__((visibility("default"), hot))
void pack_gjr_garch_pq(const double *z, double *theta, size_t p, size_t q)
{
    const size_t K = 1 + 2 * p + q;

    theta[0] = softplus(z[0]);

    // Softmax over z[1..K-1] plus slack = 0
    double m = 0.0;
    for (size_t i = 1; i < K; ++i)
        if (z[i] > m) m = z[i];

    double sum_exp = exp(-m);
    for (size_t i = 1; i < K; ++i)
        sum_exp += exp(z[i] - m);

    const double lse = m + log(sum_exp);

    for (size_t i = 1; i < K; ++i)
        theta[i] = exp(z[i] - lse);
}

__attribute__((visibility("default"), hot))
void pack_gjr_garch_studentt_pq(const double *z, double *theta, size_t p, size_t q)
{
    const size_t n_gjr = 1 + 2 * p + q;
    pack_gjr_garch_pq(z, theta, p, q);
    theta[n_gjr] = 2.0 + softplus(z[n_gjr]);
}

__attribute__((visibility("default"), hot))
void pack_gjr_garch_skewt_pq(const double *z, double *theta, size_t p, size_t q)
{
    const size_t n_gjr = 1 + 2 * p + q;
    pack_gjr_garch_pq(z, theta, p, q);
    theta[n_gjr] = 2.0 + softplus(z[n_gjr]);
    theta[n_gjr + 1] = tanh(z[n_gjr + 1]);
}

__attribute__((visibility("default"), hot))
void jacobian_gjr_garch_pq(const double *theta, double *J, size_t p, size_t q)
{
    const size_t K = 1 + 2 * p + q;
    dzeros(J, K * K);
    J[0] = softplus_deriv_from_positive(theta[0]);

    for (size_t i = 1; i < K; ++i)
        for (size_t j = 1; j < K; ++j) {
            const double delta_ij = (i == j) ? 1.0 : 0.0;
            J[i * K + j] = theta[i] * (delta_ij - theta[j]);
        }
}

__attribute__((visibility("default"), hot))
void jacobian_gjr_garch_studentt_pq(const double *theta, double *J, size_t p, size_t q)
{
    const size_t n_gjr = 1 + 2 * p + q;
    const size_t K = n_gjr + 1;
    dzeros(J, K * K);
    J[0] = softplus_deriv_from_positive(theta[0]);

    for (size_t i = 1; i < n_gjr; ++i)
        for (size_t j = 1; j < n_gjr; ++j) {
            const double delta_ij = (i == j) ? 1.0 : 0.0;
            J[i * K + j] = theta[i] * (delta_ij - theta[j]);
        }

    const double nu = theta[n_gjr];
    const double z_nu = softplus_inv(nu - 2.0);
    J[n_gjr * K + n_gjr] = softplus_deriv(z_nu);
}

__attribute__((visibility("default"), hot))
void jacobian_gjr_garch_skewt_pq(const double *theta, double *J, size_t p, size_t q)
{
    const size_t n_gjr = 1 + 2 * p + q;
    const size_t K = n_gjr + 2;
    dzeros(J, K * K);
    J[0] = softplus_deriv_from_positive(theta[0]);

    for (size_t i = 1; i < n_gjr; ++i)
        for (size_t j = 1; j < n_gjr; ++j) {
            const double delta_ij = (i == j) ? 1.0 : 0.0;
            J[i * K + j] = theta[i] * (delta_ij - theta[j]);
        }

    const double nu = theta[n_gjr];
    const double z_nu = softplus_inv(nu - 2.0);
    J[n_gjr * K + n_gjr] = softplus_deriv(z_nu);

    const double lam = theta[n_gjr + 1];
    J[(n_gjr + 1) * K + (n_gjr + 1)] = 1.0 - lam * lam;
}


// =============================================================================
// GENERAL GARCH(p,q) FUNCTIONS
// =============================================================================

__attribute__((visibility("default"), hot))
void pack_garch_pq(const double *z, double *theta, size_t p, size_t q)
{
    // z = [z_omega, z_alpha_1, ..., z_alpha_p, z_beta_1, ..., z_beta_q]
    // theta = [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
    
    const size_t K = 1 + p + q;
    
    // omega = softplus(z_omega)
    theta[0] = softplus(z[0]);
    
    // Joint softmax over z_alpha, z_beta, and slack variable 0
    // First find max for numerical stability
    double m = 0.0;  // slack variable
    for (size_t i = 1; i < K; ++i) {
        if (z[i] > m) m = z[i];
    }
    
    // Compute sum of exp(z_i - m)
    double sum_exp = exp(-m);  // slack variable contribution
    for (size_t i = 1; i < K; ++i) {
        sum_exp += exp(z[i] - m);
    }
    
    const double lse = m + log(sum_exp);
    
    // Compute alpha and beta from softmax
    for (size_t i = 1; i < K; ++i) {
        theta[i] = exp(z[i] - lse);
    }
}

__attribute__((visibility("default"), hot))
void pack_garch_studentt_pq(const double *z, double *theta, size_t p, size_t q)
{
    const size_t n_garch = 1 + p + q;
    
    // Pack GARCH params
    pack_garch_pq(z, theta, p, q);
    
    // nu = 2 + softplus(z_nu)
    theta[n_garch] = 2.0 + softplus(z[n_garch]);
}

__attribute__((visibility("default"), hot))
void pack_garch_skewt_pq(const double *z, double *theta, size_t p, size_t q)
{
    const size_t n_garch = 1 + p + q;
    
    // Pack GARCH params
    pack_garch_pq(z, theta, p, q);
    
    // nu = 2 + softplus(z_nu)
    theta[n_garch] = 2.0 + softplus(z[n_garch]);
    
    // lam = tanh(z_lam)
    theta[n_garch + 1] = tanh(z[n_garch + 1]);
}

__attribute__((visibility("default"), hot))
void jacobian_garch_pq(const double *theta, double *J, size_t p, size_t q)
{
    // Compute Jacobian J = d(theta)/d(z) for GARCH(p,q)
    // J is K x K matrix in row-major order where K = 1 + p + q
    //
    // For joint softmax over [z_alpha..., z_beta..., 0]:
    //   d(theta_i)/d(z_j) = theta_i * (delta_ij - theta_j) for i,j >= 1
    //   d(omega)/d(z_omega) follows omega = softplus(z_omega)
    
    const size_t K = 1 + p + q;
    
    // Initialize to zero
    dzeros(J, K * K);
    
    // J[0,0] = d(omega)/d(z_omega)
    J[0] = softplus_deriv_from_positive(theta[0]);
    
    // Softmax Jacobian for alpha and beta (indices 1 to K-1)
    for (size_t i = 1; i < K; ++i) {
        for (size_t j = 1; j < K; ++j) {
            const double delta_ij = (i == j) ? 1.0 : 0.0;
            J[i * K + j] = theta[i] * (delta_ij - theta[j]);
        }
    }
}

__attribute__((visibility("default"), hot))
void jacobian_garch_studentt_pq(const double *theta, double *J, size_t p, size_t q)
{
    const size_t n_garch = 1 + p + q;
    const size_t K = n_garch + 1;
    
    // Initialize to zero
    dzeros(J, K * K);
    
    // GARCH block
    J[0] = softplus_deriv_from_positive(theta[0]);  // omega
    
    for (size_t i = 1; i < n_garch; ++i) {
        for (size_t j = 1; j < n_garch; ++j) {
            const double delta_ij = (i == j) ? 1.0 : 0.0;
            J[i * K + j] = theta[i] * (delta_ij - theta[j]);
        }
    }
    
    // nu: J[n_garch, n_garch] = softplus'(z_nu)
    const double nu = theta[n_garch];
    const double z_nu = softplus_inv(nu - 2.0);
    J[n_garch * K + n_garch] = softplus_deriv(z_nu);
}

__attribute__((visibility("default"), hot))
void jacobian_garch_skewt_pq(const double *theta, double *J, size_t p, size_t q)
{
    const size_t n_garch = 1 + p + q;
    const size_t K = n_garch + 2;
    
    // Initialize to zero
    dzeros(J, K * K);
    
    // GARCH block
    J[0] = softplus_deriv_from_positive(theta[0]);  // omega
    
    for (size_t i = 1; i < n_garch; ++i) {
        for (size_t j = 1; j < n_garch; ++j) {
            const double delta_ij = (i == j) ? 1.0 : 0.0;
            J[i * K + j] = theta[i] * (delta_ij - theta[j]);
        }
    }
    
    // nu: J[n_garch, n_garch] = softplus'(z_nu)
    const double nu = theta[n_garch];
    const double z_nu = softplus_inv(nu - 2.0);
    J[n_garch * K + n_garch] = softplus_deriv(z_nu);
    
    // lam: J[n_garch+1, n_garch+1] = 1 - lam^2
    const double lam = theta[n_garch + 1];
    J[(n_garch + 1) * K + (n_garch + 1)] = 1.0 - lam * lam;
}

__attribute__((visibility("default"), hot))
void transform_grad_pq(const double *grad_theta, const double *J, double *grad_z, size_t K)
{
    // grad_z = J^T @ grad_theta
    // J is K x K row-major
    
    for (size_t i = 0; i < K; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < K; ++j) {
            sum += J[j * K + i] * grad_theta[j];
        }
        grad_z[i] = sum;
    }
}


// =============================================================================
// ARMA(p,q) + Normal
// =============================================================================

#define ARMA_TANH_BOUND 0.99

__attribute__((visibility("default"), hot))
void pack_arma_normal_pq(const double *z, double *theta, size_t p_ar, size_t q_ma)
{
    const size_t K = 1 + p_ar + q_ma;
    theta[0] = z[0];
    for (size_t i = 1; i < K; ++i) {
        theta[i] = ARMA_TANH_BOUND * tanh(z[i]);
    }
}

__attribute__((visibility("default"), hot))
void jacobian_arma_normal_pq(const double *theta, double *J, size_t p_ar, size_t q_ma)
{
    const size_t K = 1 + p_ar + q_ma;
    dzeros(J, K * K);
    J[0] = 1.0;
    for (size_t i = 1; i < K; ++i) {
        const double ratio = theta[i] / ARMA_TANH_BOUND;
        J[i * K + i] = ARMA_TANH_BOUND * (1.0 - ratio * ratio);
    }
}


// =============================================================================
// ARMA-GARCH(1,1) SPECIALIZED FUNCTIONS
// =============================================================================
//
// Parameter layout (Normal): [c, phi, theta_ma, omega, alpha, beta]   K=6
//                 (StudentT): [c, phi, theta_ma, omega, alpha, beta, nu]  K=7
//                 (SkewT):    [c, phi, theta_ma, omega, alpha, beta, nu, lam] K=8
//
// Transforms:
//   c       = z_c                    (identity)
//   phi     = 0.99 * tanh(z_phi)     (|phi| < 0.99)
//   theta_ma= 0.99 * tanh(z_theta)   (|theta| < 0.99)
//   omega   = softplus(z_omega)       (omega > 0)
//   (alpha, beta) = softmax(z_alpha, z_beta, 0)  (alpha + beta < 1)
//   nu      = 2 + softplus(z_nu)      (nu > 2)
//   lam     = tanh(z_lam)             (|lam| < 1)

__attribute__((visibility("default"), hot, flatten))
void pack_arma_garch_normal_11(const double *z, double *theta)
{
    // Mean: c (identity), phi (tanh), theta_ma (tanh)
    theta[0] = z[0];                           // c
    theta[1] = ARMA_TANH_BOUND * tanh(z[1]);   // phi
    theta[2] = ARMA_TANH_BOUND * tanh(z[2]);   // theta_ma

    // Vol: omega (softplus), alpha/beta (softmax with slack=0)
    theta[3] = softplus(z[3]);

    const double z_alpha = z[4];
    const double z_beta  = z[5];
    const double m = MAX(MAX(z_alpha, z_beta), 0.0);
    const double sum_exp = exp(z_alpha - m) + exp(z_beta - m) + exp(-m);
    const double lse = m + log(sum_exp);
    theta[4] = exp(z_alpha - lse);  // alpha
    theta[5] = exp(z_beta - lse);   // beta
}

__attribute__((visibility("default"), hot, flatten))
void pack_arma_garch_studentt_11(const double *z, double *theta)
{
    pack_arma_garch_normal_11(z, theta);
    theta[6] = 2.0 + softplus(z[6]);  // nu
}

__attribute__((visibility("default"), hot, flatten))
void pack_arma_garch_skewt_11(const double *z, double *theta)
{
    pack_arma_garch_normal_11(z, theta);
    theta[6] = 2.0 + softplus(z[6]);  // nu
    theta[7] = tanh(z[7]);             // lam
}

__attribute__((visibility("default"), hot, flatten))
void jacobian_arma_garch_normal_11(const double *theta, double *J)
{
    // J is 6x6 row-major
    dzeros(J, 36);

    // J[0,0] = d(c)/d(z_c) = 1
    J[0] = 1.0;

    // J[1,1] = d(phi)/d(z_phi) = 0.99 * (1 - (phi/0.99)^2)
    const double phi = theta[1];
    J[7] = ARMA_TANH_BOUND * (1.0 - (phi / ARMA_TANH_BOUND) * (phi / ARMA_TANH_BOUND));

    // J[2,2] = d(theta_ma)/d(z_theta) = 0.99 * (1 - (theta_ma/0.99)^2)
    const double th = theta[2];
    J[14] = ARMA_TANH_BOUND * (1.0 - (th / ARMA_TANH_BOUND) * (th / ARMA_TANH_BOUND));

    // J[3,3] = d(omega)/d(z_omega) for omega = softplus(z_omega)
    J[21] = softplus_deriv_from_positive(theta[3]);

    // Softmax block for alpha, beta at [4..5] x [4..5]
    const double alpha = theta[4];
    const double beta  = theta[5];
    J[28] = alpha * (1.0 - alpha);   // J[4,4]
    J[29] = -alpha * beta;           // J[4,5]
    J[34] = -beta * alpha;           // J[5,4]
    J[35] = beta * (1.0 - beta);    // J[5,5]
}

__attribute__((visibility("default"), hot, flatten))
void jacobian_arma_garch_studentt_11(const double *theta, double *J)
{
    // J is 7x7 row-major
    dzeros(J, 49);

    const double phi = theta[1];
    const double th  = theta[2];
    const double alpha = theta[4];
    const double beta  = theta[5];
    const double nu    = theta[6];

    J[0]  = 1.0;                                                             // J[0,0] c
    J[8]  = ARMA_TANH_BOUND * (1.0 - (phi / ARMA_TANH_BOUND) * (phi / ARMA_TANH_BOUND));  // J[1,1] phi
    J[16] = ARMA_TANH_BOUND * (1.0 - (th / ARMA_TANH_BOUND) * (th / ARMA_TANH_BOUND));   // J[2,2] theta_ma
    J[24] = softplus_deriv_from_positive(theta[3]);                          // J[3,3] omega
    J[32] = alpha * (1.0 - alpha);                                           // J[4,4]
    J[33] = -alpha * beta;                                                   // J[4,5]
    J[39] = -beta * alpha;                                                   // J[5,4]
    J[40] = beta * (1.0 - beta);                                            // J[5,5]

    const double z_nu = softplus_inv(nu - 2.0);
    J[48] = softplus_deriv(z_nu);                                            // J[6,6] nu
}

__attribute__((visibility("default"), hot, flatten))
void jacobian_arma_garch_skewt_11(const double *theta, double *J)
{
    // J is 8x8 row-major
    dzeros(J, 64);

    const double phi   = theta[1];
    const double th    = theta[2];
    const double alpha = theta[4];
    const double beta  = theta[5];
    const double nu    = theta[6];
    const double lam   = theta[7];

    J[0]  = 1.0;                                                             // J[0,0] c
    J[9]  = ARMA_TANH_BOUND * (1.0 - (phi / ARMA_TANH_BOUND) * (phi / ARMA_TANH_BOUND));  // J[1,1] phi
    J[18] = ARMA_TANH_BOUND * (1.0 - (th / ARMA_TANH_BOUND) * (th / ARMA_TANH_BOUND));   // J[2,2] theta_ma
    J[27] = softplus_deriv_from_positive(theta[3]);                          // J[3,3] omega
    J[36] = alpha * (1.0 - alpha);                                           // J[4,4]
    J[37] = -alpha * beta;                                                   // J[4,5]
    J[44] = -beta * alpha;                                                   // J[5,4]
    J[45] = beta * (1.0 - beta);                                            // J[5,5]

    const double z_nu = softplus_inv(nu - 2.0);
    J[54] = softplus_deriv(z_nu);                                            // J[6,6] nu
    J[63] = 1.0 - lam * lam;                                                // J[7,7] lam
}


// =============================================================================
// ARMA-GARCH(p,q) GENERAL FUNCTIONS
// =============================================================================
//
// Parameter layout (Normal):
//   theta = [c, phi_1..p_ar, theta_1..q_ma, omega, alpha_1..P, beta_1..Q]
//   K = 1 + p_ar + q_ma + 1 + P + Q  =  n_mean + n_vol
//
// (StudentT): K + 1  (append nu)
// (SkewT):    K + 2  (append nu, lam)

__attribute__((visibility("default"), hot))
void pack_arma_garch_normal_pq(const double *z, double *theta,
                                size_t p_ar, size_t q_ma, size_t P, size_t Q)
{
    const size_t n_mean = 1 + p_ar + q_ma;
    const size_t n_vol  = 1 + P + Q;

    // Mean parameters
    theta[0] = z[0];  // c (identity)
    for (size_t i = 0; i < p_ar; ++i)
        theta[1 + i] = ARMA_TANH_BOUND * tanh(z[1 + i]);
    for (size_t j = 0; j < q_ma; ++j)
        theta[1 + p_ar + j] = ARMA_TANH_BOUND * tanh(z[1 + p_ar + j]);

    // omega = softplus(z_omega)
    theta[n_mean] = softplus(z[n_mean]);

    // Softmax for alpha_1..P, beta_1..Q with slack=0
    const size_t n_sm = P + Q;  // number of softmax entries
    double m = 0.0;  // slack
    for (size_t i = 0; i < n_sm; ++i) {
        const double zi = z[n_mean + 1 + i];
        if (zi > m) m = zi;
    }
    double sum_exp = exp(-m);  // slack
    for (size_t i = 0; i < n_sm; ++i)
        sum_exp += exp(z[n_mean + 1 + i] - m);
    const double lse = m + log(sum_exp);
    for (size_t i = 0; i < n_sm; ++i)
        theta[n_mean + 1 + i] = exp(z[n_mean + 1 + i] - lse);
}

__attribute__((visibility("default"), hot))
void pack_arma_garch_studentt_pq(const double *z, double *theta,
                                  size_t p_ar, size_t q_ma, size_t P, size_t Q)
{
    const size_t n_mean = 1 + p_ar + q_ma;
    const size_t n_vol  = 1 + P + Q;
    pack_arma_garch_normal_pq(z, theta, p_ar, q_ma, P, Q);
    theta[n_mean + n_vol] = 2.0 + softplus(z[n_mean + n_vol]);
}

__attribute__((visibility("default"), hot))
void pack_arma_garch_skewt_pq(const double *z, double *theta,
                                size_t p_ar, size_t q_ma, size_t P, size_t Q)
{
    const size_t n_mean = 1 + p_ar + q_ma;
    const size_t n_vol  = 1 + P + Q;
    pack_arma_garch_normal_pq(z, theta, p_ar, q_ma, P, Q);
    theta[n_mean + n_vol]     = 2.0 + softplus(z[n_mean + n_vol]);
    theta[n_mean + n_vol + 1] = tanh(z[n_mean + n_vol + 1]);
}

__attribute__((visibility("default"), hot))
void jacobian_arma_garch_normal_pq(const double *theta, double *J,
                                    size_t p_ar, size_t q_ma, size_t P, size_t Q)
{
    const size_t n_mean = 1 + p_ar + q_ma;
    const size_t n_vol  = 1 + P + Q;
    const size_t K = n_mean + n_vol;

    dzeros(J, K * K);

    // c: identity
    J[0] = 1.0;

    // phi: 0.99 * sech^2
    for (size_t i = 0; i < p_ar; ++i) {
        const size_t idx = 1 + i;
        const double v = theta[idx];
        J[idx * K + idx] = ARMA_TANH_BOUND * (1.0 - (v / ARMA_TANH_BOUND) * (v / ARMA_TANH_BOUND));
    }

    // theta_ma: 0.99 * sech^2
    for (size_t j = 0; j < q_ma; ++j) {
        const size_t idx = 1 + p_ar + j;
        const double v = theta[idx];
        J[idx * K + idx] = ARMA_TANH_BOUND * (1.0 - (v / ARMA_TANH_BOUND) * (v / ARMA_TANH_BOUND));
    }

    // omega
    J[n_mean * K + n_mean] = softplus_deriv_from_positive(theta[n_mean]);

    // softmax block for alpha, beta at [n_mean+1 .. n_mean+n_vol-1]
    const size_t n_sm = P + Q;
    for (size_t i = 0; i < n_sm; ++i) {
        const size_t ri = n_mean + 1 + i;
        const double ti = theta[ri];
        for (size_t j = 0; j < n_sm; ++j) {
            const size_t cj = n_mean + 1 + j;
            const double delta = (i == j) ? 1.0 : 0.0;
            J[ri * K + cj] = ti * (delta - theta[cj]);
        }
    }
}

__attribute__((visibility("default"), hot))
void jacobian_arma_garch_studentt_pq(const double *theta, double *J,
                                      size_t p_ar, size_t q_ma, size_t P, size_t Q)
{
    const size_t n_mean = 1 + p_ar + q_ma;
    const size_t n_vol  = 1 + P + Q;
    const size_t K_base = n_mean + n_vol;
    const size_t K = K_base + 1;

    dzeros(J, K * K);

    // c: identity
    J[0] = 1.0;

    // phi
    for (size_t i = 0; i < p_ar; ++i) {
        const size_t idx = 1 + i;
        const double v = theta[idx];
        J[idx * K + idx] = ARMA_TANH_BOUND * (1.0 - (v / ARMA_TANH_BOUND) * (v / ARMA_TANH_BOUND));
    }

    // theta_ma
    for (size_t j = 0; j < q_ma; ++j) {
        const size_t idx = 1 + p_ar + j;
        const double v = theta[idx];
        J[idx * K + idx] = ARMA_TANH_BOUND * (1.0 - (v / ARMA_TANH_BOUND) * (v / ARMA_TANH_BOUND));
    }

    // omega
    J[n_mean * K + n_mean] = softplus_deriv_from_positive(theta[n_mean]);

    // softmax
    const size_t n_sm = P + Q;
    for (size_t i = 0; i < n_sm; ++i) {
        const size_t ri = n_mean + 1 + i;
        const double ti = theta[ri];
        for (size_t j = 0; j < n_sm; ++j) {
            const size_t cj = n_mean + 1 + j;
            const double delta = (i == j) ? 1.0 : 0.0;
            J[ri * K + cj] = ti * (delta - theta[cj]);
        }
    }

    // nu
    const double nu = theta[K_base];
    const double z_nu = softplus_inv(nu - 2.0);
    J[K_base * K + K_base] = softplus_deriv(z_nu);
}

__attribute__((visibility("default"), hot))
void jacobian_arma_garch_skewt_pq(const double *theta, double *J,
                                    size_t p_ar, size_t q_ma, size_t P, size_t Q)
{
    const size_t n_mean = 1 + p_ar + q_ma;
    const size_t n_vol  = 1 + P + Q;
    const size_t K_base = n_mean + n_vol;
    const size_t K = K_base + 2;

    dzeros(J, K * K);

    // c: identity
    J[0] = 1.0;

    // phi
    for (size_t i = 0; i < p_ar; ++i) {
        const size_t idx = 1 + i;
        const double v = theta[idx];
        J[idx * K + idx] = ARMA_TANH_BOUND * (1.0 - (v / ARMA_TANH_BOUND) * (v / ARMA_TANH_BOUND));
    }

    // theta_ma
    for (size_t j = 0; j < q_ma; ++j) {
        const size_t idx = 1 + p_ar + j;
        const double v = theta[idx];
        J[idx * K + idx] = ARMA_TANH_BOUND * (1.0 - (v / ARMA_TANH_BOUND) * (v / ARMA_TANH_BOUND));
    }

    // omega
    J[n_mean * K + n_mean] = softplus_deriv_from_positive(theta[n_mean]);

    // softmax
    const size_t n_sm = P + Q;
    for (size_t i = 0; i < n_sm; ++i) {
        const size_t ri = n_mean + 1 + i;
        const double ti = theta[ri];
        for (size_t j = 0; j < n_sm; ++j) {
            const size_t cj = n_mean + 1 + j;
            const double delta = (i == j) ? 1.0 : 0.0;
            J[ri * K + cj] = ti * (delta - theta[cj]);
        }
    }

    // nu
    const double nu = theta[K_base];
    const double z_nu = softplus_inv(nu - 2.0);
    J[K_base * K + K_base] = softplus_deriv(z_nu);

    // lam
    const double lam = theta[K_base + 1];
    J[(K_base + 1) * K + (K_base + 1)] = 1.0 - lam * lam;
}
