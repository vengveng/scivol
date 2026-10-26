#pragma once

#include <array>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "fit.hpp"
#include "jet.hpp"

namespace scivolapi {

// Invalid inputs throw invalid_argument. Unidentified tests and numerical
// failures return an explicit status, with NaN statistic and probability.
enum class TestStatus {
    ok, not_estimable, probability_underflow, numerical_range,
    infeasible_parameters, fourth_moment_unavailable, invalid_derivatives,
    invalid_qmle_derivatives, at_boundary, singular_information,
    singular_moment_covariance, pit_endpoint,
};

struct TestStatistic {
    double statistic = std::numeric_limits<double>::quiet_NaN();
    double p_value = std::numeric_limits<double>::quiet_NaN();
    std::size_t df = 0;
    TestStatus status = TestStatus::not_estimable;
    // Underflow keeps a finite statistic and log p-value, with an explicit
    // probability_underflow status and NaN p_value. Rejection remains defined.
    double log_p_value = std::numeric_limits<double>::quiet_NaN();
    [[nodiscard]] bool rejects(double alpha) const noexcept;
};

double normal_survival(double x);
double student_survival(double x, double df);
double chi_square_survival(double x, double df);
TestStatistic coefficient_test(double coefficient, double standard_error, std::size_t df);

struct Portmanteau {
    TestStatistic ljung_box, box_pierce;
    std::vector<double> autocorrelations;
};
// Full-sample centering and denominator; df = lags - df_adjustment.
// This scalar adjustment is the usual ARMA approximation, not a general
// correction for fitted PITs or squared GARCH residuals. Use li_mak for squared residuals.
Portmanteau portmanteau(std::span<const double> series, std::size_t lags,
                        std::size_t df_adjustment = 0);
TestStatistic dgt(std::span<const double> pit, std::size_t cells);

struct SignBias {
    // Separate intercept-plus-one-regressor tests, in sign/negative/positive order.
    std::array<TestStatistic, 3> individual;
    TestStatistic joint;  // n R^2, chi-square(3), all three regressors together
};
// Regress z_t^2 on lagged I(e<0), I(e<0)e and I(e>=0)e (Engle-Ng).
SignBias sign_bias(std::span<const double> standardized, std::span<const double> residual);
// Engle (1982): auxiliary OLS of z_t^2 on an intercept and q lagged squares.
// Uses n_aux R^2 ~ chi-square(q), with n_aux = rows - q.
TestStatistic arch_lm(std::span<const double> standardized, std::size_t lags);

struct Berkowitz {
    TestStatistic test;
    double mean = std::numeric_limits<double>::quiet_NaN(); // AR intercept
    double variance = std::numeric_limits<double>::quiet_NaN(); // innovation variance
    double ar1 = std::numeric_limits<double>::quiet_NaN();
};
// Conditional Gaussian AR(1) likelihood, conditioning both hypotheses on row 0.
// Tests intercept=0, innovation variance=1, ar1=0; no PIT endpoint clipping.
Berkowitz berkowitz(std::span<const double> pit);

struct LiMak {
    TestStatistic test;
    std::vector<double> autocorrelations;
    std::vector<double> covariance; // asymptotic covariance of sqrt(n) correlations
    std::vector<double> derivative; // lags x parameter dimension, D = -X
};
// v_t=z_t^2-1, m_t,k=v_t v_(t-k), b_t=d log h_t, a_t=(d e_t)/sqrt(h_t).
// d v_t = 2 z_t a_t - z_t^2 b_t, hence D_k=E[d m_t,k]=-E[v_(t-k)b_t]
// under centered iid innovations and predictable mean/variance sensitivities.
// The direct mean derivative has zero expectation; its score cross-covariance
// need not vanish for skew innovations, so mean parameters stay in A and psi.
// In Gaussian QMLE, psi_t=z_t a_t-v_t b_t/2. With kappa=E[v_t^2],
// mu3=E[z_t^3], Y_k=E[v_(t-k)a_t], K_k=E[m_t,k psi_t']=mu3 Y_k-kappa X_k/2.
// With negative-log scores psi, A=mean Hessian, estimator IF=-A^-1 psi:
//   w_t=m_t-D A^-1 psi_t, Sigma=mean[(w_t-mean w)(w_t-mean w)'].
// Equivalently Sigma=Omega+D A^-1 B A^-T D'-K A^-T D'-D A^-1 K',
// B=E[psi psi'], Omega=E[m m']. The sample covariance estimates every term.
// Q=n c' Sigma^-1 c ~ chi-square(lags), c=mean m. This uses the empirical
// sandwich, including moment-score cross terms, for BOTH MLE and QMLE.
// QMLE uses the Gaussian stage only, valid when density_free_recursion is true.
// This is exactly the first stage used by staged QMLE fits; shapes do not affect z.
// Requires regular information conditional on the active face and a finite
// innovation fourth moment. Closed-face normals are held fixed in the correction.
LiMak li_mak(Jet& jet, std::span<const double> theta, std::size_t lags,
             Estimator estimator = Estimator::mle, bool estimated = true);

enum class AdequacyStatus { adequate, rejected, adequacy_unresolved };
struct AdequacyCheck {
    std::string name;
    TestStatistic test;
    bool required = true, calibrated = true, holm_reject = false;
};
struct Adequacy {
    std::vector<AdequacyCheck> checks;
    AdequacyStatus status = AdequacyStatus::adequacy_unresolved;
    double level = 0.01;
    std::string volatility_gum; // empty when inapplicable
};
// Required: residual Ljung-Box and corrected Li-Mak at min(10,n/5),
// min(20,n/5), deduplicated; joint Engle-Ng; ARCH-LM for constant variance.
// Holm applies only to required checks. In-sample Berkowitz/DGT are reported
// with calibrated=false and never constrain adequacy.
Adequacy adequacy(Jet&, std::span<const double> theta, double level = 0.01,
                  Estimator = Estimator::mle);

struct Diagnostics {
    std::vector<double> pit;
    TestStatistic cell;
    std::array<Portmanteau, 4> pit_moments;
    Portmanteau residual, pit_normal;
    SignBias asymmetry;
    Berkowitz density;
    LiMak squared_residual;
};
// PIT reference p-values treat the fitted density as fixed. They are nominal
// for held-out forecasts; in-sample use requires estimation-aware calibration.
Diagnostics diagnostics(Jet&, std::span<const double> theta, std::size_t cells, std::size_t lags,
                         std::size_t df_adjustment, Estimator = Estimator::mle, bool estimated = true,
                         std::size_t pit_df_adjustment = 0);

struct DieboldMariano {
    TestStatistic test; // Student t_(n-1), two-sided, after HLN correction
    double long_run_variance = std::numeric_limits<double>::quiet_NaN();
    std::size_t bandwidth = 0;
};
// Bartlett Newey-West, default bandwidth=floor(n^(1/3)), at least horizon-1.
// HLN multiplier sqrt((n+1-2h+h(h-1)/n)/n). Inputs stay aligned; no row dropping.
DieboldMariano diebold_mariano(std::span<const double> loss_a, std::span<const double> loss_b,
                               std::size_t horizon = 1, int bandwidth = -1);

enum class MCSStatistic { maximum, range };
std::size_t mcs_block_length(std::size_t rows, int requested = -1);
struct ModelConfidenceSet {
    std::vector<std::size_t> included, eliminated;
    std::vector<double> p_values; // monotone step-down p-values, every input column
    double final_p_value = 1.0;
};
// Circular bootstrap. uniforms is B x ceil(n/block_length), iid U[0,1).
// The caller owns the sole RNG (Python: numpy Generator(PCG64(seed))). One set
// of resamples is reused at every elimination stage. Degenerate contrasts are
// handled exactly (identical losses tie; a constant nonzero gap has infinite t).
ModelConfidenceSet model_confidence_set(std::span<const double> losses, std::size_t rows,
                                        std::size_t models, std::span<const double> uniforms,
                                        std::size_t replications, std::size_t block_length,
                                        double alpha = 0.05,
                                        MCSStatistic statistic = MCSStatistic::maximum);

struct InformationCriteria { double aic, bic, hqic; };
InformationCriteria information_criteria(double log_likelihood, std::size_t parameters, std::size_t rows);
std::vector<double> location_scale_quantiles(double mean, double variance, const InnovationDensity&,
                                             std::span<const double> probabilities);

} // namespace scivolapi
