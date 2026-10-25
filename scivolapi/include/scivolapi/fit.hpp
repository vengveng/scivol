#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "correlation.hpp"
#include "model.hpp"

namespace scivolapi {

// Which formulation the optimizer solves. `theta` and `z` are independent
// paths; `automatic` is the named hybrid: a z-space solve, then exact-Newton
// refinement in theta. Every phase a fit ran is listed in FitResult::phases.
enum class Coordinates { automatic, z, theta };

// How theta is estimated and how its covariance is formed.
//   mle:  maximum likelihood; covariance = inverse observed information.
//   qmle: Gaussian quasi-maximum likelihood for the mean and variance parameters,
//         with the sandwich covariance H^-1 OPG H^-1. With a non-Normal density
//         the fit is staged: the shape parameters are then fitted by maximum
//         likelihood with the others held at their Gaussian estimates, and the
//         covariance is the two-step sandwich of the stacked score equations.
enum class Estimator { mle, qmle };

struct FitOptions {
    Estimator estimator = Estimator::mle;
    Coordinates coordinates = Coordinates::automatic;
    double accuracy = 1e-10;        // solver stopping accuracy
    int maximum_iterations = 1000;  // total budget across phases
    std::vector<double> start;      // optional theta; empty = the model's start
};

enum class Termination { converged, iteration_limit, line_search_failed, failed };

enum class Inference {
    available,
    not_feasible,          // the estimate is outside the theoretical domain
    not_stationary,        // first-order conditions not met; the fit is uncertified
    at_boundary,           // a bound or constraint is active; Hessian SEs do not apply
    not_positive_definite, // the observed information is not positive definite
    ill_conditioned,       // positive definite, but too poorly conditioned to invert reliably
};

struct FitPhase {
    std::string name;  // e.g. "z:slsqp", "theta:newton"
    Termination termination = Termination::failed;
    int iterations = 0;
    std::size_t value_requests = 0, gradient_requests = 0, hessian_requests = 0;
    double value = std::numeric_limits<double>::quiet_NaN();  // F/n_scored on unit-scaled data
};

// The only result type, built only by the finalizer, for every model.
// All quantities are in the units of the data passed to fit().
struct FitResult {
    bool estimate_available = false;  // false: no finite, feasible estimate (a fit failure)
    std::vector<std::string> names;
    std::vector<double> theta;
    double log_likelihood = std::numeric_limits<double>::quiet_NaN();  // -F
    std::vector<double> gradient;    // dF/dtheta
    std::vector<double> hessian;     // d2F/dtheta2 (dense, row-major)
    Estimator estimator = Estimator::mle;
    // Covariance of the estimator: inverse information (mle) or the sandwich
    // (qmle). Empty unless inference == available.
    std::vector<double> covariance;
    std::vector<double> std_errors;
    // Sum over scored rows of s_t s_t', s_t the row's score (stacked over both
    // stages for a staged qmle fit), and the sandwich covariance built from it.
    std::vector<double> opg;
    std::vector<double> robust_covariance;  // empty unless inference == available
    std::vector<double> robust_std_errors;
    std::vector<double> residual, variance, innovation;
    std::size_t observations = 0, scored = 0;
    double data_scale = 1.0;  // internal unit scaling used during the solve

    bool feasible = false;
    Termination termination = Termination::failed;  // of the phase that produced theta
    double kkt_residual = std::numeric_limits<double>::infinity();
    double kkt_tolerance = 0.0;
    bool stationary = false;
    Inference inference = Inference::not_stationary;
    bool success = false;  // finite, feasible, solver converged, and stationary

    std::vector<FitPhase> phases;
    std::vector<std::string> warnings;
};

// x: exogenous regressors (rows x exog, row-major), empty without them.
FitResult fit(const Model& model, std::span<const double> y, std::span<const double> x,
              const FitOptions& options = {});
inline FitResult fit(const Model& model, std::span<const double> y, const FitOptions& options = {})
{
    return fit(model, y, {}, options);
}

// A correlation model on standardized residuals eps (rows x series, row-major)
// with target matrix `target` (series x series; empty: eps'eps / rows). The
// marginal fits that produced eps are a first stage: this is the second stage of
// a staged estimator, and its covariance treats eps and the target as given.
FitResult fit(const CorrelationModel& model, std::span<const double> eps, std::size_t series,
              std::span<const double> target, const FitOptions& options = {});

}  // namespace scivolapi
