#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "model.hpp"

namespace scivolapi {

// Generated paths, each `horizon` rows: paths x horizon, row-major.
struct Simulation {
    std::size_t paths = 0, horizon = 0;
    std::vector<double> y, mean, residual, variance;
};

// Continues the model at theta past the end of the sample (y, x) along `paths`
// paths of standardized innovations z (paths x horizon), which the caller draws
// from the model's density at theta; e = sqrt(h) z. future_x holds the exogenous
// rows of the horizon (horizon x exog). The sample is filtered exactly as the
// likelihood conditions it. Throws std::invalid_argument for unusable input and
// std::domain_error when theta or a path leaves the representable range.
Simulation simulate(const Model& model, std::span<const double> theta, std::span<const double> y,
                    std::span<const double> x, std::span<const double> future_x, std::span<const double> z,
                    std::size_t paths);

// A series from the model alone (theta feasible), started at its stationary
// values: the mean's lags at the unconditional mean (exogenous columns at their
// average over x), the first variance at the unconditional variance (EGARCH: at
// exp of the unconditional mean of log h). x: horizon x exog.
Simulation simulate_series(const Model& model, std::span<const double> theta, std::span<const double> x,
                           std::span<const double> z, std::size_t paths);

struct Forecast {
    std::vector<double> mean;               // E[y_{T+k} | y_1..T]
    std::vector<double> residual_variance;  // E[h_{T+k} | y_1..T] = E[e_{T+k}^2 | y_1..T]
    std::vector<double> variance;           // Var[y_{T+k} | y_1..T] = sum_j psi_j^2 E[h_{T+k-j}]
    bool simulated = false;                 // residual variances beyond one step are path averages
};

// Forecasts for k = 1..horizon after the sample (y, x). Exact for the mean, for
// its variance given the residual variances, and for the residual variances of
// GARCH and GJR at every horizon, EGARCH at every horizon (a closed form in the
// density's exponential moments; std::domain_error where those are infinite) and
// APARCH one step ahead. APARCH beyond one step needs z (paths x horizon
// standardized draws); its residual variances are then averages over the paths.
Forecast forecast(const Model& model, std::span<const double> theta, std::span<const double> y,
                  std::span<const double> x, std::span<const double> future_x, std::size_t horizon,
                  std::span<const double> z = {}, std::size_t paths = 0);

}  // namespace scivolapi
