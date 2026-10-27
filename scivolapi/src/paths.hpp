#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <scivolapi/model.hpp>

#include "mean/regressors.hpp"
#include "model_layout.hpp"
#include "volatility/aparch.hpp"
#include "volatility/constant.hpp"
#include "volatility/egarch.hpp"
#include "volatility/garch.hpp"
#include "volatility/volatility.hpp"

// The model's recursions run forward: first over observed rows (the filter, the
// same recursion and conditioning as the likelihood), then over future rows whose
// residuals are generated from standardized innovations, e = sqrt(h) z. Each
// path continues from the same filtered state. The volatility recursions are the
// likelihood's own classes, stepped at order 0.
namespace scivolapi::detail {

// Rows are absolute: `lags` rows that only supply the mean's lags, then
// `observed` rows with given y, then `future` generated rows.
struct PathInput {
    const ModelSpec& spec;
    const Layout& layout;
    const double* theta;
    double seed;               // variance of the first row after the lags
    std::size_t lags, observed, future;
    const double* x;           // (lags + observed + future) x exog, or null without exogenous regressors
    const double* z;           // paths x future standardized innovations
    std::size_t paths;
};

// Caller-owned buffers. y holds lags + observed + future rows (the first two
// parts given); e, e2, h, state hold observed + future rows. The path outputs are
// paths x future.
struct PathBuffers {
    double* y;
    double* e;
    double* e2;
    double* h;
    double* state;
    double* scale;
    double* path_y;
    double* path_mean;
    double* path_residual;
    double* path_variance;
};

enum class PathStatus { ok, not_evaluable, numerical_range };

template<class Volatility>
PathStatus run_paths(const PathInput& in, PathBuffers& b)
{
    const std::size_t m = in.layout.mean, k = in.spec.mean.regressors(), q = in.spec.mean.ma;
    const double* theta = in.theta;
    Volatility volatility(in.spec, 0);
    if (!volatility.evaluable(theta + m) || !volatility.prepare(theta + m, 0)) return PathStatus::not_evaluable;
    VolatilityWork w{};
    w.e = b.e;
    w.e2 = b.e2;
    w.h = b.h;
    w.state = b.state;
    w.scale = b.scale;
    Sample seed;
    seed.variance_seed = in.seed;
    VolatilityStep step;

    // Conditional mean of row s (after the lags): r_s' b + sum_j ma_j e_{s-j}.
    double row[Layout::max_mean];
    auto mean = [&](std::size_t s) {
        regressor_row(in.spec.mean, b.y, in.x, in.lags + s, row);
        double mu = 0.0;
        for (std::size_t i = 0; i < k; ++i) mu += row[i] * theta[i];
        for (std::size_t j = 1, n = std::min(q, s); j <= n; ++j) mu += theta[k + j - 1] * b.e[s - j];
        return mu;
    };

    for (std::size_t s = 0; s < in.observed; ++s) {
        const double e = b.y[in.lags + s] - mean(s);
        if (!std::isfinite(e)) return PathStatus::numerical_range;
        b.e[s] = e;
        b.e2[s] = e * e;
        if (!volatility.step(theta + m, seed, s, 0, w, step)) return PathStatus::numerical_range;
    }
    for (std::size_t p = 0; p < in.paths; ++p) {
        const double* z = in.z + p * in.future;
        const std::size_t out = p * in.future;
        for (std::size_t f = 0; f < in.future; ++f) {
            const std::size_t s = in.observed + f;
            if (!volatility.step(theta + m, seed, s, 0, w, step)) return PathStatus::numerical_range;
            const double mu = mean(s), e = std::sqrt(step.h) * z[f], y = mu + e;
            if (!std::isfinite(y)) return PathStatus::numerical_range;
            b.e[s] = e;
            b.e2[s] = e * e;
            b.y[in.lags + s] = y;
            b.path_y[out + f] = y;
            b.path_mean[out + f] = mu;
            b.path_residual[out + f] = e;
            b.path_variance[out + f] = step.h;
        }
    }
    return PathStatus::ok;
}

using PathRunner = PathStatus (*)(const PathInput&, PathBuffers&);

template<class Density>
PathRunner path_runner_with(VolatilityKind kind) noexcept
{
    switch (kind) {
    case VolatilityKind::garch: return run_paths<GarchVolatility<false>>;
    case VolatilityKind::gjr: return run_paths<GarchVolatility<true>>;
    case VolatilityKind::egarch: return run_paths<EgarchVolatility<Density>>;
    case VolatilityKind::aparch: return run_paths<AparchVolatility<>>;
    case VolatilityKind::constant: return run_paths<ConstantVolatility<>>;
    }
    return nullptr;
}

PathRunner path_runner_normal(VolatilityKind);
PathRunner path_runner_student_t(VolatilityKind);
PathRunner path_runner_skew_t(VolatilityKind);
PathRunner path_runner_ged(VolatilityKind);
PathRunner path_runner_generalized_hyperbolic(VolatilityKind);
PathRunner path_runner_nig(VolatilityKind);
PathRunner path_runner_hyperbolic(VolatilityKind);
PathRunner path_runner_variance_gamma(VolatilityKind);

inline PathRunner path_runner_for(const ModelSpec& spec) noexcept
{
    switch (spec.density) {
    case DensityKind::normal: return path_runner_normal(spec.volatility);
    case DensityKind::student_t: return path_runner_student_t(spec.volatility);
    case DensityKind::skew_t: return path_runner_skew_t(spec.volatility);
    case DensityKind::ged: return path_runner_ged(spec.volatility);
    case DensityKind::generalized_hyperbolic: return path_runner_generalized_hyperbolic(spec.volatility);
    case DensityKind::nig: return path_runner_nig(spec.volatility);
    case DensityKind::hyperbolic: return path_runner_hyperbolic(spec.volatility);
    case DensityKind::variance_gamma: return path_runner_variance_gamma(spec.volatility);
    }
    return nullptr;
}

}  // namespace scivolapi::detail
