#include <scivolapi/simulate.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

#include "model_layout.hpp"
#include "paths.hpp"

namespace scivolapi {

namespace {

using detail::Layout;
using detail::PathStatus;

void require(bool ok, const char* message)
{
    if (!ok) throw std::invalid_argument(message);
}

bool finite(std::span<const double> v)
{
    for (double x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// The recursions after a run: rows [0, observed) filtered, then the last path.
struct Run {
    std::vector<double> e, h, state, scale;
    std::size_t observed = 0;
    Simulation paths;
};

// Filters `observed` rows of y after `lags` lag rows, then generates `paths`
// continuations of `horizon` rows. x covers lags + observed + horizon rows.
Run run(const Model& model, std::span<const double> theta, std::vector<double> y, std::span<const double> x,
        double seed, std::size_t lags, std::size_t horizon, std::span<const double> z, std::size_t paths)
{
    const ModelSpec& spec = model.spec();
    const Layout layout(spec);
    const std::size_t observed = y.size() - lags, rows = observed + horizon;
    require(theta.size() == model.dimension() && finite(theta), "theta must be finite with the model's dimension");
    require(z.size() == paths * horizon && finite(z), "z must hold paths x horizon finite innovations");
    require(x.size() == (lags + rows) * spec.mean.exog.size(), "exogenous regressors must cover every row of the horizon");
    require(model.evaluable(theta), "theta is outside the region where the model is defined");

    Run r;
    r.observed = observed;
    r.e.assign(rows, 0.0);
    r.h.assign(rows, 0.0);
    r.state.assign(rows, 0.0);
    r.scale.assign(rows, 0.0);
    std::vector<double> e2(rows, 0.0);
    y.resize(lags + rows, 0.0);
    Simulation& s = r.paths;
    s.paths = paths;
    s.horizon = horizon;
    s.y.assign(paths * horizon, 0.0);
    s.mean.assign(paths * horizon, 0.0);
    s.residual.assign(paths * horizon, 0.0);
    s.variance.assign(paths * horizon, 0.0);

    const detail::PathInput in{spec, layout, theta.data(), seed, lags, observed, horizon,
                               x.empty() ? nullptr : x.data(), z.data(), paths};
    detail::PathBuffers b{y.data(), r.e.data(), e2.data(), r.h.data(), r.state.data(), r.scale.data(),
                          s.y.data(), s.mean.data(), s.residual.data(), s.variance.data()};
    switch (detail::path_runner_for(spec)(in, b)) {
    case PathStatus::ok: break;
    case PathStatus::not_evaluable: throw std::domain_error("the model cannot be evaluated at theta");
    case PathStatus::numerical_range: throw std::domain_error("a path left the range representable in double precision");
    }
    return r;
}

// Sample rows and exogenous rows extended by the horizon's.
std::vector<double> joined(std::span<const double> a, std::span<const double> b)
{
    std::vector<double> out(a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

// psi weights of y in the residuals: y_t = ... + sum_j psi_j e_{t-j}, from the
// AR polynomial (AR lags plus the HAR averages' implied lags) and the MA terms.
std::vector<double> psi_weights(const ModelSpec& spec, const Layout& layout, const double* theta, std::size_t horizon)
{
    const MeanSpec& mean = spec.mean;
    std::vector<double> ar(mean.conditioning(), 0.0), psi(horizon, 0.0);
    for (std::size_t i = 0; i < mean.ar; ++i) ar[i] += theta[layout.ar + i];
    for (std::size_t k = 0; k < mean.har.size(); ++k)
        for (std::size_t i = 0; i < mean.har[k]; ++i) ar[i] += theta[layout.har + k] / static_cast<double>(mean.har[k]);
    for (std::size_t n = 0; n < horizon; ++n) {
        double v = n == 0 ? 1.0 : n <= mean.ma ? theta[layout.ma + n - 1] : 0.0;
        for (std::size_t i = 1; i <= std::min(n, ar.size()); ++i) v += ar[i - 1] * psi[n - i];
        psi[n] = v;
    }
    return psi;
}

}  // namespace

Simulation simulate(const Model& model, std::span<const double> theta, std::span<const double> y,
                    std::span<const double> x, std::span<const double> future_x, std::span<const double> z,
                    std::size_t paths)
{
    const Sample sample = model.prepare(y, x);
    return run(model, theta, {y.begin(), y.end()}, joined(x, future_x), sample.variance_seed, sample.first_scored,
               paths ? z.size() / paths : 0, z, paths)
        .paths;
}

Simulation simulate_series(const Model& model, std::span<const double> theta, std::span<const double> x,
                           std::span<const double> z, std::size_t paths)
{
    const ModelSpec& spec = model.spec();
    const Layout layout(spec);
    require(theta.size() == model.dimension() && model.feasible(theta), "theta must lie in the model's domain");
    require(paths > 0 && z.size() % paths == 0, "z must hold paths x horizon innovations");
    const std::size_t horizon = z.size() / paths, exog = spec.mean.exog.size(), lags = spec.mean.conditioning();
    require(x.size() == horizon * exog, "exogenous regressors must cover every row of the horizon");

    // Unconditional mean: (c + xbar' b) / (1 - sum of the AR polynomial).
    double level = spec.mean.constant ? theta[0] : 0.0, ar = 0.0;
    for (std::size_t i = 0; i < exog; ++i) {
        double total = 0.0;
        for (std::size_t t = 0; t < horizon; ++t) total += x[t * exog + i];
        level += theta[layout.exog + i] * total / static_cast<double>(horizon);
    }
    for (std::size_t i = 0; i < spec.mean.ar; ++i) ar += theta[layout.ar + i];
    for (std::size_t k = 0; k < spec.mean.har.size(); ++k) ar += theta[layout.har + k];
    level /= 1.0 - ar;

    // Unconditional variance. A nonstationary recursion (admitted by an
    // integrated or absent stationarity declaration) has none to start from.
    const double seed = layout.family->simulation_seed(layout, theta.data());
    if (!std::isfinite(seed) || !(seed > 0.0)) throw std::domain_error("the unconditional variance is not representable");

    std::vector<double> lagged_x(lags * exog, 0.0);
    lagged_x.insert(lagged_x.end(), x.begin(), x.end());
    return run(model, theta, std::vector<double>(lags, level), lagged_x, seed, lags, horizon, z, paths).paths;
}

Forecast forecast(const Model& model, std::span<const double> theta, std::span<const double> y,
                  std::span<const double> x, std::span<const double> future_x, std::size_t horizon,
                  std::span<const double> z, std::size_t paths)
{
    require(horizon > 0, "the horizon must be positive");
    const ModelSpec& spec = model.spec();
    const Layout layout(spec);
    const Sample sample = model.prepare(y, x);
    const std::vector<double> rows = joined(x, future_x);
    const std::vector<double> zero(horizon, 0.0);
    // With zero innovations the generated mean is E[y_{T+k}] (the mean is linear);
    // the first variance is the exact one-step forecast.
    const Run filtered = run(model, theta, {y.begin(), y.end()}, rows, sample.variance_seed, sample.first_scored,
                             horizon, zero, 1);

    Forecast f;
    f.mean = filtered.paths.mean;
    f.residual_variance.assign(horizon, 0.0);
    // Exact horizons from the model's closed form (one step ahead it is the
    // filter's next variance); beyond them, averages over the supplied paths.
    const std::size_t exact = std::min(horizon, model.capabilities().exact_variance_horizon);
    const detail::FilteredPaths past{filtered.e, filtered.h, filtered.state, filtered.observed,
                                     filtered.paths.variance[0]};
    layout.family->variance_forecast(layout, theta.data(), past, {f.residual_variance.data(), exact});
    if (horizon > exact) {
        require(paths > 0 && z.size() == paths * horizon,
                "variance forecasts beyond the exact horizons need paths x horizon innovations");
        const Simulation s = run(model, theta, {y.begin(), y.end()}, rows, sample.variance_seed, sample.first_scored,
                                 horizon, z, paths)
                                 .paths;
        for (std::size_t k = exact; k < horizon; ++k) {
            double total = 0.0;
            for (std::size_t p = 0; p < paths; ++p) total += s.variance[p * horizon + k];
            f.residual_variance[k] = total / static_cast<double>(paths);
        }
        f.simulated = true;
    }

    const std::vector<double> psi = psi_weights(spec, layout, theta.data(), horizon);
    f.variance.assign(horizon, 0.0);
    for (std::size_t k = 0; k < horizon; ++k)
        for (std::size_t j = 0; j <= k; ++j) f.variance[k] += psi[j] * psi[j] * f.residual_variance[k - j];
    return f;
}

}  // namespace scivolapi
