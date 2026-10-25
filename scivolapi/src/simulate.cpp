#include <scivolapi/simulate.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

#include "density/exponential_moment.hpp"
#include "density/moments.hpp"
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
    std::vector<double> e, h, state;
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
    require(x.size() == (lags + rows) * spec.mean.exog, "exogenous regressors must cover every row of the horizon");
    require(model.evaluable(theta), "theta is outside the region where the model is defined");

    Run r;
    r.observed = observed;
    r.e.assign(rows, 0.0);
    r.h.assign(rows, 0.0);
    r.state.assign(rows, 0.0);
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
    detail::PathBuffers b{y.data(), r.e.data(), e2.data(), r.h.data(), r.state.data(),
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

// GARCH and GJR: E[h] follows the recursion with E[e^2] = E[h] and, for GJR,
// E[e^2 1(e < 0)] = kappa E[h], kappa = E[z^2 1(z < 0)].
void linear_variance(const ModelSpec& spec, const Layout& layout, const double* theta, const Run& r,
                     std::span<double> out)
{
    const bool gjr = spec.volatility == VolatilityKind::gjr;
    double kappa = 0.0;
    if (gjr) {
        detail::MomentResult m;
        if (!detail::negative_second_moment(spec.density, layout.shape_count ? theta + layout.shapes : nullptr, 0, m))
            throw std::domain_error("the density's negative second moment is not available at theta");
        kappa = m.value;
    }
    const std::size_t T = r.observed, p = spec.p, q = spec.q;
    std::vector<double> h(r.h.begin(), r.h.begin() + static_cast<std::ptrdiff_t>(T));
    h.resize(T + out.size());
    for (std::size_t f = 0; f < out.size(); ++f) {
        const std::size_t s = T + f;
        double v = theta[layout.omega];
        for (std::size_t i = 1; i <= std::min(p, s); ++i) {
            const std::size_t k = s - i;
            const double alpha = theta[layout.alpha + i - 1], gamma = gjr ? theta[layout.gamma + i - 1] : 0.0;
            if (k < T) v += (alpha + (r.e[k] < 0.0 ? gamma : 0.0)) * r.e[k] * r.e[k];
            else v += (alpha + gamma * kappa) * h[k];
        }
        for (std::size_t j = 1; j <= std::min(q, s); ++j) v += theta[layout.beta + j - 1] * h[s - j];
        h[s] = out[f] = v;
    }
}

// EGARCH, with x = log h: for future rows x_s = d_s + sum_{u >= T} (A_su |z_u| + B_su z_u)
// (the past is deterministic), so with independent z_u
//   E[h_s] = exp(d_s) prod_u M(A_su, B_su),  M(a, b) = E exp(a |z| + b z).
void egarch_variance(const ModelSpec& spec, const Layout& layout, const double* theta, const Run& r,
                     std::span<double> out)
{
    const double* shape = layout.shape_count ? theta + layout.shapes : nullptr;
    detail::MomentResult moment;
    if (!detail::absolute_moment(spec.density, shape, 0, moment))
        throw std::domain_error("the density's absolute moment is not available at theta");
    const double m = moment.value;
    const std::size_t T = r.observed, H = out.size(), p = spec.p, q = spec.q;
    auto z = [&](std::size_t k) { return r.e[k] * std::exp(-0.5 * r.state[k]); };
    // d over rows [0, T + H); A, B over future rows s and shock rows u (H x H, offset T).
    std::vector<double> d(r.state.begin(), r.state.begin() + static_cast<std::ptrdiff_t>(T)), a(H * H, 0.0), b(H * H, 0.0);
    d.resize(T + H);
    for (std::size_t f = 0; f < H; ++f) {
        const std::size_t s = T + f;
        double v = theta[layout.omega];
        for (std::size_t i = 1; i <= std::min(p, s); ++i) {
            const std::size_t k = s - i;
            const double alpha = theta[layout.alpha + i - 1], gamma = theta[layout.gamma + i - 1];
            if (k < T) {
                v += alpha * (std::abs(z(k)) - m) + gamma * z(k);
            } else {
                v -= alpha * m;
                a[f * H + (k - T)] += alpha;
                b[f * H + (k - T)] += gamma;
            }
        }
        for (std::size_t j = 1; j <= std::min(q, s); ++j) {
            const double beta = theta[layout.beta + j - 1];
            v += beta * d[s - j];
            if (f >= j)  // row s - j is a future row: its shocks feed through beta_j
                for (std::size_t u = 0; u < f - j; ++u) {
                    a[f * H + u] += beta * a[(f - j) * H + u];
                    b[f * H + u] += beta * b[(f - j) * H + u];
                }
        }
        d[s] = v;
        double log_value = v;
        for (std::size_t u = 0; u < f; ++u) {
            double factor = 0.0;
            switch (detail::exponential_moment(spec.density, shape, a[f * H + u], b[f * H + u], factor)) {
            case detail::ExpectationStatus::ok: break;
            case detail::ExpectationStatus::infinite:
                throw std::domain_error("the EGARCH variance forecast " + std::to_string(f + 1)
                                        + " steps ahead does not exist: E exp(a|z| + b z) is infinite under this density");
            default: throw std::domain_error("the density's exponential moment could not be computed accurately");
            }
            log_value += std::log(factor);
        }
        out[f] = std::exp(log_value);
        if (!std::isfinite(out[f]) || !(out[f] > 0.0))
            throw std::domain_error("the variance forecast is not representable in double precision");
    }
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
    const std::size_t horizon = z.size() / paths, exog = spec.mean.exog, lags = spec.mean.conditioning();
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

    // Unconditional variance.
    double seed = 0.0;
    if (spec.volatility == VolatilityKind::egarch) {
        double beta = 0.0;
        for (std::size_t j = 0; j < spec.q; ++j) beta += theta[layout.beta + j];
        seed = std::exp(theta[layout.omega] / (1.0 - beta));
    } else {
        double persistence = 0.0;
        if (!layout.persistence(theta.data(), 0, persistence, nullptr))
            throw std::domain_error("the persistence is not available at theta");
        const double level_power = theta[layout.omega] / (1.0 - persistence);
        seed = spec.volatility == VolatilityKind::aparch ? std::pow(level_power, 2.0 / theta[layout.delta]) : level_power;
    }
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
    std::span<double> closed(f.residual_variance.data(), exact);
    switch (spec.volatility) {
    case VolatilityKind::garch:
    case VolatilityKind::gjr:
    case VolatilityKind::constant: linear_variance(spec, layout, theta.data(), filtered, closed); break;
    case VolatilityKind::egarch: egarch_variance(spec, layout, theta.data(), filtered, closed); break;
    case VolatilityKind::aparch: closed[0] = filtered.paths.variance[0]; break;
    }
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
