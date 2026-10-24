#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <span>

#include <scivolapi/model.hpp>

#include "density/domain.hpp"
#include "map/charts.hpp"

namespace scivolapi::detail {

// Block structure of theta for a spec, and the chart that covers each block:
//   GARCH  [omega | alpha_1..p, beta_1..q | shapes]
//   GJR    [omega | alpha_1..p, gamma_1..p, beta_1..q | shapes]
// The coefficients share one weighted simplex: sum w_k theta_k < 1 with w = 1,
// except GJR's gamma_i, weighted by kappa = E[z^2 1(z<0)].
// The single definition used by Model (maps, bounds, starts) and Jet (z-space).
struct Layout {
    std::size_t p = 0, q = 0;
    std::size_t omega = 0;         // index of omega
    std::size_t coefficients = 1;  // first ARCH coefficient
    std::size_t coefficient_count = 0;
    std::size_t shapes = 0;        // first shape parameter
    std::size_t shape_count = 0;
    std::size_t dimension = 0;
    std::size_t leverage = 0, leverage_count = 0;  // GJR gamma block (within the coefficients)
    double kappa = 0.5;                            // E[z^2 1(z<0)] for symmetric densities
    std::span<const ShapeDomain> shape_domain;
    ScalarChart omega_chart = ScalarChart::for_domain(0.0, std::numeric_limits<double>::infinity());
    std::array<ScalarChart, max_shapes> shape_chart{};

    explicit Layout(const ModelSpec& spec) noexcept
        : p(spec.p), q(spec.q), coefficient_count(spec.p + spec.q), shape_domain(shape_domains(spec.density))
    {
        if (spec.volatility == VolatilityKind::gjr) {
            leverage = coefficients + spec.p;
            leverage_count = spec.p;
            coefficient_count += spec.p;
        }
        shapes = coefficients + coefficient_count;
        shape_count = shape_domain.size();
        dimension = shapes + shape_count;
        for (std::size_t i = 0; i < shape_count; ++i)
            shape_chart[i] = ScalarChart::for_domain(shape_domain[i].lower, shape_domain[i].upper);
    }

    // Weight of coefficient k (0-based within the coefficient block).
    [[nodiscard]] double weight(std::size_t k) const noexcept
    {
        const std::size_t index = coefficients + k;
        return index >= leverage && index < leverage + leverage_count ? kappa : 1.0;
    }
    [[nodiscard]] auto weights() const noexcept
    {
        return [this](std::size_t k) { return weight(k); };
    }

    // The scalar chart of parameter `index` (omega or a shape).
    [[nodiscard]] const ScalarChart& scalar(std::size_t index) const noexcept
    {
        return index == omega ? omega_chart : shape_chart[index - shapes];
    }

    void theta(const double* z, double* theta) const noexcept
    {
        theta[omega] = omega_chart.theta(z[omega]);
        SimplexChart::theta(z + coefficients, coefficient_count, weights(), theta + coefficients);
        for (std::size_t i = 0; i < shape_count; ++i) theta[shapes + i] = shape_chart[i].theta(z[shapes + i]);
    }

    [[nodiscard]] bool z(const double* theta, double* z) const noexcept
    {
        if (!omega_chart.z(theta[omega], z[omega])
            || !SimplexChart::z(theta + coefficients, coefficient_count, weights(), z + coefficients))
            return false;
        for (std::size_t i = 0; i < shape_count; ++i)
            if (!shape_chart[i].z(theta[shapes + i], z[shapes + i])) return false;
        return true;
    }

    // Dense Jacobian d theta / d z (dimension x dimension); `share` is scratch of
    // coefficient_count values.
    void jacobian(const double* z, double* share, double* jac) const noexcept
    {
        for (std::size_t k = 0; k < dimension * dimension; ++k) jac[k] = 0.0;
        jac[omega * dimension + omega] = omega_chart.first(z[omega]);
        SimplexChart::shares(z + coefficients, coefficient_count, share);
        SimplexChart::jacobian(share, coefficient_count, weights(), coefficients, dimension, jac);
        for (std::size_t i = shapes; i < dimension; ++i) jac[i * dimension + i] = scalar(i).first(z[i]);
    }

    // Adds sum_i g_i d^2 theta_i / dz dz' to `out`; `share` must hold the shares
    // computed by jacobian() at the same z.
    void curvature(const double* z, const double* share, const double* g, double* out) const noexcept
    {
        out[omega * dimension + omega] += g[omega] * omega_chart.second(z[omega]);
        SimplexChart::curvature(share, g + coefficients, coefficient_count, weights(), coefficients, dimension, out);
        for (std::size_t i = shapes; i < dimension; ++i) out[i * dimension + i] += g[i] * scalar(i).second(z[i]);
    }

    // sum w_k theta_k over the coefficients: the persistence.
    [[nodiscard]] double persistence(const double* theta) const noexcept
    {
        double total = 0.0;
        for (std::size_t k = 0; k < coefficient_count; ++k) total += weight(k) * theta[coefficients + k];
        return total;
    }

    // Shape parameters strictly inside their open domain.
    [[nodiscard]] bool shapes_inside(const double* theta) const noexcept
    {
        for (std::size_t i = 0; i < shape_count; ++i) {
            const double v = theta[shapes + i];
            if (!(v > shape_domain[i].lower && v < shape_domain[i].upper) || std::isnan(v)) return false;
        }
        return true;
    }
};

}  // namespace scivolapi::detail
