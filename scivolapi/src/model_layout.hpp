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
// The coefficients are a linear image of one simplex. GARCH: the simplex
// coordinates are the coefficients. GJR, with kappa = E[z^2 1(z<0)]: the simplex
// holds the shock contributions c+_i = (1 - kappa) alpha_i, c-_i = kappa (alpha_i +
// gamma_i) and beta_j, so persistence = sum c+ + sum c- + sum beta < 1 and the
// domain alpha_i >= 0, alpha_i + gamma_i >= 0 is covered (gamma may be negative).
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
    static constexpr auto unit = [](std::size_t) { return 1.0; };

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

    // Persistence weight of coefficient k (0-based within the coefficient block).
    [[nodiscard]] double weight(std::size_t k) const noexcept
    {
        const std::size_t index = coefficients + k;
        return index >= leverage && index < leverage + leverage_count ? kappa : 1.0;
    }

    // Simplex coordinates v -> coefficients, in place on the coefficient block.
    void from_simplex(double* c) const noexcept
    {
        for (std::size_t i = 0; i < leverage_count; ++i) {
            const double plus = c[i], minus = c[p + i];
            c[i] = plus / (1.0 - kappa);
            c[p + i] = minus / kappa - c[i];
        }
    }
    // Coefficients -> simplex coordinates (out may alias nothing).
    void to_simplex(const double* c, double* v) const noexcept
    {
        for (std::size_t k = 0; k < coefficient_count; ++k) v[k] = c[k];
        for (std::size_t i = 0; i < leverage_count; ++i) {
            v[i] = (1.0 - kappa) * c[i];
            v[p + i] = kappa * (c[i] + c[p + i]);
        }
    }
    // g -> M' g for the coefficient block (the map is linear: theta = M v).
    void transpose_apply(const double* g, double* out) const noexcept
    {
        for (std::size_t k = 0; k < coefficient_count; ++k) out[k] = g[k];
        for (std::size_t i = 0; i < leverage_count; ++i) {
            out[i] = (g[i] - g[p + i]) / (1.0 - kappa);
            out[p + i] = g[p + i] / kappa;
        }
    }

    // The scalar chart of parameter `index` (omega or a shape).
    [[nodiscard]] const ScalarChart& scalar(std::size_t index) const noexcept
    {
        return index == omega ? omega_chart : shape_chart[index - shapes];
    }

    void theta(const double* z, double* theta) const noexcept
    {
        theta[omega] = omega_chart.theta(z[omega]);
        SimplexChart::theta(z + coefficients, coefficient_count, unit, theta + coefficients);
        from_simplex(theta + coefficients);
        for (std::size_t i = 0; i < shape_count; ++i) theta[shapes + i] = shape_chart[i].theta(z[shapes + i]);
    }

    // `v` is scratch of coefficient_count values.
    [[nodiscard]] bool z(const double* theta, double* z, double* v) const noexcept
    {
        to_simplex(theta + coefficients, v);
        if (!omega_chart.z(theta[omega], z[omega]) || !SimplexChart::z(v, coefficient_count, unit, z + coefficients))
            return false;
        for (std::size_t i = 0; i < shape_count; ++i)
            if (!shape_chart[i].z(theta[shapes + i], z[shapes + i])) return false;
        return true;
    }

    // Dense Jacobian d theta / d z (dimension x dimension); `share` is scratch of
    // coefficient_count values. The coefficient rows are M times the simplex rows.
    void jacobian(const double* z, double* share, double* jac) const noexcept
    {
        for (std::size_t k = 0; k < dimension * dimension; ++k) jac[k] = 0.0;
        jac[omega * dimension + omega] = omega_chart.first(z[omega]);
        SimplexChart::shares(z + coefficients, coefficient_count, share);
        SimplexChart::jacobian(share, coefficient_count, unit, coefficients, dimension, jac);
        for (std::size_t i = 0; i < leverage_count; ++i)
            for (std::size_t j = 0; j < coefficient_count; ++j) {
                double* plus = jac + (coefficients + i) * dimension + coefficients + j;
                double* minus = jac + (coefficients + p + i) * dimension + coefficients + j;
                *plus /= 1.0 - kappa;
                *minus = *minus / kappa - *plus;
            }
        for (std::size_t i = shapes; i < dimension; ++i) jac[i * dimension + i] = scalar(i).first(z[i]);
    }

    // Adds sum_i g_i d^2 theta_i / dz dz' to `out`; `share` must hold the shares
    // computed by jacobian() at the same z, and `v` is scratch of coefficient_count.
    // With theta = M v linear, the coefficient curvature is the simplex's with M' g.
    void curvature(const double* z, const double* share, const double* g, double* v, double* out) const noexcept
    {
        out[omega * dimension + omega] += g[omega] * omega_chart.second(z[omega]);
        transpose_apply(g + coefficients, v);
        SimplexChart::curvature(share, v, coefficient_count, unit, coefficients, dimension, out);
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
