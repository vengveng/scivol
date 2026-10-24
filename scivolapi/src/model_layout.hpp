#pragma once

#include <cstddef>

#include <scivolapi/model.hpp>

#include "map/charts.hpp"

namespace scivolapi::detail {

// Block structure of theta for a spec, and the chart that covers each block.
// The single definition used by Model (maps, bounds, starts) and Jet (z-space).
struct Layout {
    std::size_t p = 0, q = 0;
    std::size_t omega = 0;         // index of omega
    std::size_t coefficients = 1;  // first ARCH coefficient
    std::size_t coefficient_count = 0;
    std::size_t shapes = 0;        // first shape parameter
    std::size_t shape_count = 0;
    std::size_t dimension = 0;

    explicit Layout(const ModelSpec& spec) noexcept
        : p(spec.p), q(spec.q), coefficient_count(spec.p + spec.q)
    {
        shapes = coefficients + coefficient_count;
        shape_count = 0;  // Normal
        dimension = shapes + shape_count;
    }

    void theta(const double* z, double* theta) const noexcept
    {
        theta[omega] = PositiveChart::theta(z[omega]);
        SimplexChart::theta(z + coefficients, coefficient_count, theta + coefficients);
    }

    [[nodiscard]] bool z(const double* theta, double* z) const noexcept
    {
        return PositiveChart::z(theta[omega], z[omega])
            && SimplexChart::z(theta + coefficients, coefficient_count, z + coefficients);
    }

    // Dense Jacobian d theta / d z (dimension x dimension); `share` is scratch of
    // coefficient_count values.
    void jacobian(const double* z, double* share, double* jac) const noexcept
    {
        for (std::size_t k = 0; k < dimension * dimension; ++k) jac[k] = 0.0;
        PositiveChart::jacobian(z[omega], omega, dimension, jac);
        SimplexChart::shares(z + coefficients, coefficient_count, share);
        SimplexChart::jacobian(share, coefficient_count, coefficients, dimension, jac);
    }

    // Adds sum_i g_i d^2 theta_i / dz dz' to `out`; `share` must hold the shares
    // computed by jacobian() at the same z.
    void curvature(const double* z, const double* share, const double* g, double* out) const noexcept
    {
        PositiveChart::curvature(z[omega], g[omega], omega, dimension, out);
        SimplexChart::curvature(share, g + coefficients, coefficient_count, coefficients, dimension, out);
    }
};

}  // namespace scivolapi::detail
