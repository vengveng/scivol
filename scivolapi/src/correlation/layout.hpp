#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <scivolapi/correlation.hpp>

#include "../map/charts.hpp"
#include "../density/domain.hpp"

namespace scivolapi::detail::correlation {

struct Layout {
    static constexpr std::size_t max_order = 8;
    const std::size_t p, q, coefficients, dimension;
    const bool student;
    const ShapeDomain tail = domain_detail::student_t[0];
    const ScalarChart eta = ScalarChart::for_domain(tail.lower, tail.upper, tail.closed_lower);

    explicit Layout(const CorrelationSpec& spec)
        : p(spec.p), q(spec.q), coefficients(p + q),
          dimension(coefficients + (spec.density == CorrelationDensity::student_t ? 1u : 0u)),
          student(spec.density == CorrelationDensity::student_t)
    {
    }

    static double unit(std::size_t) noexcept { return 1.0; }

    [[nodiscard]] bool theta(const double* z, double* out) const noexcept
    {
        for (std::size_t i = 0; i < dimension; ++i)
            if (!std::isfinite(z[i])) return false;
        SimplexChart::theta(z, coefficients, unit, out);
        if (student) out[coefficients] = eta.theta(z[coefficients]);
        return !student || std::isfinite(out[coefficients]);
    }

    [[nodiscard]] bool z(const double* theta, double* out) const noexcept
    {
        if (!SimplexChart::z(theta, coefficients, unit, out)) return false;
        return !student || eta.z(theta[coefficients], out[coefficients]);
    }

    void jacobian(const double* z, double* out) const noexcept
    {
        double share[2 * max_order];
        SimplexChart::shares(z, coefficients, share);
        std::fill_n(out, dimension * dimension, 0.0);
        SimplexChart::jacobian(share, coefficients, unit, 0, dimension, out);
        if (student) out[coefficients * dimension + coefficients] = eta.first(z[coefficients]);
    }

    void curvature(const double* z, const double* g, double* out) const noexcept
    {
        double share[2 * max_order];
        SimplexChart::shares(z, coefficients, share);
        SimplexChart::curvature(share, g, coefficients, unit, 0, dimension, out);
        if (student) out[coefficients * dimension + coefficients] += g[coefficients] * eta.second(z[coefficients]);
    }
};

}  // namespace scivolapi::detail::correlation
