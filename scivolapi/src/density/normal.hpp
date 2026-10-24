#pragma once

#include <array>
#include <cstddef>

#include "../detail/numeric.hpp"

namespace scivolapi::detail {

// Derivatives of one observation's negative log-likelihood l(e, h; shape)
// with respect to the variance h and the shape parameters.
template<std::size_t Shapes>
struct LocalDerivatives {
    double value = 0.0;  // excludes the per-observation constant
    double h = 0.0;
    double hh = 0.0;
    std::array<double, Shapes> s{};
    std::array<double, Shapes> hs{};
    std::array<double, packed_size(Shapes)> ss{};
};

// Gaussian density with variance h:  l = 0.5 * (log 2pi + log h + e^2 / h).
struct NormalDensity {
    static constexpr std::size_t shapes = 0;

    struct Prepared {
        double constant = 0.5 * log_2pi;  // per scored observation
    };

    [[nodiscard]] static bool prepare(const double* /*shape*/, Prepared& out) noexcept
    {
        out = Prepared{};
        return true;
    }

    [[nodiscard]] static LocalDerivatives<0> observe(const Prepared&, double e, double h, int order) noexcept
    {
        LocalDerivatives<0> d;
        const double r = (e / h) * e;  // e^2 / h without forming e^2
        d.value = 0.5 * (std::log(h) + r);
        if (order >= 1) d.h = 0.5 * (1.0 - r) / h;
        if (order >= 2) d.hh = (r - 0.5) / (h * h);
        return d;
    }
};

}  // namespace scivolapi::detail
