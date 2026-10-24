#pragma once

#include "density.hpp"

namespace scivolapi::detail {

// Standard normal:  c = 0.5 log 2pi,  g = x^2 / 2.
struct NormalDensity {
    static constexpr std::size_t shapes = 0;
    static constexpr bool symmetric = true;

    struct Prepared {
        Constant<0> constant;
    };

    static std::array<double, 0> nonsmooth_points(const Prepared&) noexcept { return {}; }

    [[nodiscard]] static bool prepare(const double* /*shape*/, int /*order*/, Prepared& out) noexcept
    {
        out.constant.value = 0.5 * log_2pi;
        return true;
    }

    [[nodiscard]] static Terms<0> observe(const Prepared&, double x, int order) noexcept
    {
        Terms<0> t;
        const double x2 = x * x;
        t.g = 0.5 * x2;
        if (order >= 1) t.xg = x2;
        if (order >= 2) t.xxg = x2;
        return t;
    }
};

}  // namespace scivolapi::detail
