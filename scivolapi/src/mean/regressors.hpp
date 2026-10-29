#pragma once

#include <cstddef>

#include <scivolapi/model.hpp>

namespace scivolapi::detail {

// The mean's regressor row for observation t (t >= mean.conditioning()):
//   [1 (if constant), y_{t-1..t-ar}, mean(y_{t-1..t-w}) per HAR window, x_t],
// with x row-major (observations x exog). The single definition used for
// observed samples and for simulated continuations.
inline void regressor_row(const MeanSpec& mean, const double* y, const double* x, std::size_t t, double* row) noexcept
{
    if (mean.constant) *row++ = 1.0;
    for (std::size_t i = 1; i <= mean.ar; ++i) *row++ = y[t - i];
    for (std::size_t w : mean.har) {
        double total = 0.0;
        for (std::size_t i = 1; i <= w; ++i) total += y[t - i];
        *row++ = total / static_cast<double>(w);
    }
    for (std::size_t i = 0; i < mean.exog; ++i) *row++ = x[t * mean.exog + i];
}

}  // namespace scivolapi::detail
