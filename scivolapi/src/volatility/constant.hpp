#pragma once

#include <cmath>
#include <cstddef>

#include <scivolapi/model.hpp>

#include "../detail/numeric.hpp"
#include "volatility.hpp"

namespace scivolapi::detail {

// Constant variance, the mean model alone: theta = [sigma2], h_t = sigma2 for
// every row (no seed). lambda = log sigma2 has d lambda = 1 / sigma2 and
// d2 lambda = -1 / sigma2^2 in its own entry and nothing in the mean entries.
template<bool Coupled = false>
class ConstantVolatility {
public:
    explicit ConstantVolatility(const ModelSpec&, std::size_t mean = 0) noexcept : m_(Coupled ? mean : 0) {}

    [[nodiscard]] std::size_t own() const noexcept { return 1; }
    [[nodiscard]] std::size_t width() const noexcept { return 1 + m_; }
    [[nodiscard]] std::size_t ring_slots() const noexcept { return 1; }

    [[nodiscard]] bool evaluable(const double* theta) const noexcept { return theta[0] > 0.0 && std::isfinite(theta[0]); }
    [[nodiscard]] bool prepare(const double*, int) noexcept { return true; }

    [[nodiscard]] bool step(const double* theta, const Sample&, std::size_t t, int order, VolatilityWork& w,
                            VolatilityStep& out) const noexcept
    {
        const double h = theta[0];
        w.h[t] = h;
        out.h = h;
        out.lambda = std::log(h);
        out.inv_sigma = 1.0 / std::sqrt(h);
        if (order < 1) return true;
        const std::size_t width = this->width();
        std::fill_n(w.l1, width, 0.0);
        w.l1[0] = 1.0 / h;
        if (order >= 2) {
            std::fill_n(w.l2, packed_size(width), 0.0);
            w.l2[packed(width, 0, 0)] = -1.0 / (h * h);
        }
        return true;
    }

private:
    std::size_t m_;
};

}  // namespace scivolapi::detail
