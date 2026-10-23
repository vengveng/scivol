#pragma once
#include <cmath>
#include <cstddef>

namespace vlk::experiment::generated {
// Local sufficient-state demand, derived from the active statistical basis.
// Shape bits retain the density owner's existing shape identities and indexing.
struct LocalDensityAxes {
    bool residual = true, variance = true;
    unsigned shapes = ~0u;
    bool first = true, residual_second = true;
    constexpr bool shape(std::size_t i) const noexcept { return (shapes & (1u << i)) != 0; }
    bool local(std::size_t i, std::size_t shape_start, bool residual_slot) const noexcept {
        if (residual_slot && i == 0) return residual;
        if (i + 1 == shape_start) return variance;
        return i >= shape_start && shape(i - shape_start);
    }
    template<class Number>
    Number seed_shape(double value, std::size_t shape_id, std::size_t slot) const noexcept {
        return shape(shape_id) ? Number::variable(value, slot) : Number::constant(value);
    }
    template<class Number>
    bool finite(const Number& x, std::size_t shape_start, bool residual_slot,
        bool need_value = true) const noexcept {
        if (need_value && !std::isfinite(x.value)) return false;
        if constexpr (requires { x.gradient; })
            for (std::size_t i = 0; i < x.gradient.size(); ++i) if (local(i, shape_start, residual_slot)) {
                if ((first || residual_second) && !std::isfinite(x.gradient[i])) return false;
                if constexpr (requires { x.hessian; })
                    for (std::size_t j = i; j < x.gradient.size(); ++j) if (local(j, shape_start, residual_slot)) {
                        const auto packed = i*x.gradient.size() - i*(i+1)/2 + j;
                        if (!std::isfinite(x.hessian[packed])) return false;
                    }
            }
        return true;
    }
};
} // namespace vlk::experiment::generated
