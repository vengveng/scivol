#pragma once

#include "layout.hpp"
#include <algorithm>
#include <cmath>

namespace scivolapi::detail::dependence {

// Non-owning exact forward sensitivities. Storage always comes from the arena.
// Elementary arithmetic is private to the matrix/recursion owner, not a density.
template<int Order, class Dimensions, class Basis = IdentitySensitivityBasis>
class JetAlgebra {
    static_assert(!Basis::mapped, "Fixed-input DCC supports physical selected axes only.");
public:
    explicit constexpr JetAlgebra(Dimensions dimensions, Basis basis = {}) noexcept
        : dimensions_(dimensions), basis_(basis) {}
    [[nodiscard]] constexpr Layout get() const noexcept { return dimensions_.get(); }
    [[nodiscard]] constexpr std::size_t width() const noexcept { return jet_width<Order>(get(), basis_); }
    [[nodiscard]] constexpr std::size_t correlation() const noexcept { return correlation_width(get(), basis_); }
    [[nodiscard]] constexpr std::size_t parameters() const noexcept { return basis_.width(get().parameters); }
    [[nodiscard]] constexpr std::size_t shape() const noexcept {
        return get().student ? basis_.active_axis(get().shape_start) : inactive_axis;
    }
    // State jets omit nu (Q does not depend on it); output axes retain caller order.
    [[nodiscard]] constexpr std::size_t output_axis(std::size_t state_axis) const noexcept {
        return state_axis + std::size_t(shape() <= state_axis);
    }
    [[nodiscard]] constexpr std::size_t state_axis(std::size_t ambient_axis) const noexcept {
        const auto selected = basis_.active_axis(ambient_axis);
        return selected == inactive_axis ? inactive_axis : selected - std::size_t(shape() < selected);
    }
    [[nodiscard]] constexpr std::size_t h(std::size_t i, std::size_t j) const noexcept {
        return 1 + correlation() + packed_index(i, j, correlation());
    }
    void constant(double* x, double value) const noexcept {
        std::fill_n(x, width(), 0); x[0] = value;
    }
    void copy(double* destination, const double* source) const noexcept {
        std::copy_n(source, width(), destination);
    }
    void add(double* destination, const double* source, double scale = 1) const noexcept {
        for (std::size_t i = 0; i < width(); ++i) destination[i] += scale * source[i];
    }
    // Destination must be disjoint from the operands.
    void multiply(double* result, const double* x, const double* y) const noexcept {
        result[0] = x[0] * y[0];
        if constexpr (Order >= 1)
            for (std::size_t i = 0; i < correlation(); ++i)
                result[1 + i] = x[1 + i] * y[0] + x[0] * y[1 + i];
        if constexpr (Order >= 2)
            for (std::size_t i = 0; i < correlation(); ++i)
                for (std::size_t j = i; j < correlation(); ++j)
                    result[h(i, j)] = x[h(i, j)] * y[0] + x[0] * y[h(i, j)]
                        + x[1 + i] * y[1 + j] + x[1 + j] * y[1 + i];
    }
    void reciprocal(double* result, const double* x) const noexcept {
        const double inverse = 1 / x[0];
        result[0] = inverse;
        if constexpr (Order >= 1)
            for (std::size_t i = 0; i < correlation(); ++i)
                result[1 + i] = -(x[1 + i] / x[0]) * inverse;
        if constexpr (Order >= 2)
            for (std::size_t i = 0; i < correlation(); ++i)
                for (std::size_t j = i; j < correlation(); ++j)
                    result[h(i, j)] = (2 * (x[1 + i] / x[0]) * (x[1 + j] / x[0])
                        - x[h(i, j)] / x[0]) * inverse;
    }
    void square_root(double* result, const double* x) const noexcept {
        const double root = std::sqrt(x[0]);
        result[0] = root;
        if constexpr (Order >= 1)
            for (std::size_t i = 0; i < correlation(); ++i)
                result[1 + i] = .5 * (x[1 + i] / root);
        if constexpr (Order >= 2)
            for (std::size_t i = 0; i < correlation(); ++i)
                for (std::size_t j = i; j < correlation(); ++j)
                    result[h(i, j)] = .5 * (x[h(i, j)] / root)
                        - .25 * (x[1 + i] / root) * (x[1 + j] / root) / root;
    }
    template<bool F>
    void logarithm(double* result, const double* x) const noexcept {
        result[0] = F ? std::log(x[0]) : 0;
        if constexpr (Order >= 1)
            for (std::size_t i = 0; i < correlation(); ++i) result[1 + i] = x[1 + i] / x[0];
        if constexpr (Order >= 2)
            for (std::size_t i = 0; i < correlation(); ++i)
                for (std::size_t j = i; j < correlation(); ++j)
                    result[h(i, j)] = x[h(i, j)] / x[0]
                        - (x[1 + i] / x[0]) * (x[1 + j] / x[0]);
    }
    void add_parameter_product(double* destination, double parameter, std::size_t index,
        const double* state) const noexcept {
        add(destination, state, parameter);
        if constexpr (Order >= 1) {
            index = state_axis(index);
            if (index == inactive_axis) return;
            destination[1 + index] += state[0];
        }
        if constexpr (Order >= 2) {
            for (std::size_t j = 0; j < correlation(); ++j)
                destination[h(index, j)] += (index == j ? 2 : 1) * state[1 + j];
        }
    }
private:
    Dimensions dimensions_;
    Basis basis_;
};
} // namespace scivolapi::detail::dependence
