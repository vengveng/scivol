#pragma once

#include "mean_equation.hpp"

namespace vlk::experiment::mean {

// Storage adapters only. The mean recursion and compensated dot product stay shared.
template<std::size_t Columns = dynamic, bool Intercept = true>
struct ExternalLinearDesign {
    static constexpr bool linear = true;
    static constexpr bool residual_parameters = Columns != 0;
    static constexpr unsigned sensitivity_interleave = 4;
    const double* values;
    std::size_t external_columns;

    std::size_t columns() const noexcept {
        return extent<Columns>(external_columns + std::size_t(Intercept));
    }
    double at(std::size_t time, std::size_t column) const noexcept {
        if constexpr (Intercept)
            if (column == 0) return 1.0;
        const auto width = columns() - std::size_t(Intercept);
        return LinearDesign<> {values, width}.at(time, column - std::size_t(Intercept));
    }
    bool zero_row(std::size_t time) const noexcept {
        if constexpr (Intercept) return false;
        return LinearDesign<> {values, columns()}.zero_row(time);
    }
};

template<std::size_t P = dynamic, bool Intercept = true>
struct ConditionalArmaDesign {
    static constexpr bool linear = false;
    const double* observations;
    std::size_t order;
    conditioning::History conditioning_history;

    std::size_t columns() const noexcept {
        return extent<P>(order) + std::size_t(Intercept);
    }
    double at(std::size_t time, std::size_t column) const noexcept {
        return ArmaDesign<P> {observations, order, conditioning_history}.at(
            time, column + std::size_t(!Intercept));
    }
};

} // namespace vlk::experiment::mean
