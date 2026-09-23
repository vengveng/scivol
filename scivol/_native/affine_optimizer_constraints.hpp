// SCIVOL-NATIVE: optional optimizer equality; independent of likelihood mathematics.
#pragma once

#include "runtime_kraft_slsqp.hpp"

namespace vlk::experiment::runtime {

struct AffineEquality {
    const double* normal = nullptr;
    std::size_t width = 0u;
    double target = 0.0;

    bool active() const noexcept { return normal != nullptr; }
    bool valid(std::size_t dimension) const noexcept
    {
        if (!active()) return width == 0u;
        if (!width || width > dimension || !std::isfinite(target)) return false;
        bool nonzero = false;
        for (std::size_t j = 0; j < width; ++j) {
            if (!std::isfinite(normal[j])) return false;
            nonzero |= normal[j] != 0.0;
        }
        return nonzero;
    }
};

// Validate the immutable equality before solving. Output storage includes its row.
template<class Constraints>
struct AffineEqualityPrefix {
    const Constraints& inner;
    AffineEquality affine;

    std::size_t dimension() const noexcept { return inner.dimension(); }
    std::size_t count() const noexcept { return inner.count() + affine.active(); }
    std::size_t equalities() const noexcept
    { return runtime_kraft_detail::equality_count(inner) + affine.active(); }

    bool evaluate(kraft::ConstraintRequest request, const double* point,
                  double* values, double* jacobian) const noexcept
    {
        if (!inner.evaluate(request, point, values, jacobian)) return false;
        if (!affine.active()) return true;
        const auto previous = inner.count(), rows = count();
        if (request != kraft::ConstraintRequest::jacobian) {
            std::move_backward(values, values + previous, values + rows);
            double value = -affine.target;
            for (std::size_t j = 0; j < affine.width; ++j)
                value += affine.normal[j] * point[j];
            values[0] = value;
        }
        if (request != kraft::ConstraintRequest::value) {
            // Expand the column-major Jacobian backwards in its caller-owned buffer.
            for (std::size_t j = dimension(); j-- > 0;) {
                std::move_backward(jacobian + j * previous,
                    jacobian + (j + 1) * previous, jacobian + (j + 1) * rows);
                jacobian[j * rows] = j < affine.width ? affine.normal[j] : 0.0;
            }
        }
        return true;
    }
};

} // namespace vlk::experiment::runtime
