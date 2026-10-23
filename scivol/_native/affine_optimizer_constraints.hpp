// SCIVOL-NATIVE: optional optimizer equality; independent of likelihood mathematics.
#pragma once

#include "runtime_kraft_slsqp.hpp"

namespace vlk::experiment::runtime {

// One or more affine rows over the leading `width` coordinates. `normal` is
// row-major `rows x width`. A single row may use the scalar `target`; several
// rows require `targets`. Rank is the caller's responsibility.
struct AffineEquality {
    const double* normal = nullptr;
    std::size_t width = 0u;
    double target = 0.0;
    std::size_t rows = 1u;
    const double* targets = nullptr;

    bool active() const noexcept { return normal != nullptr; }
    std::size_t count() const noexcept { return active() ? rows : 0u; }
    double row_target(std::size_t i) const noexcept { return targets ? targets[i] : target; }
    bool valid(std::size_t dimension) const noexcept
    {
        if (!active()) return width == 0u;
        if (!width || width > dimension || !rows || rows > width) return false;
        std::size_t entries;
        if (!checked_product(rows, width, entries)) return false;
        if (rows > 1u && !targets) return false;
        for (std::size_t i = 0; i < rows; ++i) {
            if (!std::isfinite(row_target(i))) return false;
            bool nonzero = false;
            for (std::size_t j = 0; j < width; ++j) {
                const double entry = normal[i * width + j];
                if (!std::isfinite(entry)) return false;
                nonzero |= entry != 0.0;
            }
            if (!nonzero) return false;
        }
        return true;
    }
};

// Validate the immutable equalities before solving. Output storage includes their rows.
template<class Constraints>
struct AffineEqualityPrefix {
    const Constraints& inner;
    AffineEquality affine;

    std::size_t dimension() const noexcept { return inner.dimension(); }
    std::size_t count() const noexcept { return inner.count() + affine.count(); }
    std::size_t equalities() const noexcept
    { return runtime_kraft_detail::equality_count(inner) + affine.count(); }

    bool evaluate(kraft::ConstraintRequest request, const double* point,
                  double* values, double* jacobian, ConstraintColumns columns={}) const noexcept
    {
        if (!runtime_kraft_detail::invoke_constraints(inner,request,point,values,jacobian,columns)) return false;
        if (!affine.active()) return true;
        const auto previous = inner.count(), rows = count(), prefix = affine.count();
        if (request != kraft::ConstraintRequest::jacobian) {
            std::move_backward(values, values + previous, values + rows);
            for (std::size_t i = 0; i < prefix; ++i) {
                double value = -affine.row_target(i);
                for (std::size_t j = 0; j < affine.width; ++j)
                    value += affine.normal[i * affine.width + j] * point[j];
                values[i] = value;
            }
        }
        if (request != kraft::ConstraintRequest::value) {
            // Expand the column-major Jacobian backwards in its caller-owned buffer.
            // Column j's new prefix starts at j * rows >= j * previous, the end of
            // column j - 1's unmoved source, so filling it never overwrites input.
            for (std::size_t j = dimension(); j-- > 0;) {
                if (!columns.contains(j)) continue;
                std::move_backward(jacobian + j * previous,
                    jacobian + (j + 1) * previous, jacobian + (j + 1) * rows);
                for (std::size_t i = 0; i < prefix; ++i)
                    jacobian[j * rows + i] =
                        j < affine.width ? affine.normal[i * affine.width + j] : 0.0;
            }
        }
        return true;
    }
};

} // namespace vlk::experiment::runtime
