#pragma once
#include "power/scaled_logistic.hpp"
#include <cstddef>

namespace vlk::experiment::positive_coordinate {

// For an independent softplus coordinate, recover its J' H J row/column when
// the stored binary64 Jacobian cannot retain the subnormal slope's precision.
// The caller adds contracted map curvature separately. Input/output are disjoint.
template<class Width, class Axis>
void softplus_hessian_row(double coordinate, Width width_tag, Axis axis_tag,
    const double* theta_hessian, const double* jacobian, double* hessian) noexcept
{
    const ScaledLogistic point(coordinate);
    if (std::isnormal(point.value())) return;
    const std::size_t n=width_tag,axis=axis_tag;
    hessian[axis*n+axis]=point.value(point.value(theta_hessian[axis*n+axis]));
    for (std::size_t column=0;column<n;++column) {
        if (column==axis) continue;
        double row_value=0.0,column_value=0.0;
        for (std::size_t inner=0;inner<n;++inner) {
            row_value+=theta_hessian[axis*n+inner]*jacobian[inner*n+column];
            column_value+=jacobian[inner*n+column]*theta_hessian[inner*n+axis];
        }
        // Scale a representable aggregate once: individually rounded subnormal
        // terms can all vanish even when their sum is representable.
        if (std::isfinite(row_value)) row_value=point.value(row_value);
        else {
            row_value=0.0;
            for (std::size_t inner=0;inner<n;++inner)
                row_value+=point.value(theta_hessian[axis*n+inner],jacobian[inner*n+column]);
        }
        if (std::isfinite(column_value)) column_value=point.value(column_value);
        else {
            column_value=0.0;
            for (std::size_t inner=0;inner<n;++inner)
                column_value+=point.value(jacobian[inner*n+column],theta_hessian[inner*n+axis]);
        }
        hessian[axis*n+column]=row_value;
        hessian[column*n+axis]=column_value;
    }
}

} // namespace vlk::experiment::positive_coordinate
