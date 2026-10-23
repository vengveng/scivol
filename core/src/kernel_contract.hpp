#pragma once

#include <scivolapi/detail/types.hpp>
#include <optional>

namespace scivolapi::detail {

struct KernelRequirements {
    Status status = Status::invalid_argument;
    std::size_t bytes = 0, alignment = alignof(double), parameters = 0;
    int dispatch = -1; // Negative is unavailable; zero is runtime; positive is fixed.
};
struct KernelBuffers {
    std::span<double> value, gradient, hessian;
};

} // namespace scivolapi::detail
