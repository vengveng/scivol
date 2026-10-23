#pragma once
#include "models/unit_map.hpp"
#include "kernel_contract.hpp"
namespace scivolapi::detail {
// Private prepared-plan admission protects every supplied span, including
// unrequested storage. Capacity is required only for requested derivatives;
// chart H consumes G for curvature, while chart G never requires H or F.
inline Status admit_chart(std::span<const double> z, std::span<const double> theta,
    JetRequest request, std::span<std::byte> work, std::size_t bytes, std::size_t n,
    unit_map::InternalJet input, KernelBuffers output) noexcept {
    const auto mask = static_cast<unsigned>(request);
    std::size_t square=0;
    if (!n || !bytes || !unit_map::implementation::checked_product(n,n,square)
        || !mask || (mask & ~6u) || z.size() != n || theta.size() != n)
        return Status::invalid_argument;
    if (work.size() < bytes || input.gradient.size() < n
        || ((mask & 2) && output.gradient.size() < n)
        || ((mask & 4) && (input.hessian.size() < square || output.hessian.size() < square)))
        return Status::insufficient_capacity;
    using namespace unit_map::implementation;
    std::array<Range, 5> reads;
    std::array<Range, 4> writes;
    if (!range(z, reads[0]) || !range(theta, reads[1])
        || !range(input.value, reads[2]) || !range(input.gradient, reads[3])
        || !range(input.hessian, reads[4]) || !range(work, writes[0])
        || !range(output.value, writes[1]) || !range(output.gradient, writes[2])
        || !range(output.hessian, writes[3]) || !disjoint(writes)
        || (bytes && writes[0].begin % alignof(double))) return Status::invalid_argument;
    for (auto w : writes) for (auto r : reads) if (overlaps(w,r)) return Status::invalid_argument;
    return Status::ok;
}
}
