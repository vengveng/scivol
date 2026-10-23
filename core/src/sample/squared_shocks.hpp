#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace scivolapi::detail::squared_shocks {

enum class Plan : std::uint32_t { unprepared, rounded, raw_subnormal, invalid };

// A plan belongs to immutable raw/cache/history bytes. Reprepare after mutation.
// It classifies shock arithmetic, not parameters, seed validity or model history shape.

inline bool prepared(Plan plan) noexcept {
    return plan == Plan::rounded || plan == Plan::raw_subnormal;
}

// Called while immutable raw observations/history and their caches are prepared.
inline bool include(Plan& plan, double raw, double square) noexcept {
    if (!prepared(plan) || !std::isfinite(raw) || !std::isfinite(square)
        || (raw != 0.0 && square == 0.0) || square != raw * raw) {
        plan = Plan::invalid;
        return false;
    }
    if (raw == 0.0 || std::isnormal(square)) return true;
    int exponent = 0;
    const double fraction = std::frexp(raw, &exponent);
    const volatile double rounded_fraction = fraction * fraction;
    const bool exact = std::fma(fraction, fraction, -rounded_fraction) == 0.0
        && std::scalbn(square, -2 * exponent) == rounded_fraction;
    if (!exact) plan = Plan::raw_subnormal;
    return true;
}

inline Plan prepare(const double* raw, const double* squares, std::size_t n,
                    std::span<const double> history = {}) noexcept {
    if (squares == nullptr || n == 0) return Plan::invalid;
    auto plan = Plan::rounded;
    for (std::size_t i = 0; i < n; ++i) {
        if (raw != nullptr) {
            if (!include(plan, raw[i], squares[i])) return Plan::invalid;
        } else if (!std::isfinite(squares[i]) || squares[i] < 0.0) {
            return Plan::invalid;
        }
    }
    for (double value : history)
        if (!include(plan, value, value * value)) return Plan::invalid;
    return plan;
}

} // namespace scivolapi::detail::squared_shocks
