#pragma once
#include <scivolapi/detail/types.hpp>
#include <cstddef>
#include <span>

namespace profile_ordinary_test {
constexpr std::size_t dynamic_width = static_cast<std::size_t>(-1);
struct Input {
    unsigned request;
    std::size_t width, rows;
    std::span<const double> residual, first, second;
    bool force_scaled = false;
    std::size_t fixed_width = dynamic_width;
    std::size_t finalize_after = 0;
    bool checked_ordinary = false;
    double reporting_response_scale = 1;
};
struct Result {
    scivolapi::Status status = scivolapi::Status::invalid_argument;
    unsigned valid = 0;
    bool has_sse = false, has_variance = false, zero_sse_unbounded = false;
    double sse = 0, variance = 0, log_variance = 0;
    std::size_t ordinary_updates = 0, observe_normalizations = 0, final_normalizations = 0;
};
std::size_t workspace_bytes(const Input&) noexcept;
Result evaluate(const Input&, std::span<std::byte>, std::span<double>, std::span<double>) noexcept;
struct Product { double left, right; };
struct Limbs { double high, low; };
struct ProductTrace { Limbs checked, admitted; };
bool trace_products(std::span<const Product>, std::span<ProductTrace>) noexcept;
bool entry_guards_preserve_prefix() noexcept;
bool checked_fallbacks_preserved() noexcept;
} // namespace profile_ordinary_test
