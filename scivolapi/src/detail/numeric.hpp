#pragma once

#include <cmath>
#include <cstddef>

namespace scivolapi::detail {

inline constexpr double log_2pi = 1.8378770664093454835606594728112;

// Packed upper-triangular index of the symmetric (row, column) entry, row <= column.
[[nodiscard]] constexpr std::size_t packed(std::size_t size, std::size_t row, std::size_t column) noexcept
{
    return row * size - row * (row + 1u) / 2u + column;
}

[[nodiscard]] constexpr std::size_t packed_size(std::size_t size) noexcept
{
    return size * (size + 1u) / 2u;
}

// Neumaier compensated summation for additive likelihood totals.
class CompensatedSum {
public:
    void add(double x) noexcept
    {
        const double t = sum_ + x;
        compensation_ += std::abs(sum_) >= std::abs(x) ? (sum_ - t) + x : (x - t) + sum_;
        sum_ = t;
    }
    [[nodiscard]] double total() const noexcept { return sum_ + compensation_; }

private:
    double sum_ = 0.0;
    double compensation_ = 0.0;
};

// softplus(x) = log(1 + e^x) and its derivatives, stable for all finite x.
[[nodiscard]] inline double softplus(double x) noexcept
{
    return x > 0.0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

[[nodiscard]] inline double logistic(double x) noexcept
{
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

// Inverse of softplus for y > 0.
[[nodiscard]] inline double inverse_softplus(double y) noexcept
{
    return y + std::log(-std::expm1(-y));
}

}  // namespace scivolapi::detail
