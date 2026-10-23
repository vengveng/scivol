#pragma once

#include <scivolapi/detail/types.hpp>
#include "power/wide_arithmetic.hpp"
#include <boost/multiprecision/cpp_double_fp.hpp>
#include <tlfloat/tlfloat.hpp>

#include <cmath>

namespace scivolapi::detail::numerical {

// Shared binding to the vendored DD arithmetic and final boundary rounding.
struct BoostScaledPrecision {
    using Mantissa = boost::multiprecision::cpp_double_double;

    // For at most 2*SIZE_MAX products per sum, this band keeps every
    // product bit at or above 2^-504. A conservative DD intermediate bound
    // is 2^466, including addition roundoff, far inside the finite range.
    static bool ordinary_product_input(double value) noexcept {
        const double magnitude = std::abs(value);
        return value == 0.0 || (magnitude >= 0x1p-200 && magnitude <= 0x1p200);
    }
    static void accumulate_ordinary_product(Mantissa& sum, double left, double right) noexcept {
        static_assert(std::numeric_limits<std::size_t>::digits <= 64);
        // Admission includes both factors, even when one is zero. Reuse the
        // established arithmetic without the transcendental owner's tables.
        if (left == 0.0 || right == 0.0) return;
        namespace wide = vlk::experiment::wide;
        sum = wide::add_admitted_nonzero_product(wide::Real(sum),left,right).number();
    }

    static Mantissa from(double value) noexcept { return Mantissa(value); }
    static Mantissa count(std::size_t value) noexcept { return Mantissa(value); }
    static Mantissa normalize(const Mantissa& value, int& exponent) noexcept {
        return boost::multiprecision::frexp(value, &exponent);
    }
    static Mantissa scale(const Mantissa& value, int exponent) noexcept {
        return boost::multiprecision::ldexp(value, exponent);
    }
    static bool finite(const Mantissa& value) noexcept {
        return std::isfinite(value.backend().my_first()) && std::isfinite(value.backend().my_second());
    }
    static double project(const Mantissa& value) noexcept { return static_cast<double>(value); }
    static double log_abs(const Mantissa& value) noexcept {
        return static_cast<double>(boost::multiprecision::log(boost::multiprecision::abs(value)));
    }
    static Status narrow(const Mantissa& value, int exponent, double& output) noexcept {
        if (!finite(value)) return Status::numerical_range;
        if (value == Mantissa(0)) { output = 0.0; return Status::ok; }
        double candidate;
        if (exponent >= -1000 && exponent <= 1000) {
            // Exact power-of-two scaling of the rounded normal significand.
            candidate = std::scalbn(project(value), exponent);
        } else {
            // Derivatives may round to signed zero. The nuisance consumer alone
            // requires a strictly positive reconstructed variance.
            if (exponent < -1075) {
                output = std::copysign(0.0, project(value));
                return Status::ok;
            }
            if (exponent > 1026) return Status::numerical_range;
            // Only final output boundary rounding, never observation arithmetic.
            // Quad represents both DD components before the subnormal scaling.
            using Boundary = tlfloat::Quad;
            Boundary exact(value.backend().my_first());
            exact += Boundary(value.backend().my_second());
            exact = ldexp(exact, exponent);
            candidate = static_cast<double>(exact);
        }
        if (!std::isfinite(candidate)) return Status::numerical_range;
        output = candidate;
        return Status::ok;
    }
};

} // namespace scivolapi::detail::numerical
