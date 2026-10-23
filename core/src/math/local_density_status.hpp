#pragma once

#include <scivolapi/detail/types.hpp>
#include <type_traits>
#include <cmath>
#include <limits>

namespace vlk::experiment::density_status {

enum class Code { unassessed, valid, nondifferentiable, numerical_failure, invalid_input,
    unsupported_differentiation };

struct Report {
    Code code = Code::valid;
    void reset() noexcept { code=Code::valid; }
    void reject(Code reason) noexcept {
        if (code==Code::valid || code==Code::unassessed) code=reason;
    }
    [[nodiscard]] scivolapi::Status failure() const noexcept {
        if (code==Code::nondifferentiable) return scivolapi::Status::nondifferentiable;
        if (code==Code::invalid_input) return scivolapi::Status::invalid_argument;
        if (code==Code::unsupported_differentiation) return scivolapi::Status::unsupported_operation;
        return scivolapi::Status::numerical_range;
    }
};

// Retries own each output's reason. The failed fused attempt is only a fallback.
[[nodiscard]] inline scivolapi::JetValidity complete_validity(scivolapi::JetValidity result) noexcept {
    using scivolapi::Status;
    const auto mask=static_cast<unsigned>(result.requested);
    result.valid_mask&=mask;
    const auto fallback=result.status==Status::ok ? Status::numerical_range : result.status;
    result.status=Status::ok;
    for (auto [bit,reason]:{std::pair{1u,&result.f_reason},std::pair{2u,&result.g_reason},
                           std::pair{4u,&result.h_reason}}) {
        if (!(mask&bit)) *reason=Status::unsupported_operation;
        else if (result.valid_mask&bit) *reason=Status::ok;
        else {
            if (*reason==Status::ok) *reason=fallback;
            if (result.status==Status::ok) result.status=*reason;
        }
    }
    return result;
}

template<class Observation>
[[nodiscard]] inline Code status(const Observation& observation) noexcept {
    if constexpr (requires { observation.request_status; }) {
        if (observation.request_status!=Code::unassessed && observation.request_status!=Code::valid)
            return observation.request_status;
    }
    if constexpr (requires { observation.valid; })
        if (!observation.valid) return Code::numerical_failure;
    return Code::valid;
}

template<class Observation>
[[nodiscard]] inline bool accept(const Observation& observation, Report* report=nullptr) noexcept {
    const auto reason=status(observation);
    if (reason==Code::valid) return true;
    if (report) report->reject(reason);
    return false;
}

// Wider sufficient states are not certificates for their rounded doubles.
template<class Scalar>
[[nodiscard]] inline bool exact_binary64(const Scalar& value, double& rounded) noexcept {
    rounded=static_cast<double>(value);
    if constexpr (std::is_same_v<Scalar,double>) return true;
    else if constexpr (requires { value.high; value.low; })
        return value.low==0.0 && value.high==rounded;
    else return false;
}

struct BinaryEnclosure { double lower,upper; bool supported; };
template<class Scalar>
[[nodiscard]] inline BinaryEnclosure enclose(const Scalar& value) noexcept {
    double exact;
    if (exact_binary64(value,exact)) return {exact,exact,true};
    if constexpr (requires { value.high; value.low; }) {
        const volatile double sum=value.high+value.low;
        if (std::isfinite(value.high) && std::isfinite(value.low) && std::isfinite(sum))
            return {std::nextafter(sum,-std::numeric_limits<double>::infinity()),
                    std::nextafter(sum,std::numeric_limits<double>::infinity()),true};
    }
    return {0.0,0.0,false};
}

} // namespace vlk::experiment::density_status
