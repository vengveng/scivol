#pragma once

#include "../../kernel_contract.hpp"
#include "../../math/scaled_number.hpp"
#include "../../math/runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>

namespace scivolapi::detail::profile::range {

template<class Precision>
using Scaled = numerical::Scaled<Precision>;

struct Nuisance {
    Status status = Status::invalid_argument;
    bool zero_sse_unbounded = false;
    std::optional<double> sse, variance;
    double log_variance = 0.0;
};

// Request controls scratch and recursive dependencies. H retains A internally;
// F-only has a scalar moment and no derivative arrays or second-order cache.
template<class Precision, unsigned Request,
         std::size_t Width = std::numeric_limits<std::size_t>::max()>
class NormalMoments {
public:
    static_assert(Request != 0 && Request <= 7);
    using Number = Scaled<Precision>;
    static_assert(std::is_trivially_destructible_v<Number>);
    static constexpr bool First = (Request & 6u) != 0;
    static constexpr bool Second = (Request & 4u) != 0;

    static KernelRequirements requirements(std::size_t parameters) noexcept {
        KernelRequirements result;
        if constexpr (Width != std::numeric_limits<std::size_t>::max())
            if (parameters != Width) return result;
        std::size_t square;
        if (!vlk::experiment::runtime::checked_product(parameters, parameters, square)) return result;
        vlk::experiment::runtime::WorkspaceSizer size;
        if constexpr (First) size.add<Number>(parameters);
        if constexpr (Second) size.add<Number>(square);
        const auto bytes = size.size();
        if ((First && parameters && !bytes) || (Second && square && !bytes)) return result;
        result.status = Status::ok;
        result.bytes = bytes;
        result.alignment = alignof(Number);
        result.parameters = parameters;
        result.dispatch = 0;
        return result;
    }
    [[nodiscard]] Status bind(std::size_t parameters, std::span<std::byte> workspace) noexcept {
        const auto needed = requirements(parameters);
        if (needed.status != Status::ok) return needed.status;
        if (workspace.size() < needed.bytes) return Status::insufficient_capacity;
        if (needed.bytes && (!workspace.data()
            || reinterpret_cast<std::uintptr_t>(workspace.data()) % needed.alignment))
            return Status::invalid_argument;
        vlk::experiment::runtime::WorkspaceArena arena(workspace.data(), workspace.size());
        Number* a = nullptr;
        Number* b = nullptr;
        if constexpr (First) if (parameters) a = arena.take<Number>(parameters);
        if constexpr (Second) if (parameters) b = arena.take<Number>(parameters*parameters);
        if ((First && parameters && !a) || (Second && parameters && !b))
            return Status::insufficient_capacity;
        if constexpr (First)
            for (std::size_t i = 0; i < parameters; ++i) std::construct_at(a+i);
        if constexpr (Second)
            for (std::size_t i = 0; i < parameters*parameters; ++i) std::construct_at(b+i);
        parameters_ = parameters;
        a_ = a;
        b_ = b;
        sse_ = {};
        nonzero_ = false;
        prepared_n_ = 0;
        observations_ = 0;
        bound_ = true;
        ordinary_ = true;
        return Status::ok;
    }
    [[nodiscard]] Status observe(double residual, const double* first = nullptr,
        const double* second_packed = nullptr) noexcept {
        const auto width = parameters();
        if (!bound_ || (First && width && !first)
            || (Second && width && !second_packed)) return Status::invalid_argument;
        if (!std::isfinite(residual)) return Status::numerical_range;
        if (observations_ == std::numeric_limits<std::size_t>::max()) return Status::invalid_argument;
        ++observations_;
        prepared_n_ = 0;
        nonzero_ = nonzero_ || residual != 0.0;
        if (ordinary_ && ordinary_row(residual, first, second_packed))
            accumulate_row<true>(residual, first, second_packed);
        else {
            finish_ordinary_accumulation();
            accumulate_row<false>(residual, first, second_packed);
        }
        // Derivative failures remain local. They must not invalidate valid S/F.
        return sse_.valid() ? Status::ok : Status::numerical_range;
    }
    [[nodiscard]] Nuisance nuisance(std::size_t n_eff, double reporting_response_scale = 1) noexcept {
        Nuisance result;
        if (!bound_ || !n_eff || n_eff != observations_
            || !std::isfinite(reporting_response_scale) || reporting_response_scale <= 0) return result;
        finish_ordinary_accumulation();
        if (!nonzero_ && sse_.zero()) {
            result.status = Status::mathematical_domain;
            result.zero_sse_unbounded = true;
            result.sse = 0.0;
            return result;
        }
        result.status = Status::numerical_range;
        if (!sse_.positive()) return result;
        count_ = Number::count(n_eff);
        prepared_n_ = n_eff;
        if constexpr (Second) if (parameters()) denominator_ = Number::multiply(sse_,sse_);
        const auto variance = Number::divide(sse_, count_);
        // Convert the unrounded nuisance, not a rounded internal variance. This
        // also permits a finite reporting variance when its internal double
        // would underflow. The likelihood and derivative moments stay internal.
        auto reported_sse = sse_, reported_variance = variance;
        if (reporting_response_scale != 1) {
            const auto r = Number::from(reporting_response_scale);
            const auto square = Number::multiply(r, r);
            reported_sse = Number::multiply(sse_, square);
            reported_variance = Number::multiply(variance, square);
        }
        double diagnostic;
        if (reported_sse.narrow(diagnostic) == Status::ok) result.sse = diagnostic;
        double estimate;
        if (reported_variance.narrow(estimate) != Status::ok || !(estimate > 0.0) || !std::isfinite(estimate))
            return result;
        result.variance = estimate;
        if constexpr (Request & 1u) {
            result.log_variance = variance.log_positive();
            if (!std::isfinite(result.log_variance)) return result;
        }
        result.status = Status::ok;
        return result;
    }
    [[nodiscard]] Status gradient(std::size_t i, std::size_t n_eff, double& output) const noexcept {
        if constexpr (!First) return Status::unsupported_operation;
        else {
            if (!bound_ || i >= parameters() || !n_eff || n_eff != prepared_n_ || !sse_.positive())
                return Status::invalid_argument;
            return Number::divide(Number::multiply(count_, a_[i]), sse_).narrow(output);
        }
    }
    [[nodiscard]] Status hessian(std::size_t i, std::size_t j, std::size_t n_eff,
        double& output) const noexcept {
        if constexpr (!Second) return Status::unsupported_operation;
        else {
            if (!bound_ || i >= parameters() || j >= parameters() || !n_eff
                || n_eff != prepared_n_ || !sse_.positive())
                return Status::invalid_argument;
            if (j < i) std::swap(i,j);
            const auto bs = Number::multiply(b_[i*parameters()+j], sse_);
            const auto twice_aa = Number::multiply(Number::from(2.0), Number::multiply(a_[i],a_[j]));
            // Keep B*S - 2*A*A' scaled through cancellation; never narrow terms.
            const auto numerator = Number::multiply(count_, Number::subtract(bs,twice_aa));
            return Number::divide(numerator, denominator_).narrow(output);
        }
    }

private:
    bool ordinary_row(double residual, const double* first, const double* second) const noexcept {
        if (!Precision::ordinary_product_input(residual)) return false;
        const auto width = parameters();
        if constexpr (First)
            for (std::size_t i = 0; i < width; ++i)
                if (!Precision::ordinary_product_input(first[i])) return false;
        if constexpr (Second)
            for (std::size_t i = 0; i < width*(width+1)/2; ++i)
                if (!Precision::ordinary_product_input(second[i])) return false;
        return true;
    }
    template<bool Ordinary>
    static void accumulate(Number& sum, double left, double right) noexcept {
        if constexpr (Ordinary) sum.accumulate_ordinary_product(left, right);
        else sum.add_product(left, right);
    }
    template<bool Ordinary>
    void accumulate_row(double residual, const double* first, const double* second) noexcept {
        const auto width = parameters();
        accumulate<Ordinary>(sse_, residual, residual);
        if constexpr (First)
            for (std::size_t i = 0; i < width; ++i) accumulate<Ordinary>(a_[i], residual, first[i]);
        if constexpr (Second) {
            std::size_t packed = 0;
            for (std::size_t i = 0; i < width; ++i)
                for (std::size_t j = i; j < width; ++j) {
                    auto& entry = b_[i*width+j];
                    accumulate<Ordinary>(entry, first[i], first[j]);
                    accumulate<Ordinary>(entry, residual, second[packed++]);
                }
        }
    }
    void finish_ordinary_accumulation() noexcept {
        if (!ordinary_) return;
        sse_.finish_ordinary_accumulation();
        const auto width = parameters();
        if constexpr (First)
            for (std::size_t i = 0; i < width; ++i) a_[i].finish_ordinary_accumulation();
        if constexpr (Second)
            for (std::size_t i = 0; i < width; ++i)
                for (std::size_t j = i; j < width; ++j) b_[i*width+j].finish_ordinary_accumulation();
        ordinary_ = false;
    }
    [[nodiscard]] constexpr std::size_t parameters() const noexcept {
        if constexpr (Width != std::numeric_limits<std::size_t>::max()) return Width;
        else return parameters_;
    }
    struct NoSecondPreparation {};
    Number sse_, count_;
    [[no_unique_address]] std::conditional_t<Second,Number,NoSecondPreparation> denominator_;
    Number* a_ = nullptr;
    Number* b_ = nullptr;
    std::size_t parameters_ = 0;
    std::size_t prepared_n_ = 0;
    std::size_t observations_ = 0;
    bool nonzero_ = false, bound_ = false, ordinary_ = true;
};

} // namespace scivolapi::detail::profile::range
