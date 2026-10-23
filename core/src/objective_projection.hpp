#pragma once

// SCIVOLAPI: PROTOTYPE; theta restrictions only, qualification gates remain open.

#include "internal.hpp"
#include "math/local_density_status.hpp"
#include "math/runtime_workspace.hpp"
#include "active_sensitivity_basis.hpp"

namespace scivolapi::detail {

// Phase 1 affine theta selection. Original-chart restrictions are not implemented:
// fixing a theta parameter is not generally the same as fixing its chart axis.
class ObjectiveProjection {
public:
    static Outcome<ObjectiveProjection> resolve(const ParameterLayout& layout,
        DerivativeTarget target, Coordinates coordinates, ObjectiveRestriction restriction) {
        if (target != DerivativeTarget::full_theta
            && target != DerivativeTarget::fixed_free_restriction)
            return failure(Status::unsupported_operation, "Profile projection is not an affine restriction.");
        if (coordinates != Coordinates::theta && coordinates != Coordinates::original_chart)
            return failure(Status::invalid_argument, "Unknown objective coordinates.");
        if (target == DerivativeTarget::fixed_free_restriction && coordinates != Coordinates::theta)
            return failure(Status::unsupported_operation, "Restricted original-chart mathematics remains prototype debt.");
        if (!admitted(layout.entries) || layout.entries.size() != layout.full_theta_width
            || !admitted(restriction.free_axes) || !admitted(restriction.fixed_values))
            return failure(Status::invalid_argument, "Invalid ambient layout or restriction storage.");
        if (target == DerivativeTarget::full_theta
            && (!restriction.free_axes.empty() || !restriction.fixed_values.empty()))
            return failure(Status::invalid_argument, "A full-theta target cannot carry a restriction.");
        try {
            ObjectiveProjection result;
            result.ambient_layout_ = layout.id;
            result.width_ = layout.entries.size();
            result.restricted_ = target == DerivativeTarget::fixed_free_restriction;
            result.partition_.assign(result.width_, ParameterDisposition::fixed);
            result.fixed_.resize(result.width_);
            for (std::size_t i = 0; i < result.width_; ++i) {
                const auto& entry = layout.entries[i];
                if (entry.index != i || entry.id == 0 || entry.name.empty())
                    return failure(Status::invalid_argument, "Ambient entries must have unique identities and canonical indices.");
                for (std::size_t j = 0; j < i; ++j)
                    if (entry.id == layout.entries[j].id || entry.name == layout.entries[j].name)
                        return failure(Status::invalid_argument, "Ambient parameter identities and names must be unique.");
            }
            if (!result.restricted_) {
                for (const auto& entry : layout.entries) {
                    result.axes_.push_back(entry.id);
                    result.indices_.push_back(entry.index);
                }
                result.partition_.assign(result.width_, ParameterDisposition::free);
                return Outcome<ObjectiveProjection>::success(std::move(result));
            }
            std::size_t count;
            if (!checked_add(restriction.free_axes.size(), restriction.fixed_values.size(), count)
                || count != result.width_)
                return failure(Status::invalid_argument, "Restrictions require an exact free/fixed ambient partition.");
            std::vector<bool> present(result.width_, false);
            for (auto id : restriction.free_axes) {
                const auto found = std::find_if(layout.entries.begin(), layout.entries.end(),
                    [=](const auto& entry) { return entry.id == id; });
                if (found == layout.entries.end() || present[found->index])
                    return failure(Status::invalid_argument, "Free axes must be unique ambient parameter identities.");
                present[found->index] = true;
                result.axes_.push_back(id);
                result.indices_.push_back(found->index);
                result.partition_[found->index] = ParameterDisposition::free;
            }
            for (const auto& value : restriction.fixed_values) {
                const auto found = std::find_if(layout.entries.begin(), layout.entries.end(),
                    [&](const auto& entry) { return entry.name == value.name; });
                if (found == layout.entries.end() || present[found->index] || !std::isfinite(value.value))
                    return failure(Status::invalid_argument, "Fixed values must be finite, unique and disjoint from free axes.");
                present[found->index] = true;
                result.fixed_[found->index] = value.value;
            }
            return Outcome<ObjectiveProjection>::success(std::move(result));
        } catch (const std::bad_alloc&) {
            return failure(Status::allocation_failure, "Cannot allocate restriction metadata.");
        } catch (const std::length_error&) {
            return failure(Status::invalid_argument, "Restriction metadata exceeds container limits.");
        }
    }

    bool restricted() const noexcept { return restricted_; }
    std::size_t ambient_width() const noexcept { return width_; }
    std::span<const ParameterId> axes() const noexcept { return axes_; }
    std::span<const ParameterDisposition> partition() const noexcept { return partition_; }
    std::span<const std::size_t> indices() const noexcept { return indices_; }
    ActiveSensitivityBasis sensitivity_basis() const noexcept { return {width_,indices_}; }
    JetRequest ambient_request(JetRequest request) const noexcept {
        return restricted_ && indices_.empty() ? JetRequest::F : request;
    }

    // No tolerance, clipping or replacement. Preserve the complete supplied theta,
    // including its signed zeros; fixed values compare by exact double equality.
    bool matches(std::span<const double> theta) const noexcept {
        if (theta.size() != width_ || !admitted(theta) || !finite(theta)) return false;
        for (std::size_t i = 0; i < width_; ++i)
            if (partition_[i] == ParameterDisposition::fixed && theta[i] != fixed_[i]) return false;
        return true;
    }

    Status expand(std::span<const double> free, std::span<double> theta) const noexcept {
        if (!restricted_ || free.size() != indices_.size() || theta.size() != width_
            || !admitted(free) || !admitted(theta) || !finite(free)) return Status::invalid_argument;
        MemoryRange input, output;
        if (!range(free, input) || !range(theta, output) || overlap(input, output))
            return Status::invalid_argument;
        std::copy(fixed_.begin(), fixed_.end(), theta.begin());
        for (std::size_t j = 0; j < indices_.size(); ++j) theta[indices_[j]] = free[j];
        return Status::ok;
    }

    // Intermediates precede the existing backend arena, aligned to its shared 64-byte
    // workspace contract. F alone requires no full-width derivative buffers.
    bool scratch_bytes(JetRequest request, std::size_t& bytes,bool selected_evaluator=false) const noexcept {
        const auto mask = static_cast<std::uint32_t>(request);
        if (!valid_mask(mask)) return false;
        bytes = 0;
        if (!restricted_ || indices_.empty() || selected_evaluator) return true;
        vlk::experiment::runtime::WorkspaceSizer size;
        std::size_t square = 0;
        if ((mask & 2) && !size.add<double>(width_)) return false;
        if ((mask & 4) && (!checked_multiply(width_, width_, square)
            || !size.add<double>(square))) return false;
        if (!checked_add(size.size(), 63, bytes)) return false;
        bytes &= ~std::size_t{63};
        return true;
    }

    template<class AmbientEvaluator>
    JetValidity evaluate(std::span<const double> theta, JetRequest request, JetRequest maximum_request,
        std::span<std::byte> scratch, JetBuffers outputs, AmbientEvaluator&& ambient,
        bool selected_evaluator = false) const noexcept {
        const auto mask = static_cast<std::uint32_t>(request);
        const auto maximum = static_cast<std::uint32_t>(maximum_request);
        std::size_t prefix = 0, square = 0;
        if (!restricted_ || !matches(theta) || !valid_mask(mask) || !scratch_bytes(maximum_request, prefix,selected_evaluator)
            || !checked_multiply(indices_.size(), indices_.size(), square))
            return invalid(Status::invalid_argument, request);
        if ((mask & maximum) != mask) return invalid(Status::insufficient_capacity, request);
        if (scratch.size() < prefix) return invalid(Status::insufficient_capacity, request);
        if ((mask & 1) && outputs.value.empty()) return invalid(Status::insufficient_capacity, request);
        if ((mask & 2) && outputs.gradient.size() < indices_.size())
            return invalid(Status::insufficient_capacity, request);
        if ((mask & 4) && (outputs.hessian.dimension != indices_.size()
            || outputs.hessian.elements.size() < square))
            return invalid(Status::insufficient_capacity, request);
        if (!storage_admitted(theta, scratch, outputs, mask))
            return invalid(Status::invalid_argument, request);

        if (indices_.empty()) {
            // On R^0 the derivatives are empty. Probe objective existence/domain with
            // F, not unavailable ambient derivatives. Never publish an unrequested F.
            double probe = 0;
            auto value = (mask & 1) ? outputs.value.first(1) : std::span<double>{&probe, 1};
            const auto full = ambient(theta, JetRequest::F, scratch,
                JetBuffers{ambient_layout_, value, {}, {{}, width_}});
            if ((full.valid_mask & 1) == 0 || !std::isfinite(value[0]))
                return invalid((full.valid_mask & 1) ? Status::numerical_range : full.f_reason, request);
            return invalid(Status::ok, request, mask);
        }

        if (selected_evaluator) {
            // The callback already differentiates L(theta_fixed+E*u). Do not
            // request, validate or project ambient derivatives on this route.
            auto result = ambient(theta,request,scratch.subspan(prefix),outputs);
            result.requested = request;
            result.valid_mask &= mask;
            check_finite(result,outputs,indices_.size(),square);
            return result;
        }

        // The reserved maximum determines a stable backend base for point caches;
        // the actual request still determines which intermediate arrays are used.
        vlk::experiment::runtime::WorkspaceArena arena(scratch.data(), prefix);
        auto gradient = (mask & 2) ? std::span<double>{arena.take<double>(width_), width_}
                                  : std::span<double>{};
        const auto ambient_square = (mask & 4) ? width_ * width_ : 0;
        auto hessian = (mask & 4) ? std::span<double>{arena.take<double>(ambient_square), ambient_square}
                                 : std::span<double>{};
        auto result = ambient(theta, request, scratch.subspan(prefix),
            JetBuffers{ambient_layout_, (mask & 1) ? outputs.value.first(1) : std::span<double>{},
                gradient, {hessian, width_}});
        result.requested = request;
        result.valid_mask &= mask;
        // An ambient failure does not prove failure on a proper subspace. The
        // current contract carries no active-direction certificate. Empty axes
        // use the F-only branch above; a full permutation preserves the domain.
        if (indices_.size()<width_) {
            if ((mask&2) && result.g_reason==Status::nondifferentiable)
                result.g_reason=Status::unsupported_operation;
            if ((mask&4) && result.h_reason==Status::nondifferentiable)
                result.h_reason=Status::unsupported_operation;
        }
        if (result.valid_mask & 2)
            for (std::size_t j = 0; j < indices_.size(); ++j)
                outputs.gradient[j] = gradient[indices_[j]];
        if (result.valid_mask & 4)
            for (std::size_t i = 0; i < indices_.size(); ++i)
                for (std::size_t j = 0; j < indices_.size(); ++j)
                    outputs.hessian.elements[i * indices_.size() + j]
                        = hessian[indices_[i] * width_ + indices_[j]];
        check_finite(result, outputs, indices_.size(), square);
        return vlk::experiment::density_status::complete_validity(result);
    }

private:
    ObjectiveProjection() = default;
    LayoutId ambient_layout_ = 0;
    std::size_t width_ = 0;
    bool restricted_ = false;
    std::vector<ParameterId> axes_;
    std::vector<ParameterDisposition> partition_;
    std::vector<std::size_t> indices_;
    std::vector<double> fixed_;

    static Outcome<ObjectiveProjection> failure(Status status, const char* message) {
        return Outcome<ObjectiveProjection>::failure({status, message});
    }
    static JetValidity invalid(Status status, JetRequest request, std::uint32_t valid = 0) noexcept {
        const auto mask = static_cast<std::uint32_t>(request);
        const auto reason = [=](unsigned bit) {
            return (valid & bit) ? Status::ok : ((mask & bit) ? status : Status::unsupported_operation);
        };
        return {status, request, valid, reason(1), reason(2), reason(4)};
    }
    static bool storage_admitted(std::span<const double> theta, std::span<std::byte> scratch,
        JetBuffers outputs, std::uint32_t mask) noexcept {
        MemoryRange point, work;
        if (!range(theta, point) || !admitted(scratch) || !range(scratch, work)
            || (!scratch.empty() && reinterpret_cast<std::uintptr_t>(scratch.data()) % 64 != 0)
            || overlap(point, work)) return false;
        std::array<MemoryRange, 3> ranges{};
        const std::array<std::span<double>, 3> spans{outputs.value, outputs.gradient, outputs.hessian.elements};
        for (std::size_t i = 0; i < spans.size(); ++i) {
            if (!admitted(spans[i]) || !range(spans[i], ranges[i])) return false;
            // Backend scratch is mutable even when the overlapping public output is
            // unrequested. Reject that alias instead of silently changing storage.
            if (overlap(ranges[i], work) || ((mask & (1u << i)) && overlap(ranges[i], point))) return false;
        }
        for (std::size_t i = 0; i < spans.size(); ++i)
            for (std::size_t j = i + 1; j < spans.size(); ++j)
                if ((mask & ((1u << i) | (1u << j))) && overlap(ranges[i], ranges[j])) return false;
        return true;
    }
    static void check_finite(JetValidity& result, JetBuffers outputs,
        std::size_t width, std::size_t square) noexcept {
        const std::array<std::span<double>, 3> spans{
            outputs.value.first((result.valid_mask & 1) ? 1 : 0),
            outputs.gradient.first((result.valid_mask & 2) ? width : 0),
            outputs.hessian.elements.first((result.valid_mask & 4) ? square : 0)};
        const std::array<Status*, 3> reasons{&result.f_reason, &result.g_reason, &result.h_reason};
        for (std::size_t i = 0; i < spans.size(); ++i)
            if ((result.valid_mask & (1u << i)) && !finite(spans[i])) {
                result.valid_mask &= ~(1u << i);
                *reasons[i] = Status::numerical_range;
            }
        if (result.status == Status::ok && result.valid_mask != static_cast<std::uint32_t>(result.requested))
            result.status = Status::numerical_range;
    }
};

} // namespace scivolapi::detail
