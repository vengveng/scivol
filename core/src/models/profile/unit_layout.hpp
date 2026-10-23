#pragma once
#include "profile_layout.hpp"
#include "../unit_layout_roles.hpp"

namespace scivolapi::detail::profile {

// Remove only the profiled nuisance axis from already resolved mean-unit roles.
constexpr unit_map::Layout reduced_unit_layout(unit_map::Layout joint,
    std::size_t parameters) noexcept {
    const auto& roles = joint.roles;
    if (!joint.valid || roles.parameters == 0 || roles.parameters - 1 != parameters
        || roles.scale != unit_map::ScaleCoordinate::variance
        || roles.scale_axis != parameters || roles.delta != unit_map::no_axis
        || roles.beta.count) return {};
    auto reduced = roles;
    reduced.parameters = parameters;
    reduced.scale = unit_map::ScaleCoordinate::absent;
    reduced.scale_axis = unit_map::no_axis;
    return unit_map::make_layout(reduced);
}

constexpr unit_map::Layout unit_layout(ModelConfig config) noexcept {
    const auto plan = layout(config);
    if (!plan.valid) return {};
    const auto joint = unit_layout_roles::variance(plan.joint.width, plan.parameters,
        config.intercept, config.kind == Kind::linear
            ? unit_map::Block{std::size_t(config.intercept), config.design_width}
            : unit_map::Block{});
    return reduced_unit_layout(joint, plan.parameters);
}

// All declared axes use the same general profile-layout and role builders.
template<Kind Family, std::size_t P, std::size_t Q, std::size_t ExternalColumns,
         bool Intercept = true, std::size_t LinearArOrder = 0>
inline constexpr auto fixed_unit_layout = scivolapi::detail::profile::unit_layout(
    ModelConfig{Family, P, Q, ExternalColumns, 0, Intercept, true, LinearArOrder});

} // namespace scivolapi::detail::profile
