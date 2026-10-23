#pragma once
#include "unit_map.hpp"

namespace scivolapi::detail::unit_layout_roles {
struct DispatchCounts {
    std::array<std::size_t,3> fixed_preparations{},runtime_preparations{};
    std::array<std::size_t,3> fixed_pullbacks{},runtime_pullbacks{};
    std::array<std::size_t,3> fixed_selected_tiers{},runtime_selected_tiers{};
};
constexpr unit_map::Layout without_generated(unit_map::Layout layout,std::size_t generated) noexcept {
    if(!layout.valid || generated>layout.roles.exogenous.count)return {};
    layout.roles.exogenous.begin+=generated;
    layout.roles.exogenous.count-=generated;
    return unit_map::make_layout(layout.roles);
}
constexpr bool matches_model_layout(unit_map::Layout model,unit_map::Layout admitted) noexcept {
    return admitted.valid && admitted.roles.exogenous.begin>=model.roles.exogenous.begin
        && without_generated(model,admitted.roles.exogenous.begin-model.roles.exogenous.begin)==admitted;
}
template<unit_map::LayoutSpec Spec,std::size_t Generated=0,class Callback>
decltype(auto) dispatch_generated(const unit_map::Layout& admitted,Callback& callback) noexcept {
    constexpr auto candidate=without_generated(unit_map::make_layout(Spec),Generated);
    if(candidate==admitted)return callback(unit_map::FixedLayout<candidate.roles>{});
    if constexpr(Generated<Spec.exogenous.count)return dispatch_generated<Spec,Generated+1>(admitted,callback);
    else return callback(unit_map::FixedLayout<Spec>{}); // Exact rebind rejects a mismatch.
}
template<class Owner,class Callback>
decltype(auto) admit_owner(Owner,const unit_map::Layout& admitted,Callback& callback) noexcept {
    if constexpr(std::is_same_v<Owner,unit_map::RuntimeLayout>)return callback(unit_map::RuntimeLayout{admitted});
    else return dispatch_generated<Owner::value.roles>(admitted,callback);
}
// Runtime and fixed dispatch bind the same model-owned parameter layout here.
constexpr unit_map::Layout variance(std::size_t parameters,std::size_t omega,
    bool intercept=false,unit_map::Block exogenous={}) noexcept {
    return unit_map::make_layout({parameters,intercept?0:unit_map::no_axis,exogenous,
        unit_map::ScaleCoordinate::variance,omega,unit_map::no_axis,{}});
}
template<bool Aparch,class Layout>
constexpr unit_map::Layout power(const Layout& layout,bool intercept=false,
    unit_map::Block exogenous={}) noexcept {
    const auto omega=[&] {if constexpr(requires{layout.omega_index;})return layout.omega_index;else return std::size_t(0);}();
    if constexpr(Aparch)
        return unit_map::make_layout({layout.parameters,intercept?0:unit_map::no_axis,exogenous,
            unit_map::ScaleCoordinate::power,omega,layout.delta_index,{}});
    else {
        const auto q=[&] {if constexpr(requires{layout.q_vol;})return layout.q_vol;else return layout.q;}();
        return unit_map::make_layout({layout.parameters,intercept?0:unit_map::no_axis,exogenous,
            unit_map::ScaleCoordinate::log_variance,omega,unit_map::no_axis,{layout.beta_start,q}});
    }
}
}
