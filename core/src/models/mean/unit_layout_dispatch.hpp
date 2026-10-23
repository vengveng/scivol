#pragma once
#include "layout.hpp"
#include "../unit_layout_roles.hpp"
#include <span>
#include <type_traits>

namespace vlk::experiment::mean {
template<std::size_t Columns, bool Intercept> struct ExternalLinearDesign;
template<std::size_t P, bool Intercept> struct ConditionalArmaDesign;

template<bool Linear, bool Intercept, std::size_t Columns = std::dynamic_extent,
    std::size_t P = std::dynamic_extent, std::size_t Q = std::dynamic_extent>
struct SelectedMeanKernel {
    static constexpr bool linear = Linear, intercept = Intercept;
    static constexpr bool fixed = Columns != std::dynamic_extent && Q != std::dynamic_extent;
    static_assert(!fixed || Linear || P != std::dynamic_extent);
    static constexpr std::size_t columns = Columns, p = P, q = Q;
    static constexpr auto fixed_layout = [] {
        if constexpr(fixed) return layout(Columns,Q);
        else return Layout{};
    }();
    static constexpr std::size_t width = fixed ? fixed_layout.width : std::dynamic_extent;
    using Design = std::conditional_t<Linear,ExternalLinearDesign<Columns,Intercept>,
        ConditionalArmaDesign<P,Intercept>>;
    static constexpr Layout model_layout(const Layout& supplied) noexcept {
        if constexpr(fixed) return fixed_layout;
        else return supplied;
    }
};

template<class Kernel, class Owner> struct SelectedMeanBinding {
    using kernel_type = Kernel;
    Owner owner;
};

// Refining generated-column roles must retain the already selected kernel.
template<class Kernel, class Owner, class Callback>
decltype(auto) admit_selected_binding(SelectedMeanBinding<Kernel,Owner> binding,
    const scivolapi::detail::unit_map::Layout& admitted,Callback&& callback) noexcept {
    const auto refined = [&](auto owner) -> decltype(auto) {
        return callback(SelectedMeanBinding<Kernel,decltype(owner)>{owner});
    };
    return scivolapi::detail::unit_layout_roles::admit_owner(binding.owner,admitted,refined);
}

template<class Density,class Callback>
decltype(auto) dispatch_selected_binding(const Layout& plan,int dispatch,bool linear,
    bool intercept,Callback&& callback) noexcept {
    namespace u=scivolapi::detail::unit_map;
    namespace roles=scivolapi::detail::unit_layout_roles;
    const auto selected=[&]<bool Intercept>() -> decltype(auto) {
        if(linear && dispatch==1) {
            constexpr auto fixed=layout(4,0);
            constexpr auto units=roles::variance(fixed.width+Density::shape_parameters,fixed.width-1,
                Intercept,{std::size_t(Intercept),4-std::size_t(Intercept)});
            using Kernel = SelectedMeanKernel<true,Intercept,4,0,0>;
            return callback(SelectedMeanBinding<Kernel,u::FixedLayout<units.roles>>{});
        }
        if(!linear && dispatch==2) {
            constexpr auto fixed=layout(2,1);
            constexpr auto units=roles::variance(fixed.width+Density::shape_parameters,fixed.width-1,true);
            using Kernel = SelectedMeanKernel<false,true,2,1,1>;
            return callback(SelectedMeanBinding<Kernel,u::FixedLayout<units.roles>>{});
        }
        const u::RuntimeLayout owner{roles::variance(plan.width+Density::shape_parameters,
            plan.width-1,intercept,linear?u::Block{std::size_t(intercept),plan.columns-std::size_t(intercept)}:u::Block{})};
        if(linear) return callback(SelectedMeanBinding<SelectedMeanKernel<true,Intercept>,u::RuntimeLayout>{owner});
        return callback(SelectedMeanBinding<SelectedMeanKernel<false,Intercept>,u::RuntimeLayout>{owner});
    };
    return intercept?selected.template operator()<true>():selected.template operator()<false>();
}
template<class Density,class Callback>
decltype(auto) dispatch_unit_layout(const Layout& plan,int dispatch,bool linear,
    bool intercept,Callback&& callback) noexcept {
    return dispatch_selected_binding<Density>(plan,dispatch,linear,intercept,
        [&](auto binding) -> decltype(auto) {return callback(binding.owner);});
}
}
