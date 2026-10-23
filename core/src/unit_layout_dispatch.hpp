#pragma once
#include "program.hpp"
#include "models/unit_layout_roles.hpp"
#include "models/mean/unit_layout_dispatch.hpp"
#include "models/density_dispatch.hpp"
#include "models/joint/layout.hpp"
#include "models/power/egarch_layout.hpp"
#include "models/power/aparch_layout.hpp"
#include "models/joint_power/layout.hpp"
#include "models/profile/unit_layout.hpp"

namespace scivolapi::detail::unit_layout_dispatch {
namespace u=unit_map;
namespace native=vlk::experiment::runtime;
namespace mean=vlk::experiment::mean;

template<class Callback>
decltype(auto) profiled(profile_backend::ModelConfig config,const u::Layout& admitted,
    Callback&& callback) noexcept {
    const auto run=[&](auto owner) -> decltype(auto) {
        return unit_layout_roles::admit_owner(owner,admitted,callback);
    };
    switch(profile_backend::dispatch_kind(config)) {
    case 1: {
        constexpr auto units=scivolapi::detail::profile::fixed_unit_layout<
            mean_backend::Kind::linear,0,0,3>;
        return run(u::FixedLayout<units.roles>{});
    }
    case 2: {
        constexpr auto units=scivolapi::detail::profile::fixed_unit_layout<
            mean_backend::Kind::arma,1,1,0>;
        return run(u::FixedLayout<units.roles>{});
    }
    default: return run(u::RuntimeLayout{admitted});
    }
}

template<class Density,std::size_t P,std::size_t Q,bool Leverage,class Callback>
decltype(auto) zero_garch(Callback& callback) noexcept {
    constexpr auto layout=native::arma_garch_layout(0,0,P,Q,Leverage,false);
    static_assert(layout.valid);
    constexpr auto units=unit_layout_roles::variance(layout.parameter_count+Density::shape_parameters,layout.omega_index);
    return callback(u::FixedLayout<units.roles>{});
}
template<class Density,class Callback>
decltype(auto) garch(backend::ModelConfig c,Callback&& callback) noexcept {
    if(backend::dispatch_kind(c,false)!=1)return callback(u::RuntimeLayout{backend::unit_layout(c)});
    const auto selected=[&]<bool Leverage>() -> decltype(auto) {
        if(c.p==1 && c.q==1)return zero_garch<Density,1,1,Leverage>(callback);
        if(c.p==1 && c.q==2)return zero_garch<Density,1,2,Leverage>(callback);
        if(c.p==2 && c.q==1)return zero_garch<Density,2,1,Leverage>(callback);
        if(c.p==2 && c.q==2)return zero_garch<Density,2,2,Leverage>(callback);
        return callback(u::RuntimeLayout{backend::unit_layout(c)});
    };
    return c.leverage?selected.template operator()<true>():selected.template operator()<false>();
}
template<class Density,std::size_t Columns,bool Intercept,bool Leverage,class Callback>
decltype(auto) linear_garch(Callback& callback) noexcept {
    constexpr auto layout=native::joint_garch::linear_layout(Columns,1,1,Leverage);
    static_assert(layout.valid);
    constexpr auto units=unit_layout_roles::variance(layout.parameter_count+Density::shape_parameters,
        layout.omega_index,Intercept,{std::size_t(Intercept),Columns-std::size_t(Intercept)});
    return callback(u::FixedLayout<units.roles>{});
}
template<class Density,class Callback>
decltype(auto) joint_garch(joint_backend::ModelConfig c,Callback&& callback) noexcept {
    if(joint_backend::dispatch_kind(c)!=1)return callback(u::RuntimeLayout{joint_backend::unit_layout(c)});
    const auto selected=[&]<bool Leverage>() -> decltype(auto) {
        if(c.kind==joint_backend::Kind::arma) {
            constexpr auto layout=native::fixed_arma_garch_layout<1,1,1,1,Leverage>;
            constexpr auto units=unit_layout_roles::variance(layout.parameter_count+Density::shape_parameters,layout.omega_index,true);
            return callback(u::FixedLayout<units.roles>{});
        }
        if(c.design_width==0 && !c.intercept) {
            backend::ModelConfig zero{c.p_vol,c.q_vol,c.density,c.leverage};
            return garch<Density>(zero,callback);
        }
        const auto linear=[&]<bool Intercept>() -> decltype(auto) {
            if(c.design_width+std::size_t(Intercept)==4)return linear_garch<Density,4,Intercept,Leverage>(callback);
            if(c.design_width+std::size_t(Intercept)==7)return linear_garch<Density,7,Intercept,Leverage>(callback);
            return callback(u::RuntimeLayout{joint_backend::unit_layout(c)});
        };
        return c.intercept?linear.template operator()<true>():linear.template operator()<false>();
    };
    return c.leverage?selected.template operator()<true>():selected.template operator()<false>();
}
template<class Density,class Callback>
decltype(auto) constant_mean(mean_backend::ModelConfig c,Callback&& callback) noexcept {
    const bool linear=c.kind==mean_backend::Kind::linear;
    const auto plan=mean::layout(linear?c.design_width+std::size_t(c.intercept):c.p_ar+std::size_t(c.intercept),c.q_ma);
    return mean::dispatch_unit_layout<Density>(plan,mean_backend::dispatch_kind(c),linear,c.intercept,callback);
}
template<class Density,class Callback>
decltype(auto) power(power_backend::ModelConfig c,Callback&& callback) noexcept {
    if(power_backend::dispatch_kind(c)!=1)return callback(u::RuntimeLayout{power_backend::unit_layout(c)});
    if(c.kind==power_backend::Kind::aparch) {
        constexpr auto layout=native::aparch::fixed_layout<1,1,Density>;
        constexpr auto units=unit_layout_roles::power<true>(layout);
        return callback(u::FixedLayout<units.roles>{});
    }
    constexpr auto layout=native::egarch::fixed_layout<1,1,Density>;
    constexpr auto units=unit_layout_roles::power<false>(layout);
    return callback(u::FixedLayout<units.roles>{});
}
template<class Density,class Orders,bool Aparch,class Callback>
decltype(auto) fixed_joint_power(Callback& callback) noexcept {
    constexpr auto layout=native::joint_power::fixed_layout<Orders,Density,Aparch>;
    static_assert(layout.valid);
    constexpr u::Block exogenous=Orders::linear
        ?u::Block{std::size_t(Orders::intercept),Orders::columns-std::size_t(Orders::intercept)}:u::Block{};
    constexpr auto units=unit_layout_roles::power<Aparch>(layout,Orders::intercept,exogenous);
    return callback(u::FixedLayout<units.roles>{});
}
template<class Density,class Callback>
decltype(auto) joint_power(joint_power_backend::ModelConfig c,Callback&& callback) noexcept {
    const auto dispatch=joint_power_backend::dispatch_kind(c);
    if(dispatch<=0)return callback(u::RuntimeLayout{joint_power_backend::unit_layout(c)});
    const auto selected=[&]<bool Aparch>() -> decltype(auto) {
        if(dispatch==1) {
            using Orders=native::joint_power::Orders<false,true,0,1,1,1,1>;
            return fixed_joint_power<Density,Orders,Aparch>(callback);
        }
        const auto linear=[&]<bool Intercept>() -> decltype(auto) {
            using Orders=native::joint_power::Orders<true,Intercept,4,0,0,1,1>;
            return fixed_joint_power<Density,Orders,Aparch>(callback);
        };
        return c.intercept?linear.template operator()<true>():linear.template operator()<false>();
    };
    return c.kind==joint_power_backend::Kind::aparch?selected.template operator()<true>():selected.template operator()<false>();
}
template<class Config,class Callback>
decltype(auto) dispatch(Config c,Callback&& callback) noexcept {
    return joint_math::dispatch_density(c.density,[&]<class Density>() -> decltype(auto) {
        if constexpr(std::is_same_v<Config,backend::ModelConfig>)return garch<Density>(c,callback);
        else if constexpr(std::is_same_v<Config,mean_backend::ModelConfig>)return constant_mean<Density>(c,callback);
        else if constexpr(std::is_same_v<Config,joint_backend::ModelConfig>)return joint_garch<Density>(c,callback);
        else if constexpr(std::is_same_v<Config,power_backend::ModelConfig>)return power<Density>(c,callback);
        else return joint_power<Density>(c,callback);
    });
}
template<class Callback>
decltype(auto) dispatch(const program::Configuration& configuration,Callback&& callback) noexcept {
    return std::visit([&](auto c) -> decltype(auto) {
        if constexpr(std::is_same_v<decltype(c),dependence_backend::ModelConfig>)
            return callback(u::RuntimeLayout{program::unit_layout(configuration)});
        else return dispatch(c,callback);
    },configuration);
}
template<class Callback>
decltype(auto) dispatch(const program::Configuration& configuration,const u::Layout& admitted,
    Callback&& callback) noexcept {
    return dispatch(configuration,[&](auto owner) -> decltype(auto) {
        return unit_layout_roles::admit_owner(owner,admitted,callback);
    });
}
}
