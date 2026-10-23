#include "preparation_plan.hpp"
#include "models/moment_payloads.hpp"
#include "models/joint_power/order_dispatch.hpp"

namespace scivolapi::detail::preparation_plan {
namespace {
unsigned map_orders(unsigned mask) noexcept {
    return (mask&4)?7u:(mask&2)?3u:(mask&1)?1u:0u;
}
template<class Config> constexpr bool univariate=!
    std::is_same_v<Config,dependence_backend::ModelConfig>;

template<class Config>
unsigned residual_orders(Config c,const ActiveSensitivityBasis* selected) noexcept {
    if constexpr(std::is_same_v<Config,mean_backend::ModelConfig>) {
        const auto means=mean_backend::unit_layout(c).roles.scale_axis;
        return means && (!selected || !selected->fixed_block(0,means))?7u:0u;
    } else if constexpr(std::is_same_v<Config,joint_backend::ModelConfig>) {
        return c.kind==joint_backend::Kind::arma || joint_backend::unit_layout(c).roles.scale_axis?7u:0u;
    } else if constexpr(std::is_same_v<Config,joint_power_backend::ModelConfig>) {
        return joint_power_backend::program::orders(c,[]<class Orders>() {
            constexpr bool active=[] {
                if constexpr(requires {Orders::residual_parameters;})return Orders::residual_parameters;
                else return !(Orders::linear && Orders::columns==0u);
            }();
            return active?6u:0u;
        });
    } else return 0;
}

template<class Density,class Config>
Enrollment typed(Config c,unsigned mask,bool chart,const ActiveSensitivityBasis* selected) noexcept {
    Enrollment out;
    const auto n=program::parameter_count(program::Configuration{c});
    if(!mask || mask>7 || n<Density::shape_parameters || (selected && !selected->valid_for(n)))return out;
    out.mask=mask;
    out.options.density=static_cast<PreparedDensity>(c.density);
    out.options.residual_orders=residual_orders(c,selected);
    out.options.shape_axes.fill(0);
    constexpr unsigned full=(1u<<Density::shape_parameters)-1u;
    unsigned density_shapes=0;
    for(unsigned shape=0;shape<Density::shape_parameters;++shape)
        if(!selected || selected->depends(n-Density::shape_parameters+shape))density_shapes|=1u<<shape;
    for(unsigned order=0;order<3;++order)if(mask&(1u<<order))out.options.shape_axes[order]=density_shapes;

    MomentUse use=MomentUse::none;
    unsigned moments=0;std::size_t lags=0;
    if constexpr(std::is_same_v<Config,backend::ModelConfig> || std::is_same_v<Config,joint_backend::ModelConfig>) {
        if(c.leverage && chart){use=MomentUse::negative_second;moments=map_orders(mask);}
    } else if constexpr(std::is_same_v<Config,power_backend::ModelConfig>) {
        lags=c.p;
        if(c.kind==power_backend::Kind::aparch){use=MomentUse::power;moments=chart?map_orders(mask):1u;}
        else {use=MomentUse::absolute;moments=mask;}
    } else if constexpr(std::is_same_v<Config,joint_power_backend::ModelConfig>) {
        lags=c.p_vol;
        if(c.kind==joint_power_backend::Kind::aparch){use=MomentUse::power;moments=chart?map_orders(mask):1u;}
        else {use=MomentUse::wide_absolute;moments=mask;}
    }
    out.options.moments=moment_reservations<Density>(moments,use,lags);
    for(unsigned order=0;order<3;++order)if(moments&(1u<<order)) {
        // These canonical moment/map APIs genuinely consume every shape axis.
        // Absolute-moment consumers instead share the selected density basis.
        if(use==MomentUse::negative_second || use==MomentUse::power)out.options.shape_axes[order]|=full;
    }
    out.status=Status::ok;
    return out;
}
}

bool enabled(const program::Configuration& configuration) noexcept {
    return std::visit([](auto c) {
        if constexpr(univariate<decltype(c)>)return c.density>=1 && c.density<=7;
        else return false;
    },configuration);
}
Enrollment make(const program::Configuration& configuration,unsigned mask,bool chart,
    const ActiveSensitivityBasis* selected) noexcept {
    return std::visit([&](auto c)->Enrollment {
        if constexpr(univariate<decltype(c)>) {
            if(c.density<1 || c.density>7)return {};
            return with_prepared_density(static_cast<PreparedDensity>(c.density),[&]<class Density> {
                return typed<Density>(c,mask,chart,selected);
            });
        } else return {};
    },configuration);
}
}
