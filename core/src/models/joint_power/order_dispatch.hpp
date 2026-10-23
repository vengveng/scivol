#pragma once
#include "layout.hpp"
#include "../../joint_power_backend.hpp"

namespace scivolapi::detail::joint_power_backend::program {
// Metadata selection shared by numerical consumers and preparation planning.
template<class Callback>
decltype(auto) orders(ModelConfig c,Callback&& callback) {
    namespace math=vlk::experiment::runtime::joint_power;
    const auto select=[&]<bool Intercept>() -> decltype(auto) {
        if (c.mean==MeanKind::arma) {
            if constexpr (!Intercept)
                if (c.p_ar==0u && c.q_ma==0u)
                    return callback.template operator()<math::Orders<false,false,0u,0u,0u>>();
            if (dispatch_kind(c)==1) return callback.template operator()<math::Orders<false,Intercept,0,1,1,1,1>>();
            return callback.template operator()<math::Orders<false,Intercept>>();
        }
        if constexpr (!Intercept)
            if (c.design_width==0u)
                return callback.template operator()<math::Orders<true,false,0u>>();
        if (dispatch_kind(c)==2) return callback.template operator()<math::Orders<true,Intercept,4,0,0,1,1>>();
        return callback.template operator()<math::Orders<true,Intercept>>();
    };
    return c.intercept ? select.template operator()<true>() : select.template operator()<false>();
}
}
