#include "../../core/src/models/mean/location_scale_observation_assembly.hpp"
#include "../../core/src/models/unit_map.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
namespace generated=vlk::experiment::generated;
namespace model=vlk::experiment::model;
namespace units=scivolapi::detail::unit_map;
template<class Observation> constexpr bool has_factors=requires(const Observation& x) {x.residual_first_factors;};
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"normal factor line %d\n",__LINE__);std::abort();}}while(false)
template<class Number> void close(const Number& actual,double expected) {
    CHECK(vlk::experiment::finite_sensitivity(actual));
    CHECK(actual!=0 && expected!=0);
    CHECK(vlk::experiment::sensitivity_magnitude(actual/Number(expected)-Number(1))<=Number(2e-13));
}

template<unsigned Request,class Number>
void witness(int residual_exponent,int variance_exponent,bool negative,bool raw_aware=true) {
    constexpr bool F=bool(Request&1),G=bool(Request&2),H=bool(Request&4);
    const double residual=std::ldexp(negative?-1.0:1.0,residual_exponent);
    const double variance=std::ldexp(1.0,variance_exponent),x=0x1p1000;
    const auto prepared=generated::NormalDensity::prepare<F,G,H,true>(nullptr,1);
    auto observation=[&] {
        if(raw_aware)return generated::observe_with_representable_raw_square<
            generated::NormalDensity,F,G,H,true>(prepared,variance,residual);
        return generated::NormalDensity::observe<F,G,H,true>(prepared,variance,residual,residual*residual);
    }();
    observation=model::normalized_density_derivatives(observation);
    if constexpr(F)CHECK(std::isfinite(observation.value));
    if constexpr(G || H) {
        CHECK(observation.residual_first_factors.numerator==residual);
        CHECK(observation.residual_first_factors.denominator==variance);
        const double expected_g=std::ldexp(negative?1.0:-1.0,residual_exponent+1000-variance_exponent);
        const double expected_h=std::ldexp(1.0,2000-variance_exponent);
        const Number first(-x),zero(0);
        if constexpr(G) {
            close(model::compose_normalized_first(observation,first,zero),expected_g);
            close(model::compose_active_first(observation,first,zero,true,false),expected_g);
            const auto bound=model::absolute_density_derivatives(observation);
            const Number magnitude=model::compose_normalized_first(bound,Number(x),zero);
            close(magnitude,std::abs(expected_g));
            CHECK(magnitude>0);
        }
        if constexpr(H) {
            close(model::compose_normalized_second(observation,first,zero,first,zero,zero,zero),expected_h);
            close(model::compose_normalized_second(observation,zero,zero,zero,zero,first,zero),expected_g);
            close(model::compose_active_second(observation,zero,zero,zero,zero,first,zero,true,false),expected_g);
            close(model::residual_variance_product(observation,first,Number(variance)),-expected_g);
            close(model::residual_first_product<true>(observation,first),-expected_g);
            const auto bound=model::absolute_density_derivatives(observation);
            close(model::residual_variance_product(bound,Number(x),Number(variance)),std::abs(expected_g));
        }
    }
}

template<class Number> void masks() {
    const auto run=[&]<unsigned Request>() {
        for(bool negative:{false,true}) {
            witness<Request,Number>(-1000,1000,negative);
            witness<Request,Number>(-511,1023,negative);
            witness<Request,Number>(-511,1023,negative,false);
        }
    };
    run.template operator()<1>();run.template operator()<2>();run.template operator()<3>();
    run.template operator()<4>();run.template operator()<5>();run.template operator()<6>();run.template operator()<7>();
}

template<class Number> void raw_subnormal() {
    if constexpr(std::is_same_v<Number,long double>
        && std::numeric_limits<Number>::min_exponent==std::numeric_limits<double>::min_exponent) {
        const auto prepared=generated::NormalDensity::prepare<false,true,false,true>(nullptr,1);
        const auto observation=generated::observe_with_representable_raw_square<
            generated::NormalDensity,false,true,false,true>(prepared,0x1p1000,0x1p-1074);
        CHECK(!vlk::experiment::finite_sensitivity(model::residual_first_product(observation,Number(-0x1p1000))));
    } else witness<7,Number>(-1074,1000,false);
}

void ordinary_fields() {
    static_assert(std::is_same_v<decltype(model::scaled_derivative_product(0.5,1,2.0)),double>);
    static_assert(std::is_same_v<decltype(model::scaled_derivative_product(0.5,1,1,2.0,2.0)),double>);
    CHECK(model::scaled_derivative_product(0.5,1,2.0)==0.25);
    CHECK(model::scaled_derivative_product(0.5,1,1,2.0,2.0)==0.125);
    using Density=generated::NormalDensity;
    const auto prepared=Density::prepare<true,true,true,true>(nullptr,1);
    const auto raw=generated::observe_with_representable_raw_square<Density,true,true,true,true>(prepared,4.0,1.0);
    CHECK(raw.residual_first_factors.numerator==0);
    CHECK(raw.residual_gradient==0.25 && raw.residual_hessian==0.25);
    CHECK(raw.residual_variance_hessian==-0.0625);
    const auto normalized=model::normalized_density_derivatives(raw);
    CHECK(model::residual_first_product(normalized,-2.0)==-0.5);
    const auto zero=generated::observe_with_representable_raw_square<Density,true,true,true,true>(prepared,0x1p1000,0.0);
    CHECK(zero.residual_first_factors.numerator==0);
    const auto value=generated::observe_with_representable_raw_square<Density,true,false,false,true>(prepared,4.0,1.0);
    static_assert(!has_factors<decltype(value)>);
}
}

int main() {
    ordinary_fields();
    masks<double>();masks<long double>();masks<units::implementation::Quad>();masks<units::implementation::Octuple>();
    raw_subnormal<double>();raw_subnormal<long double>();
    raw_subnormal<units::implementation::Quad>();raw_subnormal<units::implementation::Octuple>();
    std::puts("normal factored score shared-owner diagnostics passed");
}
