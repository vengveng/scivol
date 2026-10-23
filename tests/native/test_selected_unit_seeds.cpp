#include "../../core/src/models/unit_map.hpp"
#include "../../core/src/math/mean/arma_mean_equation.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr,"check failed at %d\n",__LINE__); std::abort(); } } while (false)
#include <cmath>

using namespace scivolapi;
using namespace scivolapi::detail;
namespace units=scivolapi::detail::unit_map;
template<class Number>bool near(Number value,double y) {
    const double x=static_cast<double>(value);
    return std::isfinite(x) && std::abs(x-y)<=2e-12*(1+std::abs(y));
}
// Equation-owner C smoke only. Production range assertions live in the public
// consumer tests; this adapter delegates every coefficient to MapActions.
template<class Actions>struct EquationBasis {
    using number_type=units::implementation::Quad;
    static constexpr bool identity=false,mapped=true;
    ActiveSensitivityBasis axes;const Actions* actions;
    std::size_t width(std::size_t n)const{return axes.width(n);}
    std::size_t active_axis(std::size_t i)const{return axes.active_axis(i);}
    number_type first_product(std::size_t row,std::size_t a,number_type x)const{
        const auto e=actions->first(row,axes.ambient_axis(a),x);CHECK(e.valid);return e.value;
    }
    number_type second_product(std::size_t row,std::size_t a,std::size_t b,number_type x)const{
        const auto e=actions->second(row,axes.ambient_axis(a),axes.ambient_axis(b),x);CHECK(e.valid);return e.value;
    }
};
template<class Owner>struct Mapped {
    std::vector<std::max_align_t> storage;
    units::PreparedPoint<2,Owner> point;
    Mapped(const units::Plan<Owner>& plan,std::span<const double> theta){
        const auto size=units::requirements(plan,JetRequest::H);CHECK(size.status==Status::ok);
        storage.resize((size.bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
        CHECK(units::prepare(plan,theta,{reinterpret_cast<std::byte*>(storage.data()),size.bytes},point)==Status::ok);
    }
};
int main() {
    constexpr units::LayoutSpec power{5,units::no_axis,{},units::ScaleCoordinate::power,1,4,{}};
    units::Plan<units::FixedLayout<power>> plan;
    CHECK(units::make_plan(units::FixedLayout<power>{},{3,{},1},plan)==Status::ok);
    const std::array<double,5> theta{7,.8,2,3,1.4};
    const std::array<std::size_t,3> indices{4,2,1};
    const auto axes=ActiveSensitivityBasis::select(theta.size(),indices); CHECK(axes);
    Mapped mapped(plan,theta);
    const auto actions=units::map_actions<units::implementation::Quad,true>(mapped.point);CHECK(actions);
    const EquationBasis seeds{*axes,&*actions};
    const double a=std::pow(3.,-theta[4]),l=std::log(3.);
    CHECK(near(seeds.first_product(1,0,1),-theta[1]*a*l));
    CHECK(near(seeds.second_product(1,0,2,1),-a*l));
    CHECK(near(seeds.second_product(1,0,0,1),theta[1]*a*l*l));
    // Existing product owner consumes C even when the coefficient is fixed in
    // public units. Previous first/second rows are zero, h=2 is fixed.
    std::array<units::implementation::Quad,3> first{},previous{};
    std::array<units::implementation::Quad,6> second{},previous_second{};
    vlk::experiment::generated::arma_garch_equation::add_ma_first(first.data(),3,1,theta[1],2.,previous.data(),seeds);
    vlk::experiment::generated::arma_garch_equation::add_ma_second(second.data(),3,1,theta[1],previous.data(),previous_second.data(),seeds,2.);
    CHECK(near(first[0],2*theta[1]*a*l));
    CHECK(near(second[0],-2*theta[1]*a*l*l));
    CHECK(near(second[2],2*a*l));
    constexpr units::LayoutSpec logspec{6,units::no_axis,{},units::ScaleCoordinate::log_variance,0,units::no_axis,{2,3}};
    units::Plan<units::FixedLayout<logspec>> logplan;
    CHECK(units::make_plan(units::FixedLayout<logspec>{},{3,{},1},logplan)==Status::ok);
    const std::array<double,6> logtheta{.2,1,.1,.2,.3,9};
    const std::array<std::size_t,2> beta{4,2};
    const auto selected=ActiveSensitivityBasis::select(6,beta); CHECK(selected);
    Mapped logmapped(logplan,logtheta);
    const auto logactions=units::map_actions<units::implementation::Quad,true>(logmapped.point);CHECK(logactions);
    const EquationBasis logseeds{*selected,&*logactions};
    CHECK(near(logseeds.first_product(0,0,1),2*l));
    CHECK(logseeds.second_product(0,0,1,1)==0);
    // The selected contraction, not an isolated 1/r^2, needs double range.
    constexpr units::LayoutSpec variance{2,units::no_axis,{},units::ScaleCoordinate::variance,0,units::no_axis,{}};
    units::Plan<units::FixedLayout<variance>> extreme;
    CHECK(units::make_plan(units::FixedLayout<variance>{},{1e200,{},1},extreme)==Status::ok);
    const std::array<double,2> vtheta{1e100,1};
    const std::array<std::size_t,1> vaxis{0};
    const auto vb=ActiveSensitivityBasis::select(2,vaxis); CHECK(vb);
    Mapped vmapped(extreme,vtheta);
    const auto vactions=units::map_actions<units::implementation::Quad,true>(vmapped.point);CHECK(vactions);
    const EquationBasis vs{*vb,&*vactions};
    const double contracted=static_cast<double>(vs.first_product(0,0,1e300));
    CHECK(contracted>0 && std::abs(contracted/1e-100-1)<2e-12);
}
