#include "models/power/egarch_jet.hpp"
#include "models/joint_power/egarch_chart.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {
namespace native = vlk::experiment::runtime;
namespace gen = vlk::experiment::generated;
namespace chart = native::egarch_transform;
namespace joint = native::joint_power;
constexpr double sentinel = -0x1.23456789abcp+20;
std::size_t checks = 0;

void check(bool valid, int line) {
    ++checks;
    if (!valid) {
        std::fprintf(stderr,"EGARCH chart check failed at line %d\n",line);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__),__LINE__)

void equal(std::span<const double> left,std::span<const double> right) {
    CHECK(left.size()==right.size());
    for (std::size_t i=0;i<left.size();++i)
        CHECK(std::isfinite(left[i]) && std::isfinite(right[i])
            && std::abs(left[i]-right[i])<=2e-13*(1+std::max(std::abs(left[i]),std::abs(right[i]))));
}

template<class Shape> struct DensityWidth { static constexpr auto shape_parameters=Shape::parameters; };
template<std::size_t P,std::size_t Q,class Shape>
inline constexpr auto zero_layout=[] {
    native::egarch::Layout result;
    if (!native::egarch::make_layout<DensityWidth<Shape>>(P,Q,result)) return native::egarch::Layout{};
    return result;
}();

struct Result {
    bool valid=false;
    std::vector<double> theta,gradient,hessian;
    explicit Result(std::size_t width):theta(width,sentinel),gradient(width,sentinel),hessian(width*width,sentinel) {}
};

void seed_derivatives(chart::Scratch& scratch,std::size_t width,unsigned mask) {
    if (mask&6) for (std::size_t i=0;i<width;++i) scratch.theta_gradient[i]=.125*(i+1);
    if (mask&4) for (std::size_t i=0;i<width;++i) for (std::size_t j=0;j<width;++j)
        scratch.theta_hessian[i*width+j]=(i==j ? 1. : 0.)+.25/(1+i+j);
}

template<class Shape,class Layout>
Result zero_request(const Layout& layout,std::span<const double> z,unsigned mask) {
    const std::size_t width=layout.parameters;
    Result result(width);
    const auto bytes=chart::workspace_bytes(layout,0,mask);
    std::vector<double> arena((bytes+sizeof(double)-1)/sizeof(double));
    chart::Scratch scratch;
    if (!chart::bind_scratch(layout,0,mask,arena.data(),bytes,scratch)
        || !chart::map_coordinates<Shape>(layout,z.data(),(mask&6)!=0,scratch)) return result;
    std::copy_n(scratch.theta,width,result.theta.data());
    seed_derivatives(scratch,width,mask);
    if (mask&2) chart::pullback_gradient<Shape>(layout,scratch,scratch.theta_gradient,result.gradient.data());
    if ((mask&4) && !chart::pullback_hessian<Shape>(layout,z.data(),scratch,result.hessian.data())) return result;
    result.valid=true;
    return result;
}

template<const joint::EgarchLayout* Fixed,class Shape>
Result joint_request(const joint::EgarchLayout& supplied,std::span<const double> z,unsigned mask) {
    const auto& layout=joint::chart_layout<Fixed>(supplied);
    Result result(layout.parameters);
    const auto bytes=joint::egarch_chart_bytes<Fixed>(layout,0,mask);
    std::vector<double> arena((bytes+sizeof(double)-1)/sizeof(double));
    joint::EgarchChartScratch scratch;
    if (!joint::bind_egarch_chart<Fixed>(layout,0,mask,arena.data(),bytes,scratch)
        || !joint::map_egarch_chart<Shape,Fixed>(layout,z.data(),(mask&6)!=0,scratch)) return result;
    std::copy_n(scratch.power.theta,layout.parameters,result.theta.data());
    seed_derivatives(scratch.power,layout.parameters,mask);
    if (mask&2) joint::egarch_gradient<Fixed>(layout,scratch,result.gradient.data());
    if ((mask&4) && !joint::egarch_hessian<Shape,Fixed>(layout,z.data(),scratch,result.hessian.data())) return result;
    result.valid=true;
    return result;
}

void equal_requests(const Result& runtime,const Result& fixed,unsigned mask) {
    CHECK(runtime.valid==fixed.valid);
    if (!runtime.valid) return;
    equal(runtime.theta,fixed.theta);
    equal(runtime.gradient,fixed.gradient);
    equal(runtime.hessian,fixed.hessian);
    if (!(mask&2)) CHECK(std::all_of(fixed.gradient.begin(),fixed.gradient.end(),[](double x) {return x==sentinel;}));
    if (!(mask&4)) CHECK(std::all_of(fixed.hessian.begin(),fixed.hessian.end(),[](double x) {return x==sentinel;}));
}

template<auto Fixed,class Shape>
void check_zero() {
    using Policy=chart::FixedPolicyLayout<Fixed>;
    static_assert(std::is_same_v<std::remove_cv_t<decltype(Policy::q)>,std::integral_constant<std::size_t,Fixed->q>>);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(Policy::parameters)>,std::integral_constant<std::size_t,Fixed->parameters>>);
    auto supplied=*Fixed;
    supplied.q=9; supplied.parameters=99; supplied.beta_start=87; supplied.model_parameters=93;
    const auto fixed=chart::policy_layout<Fixed>(supplied);
    CHECK(chart::policy_layout<nullptr>(*Fixed).q==Fixed->q);
    std::vector<double> z(Fixed->parameters);
    for (std::size_t i=0;i<z.size();++i) z[i]=.03*(i+1);
    for (unsigned mask=1;mask<=7;++mask) {
        CHECK(chart::workspace_bytes(fixed,0,mask)==chart::workspace_bytes(*Fixed,0,mask));
        equal_requests(zero_request<Shape>(*Fixed,z,mask),zero_request<Shape>(fixed,z,mask),mask);
    }
    z[Fixed->beta_start]=40.;
    for (unsigned mask=1;mask<=7;++mask) {
        const auto runtime=zero_request<Shape>(*Fixed,z,mask), selected=zero_request<Shape>(fixed,z,mask);
        CHECK(!runtime.valid && !selected.valid);
    }
    z[Fixed->beta_start]=std::numeric_limits<double>::quiet_NaN();
    CHECK(!zero_request<Shape>(fixed,z,1).valid);
}

template<class Orders,class Shape>
void check_joint() {
    constexpr auto* Fixed=&joint::fixed_layout<Orders,DensityWidth<Shape>,false>;
    using Policy=joint::FixedEgarchPowerChart<Fixed>;
    static_assert(std::is_same_v<std::remove_cv_t<decltype(Policy::q)>,std::integral_constant<std::size_t,Fixed->q_vol>>);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(Policy::parameters)>,std::integral_constant<std::size_t,Fixed->parameters>>);
    using MeanBlock=joint::FixedMeanChartBlock<Fixed,false>;
    static_assert(MeanBlock::order==Fixed->p_ar && MeanBlock::start==Fixed->ar_start);
    auto supplied=*Fixed;
    supplied.q_vol=9; supplied.parameters=99; supplied.beta_start=87;
    std::vector<double> z(Fixed->parameters);
    for (std::size_t i=0;i<z.size();++i) z[i]=.02*(i+1);
    for (unsigned mask=1;mask<=7;++mask) {
        CHECK(joint::egarch_chart_bytes<Fixed>(supplied,0,mask)==joint::egarch_chart_bytes<nullptr>(*Fixed,0,mask));
        equal_requests(joint_request<nullptr,Shape>(*Fixed,z,mask),joint_request<Fixed,Shape>(supplied,z,mask),mask);
    }
    z[Fixed->beta_start]=40.;
    for (unsigned mask=1;mask<=7;++mask)
        CHECK(!joint_request<nullptr,Shape>(*Fixed,z,mask).valid && !joint_request<Fixed,Shape>(supplied,z,mask).valid);
}
}

extern "C" __attribute__((noinline)) bool probe_zero_fixed_map(const double* z,bool derivatives,chart::Scratch& scratch) {
    constexpr auto* Fixed=&zero_layout<1,1,gen::NoShapeTransform>;
    return chart::map_coordinates<gen::NoShapeTransform>(chart::policy_layout<Fixed>(*Fixed),z,derivatives,scratch);
}
extern "C" __attribute__((noinline)) bool probe_zero_fixed_hessian(const double* z,chart::Scratch& scratch,double* hessian) {
    constexpr auto* Fixed=&zero_layout<1,1,gen::NoShapeTransform>;
    return chart::pullback_hessian<gen::NoShapeTransform>(chart::policy_layout<Fixed>(*Fixed),z,scratch,hessian);
}
extern "C" __attribute__((noinline)) void probe_zero_fixed_gradient(chart::Scratch& scratch,double* gradient) {
    constexpr auto* Fixed=&zero_layout<1,1,gen::NoShapeTransform>;
    chart::pullback_gradient<gen::NoShapeTransform>(chart::policy_layout<Fixed>(*Fixed),scratch,scratch.theta_gradient,gradient);
}
extern "C" __attribute__((noinline)) bool probe_joint_linear_fixed_hessian(const double* z,joint::EgarchChartScratch& scratch,double* hessian) {
    using Orders=joint::Orders<true,true,4,0,0,1,1>;
    constexpr auto* Fixed=&joint::fixed_layout<Orders,DensityWidth<gen::NoShapeTransform>,false>;
    return joint::egarch_hessian<gen::NoShapeTransform,Fixed>(*Fixed,z,scratch,hessian);
}

void exercise_exported_probes() {
    constexpr auto* Zero=&zero_layout<1,1,gen::NoShapeTransform>;
    const auto layout=chart::policy_layout<Zero>(*Zero);
    std::vector<double> z(Zero->parameters,.12),g(Zero->parameters,sentinel),h(Zero->parameters*Zero->parameters,sentinel);
    const auto bytes=chart::workspace_bytes(layout,0,7);
    std::vector<double> arena((bytes+7)/8);
    chart::Scratch scratch;
    CHECK(chart::bind_scratch(layout,0,7,arena.data(),bytes,scratch));
    CHECK(probe_zero_fixed_map(z.data(),false,scratch));
    CHECK(probe_zero_fixed_map(z.data(),true,scratch));
    seed_derivatives(scratch,Zero->parameters,7);
    probe_zero_fixed_gradient(scratch,g.data());
    CHECK(probe_zero_fixed_hessian(z.data(),scratch,h.data()));
    const auto reference=zero_request<gen::NoShapeTransform>(*Zero,z,7);
    equal(g,reference.gradient); equal(h,reference.hessian);

    using Orders=joint::Orders<true,true,4,0,0,1,1>;
    constexpr auto* Joint=&joint::fixed_layout<Orders,DensityWidth<gen::NoShapeTransform>,false>;
    std::vector<double> jz(Joint->parameters,.12),jh(Joint->parameters*Joint->parameters,sentinel);
    const auto jbytes=joint::egarch_chart_bytes<Joint>(*Joint,0,7);
    std::vector<double> jarena((jbytes+7)/8);
    joint::EgarchChartScratch jscratch;
    CHECK(joint::bind_egarch_chart<Joint>(*Joint,0,7,jarena.data(),jbytes,jscratch));
    CHECK(joint::map_egarch_chart<gen::NoShapeTransform,Joint>(*Joint,jz.data(),true,jscratch));
    seed_derivatives(jscratch.power,Joint->parameters,7);
    CHECK(probe_joint_linear_fixed_hessian(jz.data(),jscratch,jh.data()));
    equal(jh,joint_request<nullptr,gen::NoShapeTransform>(*Joint,jz,7).hessian);
}

int main() {
    exercise_exported_probes();
    check_zero<&zero_layout<1,1,gen::NoShapeTransform>,gen::NoShapeTransform>();
    check_zero<&zero_layout<1,1,gen::StudentTShapeTransform>,gen::StudentTShapeTransform>();
    check_zero<&zero_layout<3,2,gen::SkewTShapeTransform>,gen::SkewTShapeTransform>();
    check_joint<joint::Orders<false,true,0,1,1,1,1>,gen::NoShapeTransform>();
    check_joint<joint::Orders<true,true,4,0,0,1,1>,gen::StudentTShapeTransform>();
    check_joint<joint::Orders<false,false,0,2,0,3,2>,gen::SkewTShapeTransform>();
    check_joint<joint::Orders<true,false,4,0,0,2,3>,gen::NoShapeTransform>();
    std::printf("EGARCH shared-chart fixed/runtime checks: %zu\n",checks);
}
