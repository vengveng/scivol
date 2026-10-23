#include "math/skew_splice.hpp"
#include "models/garch_extended_density_policy.hpp"

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

namespace { std::size_t allocations = 0, checks = 0; }
void* operator new(std::size_t size) {
    ++allocations;
    if (void* result = std::malloc(size ? size : 1)) return result;
    std::abort();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* x) noexcept { std::free(x); }
void operator delete[](void* x) noexcept { std::free(x); }

namespace {
namespace splice = vlk::experiment::skew_splice;
using C = splice::Classification;
using S = splice::RequestStatus;
void check(bool condition, const char* expression, int line) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr,"skew splice line %d: %s\n",line,expression);
        std::exit(1);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__),#__VA_ARGS__,__LINE__)

void exact_family(double nu) {
    for (int m=-15;m<=15;++m) {
        if (!m) continue;
        const auto prepared = splice::prepare(nu,m/16.0);
        const double e = -(nu==4.0 ? 1.0 : 3.0)*m;
        const double h = nu==4.0 ? (256.0+m*m)/2.0 : 1024.0+3.0*m*m;
        const auto before = allocations;
        const auto exact = splice::classify(prepared,splice::VariancePoint{e,h});
        const auto down = splice::classify(prepared,splice::VariancePoint{e,std::nextafter(h,0.0)});
        const auto up = splice::classify(prepared,splice::VariancePoint{e,std::nextafter(h,INFINITY)});
        CHECK(allocations == before);
        CHECK(!prepared.tight_enclosure_attempted());
        CHECK(exact.classification == C::true_splice && exact.used_exact_fallback);
        CHECK(down.classification == (m>0 ? C::left : C::right));
        CHECK(up.classification == (m>0 ? C::right : C::left));
        CHECK(splice::classify(prepared,splice::VariancePoint{
            std::nextafter(e,-INFINITY),h}).classification==C::left);
        CHECK(splice::classify(prepared,splice::VariancePoint{
            std::nextafter(e,INFINITY),h}).classification==C::right);
        for (unsigned mask=1;mask<8;++mask)
            CHECK(exact.request_status(mask&4) == (mask&4 ? S::nondifferentiable : S::valid));
    }
    const auto prepared = splice::prepare(nu,0.5);
    const double e = nu==4.0 ? -1.0 : -3.0, h = nu==4.0 ? 2.5 : 19.0;
    for (int exponent : {-500,-100,0,100,500}) {
        const splice::VariancePoint point{std::ldexp(e,exponent),std::ldexp(h,2*exponent)};
        CHECK(splice::classify(prepared,point).classification == C::true_splice);
        CHECK(splice::classify(prepared,splice::VariancePoint{
            point.residual,std::nextafter(point.variance,0.0)}).classification == C::left);
        CHECK(splice::classify(prepared,splice::VariancePoint{
            point.residual,std::nextafter(point.variance,INFINITY)}).classification == C::right);
    }
    const double tiny = std::numeric_limits<double>::denorm_min();
    const double maximum = std::numeric_limits<double>::max();
    const auto coefficient = splice::detail::exact_coefficient(nu);
    const auto before = allocations;
    CHECK(splice::detail::exact_dyadic_polynomial(coefficient,.5,{-maximum,tiny})<0);
    CHECK(splice::detail::exact_dyadic_polynomial(coefficient,.5,{-tiny,maximum})>0);
    CHECK(allocations == before);
    CHECK(splice::classify(prepared,splice::VariancePoint{-maximum,tiny}).classification == C::left);
    CHECK(splice::classify(prepared,splice::VariancePoint{-tiny,maximum}).classification == C::right);
}

void exact_points() {
    exact_family(4.0);
    exact_family(6.0);
    const double tiny = std::numeric_limits<double>::denorm_min();
    const auto subnormal = splice::prepare(4.0,tiny);
    CHECK(splice::classify(subnormal,splice::VariancePoint{-tiny,0.5}).classification == C::left);
    CHECK(splice::classify(subnormal,splice::VariancePoint{-tiny,std::nextafter(0.5,1.0)}).classification == C::right);
    const auto six = splice::prepare(6.,tiny);
    CHECK(splice::classify(six,splice::VariancePoint{-tiny,.25}).classification==C::left);
    CHECK(splice::classify(six,splice::VariancePoint{-tiny,.5}).classification==C::right);
}

void ordinary_and_unresolved() {
    const double degrees[] = {std::nextafter(2.0,3.0),2.01,3.0,4.0,6.0,7.0,32.99,33.0,
        127.99,128.0,1e6,1e100,std::numeric_limits<double>::max()};
    for (double nu : degrees) {
        const auto before = allocations;
        const auto prepared = splice::prepare(nu,0.3);
        CHECK(allocations == before);
        CHECK(prepared.valid_shape() && prepared.enclosure_ready());
        CHECK(splice::classify(prepared,splice::VariancePoint{1.0,1.0}).classification == C::right);
        CHECK(splice::classify(prepared,splice::VariancePoint{0.0,1.0}).classification == C::right);
        const auto far_left = splice::classify(prepared,splice::VariancePoint{-4.0,1.0});
        const auto near_zero = splice::classify(prepared,splice::VariancePoint{-0x1p-100,1.0});
        CHECK(far_left.classification == C::left && !far_left.used_exact_fallback);
        CHECK(near_zero.classification == C::right && !near_zero.used_exact_fallback);
    }
    // MP-derived nearest binary residuals to the nu=7,lambda=1/2 mode.
    // Neither double is an exact splice. The enclosure deliberately cannot decide.
    const auto general = splice::prepare(7.0,0.5);
    const auto bounds = general.coefficient_enclosure();
    CHECK(bounds.lower > 0.0 && bounds.upper-bounds.lower < 1e-11);
    for (double residual : {-0x1.66d18152f2dccp-1,-0x1.66d18152f2dcdp-1}) {
        const auto uncertain = splice::classify(general,splice::VariancePoint{residual,1.0});
        CHECK(uncertain.classification == C::unresolved);
        CHECK(uncertain.reason == splice::UnresolvedReason::overlapping_enclosures);
        CHECK(uncertain.request_status(false) == S::valid);
        CHECK(uncertain.request_status(true) == S::numerical_failure);
    }
    const auto range = splice::classify(general,splice::LogVariancePoint{-1.0,1e100});
    CHECK(range.classification == C::unresolved);
    CHECK(range.reason == splice::UnresolvedReason::log_variance_range);
    CHECK(range.request_status(false) == S::valid && range.request_status(true) == S::numerical_failure);
}

void preparation_lifetime() {
    for (double nu : {std::nextafter(2.,3.),4.,6.,7.,33.,1e100,std::numeric_limits<double>::max()}) {
        const auto fg = splice::prepare(nu,.5,splice::Preparation::value_or_gradient);
        CHECK(fg.enclosure_ready() && !fg.tight_enclosure_attempted());
        const auto before = allocations;
        CHECK(splice::classify(fg,splice::VariancePoint{1.,1.}).classification==C::right);
        CHECK(splice::classify(fg,splice::VariancePoint{-4.,1.}).classification==C::left);
        CHECK(splice::classify(fg,splice::VariancePoint{-0x1p-100,1.}).classification==C::right);
        CHECK(!fg.tight_enclosure_attempted() && allocations==before);
    }
    const auto fg = splice::prepare(7.,.5,splice::Preparation::value_or_gradient);
    const auto h = splice::prepare(7.,.5,splice::Preparation::hessian);
    CHECK(!fg.tight_enclosure_attempted() && h.tight_enclosure_attempted());
    const splice::VariancePoint near{-0x1.66d18152f2dccp-1,1.};
    CHECK(splice::classify(fg,near).classification==C::unresolved);
    CHECK(fg.tight_enclosure_attempted());
    const auto bound = fg.coefficient_enclosure();
    CHECK(bound.lower==h.coefficient_enclosure().lower && bound.upper==h.coefficient_enclosure().upper);
    CHECK(!fg.refine_enclosure());
    CHECK(splice::classify(fg,near).classification==splice::classify(h,near).classification);
    CHECK(fg.coefficient_enclosure().lower==bound.lower && fg.coefficient_enclosure().upper==bound.upper);
    const double shape[]{7.,.5};
    namespace gen = vlk::experiment::generated;
    const auto f = gen::SkewTDensity::prepare<true,false,false,true>(shape,1);
    const auto g = gen::SkewTDensity::prepare<false,true,false,true>(shape,1);
    const auto hh = gen::SkewTDensity::prepare<false,false,true,true>(shape,1);
    CHECK(!f.splice.tight_enclosure_attempted() && !g.splice.tight_enclosure_attempted());
    CHECK(hh.splice.tight_enclosure_attempted());
}

void coordinates_and_symmetric() {
    const auto skew = splice::prepare(4.0,0.5);
    const splice::LogVariancePoint original{-1.0,std::log(2.5)};
    const auto log_point = splice::classify(skew,original);
    CHECK(log_point.classification == C::unresolved);
    CHECK(log_point.request_status(true) == S::numerical_failure);
    CHECK(log_point.request_status(false) == S::valid);
    CHECK(original.residual == -1.0 && original.log_variance == std::log(2.5));
    // log(19) and its binary neighbors are actual log-state inputs, not h=19.
    const auto nu6 = splice::prepare(6.,.5);
    const double log19 = std::log(19.);
    for (double x : {std::nextafter(log19,-INFINITY),log19,std::nextafter(log19,INFINITY)}) {
        const auto result = splice::classify(nu6,splice::LogVariancePoint{-3.,x});
        CHECK(result.classification==C::unresolved);
        CHECK(result.request_status(false)==S::valid && result.request_status(true)==S::numerical_failure);
    }
    CHECK(splice::classify(skew,splice::LogVariancePoint{-1.0,0.0}).classification == C::left);
    CHECK(splice::classify(skew,splice::LogVariancePoint{-1.0,100.0}).classification == C::right);
    CHECK(splice::classify(skew,splice::LogVariancePoint{-1.0,-100.0}).classification == C::left);
    for (double zero : {0.0,-0.0}) {
        const auto prepared = splice::prepare(4.0,zero);
        CHECK(prepared.valid_shape());
        for (double residual : {-1.0,0.0,1.0}) {
            const auto result = splice::classify(prepared,splice::VariancePoint{residual,1.0});
            CHECK(result.classification == C::symmetric);
            for (unsigned mask=1;mask<8;++mask) CHECK(result.request_status(mask&4) == S::valid);
        }
    }
    CHECK(splice::classify(splice::prepare(2.0,0.5),splice::VariancePoint{0.0,1.0}).classification == C::invalid_input);
    CHECK(splice::classify(splice::prepare(4.0,1.0),splice::VariancePoint{0.0,1.0}).classification == C::invalid_input);
    CHECK(splice::classify(skew,splice::VariancePoint{0.0,0.0}).classification == C::invalid_input);
    CHECK(splice::classify(skew,splice::VariancePoint{0.0,-1.0}).classification == C::invalid_input);
    CHECK(splice::classify(skew,splice::LogVariancePoint{NAN,0.0}).classification == C::invalid_input);
    CHECK(splice::classify(skew,splice::LogVariancePoint{0.0,INFINITY}).classification == C::invalid_input);
}

void density_semantics() {
    namespace gen = vlk::experiment::generated;
    const double symmetric_shape[] = {4.0,0.0};
    const auto skew = gen::SkewTDensity::prepare<true,true,true,true>(symmetric_shape,1);
    const auto student = gen::StudentTDensity::prepare<true,true,true,true>(symmetric_shape,1);
    const auto full = gen::SkewTDensity::observe<true,true,true,true>(skew,1.0,0.0,0.0);
    const auto reduced = gen::StudentTDensity::observe<true,true,true,true>(student,1.0,0.0,0.0);
    CHECK(full.valid);
    CHECK(std::abs(full.shape_hessian[2]+skew.observation_constant.hessian[2]-4.0)<1e-13);
    CHECK(std::abs(full.variance_hessian-reduced.variance_hessian)<1e-13);
    CHECK(std::abs(full.residual_hessian-reduced.residual_hessian)<1e-13);
    CHECK(std::abs(full.shape_hessian[0]+skew.observation_constant.hessian[0]
        -reduced.shape_hessian[0])<1e-13);
    const double shape[] = {4.0,0.5};
    const auto value_state = gen::SkewTDensity::prepare<true,false,false,true>(shape,1);
    const auto gradient_state = gen::SkewTDensity::prepare<false,true,false,true>(shape,1);
    const auto value = gen::SkewTDensity::observe<true,false,false,true>(value_state,2.5,-1.0,1.0);
    const auto gradient = gen::SkewTDensity::observe<false,true,false,true>(gradient_state,2.5,-1.0,1.0);
    CHECK(value.valid && gradient.valid);
    CHECK(std::abs(value.value+value_state.observation_constant.value-std::log(8.0/3.0))<1e-13);
    CHECK(std::abs(gradient.variance_gradient-0.2)<1e-13);
    CHECK(std::abs(gradient.shape_gradient[1]+gradient_state.observation_constant.gradient[1]+0.4)<1e-13);
}

void emit(double nu, double skew, double residual, double state, bool logarithmic,
          splice::Preparation request) {
    const auto prepared = splice::prepare(nu,skew,request);
    const auto result = logarithmic
        ? splice::classify(prepared,splice::LogVariancePoint{residual,state})
        : splice::classify(prepared,splice::VariancePoint{residual,state});
    const auto bound = prepared.coefficient_enclosure();
    std::printf("%.17g %.17g %.17g %.17g %d %u %u %d %.17g %.17g %d\n",
        nu,skew,residual,state,logarithmic,unsigned(result.classification),
        unsigned(result.reason),result.used_exact_fallback,bound.lower,bound.upper,
        prepared.tight_enclosure_attempted());
}
}

int main(int argc, char** argv) {
    if ((argc == 7 || argc == 8) && std::strcmp(argv[1],"--point") == 0) {
        emit(std::strtod(argv[2],nullptr),std::strtod(argv[3],nullptr),
             std::strtod(argv[4],nullptr),std::strtod(argv[5],nullptr),std::atoi(argv[6]) != 0,
             argc==8 && std::strcmp(argv[7],"FG")==0
                 ? splice::Preparation::value_or_gradient : splice::Preparation::hessian);
        return 0;
    }
    exact_points();
    ordinary_and_unresolved();
    preparation_lifetime();
    coordinates_and_symmetric();
    density_semantics();
    std::printf("Skew splice: %zu checks passed, bounded classifier, exact fallback, no heap\n",checks);
}
