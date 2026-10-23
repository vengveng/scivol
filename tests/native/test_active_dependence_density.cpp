// Standalone header-only test: do not link dependence_backend.cpp into this
// executable, because this TU deliberately instruments the inline policy owner.
#include "models/dependence/layout.hpp"
#include "math/student_normalization.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace probe {
std::size_t first = 0, second = 0, ratios = 0, tail_first = 0, tail_second = 0;
void reset() { first=second=ratios=tail_first=tail_second=0; }
double digamma(double x) { ++first; return digamma_approx(x); }
double trigamma(double x) { ++second; return trigamma_approx(x); }
auto ratio(double x, bool g, bool h) { ++ratios; first+=g; second+=h; return vlk_student_ratio_polynomial(x,g,h); }
double first_shift(double nu,double square,double shift) { ++tail_first; return vlk_student_tail_shape_first_shift(nu,square,shift); }
double second_shift(double nu,double square,double shift) { ++tail_second; return vlk_student_tail_shape_second_shift(nu,square,shift); }
double first_tail(double nu,double square) { ++tail_first; return vlk_student_tail_shape_first(nu,square); }
double second_tail(double nu,double square) { ++tail_second; return vlk_student_tail_shape_second(nu,square); }
}
#define digamma_approx probe::digamma
#define trigamma_approx probe::trigamma
#define vlk_student_ratio_polynomial probe::ratio
#define vlk_student_tail_shape_first_shift probe::first_shift
#define vlk_student_tail_shape_second_shift probe::second_shift
#define vlk_student_tail_shape_first probe::first_tail
#define vlk_student_tail_shape_second probe::second_tail
#include "models/dependence/density_policy.hpp"
#undef digamma_approx
#undef trigamma_approx
#undef vlk_student_ratio_polynomial
#undef vlk_student_tail_shape_first_shift
#undef vlk_student_tail_shape_second_shift
#undef vlk_student_tail_shape_first
#undef vlk_student_tail_shape_second

namespace {
namespace math=scivolapi::detail::dependence;
void check(bool value,int line) {
    if (!value) { std::fprintf(stderr,"active DCC density line %d\n",line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
template<unsigned Mask>
void check_mask(double nu) {
    constexpr bool F=Mask&1,G=Mask&2,H=Mask&4;
    const std::array<double,3> residual{.2,-.4,.7};
    probe::reset();
    const math::StudentPoint<F,G,H,false> fixed(nu,3);
    const auto local=math::student_observation<F,G,H,false,true>(fixed,.83,residual);
    CHECK(fixed.nu==nu && fixed.scale==nu-2);
    CHECK(fixed.constant.first==0 && fixed.constant.second==0);
    CHECK(probe::first==0 && probe::second==0 && probe::tail_first==0 && probe::tail_second==0);
    CHECK(probe::ratios==std::size_t(F && nu>=VLK_STUDENT_ASYMPTOTIC_NU));
    if constexpr (G||H) CHECK(std::isfinite(local.delta) && local.delta>0);
    if constexpr (H) CHECK(local.delta_inverse>0 && local.delta_curvature_weight<0);
    CHECK(local.shape==0 && local.shape_shape==0 && local.delta_shape==0);
    probe::reset();
    const math::StudentPoint<F,G,H,true> active(nu,3);
    const auto all=math::student_observation<F,G,H,true,true>(active,.83,residual);
    if constexpr (F) CHECK(local.value==all.value && fixed.constant.value==active.constant.value);
    if constexpr (G||H) CHECK(local.delta==all.delta);
    if constexpr (H) CHECK(local.delta_inverse==all.delta_inverse && local.delta_curvature_weight==all.delta_curvature_weight);
    CHECK(bool(probe::first)==G && bool(probe::second)==H);
    CHECK(probe::tail_first==(G ? 4u : 0u) && probe::tail_second==(H ? 4u : 0u));
    probe::reset();
    const auto shape_only=math::student_observation<F,G,H,true,false>(active,.83,residual);
    CHECK(shape_only.delta==0 && shape_only.delta_inverse==0 && shape_only.delta_curvature_weight==0 && shape_only.delta_shape==0);
    CHECK(shape_only.value==all.value && shape_only.shape==all.shape && shape_only.shape_shape==all.shape_shape);
}
}
int main() {
    for (double nu : {4.,128.,1e15}) {
        check_mask<1>(nu); check_mask<2>(nu); check_mask<3>(nu); check_mask<4>(nu);
        check_mask<5>(nu); check_mask<6>(nu); check_mask<7>(nu);
    }
    std::puts("active DCC density demand regressions passed");
}
