#include "models/garch_chart_admission.hpp"
#include <array>
#include <bit>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace domain=vlk::experiment::domain;
namespace exact=vlk::experiment::positive_simplex;
namespace {
std::size_t checks=0;
void check(bool good,int line) {
    ++checks;
    if(!good) { std::fprintf(stderr,"GJR exact admission line %d\n",line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
constexpr double tiny=std::numeric_limits<double>::denorm_min();
constexpr domain::MomentWitness half{0.5,true,true};
static_assert(std::bit_cast<std::uint64_t>(tiny)==1);
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits==53);
static_assert(exact::AffineLayout<>::bytes<=272);
static_assert(exact::UnitAffineLayout::words==51 && exact::UnitAffineLayout::bytes==408);
using P3=std::integral_constant<std::size_t,3>;
using Q2=std::integral_constant<std::size_t,2>;
using Eight=decltype(domain::gjr_coefficient_count(P3{},Q2{}));
static_assert(Eight::value==8 && exact::CountExtent<Eight>::value==8);
static_assert(std::is_empty_v<exact::CountExtent<Eight>>);
static_assert(std::is_same_v<decltype(domain::gjr_coefficient_count(
    std::size_t{3},std::size_t{2})),std::size_t>);

domain::Admission one(double a,double g,double b,const domain::MomentWitness* m=&half,
    const domain::MomentEnclosure* enclosure=nullptr) {
    return domain::gjr_coefficients(std::size_t{1},std::size_t{1},
        [=](std::size_t) {return a;},[=](std::size_t) {return g;},
        [=](std::size_t) {return b;},m,enclosure);
}
void exact_cases() {
    using A=domain::Admission;
    // Exact persistence is 1-2^-1075, although no positive double encodes its slack.
    CHECK(one(1.0,-tiny,0.0)==A::inside);
    CHECK(one(1.0,0.0,0.0)==A::mathematical_domain);
    CHECK(one(1.0,tiny,0.0)==A::mathematical_domain);
    CHECK(one(1.0,-tiny,tiny)==A::mathematical_domain);
    CHECK(one(tiny,-tiny,0.0)==A::inside); // alpha+gamma=0 is allowed.
    CHECK(one(0.0,-tiny,0.0)==A::mathematical_domain);
    const double maximum=std::numeric_limits<double>::max();
    CHECK(one(maximum,maximum,0.0)==A::mathematical_domain);
    CHECK(one(maximum,-maximum,0.0)==A::mathematical_domain);
    CHECK(one(-0.0,-0.0,-0.0)==A::inside);
    CHECK(one(0.0,0.0,-tiny)==A::mathematical_domain);

    // .5 + (.5-2^-54) + (2^-54-2^-107) = 1-2^-107.
    std::array<double,3> alpha{0.5,0x1.fffffffffffffp-2,0x1.fffffffffffffp-55};
    const auto assess=[&](auto p,auto q) {
        return domain::gjr_coefficients(p,q,[&](std::size_t i) {return alpha[i];},
            [](std::size_t) {return 0.0;},[](std::size_t) {return 0.0;},&half);
    };
    CHECK(assess(std::size_t{3},std::size_t{2})==A::inside);
    CHECK(assess(std::integral_constant<std::size_t,3>{},
        std::integral_constant<std::size_t,2>{})==A::inside);
    alpha[2]=0x1p-54;
    CHECK(assess(std::size_t{3},std::size_t{2})==A::mathematical_domain);

    CHECK(one(0.125,0.125,0.25,nullptr)==A::inside);
    CHECK(one(1.0,tiny,0.0,nullptr)==A::mathematical_domain);
    CHECK(one(1.0,-tiny,0.0,nullptr)==A::uncertified_moment);
    const domain::MomentWitness rounded_half{0.5,true,false};
    CHECK(one(1.0,-tiny,0.0,&rounded_half)==A::uncertified_moment);
    // U is respectively 1 and 9/8, but exact half-weighted persistence is <1.
    for(double b : {0.25,0.375}) {
        CHECK(one(0.25,0.5,b)==A::inside);
        CHECK(one(0.25,0.5,b,nullptr)==A::uncertified_moment);
        CHECK(one(0.25,0.5,b,&rounded_half)==A::uncertified_moment);
    }
    const domain::MomentWitness malformed{0.25,true,true};
    CHECK(one(1.0,-tiny,0.0,&malformed)==A::numerical_range);
    CHECK(one(0.125,0.125,0.25,&malformed)==A::numerical_range);
    const domain::MomentWitness invalid{0.5,false,false};
    CHECK(one(1.0,-tiny,0.0,&invalid)==A::numerical_range);
}
void enclosure_cases() {
    using A=domain::Admission;
    const domain::MomentEnclosure inside{0.25,0.375},outside{0.5,0.75},overlap{0.25,0.75};
    CHECK(one(0.25,0.5,0.5,nullptr,&inside)==A::inside);
    CHECK(one(0.25,0.5,0.5,nullptr,&outside)==A::mathematical_domain);
    CHECK(one(0.25,0.5,0.5,nullptr,&overlap)==A::uncertified_moment);
    const domain::MomentEnclosure negative_inside{0.625,0.75},negative_outside{0.25,0.5},
        negative_overlap{0.5,0.75};
    CHECK(one(0.75,-0.5,0.5,nullptr,&negative_inside)==A::inside);
    CHECK(one(0.75,-0.5,0.5,nullptr,&negative_outside)==A::mathematical_domain);
    CHECK(one(0.75,-0.5,0.5,nullptr,&negative_overlap)==A::uncertified_moment);
    // Synthetic bounds test arithmetic only; no shipped density producer supplies them.
    const domain::MomentEnclosure smallest{tiny,tiny},half_only{0.5,0.5};
    CHECK(one(1.0,-tiny,0.0,nullptr,&smallest)==A::inside); // Slack 2^-2148.
    CHECK(one(1.0,-tiny,tiny,nullptr,&smallest)==A::mathematical_domain);
    CHECK(one(1.0,-tiny,0.0,nullptr,&half_only)==A::inside); // Slack 2^-1075.
    CHECK(one(0.25,0.5,0.5,nullptr,&half_only)==A::mathematical_domain);
    const domain::MomentEnclosure universal{-0.0,1.0};
    CHECK(one(0.25,0.5,0.5,nullptr,&universal)==A::uncertified_moment);
    for(double value : {0.25,0.5,0.75}) {
        const domain::MomentWitness numeric{value,true,false};
        CHECK(one(0.25,0.5,0.5,&numeric)==A::uncertified_moment);
    }
    const domain::MomentEnclosure contradicts_half{0.25,0.375};
    CHECK(one(0.125,0.125,0.25,&half,&contradicts_half)==A::numerical_range);
    for(const auto bounds : {domain::MomentEnclosure{-tiny,0.5},
            domain::MomentEnclosure{0.5,0x1.0000000000001p+0},
            domain::MomentEnclosure{0.75,0.25},
            domain::MomentEnclosure{0.0,std::numeric_limits<double>::infinity()},
            domain::MomentEnclosure{std::numeric_limits<double>::quiet_NaN(),1.0}}) {
        CHECK(!domain::valid_moment_enclosure(bounds));
        CHECK(one(0.125,0.125,0.25,nullptr,&bounds)==A::numerical_range);
        CHECK(one(0.25,0.5,0.5,nullptr,&bounds)==A::numerical_range);
        std::array<std::uint64_t,exact::UnitAffineLayout::words> workspace;
        workspace.fill(99);
        unsigned visits=0;
        const auto getter=[&](std::size_t) noexcept {++visits; return 0.0;};
        CHECK(domain::gjr_enclosure_admission(P3{},Q2{},getter,getter,getter,bounds,workspace)
            ==A::numerical_range);
        CHECK(visits==0);
        for(auto word : workspace) CHECK(word==99);
    }
}
template<std::size_t P,std::size_t Q>
void shared_moment_cancellation() {
    using A=domain::Admission;
    static_assert(P>=2);
    std::array<double,P> alpha{},gamma{};
    std::array<double,Q> beta{};
    alpha[0]=0.25; alpha[1]=0.5;
    gamma[0]=0.5; gamma[1]=-0.5;
    const auto assess=[&](auto p,auto q,const domain::MomentWitness* witness=nullptr) {
        return domain::gjr_coefficients(p,q,[&](std::size_t i) {return alpha[i];},
            [&](std::size_t i) {return gamma[i];},[&](std::size_t i) {return beta[i];},witness);
    };
    // U=5/4, but the shared gamma sum is exactly zero and persistence is 3/4.
    CHECK(assess(std::size_t{P},std::size_t{Q})==A::inside);
    CHECK(assess(std::integral_constant<std::size_t,P>{},std::integral_constant<std::size_t,Q>{})==A::inside);
    const domain::MomentWitness unproven{0.5,true,false};
    CHECK(assess(std::size_t{P},std::size_t{Q},&unproven)==A::inside);
    alpha[0]=0.5;
    CHECK(assess(std::size_t{P},std::size_t{Q})==A::mathematical_domain);
    CHECK(assess(std::integral_constant<std::size_t,P>{},std::integral_constant<std::size_t,Q>{})==A::mathematical_domain);
    alpha[0]=0.25; alpha[1]=0.25; // alpha[1]+gamma[1]<0 must still reject.
    CHECK(assess(std::size_t{P},std::size_t{Q})==A::mathematical_domain);
}
template<class P,class Q>
void endpoint_getter_counts(P p,Q q) {
    const std::array alpha{0.125,0.25,0.125};
    const std::array gamma{0.25,-0.125,0.0};
    const std::array beta{0.125,0.125};
    std::array<unsigned,8> visits{};
    std::array<std::uint64_t,exact::UnitAffineLayout::words+1> workspace;
    workspace.fill(99);
    const auto a=[&](std::size_t i) noexcept {++visits[i]; return alpha[i];};
    const auto g=[&](std::size_t i) noexcept {++visits[3+i]; return gamma[i];};
    const auto b=[&](std::size_t i) noexcept {++visits[6+i]; return beta[i];};
    const auto result=domain::gjr_slack_sign_at_moment(p,q,a,g,b,0x1.fffffffffffffp-1,workspace);
    CHECK(result.state==exact::AffineState::resolved && result.sign==exact::ExactSign::positive);
    for(auto n : visits) CHECK(n==1);
    CHECK(workspace.back()==99);
    visits.fill(0); workspace.fill(99);
    const auto short_result=domain::gjr_slack_sign_at_moment(p,q,a,g,b,0.5,
        std::span(workspace).first(exact::UnitAffineLayout::words-1));
    CHECK(short_result.state==exact::AffineState::insufficient_capacity
        && short_result.sign==exact::ExactSign::unavailable);
    for(auto n : visits) CHECK(n==0);
    for(auto word : workspace) CHECK(word==99);
}
void endpoint_validation() {
    endpoint_getter_counts(P3{},Q2{});
    endpoint_getter_counts(std::size_t{3},std::size_t{2});
    std::array<std::uint64_t,exact::UnitAffineLayout::words> workspace;
    workspace.fill(99);
    unsigned visits=0;
    const auto getter=[&](std::size_t) noexcept {++visits; return 0.0;};
    const auto oversized=domain::gjr_slack_sign_at_moment(std::numeric_limits<std::size_t>::max(),
        std::size_t{2},getter,getter,getter,0.5,workspace);
    CHECK(oversized.state==exact::AffineState::numerical_range && oversized.sign==exact::ExactSign::unavailable);
    CHECK(visits==0);
    for(auto word : workspace) CHECK(word==99);
    const auto nonfinite=domain::gjr_slack_sign_at_moment(std::size_t{1},std::size_t{0},
        getter,[](std::size_t) noexcept {return std::numeric_limits<double>::quiet_NaN();},
        getter,0.0,workspace);
    CHECK(nonfinite.state==exact::AffineState::invalid_terms && nonfinite.sign==exact::ExactSign::unavailable);
    const auto zero_orders=domain::gjr_coefficients(std::integral_constant<std::size_t,0>{},
        std::integral_constant<std::size_t,0>{},getter,getter,getter);
    CHECK(zero_orders==domain::Admission::inside);
}
void complete_validation() {
    using A=domain::Admission;
    for(double a : {0.125,2.0}) for(bool invalid_first : {false,true}) {
        std::array<unsigned,6> seen{};
        const auto result=domain::gjr_coefficients(std::size_t{2},std::size_t{2},
            [&](std::size_t i) {++seen[i]; return invalid_first && i==0 ? -1.0 : a;},
            [&](std::size_t i) {++seen[2+i]; return 0.0;},
            [&](std::size_t i) {++seen[4+i]; return i==1 ? std::numeric_limits<double>::quiet_NaN() : 0.0;},&half);
        CHECK(result==A::mathematical_domain);
        for(auto visits : seen) CHECK(visits==1);
    }
    unsigned visits=0;
    const auto operand=[&](std::size_t) {++visits; return 0.0;};
    CHECK(domain::gjr_coefficients(std::numeric_limits<std::size_t>::max(),
        std::size_t{1},operand,operand,operand,&half)==A::numerical_range);
    CHECK(visits==0); // Invalid layout rejected before input traversal.
    for(const auto* witness : {&half,static_cast<const domain::MomentWitness*>(nullptr)}) {
        std::array<unsigned,3> seen{};
        const auto result=domain::gjr_coefficients(std::integral_constant<std::size_t,1>{},
            std::integral_constant<std::size_t,1>{},
            [&](std::size_t) {++seen[0]; return 0.125;},
            [&](std::size_t) {++seen[1]; return -0.0625;},
            [&](std::size_t) {++seen[2]; return 0.25;},witness);
        CHECK(result==A::inside);
        // Coefficient validation, upper validation, lattice pass; no affine pass.
        for(auto n : seen) CHECK(n==3);
    }
    CHECK(domain::admission_status(A::mathematical_domain)==VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    CHECK(domain::admission_status(A::numerical_range)==VLK_JET_DOMAIN_ERROR);
    CHECK(domain::admission_status(A::uncertified_moment)==VLK_JET_UNCERTIFIED_DOMAIN);
}
}
int main() {
    const int original=std::fegetround();
    for(int mode : {FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        CHECK(std::fesetround(mode)==0);
        exact_cases();
        enclosure_cases();
        shared_moment_cancellation<2,0>();
        shared_moment_cancellation<3,2>();
        shared_moment_cancellation<4,3>();
        endpoint_validation();
        complete_validation();
    }
    CHECK(std::fesetround(original)==0);
    std::printf("GJR exact admission: %zu checks\n",checks);
}
