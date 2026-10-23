// TEST_ONLY: linear mean composition, checked arenas and generated order parity.
#include "arma_egarch_runtime_jet.hpp"
#include "aparch_runtime_jet.hpp"
#include "diagonal_optimizer_scaling.hpp"
#include "gjr_transform_policy.hpp"
#include "constant_mean_jet.hpp"
#include "power_solver_constraints.hpp"
#include "linear_power_program.hpp"
#include "egarch_retained_program.hpp"
#include <array>
#include <cassert>
#include <vector>
#include <tuple>
#include <type_traits>
#include <cstdlib>
#include <cstdio>

static std::size_t allocations=0;
void* operator new(std::size_t bytes) {
    ++allocations;
    if (void* memory=std::malloc(bytes?bytes:1)) return memory;
    std::abort();
}
void* operator new[](std::size_t bytes) {return ::operator new(bytes);}
void operator delete(void* memory) noexcept {std::free(memory);}
void operator delete[](void* memory) noexcept {std::free(memory);}
void operator delete(void* memory,std::size_t) noexcept {std::free(memory);}
void operator delete[](void* memory,std::size_t) noexcept {std::free(memory);}

using namespace vlk::experiment;

constexpr auto aparch_layout_fields(const runtime::aparch::Layout& layout)
{
    const auto& [p_ar, q_ma, p, q, maximum_lag, mean_parameters,
        model_parameters, parameters, packed, ar_start, ma_start, omega_index,
        alpha_start, gamma_start, beta_start, delta_index, joint, valid] = layout;
    return std::array<std::size_t, 18>{p_ar, q_ma, p, q, maximum_lag,
        mean_parameters, model_parameters, parameters, packed, ar_start,
        ma_start, omega_index, alpha_start, gamma_start, beta_start,
        delta_index, std::size_t(joint), std::size_t(valid)};
}

template<class Density, class Orders, bool Joint>
void check_aparch_fixed_layout()
{
    namespace ap = runtime::aparch;
    constexpr std::size_t columns = Orders::linear ? Orders::columns
        : (Joint ? 1u + Orders::p_ar + Orders::q_ma : 0u);
    constexpr auto expected = [] {
        ap::Layout result;
        if (!ap::make_mean_layout<Density>(columns, Orders::p_ar, Orders::q_ma,
                Orders::p, Orders::q, Joint, result))
            return ap::Layout{};
        return result;
    }();
    static_assert(expected.valid);
    static_assert(expected.parameters == ap::static_width<Orders, Density, Joint>);
    static_assert(expected.packed == expected.parameters * (expected.parameters + 1u) / 2u);
    constexpr auto* static_layout = &ap::fixed_layout<Orders, Density, Joint>;
    static_assert(aparch_layout_fields(*static_layout) == aparch_layout_fields(expected));
    constexpr auto policy = runtime::aparch_transform::policy_layout<static_layout>(expected);
    using Lags = std::integral_constant<std::size_t, Orders::p>;
    static_assert(std::is_same_v<std::remove_cv_t<decltype(policy.p)>, Lags>);
    static_assert(std::is_same_v<decltype(runtime::aparch_transform::layout_field<
        static_layout, &ap::Layout::p>(expected)), Lags>);
    static_assert(policy.q == expected.q && policy.model_parameters == expected.model_parameters
        && policy.delta_index == expected.delta_index && policy.alpha_start == expected.alpha_start
        && policy.gamma_start == expected.gamma_start && policy.beta_start == expected.beta_start);
    static_assert(std::is_same_v<decltype(runtime::aparch_transform::policy_layout<nullptr>(expected)),
        const ap::Layout&>);

    const auto before = allocations;
    volatile std::size_t runtime_p = Orders::p;
    ap::Layout runtime;
    assert(ap::make_mean_layout<Density>(columns, Orders::p_ar, Orders::q_ma,
        runtime_p, Orders::q, Joint, runtime));
    assert(aparch_layout_fields(expected) == aparch_layout_fields(runtime));
    assert(ap::matches_orders<Orders>(runtime));
    const auto& fixed = ap::evaluation_layout<Orders, Density, Joint>(runtime);
    assert(aparch_layout_fields(fixed) == aparch_layout_fields(runtime));
    assert(&fixed != &runtime);
    assert((&ap::evaluation_layout<ap::RuntimeOrders, Density, Joint>(runtime) == &runtime));

    // A fixed layout owns its offsets; the checked boundary owns order matching.
    runtime.omega_index = std::numeric_limits<std::size_t>::max();
    assert((aparch_layout_fields(ap::evaluation_layout<Orders, Density, Joint>(runtime))
        == aparch_layout_fields(expected)));
    ++runtime.p;
    assert(!ap::matches_orders<Orders>(runtime));
    assert(allocations == before);
}

constexpr auto egarch_layout_fields(const runtime::arma_egarch::Layout& layout)
{
    const auto& [p_ar, q_ma, p_vol, q_vol, maximum_lag, mean_parameters,
        model_parameters, parameters, packed, ar_start, ma_start, omega_index,
        alpha_start, gamma_start, beta_start, valid] = layout;
    return std::array<std::size_t, 16>{p_ar, q_ma, p_vol, q_vol, maximum_lag,
        mean_parameters, model_parameters, parameters, packed, ar_start, ma_start,
        omega_index, alpha_start, gamma_start, beta_start, std::size_t(valid)};
}

template<class Density, class Orders>
void check_egarch_fixed_layout()
{
    namespace eg = runtime::arma_egarch;
    constexpr auto* fixed = &eg::fixed_layout<Orders, Density>;
    static_assert(fixed->valid);
    static_assert(fixed->parameters == eg::static_parameter_width<Orders, Density>);
    static_assert(fixed->model_parameters == eg::static_model_width<Orders, Density>);
    static_assert(fixed->packed == fixed->parameters * (fixed->parameters + 1u) / 2u);
    static_assert(std::is_same_v<decltype(runtime::arma_egarch_transform::layout_field<
        fixed, &eg::Layout::parameters>(*fixed)),
        std::integral_constant<std::size_t, fixed->parameters>>);
    const auto before = allocations;
    volatile std::size_t runtime_p = Orders::p_vol;
    eg::Layout runtime;
    const std::size_t columns = Orders::linear ? Orders::columns
        : 1u + Orders::p_ar + Orders::q_ma;
    assert(eg::make_mean_layout<Density>(columns, Orders::p_ar, Orders::q_ma,
        runtime_p, Orders::q_vol, runtime));
    assert(egarch_layout_fields(*fixed) == egarch_layout_fields(runtime));
    assert(eg::matches_orders<Orders>(runtime));
    assert((&eg::evaluation_layout<Orders, Density>(runtime) == fixed));
    assert((&eg::evaluation_layout<eg::RuntimeOrders, Density>(runtime) == &runtime));
    runtime.omega_index = std::numeric_limits<std::size_t>::max();
    assert((egarch_layout_fields(eg::evaluation_layout<Orders, Density>(runtime))
        == egarch_layout_fields(*fixed)));
    ++runtime.p_vol;
    assert(!eg::matches_orders<Orders>(runtime));
    assert(allocations == before);
}

template<class Density>
void check_power_layout_orders()
{
    namespace ap = runtime::aparch;
    check_aparch_fixed_layout<Density, ap::FixedVolatilityOrders<1, 1>, false>();
    check_aparch_fixed_layout<Density, ap::FixedVolatilityOrders<1, 2>, false>();
    check_aparch_fixed_layout<Density, ap::FixedVolatilityOrders<2, 1>, false>();
    check_aparch_fixed_layout<Density, ap::FixedVolatilityOrders<2, 2>, false>();
    check_aparch_fixed_layout<Density, ap::FixedVolatilityOrders<3, 5>, false>();
    check_aparch_fixed_layout<Density, ap::FixedJointOrders<1, 1, 1, 1>, true>();
    check_aparch_fixed_layout<Density, ap::FixedJointOrders<2, 0, 3, 4>, true>();
    check_aparch_fixed_layout<Density, ap::FixedJointOrders<0, 2, 2, 3>, true>();
    check_aparch_fixed_layout<Density, ap::FixedJointOrders<0, 0, 4, 1>, true>();
    check_aparch_fixed_layout<Density, mean::LinearVolatilityOrders<4, 1, 1>, true>();
    check_aparch_fixed_layout<Density, mean::LinearVolatilityOrders<7, 3, 2>, true>();
    check_aparch_fixed_layout<Density, mean::LinearVolatilityOrders<11, 4, 5>, true>();
    namespace eg = runtime::arma_egarch;
    check_egarch_fixed_layout<Density, eg::FixedOrders<1, 1, 1, 1>>();
    check_egarch_fixed_layout<Density, eg::FixedOrders<2, 0, 3, 4>>();
    check_egarch_fixed_layout<Density, eg::FixedOrders<0, 2, 2, 3>>();
    check_egarch_fixed_layout<Density, eg::FixedOrders<0, 0, 4, 1>>();
    check_egarch_fixed_layout<Density, mean::LinearVolatilityOrders<4, 1, 1>>();
    check_egarch_fixed_layout<Density, mean::LinearVolatilityOrders<7, 3, 2>>();
    check_egarch_fixed_layout<Density, mean::LinearVolatilityOrders<11, 4, 5>>();
}

template<class Orders=runtime::aparch::FixedVolatilityOrders<1,1>>
void check_persistence_precision_workspace()
{
    constexpr auto* layout = &runtime::aparch::fixed_layout<Orders,generated::NormalDensity,false>;
    constexpr auto policy = runtime::aparch_transform::policy_layout<layout>(*layout);
    const auto same_cache=[](const auto& a,const auto& b) {
        assert(a.point==b.point && a.positive==b.positive && a.negative==b.negative);
        assert(a.positive_error==b.positive_error && a.negative_error==b.negative_error);
        assert(a.tolerance==b.tolerance && a.density==b.density && a.state==b.state);
    };
    const double shapes[8][3]={{0.,0.,0.},{8.,0.,0.},{8.,.2,0.},{1.7,0.,0.},
        {.1,1.8,-.7},{.1,1.8,0.},{.1,1.8,0.},{.1,3.,0.}};
    for (unsigned density=0;density<8;++density) {
        generated::PowerMomentPrecisionCache cache,fixed_cache;
        std::array<double,Orders::p> alpha,gamma;
        std::array<double,Orders::q> beta;
        alpha.fill(.1/Orders::p);gamma.fill(.2);beta.fill(.7/Orders::q);
        const auto before=allocations;
        const auto compare=[&](double power) {
            const auto actual=generated::precise_power_persistence(policy,density,shapes[density],
                power,alpha.data(),gamma.data(),beta.data(),fixed_cache);
            const auto expected=generated::precise_power_persistence(density,shapes[density],
                power,alpha.data(),gamma.data(),Orders::p,beta.data(),Orders::q,cache);
            assert(actual.valid==expected.valid && actual.refined==expected.refined);
            assert(std::isfinite(actual.slack) && std::isfinite(actual.error));
            assert(actual.slack==expected.slack && actual.error==expected.error);
            same_cache(fixed_cache,cache);
            return actual;
        };
        const auto first=compare(1.7);
        assert(first.valid && first.refined && first.slack>0. && cache.state==2);
        const auto prepared=cache;
        const auto hot=compare(1.7);
        assert(hot.slack==first.slack && hot.error==first.error);
        same_cache(cache,prepared);
        const auto positive=cache.positive,negative=cache.negative;
        gamma.fill(-.3);alpha.fill(.15/Orders::p);
        const auto second=compare(1.7);
        assert(second.valid && second.refined && second.slack!=first.slack);
        assert(cache.positive==positive && cache.negative==negative);
        same_cache(cache,prepared);
        // Changing the power invalidates the shared half-moment preparation.
        const auto changed=compare(1.9);
        assert(changed.valid && cache.point[0]==1.9 && cache.positive!=positive);
        const auto changed_cache=cache;
        alpha.fill(0.);gamma.fill(0.);beta.fill(0.);
        for (double boundary:{std::nextafter(1.,0.),1.,std::nextafter(1.,2.)}) {
            beta[0]=boundary;
            const auto result=compare(2.);
            assert(result.refined && result.valid==(boundary!=1.) && result.slack==1.-boundary);
            same_cache(cache,changed_cache);
        }
        // Zero alpha and the unit second-moment shortcut need no cold preparation.
        cache={};fixed_cache={};
        beta[0]=std::nextafter(1.,0.);
        assert(compare(2.).valid && cache.state==0);
        alpha[0]=.125;
        for (double boundary:{std::nextafter(.875,0.),.875,std::nextafter(.875,1.)}) {
            beta[0]=boundary;
            const auto result=compare(2.);
            assert(result.refined && result.valid==(boundary!=.875) && result.slack==.875-boundary);
            assert(cache.state==0);
        }
        assert(allocations==before);
    }
}

void emit_precision_moment_identities()
{
    const unsigned codes[]={0,1,1,1,1,3,2,4,4,4,5,5,6,7,7,7};
    const double shapes[][3]={{0.,0.,0.},{8.,0.,0.},{128.,0.,0.},{1e15,0.,0.},{1e100,0.,0.},{1.4,0.,0.},
        {8.,0.,0.},{0.,.1,-6.},{0.,2.,.49},{0.,20.,6.},
        {0.,.1,0.},{0.,20.,0.},{0.,.1,0.},{0.,1.50001,0.},{0.,20.,0.},{0.,50.,0.}};
    for (unsigned i=0;i<std::size(codes);++i) for (double power:{.125,.75,2.,3.25}) {
        generated::PowerMomentPrecisionCache cache;
        const double alpha=.1,gamma=.2,beta=.5;
        const auto before=allocations;
        const auto result=generated::precise_power_persistence(
            codes[i],shapes[i],power,&alpha,&gamma,1,&beta,1,cache);
        assert(result.valid && cache.state==2 && allocations==before);
        std::printf("{\"density\":%u,\"shape\":[%.17g,%.17g,%.17g],\"power\":%.17g,"
            "\"positive\":[%.17g,%.17g],\"negative\":[%.17g,%.17g]}\n",
            codes[i],shapes[i][0],shapes[i][1],shapes[i][2],power,
            cache.positive[0],cache.positive[1],cache.negative[0],cache.negative[1]);
    }
    // Unit variance checks both tails, including large Bessel arguments where
    // K itself underflows although the skewed log density remains representable.
    for (unsigned code:{2u,4u,5u,6u,7u}) for (double rho:{-.994,-.95,.95,.994}) {
        if (std::abs(rho)>.95 && code==2) continue;
        const double shape[]={code==2?8.:rho,code==2?rho:code==7?20.:2.,.3};
        generated::PowerMomentPrecisionCache cache;
        const double alpha=.1,gamma=.2,beta=.5;
        const auto before=allocations;
        const auto result=generated::precise_power_persistence(
            code,shape,2.,&alpha,&gamma,1,&beta,1,cache);
        assert(result.valid && cache.state==2 && allocations==before);
        std::printf("{\"density\":%u,\"shape\":[%.17g,%.17g,%.17g],\"power\":2,"
            "\"variance\":true,\"positive\":[%.17g,%.17g],\"negative\":[%.17g,%.17g]}\n",
            code,shape[0],shape[1],shape[2],cache.positive[0],cache.positive[1],
            cache.negative[0],cache.negative[1]);
    }
}

void emit_skew_tail_precision()
{
    const double points[][3]={{3.7,3.700001,0.},{2.0001,2.0002,0.},
        {3.7,3.700001,-.8},{2.0001,2.0002,.7},{2.7,2.701,1e-8}};
    for (const auto& point:points) {
        generated::PowerMomentPrecisionCache cache;
        const double alpha=1e-10,gamma=.2,beta=.5;
        const auto before=allocations;
        const auto result=generated::precise_power_persistence(
            2,point+1,point[0],&alpha,&gamma,1,&beta,1,cache);
        assert(result.valid && cache.state==2 && allocations==before);
        std::printf("{\"skew_tail_precision\":true,\"point\":[%.17g,%.17g,%.17g],"
            "\"positive\":[%.17g,%.17g],\"negative\":[%.17g,%.17g]}\n",
            point[0],point[1],point[2],cache.positive[0],cache.positive[1],
            cache.negative[0],cache.negative[1]);
    }
}

template<int Order>
void check_skew_standardization_consumers()
{
    using N=ad::Differential<Order,2>;
    for (double nu:{2.01,6.,100.,1e8}) for (double skew:{-.9,0.,.8}) {
        const double shape[]={nu,skew};
        const auto density=generated::SkewTDensity::prepare<true,Order>=1,Order>=2>(shape,1);
        const N degrees=N::variable(nu,0),lambda=N::variable(skew,1);
        const generated::egarch_moment_detail::SkewTAbsoluteMoment moment(
            degrees,lambda,generated::student_normalization(degrees));
        assert(density.valid && density.a.value==moment.a.value && density.b.value==moment.b.value);
        const auto same=[](const auto& left,const auto& right) {
            for (std::size_t i=0;i<left.size();++i)
                assert(std::abs(left[i]-right[i])<=4*std::numeric_limits<double>::epsilon()
                    *std::max(1.,std::abs(right[i])));
        };
        if constexpr (Order>=1) {
            same(density.a.gradient,moment.a.gradient);same(density.b.gradient,moment.b.gradient);
        }
        if constexpr (Order>=2) {
            same(density.a.hessian,moment.a.hessian);same(density.b.hessian,moment.b.hessian);
        }
    }
}

void check_sublinear_power_moment_reduction()
{
    for (double gamma : {0.,.25}) for (double delta : {.01,.2,.7,.9999,1.,1.7}) {
        const double shape[2]={8.,0.};
        const auto student=generated::asymmetric_power_moment<2,generated::StudentTDensity>(shape,gamma,delta);
        const auto skew=generated::asymmetric_power_moment<2,generated::SkewTDensity>(shape,gamma,delta);
        assert(student.valid && skew.valid);
        const auto same=[](double a,double b) {return std::abs(a-b)<=2e-9+2e-7*std::abs(b);};
        assert(same(skew.value,student.value));
        for (std::size_t i=0;i<3;++i) {
            assert(same(skew.gradient[i],student.gradient[i]));
            for (std::size_t j=i;j<3;++j)
                assert(same(skew.hessian[model::packed_upper_index(4,i,j)],
                    student.hessian[model::packed_upper_index(3,i,j)]));
        }
    }
}

template<bool Aparch>
void check_program_units_and_strict_constraint_values()
{
    using Program=runtime::LinearPowerProgram<generated::NormalDensity,Aparch>;
    Program program;
    assert(program.prepare(7,4,3));
    const auto k=program.layout.parameters;
    std::vector<double> point(k,.1),base(k,1.),units(k);
    for (std::size_t i=0;i<7;++i) base[i]=.5+double(i);
    for (double delta : {.9,1.7}) {
        if constexpr (Aparch) point[program.layout.delta_index]=delta;
        assert(program.parameter_units(point.data(),base.data(),2.,units.data()));
        for (std::size_t i=0;i<k;++i) {
            const double expected=Aparch && i==7?std::pow(2.,delta/2):base[i];
            assert(units[i]==expected);
        }
    }
    assert(!program.parameter_units(point.data(),base.data(),0.,units.data()));
    base[0]=std::numeric_limits<double>::quiet_NaN();
    assert(!program.parameter_units(point.data(),base.data(),2.,units.data()));
    const runtime::PowerConstraints<Program> constraints{program,
        mean::PolynomialConstraints{k,{},nullptr,nullptr,0},false,nullptr,nullptr};
    std::vector<double> slack(constraints.count(),1.);
    assert(constraints.feasible_values(slack.data()));
    for (double invalid : {0.,-1.,std::numeric_limits<double>::quiet_NaN()}) {
        slack.back()=invalid;
        assert(!constraints.feasible_values(slack.data()));
    }
}

struct CountedMomentDensity : generated::GedDensity {
    static inline unsigned preparations=0;
    static inline std::array<unsigned,3> orders{};
    static inline bool reject=false;
    template<bool F, bool G, bool H>
    static auto prepare(const double* shape,std::size_t n) noexcept
    {
        ++preparations;
        ++orders[H?2:G?1:0];
        auto result=generated::GedDensity::prepare<F,G,H>(shape,n);
        result.valid=result.valid && !reject;
        return result;
    }
};

void check_point_owned_power_moments()
{
    using Memo=generated::PowerMomentValues<CountedMomentDensity>;
    static_assert(!std::is_same_v<generated::PowerMomentValues<generated::NormalInverseGaussianDensity>,
        generated::PowerMomentValues<generated::VarianceGammaDensity>>);
    std::array<Memo::Entry,2> entries;
    Memo memo{entries.data(),entries.size()};
    assert(memo.reset());
    double shape=1.6;
    const auto value=[&](std::size_t slot=0) {
        return generated::asymmetric_power_moment<0,CountedMomentDensity>(
            &shape,0.15,1.7,&memo,slot);
    };
    const auto first=value();
    assert(first.valid && CountedMomentDensity::preparations==1);
    const auto again=value();
    assert(again.value==first.value && CountedMomentDensity::preparations==1);
    assert(value(1).value==first.value && CountedMomentDensity::preparations==2);
    shape=std::nextafter(shape,2.0);
    assert(value().valid && CountedMomentDensity::preparations==3);
    assert(!memo.find(0,&shape,std::nextafter(0.15,1.0),1.7));
    assert(!memo.find(0,&shape,0.15,std::nextafter(1.7,2.0)));
    const auto gradient=generated::asymmetric_power_moment<1,CountedMomentDensity>(
        &shape,0.15,1.7,&memo);
    const auto hessian=generated::asymmetric_power_moment<2,CountedMomentDensity>(
        &shape,0.15,1.7,&memo);
    assert(gradient.valid && hessian.valid && CountedMomentDensity::preparations==5);
    assert(value().valid && CountedMomentDensity::preparations==5);
    assert(!memo.find(2,&shape,0.15,1.7));
    memo.store(0,&shape,0.0,1.7,first);
    assert(memo.find(0,&shape,0.0,1.7) && !memo.find(0,&shape,-0.0,1.7));

    // An invalid computed value is reusable only at the identical point/request.
    assert(memo.reset());
    CountedMomentDensity::reject=true;
    assert(!value().valid && CountedMomentDensity::preparations==6);
    assert(!value().valid && CountedMomentDensity::preparations==6);
    CountedMomentDensity::reject=false;
    assert(memo.reset());
    assert(value().valid && CountedMomentDensity::preparations==7);
    Memo other{entries.data()+1,1};
    assert(other.reset());
    assert(memo.find(0,&shape,0.15,1.7) && !other.find(0,&shape,0.15,1.7));
}

void check_domain_reuses_prepared_scalar_moments()
{
    using Density=CountedMomentDensity;
    runtime::aparch::Layout layout;
    assert(runtime::aparch::make_linear_layout<Density>(0,2,1,layout));
    const double theta[]={.2,.05,.05,.15,.15,.5,1.7,1.6};
    assert(layout.parameters==std::size(theta));
    generated::PowerMomentBatch<0,Density> batch(theta+7,1.7,2);
    const auto value=batch(.15,0);
    assert(value.valid);
    double prepared[2]={value.value,value.value};
    const auto ordinary=generated::power_persistence<Density>(layout,theta);
    const auto before=CountedMomentDensity::preparations;
    const auto borrowed=generated::power_persistence<Density>(layout,theta,nullptr,prepared);
    assert(ordinary.valid && borrowed.valid && ordinary.slack==borrowed.slack);
    assert(runtime::aparch::valid_theta<Density>(layout,theta,nullptr,prepared));
    assert(CountedMomentDensity::preparations==before);
    assert(!runtime::aparch::valid_theta<Density>(layout,theta,nullptr,{prepared,1}));
    for (double invalid:{0.,-1.,double(INFINITY),double(NAN)}) {
        prepared[0]=invalid;
        assert(!runtime::aparch::valid_theta<Density>(layout,theta,nullptr,prepared));
    }
    assert(CountedMomentDensity::preparations==before);
}

void check_power_batch_preparation()
{
    using Density=CountedMomentDensity;
    using Memo=generated::PowerMomentValues<Density>;
    static_assert(sizeof(generated::PowerMomentBatch<0,Density>)
        <sizeof(generated::PowerMomentBatch<1,Density>));
    static_assert(sizeof(generated::PowerMomentBatch<1,Density>)
        <sizeof(generated::PowerMomentBatch<2,Density>));
    const double shape=1.6;
    const auto check=[&]<int Order,class Lags>(Lags lags) {
        const auto before=Density::orders;
        const auto memory=allocations;
        auto batch=generated::make_power_moment_batch<Order,Density>(&shape,1.7,lags);
        assert(Density::orders==before);
        for (std::size_t lag=0;lag<lags;++lag) assert(batch(-.2+.2*lag,lag).valid);
        for (unsigned order=0;order<3;++order)
            assert(Density::orders[order]==before[order]+(order==Order));
        assert(allocations==memory);
    };
    const auto orders=[&]<class Lags>(Lags lags) {
        check.template operator()<0>(lags);check.template operator()<1>(lags);
        check.template operator()<2>(lags);
    };
    orders(std::size_t{3});orders(std::integral_constant<std::size_t,1>{});
    orders(std::integral_constant<std::size_t,3>{});
    std::array<Memo::Entry,2> entries;
    Memo memo{entries.data(),entries.size()};
    assert(memo.reset());
    assert((generated::asymmetric_power_moment<0,Density>(&shape,.15,1.7,&memo).valid));
    const auto before=Density::preparations;
    generated::PowerMomentBatch<0,Density> first(&shape,1.7,2,&memo);
    const auto expected=first(.15,0);
    assert(expected.valid && Density::preparations==before+1);
    assert(!memo.find(0,&shape,.15,1.7) && memo.find(0,&shape,.15,1.7,true));
    generated::PowerMomentBatch<0,Density> warm(&shape,1.7,2,&memo);
    assert(warm(.15,0).value==expected.value && Density::preparations==before+1);
    assert(warm(.25,1).valid && Density::preparations==before+2);
}

template<int Order,class Density,std::size_t P>
void check_fixed_power_batch(const double* shape,double power)
{
    using Memo=generated::PowerMomentValues<Density>;
    using Orders=runtime::aparch::FixedVolatilityOrders<P,1>;
    constexpr auto* layout=&runtime::aparch::fixed_layout<Orders,Density,false>;
    constexpr auto policy=runtime::aparch_transform::policy_layout<layout>(*layout);
    using Lags=std::integral_constant<std::size_t,P>;
    static_assert(std::is_same_v<decltype(generated::make_power_moment_batch<Order,Density>(
        shape,power,policy.p)),generated::PowerMomentBatch<Order,Density,Lags>>);
    static_assert(std::is_same_v<decltype(generated::make_power_moment_batch<Order,Density>(
        shape,power,std::size_t{P})),generated::PowerMomentBatch<Order,Density>>);
    const auto same_entries=[](const auto& a,const auto& b) {
        for (std::size_t i=0;i<a.size();++i) {
            assert(a[i].point==b[i].point && a[i].value==b[i].value);
            assert(a[i].state==b[i].state && a[i].factored==b[i].factored);
        }
    };
    // Derivatives bypass scalar memos; scalar requests also check absent/short arenas.
    for (unsigned mode=0;mode<(Order==0?4u:1u);++mode) {
        const std::size_t slots=mode==2?0u:mode==3?P-1u:P;
        std::array<typename Memo::Entry,P+2> fixed_entries{},runtime_entries{};
        fixed_entries.front().state=runtime_entries.front().state=77;
        fixed_entries[slots+1].state=runtime_entries[slots+1].state=77;
        Memo fixed_memo{fixed_entries.data()+1,slots},runtime_memo{runtime_entries.data()+1,slots};
        assert(fixed_memo.reset() && runtime_memo.reset());
        const auto empty=fixed_entries;
        const auto before=allocations;
        for (unsigned pass=0;pass<(Order==0?3u:1u);++pass) {
            auto fixed=generated::make_power_moment_batch<Order,Density>(
                shape,power,policy.p,mode==1?nullptr:&fixed_memo);
            generated::PowerMomentBatch<Order,Density> dynamic(
                shape,power,P,mode==1?nullptr:&runtime_memo);
            for (std::size_t lag=0;lag<P;++lag) {
                const double gamma=(pass==2?-1.:1.)*(.15+.1*lag);
                const auto a=fixed(gamma,lag),b=dynamic(gamma,lag);
                assert(a.valid && b.valid && std::isfinite(a.value) && a.value==b.value);
                assert(a.gradient==b.gradient && a.hessian==b.hessian);
                if constexpr (Order>=1) for (double value:a.gradient) assert(std::isfinite(value));
                if constexpr (Order>=2) for (double value:a.hessian) assert(std::isfinite(value));
                assert(fixed.value_error_scale(gamma,a.value)==dynamic.value_error_scale(gamma,b.value));
                same_entries(fixed_entries,runtime_entries);
                if constexpr (Order==0 && generated::quadrature_power_moment<Density>) {
                    if (mode!=1 && lag<slots) {
                        assert(fixed_memo.find(lag,shape,gamma,power,P>1));
                        assert(!fixed_memo.find(lag,shape,gamma,power,P<=1));
                    }
                } else same_entries(fixed_entries,empty);
            }
        }
        assert(fixed_entries.front().state==77 && fixed_entries[slots+1].state==77);
        assert(runtime_entries.front().state==77 && runtime_entries[slots+1].state==77);
        assert(allocations==before);
    }
}

template<class Density>
void check_fixed_power_batches(const double* shape)
{
    const auto orders=[&]<int Order>() {
        for (double power:{.7,1.7}) {
            check_fixed_power_batch<Order,Density,1>(shape,power);
            check_fixed_power_batch<Order,Density,3>(shape,power);
        }
    };
    orders.template operator()<0>();orders.template operator()<1>();orders.template operator()<2>();
}

template<class Density>
void emit_power_batch_jets(const double* shape)
{
    constexpr auto width=Density::shape_parameters+2u;
    for (double power:{.7,1.7}) {
        generated::PowerMomentBatch<2,Density> full(shape,power,3);
        generated::PowerMomentBatch<1,Density> first(shape,power,3);
        generated::PowerMomentBatch<0,Density> scalar(shape,power,3);
        for (unsigned lag=0;lag<3;++lag) {
            const double gamma=std::array{-.2,.15,.4}[lag];
            const auto actual=full(gamma,lag),gradient=first(gamma,lag),value=scalar(gamma,lag);
            const auto direct=generated::asymmetric_power_moment<2,Density>(shape,gamma,power);
            assert(actual.valid && gradient.valid && value.valid && direct.valid);
            const auto close=[](double a,double b) {assert(std::abs(a-b)<=2e-9+2e-7*std::abs(b));};
            close(actual.value,direct.value);close(value.value,actual.value);
            for (unsigned j=0;j<width;++j) {
                close(actual.gradient[j],direct.gradient[j]);
                close(gradient.gradient[j],actual.gradient[j]);
            }
            for (unsigned j=0;j<actual.hessian.size();++j) close(actual.hessian[j],direct.hessian[j]);
            std::printf("{\"moment_jet\":true,\"density\":%u,\"point\":[%.17g,%.17g",
                generated::power_density_code<Density>,gamma,power);
            for (unsigned j=0;j<Density::shape_parameters;++j) std::printf(",%.17g",shape[j]);
            std::printf("],\"value\":%.17g,\"gradient\":[",actual.value);
            for (unsigned j=0;j<width;++j) std::printf("%s%.17g",j?",":"",actual.gradient[j]);
            std::printf("],\"hessian\":[");
            for (unsigned j=0;j<actual.hessian.size();++j) std::printf("%s%.17g",j?",":"",actual.hessian[j]);
            std::printf("]}\n");
        }
    }
}

void check_shared_mixture_moments(bool candidate)
{
    // Independent conditional-Normal fixed log-G AD: original points use
    // 64/128-node refinement, added boundary points use 256/512 nodes.
    // The candidate uses a different positive integral and shape-centered map.
    // Columns are rho, lambda, F, G, H.
    const double cases[][8] {
        {-.08590362405556805,3.6453348340489273,.7715073203749511,
         -.009862472299235007,.007054338222722167,.10017976779626078,
         .002955020171626496,-.00376070656535632},
        {0.,3.,.7654655446197459,0.,.010491194329724007,
         .15309310892395378,0.,-.006761288765355089},
        {-.7,1.501,.7546176422202283,-.00787917285604931,.027292946985130767,
         -.05888573695463274,.004204489894970332,-.03410361037739107},
        {.7,3.,.7757009368037271,.004375483168194582,.007231703663010958,
         -.033391318717168245,-.0012986267229490851,-.004699971744323572},
        {.98,10.,.7912653149141184,4.1082307767782485e-6,.0006587341914761069,
         -.00041849423692405334,-4.065769198173175e-7,-.00013106779552810033},
        {-.98,50.,.7965558729887814,-8.284735297221118e-7,2.6550909920430854e-5,
         -8.446149154635367e-5,1.653517210298193e-8,-1.0611089297958565e-6},
        {0.,1.5000000000000018,.7351051938957636,0.,.03893192659994804,
         .3675525969477575,0.,-.04766498602410857},
        {1e-100,1.5000000000000018,.7351051938957636,3.6755259694775785e-101,
         .03893192659994804,.3675525969477575,-3.48086633647826e-101,-.04766498602410857},
        {1e-12,50.,.795892373871819,8.039316907807189e-15,3.979262936404382e-5,
         .008039316907807174,-1.6200849684252279e-16,-1.5896360848824398e-6},
        {.9949999999999999,50.,.7965558783748009,5.027911795010374e-8,
         2.6550802413982666e-5,-2.0210802256404332e-5,
         -1.003804919100475e-9,-1.0611046393285695e-6},
    };
    using Density=generated::VarianceGammaDensity;
    const auto evaluate=[&]<int Order>(const double* point) {
        return candidate ? generated::density_moment::variance_gamma_absolute<Order>(point)
            : generated::absolute_moment<Order,Density>(point);
    };
    for (const auto& point:cases) {
        const auto f=evaluate.template operator()<0>(point);
        const auto g=evaluate.template operator()<1>(point);
        const auto h=evaluate.template operator()<2>(point);
        assert(f.valid && g.valid && h.valid);
        for (const auto& jet:{f,g,h}) assert(std::abs(jet.value-point[2])<3e-10);
        for (unsigned j=0;j<2;++j) {
            assert(std::abs(g.gradient[j]-point[3+j])<3e-9);
            assert(std::abs(h.gradient[j]-point[3+j])<3e-9);
        }
        for (unsigned j=0;j<3;++j) assert(std::abs(h.hessian[j]-point[5+j])<3e-9);
        const double reflected[]{-point[0],point[1]};
        const auto positive=evaluate.template operator()<2>(reflected);
        assert(std::abs(h.value-positive.value)<3e-13);
        assert(std::abs(h.gradient[0]+positive.gradient[0])<3e-12);
        assert(std::abs(h.gradient[1]-positive.gradient[1])<3e-12);
        const auto left=generated::negative_second_moment<2,Density>(point);
        const auto right=generated::negative_second_moment<2,Density>(reflected);
        assert(left.valid && right.valid && std::abs(left.value+right.value-1.)<3e-9);
        assert(std::abs(left.gradient[0]-right.gradient[0])<3e-8);
        assert(std::abs(left.gradient[1]+right.gradient[1])<3e-8);
    }
    const double invalid[]{0.,1.};
    assert((!evaluate.template operator()<2>(invalid).valid));
    for (double lambda:{VG_LAMBDA_MIN,3.,VG_LAMBDA_MAX}) {
        const double symmetric[]{0.,lambda};
        const auto moment=evaluate.template operator()<2>(symmetric);
        const double value=std::sqrt(2.0/(M_PI*lambda))
            *std::exp(std::lgamma(lambda+.5)-std::lgamma(lambda));
        assert(moment.valid && std::abs(moment.value-value)<1e-13);
        assert(moment.gradient[0]==0 && moment.hessian[1]==0);
        assert(std::abs(moment.hessian[0]-value/(2*lambda-1))<3e-10);
    }
}


template<int Order, class Design>
void check_residual_cancellation()
{
    const double small=std::ldexp(1.0,-27);
    double x[4]={1.0-small,0.0,0.0,0.0};
    double theta[4]={1.0+small,0.0,0.0,0.0};
    double g[4]{},h[10]{};
    mean::State state{0.0,g,h};
    mean::advance<Order,0,4>(state,{},Design{x,4},theta,1.0,0,0,4);
    assert(state.value==std::ldexp(1.0,-54));
    if constexpr (Order>=1) assert(g[0]==-x[0]);
    if constexpr (Order>=2) for (double value:h) assert(value==0.0);
}

template<class Design, std::size_t Width>
void check_deferred_normal_cancellation()
{
    constexpr std::size_t n=3;
    const double small=std::ldexp(1.0,-27);
    double x[n*4]={}, y[n]={1.,1.,1.};
    for (std::size_t t=0;t<n;++t) x[t*4]=1.-small;
    double theta[5]={1.+small,0.,0.,0.,1e-32};
    const auto plan=mean::layout(4,0);
    std::vector<std::byte> memory(mean::workspace_bytes(plan,7));
    double f, full, g[5], h[25], residuals[n];
    assert((mean::evaluate<generated::NormalDensity,1,Design,0,Width>(
        plan,{x,4},theta,y,n,0,memory.data(),memory.size(),&f,nullptr,nullptr,residuals)));
    assert((mean::evaluate<generated::NormalDensity,7,Design,0,Width>(
        plan,{x,4},theta,y,n,0,memory.data(),memory.size(),&full,g,h)));
    const double residual=std::ldexp(1.0,-54);
    const double expected=.5*n*(std::log(2*M_PI)+std::log(theta[4])+residual*residual/theta[4]);
    assert(std::abs(f-expected)<1e-12 && std::abs(f-full)<1e-12);
    for (double e:residuals) assert(e==residual);
}

template<bool Aparch, class Density=generated::NormalDensity>
void check()
{
    using Shape=generated::DensityShapeTransformT<Density>;
    using Layout=std::conditional_t<Aparch,runtime::aparch::Layout,runtime::arma_egarch::Layout>;
    using Dynamic=mean::LinearVolatilityOrders<>;
    using Fixed=mean::LinearVolatilityOrders<4,3,2>;
    Layout layout{};
    if constexpr (Aparch) assert(runtime::aparch::make_linear_layout<Density>(4,3,2,layout));
    else assert(runtime::arma_egarch::make_linear_layout<Density>(4,3,2,layout));
    if constexpr (!Aparch) {
        auto with_ma=layout;
        with_ma.q_ma=3;
        assert(runtime::arma_egarch::core_workspace_bytes<0>(with_ma)
            -runtime::arma_egarch::core_workspace_bytes<0>(layout)==3*sizeof(double));
        assert(runtime::arma_egarch::core_workspace_bytes<1>(with_ma)
            -runtime::arma_egarch::core_workspace_bytes<1>(layout)
            ==3*(1+layout.parameters)*sizeof(double));
        assert(runtime::arma_egarch::core_workspace_bytes<2>(with_ma)
            -runtime::arma_egarch::core_workspace_bytes<2>(layout)
            ==3*(1+layout.parameters+layout.packed)*sizeof(double));
    }
    constexpr std::size_t n=9;
    double y[n]{.31,-.71,.28,1.02,-.62,.11,.95,-.43,.18};
    double x[n*4];
    for (std::size_t i=0;i<n;++i)
        for (std::size_t j=0;j<4;++j) x[i*4+j]=j ? .1*(i+1)/(j+1) : 1.;
    std::vector<double> theta(layout.parameters,.02);
    theta[4]=Aparch ? .2 : -.1;
    for (std::size_t j=0;j<2;++j) theta[11+j]=.25;
    if constexpr (Aparch) theta[13]=1.5;
    if constexpr (Density::shape_parameters) theta.back()=1.7;
    const auto bytes=[&](unsigned mask) {
        if constexpr (Aparch) return runtime::aparch::workspace_bytes<Density,Shape,Dynamic,true>(layout,false,mask);
        else return runtime::arma_egarch::workspace_bytes<Density,Shape,Dynamic>(layout,false,mask);
    };
    std::vector<std::byte> arena(bytes(7));
    std::vector<double> g(layout.parameters),h(layout.parameters*layout.parameters);
    double f,residual[n],variance[n];
    const auto evaluate=[&]<class Orders>(unsigned mask,std::size_t size) {
        unsigned valid=0;
        int status;
        if constexpr (Aparch) status=runtime::aparch::evaluate<true,Density,Shape,Orders,true>(
            layout,theta.data(),theta.size(),y,n,1.,false,mask,arena.data(),size,
            mask&1?&f:nullptr,mask&2?g.data():nullptr,mask&4?h.data():nullptr,&valid,x,residual,variance);
        else status=runtime::arma_egarch::evaluate<Density,Shape,Orders>(
            layout,theta.data(),theta.size(),y,n,1.,false,mask,arena.data(),size,
            mask&1?&f:nullptr,mask&2?g.data():nullptr,mask&4?h.data():nullptr,&valid,x,residual,variance);
        assert(valid==(status==0?mask:0u));
        return status;
    };
    assert(evaluate.template operator()<Dynamic>(7,arena.size())==0);
    assert(evaluate.template operator()<mean::LinearVolatilityOrders<0>>(7,arena.size())
        ==VLK_JET_INVALID_ARGUMENT);
    const double expected_f=f;
    const auto expected_g=g,expected_h=h;
    for (unsigned mask=1;mask<=7;++mask) {
        assert(evaluate.template operator()<Fixed>(mask,bytes(mask))==0);
        if (mask&1) assert(std::abs(f-expected_f)<1e-11);
        if (mask&2) for (std::size_t i=0;i<g.size();++i) assert(std::abs(g[i]-expected_g[i])<1e-10);
        if (mask&4) for (std::size_t i=0;i<h.size();++i) assert(std::abs(h[i]-expected_h[i])<1e-9);
        assert(evaluate.template operator()<Dynamic>(mask,bytes(mask)-1)==VLK_JET_INSUFFICIENT_WORKSPACE);
        assert(variance[0]==1.);
    }
    std::fill_n(theta.data(),4,0.0);
    y[0]=0.0;
    for (double alpha : {0.0,0.02}) {
        std::fill_n(theta.data()+5,3,alpha);
        for (unsigned mask=1;mask<=7;++mask) {
            const bool first = Aparch || alpha==0.0;
            const bool second = Aparch && alpha==0.0
                && std::is_same_v<Density,generated::NormalDensity>;
            const bool valid = (!(mask&2) || first) && (!(mask&4) || second);
            assert((evaluate.template operator()<Dynamic>(mask,bytes(mask))==0)==valid);
            assert((evaluate.template operator()<Fixed>(mask,bytes(mask))==0)==valid);
        }
    }
    std::fill_n(x,4,0.0);
    assert(evaluate.template operator()<Dynamic>(7,arena.size())==0);
    assert(evaluate.template operator()<Fixed>(7,arena.size())==0);
    x[5]=std::numeric_limits<double>::quiet_NaN();
    assert(evaluate.template operator()<Dynamic>(7,arena.size())==VLK_JET_INVALID_ARGUMENT);
    const auto maximum=std::numeric_limits<std::size_t>::max();
    if constexpr (Aparch) assert(!runtime::aparch::make_linear_layout<Density>(maximum,3,2,layout));
    else assert(!runtime::arma_egarch::make_linear_layout<Density>(maximum,3,2,layout));
}

struct SparseObjective {
    std::size_t dimension() const noexcept {return 1u;}
};
struct PlannedObjective : SparseObjective {
    bool speculate_trial_gradient(std::int64_t iteration) const noexcept
    {return iteration>=3;}
};
template<class T>
concept HasTrialPlan = requires (const T& value) {value.speculate_trial_gradient(1);};
static_assert(!HasTrialPlan<runtime::DiagonalObjective<SparseObjective>>);
static_assert(HasTrialPlan<runtime::DiagonalObjective<PlannedObjective>>);

struct TrialObjective {
    bool fused_valid=true, gradient_valid=true;
    mutable unsigned values=0,gradients=0;
    std::size_t dimension() const noexcept {return 1u;}
    bool speculate_trial_gradient(std::int64_t iteration) const noexcept
    {return iteration==1;}
    bool evaluate(kraft::ObjectiveRequest request,const double* x,
        double* f,double* g,bool) const noexcept
    {
        if (request!=kraft::ObjectiveRequest::gradient) {++values;*f=x[0]*x[0];}
        if (request!=kraft::ObjectiveRequest::value) {++gradients;*g=2*x[0];}
        if (request==kraft::ObjectiveRequest::value_gradient && !fused_valid) return false;
        return request==kraft::ObjectiveRequest::value || gradient_valid;
    }
};

void check_trial_plan()
{
    TrialObjective objective;
    double x=3,f=0,g=0;
    const auto run=[&](int iteration) {
        objective.values=objective.gradients=0;
        const auto result=runtime::runtime_kraft_detail::evaluate_trial(objective,&x,&f,&g,iteration);
        assert(result.value_requests==objective.values);
        assert(result.gradient_requests==objective.gradients);
        return result;
    };
    auto trial=run(1);
    assert(trial.acceptable && trial.gradient_current && f==9 && g==6);
    x=4;
    trial=run(2);
    assert(trial.acceptable && !trial.gradient_current && f==16);
    assert(trial.gradient_requests==0); // No stale derivative from the prior plan.
    objective.fused_valid=false;
    trial=run(1);
    assert(trial.acceptable && trial.gradient_current && g==8);
    assert(trial.value_requests==2 && trial.gradient_requests==2);
    objective.gradient_valid=false;
    trial=run(1);
    assert(!trial.acceptable && !trial.gradient_current && f==16);
    trial=run(2);
    assert(trial.acceptable && !trial.gradient_current);
}

template<int Order>
void check_sparse_power_shock()
{
    using namespace model::aparch_equation;
    constexpr std::size_t width=18, means=7, packed=width*(width+1)/2;
    constexpr std::size_t alpha=9, gamma=12, delta=16;
    std::array<double,width> theta{}, residual_g{}, dense_g{}, sparse_g{}, fixed_g{};
    std::array<double,packed> residual_h{}, dense_h{}, sparse_h{}, fixed_h{};
    theta[alpha]=0.08; theta[gamma]=-0.2; theta[delta]=1.7;
    for (std::size_t i=0;i<means;++i) {
        residual_g[i]=0.03*(i+1);
        for (std::size_t j=i;j<means;++j)
            residual_h[packed_index(width,i,j)]=0.01*(1+i+j);
    }
    for (double residual_value : {-1.2,0.8,0.0}) {
        if (residual_value==0.0) {theta[gamma]=0.0;theta[delta]=2.0;}
        for (std::size_t i=0;i<width;++i)
            dense_g[i]=sparse_g[i]=fixed_g[i]=0.17*(i+1);
        for (std::size_t i=0;i<packed;++i)
            dense_h[i]=sparse_h[i]=fixed_h[i]=0.002*(i+1);
        State residual{residual_value,residual_g.data(),residual_h.data()};
        State dense{0.3,dense_g.data(),dense_h.data()};
        State sparse{0.3,sparse_g.data(),sparse_h.data()};
        State fixed{0.3,fixed_g.data(),fixed_h.data()};
        assert((add_joint_shock<Order,0>(dense,residual,false,theta.data(),alpha,gamma,delta,width)));
        assert((add_joint_shock<Order,0,true>(sparse,residual,false,theta.data(),alpha,gamma,delta,width,means)));
        assert((add_joint_shock<Order,width,true,means>(fixed,residual,false,theta.data(),alpha,gamma,delta,width,means)));
        assert(dense.value==sparse.value && dense.value==fixed.value);
        for (std::size_t i=0;i<width;++i) {
            assert(std::abs(dense_g[i]-sparse_g[i])<=1e-14*(1+std::abs(dense_g[i])));
            assert(std::abs(dense_g[i]-fixed_g[i])<=1e-14*(1+std::abs(dense_g[i])));
        }
        for (std::size_t i=0;i<packed;++i) {
            assert(std::abs(dense_h[i]-sparse_h[i])<=1e-14*(1+std::abs(dense_h[i])));
            assert(std::abs(dense_h[i]-fixed_h[i])<=1e-14*(1+std::abs(dense_h[i])));
        }
    }
}

template<int Order,std::size_t StaticWidth>
void check_scalar_state_precision()
{
    using namespace model::egarch_equation;
    struct State {
        long double value=0;
        std::array<double,3> gradient{};
        std::array<double,6> hessian{};
    };
    struct Moment {
        long double value=1;
        std::array<double,0> gradient{},hessian{};
    };
    const double theta[]{1.0,0.0,0.5};
    State current,previous;
    previous.value=std::nextafter(1.0L,2.0L);
    previous.gradient[2]=0.2;
    previous.hessian[5]=0.1;
    if constexpr (Order==0) current.gradient.fill(41.0);
    if constexpr (Order<2) current.hessian.fill(41.0);
    const long double centered=previous.value-1.0L;
    assert((add_shock<Order,StaticWidth>(current,previous,theta,0,1,Moment{},3,3)));
    assert(current.value==centered);
    if constexpr (Order>=1) {
        assert(current.gradient[0]==static_cast<double>(centered));
        assert(current.gradient[1]==static_cast<double>(previous.value));
        assert(current.gradient[2]==0.2);
    }
    if constexpr (Order>=2) {
        assert(current.hessian[packed_index(3,0,2)]==0.2);
        assert(current.hessian[packed_index(3,1,2)]==0.2);
        assert(current.hessian[5]==0.1);
    }
    add_persistence<Order,StaticWidth>(current,previous,theta,2,3);
    assert(current.value==centered+0.5L*previous.value);
    State standardized;
    if constexpr (Order==0) standardized.gradient.fill(41.0);
    if constexpr (Order<2) standardized.hessian.fill(41.0);
    long double inverse_scale=31.L;
    assert((standardize<Order,StaticWidth>(standardized,previous,current,3,&inverse_scale)));
    assert(inverse_scale==std::exp(-0.5L*current.value));
    assert(standardized.value==previous.value*std::exp(-0.5L*current.value));
    if constexpr (Order==0) {
        for (double entry:current.gradient) assert(entry==41.0);
        for (double entry:standardized.gradient) assert(entry==41.0);
    }
    if constexpr (Order<2) {
        for (double entry:current.hessian) assert(entry==41.0);
        for (double entry:standardized.hessian) assert(entry==41.0);
    }
}

struct ExtendedPointState {
    using Scalar=long double;
    unsigned* preparations;
    template<int Order,class Density>
    auto absolute_moment(const double*) const noexcept {
        static_assert(std::is_same_v<Density,generated::NormalDensity>);
        ++preparations[Order];
        struct Moment {
            bool valid=true;
            long double value=std::sqrt(2.0L/std::acos(-1.0L));
            std::array<double,0> gradient{},hessian{};
        };
        return Moment{};
    }
};

template<class Orders>
void check_retained_program()
{
    using Base=runtime::LinearPowerProgram<generated::NormalDensity,false,ExtendedPointState>;
    using Program=runtime::RetainedEgarchProgram<Base>;
    static_assert(std::is_same_v<typename Program::Layout,typename Base::Layout>);
    static_assert(Program::template fixed_layout<mean::LinearVolatilityOrders<>> == nullptr);
    static_assert(!runtime::PowerConstraints<Program>::fixed);
    const auto check_layout=[]<class Fixed>() {
        constexpr auto fixed=Program::template fixed_layout<Fixed>;
        static_assert(fixed==Base::template fixed_layout<Fixed>);
        static_assert(fixed==&runtime::arma_egarch::fixed_layout<Fixed,generated::NormalDensity>);
        using Constraints=runtime::PowerConstraints<Program,fixed>;
        static_assert(Constraints::fixed);
        using Width=decltype(std::declval<const Constraints&>()
            .template field<&Program::Layout::parameters>());
        static_assert(Width::value==fixed->parameters);
        const Constraints constraints{Program{},
            mean::PolynomialConstraints{fixed->parameters,{},nullptr,nullptr,0},
            false,nullptr,nullptr};
        assert(&constraints.layout()==fixed);
        assert(constraints.dimension()==fixed->parameters);
        assert(constraints.mean.dimension()==constraints.dimension());
    };
    check_layout.template operator()<mean::LinearVolatilityOrders<4,1,1>>();
    check_layout.template operator()<mean::LinearVolatilityOrders<7,3,2>>();
    check_layout.template operator()<mean::LinearVolatilityOrders<11,4,5>>();
    Program program;
    assert(program.prepare(2,3,2));
    unsigned preparations[3]{};
    program.egarch_point_state.preparations=preparations;
    const auto k=program.layout.parameters;
    constexpr std::size_t n=24;
    std::vector<double> y(n),x(2*n),theta(k,.01),z(k),other(k);
    for (std::size_t t=0;t<n;++t) {
        y[t]=.3+std::sin(double(t)+.2);
        x[2*t]=1.;x[2*t+1]=std::cos(double(t));
    }
    theta[program.layout.beta_start]=.2;
    theta[program.layout.beta_start+1]=.3;
    runtime::WorkspaceSizer plan;
    assert(program.plan_retention(plan,n));
    const auto bytes=plan.size(),kernel=program.bytes(7,true);
    std::vector<std::max_align_t> storage((bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
    std::vector<std::max_align_t> scratch((kernel+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
    runtime::WorkspaceArena short_arena(storage.data(),bytes-1);
    assert(!program.bind_retention(short_arena,n));
    runtime::WorkspaceArena null_arena(nullptr,bytes);
    assert(!program.bind_retention(null_arena,n));
    runtime::WorkspaceArena empty_arena(storage.data(),bytes);
    assert(!program.bind_retention(empty_arena,0));
    runtime::WorkspaceArena arena(storage.data(),bytes);
    assert(program.bind_retention(arena,n));
    runtime::WorkspaceSizer overflow;
    assert(!program.plan_retention(overflow,std::numeric_limits<std::size_t>::max()));
    runtime::WorkspaceSizer empty;
    assert(!program.plan_retention(empty,0));
    assert(program.map(theta.data(),z.data(),false,scratch.data(),kernel));
    std::vector<double> g(k),h(k*k),expected_g(k),expected_h(k*k);
    double f=0,expected_f=0;
    const auto evaluate=[&](const double* point,unsigned mask,bool transformed,
        std::size_t size=std::numeric_limits<std::size_t>::max(),bool checked=false) {
        const auto before=allocations;
        const auto status=program.template evaluate<Orders>(point,y.data(),x.data(),n,1.,mask,
            transformed,scratch.data(),std::min(size,kernel),&f,g.data(),h.data(),
            nullptr,nullptr,checked);
        assert(allocations==before);
        return status;
    };
    for (bool transformed:{false,true}) {
        const auto& point=transformed?z:theta;
        runtime::WorkspaceArena reset(storage.data(),bytes);
        assert(program.bind_retention(reset,n));
        assert(evaluate(point.data(),1,transformed)==VLK_JET_OK);
        assert(program.paths[1].complete && program.accepted==0);
        assert(evaluate(point.data(),2,transformed)==VLK_JET_OK);
        assert(program.accepted==1 && program.trial==0);
        for (unsigned mask=1;mask<8;++mask) {
            f=expected_f=37.;
            std::fill(g.begin(),g.end(),37.);std::fill(h.begin(),h.end(),37.);
            expected_g=g;expected_h=h;
            const Base& base=program;
            const auto expected=base.template evaluate<Orders>(point.data(),y.data(),x.data(),n,1.,
                mask,transformed,scratch.data(),kernel,&expected_f,expected_g.data(),
                expected_h.data(),nullptr,nullptr,false);
            assert(evaluate(point.data(),mask,transformed)==expected);
            assert(f==expected_f && g==expected_g && h==expected_h);
        }
        assert(evaluate(point.data(),2,transformed)==VLK_JET_OK);
        const auto accepted=program.accepted;
        other=point;other[0]=std::nextafter(other[0],1.);
        assert(!program.paths[accepted].matches_input(other.data(),transformed));
        assert(!program.paths[accepted].matches_input(point.data(),!transformed));
        assert(evaluate(other.data(),2,transformed)==VLK_JET_OK);
        assert(program.accepted==accepted);
        // Rejected trials cannot overwrite the accepted state or publish a
        // partially recorded path. Missing gradients remain failures.
        other[program.layout.omega_index]=std::numeric_limits<double>::quiet_NaN();
        assert(evaluate(other.data(),1,transformed)!=VLK_JET_OK);
        assert(!program.paths[program.trial].complete);
        assert(program.paths[accepted].complete);
        assert(evaluate(point.data(),2,transformed,0)!=VLK_JET_OK);
        assert(program.accepted==accepted);
        assert(evaluate(point.data(),2,transformed)==VLK_JET_OK);
        const auto old_y=program.paths[accepted].y;
        program.paths[accepted].y=nullptr;
        assert(evaluate(point.data(),2,transformed)!=VLK_JET_OK);
        program.paths[accepted].y=old_y;
        assert(evaluate(point.data(),2,transformed)==VLK_JET_OK);
        // Checked calls take the canonical validation path, not replay.
        const double saved=y[0];y[0]=std::numeric_limits<double>::quiet_NaN();
        assert(evaluate(point.data(),2,transformed,kernel,true)==VLK_JET_INVALID_ARGUMENT);
        y[0]=saved;
    }
}

struct ObservedPointState : ExtendedPointState {
    long double* states;
    std::size_t* calls;
    void observe_recursion(std::size_t time, long double residual,
        long double log_variance, long double standardized) const noexcept {
        states[3*time]=residual;
        states[3*time+1]=log_variance;
        states[3*time+2]=standardized;
        ++*calls;
    }
};

struct ScaleObservedPointState : ObservedPointState {
    long double* inverse_scales;
    void observe_recursion(std::size_t time, long double residual,
        long double log_variance, long double standardized, long double inverse_scale) const noexcept {
        ObservedPointState::observe_recursion(time,residual,log_variance,standardized);
        inverse_scales[time]=inverse_scale;
    }
};

struct ReplayedPointState : ExtendedPointState {
    const long double* states;
    const long double* scales;
    bool valid;
    bool valid_recursion(const runtime::arma_egarch::Layout&,const double*,
        const double*,std::size_t,double,const double*) const noexcept {return valid;}
    void restore_recursion(std::size_t time,long double& residual,long double& log_variance,
        long double& standardized,long double& inverse_scale) const noexcept {
        residual=states[3*time];log_variance=states[3*time+1];
        standardized=states[3*time+2];inverse_scale=scales[time];
    }
};

template<class Orders>
void check_canonical_scalar_workspace(double moving_average=0.0)
{
    namespace jet=runtime::arma_egarch;
    using Density=generated::NormalDensity;
    jet::Layout layout;
    if constexpr (Orders::linear)
        assert(jet::make_linear_layout<Density>(2,3,2,layout));
    else assert(jet::make_layout<Density>(2,1,3,2,layout));
    constexpr std::size_t n=24;
    std::vector<double> y(n),x(n*2),theta(layout.parameters,0.0);
    for (std::size_t t=0;t<n;++t) {
        y[t]=0.3+std::sin(double(t)+0.2);
        x[2*t]=1.0;x[2*t+1]=std::cos(double(t));
    }
    theta[0]=0.02;theta[1]=0.03;
    if constexpr (!Orders::linear) theta[layout.ma_start]=moving_average;
    theta[layout.omega_index]=-0.1;
    for (std::size_t j=0;j<3;++j) {
        theta[layout.alpha_start+j]=0.04;
        theta[layout.gamma_start+j]=0.01;
    }
    for (std::size_t j=0;j<2;++j) theta[layout.beta_start+j]=0.2;
    unsigned preparations[3]{};
    const ExtendedPointState point{preparations};
    const auto run=[&]<unsigned Mask>(bool short_arena=false) {
        constexpr int Order=(Mask&4)?2:(Mask&2)?1:0;
        const auto bytes=jet::core_workspace_bytes<Order,long double>(layout);
        std::vector<std::max_align_t> work((bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
        double f=31.;
        std::vector<double> g(layout.parameters,31.),h(layout.parameters*layout.parameters,31.);
        const bool valid=jet::evaluate_theta_order<Order,bool(Mask&1),bool(Mask&2),bool(Mask&4),Orders,Density>(
            layout,theta.data(),y.data(),n,1.0,work.data(),bytes-short_arena,
            &f,g.data(),h.data(),Orders::linear?x.data():nullptr,nullptr,nullptr,point);
        assert(valid!=short_arena);
        if (short_arena || !(Mask&1)) assert(f==31.);
        if (short_arena || !(Mask&2)) for (double v:g) assert(v==31.);
        if (short_arena || !(Mask&4)) for (double v:h) assert(v==31.);
        const auto control=std::tuple{f,g,h};
        std::vector<long double> states(3*n,31.L);
        unsigned observed_preparations[3]{};
        std::size_t calls=0;
        const ObservedPointState observed{{observed_preparations},states.data(),&calls};
        const bool observed_valid=jet::evaluate_theta_order<Order,bool(Mask&1),bool(Mask&2),bool(Mask&4),Orders,Density>(
            layout,theta.data(),y.data(),n,1.0,work.data(),bytes-short_arena,
            &f,g.data(),h.data(),Orders::linear?x.data():nullptr,nullptr,nullptr,observed);
        assert(observed_valid==valid);
        assert((std::tuple{f,g,h}==control));
        const auto start=Orders::linear?0:layout.maximum_lag;
        assert(calls==(short_arena?0:n-start));
        for (std::size_t t=0;t<(short_arena?n:start);++t)
            for (std::size_t j=0;j<3;++j) assert(states[3*t+j]==31.L);
        for (std::size_t t=start;!short_arena && t<n;++t) {
            assert(std::isfinite(states[3*t]) && std::isfinite(states[3*t+1]));
            assert(states[3*t+2]==states[3*t]*std::exp(-states[3*t+1]/2));
        }
        const auto expected_states=states;
        std::vector<long double> scales(n,31.L);
        std::fill(states.begin(),states.end(),31.L);
        calls=0;
        const ScaleObservedPointState scale_observed{observed,scales.data()};
        const bool scale_valid=jet::evaluate_theta_order<Order,bool(Mask&1),bool(Mask&2),bool(Mask&4),Orders,Density>(
            layout,theta.data(),y.data(),n,1.0,work.data(),bytes-short_arena,
            &f,g.data(),h.data(),Orders::linear?x.data():nullptr,nullptr,nullptr,scale_observed);
        assert(scale_valid==valid && (std::tuple{f,g,h}==control));
        assert(states==expected_states && calls==(short_arena?0:n-start));
        for (std::size_t t=0;t<n;++t)
            assert(scales[t]==(short_arena || t<start?31.L:std::exp(-states[3*t+1]/2)));
        if (!short_arena) {
            // Capture F alone, then reuse its exact state for every derivative mask.
            double captured_value=31.;
            assert((jet::evaluate_theta_order<0,true,false,false,Orders,Density>(
                layout,theta.data(),y.data(),n,1.0,work.data(),bytes,
                &captured_value,nullptr,nullptr,Orders::linear?x.data():nullptr,
                nullptr,nullptr,scale_observed)));
            assert(states==expected_states);
        }
        ReplayedPointState replay{{observed_preparations},states.data(),scales.data(),!short_arena};
        const bool replay_valid=jet::evaluate_theta_order<Order,bool(Mask&1),bool(Mask&2),bool(Mask&4),Orders,Density>(
            layout,theta.data(),y.data(),n,1.0,work.data(),bytes-short_arena,
            &f,g.data(),h.data(),Orders::linear?x.data():nullptr,nullptr,nullptr,replay);
        assert(replay_valid==valid && (std::tuple{f,g,h}==control));
        replay.valid=false;
        const std::array<unsigned,3> before_rejected{observed_preparations[0],
            observed_preparations[1],observed_preparations[2]};
        assert(!(jet::evaluate_theta_order<Order,bool(Mask&1),bool(Mask&2),bool(Mask&4),Orders,Density>(
            layout,theta.data(),y.data(),n,1.0,work.data(),bytes,
            &f,g.data(),h.data(),Orders::linear?x.data():nullptr,nullptr,nullptr,replay)));
        for (unsigned j=0;j<3;++j) assert(observed_preparations[j]==before_rejected[j]);
        assert((std::tuple{f,g,h}==control));
        return std::tuple{f,g,h};
    };
    const auto full=run.template operator()<7>();
    const auto compare=[&]<unsigned Mask>() {
        const auto actual=run.template operator()<Mask>();
        if constexpr (Mask&1) assert(std::abs(std::get<0>(actual)-std::get<0>(full))<1e-12);
        if constexpr (Mask&2) for (std::size_t j=0;j<layout.parameters;++j)
            assert(std::abs(std::get<1>(actual)[j]-std::get<1>(full)[j])<1e-12);
        if constexpr (Mask&4) for (std::size_t j=0;j<layout.parameters*layout.parameters;++j)
            assert(std::abs(std::get<2>(actual)[j]-std::get<2>(full)[j])<1e-11);
        const std::array<unsigned,3> before{preparations[0],preparations[1],preparations[2]};
        run.template operator()<Mask>(true);
        for (unsigned j=0;j<3;++j) assert(preparations[j]==before[j]);
    };
    compare.template operator()<1>();compare.template operator()<2>();
    compare.template operator()<3>();compare.template operator()<4>();
    compare.template operator()<5>();compare.template operator()<6>();
    compare.template operator()<7>();
    assert(preparations[0]==1 && preparations[1]==2 && preparations[2]==5);
}

template<class Density=generated::NormalDensity>
struct ConstraintProbeProgram {
    using density_type=Density;
    static constexpr bool aparch=true;
    runtime::aparch::Layout layout;
    generated::PowerMomentValues<density_type>* moment_values=nullptr;
};

void check_zero_coefficient_constraint_requests()
{
    using Density=generated::NormalDensity;
    const auto full=generated::asymmetric_power_moment<1,Density>(nullptr,.3,275.);
    const auto value=generated::asymmetric_power_moment<0,Density>(nullptr,.3,275.);
    assert(value.valid && std::isfinite(value.value) && !full.valid);
    for (const auto [p,q]: {std::pair{1u,1u},std::pair{3u,2u}}) {
        ConstraintProbeProgram<> program;
        auto& l=program.layout;
        assert(runtime::aparch::make_layout<Density>(0,0,p,q,false,l));
        const auto k=l.parameters;
        std::vector<double> theta(k,0.),jacobian(k,47.);
        theta[l.omega_index]=.1;
        theta[l.delta_index]=275.;
        for (std::size_t i=0;i<p;++i) theta[l.gamma_start+i]=.3;
        for (std::size_t i=0;i<q;++i) theta[l.beta_start+i]=.8/q;
        double local=0.,constraint=31.;
        const runtime::PowerConstraints constraints{program,
            mean::PolynomialConstraints{k,{},nullptr,nullptr,0},false,&local,&local};
        for (auto request:{kraft::ConstraintRequest::value,
                           kraft::ConstraintRequest::jacobian,
                           kraft::ConstraintRequest::value_jacobian}) {
            constraint=31.;std::fill(jacobian.begin(),jacobian.end(),47.);
            assert(constraints.evaluate(request,theta.data(),&constraint,jacobian.data()));
            if (request==kraft::ConstraintRequest::jacobian) assert(constraint==31.);
            else assert(std::abs(constraint-.2)<1e-15);
            if (request==kraft::ConstraintRequest::value) {
                for (double entry:jacobian) assert(entry==47.);
            } else {
                for (std::size_t j=0;j<k;++j) {
                    const double expected=j>=l.alpha_start && j<l.alpha_start+p
                        ?-value.value:j>=l.beta_start && j<l.beta_start+q?-1.:0.;
                    assert(jacobian[j]==expected);
                }
            }
        }
        for (double coefficient:{1e-310,-1e-310}) {
            theta[l.alpha_start]=coefficient;
            assert(constraints.evaluate(kraft::ConstraintRequest::value_jacobian,
                theta.data(),&constraint,jacobian.data()));
            const double contribution=coefficient*value.value;
            assert(std::abs(constraint-(.2-contribution))<1e-15);
            assert(jacobian[l.alpha_start]==-value.value);
            // Independent closed-form log derivatives at delta=275,gamma=.3.
            // The omitted opposite-side ratio is below 1e-73.
            assert(std::abs(jacobian[l.gamma_start]/(-contribution*275./1.3)-1.)<2e-12);
            assert(std::abs(jacobian[l.delta_index]/(-contribution*3.070750915218952)-1.)<2e-12);
            for (double entry:jacobian) assert(std::isfinite(entry));
        }
        theta[l.alpha_start]=0.;
        theta[l.delta_index]=400.;
        assert((!generated::asymmetric_power_moment<0,Density>(nullptr,.3,400.).valid));
        assert(runtime::aparch::valid_theta<Density>(l,theta.data()));
        theta[l.gamma_start]=1.;
        assert(!runtime::aparch::valid_theta<Density>(l,theta.data()));
        theta[l.gamma_start]=.3;
        theta[l.alpha_start]=-1.;
        assert(!runtime::aparch::valid_theta<Density>(l,theta.data()));
        theta[l.alpha_start]=0.;
        constraint=31.;std::fill(jacobian.begin(),jacobian.end(),47.);
        assert(constraints.evaluate(kraft::ConstraintRequest::value,
            theta.data(),&constraint,jacobian.data()));
        assert(std::abs(constraint-.2)<1e-15);
        for (double entry:jacobian) assert(entry==47.);
        constraint=31.;
        assert(!constraints.evaluate(kraft::ConstraintRequest::jacobian,
            theta.data(),&constraint,jacobian.data()));
        assert(constraint==31.);
    }
    const double student[]{3.},skew[]{3.,0.};
    assert((!generated::valid_power_moment_domain<generated::StudentTDensity>(student,0.,3.)));
    assert((!generated::valid_power_moment_domain<generated::SkewTDensity>(skew,0.,3.)));
    assert((generated::valid_power_moment_domain<generated::StudentTDensity>(student,0.,2.)));
    assert((generated::valid_power_moment_domain<generated::SkewTDensity>(skew,0.,2.)));
}

void check_restricted_unrepresentable_moment()
{
    using Density=generated::GedDensity;
    for (const auto [p,q]: {std::pair{1u,1u},std::pair{3u,2u},std::pair{4u,3u}}) {
        ConstraintProbeProgram<Density> program;
        auto& l=program.layout;
        assert(runtime::aparch::make_layout<Density>(0,0,p,q,false,l));
        const auto k=l.parameters;
        std::vector<double> theta(k,0.),jacobian(k,99.);
        std::vector<std::size_t> selected;
        for (std::size_t i=0;i<k;++i) if (i!=l.alpha_start) selected.push_back(i);
        theta[l.omega_index]=.002193328412278832;
        theta[l.gamma_start]=-.999999981596122;
        theta[l.delta_index]=231.2017402277763;
        theta[l.model_parameters]=1.8456427668073634;
        for (std::size_t i=0;i<q;++i) theta[l.beta_start+i]=.6/q;
        const auto ordinary=generated::asymmetric_power_moment<0,Density>(
            theta.data()+l.model_parameters,0.,theta[l.delta_index]);
        assert(ordinary.valid);
        double local=0.,constraint=71.;
        const runtime::PowerConstraints constraints{program,
            mean::PolynomialConstraints{k,{},nullptr,nullptr,0},false,&local,&local};
        const runtime::ConstraintColumns columns{selected.data(),selected.size()};
        const auto before=allocations;
        assert(!constraints.evaluate(kraft::ConstraintRequest::jacobian,
            theta.data(),&constraint,jacobian.data()));
        assert(constraint==71.);
        for (auto request:{kraft::ConstraintRequest::value,kraft::ConstraintRequest::jacobian,
                           kraft::ConstraintRequest::value_jacobian}) {
            constraint=71.;std::fill(jacobian.begin(),jacobian.end(),99.);
            assert(constraints.evaluate(request,theta.data(),&constraint,jacobian.data(),columns));
            if (request==kraft::ConstraintRequest::jacobian) assert(constraint==71.);
            else assert(std::abs(constraint-.4)<1e-15);
            for (auto i:selected) {
                const double expected=i>=l.alpha_start && i<l.alpha_start+p?-ordinary.value:
                    i>=l.beta_start && i<l.beta_start+q?-1.:0.;
                assert(jacobian[i]==(request==kraft::ConstraintRequest::value?99.:expected));
            }
        }
        theta[l.gamma_start]=-1.;
        assert(!constraints.evaluate(kraft::ConstraintRequest::jacobian,
            theta.data(),&constraint,jacobian.data(),columns));
        assert(allocations==before);
    }
}

template<std::size_t StaticWidth>
void check_zero_weight_shock_requests()
{
    using namespace model::aparch_equation;
    constexpr std::size_t width=6,alpha=2,gamma=3,delta=5;
    double theta[width]={0.,0.1,0.,0.,0.8,1020.};
    double residual_gradient[width]={1.,0.,0.,0.,0.,0.};
    State residual{2.,residual_gradient,nullptr};
    double gradient[width]{};
    State power{0.1,gradient,nullptr};
    // 2^1020 is finite, but its gamma/mean derivatives overflow. At zero
    // weight only the alpha derivative contributes to this first-order request.
    assert((add_joint_shock<1,StaticWidth>(power,residual,false,
        theta,alpha,gamma,delta,width)));
    assert(power.value==0.1);
    for (std::size_t j=0;j<width;++j) {
        assert(std::isfinite(gradient[j]));
        assert(j==alpha?std::abs(gradient[j]/std::exp2(1020.)-1.)<1e-13:gradient[j]==0.);
    }
    std::fill_n(gradient,width,0.);
    add_fixed_shock<1,StaticWidth>(power,2.,theta,alpha,gamma,delta,width);
    for (std::size_t j=0;j<width;++j) {
        assert(std::isfinite(gradient[j]));
        assert(j==alpha?std::abs(gradient[j]/std::exp2(1020.)-1.)<1e-13:gradient[j]==0.);
    }
    theta[delta]=1100.;
    std::fill_n(gradient,width,71.);
    assert((add_joint_shock<0,StaticWidth>(power,residual,false,
        theta,alpha,gamma,delta,width)));
    add_fixed_shock<0,StaticWidth>(power,2.,theta,alpha,gamma,delta,width);
    assert(power.value==0.1);
    for (double value:gradient) assert(value==71.);
}

template<std::size_t StaticWidth>
void check_underflowing_joint_shock()
{
    using namespace model::aparch_equation;
    constexpr std::size_t width=6,alpha=2,gamma=3,delta=5,packed=21;
    double theta[width]={0.,.1,1e-300,.3,.8,.1};
    double residual_gradient[width]={1.,0.,0.,0.,0.,0.};
    double residual_hessian[packed]{};
    residual_hessian[packed_index(width,0,1)]=.4;
    double gradient[width]{},hessian[packed]{};
    State residual{1e-180,residual_gradient,residual_hessian};
    State power{.1,gradient,hessian};
    const auto before=allocations;
    assert((add_joint_shock<2,StaticWidth,true,2>(power,residual,false,
        theta,alpha,gamma,delta,width,2)));
    assert(allocations==before);
    for (double value:gradient) assert(std::isfinite(value));
    for (double value:hessian) assert(std::isfinite(value));
    // 100-digit evaluation of alpha*d*(1-gamma)*B^(d-1) and its
    // derivatives. The unweighted B^(d-2) overflows in binary64.
    const double ge=9.6496109511981545e-140;
    const double hee=-8.684649856078339e40;
    assert(std::abs(gradient[0]/ge-1.)<3e-12);
    assert(std::abs(hessian[0]/hee-1.)<3e-12);
    assert(std::abs(hessian[packed_index(width,0,alpha)]/9.649610951198154e160-1.)<3e-12);
    assert(std::abs(hessian[packed_index(width,0,gamma)]/(-1.378515850171165e-140)-1.)<3e-12);
    assert(std::abs(hessian[packed_index(width,0,1)]/(.4*ge)-1.)<3e-12);
    // Zero weight does not remove the alpha/residual cross derivative.
    theta[alpha]=0.;
    std::fill_n(gradient,width,0.);std::fill_n(hessian,packed,0.);
    assert((add_joint_shock<2,StaticWidth>(power,residual,false,
        theta,alpha,gamma,delta,width)));
    assert(gradient[0]==0. && hessian[0]==0.);
    assert(std::abs(hessian[packed_index(width,0,alpha)]/9.649610951198154e160-1.)<3e-12);
    // d=1 has exactly zero residual curvature, even though B^-1 overflows.
    theta[alpha]=.08;theta[delta]=1.;residual.value=1e-310;
    std::fill_n(gradient,width,0.);std::fill_n(hessian,packed,0.);
    assert((add_joint_shock<2,StaticWidth>(power,residual,false,
        theta,alpha,gamma,delta,width)));
    assert(std::abs(gradient[0]-.056)<1e-15 && hessian[0]==0.);
    // The leverage base itself can underflow while residual curvature is finite.
    theta[gamma]=std::nextafter(1.,0.);theta[delta]=2.;
    std::fill_n(gradient,width,0.);std::fill_n(hessian,packed,0.);
    assert((add_joint_shock<2,StaticWidth>(power,residual,false,
        theta,alpha,gamma,delta,width)));
    const double t=1.-theta[gamma];
    assert(std::abs(hessian[0]/(2.*theta[alpha]*t*t)-1.)<3e-12);
}

void emit_student_tail_maps()
{
    using Density=generated::StudentTDensity;
    using Shape=generated::DensityShapeTransformT<Density>;
    namespace map=runtime::aparch_transform;
    runtime::aparch::Layout l;
    assert(runtime::aparch::make_layout<Density>(0,0,1,1,false,l));
    const auto k=l.parameters;
    std::vector<std::byte> work(map::workspace_bytes(l,8,7));
    map::Scratch s;
    assert(map::bind_scratch(l,8,7,work.data(),work.size(),s));
    for(double nu:{4.,4.3,8.,11.7}) for(double gap:{.125,0x1p-20,0x1p-40,0x1p-48}) {
        std::vector<double> z(k,0.),g(k),h(k*k),theta(k);
        z[l.gamma_start]=std::atanh(.2);
        z[l.delta_index]=std::log1p(-gap)-std::log(gap);
        assert(Shape::unpack(&nu,z.data()+l.model_parameters));
        const auto before=allocations;
        assert((map::map_coordinates<0,Density,Shape>(l,z.data(),s)));
        std::copy_n(s.theta,k,theta.data());
        assert((map::map_coordinates<2,Density,Shape>(l,z.data(),s)));
        assert(std::memcmp(s.theta,theta.data(),k*sizeof(double))==0);
        std::fill_n(s.theta_gradient,k,0.);
        std::fill_n(s.theta_hessian,k*k,0.);
        s.theta_gradient[l.alpha_start]=1./s.theta[l.alpha_start];
        map::pullback_gradient(l,s,s.theta_gradient,g.data());
        assert((map::pullback_hessian<Density,Shape>(l,z.data(),s,h.data())));
        assert(allocations==before);
        std::printf("{\"student_tail_map\":true,\"theta\":[%.17g,%.17g,%.17g],\"coordinates\":[%.17g,%.17g,%.17g],\"gradient\":[",
            s.theta[l.gamma_start],s.theta[l.delta_index],s.theta[l.model_parameters],
            z[l.gamma_start],z[l.delta_index],z[l.model_parameters]);
        const std::size_t keys[]={l.gamma_start,l.delta_index,l.model_parameters};
        for(unsigned i=0;i<3;++i) std::printf("%s%.17g",i?",":"",g[keys[i]]);
        std::printf("],\"hessian\":[");
        for(unsigned i=0;i<3;++i) for(unsigned j=0;j<3;++j)
            std::printf("%s%.17g",i||j?",":"",h[keys[i]*k+keys[j]]);
        std::puts("]}");
    }
    const double shape[]={300.}, log_weight=std::log(1e-320);
    const auto m=generated::scaled_asymmetric_power_moment<2,generated::GedDensity>(shape,.9,1200.,log_weight);
    assert(m.valid && std::isfinite(m.value));
    assert((!generated::asymmetric_power_moment<2,generated::GedDensity>(shape,.9,1200.).valid));
    std::printf("{\"scaled_log_moment\":true,\"log_weight\":%.17g,\"value\":%.17g,\"gradient\":[",log_weight,m.value);
    for(unsigned i=0;i<3;++i) std::printf("%s%.17g",i?",":"",m.gradient[i]);
    std::printf("],\"hessian\":[");
    for(unsigned i=0;i<6;++i) std::printf("%s%.17g",i?",":"",m.hessian[i]);
    std::puts("]}");
}

void check_large_moment_coordinate_derivatives()
{
    using Density=generated::NormalDensity;
    using Shape=generated::NoShapeTransform;
    namespace map=runtime::aparch_transform;
    runtime::aparch::Layout layout;
    assert(runtime::aparch::make_layout<Density>(0,0,1,1,false,layout));
    std::vector<std::byte> work(map::workspace_bytes(layout,8,7));
    map::Scratch scratch;
    assert(map::bind_scratch(layout,8,7,work.data(),work.size(),scratch));
    const double z[5]={0.,0.,0.,0.,200.};
    assert((map::map_coordinates<2,Density,Shape>(layout,z,scratch)));
    std::fill_n(scratch.theta_gradient,5,0.);
    std::fill_n(scratch.theta_hessian,25,0.);
    scratch.theta_gradient[1]=1e200;
    double g[5],h[25];
    map::pullback_gradient(layout,scratch,scratch.theta_gradient,g);
    assert((map::pullback_hessian<Density,Shape>(layout,z,scratch,h)));
    // MP70: a=1e200/(3*E|Z|^200), l=(log(2)+psi(100.5))/2,
    // a_delta=-a*l; a_delta_delta=a*(l*l-trigamma(100.5)/4).
    const auto close=[](double actual,double expected) {
        assert(std::isfinite(actual));
        assert(std::abs(actual/expected-1.)<2e-11);
    };
    close(1e200*scratch.theta[1],5000268511864.234);
    close(g[4],-13246515163950.566);
    close(h[24],35079647699013.223);
    close(h[12],-1.9901068677219654e17);
    for (double value:g) assert(std::isfinite(value));
    for (double value:h) assert(std::isfinite(value));
}

void check_weighted_moment_consumers()
{
    using Density=generated::GeneralizedHyperbolicDensity;
    using Shape=generated::DensityShapeTransformT<Density>;
    namespace map=runtime::aparch_transform;
    const double shape[]={.22917865877731938,24.999999999999993,5.9999999985845776};
    constexpr double gamma=.9999998781130509,delta=216.965990611158;
    assert((generated::asymmetric_power_moment<0,Density>(shape,gamma,delta).valid));
    assert((!generated::asymmetric_power_moment<1,Density>(shape,gamma,delta).valid));
    assert((!generated::asymmetric_power_moment<2,Density>(shape,gamma,delta).valid));
    for (const auto [p,q]:{std::pair{1u,1u},std::pair{3u,2u}}) {
        ConstraintProbeProgram<Density> program;
        auto& l=program.layout;
        assert(runtime::aparch::make_linear_layout<Density>(4,p,q,l));
        const auto k=l.parameters;
        std::vector<double> z(k,0.),theta(k),gradient(k),hessian(k*k),jacobian(k);
        z[l.delta_index]=delta;
        assert(Shape::unpack(shape,z.data()+l.model_parameters));
        for (unsigned lag=0;lag<p;++lag) {
            z[l.gamma_start+lag]=std::atanh(gamma);
            z[l.alpha_start+lag]=std::log(.2/p);
        }
        for (unsigned lag=0;lag<q;++lag) z[l.beta_start+lag]=std::log(.8/q);
        std::vector<std::byte> work(map::workspace_bytes(l,8,7));
        map::Scratch scratch;
        assert(map::bind_scratch(l,8,7,work.data(),work.size(),scratch));
        const auto check=[&]<int Order>() {
            const auto before=allocations;
            assert((map::map_coordinates<Order,Density,Shape>(l,z.data(),scratch)));
            assert(allocations==before);
            if constexpr(Order==0) std::copy_n(scratch.theta,k,theta.data());
            else assert(std::memcmp(theta.data(),scratch.theta,k*sizeof(double))==0);
            if constexpr(Order>=1) {
                std::fill_n(scratch.theta_gradient,k,0.);
                scratch.theta_gradient[l.alpha_start]=1e307;
                map::pullback_gradient(l,scratch,scratch.theta_gradient,gradient.data());
                if constexpr(Order>=2) {
                    std::fill_n(scratch.theta_hessian,k*k,0.);
                    assert((map::pullback_hessian<Density,Shape>(l,z.data(),scratch,hessian.data())));
                }
                assert(allocations==before);
                for (double value:gradient) assert(std::isfinite(value));
                for (double value:hessian) assert(std::isfinite(value));
            }
        };
        check.template operator()<0>();
        check.template operator()<1>();
        const auto first_gradient=gradient;
        check.template operator()<2>();
        for (std::size_t j=0;j<k;++j)
            assert(std::abs(first_gradient[j]-gradient[j])<2e-12*(1.+std::abs(gradient[j])));
        double local=0.,constraint=0.;
        const runtime::PowerConstraints constraints{program,
            mean::PolynomialConstraints{k,{},nullptr,nullptr,0},false,&local,&local};
        for (auto request:{kraft::ConstraintRequest::value,kraft::ConstraintRequest::jacobian,
                           kraft::ConstraintRequest::value_jacobian}) {
            constraint=31.;std::fill(jacobian.begin(),jacobian.end(),47.);
            const auto before=allocations;
            assert(constraints.evaluate(request,theta.data(),&constraint,jacobian.data()));
            assert(allocations==before);
            if (request==kraft::ConstraintRequest::jacobian) assert(constraint==31.);
            else assert(std::abs(constraint-.5)<2e-10);
            for (double value:jacobian) {
                if (request==kraft::ConstraintRequest::value) assert(value==47.);
                else assert(std::isfinite(value));
            }
        }
        const auto array=[](const double* values,std::size_t count) {
            std::printf("[");
            for(std::size_t i=0;i<count;++i) std::printf("%s%.17g",i?",":"",values[i]);
            std::printf("]");
        };
        std::printf("{\"weighted_map\":true,\"p\":%u,\"q\":%u,\"alpha\":%zu,\"gamma\":%zu,"
            "\"delta\":%zu,\"shape\":%zu,\"value\":%.17g,\"z\":",p,q,
            l.alpha_start,l.gamma_start,l.delta_index,l.model_parameters,1e307*theta[l.alpha_start]);
        array(z.data(),k);std::printf(",\"theta\":");array(theta.data(),k);
        std::printf(",\"gradient\":");array(gradient.data(),k);
        std::printf(",\"hessian\":");array(hessian.data(),k*k);
        std::printf(",\"constraint_jacobian\":");array(jacobian.data(),k);
        std::printf("}\n");
    }
}

template<class Density>
void check_request_independent_power_map(const double* shape)
{
    namespace map=runtime::aparch_transform;
    using Shape=generated::DensityShapeTransformT<Density>;
    using Memo=generated::PowerMomentValues<Density>;
    for (bool joint:{false,true}) {
        runtime::aparch::Layout layout;
        if (joint) assert(runtime::aparch::make_layout<Density>(2,1,4,3,true,layout));
        else assert(runtime::aparch::make_linear_layout<Density>(4,1,1,layout));
        std::vector<double> z(layout.parameters,.15),expected(layout.parameters);
        assert(Shape::unpack(shape,z.data()+layout.model_parameters));
        std::vector<std::byte> work(map::workspace_bytes(layout,8,7));
        std::vector<std::byte> value_work(map::map_workspace_bytes(layout));
        map::Scratch scratch;
        map::MapScratch value_scratch;
        assert(map::bind_scratch(layout,8,7,work.data(),work.size(),scratch));
        assert(map::bind_map_scratch(layout,value_work.data(),value_work.size(),value_scratch));
        std::vector<typename Memo::Entry> entries(layout.p);
        Memo memo{entries.data(),entries.size()};
        for (double power:{.7,1.7}) {
            if constexpr (map::tail_coupled_delta<Density>)
                assert(map::inverse_unit_logistic(power/shape[0],z[layout.delta_index]));
            else z[layout.delta_index]=generated::inverse_softplus(power);
            assert((map::map_values<Density,Shape>(
                layout,z.data(),true,value_scratch,expected.data())));
            const auto check=[&]<int Order>(Memo* cache) {
                const auto before=allocations;
                assert((map::map_coordinates<Order,Density,Shape>(layout,z.data(),scratch,cache)));
                assert(allocations==before);
                for (std::size_t j=0;j<layout.parameters;++j) {
                    if (std::memcmp(&expected[j],scratch.theta+j,sizeof(double)))
                        std::fprintf(stderr,"map density=%u order=%d joint=%d power=%g parameter=%zu: %.17g != %.17g\n",
                            generated::power_density_code<Density>,Order,joint,power,j,
                            scratch.theta[j],expected[j]);
                    assert(std::memcmp(&expected[j],scratch.theta+j,sizeof(double))==0);
                }
            };
            // A derivative-first call must agree even without a preceding F cache.
            for (bool cached:{false,true}) {
                assert(memo.reset());
                auto* cache=cached?&memo:nullptr;
                check.template operator()<2>(cache);
                check.template operator()<0>(cache);
                check.template operator()<1>(cache);
                check.template operator()<2>(cache);
            }
        }
        // Reusing ordinary values must retain the strict-domain precision
        // fallback. Unit variance gives an independent one-ULP sign oracle.
        expected[layout.delta_index]=2.;
        std::fill_n(expected.data()+layout.alpha_start,layout.p,0.);
        std::fill_n(expected.data()+layout.gamma_start,layout.p,0.);
        std::fill_n(expected.data()+layout.beta_start,layout.q,0.);
        expected[layout.beta_start]=.75;
        for (std::size_t lag=0;lag<layout.p;++lag) {
            const auto value=generated::asymmetric_power_moment<0,Density>(
                expected.data()+layout.model_parameters,0.,2.);
            assert(value.valid);
            scratch.moment_values[lag]=value.value;
        }
        for (bool inside:{false,true}) {
            expected[layout.alpha_start]=std::nextafter(.25,inside?0.:1.);
            const auto ordinary=generated::power_persistence<Density>(layout,expected.data());
            const auto borrowed=generated::power_persistence<Density>(layout,expected.data(),
                nullptr,{scratch.moment_values,layout.p});
            assert(ordinary.valid && borrowed.valid && borrowed.refined);
            assert(ordinary.slack==borrowed.slack && (borrowed.slack>0.)==inside);
        }
    }
}

template<bool Aparch,class Density>
void check_linear_power_mapped_evaluation(
    const runtime::LinearPowerProgram<Density,Aparch>& program,
    const double* z,const double* theta)
{
    using Shape=generated::DensityShapeTransformT<Density>;
    using Dynamic=mean::LinearVolatilityOrders<>;
    const auto& layout=program.layout;
    const double y[]={.31,-.71,.28,1.02,-.62,.11,.95,-.43,.18};
    constexpr std::size_t n=std::size(y);
    std::vector<double> x(n*layout.mean_parameters);
    for (std::size_t t=0;t<n;++t)
        for (std::size_t j=0;j<layout.mean_parameters;++j)
            x[t*layout.mean_parameters+j]=j?.1*(t+1)/(j+1):1.;
    const auto core=program.bytes(1,false),bytes=program.bytes(1,true);
    assert(core>0 && bytes>core);
    std::vector<std::byte> work(bytes);
    std::array<double,n> residuals,variances,theta_residuals,theta_variances;
    double f=31.,theta_f=31.;
    const auto before=allocations;
    // Inspect the theta actually consumed by the canonical F-only evaluation.
    if constexpr (Aparch) {
        namespace jet=runtime::aparch;
        namespace map=runtime::aparch_transform;
        assert((jet::evaluate_order<0,true,Dynamic,Density,Shape,true,false,false>(
            layout,z,y,n,1.,true,work.data(),bytes,&f,nullptr,nullptr,
            x.data(),residuals.data(),variances.data())));
        map::Scratch scratch;
        assert(map::bind_scratch(layout,core,1,work.data(),bytes,scratch));
        assert(std::memcmp(scratch.theta,theta,layout.parameters*sizeof(double))==0);
        assert((jet::evaluate_order<0,true,Dynamic,Density,Shape,true,false,false>(
            layout,theta,y,n,1.,false,work.data(),core,&theta_f,nullptr,nullptr,
            x.data(),theta_residuals.data(),theta_variances.data())));
    } else {
        namespace jet=runtime::arma_egarch;
        namespace map=runtime::arma_egarch_transform;
        assert((jet::evaluate_transformed_order<0,true,false,false,Dynamic,Density,Shape>(
            layout,z,y,n,1.,1,work.data(),bytes,&f,nullptr,nullptr,
            x.data(),residuals.data(),variances.data())));
        map::Scratch scratch;
        assert(map::bind_scratch(layout,core,1,work.data(),bytes,scratch));
        assert(std::memcmp(scratch.theta,theta,layout.parameters*sizeof(double))==0);
        assert((jet::evaluate_theta_order<0,true,false,false,Dynamic,Density>(
            layout,theta,y,n,1.,work.data(),core,&theta_f,nullptr,nullptr,
            x.data(),theta_residuals.data(),theta_variances.data())));
    }
    assert(std::isfinite(f) && std::isfinite(theta_f) && f==theta_f);
    assert(residuals==theta_residuals && variances==theta_variances);
    assert(allocations==before);
}

template<bool Aparch,class Density,class Fixed>
void check_linear_power_program_map(const double* shape)
{
    using Program=runtime::LinearPowerProgram<Density,Aparch>;
    using Shape=typename Program::Shape;
    using Dynamic=mean::LinearVolatilityOrders<>;
    constexpr auto fixed_layout=Program::template fixed_layout<Fixed>;
    static_assert(fixed_layout!=nullptr);
    static_assert(Program::template fixed_layout<Dynamic> == nullptr);
    Program program,fixed_program;
    volatile std::size_t columns=Fixed::columns,p=Fixed::p,q=Fixed::q;
    assert(program.prepare(columns,p,q));
    assert(fixed_program.prepare(Fixed::columns,Fixed::p,Fixed::q));
    const auto& layout=program.layout;
    const auto k=layout.parameters;
    // Both fixed map directions must use their own layout, including arena sizing.
    fixed_program.layout={};
    assert(!fixed_program.layout.valid && fixed_program.layout.parameters==0);
    const auto bytes=[&] {
        if constexpr (Aparch) {
            namespace map=runtime::aparch_transform;
            const auto expected=map::map_workspace_bytes(layout);
            assert(map::map_workspace_bytes<fixed_layout>(fixed_program.layout)==expected);
            return expected;
        } else {
            namespace map=runtime::arma_egarch_transform;
            const auto expected=map::map_workspace_bytes(layout);
            assert(map::map_workspace_bytes<fixed_layout>(fixed_program.layout)==expected);
            return expected;
        }
    }();
    assert(bytes>0);
    std::vector<std::byte> work(bytes);
    std::vector<double> z(k,.15),theta(k),inverse(k),output(k+2,31.);
    assert(Shape::unpack(shape,z.data()+layout.model_parameters));
    for (std::size_t j=0;j<Fixed::columns;++j) z[j]=.01*(j+1);
    for (std::size_t lag=0;lag<Fixed::p;++lag)
        z[layout.gamma_start+lag]=lag%2?-.2:.15;
    for (std::size_t lag=0;lag<Fixed::q;++lag)
        z[layout.beta_start+lag]=Aparch?-.3-.07*lag:.2/Fixed::q+.01*lag;
    if constexpr (Aparch) {
        namespace map=runtime::aparch_transform;
        z[layout.omega_index]=generated::inverse_softplus(.2);
        if constexpr (map::tail_coupled_delta<Density>)
            assert(map::inverse_unit_logistic(1.7/shape[0],z[layout.delta_index]));
        else z[layout.delta_index]=generated::inverse_softplus(1.7);
    } else z[layout.omega_index]=-.1;
    const auto before=allocations;
    assert(program.template map<Dynamic>(z.data(),theta.data(),true,work.data(),bytes));
    assert(program.template map<Dynamic>(theta.data(),inverse.data(),false,work.data(),bytes));
    for (std::size_t j=0;j<k;++j)
        assert(std::isfinite(inverse[j]) && std::abs(inverse[j]-z[j])<1e-12);
    for (bool forward:{true,false}) {
        const auto* source=forward?z.data():theta.data();
        const auto* expected=forward?theta.data():inverse.data();
        const auto check=[&]<class Orders>(const Program& candidate) {
            std::fill(output.begin(),output.end(),31.);
            assert(candidate.template map<Orders>(source,output.data()+1,forward,work.data(),bytes));
            assert(std::memcmp(output.data()+1,expected,k*sizeof(double))==0);
            assert(output.front()==31. && output.back()==31.);
            for (auto size:{bytes-1,std::size_t{0}}) {
                std::fill(output.begin(),output.end(),31.);
                assert(!candidate.template map<Orders>(source,output.data()+1,forward,work.data(),size));
                for (double entry:output) assert(entry==31.);
            }
            assert(!candidate.template map<Orders>(source,output.data()+1,forward,nullptr,bytes));
            for (double entry:output) assert(entry==31.);
        };
        check.template operator()<Dynamic>(program);
        check.template operator()<Fixed>(fixed_program);
        // The default map remains the general-order fallback, including C11/45.
        assert(program.map(source,output.data()+1,forward,work.data(),bytes));
        assert(std::memcmp(output.data()+1,expected,k*sizeof(double))==0);
        assert(output.front()==31. && output.back()==31.);
    }
    assert(allocations==before);
    check_linear_power_mapped_evaluation(program,z.data(),theta.data());
}

template<class Density>
void check_linear_power_program_maps(const double* shape)
{
    const auto check=[&]<class Orders>() {
        check_linear_power_program_map<true,Density,Orders>(shape);
        check_linear_power_program_map<false,Density,Orders>(shape);
    };
    check.template operator()<mean::LinearVolatilityOrders<4,1,1>>();
    check.template operator()<mean::LinearVolatilityOrders<7,3,2>>();
    check.template operator()<mean::LinearVolatilityOrders<11,4,5>>();
}

template<class ScalarType=double>
struct CountedEgarchPointState {
    using Scalar=ScalarType;
    std::array<unsigned,3>* calls;
    template<int Order,class Density>
    auto absolute_moment(const double* shape) const noexcept
    {
        ++(*calls)[Order];
        return generated::absolute_moment<Order,Density>(shape);
    }
};

template<class Orders,class Scalar=double>
void check_checked_point_state_forwarding()
{
    namespace jet=runtime::arma_egarch;
    using Density=generated::NormalDensity;
    using Shape=generated::NoShapeTransform;
    jet::Layout layout;
    assert(jet::make_linear_layout<Density>(4,1,1,layout));
    constexpr std::size_t n=24,k=8;
    assert(layout.parameters==k);
    const double theta[k]={.02,.03,.04,.05,-.1,.12,-.04,.7};
    double y[n],x[n*4];
    for (std::size_t i=0;i<n;++i) {
        y[i]=std::sin(.7*double(i));
        for (std::size_t j=0;j<4;++j) x[4*i+j]=std::cos(.2*double(i*j));
    }
    for (bool transformed : {false,true}) for (unsigned mask=1;mask<=7;++mask) {
        struct WiderState {double parts[2];};
        assert((jet::workspace_bytes<Density,Shape,Orders,WiderState>(layout,transformed,mask)
            > jet::workspace_bytes<Density,Shape,Orders,double>(layout,transformed,mask)));
        const auto bytes=jet::workspace_bytes<Density,Shape,Orders,Scalar>(layout,transformed,mask);
        std::vector<long double> memory((bytes+sizeof(long double)-1)/sizeof(long double)+2,919.);
        void* arena=memory.data()+1;
        std::array<double,1+k+k*k> baseline,actual;
        baseline.fill(531.);actual.fill(531.);
        unsigned valid=0;
        assert((jet::evaluate<Density,Shape,Orders>(layout,theta,k,y,n,1.,transformed,
            mask,arena,bytes,baseline.data(),baseline.data()+1,
            baseline.data()+1+k,&valid,x)==VLK_JET_OK));
        std::array<unsigned,3> calls{};
        const CountedEgarchPointState<Scalar> point{&calls};
        assert((jet::evaluate<Density,Shape,Orders>(layout,theta,k,y,n,1.,transformed,
            mask,arena,bytes,actual.data(),actual.data()+1,
            actual.data()+1+k,&valid,x,nullptr,nullptr,point)==VLK_JET_OK));
        assert(valid==mask && memory.front()==919. && memory.back()==919.);
        if constexpr (std::is_same_v<Scalar,double>) assert(actual==baseline);
        else for (std::size_t i=0;i<actual.size();++i)
            assert(std::abs(actual[i]-baseline[i])<=2e-12*(1.+std::abs(baseline[i])));
        const unsigned expected=(mask&4)?2u:((mask&2)?1u:0u);
        for (unsigned order=0;order<3;++order) assert(calls[order]==(order==expected));
        runtime::LinearPowerProgram<Density,false,CountedEgarchPointState<Scalar>> program;
        assert(program.prepare(4,1,1));
        program.egarch_point_state=point;
        assert(program.bytes(mask,transformed)==bytes);
        for (bool checked : {false,true}) {
            calls.fill(0);
            std::array<double,1+k+k*k> through_program;
            through_program.fill(531.);
            assert(program.template evaluate<Orders>(theta,y,x,n,1.,mask,transformed,
                arena,bytes,through_program.data(),through_program.data()+1,
                through_program.data()+1+k,nullptr,nullptr,checked)==VLK_JET_OK);
            assert(through_program==actual);
            for (unsigned order=0;order<3;++order) assert(calls[order]==(order==expected));
        }
        calls.fill(0);
        assert(program.template evaluate<Orders>(theta,y,x,n,1.,mask,transformed,
            arena,bytes-1,actual.data(),actual.data()+1,actual.data()+1+k,
            nullptr,nullptr)==VLK_JET_INSUFFICIENT_WORKSPACE);
        assert((calls==std::array<unsigned,3>{}));
        const double saved=x[0];x[0]=std::numeric_limits<double>::quiet_NaN();
        assert(program.template evaluate<Orders>(theta,y,x,n,1.,mask,transformed,
            arena,bytes,actual.data(),actual.data()+1,actual.data()+1+k,
            nullptr,nullptr)==VLK_JET_INVALID_ARGUMENT);
        x[0]=saved;
        assert((calls==std::array<unsigned,3>{}));
        calls.fill(0);
        assert((jet::evaluate<Density,Shape,Orders>(layout,theta,k,y,n,1.,transformed,
            mask,nullptr,0,actual.data(),actual.data()+1,actual.data()+1+k,
            &valid,x,nullptr,nullptr,point)==VLK_JET_INSUFFICIENT_WORKSPACE));
        assert((valid==0 && calls==std::array<unsigned,3>{}));
        assert((jet::evaluate<Density,Shape,Orders>(layout,theta,k,y,n,1.,transformed,
            mask,arena,bytes-1,actual.data(),actual.data()+1,actual.data()+1+k,
            &valid,x,nullptr,nullptr,point)==VLK_JET_INSUFFICIENT_WORKSPACE));
        assert((valid==0 && calls==std::array<unsigned,3>{}));
    }
}

template<unsigned Mask>
void check_scaled_power_units()
{
    namespace jet=runtime::aparch;
    using Density=generated::StudentTDensity;
    using Fixed=jet::FixedJointOrders<2,1,4,3>;
    constexpr int Order=(Mask&4)?2:(Mask&2)?1:0;
    jet::Layout layout;
    assert(jet::make_layout<Density>(2,1,4,3,true,layout));
    const auto k=layout.parameters;
    std::vector<double> theta(k,.02),g(k,71.),h(k*k,71.),expected_g(k),expected_h(k*k);
    theta[0]=.03;theta[1]=.12;theta[2]=-.08;theta[3]=.06;
    theta[layout.omega_index]=.08;
    std::fill_n(theta.data()+layout.gamma_start,4,.15);
    std::fill_n(theta.data()+layout.beta_start,3,.24);
    theta[layout.delta_index]=1.7;theta[layout.model_parameters]=8.;
    const double data[]={.3,-.7,.1,1.2,-.2,.6,-.4,.15,1.,-.9,.3,.5};
    std::vector<std::byte> workspace(jet::core_workspace_bytes<2,true>(layout));
    double expected_f=0.;
    const auto before=allocations;
    assert((jet::evaluate_theta_order<2,true,jet::RuntimeOrders,Density,true,true,true>(
        layout,theta.data(),data,std::size(data),4.,workspace.data(),workspace.size(),
        &expected_f,expected_g.data(),expected_h.data())));
    const auto check=[&]<class Orders>() {
        double f=71.;std::fill(g.begin(),g.end(),71.);std::fill(h.begin(),h.end(),71.);
        assert((jet::evaluate_theta_order<Order,true,Orders,Density,
            bool(Mask&1),bool(Mask&2),bool(Mask&4),void,true>(
                layout,theta.data(),data,std::size(data),4.,workspace.data(),workspace.size(),
                &f,g.data(),h.data())));
        const auto same=[](double a,double b) {return std::abs(a-b)<=3e-11*(1.+std::abs(b));};
        assert((Mask&1)?same(f,expected_f):f==71.);
        for(std::size_t i=0;i<k;++i) assert((Mask&2)?same(g[i],expected_g[i]):g[i]==71.);
        for(std::size_t i=0;i<k*k;++i) assert((Mask&4)?same(h[i],expected_h[i]):h[i]==71.);
    };
    check.template operator()<jet::RuntimeOrders>();
    check.template operator()<Fixed>();
    assert(allocations==before);
}

int main()
{
    check_power_layout_orders<generated::NormalDensity>();
    check_power_layout_orders<generated::StudentTDensity>();
    check_power_layout_orders<generated::SkewTDensity>();
    check_power_layout_orders<generated::GedDensity>();
    check_power_layout_orders<generated::GeneralizedHyperbolicDensity>();
    check_power_layout_orders<generated::HyperbolicDensity>();
    check_power_layout_orders<generated::NormalInverseGaussianDensity>();
    check_power_layout_orders<generated::VarianceGammaDensity>();
    check_scaled_power_units<1>();check_scaled_power_units<2>();check_scaled_power_units<3>();
    check_scaled_power_units<4>();check_scaled_power_units<5>();check_scaled_power_units<6>();
    check_scaled_power_units<7>();
    check_skew_standardization_consumers<0>();
    check_skew_standardization_consumers<1>();
    check_skew_standardization_consumers<2>();
    check_persistence_precision_workspace();
    check_persistence_precision_workspace<runtime::aparch::FixedVolatilityOrders<3,2>>();
    emit_precision_moment_identities();
    emit_skew_tail_precision();
    check_retained_program<mean::LinearVolatilityOrders<>>();
    check_retained_program<mean::LinearVolatilityOrders<2,3,2>>();
    check_checked_point_state_forwarding<mean::LinearVolatilityOrders<>>();
    check_checked_point_state_forwarding<mean::LinearVolatilityOrders<4,1,1>>();
    check_checked_point_state_forwarding<mean::LinearVolatilityOrders<>,long double>();
    check_checked_point_state_forwarding<mean::LinearVolatilityOrders<4,1,1>,long double>();
    check_program_units_and_strict_constraint_values<false>();
    check_program_units_and_strict_constraint_values<true>();
    const runtime::arma_egarch::State binary64_state{};
    static_assert(std::is_same_v<decltype(binary64_state.rounded()),
        const runtime::arma_egarch::State&>);
    assert(&binary64_state.rounded()==&binary64_state);
    check_canonical_scalar_workspace<mean::LinearVolatilityOrders<>>();
    check_canonical_scalar_workspace<mean::LinearVolatilityOrders<2,3,2>>();
    check_canonical_scalar_workspace<runtime::arma_egarch::RuntimeOrders>();
    check_canonical_scalar_workspace<runtime::arma_egarch::FixedOrders<2,1,3,2>>();
    check_canonical_scalar_workspace<runtime::arma_egarch::RuntimeOrders>(0.17);
    check_canonical_scalar_workspace<runtime::arma_egarch::FixedOrders<2,1,3,2>>(0.17);
    check_scalar_state_precision<0,0>();
    check_scalar_state_precision<1,0>();
    check_scalar_state_precision<2,0>();
    check_scalar_state_precision<0,3>();
    check_scalar_state_precision<1,3>();
    check_scalar_state_precision<2,3>();
    check_sparse_power_shock<0>();
    check_sparse_power_shock<1>();
    check_sparse_power_shock<2>();
    check_point_owned_power_moments();
    check_domain_reuses_prepared_scalar_moments();
    check_power_batch_preparation();
    check_sublinear_power_moment_reduction();
    check_large_moment_coordinate_derivatives();
    emit_student_tail_maps();
    check_weighted_moment_consumers();
    const double student_shape[]{8.},skew_shape[]{8.,.2},ged_shape[]{1.7};
    const double gh_shape[]{.1,1.8,-.7},nig_shape[]{.1,1.8},vg_shape[]{.1,3.};
    check_linear_power_program_maps<generated::NormalDensity>(nullptr);
    check_linear_power_program_maps<generated::StudentTDensity>(student_shape);
    check_linear_power_program_maps<generated::SkewTDensity>(skew_shape);
    check_linear_power_program_maps<generated::GedDensity>(ged_shape);
    check_linear_power_program_maps<generated::GeneralizedHyperbolicDensity>(gh_shape);
    check_linear_power_program_maps<generated::NormalInverseGaussianDensity>(nig_shape);
    check_linear_power_program_maps<generated::HyperbolicDensity>(nig_shape);
    check_linear_power_program_maps<generated::VarianceGammaDensity>(vg_shape);
    check_fixed_power_batches<generated::NormalDensity>(nullptr);
    check_fixed_power_batches<generated::GedDensity>(ged_shape);
    check_fixed_power_batches<generated::GeneralizedHyperbolicDensity>(gh_shape);
    check_request_independent_power_map<generated::NormalDensity>(nullptr);
    check_request_independent_power_map<generated::StudentTDensity>(student_shape);
    check_request_independent_power_map<generated::SkewTDensity>(skew_shape);
    check_request_independent_power_map<generated::GedDensity>(ged_shape);
    check_request_independent_power_map<generated::GeneralizedHyperbolicDensity>(gh_shape);
    check_request_independent_power_map<generated::NormalInverseGaussianDensity>(nig_shape);
    check_request_independent_power_map<generated::HyperbolicDensity>(nig_shape);
    check_request_independent_power_map<generated::VarianceGammaDensity>(vg_shape);
    emit_power_batch_jets<generated::SkewTDensity>(skew_shape);
    emit_power_batch_jets<generated::GeneralizedHyperbolicDensity>(gh_shape);
    emit_power_batch_jets<generated::NormalInverseGaussianDensity>(nig_shape);
    emit_power_batch_jets<generated::HyperbolicDensity>(nig_shape);
    emit_power_batch_jets<generated::VarianceGammaDensity>(vg_shape);
    check_zero_coefficient_constraint_requests();
    check_restricted_unrepresentable_moment();
    check_zero_weight_shock_requests<0>();
    check_zero_weight_shock_requests<6>();
    check_underflowing_joint_shock<0>();
    check_underflowing_joint_shock<6>();
    check_shared_mixture_moments(false);
    check_shared_mixture_moments(true);
    check_deferred_normal_cancellation<mean::LinearDesign<4>,5>();
    check_deferred_normal_cancellation<mean::LinearDesign<>,mean::dynamic>();
    check_residual_cancellation<0,mean::LinearDesign<>>();
    check_residual_cancellation<1,mean::LinearDesign<>>();
    check_residual_cancellation<2,mean::LinearDesign<>>();
    check_residual_cancellation<0,mean::LinearDesign<4>>();
    check_residual_cancellation<1,mean::LinearDesign<4>>();
    check_residual_cancellation<2,mean::LinearDesign<4>>();
    check<false>();check<true>();
    check<false,generated::GedDensity>();check<true,generated::GedDensity>();
    check_trial_plan();
    PlannedObjective objective;
    double scale=2.,point=0.;
    const runtime::DiagonalObjective adapter{objective,&scale,&point};
    assert(!adapter.speculate_trial_gradient(2));
    assert(adapter.speculate_trial_gradient(3));
}
