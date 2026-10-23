// Shared-math cases retained from native_linear_power_workspace.cpp.
#include "models/power/aparch_moment_policy.hpp"
#include "models/gjr_moment_policy.hpp"
#include "models/power/aparch_equation_program.hpp"
#include "models/power/egarch_equation_program.hpp"
#include "models/mean/constant_mean_jet.hpp"
#include "models/power/aparch_jet.hpp"
#include "models/joint_power/layout.hpp"
#include <array>
#include <cassert>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <new>
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
    assert(runtime::aparch::make_layout<Density>(2,1,layout));
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

    constexpr auto* layout=&runtime::aparch::fixed_layout<P,1,Density>;
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
    assert(runtime::aparch::make_layout<Density>(1,1,l));
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
    assert(runtime::aparch::make_layout<Density>(1,1,layout));
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


template<class Density>
void check_request_independent_power_map(const double* shape)
{
    namespace map=runtime::aparch_transform;
    using Shape=generated::DensityShapeTransformT<Density>;
    using Memo=generated::PowerMomentValues<Density>;
    for (bool joint:{false,true}) {
        runtime::aparch::Layout layout;
        if (joint) assert((runtime::joint_power::make_layout<Density,true>(2,1,0,4,3,false,true,layout)));
        else assert((runtime::joint_power::make_layout<Density,true>(0,0,4,1,1,true,false,layout)));
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

int main()
{
    emit_precision_moment_identities();
    emit_skew_tail_precision();
    check_skew_standardization_consumers<0>();
    check_skew_standardization_consumers<1>();
    check_skew_standardization_consumers<2>();
    check_sublinear_power_moment_reduction();
    check_point_owned_power_moments();
    check_domain_reuses_prepared_scalar_moments();
    check_power_batch_preparation();
    const double skew[]{8.,.2},gh[]{.1,1.8,-.7},nig[]{.1,1.8},vg[]{.1,3.};
    emit_power_batch_jets<generated::SkewTDensity>(skew);
    emit_power_batch_jets<generated::GeneralizedHyperbolicDensity>(gh);
    emit_power_batch_jets<generated::NormalInverseGaussianDensity>(nig);
    emit_power_batch_jets<generated::HyperbolicDensity>(nig);
    emit_power_batch_jets<generated::VarianceGammaDensity>(vg);
    const double student[]{8.},ged[]{1.7};
    check_fixed_power_batches<generated::NormalDensity>(nullptr);
    check_fixed_power_batches<generated::GedDensity>(ged);
    check_fixed_power_batches<generated::GeneralizedHyperbolicDensity>(gh);
    emit_student_tail_maps();
    check_large_moment_coordinate_derivatives();
    check_request_independent_power_map<generated::NormalDensity>(nullptr);
    check_request_independent_power_map<generated::StudentTDensity>(student);
    check_request_independent_power_map<generated::SkewTDensity>(skew);
    check_request_independent_power_map<generated::GedDensity>(ged);
    check_request_independent_power_map<generated::GeneralizedHyperbolicDensity>(gh);
    check_request_independent_power_map<generated::NormalInverseGaussianDensity>(nig);
    check_request_independent_power_map<generated::HyperbolicDensity>(nig);
    check_request_independent_power_map<generated::VarianceGammaDensity>(vg);
    check_shared_mixture_moments(false);
    check_shared_mixture_moments(true);
    check_residual_cancellation<0,mean::LinearDesign<>>();
    check_residual_cancellation<1,mean::LinearDesign<>>();
    check_residual_cancellation<2,mean::LinearDesign<>>();
    check_residual_cancellation<0,mean::LinearDesign<4>>();
    check_residual_cancellation<1,mean::LinearDesign<4>>();
    check_residual_cancellation<2,mean::LinearDesign<4>>();
    check_deferred_normal_cancellation<mean::LinearDesign<4>,5>();
    check_deferred_normal_cancellation<mean::LinearDesign<>,mean::dynamic>();
    check_sparse_power_shock<0>();check_sparse_power_shock<1>();check_sparse_power_shock<2>();
    check_scalar_state_precision<0,0>();check_scalar_state_precision<1,0>();
    check_scalar_state_precision<2,0>();check_scalar_state_precision<0,3>();
    check_scalar_state_precision<1,3>();check_scalar_state_precision<2,3>();
    check_zero_weight_shock_requests<0>();check_zero_weight_shock_requests<6>();
    check_underflowing_joint_shock<0>();check_underflowing_joint_shock<6>();
}
