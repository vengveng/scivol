// TEST_ONLY: linear mean composition, checked arenas and generated order parity.
#include "arma_egarch_runtime_jet.hpp"
#include "aparch_runtime_jet.hpp"
#include "diagonal_optimizer_scaling.hpp"
#include "gjr_transform_policy.hpp"
#include "constant_mean_jet.hpp"
#include "power_solver_constraints.hpp"
#include "linear_power_program.hpp"
#include <cassert>
#include <vector>
#include <tuple>

using namespace vlk::experiment;

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
    static inline bool reject=false;
    template<bool F, bool G, bool H>
    static auto prepare(const double* shape,std::size_t n) noexcept
    {
        ++preparations;
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
    assert((standardize<Order,StaticWidth>(standardized,previous,current,3)));
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

template<class Orders>
void check_canonical_scalar_workspace()
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

struct ConstraintProbeProgram {
    using density_type=generated::NormalDensity;
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
        ConstraintProbeProgram program;
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

int main()
{
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
    check_sublinear_power_moment_reduction();
    check_large_moment_coordinate_derivatives();
    check_zero_coefficient_constraint_requests();
    check_zero_weight_shock_requests<0>();
    check_zero_weight_shock_requests<6>();
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
