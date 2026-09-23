// SCIVOL-NATIVE: TEST_ONLY
#include "gh_first_order.hpp"
extern "C" {
#include "gh_math.h"
}

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>

struct Reference {
    double order, argument;
    std::array<double,6> jet;
};

static_assert(sizeof(gh_bessel_second_plan_t)<=7200,
    "Prepared Bessel state must not grow past the historical caller-owned plan.");

// Independent 70-digit log(K_v(x)) derivatives, including varying-order terms.
constexpr Reference references[] {
    {.5,.1,{1.2770838991417501,1.4933487469322395,-6.,
        2.493701642081384,-7.0133025061355205,49.99999999999999}},
    {.5,1.,{-.7742086473552726,.3613286168882226,-1.5,
        .7061585584523065,-.2773427662235548,.5}},
    {.5,10.,{-10.925501193852295,.04771854549596084,-1.05,
        .0953733561551795,-.004562909008078316,.005}},
    {3.357059340182185,8.,{-8.17055189834603,.3879408401484638,-1.1371768252535488,
        .11078407841018667,-.04418203098287694,.025067337098960526}},
    {15.5,20.,{-15.642812749540624,.7012121525963021,-1.2809039621945888,
        .03930704584783882,-.02988644605375616,.02395523774393301}},
    {20.5,8.,{10.915959009345425,1.6490369343809679,-2.759291605275581,
        .04618751539726619,-.11571267205910649,.2976275377151548}},
    {-50.,128.,{-120.58965894658779,-.3800017409128533,-1.0769734755351796,
        .007257595641696959,.0028247919442331583,.0011298788962945767}},
    {-6.5,128.,{-130.03683141752296,-.050563242864731205,-1.0051772947980184,
        .007772413699090717,.0003931727224092482,5.028898970941105e-5}},
    {.5,128.,{-130.20022377931508,.003891109044654463,-1.00390625,
        .007782179264299706,-3.028191069107381e-5,3.0517578125e-5}},
    {1.,128.,{-130.19730545845098,.007782159852961385,-1.0039289611790574,
        .007782004559144628,-6.056246693837244e-5,3.0871071400287656e-5}},
    {46.917166676811,11960.09555362426,{-11964.472418661398,.003922634642886268,
        -1.00004949835343,8.360723385704121e-5,-3.2796147365114877e-7,
        4.781761006358894e-9}},
    {49.5,128.,{-120.77875222389481,.37637058214000685,-1.0755672319955205,
        .007267029729688205,-.002800166261954983,.0011093900590215162}},
    {50.,1e8,{-100000008.98453651,4.999999974999792e-7,-1.000000005000125,
        9.99999994999875e-9,-4.999999949999376e-15,5.0002499749962504e-17}},
};

void close(double actual,double expected,double tolerance) {
    assert(std::isfinite(actual));
    if (std::abs(actual-expected)>tolerance*std::max(1.0,std::abs(expected)))
        std::fprintf(stderr,"Bessel/density mismatch: actual=%.17g expected=%.17g tolerance=%.3g\n",
            actual,expected,tolerance);
    assert(std::abs(actual-expected)<=tolerance*std::max(1.0,std::abs(expected)));
}

void density_coordinate_contract() {
    using namespace vlk::experiment;
    // Independent MP60 derivatives at residual=-1.3, log(variance)=-1.1.
    // Entries are NLL, residual, log variance, and the free shape derivatives.
    constexpr std::array<double,6> expected[] {
        {3.1653870182654757,-3.351381959195009,-1.678398273476756,
            1.2233962841989758,-.04517948140577654,.030200430082241804},
        {3.1711208873743493,-3.3649352712170404,-1.6872079262910764,
            1.3024359915425685,-.04589708339240376,0.},
        {2.829571335659548,-2.559655363458714,-1.1637759862481643,
            .5825505673841703,-.004113686507260239,0.},
        {3.022174995329909,-3.4438365362063514,-1.7384937485341285,
            1.4802149609512063,-.007886695695055288,0.},
    };
    constexpr double shapes[][3] {
        {.17,1.8,-.7},{.17,1.8,0.},{-.17,1.8,0.},
        {.07035000433159934,3.857059340182185,0.},
    };
    auto joint=std::make_unique<gh_first::Cache<5>>();
    auto fixed_residual=std::make_unique<gh_first::Cache<4>>();
    for (std::size_t family=0;family<4;++family) {
        const auto prepare=[&](auto& cache,std::size_t shape_start) {
            return family==3
                ? gh_first::prepare_variance_gamma(shapes[family],shape_start,cache)
                : gh_first::prepare_gh(shapes[family],shape_start,family==0,
                    family==1?-.5:1.,cache);
        };
        assert(prepare(*joint,2));
        assert(prepare(*fixed_residual,1));
        const auto direct=gh_first::observation_log_variance_nll<true>(*joint,-1.3,-1.1);
        close(direct.value,expected[family][0],5e-12);
        for (std::size_t j=0;j<5;++j)
            close(direct.gradient[j],expected[family][j+1],5e-12);
        for (const auto& point:std::array<std::array<double,2>,4>{{
                {-1.3,-1.1},{.2,.7},{2.1,1.3},{0.,-.2}}}) {
            const double variance=std::exp(point[1]);
            const auto theta=gh_first::observation_nll<true>(*joint,point[0],variance);
            const auto log=gh_first::observation_log_variance_nll<true>(*joint,point[0],point[1]);
            const auto sparse=gh_first::observation_log_variance_nll<false>(
                *fixed_residual,point[0],point[1]);
            close(log.value,theta.value,1e-13);
            close(sparse.value,log.value,1e-13);
            for (std::size_t j=0;j<5;++j) {
                close(log.gradient[j],theta.gradient[j]*(j==1?variance:1.),1e-13);
                if (j) close(sparse.gradient[j-1],log.gradient[j],1e-13);
            }
        }
    }
}

void variance_gamma_preparation_contract() {
    using namespace vlk::experiment;
    const double shape[]{-.11829769595989045,4.811443948964761};
    gh_shape_cache_t second;
    gh_first::Cache<4> first;
    vg_shape_value_cache_t value;
    assert(vg_shape_value_cache_init(shape[0],shape[1],&value));
    assert(gh_first::prepare_variance_gamma(shape,2,first));
    assert(vg_shape_cache_init(shape[0],shape[1],&second));
    // A one-ULP argument difference affected every observation in the saved fit.
    assert(value.theta==first.vg_theta.value);
    assert(value.theta==second.vg_theta.value);
    assert(value.argument_scale==first.vg_argument_scale.value);
    assert(value.argument_scale==second.vg_argument_scale.value);
    assert(value.log_normalizer==first.log_normalizer.value);
    assert(value.log_normalizer==second.log_normalizer.value);
}

void expansion_request_contract() {
    std::array<double,GH_BESSEL_EXPANSION_TERMS+1> coefficient,first,second;
    for (const auto& reference:references) {
        if (reference.argument<GH_BESSEL_EXPANSION_MIN_ARGUMENT) continue;
        gh_bessel_expansion_prepare(reference.order,coefficient.data(),first.data(),second.data());
        for (unsigned mask=1;mask<8;++mask) {
            std::array<double,6> output;
            output.fill(1234567.);
            assert(gh_bessel_expansion_evaluate(mask,coefficient.data(),first.data(),second.data(),
                reference.argument,&output[0],&output[1],&output[2],&output[3],&output[4],&output[5]));
            for (unsigned i=0;i<6;++i) {
                const bool requested=i==0?(mask&1):(i<3?(mask&2):(mask&4));
                if (!requested) assert(output[i]==1234567.);
                else if (!i) close(output[i],reference.jet[i],3e-15);
                else assert(std::abs(output[i]-reference.jet[i])
                    <=8e-14*std::abs(reference.jet[i])+1e-28);
            }
        }
    }
}

void shared_quadrature_storage_contract() {
    auto first=std::make_unique<gh_bessel_value_plan_t>();
    auto other=std::make_unique<gh_bessel_value_plan_t>();
    assert(gh_bessel_value_plan_init(3.,first.get()));
    assert(gh_bessel_value_plan_init(4.,other.get()));
    assert(first->cosh_t && first->cosh_t==other->cosh_t);
    const double* small_rule=first->cosh_t;
    assert(gh_bessel_value_plan_init(20.,first.get()));
    assert(gh_bessel_value_plan_init(30.,other.get()));
    assert(first->cosh_t && first->cosh_t==other->cosh_t && first->cosh_t!=small_rule);
    first->cosh_t=nullptr;
    double output=1234567.;
    assert(!gh_log_bessel_k_value_planned(first.get(),8.,&output));
    assert(output==1234567.);
}

void large_argument_density_gradient_contract() {
    using namespace vlk::experiment;
    auto cache=std::make_unique<gh_first::Cache<4>>();
    const double shape[] {-.9949103835917059,47.417166676811};
    assert(gh_first::prepare_variance_gamma(shape,2,*cache));
    const auto actual=gh_first::observation_nll<true>(*cache,-1.9715420758466602,1.);
    close(actual.value,2.809912968217987,3e-9);
    constexpr double gradient[] {-1.645610459514049,-1.1221951306926523,
        .00016476975930620224,.00028445018668820776};
    for (std::size_t j=0;j<4;++j)
        assert(std::abs(actual.gradient[j]-gradient[j])<=3e-7*std::abs(gradient[j])+3e-8);
}

void planned_rescaling_contract() {
    auto plan=std::make_unique<gh_bessel_first_plan_t>();
    for (double order : {-15.5,-3.7,0.,3.7,15.5}) {
        assert(gh_bessel_first_plan_init(order,1,plan.get()));
        for (double argument : {.01,.1,1.,8.,32.}) {
            // Historical online log-sum-exp, including exp(0) at a new maximum.
            double maximum=-INFINITY, total=0., order_total=0., argument_total=0.;
            for (std::size_t i=0;i<plan->value.count;++i) {
                const double term=plan->value.base[i]-argument*plan->value.cosh_t[i];
                if (term>maximum) {
                    const double scale=std::isfinite(maximum)?std::exp(maximum-term):0.;
                    total*=scale;
                    order_total*=scale;
                    argument_total*=scale;
                    maximum=term;
                }
                const double weight=std::exp(term-maximum);
                total+=weight;
                order_total+=weight*plan->order_score[i];
                argument_total-=weight*plan->value.cosh_t[i];
            }
            const double inverse=1./total;
            const std::array<double,3> expected{
                maximum+std::log(total),order_total*inverse,argument_total*inverse};
            std::array<double,3> actual;
            assert(gh_log_bessel_k_first_planned(plan.get(),argument,
                &actual[0],&actual[1],&actual[2]));
            for (std::size_t i=0;i<actual.size();++i)
                assert(std::bit_cast<std::uint64_t>(actual[i])
                    ==std::bit_cast<std::uint64_t>(expected[i]));
            assert(gh_log_bessel_k_value_planned(&plan->value,argument,&actual[0]));
            assert(std::bit_cast<std::uint64_t>(actual[0])
                ==std::bit_cast<std::uint64_t>(expected[0]));
        }
    }
}

int main() {
    density_coordinate_contract();
    variance_gamma_preparation_contract();
    expansion_request_contract();
    shared_quadrature_storage_contract();
    large_argument_density_gradient_contract();
    planned_rescaling_contract();
    auto second=std::make_unique<gh_bessel_second_plan_t>();
    auto value_plan=std::make_unique<gh_bessel_value_plan_t>();
    for (const auto& reference:references) {
        // Varying order deliberately disables the half-integer shortcut.
        assert(gh_bessel_second_plan_init(reference.order,1,second.get()));
        assert(second->first.value.fixed_kind==0);
        std::array<double,6> actual;
        assert(gh_log_bessel_k_second_planned(second.get(),reference.argument,
            &actual[0],&actual[1],&actual[2],&actual[3],&actual[4],&actual[5]));
        for (std::size_t i=0;i<6;++i)
            close(actual[i],reference.jet[i],i<3?3e-15:1e-13);
        if (reference.argument>=GH_BESSEL_EXPANSION_MIN_ARGUMENT)
            for (std::size_t i=1;i<6;++i)
                assert(std::abs(actual[i]-reference.jet[i])
                    <=8e-14*std::abs(reference.jet[i])+1e-28);
        assert(gh_log_bessel_k_first_planned(&second->first,reference.argument,
            &actual[0],&actual[1],&actual[2]));
        for (std::size_t i=0;i<3;++i) close(actual[i],reference.jet[i],3e-15);
        assert(gh_log_bessel_k_first(reference.order,reference.argument,1,
            &actual[0],&actual[1],&actual[2]));
        for (std::size_t i=0;i<3;++i) close(actual[i],reference.jet[i],3e-15);
        assert(gh_bessel_value_plan_init(reference.order,value_plan.get()));
        assert(gh_log_bessel_k_value_planned(value_plan.get(),reference.argument,&actual[0]));
        close(actual[0],reference.jet[0],3e-15);
    }
}
