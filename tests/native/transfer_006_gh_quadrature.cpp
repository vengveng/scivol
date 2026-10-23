// SCIVOL-NATIVE: TEST_ONLY
#include "math/gh_first_order.hpp"
#include "models/garch_extended_density_policy.hpp"
#include "models/density_moment_quadrature.hpp"
extern "C" {
#include "math/gh_math.h"
}

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <utility>

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

template<class Family, bool ResidualDerivatives>
void packed_density_contract(const double* shape, unsigned family_code) {
    using namespace vlk::experiment;
    using Density = generated::GhFamilyDensity<Family>;
    constexpr std::size_t width = Family::shape_parameters + 1u + ResidualDerivatives;
    const auto prepared = Density::template prepare<true, true, true, ResidualDerivatives>(shape, 1);
    assert(prepared.valid);
    auto source = std::make_unique<gh_shape_cache_t>();
    auto plan = std::make_unique<gh_bessel_second_plan_t>();
    assert(Family::prepare_hessian(shape, source.get()));
    assert(Family::prepare_second_plan(shape, plan.get()));
    for (double residual : {-12., -1.3, .0, .7, 15.}) {
        for (double variance : {.01, .33287108369807955, 1., 17.}) {
            gh_obs_derivs_t expected;
            assert(gh_obs_derivatives_planned(residual, variance, source.get(), plan.get(), &expected));
            const auto actual = gh_first::observation_nll<ResidualDerivatives>(
                prepared.cache, residual, variance);
            assert(ad::finite(actual));
            close(actual.value, expected.value, 3e-12);
            const auto original_index = [](std::size_t index) {
                constexpr std::size_t first_shape = 1u + ResidualDerivatives;
                return index < first_shape ? index + !ResidualDerivatives
                    : Family::derivative_source(index - first_shape);
            };
            for (std::size_t row = 0; row < width; ++row) {
                close(actual.gradient[row], expected.grad[original_index(row)], 3e-11);
                for (std::size_t column = row; column < width; ++column)
                    close(actual.hessian[ad::packed_index<width>(row, column)],
                        expected.hess[original_index(row)*GH_DERIV_DIM + original_index(column)], 5e-10);
            }
            if (residual == -1.3 && variance == .33287108369807955) {
                std::printf("{\"family\":%u,\"residual_derivatives\":%s,\"value\":%.17g,\"gradient\":[",
                    family_code, ResidualDerivatives ? "true" : "false", actual.value);
                for (std::size_t i=0; i<width; ++i)
                    std::printf("%s%.17g", i ? "," : "", actual.gradient[i]);
                std::printf("],\"hessian\":[");
                for (std::size_t i=0; i<actual.hessian.size(); ++i)
                    std::printf("%s%.17g", i ? "," : "", actual.hessian[i]);
                std::printf("]}\n");
            }
        }
    }
    for (double invalid : std::array<double,4>{NAN, INFINITY, -1., 0.}) {
        const auto observation = Density::template observe<true,true,true,ResidualDerivatives>(
            prepared, invalid, -1.3, 1.69);
        assert(!observation.valid);
    }
}

void packed_density_contracts() {
    using namespace vlk::experiment::generated;
    const double gh[] {.17, 1.8, -.7}, fixed[] {.17, 1.8}, vg[] {.07, 3.85};
    const auto check = [&]<class Family>(const double* shape, unsigned code) {
        packed_density_contract<Family, false>(shape, code);
        packed_density_contract<Family, true>(shape, code);
    };
    check.template operator()<GeneralizedHyperbolicFamily>(gh, 4);
    check.template operator()<FixedLambdaGhFamily<NormalInverseGaussianLambda>>(fixed, 5);
    check.template operator()<FixedLambdaGhFamily<HyperbolicLambda>>(fixed, 6);
    check.template operator()<VarianceGammaFamily>(vg, 7);
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

void fixed_order_small_argument_contract() {
    // Independent 70-digit K0/K1 and Bessel-ODE argument derivatives.
    constexpr double cases[][4] {
        {1e-150,345.38776394910684,-1e150,1e300},
        {1e-30,69.07755278982137,-9.999999999999999e29,9.999999999999998e59},
        {1e-8,18.420680743952364,-100000000.00000018,9999999999999982.},
        {1e-4,9.210340322844822,-10000.000932627237,99999991.67372675},
        {.001,6.907751517131147,-1000.0070237152227,999993.9762354447},
        {.03,3.5047008070791974,-33.442241283679145,1108.4689851579355},
        {.09999999999999999,2.287861712107168,-10.24630680497563,98.47626490806643},
    };
    auto plan=std::make_unique<gh_bessel_second_plan_t>();
    for (double order : {-1.,1.}) {
        assert(gh_bessel_second_plan_init(order,0,plan.get()));
        for (const auto& test:cases) {
            std::array<double,6> result;
            assert(gh_log_bessel_k_value_planned(&plan->first.value,test[0],&result[0]));
            close(result[0],test[1],2e-15);
            assert(gh_log_bessel_k_first_planned(&plan->first,test[0],
                &result[0],&result[1],&result[2]));
            assert(result[1]==0);
            close(result[2],test[2],2e-15);
            assert(gh_log_bessel_k_second_planned(plan.get(),test[0],
                &result[0],&result[1],&result[2],&result[3],&result[4],&result[5]));
            assert(result[1]==0 && result[3]==0 && result[4]==0);
            close(result[5],test[3],2e-15);
        }
        double value;
        assert(gh_log_bessel_k_value_planned(&plan->first.value,1e-310,&value));
        close(value,-std::log(1e-310),2e-15);
    }
    // A neighboring order is not the special function at exactly 1 or 1/2.
    for (double special : {-.5,.5,-1.,1.})
    for (double direction : {-INFINITY,INFINITY}) {
        const double order=std::nextafter(special,direction);
        assert(gh_bessel_value_plan_init(order,&plan->first.value));
        assert(plan->first.value.fixed_kind==0);
        assert(gh_bessel_first_plan_init(order,0,&plan->first));
        assert(plan->first.value.fixed_kind==0);
    }
    assert(gh_bessel_value_plan_init(VG_LAMBDA_MIN-.5,&plan->first.value));
    assert(plan->first.value.fixed_kind==0);
    double value;
    assert(gh_log_bessel_k_value_planned(&plan->first.value,.03,&value));
    close(value,3.5047008070792042,3e-15);
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

// Independent 90-660 digit reference; argument derivatives include subnormals.
void small_bessel_product_contract() {
    const Reference fixtures[] = {
    // 70-digit independent log(x^nu K_nu(x)) jets, including the extended range.
    {1,2,{-1.2739241220005686,1.1003010599418401,-0.8143077587637895,0.39280847799447366,0.33154856299145297,-0.07025100536480072}},
    {1.0000000000000002,2,{-1.2739241220005684,1.1003010599418401,-0.8143077587637895,0.39280847799447366,0.33154856299145297,-0.07025100536480075}},
    {1.5,2,{-0.6755963586871628,1.29103196392626,-0.6666666666666666,0.3691306421028514,0.26102608897683927,-0.1111111111111111}},
    {1.99999,2,{0.014912366067749224,1.4687309706716192,-0.551176436033084,0.34125500879266013,0.20307205019292257,-0.13055460592339777}},
    {2,2,{0.014927053394518778,1.468734383218817,-0.5511744053177436,0.34125443079634993,0.20307101785642784,-0.13055483305398385}},
    {8,2,{13.235974210429008,2.728474594038345,-0.13963511908068543,0.12772723058458602,0.019006922871998885,-0.06676135958581786}},
    {14.772133090357961,0.5,{34.125203144982514,3.3519897985362883,-0.018146153370178632,0.06999019619629264,0.0013166275339420283,-0.03626654679108246}},
    {14.772133090357961,2,{34.05733481619798,3.3568872333675133,-0.07220260795885754,0.06928581628786541,0.005181898512198661,-0.035698446855814185}},
    {15,2,{34.82404783247749,3.372553328558275,-0.07104070567112257,0.06822175069363635,0.005017495755520584,-0.03513701409352841}},
    {49.499999999999986,0.5,{176.23363243598928,4.585011400502558,-0.0051544993392975275,0.020406359454624767,0.00010627251008856325,-0.010308439365753897}},
    {49.499999999999986,2,{176.21430699395754,4.585409677696137,-0.02060961463163926,0.02038994724071234,0.0004245682147943681,-0.01029587316558812}},
    {50,2,{178.5095519664773,4.595553079909747,-0.020399493852166488,0.02018435546109763,0.0004159589749051195,-0.010191085031665749}},
    {1.0000000000000018,0.00001,{-6.06442643287614e-10,0.11593152302637395,-0.0001162885698799636,1.6449339473141076,0.0013523031484727363,-10.628857001519432}},
    {1.0000000000000018,1e-30,{2.0593574074186077e-16,0.11593151565841538,-6.919348430547128e-29,1.6449340668482222,4.7877382703318926e-27,-68.19348430547153}},
    {1.0000000000000018,1e-100,{2.0593574074186077e-16,0.11593151565841538,-2.303744408149687e-98,1.6449340668482222,5.3072382980824e-96,-229.37444081496952}},
    {1.0000000000000018,0.0001,{-4.9131360505283364e-8,0.11593199968653584,-0.0009326272371662561,1.644927694844719,0.008697935635043738,-8.326273241456157}},
    {1.1,0.00001,{0.0194422765683791,0.2693922418900753,-0.000045110660831669916,1.4332991289515675,0.00033751032547571605,-4.413279301835362}},
    {1.4999999999999998,0.00001,{0.2257913525947276,0.7296371546385132,-0.000009999900000999996,0.9348022001448931,0.000019997551489475853,-0.9999800002999965}},
    {1.5,0.00001,{0.22579135259472777,0.7296371546385134,-0.00000999990000099999,0.934802200144893,0.000019997551489475836,-0.999980000299996}},
    {1.5,1e-100,{0.22579135264472744,0.7296371545385218,-1e-100,0.9348022005446793,2e-100,-1.0}},
    {1.5000000000000002,0.00001,{0.22579135259472793,0.7296371546385136,-0.000009999900000999986,0.9348022001448928,0.000019997551489475816,-0.9999800002999956}},
    {1.9999999999999998,0.00001,{0.6931471805349451,1.1159315156834122,-0.000004999999997092788,0.6449340667982265,0.0000049999999603779955,-0.49999999915283583}},
    {2,0.00001,{0.6931471805349453,1.1159315156834124,-0.000004999999997092786,0.6449340667982264,0.000004999999960377993,-0.4999999991528357}},
    {2.0000000000000004,0.00001,{0.6931471805349458,1.1159315156834126,-0.000004999999997092784,0.6449340667982263,0.000004999999960377989,-0.4999999991528355}},
    {3,0.00001,{2.079441541667336,1.6159315156646625,-0.00000249999999996875,0.39493406684197646,0.0000012499999999375,-0.249999999990625}},
    {10,0.00001,{19.0401521051182,2.944899769626975,-5.555555555553627e-7,0.10516633568161717,6.172839506166142e-8,-0.05555555555549768}},
    {49.5,0.00001,{176.23492107830282,4.584984831286327,-1.0309278350515352e-7,0.020407455046879554,2.1256244021680674e-9,-0.010309278350515129}},
    {1.0000000000000018,0.1,{-0.014723380886877678,0.16048295676224747,-0.24630680497562799,1.4620313075944122,0.6066704217730222,-1.5237350919335906}},
    {1.5,0.1,{0.22110153244905228,0.7374275181323322,-0.09090909090909091,0.9124745259916547,0.14060580583583077,-0.8264462809917356}},
    {2,0.1,{0.6906639230371662,1.1183529409047834,-0.04939172411208572,0.6403735273747413,0.047310849992016187,-0.48419126577333604}},
    {49.5,0.1,{176.23486953193955,4.584985894096776,-0.0010309267163043415,0.020407411219717425,0.00002125617433530775,-0.01030924478814892}},
    {1.0000000000000018,1e-310,{2.0593574074186077e-16,0.11593151565841538,-7.139173103429051e-308,1.6449340668482222,5.096779260076802e-305,-712.9173103429098}},
    {1.5,1e-310,{0.22579135264472744,0.7296371545385218,-1e-310,0.9348022005446793,2e-310,-1.0}},
    {1.8,1e-310,{0.48343387153358414,0.9781386138538068,-6.25e-311,0.7369741375017003,7.8125e-311,-0.625}},
    {49.5,1e-310,{176.23492107830333,4.5849848312863175,-1.030927835053e-312,0.02040745504687999,2.1256244022e-314,-0.010309278350515464}},
    };
    for (const auto& fixture:fixtures) {
        for (unsigned mask=1;mask<8;++mask) {
            gh_bessel_product_plan_t plan;
            const unsigned level=(mask&4)?2:((mask&2)?1:0);
            assert(gh_bessel_product_prepare(fixture.order,level,&plan));
            std::array<double,6> actual; actual.fill(1234567.);
            assert(gh_bessel_product_evaluate(mask,&plan,fixture.argument,actual.data()));
            for (unsigned i=0;i<6;++i) {
                if (mask&(i==0?1:(i<3?2:4))) {
                    assert(std::abs(actual[i]-fixture.jet[i])<=2e-12*std::abs(fixture.jet[i])+2e-14);
                } else assert(actual[i]==1234567.);
            }
        }
    }
    gh_bessel_product_plan_t plan;
    std::array<double,6> output; output.fill(1234567.);
    assert(gh_bessel_product_prepare(1.1,0,&plan));
    assert(!gh_bessel_product_evaluate(7,&plan,.01,output.data()));
    for (double x:{0.,-1.,double(INFINITY),double(NAN),std::nextafter(GH_BESSEL_PRODUCT_MAX_ARGUMENT,INFINITY)})
        assert(!gh_bessel_product_evaluate(1,&plan,x,output.data()));
    for (double item:output) assert(item==1234567.);
}


void variance_gamma_center_contract() {
    using namespace vlk::experiment;
    using Family=generated::VarianceGammaFamily;
    struct Case { double rho, shape, argument, variance; };
    const Case cases[] = {
        {0.,VG_LAMBDA_MIN,1e-5,1.}, {.3,VG_LAMBDA_MIN,1e-5,.7},
        {-.6,1.7,-1e-7,1.3}, {.4,2.5,1e-8,.5},
        {0.,3.,0.,1.}, {0.,50.,1e-5,1.},
        // Leave enough separation to survive the skew-center coordinate round-trip.
        {.1,2.,.1-1e-10,1.}, {.1,2.,.1+1e-10,1.},
    };
    for (const auto& point:cases) {
        const double shape[]={point.rho,point.shape};
        const double theta=std::sqrt(2.)*point.rho/std::sqrt(1.-point.rho*point.rho);
        const double scale=std::sqrt(point.shape*(1.+theta*theta));
        const double residual=std::sqrt(point.variance)
            *(point.argument/std::sqrt(theta*theta+2.)-theta*point.shape)/scale;
        if (point.argument>.09) {
            const double recovered=std::sqrt(theta*theta+2.)
                *std::abs(scale*residual/std::sqrt(point.variance)+theta*point.shape);
            assert((recovered>GH_BESSEL_PRODUCT_MAX_ARGUMENT)
                ==(point.argument>GH_BESSEL_PRODUCT_MAX_ARGUMENT));
        }
        auto prepared=std::make_unique<generated::GhSecondPrepared<Family,true>>(shape);
        auto first=std::make_unique<generated::GhFirstPrepared<Family,true>>(shape);
        auto source=std::make_unique<gh_shape_cache_t>();
        assert(prepared->valid && first->valid && Family::prepare_hessian(shape,source.get()));
        const auto actual=gh_first::observation_nll<true>(prepared->cache,residual,point.variance);
        const auto gradient=gh_first::observation_nll<true>(first->cache,residual,point.variance);
        gh_obs_derivs_t legacy;
        assert(gh_obs_derivatives(residual,point.variance,source.get(),&legacy));
        vg_shape_value_cache_t value;
        assert(vg_shape_value_cache_init(point.rho,point.shape,&value));
        close(vg_obs_nll_value(residual,point.variance,&value),actual.value,2e-13);
        close(actual.value,legacy.value,2e-13);
        close(gradient.value,actual.value,2e-13);
        for (std::size_t i=0;i<4;++i) {
            close(actual.gradient[i],gradient.gradient[i],2e-12);
            close(actual.gradient[i],legacy.grad[i],2e-12);
            for (std::size_t j=i;j<4;++j)
                close(actual.hessian[ad::packed_index<4>(i,j)],legacy.hess[5*i+j],3e-11);
        }
        std::printf("{\"near_center\":true,\"point\":[%.17g,%.17g,%.17g,%.17g],\"value\":%.17g,\"gradient\":[",
            residual,point.variance,point.rho,point.shape,actual.value);
        for (std::size_t i=0;i<4;++i) std::printf("%s%.17g",i?",":"",actual.gradient[i]);
        std::printf("],\"hessian\":[");
        for (std::size_t i=0;i<10;++i) std::printf("%s%.17g",i?",":"",actual.hessian[i]);
        std::printf("]}\n");
    }
}

void variance_gamma_exact_center_curvature_contract() {
    using namespace vlk::experiment;
    using Family=generated::VarianceGammaFamily;
    const double shape[]{0.,VG_LAMBDA_MIN};
    auto prepared=std::make_unique<generated::GhSecondPrepared<Family,true>>(shape);
    auto source=std::make_unique<gh_shape_cache_t>();
    assert(prepared->valid && Family::prepare_hessian(shape,source.get()));
    const auto actual=gh_first::observation_nll<true>(prepared->cache,0.,1.);
    gh_obs_derivs_t legacy;
    assert(gh_obs_derivatives(0.,1.,source.get(),&legacy));
    // Conditional inverse-Gamma moments at the exact center, rounded at 90 digits.
    // This curvature is genuinely large; a nearby nonzero residual is not the limit.
    constexpr double gradient[]{0.,.5,0.,.28037230554677517};
    constexpr double hessian[]{
        844424930131969.,0.,1462586882166368.8,0.,-.5,0.,0.,
        2533274790395904.,0.,-.48790964408132265,
    };
    close(actual.value,.5954237415153458,2e-13);
    close(legacy.value,actual.value,2e-13);
    for (std::size_t i=0;i<4;++i) {
        // Match the independent full-density contract for gamma derivatives.
        close(actual.gradient[i],gradient[i],2e-9);
        close(legacy.grad[i],gradient[i],2e-9);
        for (std::size_t j=i;j<4;++j) {
            const auto k=ad::packed_index<4>(i,j);
            const double tolerance=i==3 ? 2e-8 : 3e-12;
            close(actual.hessian[k],hessian[k],tolerance);
            close(legacy.hess[5*i+j],hessian[k],tolerance);
        }
    }
}

void first_preparation_reuse_contract() {
    using vlk::experiment::gh_first::FirstPreparation;
    FirstPreparation shared;
    for (double order:{-.5,.3,1.2,4.,5.}) {
        for (int derivative:{0,1}) {
            for (double argument:{.2,1.5,std::nextafter(1.5,2.)}) {
                std::array<double,3> expected,actual;
                assert(gh_log_bessel_k_first(order,argument,derivative,
                    &expected[0],&expected[1],&expected[2]));
                for (int repeat=0;repeat<2;++repeat) {
                    assert(shared.first(order,argument,derivative,&actual[0],&actual[1],&actual[2]));
                    for (unsigned i=0;i<3;++i)
                        assert(FirstPreparation::identical(actual[i],expected[i]));
                }
                assert(shared.count<=shared.entries.size());
            }
        }
    }
    assert(shared.count==shared.entries.size());
    const auto count=shared.count;
    std::array<double,3> invalid{};
    assert(!shared.first(.3,-1.,1,&invalid[0],&invalid[1],&invalid[2]));
    assert(shared.count==count);

    auto first=std::make_unique<gh_bessel_first_plan_t>();
    auto companion=std::make_unique<gh_bessel_first_plan_t>();
    auto independent=std::make_unique<gh_bessel_first_plan_t>();
    assert(shared.plan(.3,1,first.get()));
    assert(shared.retained_plan==first.get());
    for (const auto [order,derivative]:{std::pair{.3,1},std::pair{.3,0},std::pair{1.3,1}}) {
        assert(shared.plan(order,derivative,companion.get()));
        assert(gh_bessel_first_plan_init(order,derivative,independent.get()));
        for (double argument:{.02,1.,12.,200.}) {
            std::array<double,3> actual,expected;
            assert(gh_log_bessel_k_first_planned(companion.get(),argument,&actual[0],&actual[1],&actual[2]));
            assert(gh_log_bessel_k_first_planned(independent.get(),argument,&expected[0],&expected[1],&expected[2]));
            for (unsigned i=0;i<3;++i)
                assert(FirstPreparation::identical(actual[i],expected[i]));
        }
    }
    assert(!shared.plan(NAN,1,companion.get()));
    assert(shared.retained_plan==first.get());
    assert(shared.plan(.3,1,companion.get()));
}

void moment_quadrature_scaling_contract() {
    namespace moment=vlk::experiment::generated::density_moment;
    unsigned baseline_calls=0;
    for (int exponent:{0,-512,512}) {
        const double scale=std::scalbn(1.,exponent);
        unsigned calls=0;
        const auto integrand=[&](double x) {
            ++calls;
            moment::Jet<1> result;
            result.valid=true;
            result.value=scale*std::exp(x);
            result.gradient[0]=scale*std::sin(80.*x);
            result.hessian[0]=scale*std::cos(80.*x);
            return result;
        };
        constexpr double boundaries[]={0.,1.};
        const auto result=moment::integrate<2,1>(integrand,boundaries,2,scale);
        assert(result.valid);
        assert(std::abs(result.value/scale-std::expm1(1.))<1e-12);
        assert(std::abs(result.gradient[0]/scale-(1.-std::cos(80.))/80.)<1e-12);
        assert(std::abs(result.hessian[0]/scale-std::sin(80.)/80.)<1e-12);
        if (exponent==0) baseline_calls=calls;
        else assert(calls==baseline_calls);
        assert((!moment::integrate<2,1>(integrand,boundaries,2,-1.).valid));
        assert((!moment::integrate<2,1>(integrand,boundaries,2,INFINITY).valid));
        const auto positive=[&](double x) {
            auto value=integrand(x);
            value.gradient[0]=value.hessian[0]=value.value;
            return value;
        };
        // An underflowed scaled absolute tolerance leaves the relative gate active.
        const auto relative_only=moment::integrate<2,1>(positive,boundaries,2,0.);
        assert(relative_only.valid);
        assert(std::abs(relative_only.value/scale-std::expm1(1.))<1e-12);
        assert(relative_only.value==relative_only.gradient[0]);
        assert(relative_only.value==relative_only.hessian[0]);
    }
    moment::Jet<1> high,low;
    high.valid=low.valid=true;
    high.value=INFINITY;
    assert(!moment::interval_converged<2>(high,low,1.));
    high.value=0.;high.gradient[0]=INFINITY;
    assert(!moment::interval_converged<2>(high,low,1.));
    high.gradient[0]=0.;high.hessian[0]=INFINITY;
    assert(!moment::interval_converged<2>(high,low,1.));
}

int main() {
    moment_quadrature_scaling_contract();
    first_preparation_reuse_contract();
    small_bessel_product_contract();
    variance_gamma_center_contract();
    variance_gamma_exact_center_curvature_contract();
    packed_density_contracts();
    density_coordinate_contract();
    variance_gamma_preparation_contract();
    expansion_request_contract();
    shared_quadrature_storage_contract();
    fixed_order_small_argument_contract();
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
