// SCIVOL-NATIVE: TEST_ONLY
extern "C" {
#include "math/gh_math.h"
}

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

namespace {

struct Reference {
    double x;
    std::array<double, 3> jet; // log K1, argument slope, argument curvature.
};

// MP80 adjacent-order K derivatives at exact binary64 inputs, rounded once.
// Provenance: transition_phase1_20261014/bessel_curvature/scalar_results.json.
constexpr Reference references[] {
    {0x1.47ae147ae147bp-7, {4.6049090930892688789, -100.04722477574570405, 9996.2752922459858382}},
    {0x1.999999999999ap-5, {2.9912057606285360478, -20.156418130125988597, 397.84717076604800923}},
    {0x1.9999999999999p-4, {2.287861712107167822, -10.246306804975629845, 98.476264908066426074}},
    {0x1.999999999999ap-4, {2.287861712107167822, -10.246306804975628069, 98.476264908066397652}},
    {0x1.999999999999bp-4, {2.2878617121071673779, -10.246306804975626292, 98.47626490806636923}},
    {0x1.97a967f7524b4p-2, {0.78702845445237046285, -3.021102424884476445, 5.7711797724270734733}},
    {0x1.97a967f7524b5p-2, {0.7870284544523702408, -3.021102424884476001, 5.7711797724270716969}},
    {0x1.97a967f7524b6p-2, {0.78702845445237012978, -3.0211024248844755569, 5.7711797724270699206}},
    {0x1.95bb8f6d46054p+0, {-1.4032035432618605864, -1.4105442439912174724, 0.29846535675726199033}},
    {0x1.95bb8f6d46055p+0, {-1.4032035432618608084, -1.4105442439912174724, 0.29846535675726187931}},
    {0x1.95bb8f6d46056p+0, {-1.4032035432618612525, -1.4105442439912172503, 0.29846535675726182379}},
    {0x1.0cccccccccccdp+1, {
        -2.09763474667773632311866564188599340475,
        -1.29726328500700529747727198629496620486,
         0.161609760419253055424669844331390336010}},
    {0x1.93d00d234899bp+2, {-6.9495446910841760513, -1.0874183734394338163, 0.014984343163561343951}},
    {0x1.93d00d234899cp+2, {-6.9495446910841769395, -1.0874183734394338163, 0.014984343163561338746}},
    {0x1.93d00d234899dp+2, {-6.9495446910841787158, -1.0874183734394338163, 0.014984343163561335277}},
    {0x1.9000000000000p+4, {-26.368936564922208277, -1.0205771804610790365, 0.00084530594055738173323}},
    {0x1.91e6de449ff78p+4, {-26.490240820268898148, -1.0204771911901018289, 0.00083712417312863042669}},
    {0x1.91e6de449ff79p+4, {-26.4902408202689017, -1.0204771911901018289, 0.00083712417312863020985}},
    {0x1.91e6de449ff7ap+4, {-26.490240820268905253, -1.0204771911901018289, 0.00083712417312862988459}},
    {0x1.8ffffffffffffp+6, {-102.07306232835991011, -1.0050371298393054875, 5.0738942764103046011e-5}},
    {0x1.9000000000000p+6, {-102.07306232835992432, -1.0050371298393054875, 5.0738942764103032459e-5}},
    {0x1.9000000000001p+6, {-102.07306232835993853, -1.0050371298393054875, 5.073894276410301213e-5}},
    {0x1.fffffffffffffp+6, {-130.19730545845095548, -1.0039289611790573886, 3.0871071400287662917e-5}},
    {0x1.0000000000000p+7, {-130.19730545845098391, -1.0039289611790573886, 3.087107140028765614e-5}},
    {0x1.0000000000001p+7, {-130.19730545845101233, -1.0039289611790573886, 3.0871071400287642588e-5}},
    {0x1.f400000000000p+9, {-1003.2277114741824562, -1.00050037462549124, 5.0074887696454233725e-7}},
    {0x1.e848000000000p+19, {-1000006.6819635513239, -1.0000005000003751032, 5.0000074999887496451e-13}},
    {0x1.d1a94a2000000p+39, {-1000000000013.5897217, -1.0000000000005000445, 5.0000000000074998235e-25}},
};


struct SingularReference {
    double x;
    std::array<double, 3> jet;
    bool first_representable, second_representable;
};

struct DensityReference {
    const char* name;
    std::array<double, 5> point; // residual, variance, rho, zeta, lambda.
    int estimate_lambda;
    double value;
    std::array<double, 5> gradient;
    std::array<double, 25> hessian;
};

// Independent MP80/120 pins; the two precisions round to identical binary64 values.
// Provenance: bessel_curvature/regression_pins_mp80_120.json.
constexpr Reference transition_references[] {
    {0x1.fffffffffffffp-2, {0.50467139730465127023, -2.5580754184765854831, 3.5724009903390143705}},
    {0x1.0000000000000p-1, {0.50467139730465115921, -2.5580754184765854831, 3.5724009903390134824}},
    {0x1.0000000000001p-1, {0.50467139730465093717, -2.558075418476585039, 3.572400990339011706}},
};
constexpr SingularReference singular_references[] {
    {0x0.0000000000001p-1022, {744.44007192138121809, 0.0, 0.0}, false, false},
    {0x0.4000000000000p-1022, {709.7827128933839731, 0.0, 0.0}, false, false},
    {0x0.8000000000000p-1022, {709.08956571282408277, -8.9884656743115795386e+307, 0.0}, true, false},
    {0x1.0000000000000p-1022, {708.39641853226407875, -4.4942328371557897693e+307, 0.0}, true, false},
    {0x1.fffffffffffffp-513, {354.89135644669198655, -1.3407807929942600077e+154, 0.0}, true, false},
    {0x1.0000000000000p-512, {354.89135644669198655, -1.34078079299425971e+154, 0.0}, true, false},
    {0x1.0000000000001p-512, {354.89135644669198655, -1.3407807929942594122e+154, 1.7976931348623151094e+308}, true, true},
    {0x1.0000000000000p-511, {354.19820926613203937, -6.7039039649712985498e+153, 4.4942328371557897693e+307}, true, true},
    {0x1.0000000000000p-500, {346.57359027997267731, -3.27339060789614187e+150, 1.0715086071862673209e+301}, true, true},
};
constexpr DensityReference density_references[] {
    {"gh-lambda-one", {0x1.7ae147ae147aep-2, 0x1.4cccccccccccdp+0, 0x1.70a3d70a3d70ap-3, 0x1.999999999999ap-4, 0x1.0000000000000p+0}, 1, 1.153574434889293121,
        {1.0652289531229957031, 0.23302511051711213841, 0.99918220668620538749, -0.12363676828763113524, -0.064142209870436728281},
        {0.027184664863887916808, -0.41357203043178242252, -0.68310533680699780934, -0.076407866832729862128, -0.52398232792150301051,
         -0.41357203043178242252, -0.12039560375940953907, 0.097211144084072761973, 0.010873427203119248596, 0.07456671589652157095,
         -0.68310533680699780934, 0.097211144084072761973, -2.4626017633166359388, -0.0033111274601941827547, -0.3607250330281167594,
         -0.076407866832729862128, 0.010873427203119248596, -0.0033111274601941827547, -0.35036085928533045042, 0.40265104712766541795,
         -0.52398232792150301051, 0.07456671589652157095, -0.3607250330281167594, 0.40265104712766541795, 0.10628789574782190885}},
    {"gh-general", {0x1.7ae147ae147aep-2, 0x1.4cccccccccccdp+0, 0x1.70a3d70a3d70ap-3, 0x1.999999999999ap-4, 0x1.6666666666666p-2}, 1, 1.2121627791656481854,
        {1.6287174022867985101, 0.1528363696745709821, 1.2658890101263151617, -0.94388586556067333166, -0.076477422358812330305},
        {-1.3489960562217317985, -0.43445725441721450011, -1.7801967973420023128, -0.73361063355343392178, -1.3687795439747589832,
         -0.43445725441721450011, -0.055739828928758679283, 0.25333569808328493256, 0.10439843631337329677, 0.19478785818102339888,
         -1.7801967973420023128, 0.25333569808328493256, -2.7883941802556861944, 0.14860054593831403813, -0.42157384501968137025,
         -0.73361063355343392178, 0.10439843631337329677, 0.14860054593831403813, 6.358931793109954711, 2.4903360600784449908,
         -1.3687795439747589832, 0.19478785818102339888, -0.42157384501968137025, 2.4903360600784449908, -0.32004026989949702076}},
    {"hyperbolic", {0x1.7ae147ae147aep-2, 0x1.4cccccccccccdp+0, 0x1.70a3d70a3d70ap-3, 0x1.999999999999ap-4, 0x1.0000000000000p+0}, 0, 1.153574434889293121,
        {1.0652289531229957031, 0.23302511051711213841, 0.99918220668620538749, -0.12363676828763113524, 0},
        {0.027184664863887916808, -0.41357203043178242252, -0.68310533680699780934, -0.076407866832729862128, 0,
         -0.41357203043178242252, -0.12039560375940953907, 0.097211144084072761973, 0.010873427203119248596, 0,
         -0.68310533680699780934, 0.097211144084072761973, -2.4626017633166359388, -0.0033111274601941827547, 0,
         -0.076407866832729862128, 0.010873427203119248596, -0.0033111274601941827547, -0.35036085928533045042, 0,
         0, 0, 0, 0, 0}},
    {"hyperbolic-high-rho", {0x1.7ae147ae147aep-2, 0x1.4cccccccccccdp+0, 0x1.999999999999ap-1, 0x1.999999999999ap-4, 0x1.0000000000000p+0}, 0, 1.4427686040362346365,
        {0.88282385024213705371, 0.25898275977323431674, 0.11651793962972210872, -0.050645125471211208112, 0},
        {0.0023494981097761558249, -0.33988198636259775443, -0.062525830053438341549, 0.0026001893334030353605, 0,
         -0.33988198636259775443, -0.15084968638165674459, 0.0088979065845277635627, -0.00037002694359966267768, 0,
         -0.062525830053438341549, 0.0088979065845277635627, -0.67961721029572541575, 0.13641754466661645817, 0,
         0.0026001893334030353605, -0.00037002694359966267768, 0.13641754466661645817, -0.29756702935084139439, 0,
         0, 0, 0, 0, 0}},
};
static_assert(GH_DERIV_DIM == 5 && GH_DERIV_E == 0 && GH_DERIV_H == 1
    && GH_DERIV_RHO == 2 && GH_DERIV_ZETA == 3 && GH_DERIV_LAMBDA == 4,
    "Update the independent density parameter layout before changing GH derivatives.");

static_assert(GH_BESSEL_FIXED_ONE_MIN_ARGUMENT == 100.0,
              "Update the independent fixtures when the fixed K1 branch changes.");

constexpr double sentinel = -0x1.23456789abcdep+40;
using Partials = std::array<double, 6>;

struct Plans {
    gh_bessel_value_plan_t value;
    gh_bessel_first_plan_t first;
    gh_bessel_second_plan_t second;
};

void require(bool condition, const char* reason)
{
    if (!condition) {
        std::fprintf(stderr, "K1 curvature: %s\n", reason);
        std::exit(1);
    }
}

void check_scalar(double actual, const Reference& reference, unsigned derivative)
{
    const double expected = reference.jet[derivative];
    double tolerance = 2e-15 * std::max(1.0, std::abs(expected));
    if (derivative && reference.x >= 0.5 && reference.x < 100.0) {
        // Established scalar approximation budget, not a consumer-H guarantee.
        tolerance = 2e-11 * std::max(1.0, std::abs(expected));
        if (reference.x == 0x1.0cccccccccccdp+1)
            tolerance = 8 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, std::abs(expected));
    }
    if (derivative == 2 && reference.x >= 100.0)
        tolerance = 8e-14 * std::abs(expected); // No absolute floor hides tail cancellation.
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::fprintf(stderr, "K1 x=%a derivative=%u actual=%.17g MP=%.17g tolerance=%.3g\n",
                     reference.x, derivative, actual, expected, tolerance);
        std::exit(1);
    }
    if (derivative == 2) require(actual > 0.0, "log K1 curvature lost positivity");
}

// Below 0.5, the original 2e-15 scaled small-series gate now also covers the extension.
void check_projection(const Partials& sparse, const Partials& full, unsigned count)
{
    for (unsigned i = 0; i < count; ++i)
        require(std::bit_cast<std::uint64_t>(sparse[i]) == std::bit_cast<std::uint64_t>(full[i]),
                "planned/unplanned request projection changed");
    for (unsigned i = count; i < sparse.size(); ++i)
        require(sparse[i] == sentinel, "a lower-order intention wrote higher-order outputs");
}

template<unsigned Mask>
void check_tail_intention(const Reference& reference, const gh_bessel_value_plan_t& plan)
{
    static_assert(Mask > 0 && Mask < 8);
    std::array<double, 3> full {};
    require(gh_bessel_fixed_one_evaluate(7, plan.expansion, reference.x,
                &full[0], &full[1], &full[2]), "shared tail full request failed");
    for (bool null_unrequested : {false, true}) {
        std::array<double, 3> sparse {sentinel, sentinel, sentinel};
        const auto output = [&](unsigned i) {
            return null_unrequested && !(Mask & (1u << i)) ? nullptr : &sparse[i];
        };
        require(gh_bessel_fixed_one_evaluate(Mask, plan.expansion, reference.x,
                    output(0), output(1), output(2)), "shared tail sparse request failed");
        for (unsigned i = 0; i < sparse.size(); ++i) {
            if (Mask & (1u << i)) {
                check_scalar(sparse[i], reference, i);
                require(std::bit_cast<std::uint64_t>(sparse[i]) == std::bit_cast<std::uint64_t>(full[i]),
                        "shared tail mask is not a full-request projection");
            } else {
                require(sparse[i] == sentinel, "shared tail wrote an unrequested output");
            }
        }
    }
}

void check_owner_intentions(const Reference& reference, double order, const Plans& plans)
{
    Partials full {};
    require(gh_log_bessel_k_second_planned(&plans.second, reference.x,
                &full[0], &full[1], &full[2], &full[3], &full[4], &full[5]),
            "shared GH second-order intention failed");
    require(full[1] == 0.0 && full[3] == 0.0 && full[4] == 0.0,
            "fixed order acquired order sensitivities");
    check_scalar(full[0], reference, 0);
    check_scalar(full[2], reference, 1);
    check_scalar(full[5], reference, 2);

    // This interface exposes F, F|G and F|G|H, not pure G/H argument requests.
    Partials value, first, unplanned;
    value.fill(sentinel);
    first.fill(sentinel);
    unplanned.fill(sentinel);
    require(gh_log_bessel_k_value_planned(&plans.value, reference.x, &value[0]),
            "shared GH value intention failed");
    require(gh_log_bessel_k_first_planned(&plans.first, reference.x,
                &first[0], &first[1], &first[2]), "shared GH first-order intention failed");
    require(gh_log_bessel_k_first(order, reference.x, 0,
                &unplanned[0], &unplanned[1], &unplanned[2]), "shared GH unplanned first intention failed");
    check_projection(value, full, 1);
    check_projection(first, full, 3);
    check_projection(unplanned, full, 3);

    if (reference.x >= GH_BESSEL_FIXED_ONE_MIN_ARGUMENT) {
        check_tail_intention<1>(reference, plans.value);
        check_tail_intention<2>(reference, plans.value);
        check_tail_intention<3>(reference, plans.value);
        check_tail_intention<4>(reference, plans.value);
        check_tail_intention<5>(reference, plans.value);
        check_tail_intention<6>(reference, plans.value);
        check_tail_intention<7>(reference, plans.value);
    }
}


void check_singular_intentions(const SingularReference& reference, double order, const Plans& plans)
{
    const Reference expected {reference.x, reference.jet};
    Partials value, first, unplanned, second;
    value.fill(sentinel);
    first.fill(sentinel);
    unplanned.fill(sentinel);
    second.fill(sentinel);
    require(gh_log_bessel_k_value_planned(&plans.value, reference.x, &value[0]),
            "tiny positive argument lost its finite value intention");
    check_scalar(value[0], expected, 0);
    const bool first_valid = gh_log_bessel_k_first_planned(&plans.first, reference.x,
        &first[0], &first[1], &first[2]) != 0;
    const bool unplanned_valid = gh_log_bessel_k_first(order, reference.x, 0,
        &unplanned[0], &unplanned[1], &unplanned[2]) != 0;
    const bool second_valid = gh_log_bessel_k_second_planned(&plans.second, reference.x,
        &second[0], &second[1], &second[2], &second[3], &second[4], &second[5]) != 0;
    require(first_valid == reference.first_representable
                && unplanned_valid == reference.first_representable
                && second_valid == reference.second_representable,
            "tiny-argument request-specific representability changed");
    if (first_valid) {
        require(first[1] == 0.0 && unplanned[1] == 0.0, "fixed tiny order acquired a score");
        check_scalar(first[0], expected, 0);
        check_scalar(first[2], expected, 1);
        check_projection(value, first, 1);
        check_projection(unplanned, first, 3);
    }
    if (second_valid) {
        require(second[1] == 0.0 && second[3] == 0.0 && second[4] == 0.0,
                "fixed tiny order acquired curvature");
        check_scalar(second[0], expected, 0);
        check_scalar(second[2], expected, 1);
        check_scalar(second[5], expected, 2);
        check_projection(first, second, 3);
    }
    // Failed requests do not promise transactional output buffers.
}

void check_density_scalar(double actual, double expected, double tolerance,
                          const DensityReference& reference, const char* intention, unsigned index)
{
    const double bound = tolerance * std::max(1.0, std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual - expected) > bound) {
        std::fprintf(stderr, "%s %s[%u] actual=%.17g MP=%.17g tolerance=%.3g\n",
                     reference.name, intention, index, actual, expected, bound);
        std::exit(1);
    }
}

void check_complete_low_zeta_consumer(const DensityReference& reference)
{
    struct DensityPlans {
        gh_shape_value_cache_t value;
        gh_shape_cache_t first, second;
        gh_bessel_value_plan_t value_plan;
        gh_bessel_second_plan_t second_plan;
    };
    auto plans = std::make_unique<DensityPlans>();
    const auto& p = reference.point;
    require(gh_shape_value_cache_init(p[2], p[3], p[4], &plans->value), "density value preparation failed");
    require(gh_shape_cache_init_gradient(p[2], p[3], p[4], reference.estimate_lambda, &plans->first),
            "density first-order preparation failed");
    require(gh_shape_cache_init(p[2], p[3], p[4], reference.estimate_lambda, &plans->second),
            "density second-order preparation failed");
    require(gh_bessel_value_plan_init(p[4] - 0.5, &plans->value_plan), "density value plan failed");
    require(gh_bessel_second_plan_init(p[4] - 0.5, reference.estimate_lambda, &plans->second_plan),
            "density curvature plan failed");
    check_density_scalar(gh_obs_nll_value(p[0], p[1], &plans->value), reference.value,
                         2e-13, reference, "F", 0);
    check_density_scalar(gh_obs_nll_value_planned(p[0], p[1], &plans->value, &plans->value_plan),
                         reference.value, 2e-13, reference, "planned F", 0);
    gh_obs_gradient_t first {};
    gh_obs_derivs_t second {}, planned {};
    require(gh_obs_gradient_prepared(p[0], p[1], &plans->first, &first), "density F|G failed");
    require(gh_obs_derivatives(p[0], p[1], &plans->second, &second), "density F|G|H failed");
    require(gh_obs_derivatives_planned(p[0], p[1], &plans->second, &plans->second_plan, &planned),
            "density planned F|G|H failed");
    check_density_scalar(first.value, reference.value, 2e-13, reference, "F|G value", 0);
    check_density_scalar(second.value, reference.value, 2e-13, reference, "F|G|H value", 0);
    check_density_scalar(planned.value, reference.value, 2e-13, reference, "planned F|G|H value", 0);
    for (unsigned i = 0; i < GH_DERIV_DIM; ++i) {
        check_density_scalar(first.grad[i], reference.gradient[i], 2e-11, reference, "G", i);
        check_density_scalar(second.grad[i], reference.gradient[i], 2e-11, reference, "F|G|H G", i);
        check_density_scalar(planned.grad[i], reference.gradient[i], 2e-11, reference, "planned G", i);
    }
    for (unsigned i = 0; i < GH_DERIV_DIM * GH_DERIV_DIM; ++i) {
        check_density_scalar(second.hess[i], reference.hessian[i], 2e-11, reference, "H", i);
        check_density_scalar(planned.hess[i], reference.hessian[i], 2e-11, reference, "planned H", i);
    }
    if (!reference.estimate_lambda) {
        require(first.grad[GH_DERIV_LAMBDA] == 0.0 && second.grad[GH_DERIV_LAMBDA] == 0.0
                    && planned.grad[GH_DERIV_LAMBDA] == 0.0,
                "fixed Hyperbolic lambda acquired a score");
        for (unsigned i = 0; i < GH_DERIV_DIM; ++i) {
            const unsigned row = GH_DERIV_LAMBDA * GH_DERIV_DIM + i;
            const unsigned column = i * GH_DERIV_DIM + GH_DERIV_LAMBDA;
            require(second.hess[row] == 0.0 && second.hess[column] == 0.0
                        && planned.hess[row] == 0.0 && planned.hess[column] == 0.0,
                    "fixed Hyperbolic lambda acquired a cross derivative");
        }
    }
}

} // namespace

int main()
{
    // Caller-owned plans stay off the bounded kernel/test stack.
    auto plans = std::make_unique<Plans>();
    for (double order : {-1.0, 1.0}) {
        require(gh_bessel_value_plan_init(order, &plans->value), "value plan preparation failed");
        require(gh_bessel_first_plan_init(order, 0, &plans->first), "first plan preparation failed");
        require(gh_bessel_second_plan_init(order, 0, &plans->second), "second plan preparation failed");
        require(plans->value.fixed_kind == 2 && plans->first.value.fixed_kind == 2
                    && plans->second.first.value.fixed_kind == 2,
                "fixed K1 dispatch did not reach the shared owner");
        for (const auto& reference : references)
            check_owner_intentions(reference, order, *plans);
        for (const auto& reference : transition_references)
            check_owner_intentions(reference, order, *plans);
        for (const auto& reference : singular_references)
            check_singular_intentions(reference, order, *plans);
    }
    for (const auto& reference : density_references)
        check_complete_low_zeta_consumer(reference);
    std::puts("K1 curvature: MP pins, branch neighbors, shared owner intentions and tail masks passed");
}
