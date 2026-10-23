#include "backend.hpp"
#include "models/garch_generated_jet.hpp"
#include "models/gjr_parameter_admission.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace runtime = vlk::experiment::runtime;
namespace admission = runtime::gjr_admission;
namespace coordinate = runtime::gjr_coordinate;
namespace generated = vlk::experiment::generated;
namespace domain = vlk::experiment::domain;
namespace backend = scivolapi::detail::backend;
using scivolapi::Status;

namespace {
std::size_t checks = 0;
void check(bool good, int line)
{
    ++checks;
    if (!good) {
        std::fprintf(stderr, "GJR public admission line %d\n", line);
        std::exit(1);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), __LINE__)
constexpr double tiny = std::numeric_limits<double>::denorm_min();
static_assert(std::bit_cast<std::uint64_t>(tiny) == 1);
static_assert(std::numeric_limits<double>::is_iec559);

Status translated(std::int32_t status)
{
    switch (status) {
    case VLK_JET_OK: return Status::ok;
    case VLK_JET_MATHEMATICAL_DOMAIN_ERROR: return Status::mathematical_domain;
    case VLK_JET_DOMAIN_ERROR: return Status::numerical_range;
    case VLK_JET_UNCERTIFIED_DOMAIN: return Status::unsupported_operation;
    default: return Status::invalid_argument;
    }
}

template<class Density> constexpr unsigned code()
{
    if constexpr (std::is_same_v<Density, generated::NormalDensity>) return 0;
    else if constexpr (std::is_same_v<Density, generated::StudentTDensity>) return 1;
    else if constexpr (std::is_same_v<Density, generated::SkewTDensity>) return 2;
    else if constexpr (std::is_same_v<Density, generated::GedDensity>) return 3;
    else if constexpr (std::is_same_v<Density, generated::GeneralizedHyperbolicDensity>) return 4;
    else if constexpr (std::is_same_v<Density, generated::NormalInverseGaussianDensity>) return 5;
    else if constexpr (std::is_same_v<Density, generated::HyperbolicDensity>) return 6;
    else {
        static_assert(std::is_same_v<Density, generated::VarianceGammaDensity>);
        return 7;
    }
}

template<class Density> constexpr auto symmetric_shape()
{
    if constexpr (code<Density>() == 0) return std::array<double, 0>{};
    else if constexpr (code<Density>() == 1) return std::array{8.0};
    else if constexpr (code<Density>() == 2) return std::array{8.0, 0.0};
    else if constexpr (code<Density>() == 3) return std::array{1.5};
    else if constexpr (code<Density>() == 4) return std::array{0.0, 1.5, 0.7};
    else if constexpr (code<Density>() == 7) return std::array{0.0, 2.3};
    else return std::array{0.0, 1.5};
}

template<std::size_t P, std::size_t Q, class Density>
struct Fixture {
    using PTag = std::integral_constant<std::size_t, P>;
    using QTag = std::integral_constant<std::size_t, Q>;
    static constexpr auto fixed = coordinate::map_layout<Density>(PTag{}, QTag{});
    static constexpr std::size_t K = fixed.parameters;
    using Jet = generated::GarchJet<P, Q, Density, generated::GjrRecursion<P, Q>>;
    static_assert(K == Jet::K);
    static_assert(std::remove_cvref_t<decltype(fixed.p)>::value == P);
    static_assert(std::remove_cvref_t<decltype(fixed.q)>::value == Q);
    static_assert(std::remove_cvref_t<decltype(fixed.shape)>::value == Density::shape_parameters);
    static_assert(std::remove_cvref_t<decltype(fixed.variance_parameters)>::value == Jet::V);
    static_assert(std::remove_cvref_t<decltype(fixed.parameters)>::value == Jet::K);
    static_assert(std::remove_cvref_t<decltype(fixed.alpha_start)>::value == generated::GjrRecursion<P, Q>::alpha_start);
    static_assert(std::remove_cvref_t<decltype(fixed.gamma_start)>::value == generated::GjrRecursion<P, Q>::gamma_start);
    static_assert(std::remove_cvref_t<decltype(fixed.beta_start)>::value == generated::GjrRecursion<P, Q>::beta_start);

    static auto point()
    {
        std::array<double, K> theta{};
        theta[0] = 0.125;
        const auto shape = symmetric_shape<Density>();
        std::copy(shape.begin(), shape.end(), theta.begin() + fixed.variance_parameters);
        return theta;
    }

    static void metadata(const admission::Evidence& evidence, bool is_fixed)
    {
        const auto general = coordinate::map_layout<Density>(std::size_t{P}, std::size_t{Q});
        CHECK(evidence.owner_calls == 1);
        CHECK(evidence.fixed == is_fixed);
#define SCIVOL_CHECK_FIELD(FIELD) \
        CHECK(evidence.FIELD == general.FIELD && general.FIELD == fixed.FIELD)
        SCIVOL_CHECK_FIELD(p);
        SCIVOL_CHECK_FIELD(q);
        SCIVOL_CHECK_FIELD(shape);
        SCIVOL_CHECK_FIELD(variance_parameters);
        SCIVOL_CHECK_FIELD(parameters);
        SCIVOL_CHECK_FIELD(alpha_start);
        SCIVOL_CHECK_FIELD(gamma_start);
        SCIVOL_CHECK_FIELD(beta_start);
#undef SCIVOL_CHECK_FIELD
    }

    static void expect(const std::array<double, K>& theta, std::int32_t expected)
    {
        admission::Evidence evidence;
        evidence.owner_calls = 99;
        const auto status = runtime::garch_parameter_status(
            P, Q, code<Density>(), true, theta.data(), &evidence);
        CHECK(status == expected);
        metadata(evidence, P <= 2 && Q <= 2);
        CHECK(evidence.status == status);
        CHECK(runtime::garch_parameter_status(P, Q, code<Density>(), true, theta.data()) == status);
        CHECK(backend::parameter_status({P, Q, code<Density>(), true}, theta.data()) == translated(status));
        CHECK(Jet::parameter_status(theta.data()) == status);

        admission::Evidence fixed_evidence, general_evidence;
        const auto fixed_status = admission::admit<Density>(PTag{}, QTag{}, theta.data(), &fixed_evidence);
        const auto general_status = admission::admit<Density>(std::size_t{P}, std::size_t{Q}, theta.data(), &general_evidence);
        CHECK(domain::admission_status(fixed_status) == status);
        CHECK(general_status == fixed_status);
        metadata(fixed_evidence, true);
        metadata(general_evidence, false);
        CHECK(fixed_evidence.status == status && general_evidence.status == status);
    }

    static void precheck(const std::array<double, K>& theta, std::int32_t expected)
    {
        admission::Evidence evidence;
        evidence.owner_calls = 99;
        CHECK(runtime::garch_parameter_status(P, Q, code<Density>(), true, theta.data(), &evidence) == expected);
        CHECK(evidence.owner_calls == 0);
        CHECK(backend::parameter_status({P, Q, code<Density>(), true}, theta.data()) == translated(expected));
        CHECK(Jet::parameter_status(theta.data()) == expected);
    }
};

template<std::size_t P, std::size_t Q, class Density>
void boundary_cases()
{
    using F = Fixture<P, Q, Density>;
    constexpr auto layout = F::fixed;
    constexpr std::size_t a = layout.alpha_start + P - 1;
    constexpr std::size_t g = layout.gamma_start + P - 1;
    constexpr std::size_t b = layout.beta_start + Q - 1;
    auto theta = F::point();
    F::expect(theta, VLK_JET_OK);
    theta[a] = -0.0; theta[g] = -0.0; theta[b] = -0.0;
    F::expect(theta, VLK_JET_OK);
    theta[a] = 0.25; theta[g] = 0.5; theta[b] = 0.375;
    F::expect(theta, VLK_JET_OK); // Upper bound > 1; exact half persistence < 1.
    theta[b] = 0.5;
    F::expect(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    theta[b] = 0.0;
    theta[a] = 1.0;
    for (double gamma : {-tiny, 0.0, tiny}) {
        theta[g] = gamma;
        F::expect(theta, gamma < 0 ? VLK_JET_OK : VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    }
    theta[a] = tiny; theta[g] = -tiny;
    F::expect(theta, VLK_JET_OK); // alpha + gamma == 0 is admitted.
    theta[a] = 0.0;
    F::expect(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    theta = F::point(); theta[b] = -tiny;
    F::expect(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    theta = F::point(); theta[a] = -tiny;
    F::expect(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    theta = F::point(); theta[a] = std::numeric_limits<double>::max();
    theta[g] = -std::numeric_limits<double>::max();
    F::expect(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);

    for (double omega : {0.0, -tiny, std::numeric_limits<double>::infinity()}) {
        theta = F::point(); theta[0] = omega;
        F::precheck(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    }
    theta = F::point(); theta[b] = std::numeric_limits<double>::quiet_NaN();
    F::precheck(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);

    if constexpr (!generated::exact_half_moment<Density>) {
        theta = F::point();
        constexpr std::size_t skew = layout.variance_parameters + (code<Density>() == 2 ? 1 : 0);
        for (double value : {0.0, -0.0}) {
            theta[skew] = value;
            theta[a] = 1.0; theta[g] = -tiny;
            F::expect(theta, VLK_JET_OK);
        }
        for (double value : {tiny, -tiny, 0.2}) {
            theta[skew] = value;
            theta[a] = 1.0; theta[g] = -tiny;
            F::expect(theta, VLK_JET_UNCERTIFIED_DOMAIN);
            theta[a] = 0.125; theta[g] = 0.125; theta[b] = 0.25;
            F::expect(theta, VLK_JET_OK);
            theta[a] = 1.0; theta[g] = tiny; theta[b] = 0.0;
            F::expect(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
        }
    }
    if constexpr (code<Density>() >= 4) {
        theta = F::point();
        theta[layout.variance_parameters] = 1.0;
        F::precheck(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
        theta[layout.variance_parameters] = GH_NUMERICAL_RHO_LIMIT;
        F::precheck(theta, VLK_JET_DOMAIN_ERROR);
        theta = F::point();
        theta[layout.variance_parameters + 1] = 0.0;
        F::precheck(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
        theta[layout.variance_parameters + 1] = tiny;
        F::precheck(theta, VLK_JET_DOMAIN_ERROR);
    } else if constexpr (code<Density>() == 1 || code<Density>() == 2) {
        theta = F::point(); theta[layout.variance_parameters] = 2.0;
        F::precheck(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    } else if constexpr (code<Density>() == 3) {
        theta = F::point(); theta[layout.variance_parameters] = 0.0;
        F::precheck(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    }
    if constexpr (P >= 2) {
        theta = F::point();
        theta[layout.alpha_start] = 0.25;
        theta[layout.alpha_start + 1] = 0.5;
        theta[layout.gamma_start] = 0.5;
        theta[layout.gamma_start + 1] = -0.5;
        F::expect(theta, VLK_JET_OK);
        if constexpr (!generated::exact_half_moment<Density>) {
            constexpr std::size_t skew = layout.variance_parameters + (code<Density>() == 2 ? 1 : 0);
            theta[skew] = 0.2; // No half identity or numerical moment is needed.
            F::expect(theta, VLK_JET_OK);
        }
        theta[layout.alpha_start] = 0.5;
        F::expect(theta, VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    }
}

template<class Density>
void density_orders()
{
    boundary_cases<1, 1, Density>();
    boundary_cases<1, 2, Density>();
    boundary_cases<2, 1, Density>();
    boundary_cases<2, 2, Density>();
    boundary_cases<3, 2, Density>();
    boundary_cases<1, 3, Density>();
    boundary_cases<4, 3, Density>();
}

void invalid_entries()
{
    const double theta[]{0.125, 0.125, 0.125, 0.25};
    admission::Evidence evidence;
    const auto invalid = [&](std::size_t p, std::size_t q, unsigned density, const double* point) {
        evidence.owner_calls = 99;
        CHECK(runtime::garch_parameter_status(p, q, density, true, point, &evidence) == VLK_JET_INVALID_ARGUMENT);
        CHECK(evidence.owner_calls == 0);
    };
    invalid(0, 1, 0, theta);
    invalid(1, 0, 0, theta);
    invalid(std::numeric_limits<std::size_t>::max(), 1, 0, theta);
    invalid(1, std::numeric_limits<std::size_t>::max(), 0, theta);
    invalid(1, 1, 8, theta);
    invalid(1, 1, 0, nullptr);
    const double garch[]{0.125, 0.125, 0.25};
    CHECK(runtime::garch_parameter_status(1, 1, 0, false, garch, &evidence) == VLK_JET_OK);
    CHECK(evidence.owner_calls == 0);
}
} // namespace

int main()
{
    const int original = std::fegetround();
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        CHECK(std::fesetround(mode) == 0);
        density_orders<generated::NormalDensity>();
        density_orders<generated::StudentTDensity>();
        density_orders<generated::SkewTDensity>();
        density_orders<generated::GedDensity>();
        density_orders<generated::GeneralizedHyperbolicDensity>();
        density_orders<generated::NormalInverseGaussianDensity>();
        density_orders<generated::HyperbolicDensity>();
        density_orders<generated::VarianceGammaDensity>();
        invalid_entries();
    }
    CHECK(std::fesetround(original) == 0);
    std::printf("GJR public admission: %zu checks\n", checks);
}
