#include "bessel.hpp"

#include <cmath>
#include <limits>

namespace scivolapi::detail::bessel {
namespace {

// K0 + log(z/2) I0 and z K1 - z log(z/2) I1 on [0,2].
// The large-argument rational approximates sqrt(2z/pi) exp(z) K1
// at x = 2/z. Its derivatives give K0/K1 without subtracting it from one.
// Coefficients and independent mpmath references: tests/math/nig_reference.py.
constexpr double k1_numerator[] = {
    1.0,
    8.2224154254767075539278733,
    25.400845405930472895185052,
    37.888176147160344205691892,
    29.269580817563390370239412,
    11.693766022418146250225707,
    2.2724848106160729589626316,
    0.18323410444130533149501673,
    0.0039903105624820594245977669,
};
constexpr double k1_denominator[] = {
    1.0,
    8.0349154254767091676155885,
    23.923595638653296885066111,
    33.625082494973253327311611,
    23.571790075638973538881936,
    8.016265527714835493072579,
    1.1855086805316647253608957,
    0.058561127602871559151972873,
    0.00041174958623321763072282874,
};
constexpr double k0_small[] = {
    -0.5772156649015328606065121,
    0.4227843350984671393934879,
    0.230696083774616784848372,
    0.03489215745643890202018948,
    0.00261478761880520915403962,
    0.0001184803936410972550504737,
    0.000003612624103199203586792994,
    7.935096521304209170037892e-8,
    1.316748673038564743790958e-9,
    1.709994072705807843881882e-11,
    1.785934656987074076941178e-13,
    1.533034340320847270995185e-15,
    1.100927095972574452165741e-17,
    6.712740659979047509869663e-20,
    3.518851972639804955703383e-22,
    1.602920285489642457203402e-24,
    6.404178637511446077949925e-27,
};
constexpr double k1_small[] = {
    1.0,
    0.1544313298030657212130242,
    -0.6727843350984671393934879,
    -0.1815751669608556343433591,
    -0.01918218983933056212120585,
    -0.001115359491966528106060293,
    -0.00004142247689271143069583691,
    -0.000001071545914091180868587366,
    -2.045286003593877941287503e-8,
    -3.002048746589187885928091e-10,
    -3.495928729692881920823059e-12,
    -3.309914735250272068041229e-14,
    -2.598641132101128735094877e-16,
    -1.719523282699256524102857e-18,
    -9.721207518823618016463068e-21,
    -4.750281743327666989607358e-23,
    -2.026493760432857908224114e-25,
};
constexpr double i0_small[] = {
    1.0,
    1.0,
    0.25,
    0.02777777777777777777777778,
    0.001736111111111111111111111,
    0.00006944444444444444444444444,
    0.000001929012345679012345679012,
    3.936759889140841521793903e-8,
    6.151187326782564877802973e-10,
    7.594058428126623305929596e-12,
    7.594058428126623305929596e-14,
    6.276081345559192814817848e-16,
    4.358389823304995010290172e-18,
    2.578928889529582846325546e-20,
    1.315780045678358595064054e-22,
    5.847911314126038200284684e-25,
    2.284340357080483671986205e-27,
};
constexpr double zi1_small[] = {
    0.0,
    2.0,
    1.0,
    0.1666666666666666666666667,
    0.01388888888888888888888889,
    0.0006944444444444444444444444,
    0.00002314814814814814814814815,
    0.0000005511463844797178130511464,
    9.841899722852103804484757e-9,
    1.366930517062792195067327e-10,
    1.518811685625324661185919e-12,
    1.380737896023022419259927e-14,
    1.046013557593198802469641e-16,
    6.705215112776915400446419e-19,
    3.684184127899404066179351e-21,
    1.754373394237811460085405e-23,
    7.309889142657547750355855e-26,
};

template<std::size_t N>
double polynomial(const double (&c)[N], double x) noexcept
{
    double sum = c[N - 1];
    for (std::size_t k = N - 1; k > 0; --k) sum = std::fma(sum, x, c[k - 1]);
    return sum;
}

template<std::size_t N>
OrderJet polynomial_jet(const double (&c)[N], double x, unsigned want) noexcept
{
    OrderJet sum{c[N - 1]};
    for (std::size_t k = N - 1; k > 0; --k) {
        if ((want & argument) && !(want & first_only)) sum.h = std::fma(sum.h, x, 2.0 * sum.g);
        if (want & argument) sum.g = std::fma(sum.g, x, sum.f);
        sum.f = std::fma(sum.f, x, c[k - 1]);
    }
    return sum;
}

}  // namespace

bool log_zk_scaled_nig(double z, unsigned want, LogK& out) noexcept
{
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr double log_two = 0.69314718055994530941723212145818;
    constexpr double half_log_pi_half = 0.22579135264472743236309761494744;
    out = {nan, nan, nan, nan, nan, nan};
    if (!(z > 0.0) || !std::isfinite(z) || (want & ~(argument | first_only)) || want == first_only)
        return false;
    LogK result = out;
    const double lz = std::log(z);
    if (z <= 2.0) {
        const double w = (0.25 * z) * z, l = lz - log_two;
        const double product = polynomial(k1_small, w) + l * polynomial(zi1_small, w);
        result.value = z + std::log(product) - 2.0 * lz;
        if (want & argument) {
            const double k0 = polynomial(k0_small, w) - l * polynomial(i0_small, w);
            const double ratio = z * (k0 / product);
            result.z = (1.0 - ratio) - 2.0 / z;
            if (!(want & first_only)) result.zz = (2.0 / z) / z + 1.0 - ratio / z - ratio * ratio;
        }
    } else {
        const double inverse = 1.0 / z, x = 2.0 * inverse;
        const auto numerator = polynomial_jet(k1_numerator, x, want);
        const auto denominator = polynomial_jet(k1_denominator, x, want);
        result.value = half_log_pi_half - 1.5 * lz + std::log(numerator.f / denominator.f);
        if (want & argument) {
            const double p = numerator.g / numerator.f, q = denominator.g / denominator.f;
            const double slope = p - q;
            result.z = (-1.5 - x * slope) * inverse;
            if (!(want & first_only)) {
                const double curvature = numerator.h / numerator.f - p * p - denominator.h / denominator.f + q * q;
                result.zz = ((1.5 + 2.0 * x * slope + x * x * curvature) * inverse) * inverse;
            }
        }
    }
    if (!std::isfinite(result.value) || ((want & argument) && !std::isfinite(result.z))
        || ((want & argument) && !(want & first_only) && !std::isfinite(result.zz))) return false;
    out = result;
    return true;
}

}  // namespace scivolapi::detail::bessel
