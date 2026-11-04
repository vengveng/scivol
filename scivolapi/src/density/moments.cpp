#include "moments_dispatch.hpp"
#include "family.hpp"
#include "skew_t.hpp"

#include <cmath>
#include <limits>

namespace scivolapi::detail {
namespace {
bool dispatch(DensityKind kind, moment_detail::Moment moment, const double* shape, double gamma, double delta,
              int order, MomentResult& out) noexcept
{
    using namespace moment_detail;
    switch (kind) {
    case DensityKind::normal: return evaluate<NormalDensity>(moment, shape, gamma, delta, order, out);
    case DensityKind::student_t: return evaluate<StudentTDensity>(moment, shape, gamma, delta, order, out);
    case DensityKind::skew_t: return evaluate<SkewTDensity>(moment, shape, gamma, delta, order, out);
    case DensityKind::ged: return evaluate<GedDensity>(moment, shape, gamma, delta, order, out);
    case DensityKind::generalized_hyperbolic: return gh(moment, shape, gamma, delta, order, out);
    case DensityKind::nig: return nig(moment, shape, gamma, delta, order, out);
    case DensityKind::hyperbolic: return hyperbolic(moment, shape, gamma, delta, order, out);
    case DensityKind::variance_gamma: return vg(moment, shape, gamma, delta, order, out);
    }
    return false;
}
}  // namespace

bool negative_second_moment(DensityKind kind, const double* shape, int order, MomentResult& out) noexcept
{
    return dispatch(kind, moment_detail::Moment::negative_second, shape, 0.0, 2.0, order, out);
}

bool absolute_moment(DensityKind kind, const double* shape, int order, MomentResult& out) noexcept
{
    return dispatch(kind, moment_detail::Moment::absolute, shape, 0.0, 1.0, order, out);
}

bool power_moment(DensityKind kind, const double* shape, double gamma, double delta, int order, MomentResult& out) noexcept
{
    return dispatch(kind, moment_detail::Moment::power, shape, gamma, delta, order, out);
}

// With M(g) = E[(|z| - g z)^k] = (1 - g)^k P + (1 + g)^k N, P = E[z^k 1(z > 0)]
// and N = E[|z|^k 1(z < 0)], the pair g = -1/2, 1/2 gives P and N, and
// E[z^k] = P + (-1)^k N.
bool standardized_moment(DensityKind kind, const double* shape, unsigned k, double& value) noexcept
{
    if (k == 0) return false;
    const DensityFamily& family = density_family(kind);
    if (family.tail_index() != DensityFamily::no_tail_index && double(k) * shape[family.tail_index()] >= 1.0) {
        value = k % 2 == 0 ? std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
        return true;
    }
    if (k == 1) { value = 0.0; return true; }
    if (k == 2) { value = 1.0; return true; }
    MomentResult minus, plus;
    if (!power_moment(kind, shape, -0.5, double(k), 0, minus) || !power_moment(kind, shape, 0.5, double(k), 0, plus))
        return false;
    const double large = std::pow(1.5, k), small = std::pow(0.5, k), det = large * large - small * small;
    const double positive = (large * minus.value - small * plus.value) / det;
    const double negative = (large * plus.value - small * minus.value) / det;
    value = positive + (k % 2 == 0 ? negative : -negative);
    return std::isfinite(value);
}
}  // namespace scivolapi::detail
