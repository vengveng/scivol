#include "moments_dispatch.hpp"
#include "skew_t.hpp"

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
}  // namespace scivolapi::detail
