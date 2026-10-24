#include "moments_dispatch.hpp"
#include "gh.hpp"

namespace scivolapi::detail::moment_detail {
bool gh(Moment moment, const double* shape, double gamma, double delta, int order, MomentResult& out) noexcept
{
    return evaluate<GhFamilyDensity<GeneralizedHyperbolic>>(moment, shape, gamma, delta, order, out);
}
}  // namespace scivolapi::detail::moment_detail
