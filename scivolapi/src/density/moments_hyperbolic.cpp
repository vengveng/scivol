#include "moments_dispatch.hpp"
#include "gh.hpp"

namespace scivolapi::detail::moment_detail {
bool hyperbolic(Moment moment, const double* shape, double gamma, double delta, int order, MomentResult& out) noexcept
{
    return evaluate<GhFamilyDensity<Hyperbolic>>(moment, shape, gamma, delta, order, out);
}
}  // namespace scivolapi::detail::moment_detail
