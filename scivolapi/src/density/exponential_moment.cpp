#include "exponential_moment.hpp"

#include <cmath>

#include "family.hpp"

namespace scivolapi::detail {

ExpectationStatus exponential_moment(DensityKind kind, const double* shape,
                                     double a, double b, double& value) noexcept
{
    using Status = ExpectationStatus;
    if (!std::isfinite(a) || !std::isfinite(b)) return Status::numerical_failure;
    const DensityFamily* family = find_density_family(kind);
    if (!family) return Status::numerical_failure;
    const auto domains = family->shapes();
    if (!domains.empty() && !shape) return Status::numerical_failure;
    for (std::size_t i = 0; i < domains.size(); ++i)
        if (!domains[i].contains(shape[i]))
            return Status::numerical_failure;
    if (a == 0.0 && b == 0.0) { value = 1.0; return Status::ok; }
    return family->exponential_moment(shape, a, b, value);
}

}  // namespace scivolapi::detail
