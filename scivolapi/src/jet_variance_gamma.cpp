#include "density/gh.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Kernel kernel_branches_variance_gamma(const ModelSpec& spec)
{
    return kernel_runtime<GhFamilyDensity<VarianceGamma>, true, true>(spec);
}

Kernel kernel_variance_gamma(const ModelSpec& spec)
{
    return kernel_with<GhFamilyDensity<VarianceGamma>>(spec);
}

PathRunner path_runner_variance_gamma(VolatilityKind kind)
{
    return path_runner_with<GhFamilyDensity<VarianceGamma>>(kind);
}

}  // namespace scivolapi::detail
