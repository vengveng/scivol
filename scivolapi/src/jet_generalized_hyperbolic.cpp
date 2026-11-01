#include "density/gh.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Kernel kernel_branches_generalized_hyperbolic(const ModelSpec& spec)
{
    return kernel_runtime<GhFamilyDensity<GeneralizedHyperbolic>, true, true>(spec);
}

Kernel kernel_generalized_hyperbolic(const ModelSpec& spec)
{
    return kernel_with<GhFamilyDensity<GeneralizedHyperbolic>>(spec);
}

PathRunner path_runner_generalized_hyperbolic(VolatilityKind kind)
{
    return path_runner_with<GhFamilyDensity<GeneralizedHyperbolic>>(kind);
}

}  // namespace scivolapi::detail
