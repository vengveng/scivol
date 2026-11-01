#include "density/gh.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Kernel kernel_branches_hyperbolic(const ModelSpec& spec)
{
    return kernel_runtime<GhFamilyDensity<Hyperbolic>, true, true>(spec);
}

Kernel kernel_hyperbolic(const ModelSpec& spec)
{
    return kernel_with<GhFamilyDensity<Hyperbolic>>(spec);
}

PathRunner path_runner_hyperbolic(VolatilityKind kind)
{
    return path_runner_with<GhFamilyDensity<Hyperbolic>>(kind);
}

}  // namespace scivolapi::detail
