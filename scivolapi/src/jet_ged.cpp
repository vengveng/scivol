#include "density/ged.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Kernel kernel_branches_ged(const ModelSpec& spec)
{
    return kernel_runtime<GedDensity, true, true>(spec);
}

Kernel kernel_ged(const ModelSpec& spec)
{
    return kernel_with<GedDensity>(spec);
}

PathRunner path_runner_ged(VolatilityKind kind)
{
    return path_runner_with<GedDensity>(kind);
}

}  // namespace scivolapi::detail
