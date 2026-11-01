#include "density/normal.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Kernel kernel_branches_normal(const ModelSpec& spec)
{
    return kernel_runtime<NormalDensity, true, true>(spec);
}

Kernel kernel_normal(const ModelSpec& spec)
{
    return kernel_with<NormalDensity>(spec);
}

PathRunner path_runner_normal(VolatilityKind kind)
{
    return path_runner_with<NormalDensity>(kind);
}

}  // namespace scivolapi::detail
