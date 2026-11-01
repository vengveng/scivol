#include "density/gh.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Kernel kernel_branches_nig(const ModelSpec& spec)
{
    return kernel_runtime<GhFamilyDensity<NormalInverseGaussian>, true, true>(spec);
}

Kernel kernel_nig(const ModelSpec& spec)
{
    return kernel_with<GhFamilyDensity<NormalInverseGaussian>>(spec);
}

PathRunner path_runner_nig(VolatilityKind kind)
{
    return path_runner_with<GhFamilyDensity<NormalInverseGaussian>>(kind);
}

}  // namespace scivolapi::detail
