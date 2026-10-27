#include "density/gh.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_branches_nig(const ModelSpec& spec)
{
    return assembler_coupled<GhFamilyDensity<NormalInverseGaussian>, true, true>(spec.volatility);
}

Assembler assembler_nig(const ModelSpec& spec)
{
    return assembler_with<GhFamilyDensity<NormalInverseGaussian>>(spec);
}

PathRunner path_runner_nig(VolatilityKind kind)
{
    return path_runner_with<GhFamilyDensity<NormalInverseGaussian>>(kind);
}

}  // namespace scivolapi::detail
