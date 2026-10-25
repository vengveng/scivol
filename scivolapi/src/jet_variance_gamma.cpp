#include "density/gh.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_variance_gamma(VolatilityKind kind, bool mean)
{
    return assembler_with<GhFamilyDensity<VarianceGamma>>(kind, mean);
}

PathRunner path_runner_variance_gamma(VolatilityKind kind)
{
    return path_runner_with<GhFamilyDensity<VarianceGamma>>(kind);
}

}  // namespace scivolapi::detail
