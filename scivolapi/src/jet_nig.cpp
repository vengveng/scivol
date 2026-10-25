#include "density/gh.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_nig(VolatilityKind kind, bool mean)
{
    return assembler_with<GhFamilyDensity<NormalInverseGaussian>>(kind, mean);
}

PathRunner path_runner_nig(VolatilityKind kind)
{
    return path_runner_with<GhFamilyDensity<NormalInverseGaussian>>(kind);
}

}  // namespace scivolapi::detail
