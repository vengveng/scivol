#include "density/gh.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_generalized_hyperbolic(const ModelSpec& spec)
{
    return assembler_with<GhFamilyDensity<GeneralizedHyperbolic>>(spec);
}

PathRunner path_runner_generalized_hyperbolic(VolatilityKind kind)
{
    return path_runner_with<GhFamilyDensity<GeneralizedHyperbolic>>(kind);
}

}  // namespace scivolapi::detail
