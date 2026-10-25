#include "density/gh.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_hyperbolic(VolatilityKind kind, bool mean)
{
    return assembler_with<GhFamilyDensity<Hyperbolic>>(kind, mean);
}

PathRunner path_runner_hyperbolic(VolatilityKind kind)
{
    return path_runner_with<GhFamilyDensity<Hyperbolic>>(kind);
}

}  // namespace scivolapi::detail
