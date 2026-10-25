#include "density/ged.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_ged(VolatilityKind kind, bool mean)
{
    return assembler_with<GedDensity>(kind, mean);
}

PathRunner path_runner_ged(VolatilityKind kind)
{
    return path_runner_with<GedDensity>(kind);
}

}  // namespace scivolapi::detail
