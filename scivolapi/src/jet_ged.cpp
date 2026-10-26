#include "density/ged.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_ged(const ModelSpec& spec)
{
    return assembler_with<GedDensity>(spec);
}

PathRunner path_runner_ged(VolatilityKind kind)
{
    return path_runner_with<GedDensity>(kind);
}

}  // namespace scivolapi::detail
