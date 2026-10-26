#include "density/normal.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_normal(const ModelSpec& spec)
{
    return assembler_with<NormalDensity>(spec);
}

PathRunner path_runner_normal(VolatilityKind kind)
{
    return path_runner_with<NormalDensity>(kind);
}

}  // namespace scivolapi::detail
