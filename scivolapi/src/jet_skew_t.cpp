#include "density/skew_t.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_branches_skew_t(const ModelSpec& spec)
{
    return assembler_coupled<SkewTDensity, true, true>(spec.volatility);
}

Assembler assembler_skew_t(const ModelSpec& spec)
{
    return assembler_with<SkewTDensity>(spec);
}

PathRunner path_runner_skew_t(VolatilityKind kind)
{
    return path_runner_with<SkewTDensity>(kind);
}

}  // namespace scivolapi::detail
