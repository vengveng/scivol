#include "density/normal.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_branches_normal(const ModelSpec& spec)
{
    return assembler_coupled<NormalDensity, true, true>(spec.volatility);
}

Assembler assembler_normal(const ModelSpec& spec)
{
    if (spec.volatility == VolatilityKind::garch && spec.p == 1 && spec.q == 0 && spec.mean.width() == 0)
        return assemble_requested<NoMean, GarchVolatility<false, false, 1, 0>, NormalDensity>;
    return assembler_with<NormalDensity>(spec);
}

PathRunner path_runner_normal(VolatilityKind kind)
{
    return path_runner_with<NormalDensity>(kind);
}

}  // namespace scivolapi::detail
