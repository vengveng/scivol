#include "density/gh.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_variance_gamma(VolatilityKind kind)
{
    return assembler_with<GhFamilyDensity<VarianceGamma>>(kind);
}

}  // namespace scivolapi::detail
