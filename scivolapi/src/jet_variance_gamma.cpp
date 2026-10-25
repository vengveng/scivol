#include "density/gh.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_variance_gamma(VolatilityKind kind, bool mean)
{
    return assembler_with<GhFamilyDensity<VarianceGamma>>(kind, mean);
}

}  // namespace scivolapi::detail
