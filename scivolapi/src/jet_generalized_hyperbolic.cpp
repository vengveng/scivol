#include "density/gh.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_generalized_hyperbolic(VolatilityKind kind)
{
    return assembler_with<GhFamilyDensity<GeneralizedHyperbolic>>(kind);
}

}  // namespace scivolapi::detail
