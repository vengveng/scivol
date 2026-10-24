#include "density/gh.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_hyperbolic(VolatilityKind kind)
{
    return assembler_with<GhFamilyDensity<Hyperbolic>>(kind);
}

}  // namespace scivolapi::detail
