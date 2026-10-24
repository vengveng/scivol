#include "density/gh.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_nig(VolatilityKind kind)
{
    return assembler_with<GhFamilyDensity<NormalInverseGaussian>>(kind);
}

}  // namespace scivolapi::detail
