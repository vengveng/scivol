#include "density/gh.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_nig(VolatilityKind kind, bool mean)
{
    return assembler_with<GhFamilyDensity<NormalInverseGaussian>>(kind, mean);
}

}  // namespace scivolapi::detail
