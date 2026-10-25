#include "density/ged.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_ged(VolatilityKind kind, bool mean)
{
    return assembler_with<GedDensity>(kind, mean);
}

}  // namespace scivolapi::detail
