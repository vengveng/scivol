#include "density/ged.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_ged(VolatilityKind kind)
{
    return assembler_with<GedDensity>(kind);
}

}  // namespace scivolapi::detail
