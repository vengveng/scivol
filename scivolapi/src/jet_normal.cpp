#include "density/normal.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_normal(VolatilityKind kind, bool mean)
{
    return assembler_with<NormalDensity>(kind, mean);
}

}  // namespace scivolapi::detail
