#include "density/normal.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_normal(VolatilityKind kind)
{
    return assembler_with<NormalDensity>(kind);
}

}  // namespace scivolapi::detail
