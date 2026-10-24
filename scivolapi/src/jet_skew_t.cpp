#include "density/skew_t.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_skew_t(VolatilityKind kind)
{
    return assembler_with<SkewTDensity>(kind);
}

}  // namespace scivolapi::detail
