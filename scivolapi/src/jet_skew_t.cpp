#include "density/skew_t.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_skew_t(VolatilityKind kind, bool mean)
{
    return assembler_with<SkewTDensity>(kind, mean);
}

}  // namespace scivolapi::detail
