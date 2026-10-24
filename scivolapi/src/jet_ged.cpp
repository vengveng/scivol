#include "density/ged.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

ThetaResult assemble_ged(const AssembleInput& in, Work& work, double& value)
{
    return assemble<GedDensity>(in, work, value);
}

}  // namespace scivolapi::detail
