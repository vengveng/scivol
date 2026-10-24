#include "density/normal.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

ThetaResult assemble_normal(const AssembleInput& in, Work& work, double& value)
{
    return assemble<NormalDensity>(in, work, value);
}

}  // namespace scivolapi::detail
