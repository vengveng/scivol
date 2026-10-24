#include "density/gh.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

ThetaResult assemble_nig(const AssembleInput& in, Work& work, double& value)
{
    return assemble<GhFamilyDensity<NormalInverseGaussian>>(in, work, value);
}

}  // namespace scivolapi::detail
