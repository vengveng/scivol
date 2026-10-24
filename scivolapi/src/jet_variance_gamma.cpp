#include "density/gh.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

ThetaResult assemble_variance_gamma(const AssembleInput& in, Work& work, double& value)
{
    return assemble<GhFamilyDensity<VarianceGamma>>(in, work, value);
}

}  // namespace scivolapi::detail
