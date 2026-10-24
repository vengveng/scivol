#include "density/gh.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

ThetaResult assemble_generalized_hyperbolic(const AssembleInput& in, Work& work, double& value)
{
    return assemble<GhFamilyDensity<GeneralizedHyperbolic>>(in, work, value);
}

}  // namespace scivolapi::detail
