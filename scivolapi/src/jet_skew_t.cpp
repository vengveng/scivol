#include "density/skew_t.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

ThetaResult assemble_skew_t(const AssembleInput& in, Work& work, double& value)
{
    return assemble<SkewTDensity>(in, work, value);
}

}  // namespace scivolapi::detail
