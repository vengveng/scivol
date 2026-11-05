// The GeneralizedHyperbolic density.

#include "families.hpp"
#include "gh_family.hpp"

namespace scivolapi::detail {
namespace {

const GhMemberFamily<GeneralizedHyperbolic> family("GeneralizedHyperbolic", domain_detail::gh);

}  // namespace

const DensityFamily& generalized_hyperbolic_family() noexcept { return family; }

}  // namespace scivolapi::detail
