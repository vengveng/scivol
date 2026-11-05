// The Hyperbolic density.

#include "families.hpp"
#include "gh_family.hpp"

namespace scivolapi::detail {
namespace {

const GhMemberFamily<Hyperbolic> family("Hyperbolic", domain_detail::fixed_gh, domain_detail::hyperbolic_fixed);

}  // namespace

const DensityFamily& hyperbolic_family() noexcept { return family; }

}  // namespace scivolapi::detail
