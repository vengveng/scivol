// The VarianceGamma density.

#include "families.hpp"
#include "gh_family.hpp"

namespace scivolapi::detail {
namespace {

const GhMemberFamily<VarianceGamma> family("VarianceGamma", domain_detail::vg);

}  // namespace

const DensityFamily& variance_gamma_family() noexcept { return family; }

}  // namespace scivolapi::detail
