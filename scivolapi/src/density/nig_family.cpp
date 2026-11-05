// The NormalInverseGaussian density.

#include "families.hpp"
#include "gh_family.hpp"

namespace scivolapi::detail {
namespace {

const GhMemberFamily<NormalInverseGaussian> family("NormalInverseGaussian", domain_detail::fixed_gh, domain_detail::nig_fixed);

}  // namespace

const DensityFamily& nig_family() noexcept { return family; }

}  // namespace scivolapi::detail
