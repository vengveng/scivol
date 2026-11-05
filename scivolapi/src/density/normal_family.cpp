// The Normal density.

#include "distribution_point.hpp"
#include "exponential_moment_methods.hpp"
#include "families.hpp"
#include "family_of.hpp"
#include "normal.hpp"

namespace scivolapi::detail {
namespace {

class Normal final : public DensityFamilyOf<NormalDensity> {
public:
    using DensityFamilyOf::DensityFamilyOf;
    std::span<const ShapeDomain> shapes() const noexcept override { return {}; }
    ExpectationStatus exponential_moment(const double*, double a, double b, double& value) const noexcept override
    {
        return exponential_detail::normal_moment(a, b, value);
    }
    bool distribution_point(const double*, distribution_detail::Point& p) const noexcept override
    {
        return distribution_detail::prepare_normal(p);
    }
};

const Normal family("Normal");

}  // namespace

const DensityFamily& normal_family() noexcept { return family; }

}  // namespace scivolapi::detail
