// The generalized error distribution in nu.

#include <cmath>
#include <limits>

#include "../model_layout.hpp"
#include "../restriction_builder.hpp"
#include "distribution_point.hpp"
#include "exponential_moment_methods.hpp"
#include "families.hpp"
#include "family_of.hpp"
#include "ged.hpp"

namespace scivolapi::detail {
namespace {

// The Normal at nu = 2; for nu <= 1 the log density's kink at zero is not
// Lipschitz in the residual.
class Ged final : public DensityFamilyOf<GedDensity> {
public:
    using DensityFamilyOf::DensityFamilyOf;
    std::span<const ShapeDomain> shapes() const noexcept override { return domain_detail::ged; }
    bool residual_kinks() const noexcept override { return true; }
    double residual_kink_order(const double* shape) const noexcept override
    {
        return shape[0] <= 1 ? std::numeric_limits<double>::quiet_NaN() : 0.0;
    }
    void restrictions(const Layout& l, RestrictionBuilder& out) const override
    {
        ModelSpec c = out.spec();
        c.density = DensityKind::normal;
        out.add(c, "density.normal", l.shapes, RestrictionType::interior_equality, 2.0);
    }
    ExpectationStatus exponential_moment(const double* shape, double a, double b, double& value) const noexcept override
    {
        return exponential_detail::ged_moment(shape, a, b, value);
    }
    bool distribution_point(const double* shape, distribution_detail::Point& p) const noexcept override
    {
        return distribution_detail::prepare_ged(shape, p);
    }
};

const Ged family("GED");

}  // namespace

const DensityFamily& ged_family() noexcept { return family; }

}  // namespace scivolapi::detail
