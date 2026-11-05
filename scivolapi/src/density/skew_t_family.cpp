// Hansen's (1994) skewed t in eta = 1/nu and lambda.

#include "../model_layout.hpp"
#include "../restriction_builder.hpp"
#include "families.hpp"
#include "skew_t.hpp"
#include "t_family.hpp"

namespace scivolapi::detail {
namespace {

class SkewT final : public TFamily<SkewTDensity> {
public:
    using TFamily::TFamily;
    std::span<const ShapeDomain> shapes() const noexcept override { return domain_detail::skew_t; }
    void restrictions(const Layout& l, RestrictionBuilder& out) const override
    {
        using T = RestrictionType;
        ModelSpec c = out.spec();
        c.density = DensityKind::student_t;
        out.add(c, "density.symmetry", l.shapes + 1, T::interior_equality);
        c.density = DensityKind::normal;
        out.add(c, "density.normal", {{l.shapes, T::closed_boundary}, {l.shapes + 1, T::interior_equality}});
    }
    ExpectationStatus exponential_moment(const double* shape, double a, double b, double& value) const noexcept override
    {
        return exponential_detail::polynomial_moment<SkewTDensity>(shape, a, b, value);
    }
    bool distribution_point(const double* shape, distribution_detail::Point& p) const noexcept override
    {
        return distribution_detail::prepare_student(shape, true, p);
    }
};

const SkewT family("SkewT");

}  // namespace

const DensityFamily& skew_t_family() noexcept { return family; }

}  // namespace scivolapi::detail
