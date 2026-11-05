// The standardized Student-t density in eta = 1/nu.

#include "../model_layout.hpp"
#include "../restriction_builder.hpp"
#include "families.hpp"
#include "student_t.hpp"
#include "t_family.hpp"

namespace scivolapi::detail {
namespace {

class StudentT final : public TFamily<StudentTDensity> {
public:
    using TFamily::TFamily;
    std::span<const ShapeDomain> shapes() const noexcept override { return domain_detail::student_t; }
    void restrictions(const Layout& l, RestrictionBuilder& out) const override
    {
        ModelSpec c = out.spec();
        c.density = DensityKind::normal;
        out.add(c, "density.normal", l.shapes, RestrictionType::closed_boundary);
    }
    ExpectationStatus exponential_moment(const double* shape, double a, double b, double& value) const noexcept override
    {
        return shape[0] == 0.0 ? exponential_detail::normal_moment(a, b, value)
                               : exponential_detail::polynomial_moment<StudentTDensity>(shape, a, b, value);
    }
    bool distribution_point(const double* shape, distribution_detail::Point& p) const noexcept override
    {
        return distribution_detail::prepare_student(shape, false, p);
    }
};

const StudentT family("StudentT");

}  // namespace

const DensityFamily& student_t_family() noexcept { return family; }

}  // namespace scivolapi::detail
