// The innovation densities' descriptions beyond their likelihood terms: names,
// derived shapes, tails, residual kinks and nested restrictions.

#include "family.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "../model_layout.hpp"
#include "../restriction_builder.hpp"
#include "domain.hpp"

namespace scivolapi::detail {
namespace {

using T = RestrictionType;

class Named : public DensityFamily {
public:
    explicit constexpr Named(const char* name) : name_(name) {}
    const char* component() const noexcept override { return name_; }

private:
    const char* name_;
};

// Student-t and Hansen's skewed t share the reciprocal tail eta = 1/nu:
// eta = 0 is the Normal tail and E|z|^r is finite for r eta < 1.
class TFamily : public Named {
public:
    using Named::Named;
    std::vector<std::pair<std::string, double>> derived(const double* shape) const override
    {
        if (!shape || !domain_detail::student_t[0].contains(shape[0]))
            throw std::invalid_argument("density shape is outside its domain");
        return {{"nu", shape[0] == 0.0 ? std::numeric_limits<double>::infinity() : 1.0 / shape[0]}};
    }
    std::size_t tail_index() const noexcept override { return 0; }
};

class StudentT final : public TFamily {
public:
    using TFamily::TFamily;
    void restrictions(const Layout& l, RestrictionBuilder& out) const override
    {
        ModelSpec c = out.spec();
        c.density = DensityKind::normal;
        out.add(c, "density.normal", l.shapes, T::closed_boundary);
    }
};

class SkewT final : public TFamily {
public:
    using TFamily::TFamily;
    void restrictions(const Layout& l, RestrictionBuilder& out) const override
    {
        ModelSpec c = out.spec();
        c.density = DensityKind::student_t;
        out.add(c, "density.symmetry", l.shapes + 1, T::interior_equality);
        c.density = DensityKind::normal;
        out.add(c, "density.normal", {{l.shapes, T::closed_boundary}, {l.shapes + 1, T::interior_equality}});
    }
};

// GED: the Normal at nu = 2; for nu <= 1 the log density's kink at zero is not
// Lipschitz in the residual.
class Ged final : public Named {
public:
    using Named::Named;
    bool residual_kinks() const noexcept override { return true; }
    double residual_kink_order(const double* shape) const noexcept override
    {
        return shape[0] <= 1 ? std::numeric_limits<double>::quiet_NaN() : 0.0;
    }
    void restrictions(const Layout& l, RestrictionBuilder& out) const override
    {
        ModelSpec c = out.spec();
        c.density = DensityKind::normal;
        out.add(c, "density.normal", l.shapes, T::interior_equality, 2.0);
    }
};

const Named normal("Normal");
const StudentT student_t("StudentT");
const SkewT skew_t("SkewT");
const Ged ged("GED");
const Named generalized_hyperbolic("GeneralizedHyperbolic");
const Named nig("NormalInverseGaussian");
const Named hyperbolic("Hyperbolic");
const Named variance_gamma("VarianceGamma");

}  // namespace

const DensityFamily& density_family(DensityKind kind) noexcept
{
    switch (kind) {
    case DensityKind::normal: return normal;
    case DensityKind::student_t: return student_t;
    case DensityKind::skew_t: return skew_t;
    case DensityKind::ged: return ged;
    case DensityKind::generalized_hyperbolic: return generalized_hyperbolic;
    case DensityKind::nig: return nig;
    case DensityKind::hyperbolic: return hyperbolic;
    case DensityKind::variance_gamma: return variance_gamma;
    }
    return normal;  // unreachable: the switch covers every kind
}

}  // namespace scivolapi::detail
