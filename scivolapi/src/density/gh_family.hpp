#pragma once

#include <span>

#include "distribution_point.hpp"
#include "exponential_moment_methods.hpp"
#include "family_of.hpp"
#include "gh.hpp"

namespace scivolapi::detail {

// The normal mean-variance mixtures with a generalized inverse Gaussian mixing
// law (GH, NIG, Hyperbolic, Variance-Gamma): one Member each, sharing the
// moment, exponential-moment and distribution code.
template<class Member>
class GhMemberFamily final : public DensityFamilyOf<GhFamilyDensity<Member>> {
public:
    constexpr GhMemberFamily(const char* name, std::span<const ShapeDomain> shapes,
                             std::span<const FixedShape> fixed = {})
        : DensityFamilyOf<GhFamilyDensity<Member>>(name), shapes_(shapes), fixed_(fixed)
    {
    }
    std::span<const ShapeDomain> shapes() const noexcept override { return shapes_; }
    std::span<const FixedShape> fixed_shapes() const noexcept override { return fixed_; }
    ExpectationStatus exponential_moment(const double* shape, double a, double b, double& value) const noexcept override
    {
        return exponential_detail::gh_moment<Member>(shape, Member::index(shape), a, b, value);
    }
    bool distribution_point(const double* shape, distribution_detail::Point& p) const noexcept override
    {
        return distribution_detail::prepare_gh<Member>(shape, p);
    }

private:
    std::span<const ShapeDomain> shapes_;
    std::span<const FixedShape> fixed_;
};

}  // namespace scivolapi::detail
