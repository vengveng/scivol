#pragma once

#include <limits>
#include <stdexcept>

#include "distribution_point.hpp"
#include "exponential_moment_methods.hpp"
#include "family_of.hpp"

namespace scivolapi::detail {

// Student-t and Hansen's skewed t share the reciprocal tail eta = 1/nu:
// eta = 0 is the Normal tail and E|z|^r is finite for r eta < 1.
template<class Density>
class TFamily : public DensityFamilyOf<Density> {
public:
    using DensityFamilyOf<Density>::DensityFamilyOf;
    std::vector<std::pair<std::string, double>> derived(const double* shape) const override
    {
        if (!shape || !domain_detail::student_t[0].contains(shape[0]))
            throw std::invalid_argument("density shape is outside its domain");
        return {{"nu", shape[0] == 0.0 ? std::numeric_limits<double>::infinity() : 1.0 / shape[0]}};
    }
    std::size_t tail_index() const noexcept override { return 0; }
};

}  // namespace scivolapi::detail
