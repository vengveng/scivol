#pragma once

#include "family.hpp"
#include "moments.hpp"

namespace scivolapi::detail::moment_detail {

template<class Density>
bool evaluate(Moment moment, const double* shape, double gamma, double delta, int order, MomentResult& out) noexcept
{
    if (moment == Moment::power) {
        Dual2<Density::shapes + 2> d;
        return power_moment<Density>(shape, gamma, delta, order, d) && store(d, order, out);
    }
    Dual2<Density::shapes> d;
    const bool ok = moment == Moment::absolute ? absolute_moment<Density>(shape, order, d)
                                               : negative_second_moment<Density>(shape, order, d);
    return ok && store(d, order, out);
}

}  // namespace scivolapi::detail::moment_detail
