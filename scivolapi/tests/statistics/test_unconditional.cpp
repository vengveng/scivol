#ifdef NDEBUG
#undef NDEBUG
#endif
#include <scivolapi/model.hpp>
#include <cassert>
#include <cmath>
#include <vector>

using namespace scivolapi;

int main()
{
    ModelSpec spec;
    spec.mean.constant = true;
    spec.mean.ar = 2;
    spec.mean.ma = 1;
    Model mean(spec);
    std::vector<double> theta{.4, 1.2, -.4, .6, .1, .1, .8};
    assert(std::abs(mean.unconditional_mean(theta)-2.0) < 1e-13);
    theta[1] = .5;
    theta[2] = .6;
    assert(std::isnan(mean.unconditional_mean(theta)));

    spec.mean = {};
    spec.volatility = VolatilityKind::egarch;
    Model egarch(spec);
    // Independent mpmath product, 60 decimal digits, 500 terms.
    const double reference = .8524073275321758962027921957515552547319;
    assert(std::abs(egarch.unconditional_variance(std::vector<double>{-.02, .15, -.08, .9})/reference-1) < 3e-11);
    spec.density = DensityKind::student_t;
    Model student(spec);
    assert(std::isinf(student.unconditional_variance(std::vector<double>{-.02, .15, -.08, .9, .1})));
    assert(std::abs(student.unconditional_variance(std::vector<double>{-.02, .15, -.08, .9, 0.})/reference-1) < 3e-11);

    spec.density = DensityKind::normal;
    spec.volatility = VolatilityKind::aparch;
    Model aparch(spec);
    theta = {.1, .08, .3, .7, 2.};
    const double expected = .1/(1-.08*(1+.3*.3)-.7);
    assert(std::abs(aparch.unconditional_variance(theta)/expected-1) < 1e-13);
    theta[4] = std::nextafter(2., 1.);
    assert(std::isnan(aparch.unconditional_variance(theta)));
}
