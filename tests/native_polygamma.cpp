#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math_and_helpers.h"

#ifdef __cplusplus
#include "fixed_differential.hpp"
namespace ad = vlk::experiment::ad;

template<int Order>
void loggamma_jet(double x, bool chain)
{
    using Number = ad::Differential<Order, 1>;
    const Number coordinate = Number::variable(chain ? log(x) : x, 0);
    const Number argument = chain ? ad::exp(coordinate) : coordinate;
    const Number value = ad::lgamma(argument);
    printf("%.17g %.17g %.17g", argument.value, value.value, value.gradient[0]);
    if constexpr (Order == 2) printf(" %.17g", value.hessian[0]);
    printf("\n");
}

void ged_jet(double shape, double residual, double variance)
{
    using Number = ad::Differential<2, 1>;
    const Number nu = Number::variable(shape, 0);
    const Number inverse = Number::constant(1.0) / nu;
    const Number scale = Number::constant(0.5) * (
        ad::lgamma(inverse) - ad::lgamma(Number::constant(3.0) * inverse));
    const Number constant = Number::constant(log(2.0)) + scale
        + ad::lgamma(inverse) - ad::log(nu);
    const Number local = constant + Number::constant(0.5 * log(variance))
        + ad::exp(nu * (Number::constant(log(fabs(residual))
            - 0.5 * log(variance)) - scale));
    for (const auto& value : {scale, constant, local})
        printf("%.17g %.17g %.17g ", value.value, value.gradient[0], value.hessian[0]);
    printf("\n");
}
#endif

int main(int argc, char** argv)
{
    if (argc < 3) return 1;
    if (strcmp(argv[1], "--scalar") == 0) {
        for (int i = 2; i < argc; ++i) {
            const double x = strtod(argv[i], NULL);
            printf("%.17g %.17g\n", digamma_approx(x), trigamma_approx(x));
        }
        return 0;
    }
#ifdef __cplusplus
    const double x = strtod(argv[2], NULL);
    if (strcmp(argv[1], "--ged") == 0 && argc == 5) {
        ged_jet(x, strtod(argv[3], NULL), strtod(argv[4], NULL));
    } else if (strcmp(argv[1], "--first") == 0) {
        loggamma_jet<1>(x, false);
    } else if (strcmp(argv[1], "--second") == 0) {
        loggamma_jet<2>(x, false);
    } else if (strcmp(argv[1], "--chain") == 0) {
        loggamma_jet<2>(x, true);
    } else {
        return 1;
    }
    return 0;
#else
    return 1;
#endif
}
