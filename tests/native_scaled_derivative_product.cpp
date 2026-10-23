// SCIVOL-NATIVE: TEST_ONLY; range-safe frozen-unit contraction controls.
#include "garch_observation_assembly.hpp"

#include <cstdio>

int main()
{
    using vlk::experiment::model::scaled_derivative_product;
    unsigned factors;
    double coefficient, row, column, row_unit, column_unit;
    while (std::scanf("%u%lf%lf%lf", &factors, &coefficient, &row, &row_unit) == 4) {
        if (factors == 1u)
            std::printf("%.17g\n", scaled_derivative_product(coefficient, row, row_unit));
        else {
            if (std::scanf("%lf%lf", &column, &column_unit) != 2) return 1;
            std::printf("%.17g\n", scaled_derivative_product(
                coefficient, row, column, row_unit, column_unit));
        }
    }
}
