#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "math_and_helpers.h"

__attribute__((visibility("default"), hot, flatten))
double normal_ll(const double* __restrict sigma2, 
                 const double* __restrict residuals2, 
                 size_t n) {

    double log_like_acc = 0;
    for (size_t i = 0; i < n; ++i) {
        log_like_acc += log(sigma2[i]) + (residuals2[i] / sigma2[i]);
    }

    return 0.5 * log_like_acc;
}
