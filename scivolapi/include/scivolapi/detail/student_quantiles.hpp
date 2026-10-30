#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace scivolapi::detail {

// u: each residual's smaller marginal tail probability min(F, 1 - F) in (0, 1/2];
// its side is the sign of the Gaussian score `normal`.
struct StudentProbabilities {
    std::vector<double> u, normal, identity_eta;
    std::vector<std::size_t> permutation;
    std::span<const double> original;
    std::size_t series = 1;
};

// Unscaled t scores w = x / sqrt(1 - 2 eta), and their eta derivatives.
struct StudentQuantileCache {
    double eta = std::numeric_limits<double>::quiet_NaN();
    int order = -1;
    std::span<double> x, first, second, iterate, area;
};

}  // namespace scivolapi::detail
