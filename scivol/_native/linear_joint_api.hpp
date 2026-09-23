// SCIVOL-NATIVE: shared checked ABI for linear-mean volatility programs.
#pragma once

#include <cstddef>
#include <cstdint>

namespace scivol::linear_joint {

struct Plan {
    unsigned family, density;
    std::size_t columns, p, q, n;
    bool fixed;
};

struct Evaluation {
    const double *theta, *y, *x;
    double h0;
    unsigned request;
    bool transformed;
    void* workspace;
    std::size_t bytes;
    double *value, *gradient, *hessian, *residuals, *variances;
};

struct Domain {
    std::size_t order, width, offset;
    const double* basis;
};

struct Solve {
    double* point;
    const double *y, *x;
    double h0;
    Domain mean;
    const double *lower, *upper, *scales;
    void* workspace;
    std::size_t bytes;
    double *theta, *gradient, *value;
    std::int64_t* stats;
    double accuracy;
    std::int64_t iterations;
    bool transformed;
    bool speculate;
    const double* mean_normal = nullptr;
    double mean_target = 0.0;
    std::int64_t* requests = nullptr; // Optional three-entry F, G, F|G attempt counts.
    bool curvature_reset = false;
    std::int64_t* reset_count = nullptr; // Optional one-entry policy reset count.
};

std::size_t power_workspace(const Plan&, unsigned, bool);
std::size_t power_solve_workspace(const Plan&, const Domain&, bool, bool affine = false);
bool power_evaluate(const Plan&, const Evaluation&);
bool power_inverse(const Plan&, const double*, double*, void*, std::size_t);
bool power_fit(const Plan&, const Solve&);
bool power_constraints(const Plan&, const Domain&, const double*, double*, double*, void*, std::size_t);

std::size_t garch_workspace(const Plan&, unsigned, bool);
std::size_t garch_solve_workspace(const Plan&, const Domain&, bool);
bool garch_evaluate(const Plan&, const Evaluation&);
bool garch_inverse(const Plan&, const double*, double*, void*, std::size_t);
bool garch_fit(const Plan&, const Solve&);
bool garch_constraints(const Plan&, const Domain&, const double*, double*, double*, void*, std::size_t);

} // namespace scivol::linear_joint
