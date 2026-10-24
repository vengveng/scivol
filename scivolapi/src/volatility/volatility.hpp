#pragma once

#include <cstddef>

// The volatility recursion concept. A recursion depends on the leading
// `width()` entries of theta and advances one observation at a time:
//
//   std::size_t width() const;       leading theta entries it depends on
//   std::size_t ring_slots() const;  past sensitivity vectors it keeps
//   bool evaluable(const double* theta) const;
//   bool step(theta, sample, t, order, VolatilityWork&, VolatilityStep&) const;
//
// `step` writes h_t into work.h and reports lambda_t = log h_t; for order >= 1
// it writes d lambda_t / d theta into work.l1 (width) and for order >= 2 the
// packed second derivatives into work.l2. It keeps whatever history it needs in
// the rings (its natural state's sensitivities). False: h_t is not representable.
namespace scivolapi::detail {

struct VolatilityStep {
    double h = 0.0;
    double lambda = 0.0;
};

struct VolatilityWork {
    const double* e;      // residuals (n)
    const double* e2;     // squared residuals (n)
    double* h;            // variance path (n)
    double* state;        // the recursion's natural state path, if it differs from h (n)
    double* first_ring;   // slots * width
    double* second_ring;  // slots * packed(width)
    double* d;            // current first sensitivities (width)
    double* s;            // current second sensitivities (packed width)
    double* l1;           // d lambda / d theta (width)
    double* l2;           // d2 lambda / d theta2 (packed width)
};

}  // namespace scivolapi::detail
