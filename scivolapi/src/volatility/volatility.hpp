#pragma once

#include <cstddef>

// The volatility recursion concept. A recursion depends on its own leading block of
// theta (`own()` entries from its first parameter on; EGARCH's includes the shapes)
// and advances one observation at a time:
//
//   std::size_t own() const;         entries of its own block
//   std::size_t width() const;       own() plus, when Coupled, the mean entries
//   std::size_t ring_slots() const;  past sensitivity vectors it keeps
//   bool evaluable(const double* theta) const;
//   bool prepare(const double* theta, int order);
//   bool step(theta, sample, t, order, VolatilityWork&, VolatilityStep&) const;
//
// `theta` points at the recursion's first parameter. `step` writes h_t into
// work.h and reports lambda_t = log h_t; for order >= 1 it writes d lambda_t into
// work.l1 and for order >= 2 the packed second derivatives into work.l2, both
// over its local coordinates [own block | mean entries]. It keeps whatever history
// it needs in the rings. False: h_t is not representable.
//
// Coupled recursions (a mean model is present) differentiate through the past
// residuals as well, using d e_k / d(mean) and d2 e_k / d(mean)^2 from the mean
// rings. Where the innovation term has a genuine kink at e_k = 0 (GJR's
// indicator, EGARCH's |z|, APARCH's power for small delta) the step lowers
// `VolatilityStep::defined`.
namespace scivolapi::detail {

struct VolatilityStep {
    double h = 0.0;
    double lambda = 0.0;
    int defined = 2;  // highest derivative order that exists from here on
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

    // The mean's residual sensitivities for recent observations (Coupled only).
    std::size_t mean = 0;        // mean entries
    std::size_t mean_slots = 1;  // ring slots, observation k in slot k % mean_slots
    const double* de = nullptr;  // mean_slots * mean
    const double* dde = nullptr; // mean_slots * packed(mean); null when e is linear in the mean
};

}  // namespace scivolapi::detail
