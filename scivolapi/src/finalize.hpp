#pragma once

#include <vector>

#include <scivolapi/fit.hpp>
#include <scivolapi/jet.hpp>

namespace scivolapi::detail {

// One candidate point produced by an optimizer phase, in unit-scaled theta.
struct Candidate {
    std::vector<double> theta;
    double value = 0.0;           // F / n_scored
    std::size_t phase = 0;        // index into phases
    bool accepted = false;        // the phase's accepted iterate (vs. its best feasible trial)
};

// The single finalizer for every model. Qualification (feasibility, KKT,
// boundary activity) uses the unit-scaled problem the optimizer solved; the
// reported likelihood, derivatives, paths and covariance come from the jet on
// the original data, so no derivative transformation rules are needed.
FitResult finalize(const Model& model, Jet& unit_jet, Jet& data_jet, double data_scale,
                   const Candidate& selected, std::vector<FitPhase> phases, double accuracy);

}  // namespace scivolapi::detail
