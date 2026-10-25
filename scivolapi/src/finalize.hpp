#pragma once

#include <span>
#include <vector>

#include <scivolapi/fit.hpp>
#include <scivolapi/correlation.hpp>
#include <scivolapi/jet.hpp>

namespace scivolapi::detail {

// One candidate point produced by an optimizer phase, in unit-scaled theta.
struct Candidate {
    std::vector<double> theta;
    double value = 0.0;           // F / n_scored
    std::size_t phase = 0;        // index into phases
    bool accepted = false;        // the phase's accepted iterate (vs. its best feasible trial)
};

// The single finalizer for every model (instantiated for Model/Jet and
// CorrelationModel/CorrelationJet). Qualification (feasibility, KKT, boundary
// activity) uses the unit-scaled problem the optimizer solved; the reported
// likelihood, derivatives, paths and covariance come from the jet on the
// original data, so no derivative transformation rules are needed.
template<class ModelT, class JetT>
FitResult finalize(const ModelT& model, JetT& unit_jet, JetT& data_jet, double data_scale,
                   const Candidate& selected, std::vector<FitPhase> phases, double accuracy);

// Covariance as the inverse of a symmetric information matrix (d x d), or why
// it is not available (not positive definite, ill conditioned).
Inference invert_information(const std::vector<double>& information, std::size_t d,
                             std::vector<double>& covariance);
// sum_t s_t s_t' over the rows of `scores` (rows x d, row-major).
std::vector<double> outer_products(std::span<const double> scores, std::size_t rows, std::size_t d);
// M B M' for d x d matrices.
std::vector<double> sandwich(const std::vector<double>& m, const std::vector<double>& b, std::size_t d);
// Square roots of the diagonal.
std::vector<double> standard_errors(const std::vector<double>& covariance, std::size_t d);

}  // namespace scivolapi::detail
