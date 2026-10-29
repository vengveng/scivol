#pragma once

#include <span>
#include <vector>

#include <scivolapi/fit.hpp>
#include <scivolapi/correlation.hpp>
#include <scivolapi/jet.hpp>

#include "optimize/optimizer.hpp"

namespace scivolapi::detail {

// One candidate point produced by an optimizer phase, in unit-scaled theta.
struct Candidate {
    std::vector<double> theta;
    double value = 0.0;           // F / n_scored
    std::size_t phase = 0;        // index into phases
    bool accepted = false;        // the phase's accepted iterate (vs. its best feasible trial)
    bool kink = false;
    std::vector<double> gradient;  // at theta on the theta problem, when the phase supplies it
};

// The single finalizer for every model (instantiated for Model/Jet and
// CorrelationModel/CorrelationJet). Qualification (feasibility, KKT, boundary
// activity) uses the unit-scaled problem the optimizer solved; the reported
// likelihood, derivatives, paths and covariance come from the jet on the
// original data, so no derivative transformation rules are needed.
template<class ModelT, class JetT>
FitResult finalize(const ModelT& model, JetT& unit_jet, JetT& data_jet, double data_scale,
                   const Candidate& selected, std::vector<FitPhase> phases, double accuracy,
                   std::span<const double> reported_theta = {});

// The Newton step of the local quadratic model of the problem's objective at
// x (gradient g, dense Hessian h): coordinates on closed bounds are held,
// constraints within `active` of binding are linearized as equalities. False
// where the reduced Hessian is not positive definite, so the model has no
// minimizer to locate.
bool face_newton_step(const optimize::Problem& problem, std::span<const double> x, std::span<const double> g,
                      std::span<const double> h, double active, std::vector<double>& step);
// Whether the step from x crosses one of the problem's open bounds.
bool crosses_open_bound(const optimize::Problem& problem, std::span<const double> x, std::span<const double> step);

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
