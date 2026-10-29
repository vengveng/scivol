#pragma once

#include "fit.hpp"

namespace scivolapi {

enum class RestrictFitStatus { reused, fit_needed, parent_uncertified, incompatible, child_uncertified };
struct RestrictedFit {
    RestrictFitStatus status = RestrictFitStatus::fit_needed;
    FitResult result;
};
RestrictedFit restrict_fit(const Model& parent, const FitResult& fitted, const Restriction& restriction,
                          const ComparisonPlan& plan);

enum class LRStatus {
    ok, unsupported, incompatible, uncertified, negative_statistic,
    singular_information, unresolved, invalid_argument,
};
enum class ReferenceLaw { chi_square, chi_bar_square, cone_projection };
struct LRResult {
    LRStatus status = LRStatus::invalid_argument;
    double statistic = std::numeric_limits<double>::quiet_NaN();
    ReferenceLaw law = ReferenceLaw::chi_square;
    std::size_t interior_df = 0;
    std::vector<double> weights;
    double p_value = std::numeric_limits<double>::quiet_NaN();
    double numerical_error = std::numeric_limits<double>::infinity();
    std::string message;
};

struct LRWeights {
    LRStatus status = LRStatus::invalid_argument;
    std::vector<double> weights;
    // Sum of absolute weight errors, hence an absolute p-value error bound.
    double numerical_error = std::numeric_limits<double>::infinity();
};

// Efficient boundary information J, row-major. Deterministic conditioning
// cubature for blocks larger than three, with a bounded subdivision budget.
LRWeights chi_bar_weights(std::span<const double> information, std::size_t boundaries,
                         double tolerance = 1e-6, std::size_t maximum_cells = 32768);
// Difference of squared information-metric distances to nested tangent cones.
// Parent: normals * h >= 0; child adds equalities * h = 0. Row-major inputs.
// Fixed-seed mt19937_64 / Box-Muller Gaussian simulation, with a 99.99% Chernoff
// p-value interval (numerical_error is its maximum half-width, not a rigorous
// cubature bound). An interval crossing level is unresolved. No chi-bar weights
// are asserted for a general retained boundary nuisance (Self--Liang, Case 8).
LRResult cone_projection_test(std::span<const double> information, std::size_t dimension,
                             std::span<const double> normals, std::span<const double> equalities,
                             double statistic, double level = 0.01, std::size_t draws = 32768);
LRResult restriction_test(const Model& parent, const Model& child, const Restriction& restriction,
                          const FitResult& parent_fit, const FitResult& child_fit,
                          std::span<const double> parent_information, double level = 0.01,
                          double tolerance = 1e-6, std::size_t maximum_cells = 32768);
// Uses the parent's score outer products at the embedded null estimate, a
// consistent Fisher information estimator under the correct-likelihood null.
// Unlike an observed Hessian at a strongly false null it is positive semidefinite.
LRResult restriction_test(const Model& parent, const Model& child, const Restriction& restriction,
                          const FitResult& parent_fit, const FitResult& child_fit,
                          const ComparisonPlan& plan, double level = 0.01,
                          double tolerance = 1e-6, std::size_t maximum_cells = 32768);

}  // namespace scivolapi
