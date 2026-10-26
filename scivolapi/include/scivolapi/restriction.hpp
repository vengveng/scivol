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
    singular_information, retained_boundary, unresolved, invalid_argument,
};
enum class ReferenceLaw { chi_square, chi_bar_square };
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
double chi_square_survival(double statistic, std::size_t degrees);
LRResult restriction_test(const Model& parent, const Model& child, const Restriction& restriction,
                          const FitResult& parent_fit, const FitResult& child_fit,
                          std::span<const double> parent_information, double level = 0.01,
                          double tolerance = 1e-6, std::size_t maximum_cells = 32768);

}  // namespace scivolapi
