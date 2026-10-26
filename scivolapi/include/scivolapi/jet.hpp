#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "model.hpp"

namespace scivolapi {

// Requested outputs. Any non-empty combination is valid.
enum Request : unsigned {
    value = 1u,     // F: complete additive negative log-likelihood
    gradient = 2u,  // G
    hessian = 4u,   // H (dense, row-major, symmetric)
    paths = 8u,     // residuals, variances, standardized innovations
    scores = 16u,   // each scored row's contribution to G (theta coordinates only)
    terms = 32u,    // each scored row's term of F, its share of the constants included
    log_variance_sensitivities = 64u, // d log h_t / d theta, scored rows x dimension
    residual_sensitivities = 128u,    // d e_t / d theta, scored rows x dimension
};

enum class Status {
    ok,
    invalid_argument,  // wrong sizes or empty request
    not_evaluable,     // the likelihood is undefined at this point
    numerical_range,   // defined, but not representable in double precision
};

struct Paths {
    std::span<double> residual;
    std::span<double> variance;
    std::span<double> innovation;
};

// Caller-provided outputs. Unrequested outputs are not touched.
struct JetOutput {
    double value = std::numeric_limits<double>::quiet_NaN();
    std::span<double> gradient;  // dimension
    std::span<double> hessian;   // dimension * dimension
    Paths paths;                 // sample size each
    std::span<double> scores;    // scored rows * dimension, row-major
    std::span<double> terms;     // scored rows
    // Row-major, including zero columns for parameters a path does not depend on.
    // In z these are first derivatives at the original z, multiplied by d theta/dz.
    std::span<double> log_variance_sensitivities;
    std::span<double> residual_sensitivities;
};

struct JetResult {
    Status status = Status::invalid_argument;
    unsigned valid = 0u;  // which requested outputs hold valid numbers
};

// The model's likelihood jet on one sample. The only source of likelihood
// values, derivatives and fitted paths. Owns its workspace (allocated once);
// evaluations allocate nothing. Borrows the model and the sample.
class Jet {
public:
    Jet(const Model& model, const Sample& sample);

    [[nodiscard]] const Model& model() const noexcept { return model_; }
    [[nodiscard]] const Sample& sample() const noexcept { return sample_; }

    // Derivatives with respect to theta.
    JetResult theta(std::span<const double> theta, unsigned request, JetOutput& out);
    // Derivatives of L(theta(z)) with respect to z, including map curvature.
    // Scores are not available in z.
    JetResult z(std::span<const double> z, unsigned request, JetOutput& out);

private:
    const Model& model_;
    Sample sample_;
    std::vector<double> work_;
};

}  // namespace scivolapi
