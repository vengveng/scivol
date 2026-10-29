#pragma once

#include <scivolapi/correlation.hpp>

#include <algorithm>

#include "../detail/numeric.hpp"

#include "../density/distribution.hpp"

namespace scivolapi::detail::correlation {

// F = G: preserve the original residual without a CDF/quantile round trip.
inline bool identity(const InnovationDensity& marginal, CorrelationDensity copula, double eta = 0.0)
{
    if (copula == CorrelationDensity::normal)
        return marginal.density == DensityKind::normal;
    return marginal.density == DensityKind::student_t && marginal.shape[0] == eta;
}

inline Status copula_scores(const CorrelationSample& sample, bool student, double eta, int order,
                            StudentQuantileCache& cache)
{
    const std::size_t count = sample.eps.size();
    if (!student) {
        std::copy_n(sample.probabilities.normal.data(), count, cache.x.data());
        return Status::ok;
    }
    const auto status = student_quantiles(sample.probabilities, eta, order, cache);
    if (status != ExpectationStatus::ok) return Status::numerical_range;
    return Status::ok;
}

inline void score_target(const double* x, std::size_t rows, std::size_t n, double* target,
                          const double* first = nullptr, double* target_first = nullptr,
                          const double* second = nullptr, double* target_second = nullptr)
{
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            CompensatedSum sum, d1, d2;
            for (std::size_t t = 0; t < rows; ++t) {
                const std::size_t a = t * n + i, b = t * n + j;
                const double scale = static_cast<double>(rows);
                sum.add((x[a] / scale) * x[b]);
                if (first) d1.add((first[a] * x[b] + x[a] * first[b]) / scale);
                if (second) d2.add((second[a] * x[b] + 2.0 * first[a] * first[b] + x[a] * second[b]) / scale);
            }
            target[i * n + j] = target[j * n + i] = sum.total();
            if (first) target_first[i * n + j] = target_first[j * n + i] = d1.total();
            if (second) target_second[i * n + j] = target_second[j * n + i] = d2.total();
        }
}

}  // namespace scivolapi::detail::correlation
