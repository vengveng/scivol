#pragma once

#include "profile_backend.hpp"
#include "math/fixed_differential.hpp"
#include <array>
#include <cstdlib>
#include <vector>

namespace scivolapi::test::profile_oracle {
namespace ad = vlk::experiment::ad;
namespace backend = detail::profile_backend;
namespace mean = detail::mean_backend;

// Existing native scalar objective, independent of profile assembly/recursion.
// Shared elementary AD arithmetic makes this a smoke oracle, not JAX evidence.
template<std::size_t K>
ad::Differential<2,K> scalar_profile(backend::ModelConfig config, backend::SampleView sample,
    std::span<const double> point, Coordinates coordinates, std::span<const double> actual_theta,
    double* variance = nullptr) {
    using D = ad::Differential<2,K>;
    std::array<D,K> theta;
    for (std::size_t i = 0; i < K; ++i) theta[i] = D::variable(point[i], i);
    const auto columns = config.kind == mean::Kind::arma
        ? config.p_ar + std::size_t(config.intercept)
        : config.design_width + std::size_t(config.intercept);
    if (coordinates == Coordinates::original_chart) {
        const auto ar_order = config.kind == mean::Kind::linear ? config.linear_ar_order : config.p_ar;
        const std::array blocks{std::array{std::size_t(config.intercept), ar_order},
            std::array{columns, config.q_ma}};
        for (std::size_t block = 0; block < blocks.size(); ++block) {
            const auto start = blocks[block][0], order = blocks[block][1];
            std::array<D,K> coefficients{};
            for (std::size_t stage = 0; stage < order; ++stage) {
                const auto reflection = ad::tanh(theta[start+stage]);
                const auto previous = coefficients;
                for (std::size_t row = 0; row < stage; ++row)
                    coefficients[row] = previous[row] - reflection*previous[stage-1-row];
                coefficients[stage] = reflection;
            }
            for (std::size_t i = 0; i < order; ++i)
                theta[start+i] = block == 0 ? coefficients[i] : -coefficients[i];
        }
        // Values anchor to actual mapped doubles; derivatives stay at original z.
        for (std::size_t i = 0; i < K; ++i) theta[i].value = actual_theta[i];
    }
    std::vector<D> residual(sample.observations.size(), D::constant(0));
    D sum = D::constant(0);
    if (!sample.recursion_begin) std::abort();
    for (std::size_t t = *sample.recursion_begin; t < residual.size(); ++t) {
        auto e = D::constant(sample.observations[t]);
        std::size_t column = 0;
        if (config.intercept) e = e-theta[column++];
        if (config.kind == mean::Kind::arma) {
            for (std::size_t lag = 0; lag < config.p_ar; ++lag) {
                const double previous = !sample.observation_history.empty() && t-*sample.recursion_begin <= lag
                    ? sample.observation_history[lag-(t-*sample.recursion_begin)] : sample.observations[t-1-lag];
                e = e-theta[column++]*D::constant(previous);
            }
        } else {
            for (std::size_t j = 0; j < config.design_width; ++j)
                e = e-theta[column++]*D::constant(sample.design[t*config.design_width+j]);
        }
        for (std::size_t lag = 0; lag < config.q_ma; ++lag) {
            const auto previous = !sample.innovation_history.empty() && t-*sample.recursion_begin <= lag
                ? D::constant(sample.innovation_history[lag-(t-*sample.recursion_begin)]) : residual[t-1-lag];
            e = e-theta[column+lag]*previous;
        }
        residual[t] = e;
        if (t >= sample.first_scored) sum = sum+e*e;
    }
    const double n = double(residual.size()-sample.first_scored);
    if (variance) *variance = sum.value / n;
    const double log_two_pi = std::log(2.0*std::acos(-1.0));
    return D::constant(0.5*n)*(D::constant(1.0+log_two_pi) + ad::log(sum/D::constant(n)));
}
} // namespace scivolapi::test::profile_oracle
