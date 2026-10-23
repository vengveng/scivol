#pragma once

#include "density_policy.hpp"
#include "jet_algebra.hpp"
#include "parameter_domain.hpp"

namespace scivolapi::detail::dependence {

template<int Order, class Dimensions, class Basis = IdentitySensitivityBasis>
struct Scratch {
    JetAlgebra<Order, Dimensions, Basis> algebra;
    double *history, *current, *correlation, *cholesky, *forward, *scalars;
    Scratch(Dimensions dimensions, std::span<std::byte> memory, Basis basis = {}) noexcept
        : algebra(dimensions, basis) {
        const auto l = algebra.get();
        const auto stride = algebra.width();
        workspace::WorkspaceArena arena(memory.data(), memory.size());
        history = arena.take<double>(l.q * l.matrix * stride);
        current = arena.take<double>(l.matrix * stride);
        correlation = arena.take<double>(l.matrix * stride);
        cholesky = arena.take<double>(l.matrix * stride);
        forward = arena.take<double>(l.series * stride);
        scalars = arena.take<double>(6 * stride);
    }
    [[nodiscard]] double* cell(double* matrix, std::size_t element) const noexcept {
        return matrix + element * algebra.width();
    }
    [[nodiscard]] const double* shock(SampleView sample, std::size_t t, std::size_t lag) const noexcept {
        const auto elapsed = t - sample.recursion_begin;
        if (elapsed > lag) return sample.residuals.data() + (t - lag - 1) * algebra.get().series;
        if (sample.residual_history.empty()) return nullptr;
        return sample.residual_history.data() + (lag - elapsed) * algebra.get().series;
    }
    void initialize(SampleView sample) const noexcept {
        const auto l = algebra.get();
        for (std::size_t lag = 0; lag < l.q; ++lag)
            for (std::size_t ij = 0; ij < l.matrix; ++ij)
                algebra.constant(cell(history, lag * l.matrix + ij), sample.correlation_history.empty()
                    ? sample.seed[ij] : sample.correlation_history[lag * l.matrix + ij]);
    }
    void advance(SampleView sample, std::span<const double> theta, double intercept,
        std::size_t t) const noexcept {
        const auto l = algebra.get();
        for (std::size_t ij = 0; ij < l.matrix; ++ij) {
            auto* point = cell(current, ij);
            algebra.constant(point, intercept * sample.target[ij]);
            if constexpr (Order >= 1)
                for (std::size_t k = 0; k < algebra.correlation(); ++k) point[1 + k] = -sample.target[ij];
        }
        for (std::size_t lag = 0; lag < l.p; ++lag) {
            const double* residual = shock(sample, t, lag);
            if (!residual) continue;
            for (std::size_t i = 0; i < l.series; ++i)
                for (std::size_t j = 0; j < l.series; ++j) {
                    const double outer = residual[i] * residual[j];
                    auto* point = cell(current, i * l.series + j);
                    point[0] += theta[lag] * outer;
                    if constexpr (Order >= 1) {
                        const auto axis = algebra.state_axis(lag);
                        if (axis != inactive_axis) point[1 + axis] += outer;
                    }
                }
        }
        for (std::size_t lag = 0; lag < l.q; ++lag)
            for (std::size_t ij = 0; ij < l.matrix; ++ij)
                algebra.add_parameter_product(cell(current, ij), theta[l.beta_start + lag],
                    l.beta_start + lag, cell(history, lag * l.matrix + ij));
    }
    void retain() const noexcept {
        const auto l = algebra.get();
        const auto count = l.matrix * algebra.width();
        for (std::size_t lag = l.q - 1; lag > 0; --lag)
            std::copy_n(history + (lag - 1) * count, count, history + lag * count);
        std::copy_n(current, count, history);
    }
    [[nodiscard]] Status normalize() const noexcept {
        const auto l = algebra.get();
        for (std::size_t i = 0; i < l.series; ++i) {
            const auto* diagonal_i = cell(current, i * l.series + i);
            if (!(diagonal_i[0] > 0) || !finite(diagonal_i[0])) return Status::numerical_range;
            algebra.constant(cell(correlation, i * l.series + i), 1);
            for (std::size_t j = i + 1; j < l.series; ++j) {
                const auto* diagonal_j = cell(current, j * l.series + j);
                if (!(diagonal_j[0] > 0) || !finite(diagonal_j[0])) return Status::numerical_range;
                const auto* q = cell(current, i * l.series + j);
                auto* r = cell(correlation, i * l.series + j);
                const double root_i = std::sqrt(diagonal_i[0]), root_j = std::sqrt(diagonal_j[0]);
                const double value = q[0] / root_i / root_j;
                r[0] = value;
                if (!finite(value)) return Status::numerical_range;
                if constexpr (Order >= 1)
                    for (std::size_t k = 0; k < algebra.correlation(); ++k)
                        r[1 + k] = q[1 + k] / root_i / root_j - .5 * value
                            * (diagonal_i[1 + k] / diagonal_i[0] + diagonal_j[1 + k] / diagonal_j[0]);
                if constexpr (Order >= 2) {
                    for (std::size_t k = 0; k < algebra.correlation(); ++k)
                        for (std::size_t m = k; m < algebra.correlation(); ++m) {
                            const auto h = algebra.h(k, m);
                            const double ik = diagonal_i[1 + k] / diagonal_i[0];
                            const double im = diagonal_i[1 + m] / diagonal_i[0];
                            const double jk = diagonal_j[1 + k] / diagonal_j[0];
                            const double jm = diagonal_j[1 + m] / diagonal_j[0];
                            r[h] = q[h] / root_i / root_j
                                - .5 * (q[1 + k] / root_i / root_j) * (im + jm)
                                - .5 * (q[1 + m] / root_i / root_j) * (ik + jk)
                                + value * (.25 * (ik + jk) * (im + jm) + .5 * (ik * im + jk * jm)
                                    - .5 * (diagonal_i[h] / diagonal_i[0] + diagonal_j[h] / diagonal_j[0]));
                        }
                }
                algebra.copy(cell(correlation, j * l.series + i), r);
            }
        }
        return Status::ok;
    }
    template<bool F>
    [[nodiscard]] Status sufficient_state(std::span<const double> residual,
        double* logdet, double* delta) const noexcept {
        const auto l = algebra.get();
        auto* work = cell(scalars, 2);
        auto* product = cell(scalars, 3);
        auto* inverse = cell(scalars, 4);
        auto* logarithm = cell(scalars, 5);
        algebra.constant(logdet, 0); algebra.constant(delta, 0);
        for (std::size_t i = 0; i < l.series; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                algebra.copy(work, cell(correlation, i * l.series + j));
                for (std::size_t k = 0; k < j; ++k) {
                    algebra.multiply(product, cell(cholesky, i * l.series + k),
                        cell(cholesky, j * l.series + k));
                    algebra.add(work, product, -1);
                }
                auto* output = cell(cholesky, i * l.series + j);
                if (i == j) {
                    if (!(work[0] > 0) || !finite(work[0])) return Status::numerical_range;
                    algebra.square_root(output, work);
                    algebra.template logarithm<F>(logarithm, output);
                    algebra.add(logdet, logarithm, 2);
                } else {
                    algebra.reciprocal(inverse, cell(cholesky, j * l.series + j));
                    algebra.multiply(output, work, inverse);
                }
                if (!finite(output[0])) return Status::numerical_range;
            }
            algebra.constant(work, residual[i]);
            for (std::size_t j = 0; j < i; ++j) {
                algebra.multiply(product, cell(cholesky, i * l.series + j), cell(forward, j));
                algebra.add(work, product, -1);
            }
            algebra.reciprocal(inverse, cell(cholesky, i * l.series + i));
            algebra.multiply(cell(forward, i), work, inverse);
            algebra.multiply(product, cell(forward, i), cell(forward, i));
            algebra.add(delta, product);
        }
        return finite(delta[0]) ? Status::ok : Status::numerical_range;
    }
};

[[nodiscard]] inline JetValidity failed(JetRequest request, Status reason) noexcept {
    JetValidity validity;
    validity.requested = request; validity.status = reason;
    const auto mask = static_cast<unsigned>(request);
    if (mask & 1) validity.f_reason = reason;
    if (mask & 2) validity.g_reason = reason;
    if (mask & 4) validity.h_reason = reason;
    return validity;
}

template<Law Density, unsigned Mask, bool Shape, bool Correlation, class Dimensions, class Basis>
[[nodiscard]] JetValidity evaluate_selected_kernel(Dimensions dimensions, SampleView sample,
    std::span<const double> theta, std::span<std::byte> memory,
    dependence_backend::Buffers outputs, Basis basis) noexcept
{
    constexpr bool F = (Mask & 1) != 0, G = (Mask & 2) != 0, H = (Mask & 4) != 0;
    constexpr int Order = Correlation ? (H ? 2 : G ? 1 : 0) : 0;
    static_assert(Mask > 0 && Mask <= 7);
    const auto l = dimensions.get();
    const auto bytes = workspace_bytes<Order>(l, basis);
    if (!bytes) return failed(static_cast<JetRequest>(Mask), Status::invalid_argument);
    if (memory.size() < bytes) return failed(static_cast<JetRequest>(Mask), Status::insufficient_capacity);
    const auto coefficient_point = parameter_point(l, theta);
    if (coefficient_point.status != Status::ok)
        return failed(static_cast<JetRequest>(Mask), coefficient_point.status);
    Scratch<Order, Dimensions, Basis> state(dimensions, memory, basis);
    const auto width = state.algebra.parameters(), shape = state.algebra.shape();
    const auto dense = width * width;
    if (!state.scalars) return failed(static_cast<JetRequest>(Mask), Status::insufficient_capacity);
    state.initialize(sample);
    const double intercept = coefficient_point.intercept;
    const auto rows = sample.residuals.size() / l.series;
    const auto scored = rows - sample.first_scored;
    const double nu = Density == Law::studentt ? theta[l.shape_start] : 0;
    // Conditional type prevents Normal requests from constructing Student state.
    struct NormalPoint { Normalizer constant{}; };
    const auto point = [&] {
        if constexpr (Density == Law::studentt) return StudentPoint<F, G, H, Shape>(nu, l.series);
        else return NormalPoint{};
    }();
    const auto constant = point.constant;
    if constexpr (F) outputs.value[0] = static_cast<double>(scored) * constant.value;
    if constexpr (G) {
        std::fill_n(outputs.gradient.data(), width, 0);
        if constexpr (Density == Law::studentt && Shape)
            outputs.gradient[shape] = static_cast<double>(scored) * constant.first;
    }
    if constexpr (H) {
        std::fill_n(outputs.hessian.data(), dense, 0);
        if constexpr (Density == Law::studentt && Shape)
            outputs.hessian[shape * width + shape]
                = static_cast<double>(scored) * constant.second;
    }
    auto* logdet = state.cell(state.scalars, 0);
    auto* delta = state.cell(state.scalars, 1);
    for (std::size_t t = sample.recursion_begin; t < rows; ++t) {
        state.advance(sample, theta, intercept, t);
        for (std::size_t ij = 0; ij < l.matrix; ++ij)
            if (!finite(state.cell(state.current, ij)[0]))
                return failed(static_cast<JetRequest>(Mask), Status::numerical_range);
        if (t >= sample.first_scored) {
            const auto residual = sample.residuals.subspan(t * l.series, l.series);
            const auto normalization = state.normalize();
            if (normalization != Status::ok) return failed(static_cast<JetRequest>(Mask), normalization);
            const auto factorization = state.template sufficient_state<F>(residual, logdet, delta);
            if (factorization != Status::ok) return failed(static_cast<JetRequest>(Mask), factorization);
            DensityPartials local;
            if constexpr (Density == Law::studentt)
                local = student_observation<F, G, H, Shape, Correlation>(point, delta[0], residual);
            else {
                local.delta = .5;
                if constexpr (F) {
                    local.value = .5 * delta[0];
                    for (double e : residual) local.value -= .5 * e * e;
                }
            }
            if constexpr (F) outputs.value[0] += .5 * logdet[0] + local.value;
            if constexpr (G) {
                for (std::size_t k = 0; k < state.algebra.correlation(); ++k)
                    outputs.gradient[state.algebra.output_axis(k)]
                        += .5 * logdet[1 + k] + local.delta * delta[1 + k];
                if constexpr (Density == Law::studentt && Shape) outputs.gradient[shape] += local.shape;
            }
            if constexpr (H) {
                for (std::size_t k = 0; k < state.algebra.correlation(); ++k)
                    for (std::size_t j = k; j < state.algebra.correlation(); ++j) {
                        const auto h = state.algebra.h(k, j);
                        double increment = .5 * logdet[h] + local.delta * delta[h];
                        if constexpr (Density == Law::studentt) {
                            const double scaled_k = delta[1 + k] * local.delta_inverse;
                            const double scaled_j = delta[1 + j] * local.delta_inverse;
                            increment += local.delta_curvature_weight * scaled_k * scaled_j;
                        }
                        const auto out_k = state.algebra.output_axis(k), out_j = state.algebra.output_axis(j);
                        outputs.hessian[out_k * width + out_j] += increment;
                        if (k != j) outputs.hessian[out_j * width + out_k] += increment;
                    }
                if constexpr (Density == Law::studentt && Shape) {
                    for (std::size_t k = 0; k < state.algebra.correlation(); ++k) {
                        const double increment = local.delta_shape * delta[1 + k];
                        const auto out = state.algebra.output_axis(k);
                        outputs.hessian[out * width + shape] += increment;
                        outputs.hessian[shape * width + out] += increment;
                    }
                    outputs.hessian[shape * width + shape] += local.shape_shape;
                }
            }
        }
        state.retain();
    }
    auto validity = failed(static_cast<JetRequest>(Mask), Status::numerical_range);
    const auto accept = [&](std::span<const double> values, unsigned bit, Status& reason) {
        for (double value : values) if (!finite(value)) return;
        validity.valid_mask |= bit; reason = Status::ok;
    };
    if constexpr (F) accept(outputs.value.first(1), 1, validity.f_reason);
    if constexpr (G) accept(outputs.gradient.first(width), 2, validity.g_reason);
    if constexpr (H) accept(outputs.hessian.first(dense), 4, validity.h_reason);
    if (validity.valid_mask == Mask) validity.status = Status::ok;
    return validity;
}

template<Law Density, unsigned Mask, class Dimensions, class Basis = IdentitySensitivityBasis>
[[nodiscard]] JetValidity evaluate_kernel(Dimensions dimensions, SampleView sample,
    std::span<const double> theta, std::span<std::byte> memory,
    dependence_backend::Buffers outputs, Basis basis = {}) noexcept
{
    if constexpr (Basis::identity) {
        return evaluate_selected_kernel<Density, Mask, Density == Law::studentt, true>(
            dimensions, sample, theta, memory, outputs, basis);
    } else {
        const auto l = dimensions.get();
        const bool correlation = correlation_width(l, basis) != 0;
        if constexpr (Density == Law::studentt) {
            if (basis.depends(l.shape_start)) {
                if (correlation) return evaluate_selected_kernel<Density, Mask, true, true>(
                    dimensions, sample, theta, memory, outputs, basis);
                return evaluate_selected_kernel<Density, Mask, true, false>(
                    dimensions, sample, theta, memory, outputs, basis);
            }
        }
        if (correlation) return evaluate_selected_kernel<Density, Mask, false, true>(
            dimensions, sample, theta, memory, outputs, basis);
        return evaluate_selected_kernel<Density, Mask, false, false>(
            dimensions, sample, theta, memory, outputs, basis);
    }
}

template<Law Density, class Dimensions, class Basis = IdentitySensitivityBasis>
[[nodiscard]] JetValidity dispatch_request(Dimensions dimensions, SampleView sample,
    std::span<const double> theta, JetRequest request, std::span<std::byte> workspace,
    dependence_backend::Buffers buffers, Basis basis = {}) noexcept
{
    switch (request) {
    case JetRequest::F: return evaluate_kernel<Density, 1>(dimensions, sample, theta, workspace, buffers, basis);
    case JetRequest::G: return evaluate_kernel<Density, 2>(dimensions, sample, theta, workspace, buffers, basis);
    case JetRequest::FG: return evaluate_kernel<Density, 3>(dimensions, sample, theta, workspace, buffers, basis);
    case JetRequest::H: return evaluate_kernel<Density, 4>(dimensions, sample, theta, workspace, buffers, basis);
    case JetRequest::FH: return evaluate_kernel<Density, 5>(dimensions, sample, theta, workspace, buffers, basis);
    case JetRequest::GH: return evaluate_kernel<Density, 6>(dimensions, sample, theta, workspace, buffers, basis);
    case JetRequest::FGH: return evaluate_kernel<Density, 7>(dimensions, sample, theta, workspace, buffers, basis);
    }
    return failed(request, Status::invalid_argument);
}
} // namespace scivolapi::detail::dependence
