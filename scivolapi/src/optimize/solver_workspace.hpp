#pragma once

#include "slsqp_backend.h"
#include "nnls_workspace.hpp"

#include <memory>

namespace scivolapi::optimize::detail {

struct SolverWorkspace {
    std::unique_ptr<std::byte[]> memory;
    double *buffer, *multipliers, *x, *lower, *upper, *gradient, *trial_gradient;
    double *values, *normals, *trial_values, *trial_normals, *constraint_rows;
    double *hessian, *factor, *packed_factor, *direction, *projected, *trial;
    double *qp_lower, *qp_upper, *qp_gradient, *correction;
    CBLAS_INT* indices;
    ScivolNnlsWorkspace nnls;
    std::size_t buffer_count = 0, multiplier_count = 0;

    bool allocate(std::size_t n, std::size_t m)
    {
        std::size_t square, rectangular, triangular;
        if (!checked_product(n, n, square) || !checked_product(m, n, rectangular)
            || !checked_product(n, n + 1, triangular)) return false;
        triangular /= 2;
        buffer_count = 28;
        const std::size_t terms[][2] = {
            {triangular, m == 0 ? 5u : 1u}, {rectangular, 3},
            {m, 9}, {square, 8}, {n, 35}};
        for (const auto& term : terms) {
            std::size_t contribution;
            if (!checked_product(term[0], term[1], contribution)
                || !checked_sum(buffer_count, contribution, buffer_count)) return false;
        }
        if (!checked_sum(m, 2 * n + 2, multiplier_count)) return false;
        nnls.bytes = nnls::sqp_workspace_bytes(n, m);
        // The C core uses BLAS integers for matrix indices as well as dimensions.
        if (!nnls.bytes || buffer_count > static_cast<std::size_t>(CBLAS_INT_MAX)) return false;

        WorkspaceSizer size;
        size.add<double>(buffer_count);
        size.add<CBLAS_INT>(multiplier_count);
        size.add<double>(multiplier_count);
        size.add<double>(7 * n);  // x, bounds, gradients, direction, projected
        size.add<double>(2 * std::max(m, std::size_t{1}));
        size.add<double>(3 * std::max(rectangular, std::size_t{1}));
        size.add<double>(2 * square + triangular);
        size.add<double>(3 * n + 2 * (n + 1)); // trial, QP gradient/correction, step bounds
        size.add<std::byte>(nnls.bytes);
        const auto bytes = size.size();
        if (!bytes) return false;
        memory.reset(new std::byte[bytes]{});
        WorkspaceArena arena(memory.get(), bytes);
        buffer = arena.take<double>(buffer_count);
        indices = arena.take<CBLAS_INT>(multiplier_count);
        multipliers = arena.take<double>(multiplier_count);
        x = arena.take<double>(n);
        lower = arena.take<double>(n);
        upper = arena.take<double>(n);
        gradient = arena.take<double>(n);
        trial_gradient = arena.take<double>(n);
        direction = arena.take<double>(n);
        projected = arena.take<double>(n);
        values = arena.take<double>(std::max(m, std::size_t{1}));
        trial_values = arena.take<double>(std::max(m, std::size_t{1}));
        normals = arena.take<double>(std::max(rectangular, std::size_t{1}));
        trial_normals = arena.take<double>(std::max(rectangular, std::size_t{1}));
        constraint_rows = arena.take<double>(std::max(rectangular, std::size_t{1}));
        hessian = arena.take<double>(square);
        factor = arena.take<double>(square);
        packed_factor = arena.take<double>(triangular);
        trial = arena.take<double>(n);
        qp_gradient = arena.take<double>(n);
        correction = arena.take<double>(n);
        qp_lower = arena.take<double>(n + 1);
        qp_upper = arena.take<double>(n + 1);
        nnls.memory = arena.take<std::byte>(nnls.bytes);
        return nnls.memory != nullptr;
    }
};

} // namespace scivolapi::optimize::detail
