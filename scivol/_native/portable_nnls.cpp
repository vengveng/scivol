// SCIVOL-NATIVE: CORE
#if !defined(BLAS_SYMBOL_PREFIX)
#  define BLAS_SYMBOL_PREFIX vlk_
#endif
#if !defined(NO_APPEND_FORTRAN)
#  define NO_APPEND_FORTRAN
#endif

#if defined(_MSC_VER)
#  include <malloc.h>
#  define VLK_STACK_ALLOCATE(bytes) _alloca(bytes)
#  define VLK_NNLS_RESTRICT __restrict
#else
#  define VLK_STACK_ALLOCATE(bytes) __builtin_alloca(bytes)
#  define VLK_NNLS_RESTRICT __restrict__
#endif

#if !defined(restrict)
#  define VLK_RESTORE_RESTRICT
#  define restrict VLK_NNLS_RESTRICT
#endif
extern "C" {
#include "third_party/scipy_slsqp/nnls.h"
}
#if defined(VLK_RESTORE_RESTRICT)
#  undef restrict
#  undef VLK_RESTORE_RESTRICT
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr std::size_t maximum_stack_bytes = 128u * 1024u;

bool planned_stack_bytes(
    std::size_t rows, std::size_t columns, std::size_t& bytes)
{
    if (columns == 0u
        || rows > maximum_stack_bytes / sizeof(double) / columns) {
        return false;
    }
    const std::size_t matrix_elements = rows * columns;
    const std::size_t rhs_elements = std::max(rows, columns);
    const std::size_t work_elements = matrix_elements + 3u * columns + 1u;
    const std::size_t double_elements =
        columns + matrix_elements + rhs_elements + work_elements;
    if (double_elements > maximum_stack_bytes / sizeof(double))
        return false;
    const std::size_t double_bytes = double_elements * sizeof(double);
    const std::size_t integer_bytes = 2u * columns * sizeof(CBLAS_INT);
    const std::size_t flag_bytes = columns * sizeof(unsigned char);
    if (integer_bytes > maximum_stack_bytes - double_bytes
        || flag_bytes > maximum_stack_bytes - double_bytes - integer_bytes) {
        return false;
    }
    bytes = double_bytes + integer_bytes + flag_bytes;
    return true;
}

}  // namespace

/*
 * Refactorized Lawson-Hanson NNLS.
 *
 * Rebuilding the small passive least-squares problem avoids the backend-
 * dependent cycling seen with incremental QR updates. Scratch storage is
 * bounded, automatic, and rejected before the stack pointer moves.
 */
extern "C" void __nnls(
    const CBLAS_INT m,
    const CBLAS_INT n,
    double* VLK_NNLS_RESTRICT a,
    double* VLK_NNLS_RESTRICT b,
    double* VLK_NNLS_RESTRICT x,
    double* VLK_NNLS_RESTRICT dual,
    double* VLK_NNLS_RESTRICT residual,
    CBLAS_INT* VLK_NNLS_RESTRICT passive,
    const std::int64_t maxiter,
    double* rnorm,
    std::int64_t* info)
{
    *info = 1;
    *rnorm = 0.0;
    if (m <= 0 || n <= 0) {
        *info = 2;
        return;
    }

    const std::size_t rows = static_cast<std::size_t>(m);
    const std::size_t columns = static_cast<std::size_t>(n);
    std::size_t stack_bytes = 0u;
    if (!planned_stack_bytes(rows, columns, stack_bytes)) {
        *info = 2;
        return;
    }
    static_cast<void>(stack_bytes);

    auto* candidate = static_cast<double*>(
        VLK_STACK_ALLOCATE(columns * sizeof(double)));
    auto* active_matrix = static_cast<double*>(
        VLK_STACK_ALLOCATE(rows * columns * sizeof(double)));
    auto* active_rhs = static_cast<double*>(
        VLK_STACK_ALLOCATE(std::max(rows, columns) * sizeof(double)));
    auto* least_squares_work = static_cast<double*>(VLK_STACK_ALLOCATE(
        (rows * columns + 3u * columns + 1u) * sizeof(double)));
    auto* active_permutation = static_cast<CBLAS_INT*>(
        VLK_STACK_ALLOCATE(columns * sizeof(CBLAS_INT)));
    auto* active_to_original = static_cast<CBLAS_INT*>(
        VLK_STACK_ALLOCATE(columns * sizeof(CBLAS_INT)));
    auto* eligible = static_cast<unsigned char*>(
        VLK_STACK_ALLOCATE(columns * sizeof(unsigned char)));

    for (CBLAS_INT column = 0; column < n; ++column) {
        x[column] = 0.0;
        passive[column] = 0;
        eligible[column] = 1;
    }
    const std::int64_t iteration_limit =
        (maxiter > 0 ? maxiter : 3 * static_cast<std::int64_t>(n)) * 32 + 32;
    std::int64_t iterations = 0;

    while (true) {
        for (CBLAS_INT row = 0; row < m; ++row) {
            double fitted = 0.0;
            for (CBLAS_INT column = 0; column < n; ++column)
                fitted += a[row + column * m] * x[column];
            residual[row] = b[row] - fitted;
        }

        CBLAS_INT entering = -1;
        double maximum_dual = 0.0;
        for (CBLAS_INT column = 0; column < n; ++column) {
            double correlation = 0.0;
            for (CBLAS_INT row = 0; row < m; ++row)
                correlation += a[row + column * m] * residual[row];
            dual[column] = correlation;
            if (!passive[column] && eligible[column]
                && correlation > maximum_dual) {
                maximum_dual = correlation;
                entering = column;
            }
        }
        if (entering < 0)
            break;
        passive[entering] = 1;

        while (true) {
            if (++iterations > iteration_limit) {
                *info = 3;
                goto finish;
            }

            CBLAS_INT active_count = 0;
            for (CBLAS_INT column = 0; column < n; ++column) {
                candidate[column] = 0.0;
                if (!passive[column])
                    continue;
                active_to_original[active_count] = column;
                for (CBLAS_INT row = 0; row < m; ++row) {
                    active_matrix[row + active_count * m] =
                        a[row + column * m];
                }
                ++active_count;
            }
            for (CBLAS_INT row = 0; row < m; ++row)
                active_rhs[row] = b[row];
            for (CBLAS_INT row = m; row < active_count; ++row)
                active_rhs[row] = 0.0;
            for (CBLAS_INT column = 0; column < active_count; ++column)
                active_permutation[column] = 0;

            CBLAS_INT one = 1;
            CBLAS_INT least_squares_rows = m;
            CBLAS_INT leading_rhs = std::max(m, active_count);
            CBLAS_INT rank = 0;
            CBLAS_INT linear_info = 0;
            CBLAS_INT work_size = m * active_count + 3 * active_count + 1;
#if defined(VLK_NNLS_RCOND)
            double rank_tolerance = VLK_NNLS_RCOND;
#else
            double rank_tolerance = std::sqrt(DBL_EPSILON) / 32.0;
#endif
            BLAS_FUNC(dgelsy)(
                &least_squares_rows,
                &active_count,
                &one,
                active_matrix,
                &least_squares_rows,
                active_rhs,
                &leading_rhs,
                active_permutation,
                &rank_tolerance,
                &rank,
                least_squares_work,
                &work_size,
                &linear_info);
            if (linear_info != 0) {
                *info = 3;
                goto finish;
            }
            if (rank < active_count) {
                passive[entering] = 0;
                eligible[entering] = 0;
                break;
            }
            for (CBLAS_INT index = 0; index < active_count; ++index)
                candidate[active_to_original[index]] = active_rhs[index];

            bool all_positive = true;
            for (CBLAS_INT column = 0; column < n; ++column) {
                if (passive[column] && candidate[column] <= 0.0) {
                    all_positive = false;
                    break;
                }
            }
            if (all_positive) {
                for (CBLAS_INT column = 0; column < n; ++column)
                    x[column] = candidate[column];
                break;
            }

            double step = 1.0;
            for (CBLAS_INT column = 0; column < n; ++column) {
                if (!passive[column] || candidate[column] > 0.0)
                    continue;
                const double denominator = x[column] - candidate[column];
                if (denominator > 0.0)
                    step = std::min(step, x[column] / denominator);
            }
            bool entering_removed = false;
            bool previous_member_removed = false;
            for (CBLAS_INT column = 0; column < n; ++column) {
                x[column] += step * (candidate[column] - x[column]);
                if (passive[column] && x[column] <= 0.0) {
                    x[column] = 0.0;
                    passive[column] = 0;
                    entering_removed |= column == entering;
                    previous_member_removed |= column != entering;
                }
            }
            if (previous_member_removed) {
                for (CBLAS_INT column = 0; column < n; ++column)
                    eligible[column] = 1;
            }
            if (entering_removed) {
                eligible[entering] = 0;
                break;
            }
        }
    }

finish:
    *rnorm = 0.0;
    for (CBLAS_INT row = 0; row < m; ++row) {
        double fitted = 0.0;
        for (CBLAS_INT column = 0; column < n; ++column)
            fitted += a[row + column * m] * x[column];
        residual[row] = b[row] - fitted;
        *rnorm += residual[row] * residual[row];
    }
    *rnorm = std::sqrt(*rnorm);
    for (CBLAS_INT column = 0; column < n; ++column) {
        double correlation = 0.0;
        for (CBLAS_INT row = 0; row < m; ++row)
            correlation += a[row + column * m] * residual[row];
        dual[column] = correlation;
    }
}

#undef VLK_STACK_ALLOCATE
#undef VLK_NNLS_RESTRICT
