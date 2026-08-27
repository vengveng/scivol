// SCIVOL-NATIVE: CORE
#define BLAS_SYMBOL_PREFIX vlk_
#define NO_APPEND_FORTRAN

extern "C" {
#include "third_party/scipy_slsqp/blaslapack_declarations.h"
}

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace {

[[nodiscard]] bool option(const char* value, char expected)
{
    return value != nullptr
        && (*value == expected || *value == expected + ('a' - 'A'));
}

[[nodiscard]] std::ptrdiff_t vector_start(CBLAS_INT size, CBLAS_INT increment)
{
    return increment >= 0
        ? 0
        : static_cast<std::ptrdiff_t>(1 - size) * increment;
}

[[nodiscard]] std::size_t lower_packed_index(
    CBLAS_INT row,
    CBLAS_INT column,
    CBLAS_INT dimension)
{
    return static_cast<std::size_t>(
        column * dimension - column * (column - 1) / 2 + row - column);
}

double stable_norm(CBLAS_INT size, const double* values, CBLAS_INT increment)
{
    double scale = 0.0;
    double sum_squares = 1.0;
    std::ptrdiff_t index = vector_start(size, increment);
    for (CBLAS_INT item = 0; item < size; ++item, index += increment) {
        const double magnitude = std::abs(values[index]);
        if (magnitude == 0.0)
            continue;
        if (scale < magnitude) {
            const double ratio = scale / magnitude;
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = magnitude;
        } else {
            const double ratio = magnitude / scale;
            sum_squares += ratio * ratio;
        }
    }
    return scale == 0.0 ? 0.0 : scale * std::sqrt(sum_squares);
}

void standard_householder(
    CBLAS_INT size,
    double* alpha,
    double* tail,
    CBLAS_INT increment,
    double* tau)
{
    if (size <= 1) {
        *tau = 0.0;
        return;
    }
    const double tail_norm = stable_norm(size - 1, tail, increment);
    if (tail_norm == 0.0) {
        *tau = 0.0;
        return;
    }
    const double beta = -std::copysign(std::hypot(*alpha, tail_norm), *alpha);
    *tau = (beta - *alpha) / beta;
    const double inverse = 1.0 / (*alpha - beta);
    std::ptrdiff_t index = vector_start(size - 1, increment);
    for (CBLAS_INT item = 0; item < size - 1; ++item, index += increment)
        tail[index] *= inverse;
    *alpha = beta;
}

}  // namespace

extern "C" {

void BLAS_FUNC(daxpy)(
    CBLAS_INT* n,
    double* alpha,
    double* x,
    CBLAS_INT* incx,
    double* y,
    CBLAS_INT* incy)
{
    std::ptrdiff_t x_index = vector_start(*n, *incx);
    std::ptrdiff_t y_index = vector_start(*n, *incy);
    for (CBLAS_INT item = 0; item < *n; ++item) {
        y[y_index] += *alpha * x[x_index];
        x_index += *incx;
        y_index += *incy;
    }
}

void BLAS_FUNC(dscal)(CBLAS_INT* n, double* alpha, double* x, CBLAS_INT* incx)
{
    std::ptrdiff_t index = vector_start(*n, *incx);
    for (CBLAS_INT item = 0; item < *n; ++item, index += *incx)
        x[index] *= *alpha;
}

void BLAS_FUNC(dcopy)(
    CBLAS_INT* n,
    double* x,
    CBLAS_INT* incx,
    double* y,
    CBLAS_INT* incy)
{
    std::ptrdiff_t x_index = vector_start(*n, *incx);
    std::ptrdiff_t y_index = vector_start(*n, *incy);
    for (CBLAS_INT item = 0; item < *n; ++item) {
        y[y_index] = x[x_index];
        x_index += *incx;
        y_index += *incy;
    }
}

double BLAS_FUNC(dnrm2)(CBLAS_INT* n, double* x, CBLAS_INT* incx)
{
    return stable_norm(*n, x, *incx);
}

double BLAS_FUNC(ddot)(
    CBLAS_INT* n,
    double* x,
    CBLAS_INT* incx,
    double* y,
    CBLAS_INT* incy)
{
    double result = 0.0;
    std::ptrdiff_t x_index = vector_start(*n, *incx);
    std::ptrdiff_t y_index = vector_start(*n, *incy);
    for (CBLAS_INT item = 0; item < *n; ++item) {
        result += x[x_index] * y[y_index];
        x_index += *incx;
        y_index += *incy;
    }
    return result;
}

void BLAS_FUNC(dgemv)(
    char* trans,
    CBLAS_INT* m,
    CBLAS_INT* n,
    double* alpha,
    double* a,
    CBLAS_INT* lda,
    double* x,
    CBLAS_INT* incx,
    double* beta,
    double* y,
    CBLAS_INT* incy)
{
    const bool transpose = option(trans, 'T');
    const CBLAS_INT output_size = transpose ? *n : *m;
    std::ptrdiff_t y_index = vector_start(output_size, *incy);
    for (CBLAS_INT item = 0; item < output_size; ++item, y_index += *incy)
        y[y_index] *= *beta;

    if (!transpose) {
        std::ptrdiff_t x_index = vector_start(*n, *incx);
        for (CBLAS_INT column = 0; column < *n; ++column, x_index += *incx) {
            const double factor = *alpha * x[x_index];
            y_index = vector_start(*m, *incy);
            for (CBLAS_INT row = 0; row < *m; ++row, y_index += *incy)
                y[y_index] += factor * a[row + column * *lda];
        }
    } else {
        std::ptrdiff_t output = vector_start(*n, *incy);
        for (CBLAS_INT column = 0; column < *n; ++column, output += *incy) {
            double dot = 0.0;
            std::ptrdiff_t x_index = vector_start(*m, *incx);
            for (CBLAS_INT row = 0; row < *m; ++row, x_index += *incx)
                dot += a[row + column * *lda] * x[x_index];
            y[output] += *alpha * dot;
        }
    }
}

void BLAS_FUNC(dtpmv)(
    char* uplo,
    char* trans,
    char* diag,
    CBLAS_INT* n,
    double* packed,
    double* x,
    CBLAS_INT* incx)
{
    if (!option(uplo, 'L') || *incx != 1)
        return;
    const bool unit = option(diag, 'U');
    if (option(trans, 'T')) {
        for (CBLAS_INT row = 0; row < *n; ++row) {
            double value = unit
                ? x[row]
                : packed[lower_packed_index(row, row, *n)] * x[row];
            for (CBLAS_INT column = row + 1; column < *n; ++column) {
                value += packed[lower_packed_index(column, row, *n)]
                    * x[column];
            }
            x[row] = value;
        }
    } else {
        for (CBLAS_INT row = *n - 1; row >= 0; --row) {
            double value = unit
                ? x[row]
                : packed[lower_packed_index(row, row, *n)] * x[row];
            for (CBLAS_INT column = 0; column < row; ++column) {
                value += packed[lower_packed_index(row, column, *n)]
                    * x[column];
            }
            x[row] = value;
        }
    }
}

void BLAS_FUNC(dtpsv)(
    char* uplo,
    char* trans,
    char* diag,
    CBLAS_INT* n,
    double* packed,
    double* x,
    CBLAS_INT* incx)
{
    if (!option(uplo, 'L') || option(trans, 'T') || *incx != 1)
        return;
    const bool unit = option(diag, 'U');
    for (CBLAS_INT row = 0; row < *n; ++row) {
        double value = x[row];
        for (CBLAS_INT column = 0; column < row; ++column) {
            value -= packed[lower_packed_index(row, column, *n)] * x[column];
        }
        if (!unit)
            value /= packed[lower_packed_index(row, row, *n)];
        x[row] = value;
    }
}

void BLAS_FUNC(dtrsv)(
    char* uplo,
    char* trans,
    char* diag,
    CBLAS_INT* n,
    double* a,
    CBLAS_INT* lda,
    double* x,
    CBLAS_INT* incx)
{
    if (!option(uplo, 'U') || *incx != 1)
        return;
    const bool unit = option(diag, 'U');
    if (!option(trans, 'T')) {
        for (CBLAS_INT row = *n - 1; row >= 0; --row) {
            double value = x[row];
            for (CBLAS_INT column = row + 1; column < *n; ++column)
                value -= a[row + column * *lda] * x[column];
            x[row] = unit ? value : value / a[row + row * *lda];
        }
    } else {
        for (CBLAS_INT row = 0; row < *n; ++row) {
            double value = x[row];
            for (CBLAS_INT column = 0; column < row; ++column)
                value -= a[column + row * *lda] * x[column];
            x[row] = unit ? value : value / a[row + row * *lda];
        }
    }
}

void BLAS_FUNC(dtrsm)(
    char* side,
    char* uplo,
    char* trans,
    char* diag,
    CBLAS_INT* m,
    CBLAS_INT* n,
    double* alpha,
    double* a,
    CBLAS_INT* lda,
    double* b,
    CBLAS_INT* ldb)
{
    if (!option(side, 'R') || !option(uplo, 'U') || option(trans, 'T'))
        return;
    const bool unit = option(diag, 'U');
    for (CBLAS_INT row = 0; row < *m; ++row) {
        for (CBLAS_INT column = 0; column < *n; ++column) {
            double value = *alpha * b[row + column * *ldb];
            for (CBLAS_INT prior = 0; prior < column; ++prior) {
                value -= b[row + prior * *ldb] * a[prior + column * *lda];
            }
            b[row + column * *ldb] = unit
                ? value
                : value / a[column + column * *lda];
        }
    }
}

void BLAS_FUNC(dlarfgp)(
    CBLAS_INT* n,
    double* alpha,
    double* x,
    CBLAS_INT* incx,
    double* tau)
{
    if (*n <= 0) {
        *tau = 0.0;
        return;
    }
    const double x_norm = stable_norm(*n - 1, x, *incx);
    const double epsilon = std::numeric_limits<double>::epsilon();
    if (x_norm <= epsilon * std::abs(*alpha)) {
        if (*alpha >= 0.0) {
            *tau = 0.0;
        } else {
            *tau = 2.0;
            std::ptrdiff_t index = vector_start(*n - 1, *incx);
            for (CBLAS_INT item = 0; item < *n - 1; ++item, index += *incx)
                x[index] = 0.0;
            *alpha = -*alpha;
        }
        return;
    }

    double beta = std::copysign(std::hypot(*alpha, x_norm), *alpha);
    const double saved_alpha = *alpha;
    double scale;
    *alpha += beta;
    if (beta < 0.0) {
        beta = -beta;
        *tau = -*alpha / beta;
        scale = 1.0 / *alpha;
    } else {
        const double cancellation_safe = x_norm * (x_norm / *alpha);
        *tau = cancellation_safe / beta;
        scale = -1.0 / cancellation_safe;
    }
    if (std::abs(*tau) <= std::numeric_limits<double>::min()) {
        if (saved_alpha >= 0.0) {
            *tau = 0.0;
        } else {
            *tau = 2.0;
            scale = 0.0;
            beta = -saved_alpha;
        }
    }
    std::ptrdiff_t index = vector_start(*n - 1, *incx);
    for (CBLAS_INT item = 0; item < *n - 1; ++item, index += *incx)
        x[index] *= scale;
    *alpha = beta;
}

void BLAS_FUNC(dlartgp)(
    double* f,
    double* g,
    double* cosine,
    double* sine,
    double* radius)
{
    if (*g == 0.0) {
        *cosine = std::copysign(1.0, *f);
        *sine = 0.0;
        *radius = std::abs(*f);
        return;
    }
    if (*f == 0.0) {
        *cosine = 0.0;
        *sine = std::copysign(1.0, *g);
        *radius = std::abs(*g);
        return;
    }
    *radius = std::hypot(*f, *g);
    *cosine = *f / *radius;
    *sine = *g / *radius;
}

void BLAS_FUNC(dlarf)(
    char* side,
    CBLAS_INT* m,
    CBLAS_INT* n,
    double* vector,
    CBLAS_INT* increment,
    double* tau,
    double* matrix,
    CBLAS_INT* leading_dimension,
    double* work)
{
    if (*tau == 0.0 || *m == 0 || *n == 0)
        return;
    if (option(side, 'L')) {
        for (CBLAS_INT column = 0; column < *n; ++column) {
            double dot = 0.0;
            std::ptrdiff_t vector_index = vector_start(*m, *increment);
            for (CBLAS_INT row = 0; row < *m; ++row, vector_index += *increment)
                dot += vector[vector_index] * matrix[row + column * *leading_dimension];
            work[column] = dot;
        }
        for (CBLAS_INT column = 0; column < *n; ++column) {
            std::ptrdiff_t vector_index = vector_start(*m, *increment);
            for (CBLAS_INT row = 0; row < *m; ++row, vector_index += *increment) {
                matrix[row + column * *leading_dimension] -=
                    *tau * vector[vector_index] * work[column];
            }
        }
    } else {
        for (CBLAS_INT row = 0; row < *m; ++row) {
            double dot = 0.0;
            std::ptrdiff_t vector_index = vector_start(*n, *increment);
            for (CBLAS_INT column = 0; column < *n;
                 ++column, vector_index += *increment) {
                dot += matrix[row + column * *leading_dimension]
                    * vector[vector_index];
            }
            work[row] = dot;
        }
        for (CBLAS_INT row = 0; row < *m; ++row) {
            std::ptrdiff_t vector_index = vector_start(*n, *increment);
            for (CBLAS_INT column = 0; column < *n;
                 ++column, vector_index += *increment) {
                matrix[row + column * *leading_dimension] -=
                    *tau * work[row] * vector[vector_index];
            }
        }
    }
}

void BLAS_FUNC(dgeqr2)(
    CBLAS_INT* m,
    CBLAS_INT* n,
    double* a,
    CBLAS_INT* lda,
    double* tau,
    double* work,
    CBLAS_INT* info)
{
    *info = 0;
    const CBLAS_INT reflectors = std::min(*m, *n);
    CBLAS_INT one = 1;
    for (CBLAS_INT index = 0; index < reflectors; ++index) {
        CBLAS_INT length = *m - index;
        standard_householder(
            length,
            &a[index + index * *lda],
            &a[index + 1 + index * *lda],
            one,
            &tau[index]);
        if (index + 1 < *n) {
            const double diagonal = a[index + index * *lda];
            a[index + index * *lda] = 1.0;
            CBLAS_INT columns = *n - index - 1;
            BLAS_FUNC(dlarf)(
                const_cast<char*>("L"),
                &length,
                &columns,
                &a[index + index * *lda],
                &one,
                &tau[index],
                &a[index + (index + 1) * *lda],
                lda,
                work);
            a[index + index * *lda] = diagonal;
        }
    }
}

void BLAS_FUNC(dgeqrf)(
    CBLAS_INT* m,
    CBLAS_INT* n,
    double* a,
    CBLAS_INT* lda,
    double* tau,
    double* work,
    double* lwork,
    CBLAS_INT* info)
{
    BLAS_FUNC(dgeqr2)(m, n, a, lda, tau, work + 1, info);
    work[0] = std::max(1.0, *lwork);
}

void BLAS_FUNC(dorm2r)(
    char* side,
    char* trans,
    CBLAS_INT* m,
    CBLAS_INT* n,
    CBLAS_INT* k,
    double* a,
    CBLAS_INT* lda,
    double* tau,
    double* c,
    CBLAS_INT* ldc,
    double* work,
    CBLAS_INT* info)
{
    *info = 0;
    const bool left = option(side, 'L');
    const bool no_transpose = option(trans, 'N');
    const bool forward = (left && !no_transpose) || (!left && no_transpose);
    const CBLAS_INT begin = forward ? 0 : *k - 1;
    const CBLAS_INT end = forward ? *k : -1;
    const CBLAS_INT step = forward ? 1 : -1;
    CBLAS_INT one = 1;
    for (CBLAS_INT index = begin; index != end; index += step) {
        CBLAS_INT rows = left ? *m - index : *m;
        CBLAS_INT columns = left ? *n : *n - index;
        double* block = left ? &c[index] : &c[index * *ldc];
        const double diagonal = a[index + index * *lda];
        a[index + index * *lda] = 1.0;
        BLAS_FUNC(dlarf)(
            side,
            &rows,
            &columns,
            &a[index + index * *lda],
            &one,
            &tau[index],
            block,
            ldc,
            work);
        a[index + index * *lda] = diagonal;
    }
}

void BLAS_FUNC(dgerq2)(
    CBLAS_INT* m,
    CBLAS_INT* n,
    double* a,
    CBLAS_INT* lda,
    double* tau,
    double* work,
    CBLAS_INT* info)
{
    *info = 0;
    const CBLAS_INT reflectors = std::min(*m, *n);
    for (CBLAS_INT index = reflectors - 1; index >= 0; --index) {
        const CBLAS_INT row = *m - reflectors + index;
        CBLAS_INT length = *n - reflectors + index + 1;
        standard_householder(
            length,
            &a[row + (length - 1) * *lda],
            &a[row],
            *lda,
            &tau[index]);
        if (row > 0) {
            const double diagonal = a[row + (length - 1) * *lda];
            a[row + (length - 1) * *lda] = 1.0;
            CBLAS_INT rows = row;
            BLAS_FUNC(dlarf)(
                const_cast<char*>("R"),
                &rows,
                &length,
                &a[row],
                lda,
                &tau[index],
                a,
                lda,
                work);
            a[row + (length - 1) * *lda] = diagonal;
        }
    }
}

void BLAS_FUNC(dormr2)(
    char* side,
    char* trans,
    CBLAS_INT* m,
    CBLAS_INT* n,
    CBLAS_INT* k,
    double* a,
    CBLAS_INT* lda,
    double* tau,
    double* c,
    CBLAS_INT* ldc,
    double* work,
    CBLAS_INT* info)
{
    *info = 0;
    const bool left = option(side, 'L');
    const bool no_transpose = option(trans, 'N');
    const bool forward = (left && !no_transpose) || (!left && no_transpose);
    const CBLAS_INT begin = forward ? 0 : *k - 1;
    const CBLAS_INT end = forward ? *k : -1;
    const CBLAS_INT step = forward ? 1 : -1;
    const CBLAS_INT orthogonal_dimension = left ? *m : *n;
    for (CBLAS_INT index = begin; index != end; index += step) {
        CBLAS_INT rows = left ? *m - *k + index + 1 : *m;
        CBLAS_INT columns = left ? *n : *n - *k + index + 1;
        const CBLAS_INT diagonal_column =
            orthogonal_dimension - *k + index;
        const double diagonal = a[index + diagonal_column * *lda];
        a[index + diagonal_column * *lda] = 1.0;
        BLAS_FUNC(dlarf)(
            side,
            &rows,
            &columns,
            &a[index],
            lda,
            &tau[index],
            c,
            ldc,
            work);
        a[index + diagonal_column * *lda] = diagonal;
    }
}

void BLAS_FUNC(dgelsy)(
    CBLAS_INT* m,
    CBLAS_INT* n,
    CBLAS_INT* nrhs,
    double* a,
    CBLAS_INT* lda,
    double* b,
    CBLAS_INT*,
    CBLAS_INT* permutation,
    double* rcond,
    CBLAS_INT* rank,
    double* work,
    CBLAS_INT*,
    CBLAS_INT* info)
{
    *info = 0;
    *rank = 0;
    if (*nrhs != 1 || *m < *n)
        return;

    double* triangular = work;
    double* projected_rhs = work + static_cast<std::size_t>(*n) * *n;
    double* solution = projected_rhs + *n;
    std::fill_n(triangular, static_cast<std::size_t>(*n) * *n, 0.0);
    for (CBLAS_INT column = 0; column < *n; ++column)
        permutation[column] = column;

    double first_diagonal = 0.0;
    for (CBLAS_INT pivot = 0; pivot < *n; ++pivot) {
        CBLAS_INT selected = pivot;
        double selected_norm = -1.0;
        for (CBLAS_INT column = pivot; column < *n; ++column) {
            const double norm = stable_norm(*m, &a[column * *lda], 1);
            if (norm > selected_norm) {
                selected = column;
                selected_norm = norm;
            }
        }
        if (selected != pivot) {
            for (CBLAS_INT row = 0; row < *m; ++row) {
                std::swap(
                    a[row + pivot * *lda],
                    a[row + selected * *lda]);
            }
            std::swap(permutation[pivot], permutation[selected]);
            for (CBLAS_INT row = 0; row < pivot; ++row) {
                std::swap(
                    triangular[row + pivot * *n],
                    triangular[row + selected * *n]);
            }
        }

        const double diagonal = stable_norm(*m, &a[pivot * *lda], 1);
        if (pivot == 0)
            first_diagonal = diagonal;
        const double tolerance = (*rcond >= 0.0 ? *rcond : 0.0)
            * std::max(first_diagonal, 1.0);
        if (!(diagonal > tolerance))
            break;
        triangular[pivot + pivot * *n] = diagonal;
        for (CBLAS_INT row = 0; row < *m; ++row)
            a[row + pivot * *lda] /= diagonal;

        double rhs_projection = 0.0;
        for (CBLAS_INT row = 0; row < *m; ++row)
            rhs_projection += a[row + pivot * *lda] * b[row];
        projected_rhs[pivot] = rhs_projection;

        for (CBLAS_INT column = pivot + 1; column < *n; ++column) {
            double projection = 0.0;
            for (CBLAS_INT row = 0; row < *m; ++row) {
                projection += a[row + pivot * *lda]
                    * a[row + column * *lda];
            }
            triangular[pivot + column * *n] = projection;
            for (CBLAS_INT row = 0; row < *m; ++row) {
                a[row + column * *lda] -=
                    projection * a[row + pivot * *lda];
            }
        }
        ++*rank;
    }

    std::fill_n(solution, *n, 0.0);
    for (CBLAS_INT row = *rank - 1; row >= 0; --row) {
        double value = projected_rhs[row];
        for (CBLAS_INT column = row + 1; column < *rank; ++column) {
            value -= triangular[row + column * *n] * solution[column];
        }
        solution[row] = value / triangular[row + row * *n];
    }
    std::fill_n(b, std::max(*m, *n), 0.0);
    for (CBLAS_INT column = 0; column < *n; ++column)
        b[permutation[column]] = solution[column];
}

}  // extern "C"
