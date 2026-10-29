#include "test_support.hpp"
#include "optimize/solver_workspace.hpp"

int main()
{
    for (double scale : {1e-310, 1e-100, 1.0, 1e308}) {
        for (double sign : {-1.0, 1.0}) {
            int n = 2, one = 1, info = 0;
            double alpha = sign * scale, tail = scale, tau;
            BLAS_FUNC(dlarfgp)(&n, &alpha, &tail, &one, &tau);
            close(tau, 1 - sign / std::sqrt(2.0), 3e-15);
            assert(std::isfinite(alpha));
            close(tail, -1 / (std::sqrt(2.0) * tau), 3e-15);
            double matrix[]{sign * scale, scale}, work[8], lwork = 8;
            BLAS_FUNC(dgeqrf)(&n, &one, matrix, &n, &tau, work, &lwork, &info);
            assert(info == 0 && std::isfinite(matrix[0]));
            close(tau, 1 + 1 / std::sqrt(2.0), 3e-15);
        }
    }
    for (double scale : {1e-100, 1e-12, 1.0, 1e100}) {
        int n = 2, one = 1, rank = 0, info = 0, lwork = 32, permutation[2]{};
        double a[]{scale, 0, 0, scale}, b[]{scale, 2 * scale}, work[32], rcond = 1e-10;
        BLAS_FUNC(dgelsy)(&n, &n, &one, a, &n, b, &n, permutation, &rcond, &rank, work, &lwork, &info);
        assert(info == 0 && rank == 2); close(b[0], 1); close(b[1], 2);
    }

    int n = 2, one = 1;
    double zero = 0, two = 2, a[]{1, 2, 3, 4}, x[]{2, 3}, y[]{NAN, NAN};
    BLAS_FUNC(dgemv)("N", &n, &n, &two, a, &n, x, &one, &zero, y, &one);
    close(y[0], 22); close(y[1], 32);
    BLAS_FUNC(dgemv)("T", &n, &n, &zero, nullptr, &n, nullptr, &one, &zero, y, &one);
    close(y[0], 0); close(y[1], 0);
    BLAS_FUNC(daxpy)(&n, &zero, nullptr, &one, nullptr, &one);
    y[0] = y[1] = NAN;
    BLAS_FUNC(dscal)(&n, &zero, y, &one); close(y[0], 0); close(y[1], 0);
    double b[]{NAN, NAN, NAN, NAN};
    BLAS_FUNC(dtrsm)("R", "U", "N", "N", &n, &n, &zero, nullptr, &n, b, &n);
    for (double v : b) close(v, 0);

    // Exact separable primal steps remain valid at gradients that break LDP.
    scivolapi::optimize::detail::SolverWorkspace w;
    assert(w.allocate(2, 0));
    double factor[]{1, 0, 2}, g[]{1e15, -1e15}, lo[]{-1, -1}, hi[]{1, 1}, d[2];
    std::int64_t mode;
    scivolapi_opt_lsq(2, 0, 0, factor, g, w.normals, w.values, lo, hi, d,
        w.multipliers, w.buffer, w.indices, &mode, w.nnls);
    assert(mode == 1); close(d[0], -1); close(d[1], 1);

    // Scratch exceeds the removed 128 KiB cap. Duplicate columns test rank loss.
    const int rows = 72, columns = 144;
    std::vector<double> matrix(rows * columns), rhs(rows), solution(columns), dual(columns), residual(rows);
    std::vector<int> passive(columns);
    for (int i = 0; i < rows; ++i) {
        matrix[i + i * rows] = matrix[i + (i + rows) * rows] = 1;
        rhs[i] = (i % 2 ? -1 : 1) * (i + 1) * 0.125;
    }
    const auto bytes = scivolapi::optimize::nnls::workspace_bytes(rows, columns);
    assert(bytes > 128 * 1024);
    std::vector<double> arena((bytes + 7) / 8 + 2, 919);
    double norm;
    __nnls(rows, columns, matrix.data(), rhs.data(), solution.data(), dual.data(), residual.data(),
        passive.data(), 0, &norm, &mode, {arena.data() + 1, bytes});
    assert(mode == 1 && arena.front() == 919 && arena.back() == 919);
    for (int i = 0; i < rows; ++i) close(solution[i] + solution[i + rows], std::max(0.0, rhs[i]));
    for (int i = 0; i < columns; ++i) {
        assert(solution[i] >= 0 && dual[i] <= 1e-12);
        assert(std::abs(solution[i] * dual[i]) < 1e-11);
    }
}
