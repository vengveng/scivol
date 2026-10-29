#include "kinks.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace scivolapi::optimize {

Stationarity kink_stationarity(const Problem& p, std::span<const double> x, KinkWorkspace& w)
{
    Stationarity s;
    const auto n = p.dimension, k = w.out.count;
    if (!w.out.supported || !k) return s;
    std::size_t count = 0;
    auto add = [&](const double* a, double sign, double lo, double hi) {
        for (std::size_t i = 0; i < n; ++i) w.columns[count*n+i] = sign*a[i];
        w.lower[count] = lo; w.upper[count] = hi;
        ++count;
    };
    for (std::size_t j = 0; j < k; ++j) {
        if (!(w.weights[j] >= 0) || !std::isfinite(w.weights[j])) return s;
        add(w.normals.data()+j*n, 1, -w.weights[j], w.weights[j]);
    }
    const double infinity = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        double sign = 0;
        if (!p.lower.empty() && x[i] == p.lower[i]) sign = -1;
        if (!p.upper.empty() && x[i] == p.upper[i]) sign = 1;
        if (!sign) continue;
        std::fill_n(w.columns.data()+count*n, n, 0.0);
        w.columns[count*n+i] = sign;
        w.lower[count] = 0; w.upper[count] = infinity; ++count;
    }
    if (p.constraint_count) {
        if (!p.constraints(x.data(), w.c.data(), w.jac.data())) return s;
        for (std::size_t j = 0; j < p.constraint_count; ++j)
            if (w.c[j] <= 64*std::numeric_limits<double>::epsilon())
                add(w.jac.data()+j*n, -1, 0, infinity);
    }
    std::copy(w.gradient.begin(), w.gradient.end(), w.residual.begin());
    std::fill_n(w.coefficients.data(), count, 0.0);
    for (int sweep = 0; sweep < 4096; ++sweep) {
        double largest = 0;
        for (std::size_t j = 0; j < count; ++j) {
            const double* a = w.columns.data()+j*n;
            double norm = 0, product = 0;
            for (std::size_t i = 0; i < n; ++i) { norm += a[i]*a[i]; product += a[i]*w.residual[i]; }
            if (!(norm > 0)) continue;
            const double next = std::clamp(w.coefficients[j]-product/norm, w.lower[j], w.upper[j]);
            const double delta = next-w.coefficients[j];
            for (std::size_t i = 0; i < n; ++i) w.residual[i] += delta*a[i];
            largest = std::max(largest, std::abs(delta)*std::sqrt(norm));
            w.coefficients[j] = next;
        }
        if (largest <= 8*std::numeric_limits<double>::epsilon()) break;
    }
    // Recompute the certificate to avoid accumulating coordinate-update error.
    s.residual = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double r = w.gradient[i];
        for (std::size_t j = 0; j < count; ++j) r += w.coefficients[j]*w.columns[j*n+i];
        if (!std::isfinite(r)) return Stationarity{};
        s.residual = std::max(s.residual, std::abs(r));
    }
    for (std::size_t j = k; j < count; ++j) s.binding |= w.coefficients[j] > 0;
    return s;
}

} // namespace scivolapi::optimize
