#pragma once

#include <cstddef>

#include "layout.hpp"

namespace scivolapi::detail::correlation {

// Q_t = (1 - sum a - sum b) Qbar + sum_{lag <= p_t} a_lag e_{t-lag} e_{t-lag}' + sum_{lag <= q_t} b_lag Q_{t-lag},
// with the available lags p_t, q_t and accessors eps(lag), q(lag) for the past rows.
// The single definition of the recursion's value, used by the jet and by
// simulated continuations.
template<class Eps, class Lagged>
void q_step(const Layout& layout, const double* theta, const double* target, std::size_t n, std::size_t p_t,
            std::size_t q_t, const Eps& eps, const Lagged& lagged, double* q) noexcept
{
    double slack = 1.0;
    for (std::size_t a = 0; a < layout.coefficients; ++a) slack -= theta[a];
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) q[i * n + j] = slack * target[i * n + j];
    for (std::size_t lag = 1; lag <= p_t; ++lag) {
        const double* e = eps(lag);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j <= i; ++j) q[i * n + j] += theta[lag - 1] * (e[i] * e[j]);
    }
    for (std::size_t lag = 1; lag <= q_t; ++lag) {
        const double* previous = lagged(lag);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j <= i; ++j)
                q[i * n + j] += theta[layout.p + lag - 1] * previous[i * n + j];
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) q[j * n + i] = q[i * n + j];
}

}  // namespace scivolapi::detail::correlation
