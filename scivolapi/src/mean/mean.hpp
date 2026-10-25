#pragma once

#include <algorithm>
#include <cstddef>

#include <scivolapi/model.hpp>

#include "../detail/numeric.hpp"

// The conditional mean. Every supported mean is one ARMAX equation over the
// scored rows s = 0, 1, ...:
//
//   e_s = y_s - r_s' b - sum_{j<=min(Q,s)} theta_j e_{s-j},
//
// where r_s is the regressor row the Model prepared once per sample (constant,
// AR lags, HAR averages, exogenous columns) and pre-sample residuals are zero.
// With mean entries [b | theta] and the extended row v_s = [r_s, e_{s-1}, ..., e_{s-Q}],
//
//   E_s = de_s = -v_s - sum_j theta_j E_{s-j},
//   F_s = d2e_s = -sum_j (theta_j F_{s-j} + [a = theta_j] E_{s-j}[b] + [b = theta_j] E_{s-j}[a]),
//
// so without MA terms E_s = -r_s and F_s = 0: the residual is linear in the mean.
namespace scivolapi::detail {

// Residuals are the data.
struct NoMean {
    static constexpr bool present = false;
};

struct MeanWork {
    double* e;           // residuals (scored rows)
    double* e2;          // squared residuals
    std::size_t slots;   // ring slots, row s in slot s % slots
    double* de;          // slots * width
    double* dde;         // slots * packed(width), used with MA terms only
};

class ArmaxMean {
public:
    static constexpr bool present = true;

    explicit ArmaxMean(const MeanSpec& spec) noexcept : k_(spec.regressors()), q_(spec.ma) {}

    [[nodiscard]] std::size_t width() const noexcept { return k_ + q_; }
    [[nodiscard]] bool linear() const noexcept { return q_ == 0; }

    // Residual of scored row s, with E_s (order >= 1) and F_s (order >= 2) in the rings.
    void step(const double* theta, const Sample& sample, std::size_t s, int order, MeanWork& w) const noexcept
    {
        const double* r = sample.design.data() + s * k_;
        const double* ma = theta + k_;
        double e = sample.y[sample.first_scored + s];
        for (std::size_t k = 0; k < k_; ++k) e -= r[k] * theta[k];
        for (std::size_t j = 1, n = std::min(q_, s); j <= n; ++j) e -= ma[j - 1] * w.e[s - j];
        w.e[s] = e;
        w.e2[s] = e * e;
        if (order < 1) return;

        const std::size_t m = width(), pm = packed_size(m);
        double* de = w.de + (s % w.slots) * m;
        for (std::size_t k = 0; k < k_; ++k) de[k] = -r[k];
        for (std::size_t j = 1; j <= q_; ++j) de[k_ + j - 1] = j <= s ? -w.e[s - j] : 0.0;
        for (std::size_t j = 1, n = std::min(q_, s); j <= n; ++j) {
            const double* previous = w.de + ((s - j) % w.slots) * m;
            for (std::size_t a = 0; a < m; ++a) de[a] -= ma[j - 1] * previous[a];
        }
        if (order < 2 || linear()) return;

        double* dde = w.dde + (s % w.slots) * pm;
        std::fill_n(dde, pm, 0.0);
        for (std::size_t j = 1, n = std::min(q_, s); j <= n; ++j) {
            const std::size_t slot = (s - j) % w.slots, t = k_ + j - 1;
            const double* previous = w.dde + slot * pm;
            const double* first = w.de + slot * m;
            for (std::size_t k = 0; k < pm; ++k) dde[k] -= ma[j - 1] * previous[k];
            for (std::size_t a = 0; a < m; ++a) dde[a <= t ? packed(m, a, t) : packed(m, t, a)] -= first[a];
            dde[packed(m, t, t)] -= first[t];  // the diagonal receives both terms
        }
    }

private:
    std::size_t k_, q_;
};

}  // namespace scivolapi::detail
