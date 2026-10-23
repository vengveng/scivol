// SCIVOL-NATIVE: shared residual chart; integration candidate, no default routing.
#pragma once
#include "anchored_mean_chart.hpp"
#include "arma_residual_precision.hpp"

namespace vlk::experiment::mean_chart {
namespace mean = vlk::experiment::mean;
namespace rt = vlk::experiment::runtime;
namespace eq = vlk::experiment::generated::arma_garch_equation;
using Real = vlk::experiment::wide::Real;

enum class PreparedIntention { derivatives, value_basis };

enum class Status { unprepared, ready, invalid, rank_fallback, nonfinite };

struct MatrixLayout {
    std::size_t m = 0, packed = 0, k = 0, state = 0, history = 0;
    std::size_t anchors = 0, square = 0, first = 0, second = 0, bytes = 0;
    constexpr bool operator==(const MatrixLayout&) const = default;
};

template<int Order, class Layout, PreparedIntention Intention = PreparedIntention::derivatives,
         class Number = Real>
constexpr MatrixLayout make_matrix_layout(const Layout& l, std::size_t k) noexcept {
    static_assert(Order == 1 || Order == 2);
    MatrixLayout w;
    std::size_t pair = 0, count = 0, next = 0;
    if (!mean::valid_arma_mean_layout(l) || !k || k > l.ma_start) return w;
    if constexpr (Intention == PreparedIntention::derivatives)
        if (!rt::checked_sum(l.mean_parameters, 1, next)
            || !rt::checked_product(l.mean_parameters, next, pair)) return w;
    w.m = Intention == PreparedIntention::value_basis ? k : l.mean_parameters;
    w.packed = Intention == PreparedIntention::value_basis ? 0 : pair / 2; w.k = k;
    if (!rt::checked_sum(1, w.m, w.state)
        || (Order == 2 && !rt::checked_sum(w.state, w.packed, w.state))
        || !rt::checked_product(l.q_ma, w.state, w.history)
        || !rt::checked_product(k, w.state, w.anchors)
        || !rt::checked_product(k, k, w.square)
        || (Intention != PreparedIntention::value_basis && !rt::checked_product(k, w.m, w.first))
        || (Order == 2 && !rt::checked_product(k, w.packed, w.second))) return {};
    rt::WorkspaceSizer s;
    for (auto n : {w.history, w.state, w.anchors, w.square, w.first, w.second, k})
        if (!rt::checked_sum(count, n, count)) return {};
    if (!s.add<Number>(count) || !s.add<std::size_t>(k)) return {};
    w.bytes = s.size();
    return w;
}

// Selected theta slots become rho slots in the same full-width coordinate array.
template<int Order, class Layout, const Layout* Fixed = nullptr,
         std::size_t FixedK = mean::dynamic,
         PreparedIntention Intention = PreparedIntention::derivatives,
         class PrimalPlan = mean::ArmaResidualPrecision<Layout, Fixed>, class Number = Real>
class ResidualPreservingChart {
    static_assert(Order == 1 || Order == 2);
    static_assert(Intention != PreparedIntention::value_basis || Order == 1);
    using Primal = PrimalPlan;
    using Real = Number;
    const Layout& supplied_;
    const double* theta_;
    const double* data_;
    const std::size_t *anchors_, *selected_;
    std::size_t n_, k_, next_ = 0;
    Primal* primal_;
    Real *history_ = nullptr, *current_ = nullptr, *rows_ = nullptr, *lu_ = nullptr;
    Real *map_g_ = nullptr, *map_h_ = nullptr, *rhs_ = nullptr;
    std::size_t* swaps_ = nullptr;
    Status status_ = Status::unprepared;
    double pivot_ratio_ = 0;
    double raw_identity_first_ = 0, raw_identity_second_ = 0;

    static constexpr MatrixLayout fixed_matrix = [] {
        if constexpr (Fixed != nullptr && FixedK != mean::dynamic)
            return make_matrix_layout<Order, Layout, Intention, Number>(*Fixed, FixedK);
        return MatrixLayout{};
    }();
    const Layout& layout() const noexcept { return mean::arma_layout<Layout, Fixed>(supplied_); }
    MatrixLayout matrix() const noexcept {
        if constexpr (Fixed != nullptr && FixedK != mean::dynamic) return fixed_matrix;
        return make_matrix_layout<Order, Layout, Intention, Number>(layout(), mean::extent<FixedK>(k_));
    }
    static bool finite(Real x) noexcept { return finite_sensitivity(x); }
    static double magnitude(Real x) noexcept { return std::abs(static_cast<double>(x)); }
    std::size_t packed(std::size_t i, std::size_t j) const noexcept {
        if (i > j) std::swap(i, j);
        return eq::packed_index(matrix().m, i, j);
    }
    bool selected(std::size_t j) const noexcept {
        for (std::size_t r = 0; r < matrix().k; ++r) if (selected_[r] == j) return true;
        return false;
    }
    void rewind() noexcept {
        std::fill_n(history_, matrix().history, Real(0));
        primal_->rewind(); next_ = layout().maximum_lag;
    }
    void retain(std::size_t time) noexcept {
        if (layout().q_ma)
            std::copy_n(current_, matrix().state,
                        history_ + (time % layout().q_ma) * matrix().state);
    }

    template<bool Chart>
    void advance_state(std::size_t time) noexcept {
        const auto w = matrix();
        std::fill_n(current_, w.state, Real(0));
        current_[0] = mean_replay::lift<Real>(primal_->advance(time));
        auto* g = current_ + 1;
        auto* h = g + w.m;
        if constexpr (Intention == PreparedIntention::value_basis) {
            static_assert(!Chart);
            mean::ArmaDesign<Fixed != nullptr ? Fixed->p_ar : mean::dynamic> design{data_, layout().p_ar};
            // Only selected linear forcing columns pass through the shared inverse-MA equation.
            for (std::size_t r = 0; r < w.k; ++r)
                g[r] = mean::inverse_ma_value<Real, Layout, Fixed>(
                    Real(-design.at(time, selected_[r])), layout(), theta_, [&](std::size_t lag) {
                        return history_[((time-1-lag) % layout().q_ma) * w.state+1+r];
                    });
        } else {
            if constexpr (!Chart) {
                eq::initialize_residual_first(g, w.m, 0);
                for (std::size_t lag = 0; lag < layout().p_ar; ++lag)
                    eq::add_ar_first(g, layout().ar_start + lag, data_[time-1-lag]);
            } else {
                mean::ArmaDesign<Fixed != nullptr ? Fixed->p_ar : mean::dynamic> design{data_, layout().p_ar};
                for (std::size_t j = 0; j < design.columns(); ++j)
                    if (!selected(j)) eq::add_ar_first(g, j, design.at(time, j));
                for (std::size_t r = 0; r < w.k; ++r) {
                    const double feature = design.at(time, selected_[r]);
                    for (std::size_t i = 0; i < w.m; ++i) g[i] -= feature * map_g_[r*w.m+i];
                    if constexpr (Order == 2)
                        for (std::size_t i = 0; i < w.packed; ++i) h[i] -= feature * map_h_[r*w.packed+i];
                }
            }
            for (std::size_t lag = 0; lag < layout().q_ma; ++lag) {
                const auto ma = layout().ma_start + lag;
                const auto* previous = history_ + ((time-1-lag) % layout().q_ma) * w.state;
                eq::add_ma_first(g, w.m, ma, theta_[ma], previous[0], previous+1);
                if constexpr (Order == 2)
                    eq::add_ma_second(h, w.m, ma, theta_[ma], previous+1, previous+1+w.m);
            }
        }
    }

    bool factor() noexcept {
        const auto w = matrix();
        double scale = 0, smallest = INFINITY;
        for (std::size_t r = 0; r < w.k; ++r)
            for (std::size_t c = 0; c < w.k; ++c) {
                lu_[r*w.k+c] = -rows_[r*w.state+1+
                    (Intention == PreparedIntention::value_basis ? c : selected_[c])];
                if (!finite(lu_[r*w.k+c])) return false;
                scale = std::max(scale, magnitude(lu_[r*w.k+c]));
            }
        // This is chart rejection only. It never changes the statistical domain.
        const double cutoff = 64 * std::numeric_limits<double>::epsilon() * scale;
        for (std::size_t j = 0; j < w.k; ++j) {
            std::size_t pivot = j;
            for (std::size_t i = j+1; i < w.k; ++i)
                if (magnitude(lu_[i*w.k+j]) > magnitude(lu_[pivot*w.k+j])) pivot = i;
            const double diagonal = magnitude(lu_[pivot*w.k+j]);
            if (!(diagonal > cutoff)) return false;
            smallest = std::min(smallest, diagonal);
            swaps_[j] = pivot;
            for (std::size_t c = 0; c < w.k; ++c) std::swap(lu_[j*w.k+c], lu_[pivot*w.k+c]);
            for (std::size_t i = j+1; i < w.k; ++i) {
                lu_[i*w.k+j] /= lu_[j*w.k+j];
                for (std::size_t c = j+1; c < w.k; ++c)
                    lu_[i*w.k+c] -= lu_[i*w.k+j] * lu_[j*w.k+c];
            }
        }
        pivot_ratio_ = smallest / scale;
        return std::all_of(lu_, lu_+w.square, finite);
    }
    void solve() noexcept {
        const auto w = matrix();
        for (std::size_t j = 0; j < w.k; ++j) std::swap(rhs_[j], rhs_[swaps_[j]]);
        for (std::size_t i = 0; i < w.k; ++i)
            for (std::size_t j = 0; j < i; ++j) rhs_[i] -= lu_[i*w.k+j] * rhs_[j];
        for (std::size_t i = w.k; i-- > 0;) {
            for (std::size_t j = i+1; j < w.k; ++j) rhs_[i] -= lu_[i*w.k+j] * rhs_[j];
            rhs_[i] /= lu_[i*w.k+i];
        }
    }
    bool inverse_derivatives() noexcept {
        const auto w = matrix();
        for (std::size_t i = 0; i < w.m; ++i) {
            for (std::size_t r = 0; r < w.k; ++r)
                rhs_[r] = (selected(i) ? Real(0) : rows_[r*w.state+1+i])
                    - Real(i == selected_[r] ? 1 : 0);
            solve();
            for (std::size_t r = 0; r < w.k; ++r) map_g_[r*w.m+i] = rhs_[r];
        }
        if constexpr (Order == 2)
            for (std::size_t i = 0; i < w.m; ++i)
                for (std::size_t j = i; j < w.m; ++j) {
                    const auto ij = packed(i,j);
                    for (std::size_t r = 0; r < w.k; ++r) {
                        const auto* eh = rows_ + r*w.state+1+w.m;
                        rhs_[r] = selected(i) || selected(j) ? Real(0) : eh[ij];
                        for (std::size_t c = 0; c < w.k; ++c) {
                            const Real bi = selected(i) ? Real(0) : -eh[packed(selected_[c],i)];
                            const Real bj = selected(j) ? Real(0) : -eh[packed(selected_[c],j)];
                            rhs_[r] -= bi * map_g_[c*w.m+j] + bj * map_g_[c*w.m+i];
                        }
                    }
                    solve();
                    for (std::size_t r = 0; r < w.k; ++r) map_h_[r*w.packed+ij] = rhs_[r];
                }
        return std::all_of(map_g_, map_g_+w.first, finite)
            && (Order != 2 || std::all_of(map_h_, map_h_+w.second, finite));
    }

public:
    ResidualPreservingChart(const Layout& l, const double* theta, const double* data,
        std::size_t n, const std::size_t* anchors, const std::size_t* selected,
        std::size_t k, Primal* primal) noexcept
        : supplied_(l), theta_(theta), data_(data), anchors_(anchors), selected_(selected),
          n_(n), k_(k), primal_(primal) {}

    static constexpr MatrixLayout workspace_layout(const Layout& l, std::size_t k) noexcept {
        if constexpr (Fixed != nullptr && FixedK != mean::dynamic) return fixed_matrix;
        return make_matrix_layout<Order, Layout, Intention, Number>(mean::arma_layout<Layout,Fixed>(l), mean::extent<FixedK>(k));
    }
    static constexpr std::size_t workspace_bytes(const Layout& l, std::size_t k) noexcept {
        return workspace_layout(l,k).bytes;
    }
    bool prepare(void* memory, std::size_t bytes) noexcept {
        status_ = Status::invalid;
        raw_identity_first_ = raw_identity_second_ = 0;
        const auto w = matrix();
        if (!w.bytes || !memory || bytes < w.bytes || !theta_ || !data_ || !anchors_
            || !selected_ || !primal_ || !primal_->ready() || k_ != w.k
            || reinterpret_cast<std::uintptr_t>(memory) % alignof(Real)) return false;
        std::size_t last = layout().maximum_lag;
        for (std::size_t r = 0; r < w.k; ++r) {
            if (anchors_[r] < layout().maximum_lag || anchors_[r] >= n_
                || selected_[r] >= layout().ma_start) return false;
            for (std::size_t c = 0; c < r; ++c)
                if (anchors_[r] == anchors_[c] || selected_[r] == selected_[c]) return false;
            last = std::max(last, anchors_[r]);
        }
        rt::WorkspaceArena arena(memory,bytes);
        history_ = take_sensitivities<Real>(arena,w.history); current_ = take_sensitivities<Real>(arena,w.state);
        rows_ = take_sensitivities<Real>(arena,w.anchors); lu_ = take_sensitivities<Real>(arena,w.square);
        map_g_ = take_sensitivities<Real>(arena,w.first); map_h_ = take_sensitivities<Real>(arena,w.second);
        rhs_ = take_sensitivities<Real>(arena,w.k); swaps_ = take_sensitivities<std::size_t>(arena,w.k);
        if (!swaps_) return false;
        rewind();
        for (std::size_t t = layout().maximum_lag; t <= last; ++t) {
            advance_state<false>(t);
            if (!std::all_of(current_,current_+w.state,finite)) {
                status_ = Status::nonfinite; return false;
            }
            for (std::size_t r = 0; r < w.k; ++r)
                if (t == anchors_[r]) std::copy_n(current_,w.state,rows_+r*w.state);
            retain(t);
        }
        if (!factor()) { status_ = Status::rank_fallback; return false; }
        if constexpr (Intention == PreparedIntention::derivatives)
            if (!inverse_derivatives()) { status_ = Status::nonfinite; return false; }
        rewind(); status_ = Status::ready;
        return true;
    }

    // Solve the caller's requested RHS. No inverse matrix or unused map derivatives.
    bool solve_basis(Real* values) noexcept {
        static_assert(Intention == PreparedIntention::value_basis);
        const auto w = matrix();
        if (status_ != Status::ready || !values
            || !std::all_of(values, values+w.k, finite)) return false;
        std::copy_n(values, w.k, rhs_);
        solve();
        if (!std::all_of(rhs_, rhs_+w.k, finite)) return false;
        std::copy_n(rhs_, w.k, values);
        return true;
    }

    template<class State>
    bool advance(std::size_t time, State& output) noexcept {
        static_assert(Intention == PreparedIntention::derivatives);
        if (status_ != Status::ready || time != next_ || time >= n_ || !output.gradient
            || (Order == 2 && !output.hessian)) return false;
        const auto w = matrix();
        advance_state<true>(time);
        for (std::size_t r = 0; r < w.k; ++r)
            if (time == anchors_[r]) {
                // Audit the recurrence before installing the exact chart identities.
                for (std::size_t i = 0; i < w.m; ++i)
                    raw_identity_first_ = std::max(raw_identity_first_,
                        magnitude(current_[1+i] - Real(i == selected_[r] ? 1 : 0)));
                if constexpr (Order == 2)
                    for (std::size_t i = 0; i < w.packed; ++i)
                        raw_identity_second_ = std::max(raw_identity_second_, magnitude(current_[1+w.m+i]));
                std::fill_n(current_+1,w.m,Real(0)); current_[1+selected_[r]] = Real(1);
                if constexpr (Order == 2) std::fill_n(current_+1+w.m,w.packed,Real(0));
            }
        if (!std::all_of(current_,current_+w.state,[](Real x) {
                return finite_sensitivity(x) && std::isfinite(static_cast<double>(x));
            })) {
            status_ = Status::nonfinite; return false;
        }
        retain(time); ++next_;
        output.value = static_cast<double>(current_[0]);
        std::fill_n(output.gradient,layout().parameters,0.);
        for (std::size_t i = 0; i < w.m; ++i) output.gradient[i] = static_cast<double>(current_[1+i]);
        if constexpr (Order == 2) {
            std::fill_n(output.hessian,layout().packed,0.);
            for (std::size_t i = 0; i < w.m; ++i)
                for (std::size_t j = i; j < w.m; ++j)
                    output.hessian[eq::packed_index(layout().parameters,i,j)]
                        = static_cast<double>(current_[1+w.m+packed(i,j)]);
        }
        return true;
    }
    Status status() const noexcept { return status_; }
    double pivot_ratio() const noexcept { return pivot_ratio_; }
    double raw_identity_first() const noexcept { return raw_identity_first_; }
    double raw_identity_second() const noexcept { return raw_identity_second_; }
    Real anchor_value(std::size_t r) const noexcept { return rows_[r*matrix().state]; }
    Real coefficient_basis(std::size_t r, std::size_t c) const noexcept {
        return -rows_[r*matrix().state+1+
            (Intention == PreparedIntention::value_basis ? c : selected_[c])];
    }
    Real map_first(std::size_t r, std::size_t i) const noexcept {
        static_assert(Intention == PreparedIntention::derivatives);
        return map_g_[r*matrix().m+i];
    }
    Real map_second(std::size_t r, std::size_t i, std::size_t j) const noexcept {
        static_assert(Intention == PreparedIntention::derivatives);
        if constexpr (Order == 2) return map_h_[r*matrix().packed+packed(i,j)];
        return Real(0);
    }
};
} // namespace vlk::experiment::mean_chart
