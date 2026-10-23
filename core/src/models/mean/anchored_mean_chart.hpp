// SCIVOL-NATIVE: integration candidate; anchored ARMA coordinates, no solver policy.
#ifndef SCIVOL_CORE_ANCHORED_MEAN_CHART_HPP
#define SCIVOL_CORE_ANCHORED_MEAN_CHART_HPP

#include "mean_equation.hpp"
#include "../../math/runtime_workspace.hpp"
#include "../../math/mean/mean_precision.hpp"

namespace vlk::experiment::mean {

// The caller owns conditioning and domain checks. Only derivatives change
// coordinates; the primal always uses the actual supplied theta.
template<int Order, class Layout, const Layout* FixedLayout = nullptr,
         bool MeanOnly = false, class PrimalPlan = void, class Number = wide::Real>
class AnchoredArmaChart {
    static_assert(Order == 1 || Order == 2);
    using Real = Number;
    struct NoPrimalPlan {};
    [[no_unique_address]] std::conditional_t<std::is_void_v<PrimalPlan>,
        NoPrimalPlan, PrimalPlan*> primal_{};
    const Layout& supplied_;
    const double* theta_;
    const double* data_;
    std::size_t observation_count_, pivot_, next_ = 0u;
    Real *residual_history_ = nullptr, *basis_history_ = nullptr;
    Real *residual_ = nullptr, *basis_ = nullptr, *anchor_basis_ = nullptr;
    Real* intercept_ = nullptr;
    Real error_{}, original_residual_{};
    bool prepared_ = false;
    const Real* retained_error_ = nullptr;

    static const Layout& select_layout(const Layout& supplied) noexcept {
        return arma_layout<Layout, FixedLayout>(supplied);
    }
    const Layout& layout() const noexcept { return select_layout(supplied_); }
    std::size_t width() const noexcept { return layout().mean_parameters; }
    std::size_t packed() const noexcept { return width() * (width() + 1u) / 2u; }
    std::size_t state_size() const noexcept {
        return 1u + width() + (Order == 2 ? packed() : 0u);
    }
    static bool finite(Real x) noexcept {
        return finite_sensitivity(x);
    }
    Real* first(Real* state) const noexcept { return state + 1u; }
    Real* second(Real* state) const noexcept { return state + 1u + width(); }
    void clear_history() noexcept {
        next_ = layout().maximum_lag;
        const auto count = layout().q_ma * state_size();
        std::fill_n(residual_history_, count, Real(0.));
        std::fill_n(basis_history_, count, Real(0.));
        if constexpr (!std::is_void_v<PrimalPlan>) primal_->rewind();
    }

    template<bool Basis, bool Chart>
    void advance_state(Real* current, Real* history, std::size_t time) noexcept {
        using namespace generated::arma_garch_equation;
        const auto& l = layout();
        const auto m = width(), size = state_size();
        std::fill_n(current, size, Real(0.));
        const auto lagged = [&](std::size_t lag) {
            return history[((time - 1u - lag) % l.q_ma) * size];
        };
        if constexpr (Basis)
            current[0] = inverse_ma_value<Real, Layout, FixedLayout>(Real(1.), l, theta_, lagged);
        else {
            if constexpr (std::is_void_v<PrimalPlan>)
                current[0] = arma_residual_value<Real, Layout, FixedLayout>(l, theta_, data_, time, lagged);
            else
                current[0] = mean_replay::lift<Real>(primal_->advance(time));
            if constexpr (Chart) {
                for (std::size_t i = 0; i < m; ++i)
                    first(current)[i] = -first(intercept_)[i];
                if constexpr (Order == 2)
                    for (std::size_t k = 0; k < packed(); ++k)
                        second(current)[k] = -second(intercept_)[k];
            } else initialize_residual_first(first(current), m, 0u);
            for (std::size_t lag = 0; lag < l.p_ar; ++lag) {
                const auto index = l.ar_start + lag;
                const double observed = data_[time - 1u - lag];
                if constexpr (Chart) first(current)[index] -= Real(observed);
                else add_ar_first(first(current), index, observed);
            }
        }
        for (std::size_t lag = 0; lag < l.q_ma; ++lag) {
            const auto ma = l.ma_start + lag;
            const auto* previous = history + ((time - 1u - lag) % l.q_ma) * size;
            add_ma_first(first(current), m, ma, theta_[ma], previous[0], previous + 1u);
            if constexpr (Order == 2)
                add_ma_second(second(current), m, ma, theta_[ma],
                    previous + 1u, previous + 1u + m);
        }
    }
    void save(Real* history, const Real* current, std::size_t time) noexcept {
        if (layout().q_ma != 0u)
            std::copy_n(current, state_size(),
                history + (time % layout().q_ma) * state_size());
    }

public:
    AnchoredArmaChart(const Layout& supplied, const double* theta,
        const double* data, std::size_t observation_count, std::size_t pivot,
        PrimalPlan* primal = nullptr, const Real* retained_error = nullptr) noexcept
        : supplied_(supplied), theta_(theta), data_(data),
          observation_count_(observation_count), pivot_(pivot), retained_error_(retained_error) {
        if constexpr (!std::is_void_v<PrimalPlan>) primal_ = primal;
    }

    static std::size_t workspace_bytes(const Layout& supplied) noexcept {
        const auto& l = select_layout(supplied);
        std::size_t pair = 0u, size = 0u, rows = 0u, count = 0u, full_pair = 0u;
        if (!valid_arma_mean_layout(l)
            || !runtime::checked_sum(l.parameters, 1u, full_pair)
            || !runtime::checked_product(l.parameters, full_pair, full_pair)
            || l.packed != full_pair / 2u
            || !runtime::checked_sum(l.mean_parameters, 1u, pair)
            || !runtime::checked_product(l.mean_parameters, pair, pair)
            || !runtime::checked_sum(1u, l.mean_parameters, size)
            || (Order == 2 && !runtime::checked_sum(size, pair / 2u, size))
            || !runtime::checked_product(2u, l.q_ma, rows)
            || !runtime::checked_sum(rows, 4u, rows)
            || !runtime::checked_product(rows, size, count)) return 0u;
        runtime::WorkspaceSizer sizer;
        return sizer.add<Real>(count) ? sizer.size() : 0u;
    }

    bool prepare(void* memory, std::size_t bytes) noexcept {
        prepared_ = false;
        const auto required = workspace_bytes(layout());
        if (!theta_ || !data_ || !memory || !required || bytes < required
            || pivot_ < layout().maximum_lag || pivot_ >= observation_count_) return false;
        if constexpr (!std::is_void_v<PrimalPlan>)
            if (!primal_ || !primal_->ready()) return false;
        runtime::WorkspaceArena arena(memory, bytes);
        const auto size = state_size(), history = layout().q_ma * size;
        residual_history_ = take_sensitivities<Real>(arena,history);
        basis_history_ = take_sensitivities<Real>(arena,history);
        residual_ = take_sensitivities<Real>(arena,size); basis_ = take_sensitivities<Real>(arena,size);
        anchor_basis_ = take_sensitivities<Real>(arena,size); intercept_ = take_sensitivities<Real>(arena,size);
        if (!intercept_) return false;
        clear_history();
        for (std::size_t t = layout().maximum_lag; t <= pivot_; ++t) {
            advance_state<false, false>(residual_, residual_history_, t);
            advance_state<true, false>(basis_, basis_history_, t);
            save(residual_history_, residual_, t); save(basis_history_, basis_, t);
        }
        if (!finite(basis_[0]) || basis_[0] == Real(0.) || !finite(residual_[0])) return false;
        std::copy_n(basis_, size, anchor_basis_);
        original_residual_ = residual_[0];
        // rho is zero at the chart origin, but its residual is rho - B(psi) E.
        error_ = retained_error_ ? *retained_error_ : -residual_[0] / basis_[0];
        intercept_[0] = Real(theta_[0]);
        first(intercept_)[0] = -Real(1.) / basis_[0];
        for (std::size_t i = 1u; i < width(); ++i)
            first(intercept_)[i] = (first(residual_)[i] + error_ * first(basis_)[i]) / basis_[0];
        if constexpr (Order == 2) {
            for (std::size_t i = 0; i < width(); ++i) {
                for (std::size_t j = i; j < width(); ++j) {
                    const auto k = generated::arma_garch_equation::packed_index(width(), i, j);
                    second(intercept_)[k] = ((i && j ? second(residual_)[k] : Real(0.))
                        + error_ * second(basis_)[k] - first(intercept_)[i] * first(basis_)[j]
                        - first(intercept_)[j] * first(basis_)[i]) / basis_[0];
                }
            }
        }
        if (!std::all_of(intercept_, intercept_ + size, finite)) return false;
        clear_history();
        prepared_ = true;
        return true;
    }

    template<class State>
    bool advance(std::size_t time, State& output) noexcept {
        if (!prepared_ || time != next_ || time >= observation_count_
            || !output.gradient || (Order == 2 && !output.hessian)) return false;
        advance_state<false, true>(residual_, residual_history_, time);
        if (time == pivot_) {
            for (std::size_t i = 0; i < width(); ++i)
                first(residual_)[i] = Real(i == 0u ? 1. : 0.) - error_ * first(anchor_basis_)[i];
            if constexpr (Order == 2)
                for (std::size_t k = 0; k < packed(); ++k)
                    second(residual_)[k] = -error_ * second(anchor_basis_)[k];
        }
        if (!std::all_of(residual_, residual_ + state_size(), [](Real x) {
                return finite_sensitivity(x) && std::isfinite(static_cast<double>(x));
            })) {
            prepared_ = false;
            return false;
        }
        save(residual_history_, residual_, time);
        ++next_;
        output.value = static_cast<double>(residual_[0]);
        const auto output_width = MeanOnly ? width() : layout().parameters;
        std::fill_n(output.gradient, output_width, 0.);
        for (std::size_t i = 0; i < width(); ++i)
            output.gradient[i] = static_cast<double>(first(residual_)[i]);
        if constexpr (Order == 2) {
            std::fill_n(output.hessian, output_width * (output_width + 1u) / 2u, 0.);
            for (std::size_t i = 0; i < width(); ++i)
                for (std::size_t j = i; j < width(); ++j)
                    output.hessian[generated::arma_garch_equation::packed_index(output_width, i, j)]
                        = static_cast<double>(second(residual_)[generated::arma_garch_equation::packed_index(width(), i, j)]);
        }
        return true;
    }

    Real anchor_error() const noexcept { return error_; }
    Real pivot_value() const noexcept { return original_residual_; }
    Real intercept_basis() const noexcept { return anchor_basis_[0]; }
    Real map_first(std::size_t i) const noexcept { return first(intercept_)[i]; }
    Real map_second(std::size_t i, std::size_t j) const noexcept {
        static_assert(Order == 2);
        if (i > j) std::swap(i,j);
        return second(intercept_)[generated::arma_garch_equation::packed_index(width(),i,j)];
    }
    // Read after prepare, before advance changes the residual derivative state.
    Real prepared_coefficient_basis(std::size_t column) const noexcept {
        return -first(residual_)[column];
    }
};

} // namespace vlk::experiment::mean
#endif
