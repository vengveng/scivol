// SCIVOL-NATIVE: integration candidate; adaptive scalar primal, not derivative certification.
#pragma once
#include "../../math/mean/mean_precision.hpp"
#include "mean_equation.hpp"
#include "../../math/runtime_workspace.hpp"

namespace vlk::experiment::mean {

template<class Layout, const Layout* Fixed = nullptr>
class ArmaResidualPrecision {
    using Fast = mean_screen::Scalar;
    using Quick = mean_replay::Quick;
    using Exact = mean_replay::Accurate;
    using Interval = mean_replay::Interval;
    static constexpr auto ar_order = Fixed != nullptr ? Fixed->p_ar : mean_replay::dynamic;
    static constexpr auto ma_order = Fixed != nullptr ? Fixed->q_ma : mean_replay::dynamic;
    struct Checkpoint {
        std::size_t next = 0u;
        double defect = 0.;
        bool initialized = false;
    };

    const Layout& supplied_;
    const double* theta_;
    mean_screen::ObservationSummary sample_;
    mean_replay::Counts* counts_;
    std::size_t next_ = 0u;
    Fast* fast_ = nullptr;
    Quick* quick_ = nullptr;
    Exact* exact_ = nullptr;
    double gain_ = INFINITY;
    mean_screen::Plan screen_;
    Checkpoint quick_checkpoint_, exact_checkpoint_;
    bool ready_ = false, supported_ = true;

    const Layout& layout() const noexcept { return arma_layout<Layout, Fixed>(supplied_); }

    template<class Scalar>
#if defined(_MSC_VER)
    __forceinline
#else
    __attribute__((always_inline)) inline
#endif
    Scalar step(std::size_t time, Scalar* history) noexcept {
        const auto q = layout().q_ma;
        const auto value = arma_residual_value<Scalar, Layout, Fixed>(
            layout(), theta_, sample_.data(), time, [&](std::size_t lag) {
                const auto previous = history[(time - 1u - lag) % q];
                return std::isfinite(gain_) ? Scalar(previous.value, 0.) : previous;
            });
        if (q) history[time % q] = value;
        return value;
    }

    template<class Scalar>
    Scalar replay(std::size_t time, Scalar* history, Checkpoint& checkpoint) noexcept {
        if (counts_) ++counts_->replays;
        if (!checkpoint.initialized) {
            std::fill_n(history, layout().q_ma, Scalar(0.));
            checkpoint.next = layout().maximum_lag;
            checkpoint.initialized = true;
        }
        Scalar result;
        const auto begin = layout().q_ma ? checkpoint.next : time;
        for (std::size_t t = begin; t <= time; ++t) {
            result = step(t, history);
            checkpoint.defect = layout().q_ma
                ? mean_replay::add_bound(checkpoint.defect, result.error) : result.error;
            if (counts_) ++counts_->observations;
        }
        checkpoint.next = time + 1u;
        if (std::isfinite(gain_))
            result.error = mean_replay::multiply_bound(gain_, checkpoint.defect);
        return result;
    }

#if defined(_MSC_VER)
    __declspec(noinline)
#else
    __attribute__((noinline))
#endif
    mean_replay::Fast recover(std::size_t time) noexcept {
        auto result = mean_replay::narrow(replay(time, quick_, quick_checkpoint_));
        if (!result.resolved()) {
            result = mean_replay::narrow(replay(time, exact_, exact_checkpoint_));
            if (!std::isfinite(gain_))
                for (std::size_t lag = 0; lag < layout().q_ma; ++lag)
                    fast_[lag] = mean_replay::narrow(exact_[lag]);
        } else if (!std::isfinite(gain_)) {
            for (std::size_t lag = 0; lag < layout().q_ma; ++lag)
                fast_[lag] = mean_replay::narrow(quick_[lag]);
        }
        if (!result.resolved() && counts_) ++counts_->unresolved;
        return result;
    }

public:
    ArmaResidualPrecision(const Layout& layout, const double* theta,
        const mean_screen::ObservationSummary& sample, mean_replay::Counts* counts = nullptr) noexcept
        : supplied_(layout), theta_(theta), sample_(sample), counts_(counts) {}

    ArmaResidualPrecision(const Layout& layout, const double* theta,
        const double* data, std::size_t count, mean_replay::Counts* counts = nullptr) noexcept
        : ArmaResidualPrecision(layout, theta, mean_screen::ObservationSummary::prepare(data, count), counts) {}

    static std::size_t workspace_bytes(const Layout& supplied) noexcept {
        const auto& l = arma_layout<Layout, Fixed>(supplied);
        if (!valid_arma_mean_layout(l)) return 0u;
        runtime::WorkspaceSizer s;
        if (!s.add<Fast>(l.q_ma) || !s.add<Quick>(l.q_ma) || !s.add<Exact>(l.q_ma)
            || !s.add<Interval>(l.q_ma) || !s.add<Interval>(l.q_ma)) return 0u;
        return s.size();
    }

    bool prepare(void* memory, std::size_t bytes) noexcept {
        ready_ = false;
        const auto& l = layout();
        const auto required = workspace_bytes(l);
        if (!valid_arma_mean_layout(l) || !theta_ || !sample_.data() || sample_.size() <= l.maximum_lag
            || (l.q_ma && (!memory || !required)) || bytes < required) return false;
        runtime::WorkspaceArena arena(memory, bytes);
        fast_ = take_sensitivities<Fast>(arena,l.q_ma);
        quick_ = take_sensitivities<Quick>(arena,l.q_ma);
        exact_ = take_sensitivities<Exact>(arena,l.q_ma);
        auto* current = take_sensitivities<Interval>(arena,l.q_ma);
        auto* next = take_sensitivities<Interval>(arena,l.q_ma);
        if (l.q_ma && (!fast_ || !quick_ || !exact_ || !current || !next)) return false;
        gain_ = mean_replay::filter_gain<ma_order>(theta_ + l.ma_start, l.q_ma, current, next);
        screen_ = mean_screen::prepare<ar_order, ma_order>(theta_, l.p_ar, l.q_ma, sample_,
                                                        sample_.size() - l.maximum_lag, gain_);
        ready_ = true;
        rewind();
        return true;
    }

    bool ready() const noexcept { return ready_; }

    void rewind() noexcept {
        if (!ready_) return;
        std::fill_n(fast_, layout().q_ma, Fast(0.));
        quick_checkpoint_ = {};
        exact_checkpoint_ = {};
        supported_ = true;
        next_ = layout().maximum_lag;
    }

#if defined(_MSC_VER)
    __forceinline
#else
    __attribute__((always_inline)) inline
#endif
    wide::Real advance(std::size_t time) noexcept {
        if (!ready_ || time != next_ || time >= sample_.size()) return wide::Real(NAN);
        ++next_;
        const auto primal = step(time, fast_);
        supported_ = supported_ && primal.supported;
        if (supported_ && screen_.resolved(primal.value)) return primal.value;
        const auto corrected = recover(time);
        return corrected.resolved() ? corrected.value : wide::Real(NAN);
    }
};

// Borrow immutable sample metadata and caller-owned scratch across parameter requests.
template<class Layout, const Layout* Fixed = nullptr>
struct ArmaResidualPrecisionContext {
    using Plan = ArmaResidualPrecision<Layout, Fixed>;
    mean_screen::ObservationSummary sample;
    void* workspace = nullptr;
    std::size_t workspace_size = 0u;
    mean_replay::Counts* counts = nullptr;

    bool matches_sample(const double* data, std::size_t count) const noexcept {
        return sample.data() == data && sample.size() == count;
    }

    Plan prepare(const Layout& layout, const double* theta) const noexcept {
        Plan plan(layout, theta, sample, counts);
        (void)plan.prepare(workspace, workspace_size);
        return plan;
    }
};

} // namespace vlk::experiment::mean
