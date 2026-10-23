// Internal model coordinates; no optimizer, selection policy, or public routing.
#pragma once
#include "residual_mean_mapper.hpp"

namespace vlk::experiment::mean_chart {

enum class CoordinateKind { anchored, residual };

// The selected DD coefficients are part of the original-z point. The shared
// scalar equation evaluates their affine correction before narrowing the state.
template<class Layout, const Layout* Fixed = nullptr, std::size_t FixedK = mean::dynamic>
class ExtendedCoefficientPrimal {
    using Accurate = mean_replay::Accurate;
    const Layout& supplied_;
    const double *base_, *data_;
    const Real* coefficients_;
    const std::size_t* selected_;
    std::size_t n_, k_, next_ = 0;
    Accurate* history_ = nullptr;
    bool ready_ = false;
    const Layout& layout() const noexcept { return mean::arma_layout<Layout,Fixed>(supplied_); }
public:
    ExtendedCoefficientPrimal(const Layout& layout, const double* base,
        const double* data, std::size_t n, const std::size_t* selected,
        std::size_t k, const Real* coefficients) noexcept
        : supplied_(layout), base_(base), data_(data), coefficients_(coefficients),
          selected_(selected), n_(n), k_(k) {}
    static std::size_t workspace_bytes(const Layout& supplied) noexcept {
        const auto& l = mean::arma_layout<Layout,Fixed>(supplied);
        rt::WorkspaceSizer s;
        return mean::valid_arma_mean_layout(l) && s.add<Accurate>(l.q_ma) ? s.size() : 0;
    }
    bool prepare(void* memory, std::size_t bytes) noexcept {
        ready_ = false;
        if (!base_ || !data_ || !selected_ || !coefficients_ || n_ <= layout().maximum_lag
            || bytes < workspace_bytes(layout()) || (layout().q_ma && !memory)) return false;
        rt::WorkspaceArena arena(memory,bytes);
        history_ = take_sensitivities<Accurate>(arena,layout().q_ma);
        if (layout().q_ma && !history_) return false;
        ready_ = true; rewind(); return true;
    }
    bool ready() const noexcept { return ready_; }
    void rewind() noexcept {
        std::fill_n(history_,layout().q_ma,Accurate(0));
        next_ = layout().maximum_lag;
    }
    Real advance(std::size_t time) noexcept {
        if (!ready_ || time != next_ || time >= n_) return Real(NAN);
        const auto& l = layout();
        auto value = mean::arma_residual_value<Accurate,Layout,Fixed>(
            l,base_,data_,time,[&](std::size_t lag) {
                return history_[(time-1-lag)%l.q_ma];
            });
        mean::ArmaDesign<Fixed != nullptr ? Fixed->p_ar : mean::dynamic> design{data_,l.p_ar};
        for (std::size_t r = 0; r < mean::extent<FixedK>(k_); ++r) {
            const auto shift = Accurate(coefficients_[r].high)+Accurate(coefficients_[r].low)
                -Accurate(base_[selected_[r]]);
            value -= shift*Accurate(design.at(time,selected_[r]));
        }
        if (l.q_ma) history_[time%l.q_ma] = value;
        ++next_;
        const auto narrowed = mean_replay::narrow(value);
        if (!narrowed.resolved()) { ready_ = false; return Real(NAN); }
        return narrowed.value;
    }
};

struct CoordinateOutput {
    double* value = nullptr;       // N
    double* jacobian = nullptr;    // N by N, row-major
    double* hessian = nullptr;     // N by packed(N), upper triangle per output
};

// Mask bits F=1, G=2, H=4 select public map outputs, not traversal count.
// H internally needs first-order state but never writes an unrequested Jacobian.
// Private precondition: borrowed inputs, arena, and output ranges are disjoint.
// A future public caller must enforce this before entering the prepared owner.
template<CoordinateKind Kind, unsigned Mask, class Layout,
         const Layout* Fixed = nullptr, std::size_t FixedK = mean::dynamic>
class MeanCoordinateJet {
    static_assert(Mask > 0 && Mask < 8);
    static constexpr int order = Mask & 4u ? 2 : 1;
    using WideNumber = tlfloat::Octuple;
    static_assert(alignof(WideNumber) >= alignof(Real));
    using Mapper = OffOriginMapper<Layout,Fixed,FixedK>;
    using Primal = ExtendedCoefficientPrimal<Layout,Fixed,FixedK>;
    template<class Number>
    using Residual = ResidualPreservingChart<order,Layout,Fixed,FixedK,
        PreparedIntention::derivatives,Primal,Number>;
    template<class Number>
    using Anchored = mean::AnchoredArmaChart<order,Layout,Fixed,false,Primal,Number>;
    template<class Number>
    using Chart = std::conditional_t<Kind == CoordinateKind::anchored,Anchored<Number>,Residual<Number>>;
    const Layout& supplied_;
    const double *origin_, *data_;
    const std::size_t *anchors_, *selected_;
    std::size_t n_, k_;
    Mapper mapper_;
    void *mapper_memory_ = nullptr, *primal_memory_ = nullptr, *chart_memory_ = nullptr;
    double *rounded_ = nullptr, *evidence_ = nullptr;
    bool ready_ = false;
    int status_ = -1;
    const Layout& layout() const noexcept { return mean::arma_layout<Layout,Fixed>(supplied_); }
    template<class Number>
    static bool representable(Number x) noexcept {
        return finite_sensitivity(x) && std::isfinite(static_cast<double>(x));
    }
    template<class Number>
    static std::size_t chart_bytes_for(const Layout& l, std::size_t k) noexcept {
        if constexpr (Kind == CoordinateKind::anchored) return Chart<Number>::workspace_bytes(l);
        else return Chart<Number>::workspace_bytes(l,k);
    }
    static std::size_t chart_bytes(const Layout& l, std::size_t k) noexcept {
        const auto ordinary = chart_bytes_for<Real>(l,k), wide = chart_bytes_for<WideNumber>(l,k);
        return ordinary && wide ? std::max(ordinary,wide) : 0;
    }
    template<class C>
    static auto first(const C& chart, std::size_t r, std::size_t i) noexcept {
        if constexpr (Kind == CoordinateKind::anchored) return chart.map_first(i);
        else return chart.map_first(r,i);
    }
    template<class C>
    static auto second(const C& chart, std::size_t r, std::size_t i, std::size_t j) noexcept {
        if constexpr (Kind == CoordinateKind::anchored) return chart.map_second(i,j);
        else return chart.map_second(r,i,j);
    }
    template<class C>
    bool project(const C& chart, CoordinateOutput out) noexcept {
        const auto& l = layout();
        const auto k = mean::extent<FixedK>(k_);
        // Validate before committing any public output, including narrowing overflow.
        for (std::size_t r = 0; r < k; ++r)
            for (std::size_t i = 0; i < l.mean_parameters; ++i) {
                if constexpr (Mask & 2u)
                    if (!representable(first(chart,r,i))) return false;
                if constexpr (Mask & 4u)
                    for (std::size_t j = i; j < l.mean_parameters; ++j)
                        if (!representable(second(chart,r,i,j))) return false;
            }
        if constexpr (Mask & 1u) std::copy_n(rounded_,l.parameters,out.value);
        if constexpr (Mask & 2u) {
            std::fill_n(out.jacobian,l.parameters*l.parameters,0.);
            for (std::size_t i = 0; i < l.parameters; ++i) out.jacobian[i*l.parameters+i] = 1.;
            for (std::size_t r = 0; r < k; ++r) {
                auto* row = out.jacobian+selected_[r]*l.parameters;
                std::fill_n(row,l.parameters,0.);
                for (std::size_t i = 0; i < l.mean_parameters; ++i)
                    row[i] = static_cast<double>(first(chart,r,i));
            }
        }
        if constexpr (Mask & 4u) {
            std::fill_n(out.hessian,l.parameters*l.packed,0.);
            for (std::size_t r = 0; r < k; ++r)
                for (std::size_t i = 0; i < l.mean_parameters; ++i)
                    for (std::size_t j = i; j < l.mean_parameters; ++j)
                        out.hessian[selected_[r]*l.packed+eq::packed_index(l.parameters,i,j)]
                            = static_cast<double>(second(chart,r,i,j));
        }
        return true;
    }
    template<class Number>
    bool map_derivatives(CoordinateOutput out, Real error) noexcept {
        status_ = -3;
        Primal primal(layout(),mapper_.base_point(),data_,n_,selected_,k_,mapper_.mapped_coefficients());
        if (!primal.prepare(primal_memory_,Primal::workspace_bytes(layout()))) return false;
        if constexpr (Kind == CoordinateKind::anchored) {
            const auto retained_error = mean_replay::lift<Number>(error);
            Chart<Number> chart(layout(),mapper_.base_point(),data_,n_,anchors_[0],&primal,&retained_error);
            if (!chart.prepare(chart_memory_,chart_bytes_for<Number>(layout(),k_)) || !project(chart,out))
                return false;
        } else {
            Chart<Number> chart(layout(),mapper_.base_point(),data_,n_,anchors_,selected_,k_,&primal);
            if (!chart.prepare(chart_memory_,chart_bytes_for<Number>(layout(),k_))) {
                if (chart.status() == Status::rank_fallback) status_ = -4;
                return false;
            }
            if (!project(chart,out)) return false;
        }
        status_ = 0; return true;
    }
public:
    MeanCoordinateJet(const Layout& l, const double* origin, const double* data,
        std::size_t n, const std::size_t* anchors, const std::size_t* selected,
        std::size_t k) noexcept
        : supplied_(l), origin_(origin), data_(data), anchors_(anchors),
          selected_(selected), n_(n), k_(k), mapper_(l,origin,data,n,anchors,selected,k) {}
    static std::size_t workspace_bytes(const Layout& supplied, std::size_t k) noexcept {
        const auto& l = mean::arma_layout<Layout,Fixed>(supplied);
        const auto w = Mapper::workspace_layout(l,k);
        std::size_t square = 0, cube = 0, adjacent = 0, product = 0;
        if (!w.bytes || !rt::checked_sum(l.parameters,1,adjacent)
            || !rt::checked_product(l.parameters,adjacent,product) || l.packed != product/2
            || !rt::checked_product(l.parameters,l.parameters,square)
            || !rt::checked_product(l.parameters,l.packed,cube)) return 0;
        rt::WorkspaceSizer s;
        if (!s.add<double>(l.parameters) || !s.add<double>(w.evidence)
            || !s.add<Real>(0) || !s.add<std::byte>(w.bytes)) return 0;
        if constexpr (Mask & 6u) {
            if (!chart_bytes(l,k) || !s.add<mean_replay::Accurate>(0)
                || !s.add<std::byte>(Primal::workspace_bytes(l))
                || !s.add<WideNumber>(0) || !s.add<std::byte>(chart_bytes(l,k))) return 0;
        }
        return s.size();
    }
    bool prepare(void* memory, std::size_t bytes) noexcept {
        ready_ = false;
        status_ = -1;
        const auto required = workspace_bytes(layout(),k_);
        if (!memory || !required || bytes < required || !anchors_ || !selected_
            || reinterpret_cast<std::uintptr_t>(memory) % alignof(mean_replay::Accurate)) return false;
        if constexpr (Kind == CoordinateKind::anchored)
            if (k_ != 1 || selected_[0] != 0) return false;
        rt::WorkspaceArena arena(memory,bytes);
        const auto w = Mapper::workspace_layout(layout(),k_);
        rounded_ = take_sensitivities<double>(arena,layout().parameters);
        evidence_ = take_sensitivities<double>(arena,w.evidence);
        (void)arena.take<Real>(0); mapper_memory_ = arena.take<std::byte>(w.bytes);
        if constexpr (Mask & 6u) {
            (void)arena.take<mean_replay::Accurate>(0);
            primal_memory_ = arena.take<std::byte>(Primal::workspace_bytes(layout()));
            (void)arena.take<WideNumber>(0); chart_memory_ = arena.take<std::byte>(chart_bytes(layout(),k_));
            if (!chart_memory_) return false;
        }
        if (!mapper_memory_) return false;
        status_ = mapper_.prepare(mapper_memory_,w.bytes);
        ready_ = status_ == 0;
        return ready_;
    }
    bool map(const double* original_z, CoordinateOutput out) noexcept {
        status_ = -1;
        if (!ready_ || !original_z || ((Mask & 1u) && !out.value)
            || ((Mask & 2u) && !out.jacobian) || ((Mask & 4u) && !out.hessian)) return false;
        const Real error = mapper_.origin_anchor_error();
        status_ = mapper_.map(original_z,rounded_,evidence_,
            Kind == CoordinateKind::anchored ? &error : nullptr);
        if (status_ != 0) return false;
        if constexpr (!(Mask & 6u)) {
            std::copy_n(rounded_,layout().parameters,out.value); return true;
        } else {
            if (map_derivatives<Real>(out,error)) return true;
            if (status_ != -3) return false;
            // Reuse the same policy, original point and arena with a wider scalar.
            return map_derivatives<WideNumber>(out,error);
        }
    }
    // Existing mapper meanings: -1 invalid/unprepared, -3 numerical range, -4 rank.
    int status() const noexcept { return status_; }
};
} // namespace vlk::experiment::mean_chart
