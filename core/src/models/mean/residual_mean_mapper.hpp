// SCIVOL-NATIVE: value-only residual-coordinate mapping; no fitting policy.
#pragma once
#include "residual_mean_chart.hpp"

namespace vlk::experiment::mean_chart {

struct MapperLayout {
    std::size_t primal = 0, chart = 0, parameters = 0, k = 0, replay = 0, evidence = 0, bytes = 0;
    constexpr bool operator==(const MapperLayout&) const = default;
};

template<class Layout, const Layout* Fixed = nullptr,
         std::size_t FixedK = mean::dynamic>
class OffOriginMapper {
    using Primal = mean::ArmaResidualPrecision<Layout, Fixed>;
    using Chart = ResidualPreservingChart<1, Layout, Fixed, FixedK, PreparedIntention::value_basis>;
    using Accurate = vlk::experiment::mean_replay::Accurate;
    const Layout& supplied_;
    const double *origin_, *data_;
    const std::size_t *anchors_, *selected_;
    std::size_t n_, k_, last_ = 0;
    vlk::experiment::mean_screen::ObservationSummary sample_ =
        vlk::experiment::mean_screen::ObservationSummary::prepare(nullptr,0);
    void *primal_memory_ = nullptr, *chart_memory_ = nullptr;
    double *point_ = nullptr, *rounded_ = nullptr;
    Real *targets_ = nullptr, *rhs_ = nullptr, *mapped_ = nullptr;
    Accurate *diagnostic_history_ = nullptr, *diagnostic_origin_ = nullptr, *diagnostic_actual_ = nullptr;
    double* evidence_ = nullptr;
    double origin_pivot_ = 0;
    Real origin_error_{};
    bool ready_ = false, sample_prepared_ = false;

    const Layout& layout() const noexcept { return mean::arma_layout<Layout,Fixed>(supplied_); }
    static bool finite(Real x) noexcept {
        return std::isfinite(x.high) && std::isfinite(x.low);
    }
    bool diagnostic_replay(const double* theta, Accurate* anchors) noexcept {
        const auto q = layout().q_ma;
        std::fill_n(diagnostic_history_,q,Accurate(0));
        for (std::size_t t = layout().maximum_lag; t <= last_; ++t) {
            // Storage/traversal only. The existing equation and bounded scalar own all arithmetic.
            const auto value = mean::arma_residual_value<Accurate,Layout,Fixed>(
                layout(),theta,data_,t,[&](std::size_t lag) {
                    return diagnostic_history_[(t-1-lag)%q];
                });
            if (q) diagnostic_history_[t%q] = value;
            for (std::size_t r = 0; r < mean::extent<FixedK>(k_); ++r)
                if (t == anchors_[r]) {
                    const auto narrowed = vlk::experiment::mean_replay::narrow(value);
                    if (!narrowed.resolved()) return false;
                    anchors[r] = value;
                }
        }
        return true;
    }

public:
    using Sample = vlk::experiment::mean_screen::ObservationSummary;

    OffOriginMapper(const Layout& l, const double* origin, const double* data,
        std::size_t n, const std::size_t* anchors, const std::size_t* selected,
        std::size_t k) noexcept
        : supplied_(l), origin_(origin), data_(data), anchors_(anchors), selected_(selected),
          n_(n), k_(k) {}

    OffOriginMapper(const Layout& l, const double* origin, const Sample& sample,
        const std::size_t* anchors, const std::size_t* selected, std::size_t k) noexcept
        : supplied_(l), origin_(origin), data_(sample.data()), anchors_(anchors), selected_(selected),
          n_(sample.size()), k_(k), sample_(sample), sample_prepared_(true) {}

    static MapperLayout workspace_layout(const Layout& supplied, std::size_t k) noexcept {
        const auto& l = mean::arma_layout<Layout,Fixed>(supplied);
        MapperLayout w;
        w.k = mean::extent<FixedK>(k); w.parameters = l.parameters; w.replay = l.q_ma;
        if (!mean::valid_arma_mean_layout(l) || !w.k || w.k > l.ma_start
            || !rt::checked_product(w.k,8,w.evidence) || !rt::checked_sum(w.evidence,2,w.evidence)) return {};
        w.primal = Primal::workspace_bytes(l); w.chart = Chart::workspace_bytes(l,w.k);
        if ((l.q_ma && !w.primal) || !w.chart) return {};
        rt::WorkspaceSizer s;
        if (!s.add<Accurate>(w.replay) || !s.add<Accurate>(w.k) || !s.add<Accurate>(w.k)
            || !s.add<Real>(w.k) || !s.add<Real>(w.k) || !s.add<Real>(w.k) || !s.add<double>(w.evidence)
            || !s.add<double>(w.parameters) || !s.add<double>(w.parameters)
            || !s.add<Real>(0) || !s.add<std::byte>(w.primal)
            || !s.add<Real>(0) || !s.add<std::byte>(w.chart)) return {};
        w.bytes = s.size();
        return w;
    }

    // A summary borrows immutable data; mismatched summaries invalidate preparation.
    int prepare(void* memory, std::size_t bytes, const Sample& sample) noexcept {
        ready_ = false;
        if (sample.data() != data_ || sample.size() != n_ || !std::isfinite(sample.maximum())) return -1;
        sample_ = sample; sample_prepared_ = true;
        return prepare(memory, bytes);
    }

    // Borrowed inputs must remain immutable and disjoint from the arena and outputs.
    int prepare(void* memory, std::size_t bytes) noexcept {

        ready_ = false;
        const auto w = workspace_layout(layout(),k_);
        if (!memory || !w.bytes || bytes < w.bytes || k_ != w.k
            || reinterpret_cast<std::uintptr_t>(memory) % alignof(Real)
            || !origin_ || !data_ || !anchors_ || !selected_ || n_ <= layout().maximum_lag
            || !std::all_of(origin_,origin_+w.parameters,[](double x){return std::isfinite(x);})) return -1;
        if (!sample_prepared_) {

            sample_ = Sample::prepare(data_,n_); sample_prepared_ = true;
        }
        if (sample_.data() != data_ || sample_.size() != n_
            || !std::isfinite(sample_.maximum())) return -1;
        last_ = layout().maximum_lag;
        for (std::size_t r = 0; r < w.k; ++r) {
            if (anchors_[r] < layout().maximum_lag || anchors_[r] >= n_
                || selected_[r] >= layout().ma_start) return -1;
            for (std::size_t c = 0; c < r; ++c)
                if (anchors_[r] == anchors_[c] || selected_[r] == selected_[c]) return -1;
            last_ = std::max(last_,anchors_[r]);
        }
        rt::WorkspaceArena arena(memory,bytes);
        diagnostic_history_ = take_sensitivities<Accurate>(arena,w.replay);
        diagnostic_origin_ = take_sensitivities<Accurate>(arena,w.k); diagnostic_actual_ = take_sensitivities<Accurate>(arena,w.k);
        targets_ = take_sensitivities<Real>(arena,w.k); rhs_ = take_sensitivities<Real>(arena,w.k);
        mapped_ = take_sensitivities<Real>(arena,w.k);
        evidence_ = take_sensitivities<double>(arena,w.evidence);
        point_ = take_sensitivities<double>(arena,w.parameters); rounded_ = take_sensitivities<double>(arena,w.parameters);
        (void)arena.take<Real>(0); primal_memory_ = arena.take<std::byte>(w.primal);
        (void)arena.take<Real>(0); chart_memory_ = arena.take<std::byte>(w.chart);
        if (!chart_memory_) return -1;
        Primal primal(layout(),origin_,sample_);

        if (!primal.prepare(primal_memory_,w.primal)) return -3;
        Chart chart(layout(),origin_,data_,n_,anchors_,selected_,w.k,&primal);

        if (!chart.prepare(chart_memory_,w.chart))
            return chart.status() == Status::rank_fallback ? -4 : -3;
        for (std::size_t r = 0; r < w.k; ++r) targets_[r] = chart.anchor_value(r);
        if (w.k == 1 && selected_[0] == 0)
            origin_error_ = -targets_[0]/chart.coefficient_basis(0,0);
        if (!diagnostic_replay(origin_,diagnostic_origin_)) return -3;
        origin_pivot_ = chart.pivot_ratio(); ready_ = true;
        return 0;
    }

    // Selected coordinate slots are rho displacements; every other slot is absolute theta.
    // Evidence per anchor: target, requested, actual, actual displacement (two limbs each).
    int map(const double* coordinates, double* output, double* evidence,
            const Real* anchored_error = nullptr) noexcept {
        const auto w = workspace_layout(layout(),k_);
        if (!ready_ || !coordinates || !output || !evidence
            || (anchored_error && (w.k != 1 || selected_[0] != 0 || !finite(*anchored_error)))
            || !std::all_of(coordinates,coordinates+w.parameters,[](double x){return std::isfinite(x);}))
            return -1;
        std::copy_n(coordinates,w.parameters,point_);
        for (std::size_t r = 0; r < w.k; ++r) point_[selected_[r]] = origin_[selected_[r]];
        Primal primal(layout(),point_,sample_);

        if (!primal.prepare(primal_memory_,w.primal)) return -3;
        Chart chart(layout(),point_,data_,n_,anchors_,selected_,w.k,&primal);

        if (!chart.prepare(chart_memory_,w.chart))
            return chart.status() == Status::rank_fallback ? -4 : -3;
        for (std::size_t r = 0; r < w.k; ++r)
            rhs_[r] = chart.anchor_value(r)-Real(coordinates[selected_[r]])
                + (anchored_error ? chart.coefficient_basis(0,0)*(*anchored_error) : -targets_[r]);
        std::copy_n(coordinates,w.parameters,rounded_);
        if (!chart.solve_basis(rhs_)) return -3;
        for (std::size_t r = 0; r < w.k; ++r) {
            const Real shift = rhs_[r];
            const Real mapped = Real(origin_[selected_[r]])+shift;
            if (!finite(mapped)) return -3;
            mapped_[r] = mapped;
            rounded_[selected_[r]] = shift == Real(0) ? origin_[selected_[r]] : static_cast<double>(mapped);
        }
        if (!std::all_of(rounded_,rounded_+w.parameters,[](double x){return std::isfinite(x);})) return -3;
        // Diagnostics use Octuple; this does not alter the DD affine inversion above.
        // The adaptive primal certifies binary64 rounding, not its low limb as a full DD result.
        if (!diagnostic_replay(rounded_,diagnostic_actual_)) return -3;
        for (std::size_t r = 0; r < w.k; ++r) {
            const Real requested = anchored_error
                ? Real(coordinates[0])-chart.coefficient_basis(0,0)*(*anchored_error)
                : vlk::experiment::mean_replay::narrow(
                    diagnostic_origin_[r]+Accurate(coordinates[selected_[r]])).value;
            if (!finite(requested)) return -3;
        }
        for (std::size_t r = 0; r < w.k; ++r) {
            const Real target = vlk::experiment::mean_replay::narrow(diagnostic_origin_[r]).value;
            const Real requested = anchored_error
                ? Real(coordinates[0])-chart.coefficient_basis(0,0)*(*anchored_error)
                : vlk::experiment::mean_replay::narrow(
                    diagnostic_origin_[r]+Accurate(coordinates[selected_[r]])).value;
            const Real actual = vlk::experiment::mean_replay::narrow(diagnostic_actual_[r]).value;
            const Real displacement = vlk::experiment::mean_replay::narrow(
                diagnostic_actual_[r]-diagnostic_origin_[r]).value;
            if (!finite(target) || !finite(requested) || !finite(actual) || !finite(displacement)) return -3;
            evidence_[8*r] = target.high; evidence_[8*r+1] = target.low;
            evidence_[8*r+2] = requested.high; evidence_[8*r+3] = requested.low;
            evidence_[8*r+4] = actual.high; evidence_[8*r+5] = actual.low;
            evidence_[8*r+6] = displacement.high; evidence_[8*r+7] = displacement.low;
        }
        evidence_[8*w.k] = origin_pivot_; evidence_[8*w.k+1] = chart.pivot_ratio();
        std::copy_n(rounded_,w.parameters,output);
        std::copy_n(evidence_,w.evidence,evidence);
        return 0;
    }
    // Valid until the next map call; never reconstruct this point from rounded output.
    Real mapped_coefficient(std::size_t r) const noexcept { return mapped_[r]; }
    Real origin_residual(std::size_t r) const noexcept { return targets_[r]; }
    Real origin_anchor_error() const noexcept { return origin_error_; }
    const Real* mapped_coefficients() const noexcept { return mapped_; }
    const double* base_point() const noexcept { return point_; }
};
} // namespace vlk::experiment::mean_chart
