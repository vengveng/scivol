// SCIVOL-NATIVE: CORE
#pragma once
#include "residual_mean_chart.hpp"
#include "native_request_accounting.hpp"

#include "third_party/scipy_slsqp/scipy_blas_defines.h"
extern "C" {
double vlk_dnrm2(CBLAS_INT*,double*,CBLAS_INT*);
void vlk_dgeqr2(CBLAS_INT*,CBLAS_INT*,double*,CBLAS_INT*,double*,double*,CBLAS_INT*);
void vlk_dlarf(const char*,CBLAS_INT*,CBLAS_INT*,double*,CBLAS_INT*,double*,double*,CBLAS_INT*,double*);
}

namespace vlk::experiment::mean_chart::anchor_selection {
namespace mean = vlk::experiment::mean;
namespace chart = vlk::experiment::mean_chart;
namespace rt = vlk::experiment::runtime;
using Real = vlk::experiment::wide::Real;

enum class Status : int { ready = 0, invalid = -1, rank_failure = -2,
    nonfinite = -3, precision_unresolved = -4, backend_failure = -5 };

using vlk::experiment::wide::finite;

struct SelectionLayout {
    std::size_t columns = 0, k = 0, design = 0, history = 0, bytes = 0;
    constexpr bool operator==(const SelectionLayout&) const = default;
};

template<class Layout>
constexpr SelectionLayout make_selection_layout(const Layout& l, std::size_t k) noexcept {
    SelectionLayout w;
    if (!mean::valid_arma_mean_layout(l) || !k || k > l.ma_start
        || l.ma_start > std::size_t(std::numeric_limits<CBLAS_INT>::max())) return w;
    w.columns = l.ma_start; w.k = k;
    rt::WorkspaceSizer s;
    if (!rt::checked_product(k, w.columns, w.design)
        || !rt::checked_product(l.q_ma, w.columns, w.history)
        || !s.add<std::size_t>(k) || !s.add<std::size_t>(w.columns)
        || !s.add<Real>(k) || !s.add<double>(w.design)
        || !s.add<double>(w.columns) || !s.add<double>(k)
        || !s.add<Real>(w.history) || !s.add<Real>(w.columns)) return {};
    w.bytes = s.size();
    return w;
}

struct Selection {
    Status status = Status::invalid;
    std::size_t rank = 0;
    const std::size_t *anchors = nullptr, *columns = nullptr;
    const Real* residuals = nullptr;
    const double* diagonal = nullptr;
    double design_scale = 0, chart_pivot_ratio = 0;
};

// Pivoting orchestrates the existing native norm, reflector and application owners.
// matrix is column-major k x columns; all scratch is caller-owned.
template<std::size_t FixedRows = mean::dynamic, std::size_t FixedColumns = mean::dynamic>
inline Status select_columns(double* matrix, std::size_t runtime_k, std::size_t runtime_columns,
    std::size_t* permutation, double* work, double* diagonal,
    std::size_t& rank, double& scale) noexcept {
    const auto k = mean::extent<FixedRows>(runtime_k);
    const auto columns = mean::extent<FixedColumns>(runtime_columns);
    rank = 0; scale = 0;
    if (!matrix || !permutation || !work || !diagonal || !k || k > columns
        || columns > std::size_t(std::numeric_limits<CBLAS_INT>::max())) return Status::invalid;
    for (std::size_t c = 0; c < columns; ++c) {
        permutation[c] = c;
        for (std::size_t r = 0; r < k; ++r) {
            const double value = matrix[r+c*k];
            if (!std::isfinite(value)) return Status::nonfinite;
            scale = std::max(scale, std::abs(value));
        }
    }
    // Same declared chart-selection cutoff as the MP120/SciPy pilot.
    const double cutoff = 64 * std::numeric_limits<double>::epsilon() * scale;
    CBLAS_INT one = 1, lda = CBLAS_INT(k);
    for (std::size_t p = 0; p < k; ++p) {
        CBLAS_INT rows = CBLAS_INT(k-p);
        std::size_t pivot = p;
        double largest = -1;
        for (std::size_t c = p; c < columns; ++c) {
            const double norm = vlk_dnrm2(&rows, matrix+p+c*k, &one);
            if (!std::isfinite(norm)) return Status::nonfinite;
            if (norm > largest) { largest = norm; pivot = c; }
        }
        if (pivot != p) {
            for (std::size_t r = 0; r < k; ++r) std::swap(matrix[r+p*k], matrix[r+pivot*k]);
            std::swap(permutation[p], permutation[pivot]);
        }
        double tau = 0;
        CBLAS_INT info = 0;
        vlk_dgeqr2(&rows, &one, matrix+p+p*k, &lda, &tau, work, &info);
        if (info) return Status::backend_failure;
        const double value = matrix[p+p*k];
        diagonal[p] = std::abs(value);
        if (!std::isfinite(tau) || !std::isfinite(value)) return Status::nonfinite;
        if (!(diagonal[p] > cutoff)) return Status::rank_failure;
        ++rank;
        CBLAS_INT remaining = CBLAS_INT(columns-p-1);
        if (remaining) {
            matrix[p+p*k] = 1;
            vlk_dlarf("L", &rows, &remaining, matrix+p+p*k, &one, &tau,
                matrix+p+(p+1)*k, &lda, work);
            matrix[p+p*k] = value;
        }
        for (std::size_t c = p+1; c < columns; ++c)
            for (std::size_t r = p; r < k; ++r)
                if (!std::isfinite(matrix[r+c*k])) return Status::nonfinite;
    }
    return Status::ready;
}

template<class Layout, const Layout* Fixed = nullptr, std::size_t FixedK = mean::dynamic>
struct Selector {
    using Primal = mean::ArmaResidualPrecision<Layout, Fixed>;
    using Chart = chart::ResidualPreservingChart<1, Layout, Fixed, FixedK,
        chart::PreparedIntention::value_basis>;
    static constexpr SelectionLayout fixed_selection = [] {
        if constexpr (Fixed != nullptr && FixedK != mean::dynamic)
            return make_selection_layout(*Fixed, FixedK);
        return SelectionLayout{};
    }();

    static constexpr SelectionLayout selection_layout(const Layout& supplied, std::size_t k) noexcept {
        if constexpr (Fixed != nullptr && FixedK != mean::dynamic) return fixed_selection;
        return make_selection_layout(mean::arma_layout<Layout, Fixed>(supplied), mean::extent<FixedK>(k));
    }
    static constexpr SelectionLayout effective_selection(const SelectionLayout& supplied) noexcept {
        if constexpr (Fixed != nullptr && FixedK != mean::dynamic) return fixed_selection;
        return supplied;
    }
    static std::size_t reusable_bytes(const Layout& l, std::size_t k) noexcept {
        return Chart::workspace_bytes(l,k);
    }
    static std::size_t workspace_bytes(const Layout& supplied, std::size_t k) noexcept {
        const auto& l = mean::arma_layout<Layout, Fixed>(supplied);
        const auto w = selection_layout(l,k);
        rt::WorkspaceSizer s;
        if (!w.bytes || k != w.k || !s.add<std::byte>(w.bytes)
            || !s.add<Real>(0) || !s.add<std::byte>(Primal::workspace_bytes(l))
            || !s.add<Real>(0) || !s.add<std::byte>(reusable_bytes(l,k))) return 0;
        return s.size();
    }

    // Only an explicit development evidence request materializes residuals.
    static Status retain_smallest(Primal& primal, const Layout& supplied,
        std::size_t n, const SelectionLayout& supplied_workspace, std::size_t* anchors,
        Real* residuals, Real* evidence) noexcept {
        const auto& l = mean::arma_layout<Layout, Fixed>(supplied);
        const auto w = effective_selection(supplied_workspace);
        std::fill_n(anchors,w.k,n);
        std::fill_n(residuals,w.k,Real(INFINITY));
        primal.rewind();
        for (std::size_t t = l.maximum_lag; t < n; ++t) {
            const Real residual = primal.advance(t);
            if (!finite(residual)) return Status::precision_unresolved;
            if (evidence) evidence[t] = residual;
            std::size_t time = t;
            Real value = residual;
            for (std::size_t r = 0; r < w.k; ++r) {
                using vlk::experiment::wide::abs;
                if (abs(value) < abs(residuals[r])) {
                    std::swap(time,anchors[r]); std::swap(value,residuals[r]);
                }
            }
        }
        return Status::ready;
    }

    static __attribute__((noinline)) Status prepare_design(const Layout& supplied,
        const double* theta, const double* data, std::size_t n,
        const SelectionLayout& supplied_workspace, const std::size_t* anchors,
        double* design, Real* history, Real* current) noexcept {
        const auto& l = mean::arma_layout<Layout, Fixed>(supplied);
        const auto w = effective_selection(supplied_workspace);
        std::fill_n(history,w.history,Real(0));
        const auto last = *std::max_element(anchors,anchors+w.k);
        if (last>=n) return Status::invalid;
        mean::ArmaDesign<Fixed!=nullptr ? Fixed->p_ar : mean::dynamic> forcing{data,l.p_ar};
        for (std::size_t t=l.maximum_lag;t<=last;++t) {
            for (std::size_t c = 0; c < w.columns; ++c) {
                current[c] = mean::inverse_ma_value<Real,Layout,Fixed>(Real(forcing.at(t,c)),l,theta,
                    [&](std::size_t lag) { return history[((t-1-lag)%l.q_ma)*w.columns+c]; });
                if (!finite(current[c])) return Status::nonfinite;
            }
            for (std::size_t r=0;r<w.k;++r) if (t==anchors[r])
                for (std::size_t c=0;c<w.columns;++c) {
                    design[r+c*w.k]=static_cast<double>(current[c]);
                    if (!std::isfinite(design[r+c*w.k])) return Status::nonfinite;
                }
            if (l.q_ma) std::copy_n(current,w.columns,history+(t%l.q_ma)*w.columns);
        }
        return Status::ready;
    }

    // The checked ABI owns buffer/sample validation. A native controller may pass
    // its once-per-solve ObservationSummary here; no point/request allocation.
    static Selection select(const Layout& supplied, const double* theta,
        const vlk::experiment::mean_screen::ObservationSummary& sample,
        std::size_t k, void* memory, std::size_t bytes,
        vlk::experiment::mean_replay::Counts* counts = nullptr,
        Real* residual_evidence = nullptr, double* design_evidence = nullptr) noexcept {
        Selection result;
        const auto& l = mean::arma_layout<Layout, Fixed>(supplied);
        const auto w = selection_layout(l,k);
        const auto required = workspace_bytes(l,k);
        if (!required || !theta || !sample.data() || sample.size() <= l.maximum_lag
            || sample.size()-l.maximum_lag < k || !memory || bytes < required
            || reinterpret_cast<std::uintptr_t>(memory) % 16) return result;
        rt::WorkspaceArena arena(memory,bytes);
        auto* anchors = arena.take<std::size_t>(w.k);
        auto* columns = arena.take<std::size_t>(w.columns);
        auto* residuals = arena.take<Real>(w.k);
        auto* design = arena.take<double>(w.design);
        auto* work = arena.take<double>(w.columns);
        auto* diagonal = arena.take<double>(w.k);
        auto* history = arena.take<Real>(w.history);
        auto* current = arena.take<Real>(w.columns);
        (void)arena.take<Real>(0);
        auto* primal_memory = arena.take<std::byte>(Primal::workspace_bytes(l));
        (void)arena.take<Real>(0);
        auto* scratch = arena.take<std::byte>(reusable_bytes(l,k));
        if (!scratch) return result;
        Primal primal(l,theta,sample,counts);
        request_accounting::prepare(request_accounting::Preparation::mean_primal,0u);
        if (!primal.prepare(primal_memory,Primal::workspace_bytes(l))) return result;
        result.status = retain_smallest(primal,l,sample.size(),w,anchors,residuals,residual_evidence);
        if (result.status != Status::ready) return result;
        result.status = prepare_design(l,theta,sample.data(),sample.size(),w,
            anchors,design,history,current);
        if (result.status != Status::ready) return result;
        if (design_evidence) std::copy_n(design,w.design,design_evidence);
        result.status = select_columns<FixedK,Fixed!=nullptr ? Fixed->ma_start : mean::dynamic>(
            design,w.k,w.columns,columns,work,diagonal,
            result.rank,result.design_scale);
        if (result.status != Status::ready) return result;
        Chart canonical(l,theta,sample.data(),sample.size(),anchors,columns,k,&primal);
        request_accounting::prepare(request_accounting::Preparation::residual_chart,1u);
        if (!canonical.prepare(scratch,reusable_bytes(l,k))) {
            result.status = canonical.status() == chart::Status::rank_fallback
                ? Status::rank_failure : Status::nonfinite;
            return result;
        }
        result.anchors = anchors; result.columns = columns;
        result.residuals = residuals; result.diagonal = diagonal;
        result.chart_pivot_ratio = canonical.pivot_ratio();
        return result;
    }
};
} // namespace vlk::experiment::mean_chart::anchor_selection
