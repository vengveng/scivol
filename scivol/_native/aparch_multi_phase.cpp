// SCIVOL-NATIVE: CORE
#include "aparch_multi_phase_api.hpp"
#include "aparch_selected_phase.hpp"

namespace vlk::experiment::runtime::aparch::multi_phase::detail {
using Narrow = ap::FixedJointOrders<1,1,1,1>;
using Wide = ap::FixedJointOrders<2,1,3,2>;

template<class Call>
auto density(std::uint32_t code, Call call) noexcept {
#define VLK_MULTI_DENSITY(CODE, POLICY) case CODE: return call.template operator()<gen::POLICY>()
    switch (code) {
        VLK_MULTI_DENSITY(0, NormalDensity);
        VLK_MULTI_DENSITY(1, StudentTDensity);
        VLK_MULTI_DENSITY(2, SkewTDensity);
        VLK_MULTI_DENSITY(3, GedDensity);
        VLK_MULTI_DENSITY(4, GeneralizedHyperbolicDensity);
        VLK_MULTI_DENSITY(5, NormalInverseGaussianDensity);
        VLK_MULTI_DENSITY(6, HyperbolicDensity);
        VLK_MULTI_DENSITY(7, VarianceGammaDensity);
    }
#undef VLK_MULTI_DENSITY
    return decltype(call.template operator()<gen::NormalDensity>()){};
}

bool specification_valid(const Specification* spec) noexcept {
    bd::Range range;
    return spec && reinterpret_cast<std::uintptr_t>(spec)%alignof(Specification) == 0
        && bd::memory_range(spec,sizeof(*spec),range) && spec->density <= 7 && spec->k;
}

template<class D>
bool layout(const Specification& spec, ap::Layout& l) noexcept {
    return ap::make_layout<D>(spec.ar,spec.ma,spec.p,spec.q,true,l)
        && spec.k <= l.ma_start && extents(l,spec.k).n;
}

int kind(const Specification& spec, const ap::Layout& l) noexcept {
    if (spec.force_runtime) return 0;
    if (spec.k == 2 && ap::matches_orders<Wide>(l)) return 1;
    if ((spec.k == 1 || spec.k == 2) && ap::matches_orders<Narrow>(l)) return 2;
    return 0;
}

template<class D, class Call>
auto orders(const Specification& spec, const ap::Layout& l, Call call) noexcept {
    if (kind(spec,l) == 1) return call.template operator()<SelectedPhase<D,Wide,2>>();
    if (kind(spec,l) == 2) {
        if (spec.k == 2) return call.template operator()<SelectedPhase<D,Narrow,2>>();
        return call.template operator()<SelectedPhase<D,Narrow,1>>();
    }
    return call.template operator()<SelectedPhase<D,ap::RuntimeOrders>>();
}

struct Buffer { const void* data; std::size_t count, item_size, alignment; };
bool disjoint(const Buffer (&buffers)[10]) noexcept {
    bd::Range ranges[10];
    for (std::size_t i=0; i<10; ++i) {
        std::size_t bytes=0;
        if (!buffers[i].data || reinterpret_cast<std::uintptr_t>(buffers[i].data)%buffers[i].alignment
            || !rt::checked_product(buffers[i].count,buffers[i].item_size,bytes)
            || !bd::memory_range(buffers[i].data,bytes,ranges[i])) return false;
        for (std::size_t j=0; j<i; ++j)
            if (bd::intersects(ranges[i],ranges[j])) return false;
    }
    return true;
}
}

using namespace vlk::experiment::runtime::aparch::multi_phase;

extern "C" std::size_t vlk_arma_aparch_multi_phase_workspace_bytes(const Specification* spec) {
    return vlk_arma_aparch_multi_phase_strategy_workspace_bytes(spec,std::uint32_t(Strategy::held_residuals));
}

extern "C" std::size_t vlk_arma_aparch_multi_phase_strategy_workspace_bytes(
    const Specification* spec, std::uint32_t strategy) {
    if (!detail::specification_valid(spec) || strategy > std::uint32_t(Strategy::free_chart)) return 0;
    return detail::density(spec->density,[&]<class D>() -> std::size_t {
        ap::Layout l;
        if (!detail::layout<D>(*spec,l)) return 0;
        return detail::orders<D>(*spec,l,[&]<class P>() { return P::workspace(l,spec->k,Strategy(strategy)).bytes; });
    });
}

extern "C" std::int32_t vlk_arma_aparch_multi_phase_dispatch_kind(const Specification* spec) {
    if (!detail::specification_valid(spec)) return -1;
    return detail::density(spec->density,[&]<class D>() {
        ap::Layout l;
        return detail::layout<D>(*spec,l) ? detail::kind(*spec,l) : -1;
    });
}

extern "C" std::int32_t vlk_arma_aparch_multi_phase_run(
    const Specification* spec, const double* initial, std::size_t parameter_count,
    const double* observations, std::size_t observation_count, double initial_variance,
    std::size_t maximum_steps, void* workspace, std::size_t workspace_bytes,
    std::size_t* anchors, std::size_t anchor_count, std::size_t* columns, std::size_t column_count,
    double* continuation, std::size_t continuation_count, double* best, std::size_t best_count,
    double* summary, std::size_t summary_capacity, double* trace, std::size_t trace_count) {
    return vlk_arma_aparch_multi_phase_strategy_run(spec,initial,parameter_count,
        observations,observation_count,initial_variance,maximum_steps,workspace,workspace_bytes,
        anchors,anchor_count,columns,column_count,continuation,continuation_count,best,best_count,
        summary,summary_capacity,trace,trace_count,std::uint32_t(Strategy::held_residuals));
}

extern "C" std::int32_t vlk_arma_aparch_multi_phase_strategy_run(
    const Specification* spec, const double* initial, std::size_t parameter_count,
    const double* observations, std::size_t observation_count, double initial_variance,
    std::size_t maximum_steps, void* workspace, std::size_t workspace_bytes,
    std::size_t* anchors, std::size_t anchor_count, std::size_t* columns, std::size_t column_count,
    double* continuation, std::size_t continuation_count, double* best, std::size_t best_count,
    double* summary, std::size_t summary_capacity, double* trace, std::size_t trace_count,
    std::uint32_t strategy) {
    vlk::experiment::request_accounting::PhaseScope accounting_phase(
        vlk::experiment::request_accounting::Phase::refinement);
    if (!detail::specification_valid(spec) || strategy > std::uint32_t(Strategy::free_chart) || !maximum_steps
        || !std::isfinite(initial_variance) || initial_variance <= 0) return -1;
    return detail::density(spec->density,[&]<class D>() {
        ap::Layout l;
        if (!detail::layout<D>(*spec,l) || parameter_count != l.parameters
            || observation_count <= l.maximum_lag || observation_count-l.maximum_lag < spec->k
            || anchor_count < spec->k || column_count < spec->k
            || continuation_count < l.parameters || best_count < l.parameters
            || summary_capacity < (strategy == std::uint32_t(Strategy::free_chart)
                ? free_summary_count : summary_count)) return -1;
        std::size_t required_trace=0;
        if (!rt::checked_product(maximum_steps,extents(l,spec->k).trace,required_trace)
            || trace_count < required_trace) return -1;
        const detail::Buffer buffers[] = {
            {initial,parameter_count,sizeof(double),alignof(double)},
            {observations,observation_count,sizeof(double),alignof(double)},
            {workspace,workspace_bytes,1,strategy == std::uint32_t(Strategy::free_chart)
                ? alignof(SelectionAlignment) : alignof(Real)},
            {anchors,anchor_count,sizeof(std::size_t),alignof(std::size_t)},
            {columns,column_count,sizeof(std::size_t),alignof(std::size_t)},
            {continuation,continuation_count,sizeof(double),alignof(double)},
            {best,best_count,sizeof(double),alignof(double)},
            {summary,summary_capacity,sizeof(double),alignof(double)},
            {trace,trace_count,sizeof(double),alignof(double)},
            {spec,1,sizeof(*spec),alignof(Specification)}};
        if (!detail::disjoint(buffers)) return -1;
        return detail::orders<D>(*spec,l,[&]<class P>() {
            const auto plan = P::workspace(l,spec->k,Strategy(strategy));
            if (!plan.bytes || workspace_bytes < plan.bytes) return -1;
            if (!std::all_of(initial,initial+parameter_count,[](double v){return std::isfinite(v);})) return -1;
            // One immutable-sample scan, reused by selection, mapping and every jet request.
            vlk::experiment::request_accounting::prepare(
                vlk::experiment::request_accounting::Preparation::sample,0u);
            const auto sample = Sample::prepare(observations,observation_count);
            if (!std::isfinite(sample.maximum())) return -1;
            const auto status = P::run(l,plan,sample,initial,initial_variance,spec->k,maximum_steps,
                workspace,workspace_bytes,anchors,columns,continuation,best,summary,trace);
            return status.selection ? status.selection-10 : status.phase;
        });
    });
}
