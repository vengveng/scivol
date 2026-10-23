// SCIVOL-NATIVE: CORE
#pragma once
#include "mean_anchor_selection.hpp"
#include "aparch_multi_phase.hpp"

namespace vlk::experiment::runtime::aparch::multi_phase {
template<class Density, class Orders, std::size_t FixedK = mean::dynamic>
struct SelectedPhase {
    static constexpr const ap::Layout* fixed = Orders::fixed
        ? &ap::fixed_layout<Orders,Density,true> : nullptr;
    using Selector = vlk::experiment::mean_chart::anchor_selection::Selector<ap::Layout,fixed,FixedK>;
    using Refinement = Phase<Density,Orders,FixedK>;
    struct Workspace { Plan phase; std::size_t selector = 0, bytes = 0; };
    struct Result { int selection = -1, phase = -1; std::size_t rank = 0; };

    static Workspace workspace(const ap::Layout& l, std::size_t k,
        Strategy strategy = Strategy::held_residuals) noexcept {
        Workspace w;
        w.phase = Refinement::workspace(l,k,strategy);
        w.selector = Selector::workspace_bytes(l,k);
        rt::WorkspaceSizer size;
        if (!w.phase.bytes || !w.selector || !size.add<std::byte>(w.selector)) return {};
        const bool aligned = strategy == Strategy::free_chart
            ? size.add<SelectionAlignment>(0) : size.add<Real>(0);
        if (!aligned || !size.add<std::byte>(w.phase.bytes)) return {};
        w.bytes = size.size();
        return w;
    }

    // The checked caller validates immutable inputs once and owns every output.
    // Selection remains alive until refinement ends: its arena owns both index lists.
    static Result run(const ap::Layout& l, const Workspace& w, const Sample& sample,
        const double* initial, double h0, std::size_t k, std::size_t steps,
        void* memory, std::size_t bytes, std::size_t* anchors, std::size_t* columns,
        double* continuation, double* best, double* result, double* trace) noexcept {
        Result status;
        const bool free = w.phase.strategy == Strategy::free_chart;
        const auto alignment = free ? alignof(SelectionAlignment) : alignof(Real);
        if (!w.bytes || !memory || bytes < w.bytes
            || reinterpret_cast<std::uintptr_t>(memory)%alignment) return status;
        rt::WorkspaceArena arena(memory,bytes);
        auto* selection_memory = arena.take<std::byte>(w.selector);
        if (free) (void)arena.take<SelectionAlignment>(0);
        else (void)arena.take<Real>(0);
        auto* phase_memory = arena.take<std::byte>(w.phase.bytes);
        if (!phase_memory) return status;
        const auto selected = Selector::select(l,initial,sample,k,selection_memory,w.selector);
        status.selection = int(selected.status); status.rank = selected.rank;
        if (selected.status != vlk::experiment::mean_chart::anchor_selection::Status::ready) return status;
        Refinement phase(l,w.phase,sample,h0,selected.anchors,selected.columns);
        if (!phase.bind(phase_memory,w.phase.bytes)) return status;
        status.phase = phase.run(initial,steps,continuation,best,result,trace);
        if (!status.phase) {
            phase.copy_selection(anchors,columns);
        }
        return status;
    }
};
} // namespace vlk::experiment::runtime::aparch::multi_phase
