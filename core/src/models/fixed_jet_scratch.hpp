#pragma once

#include "jet_contract.hpp"
#include "../math/runtime_workspace.hpp"
#include <array>
#include <memory>
#include <new>
#include <type_traits>

namespace vlk::experiment::generated {

struct FixedJetWorkspace {
    void* memory=nullptr;
    std::size_t bytes=0;

    [[nodiscard]] std::int32_t admit(std::size_t required) const noexcept {
        if (!required) return VLK_JET_OK;
        if (!memory || bytes<required) return VLK_JET_INSUFFICIENT_WORKSPACE;
        const auto address=reinterpret_cast<std::uintptr_t>(memory);
        if (address%alignof(double) || bytes>std::numeric_limits<std::uintptr_t>::max()-address)
            return VLK_JET_INVALID_ARGUMENT;
        return VLK_JET_OK;
    }
    [[nodiscard]] FixedJetWorkspace tail(std::size_t used) const noexcept {
        if (!used) return *this;
        if (admit(used)!=VLK_JET_OK) return {};
        return {static_cast<std::byte*>(memory)+used,bytes-used};
    }
    [[nodiscard]] bool overlaps(const void* data,std::size_t count) const noexcept {
        if (!data || !count || !memory || !bytes) return false;
        const auto begin=reinterpret_cast<std::uintptr_t>(data);
        const auto own=reinterpret_cast<std::uintptr_t>(memory);
        if (count>std::numeric_limits<std::uintptr_t>::max()-begin) return true;
        return begin<own+bytes && own<begin+count;
    }
};

template<std::size_t Q>
consteval std::size_t affine_remainder_elements() {
    static_assert(Q<std::numeric_limits<std::size_t>::max());
    static_assert(Q+1<=std::numeric_limits<std::size_t>::max()/sizeof(double)/(Q+1),
        "Fixed affine remainder size is not representable");
    return (Q+1)*(Q+1);
}

// T retains its std::array extents. Large storage contains only an arena pointer.
template<class T,std::size_t Budget=4096,bool Stack=(sizeof(T)<=Budget)>
class FixedScratchObject;

template<class T,std::size_t Budget>
class FixedScratchObject<T,Budget,true> {
    T value;
public:
    static constexpr std::size_t workspace_bytes=0;
    explicit FixedScratchObject(FixedJetWorkspace={}) noexcept {}
    [[nodiscard]] bool valid() const noexcept { return true; }
    T& get() noexcept { return value; }
};

template<class T,std::size_t Budget>
class FixedScratchObject<T,Budget,false> {
    T* value=nullptr;
public:
    static_assert(std::is_trivially_destructible_v<T>);
    static_assert(alignof(T)<=alignof(double));
    static constexpr std::size_t workspace_bytes=[] {
        runtime::WorkspaceSizer size;
        return size.template add<T>(1)?size.size():0;
    }();
    static_assert(workspace_bytes>0);
    explicit FixedScratchObject(FixedJetWorkspace workspace={}) noexcept {
        if (workspace.admit(workspace_bytes)!=VLK_JET_OK) return;
        runtime::WorkspaceArena arena(workspace.memory,workspace.bytes);
        if (auto* slot=arena.template take<T>(1))
            value=::new (static_cast<void*>(slot)) T;
    }
    [[nodiscard]] bool valid() const noexcept { return value!=nullptr; }
    T& get() noexcept { return *value; }
};

template<std::size_t P,std::size_t Q,std::size_t V,std::size_t K,unsigned Order,bool Gradient>
struct FixedGarchBuffers {
    static constexpr auto limit=std::numeric_limits<std::size_t>::max();
    static_assert(V>0 && K>=V && V<limit && K<limit);
    static_assert(V<=limit/(V+1) && K<=limit/(K+1));
    static constexpr std::size_t V2=V*(V+1)/2, K2=K*(K+1)/2;
    static_assert(P<=limit/sizeof(double) && Q<=limit/sizeof(double));
    static_assert(Order<1 || Q<=limit/sizeof(double)/V);
    static_assert(Order<2 || Q<=limit/sizeof(double)/V2);
    std::array<double,(Gradient?K:0)> accumulated_gradient{};
    std::array<double,(Order==2?K2:0)> accumulated_hessian{};
    std::array<double,P> residual_ring{};
    std::array<double,Q> variance_ring{};
    std::array<std::array<double,V>,(Order>=1?Q:0)> first_ring{};
    std::array<std::array<double,V2>,(Order==2?Q:0)> second_ring{};
    std::array<double,(Order>=1?V:0)> initial_first{};
    std::array<double,(Order==2?V2:0)> initial_second{};
    std::array<double,(Order>=1?V:0)> current_first;
    std::array<double,(Order==2?V2:0)> current_second;
};

template<std::size_t K,std::size_t S,unsigned Order,bool GradientMatrix>
struct FixedTransformBuffers {
    static_assert(K>0 && K<=std::numeric_limits<std::size_t>::max()/sizeof(double)/K);
    std::array<double,K> theta{};
    std::array<double,(Order>=1?K:0)> theta_gradient{};
    std::array<double,(Order==2?K*K:0)> theta_hessian{};
    std::array<double,(Order>=1?S:0)> shifted_weights;
    std::array<double,((Order==2 || (Order==1 && GradientMatrix))?K*K:0)> matrix{};
    std::array<double,(Order==2?K*K:0)> intermediate{};
};

[[nodiscard]] constexpr std::size_t combined_fixed_workspace(std::size_t first,std::size_t second) {
    std::size_t total=0;
    return runtime::checked_sum(first,second,total)?total:std::numeric_limits<std::size_t>::max();
}

template<class Jet,class Sample>
[[nodiscard]] std::int32_t admit_fixed_workspace(FixedJetWorkspace workspace,const Sample& sample,
    const double* parameters,std::size_t n,unsigned request,double* f,double* g,double* h) noexcept
{
    const auto required=Jet::workspace_bytes(request);
    const auto status=workspace.admit(required);
    if (status!=VLK_JET_OK || !required) return status;
    workspace.bytes=required;
    std::size_t sample_bytes=0;
    if (!runtime::checked_product(n,sizeof(double),sample_bytes)) return VLK_JET_INVALID_ARGUMENT;
    static_assert(Jet::K>0 && Jet::K<=std::numeric_limits<std::size_t>::max()/sizeof(double)/Jet::K);
    if (workspace.overlaps(parameters,Jet::K*sizeof(double))
        || workspace.overlaps(sample.residuals,sample_bytes)
        || workspace.overlaps(sample.residuals2,sample_bytes)
        || sample.history.overlaps(workspace.memory,required)
        || workspace.overlaps(f,sizeof(double))
        || workspace.overlaps(g,Jet::K*sizeof(double))
        || workspace.overlaps(h,Jet::K*Jet::K*sizeof(double)))
        return VLK_JET_INVALID_ARGUMENT;
    return VLK_JET_OK;
}

} // namespace vlk::experiment::generated
