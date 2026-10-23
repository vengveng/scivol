// TEST_ONLY: model-free mathematical owners, no native backend or fitting policy.
#include "models/mean/mean_coordinate_jet.hpp"
#include <cstdint>
#include <limits>

namespace coordinate_006 {
using namespace vlk::experiment;
struct Layout {
    std::size_t p_ar=0, q_ma=0, maximum_lag=0, ar_start=1, ma_start=0;
    std::size_t mean_parameters=0, parameters=0, packed=0;
    bool valid=false, joint=true;
};
struct Spec {
    std::size_t p, q, start, tail, n, k, kind, mask, fixed, mutation;
};
constexpr Layout make_layout(std::size_t p, std::size_t q,
                             std::size_t start, std::size_t tail) {
    Layout l;
    l.p_ar=p; l.q_ma=q; l.maximum_lag=start;
    std::size_t adjacent=0, product=0;
    l.valid=runtime::checked_sum(1,p,l.ma_start)
        && runtime::checked_sum(l.ma_start,q,l.mean_parameters)
        && runtime::checked_sum(l.mean_parameters,tail,l.parameters)
        && runtime::checked_sum(l.parameters,1,adjacent)
        && runtime::checked_product(l.parameters,adjacent,product);
    l.packed=product/2; return l;
}
template<std::size_t P, std::size_t Q, std::size_t S>
inline constexpr Layout fixed_layout=make_layout(P,Q,S,2);
template<class Result>
constexpr Result invalid_dispatch() {
    if constexpr (std::is_same_v<Result,int>) return -1;
    return 0;
}

template<class Operation>
auto dispatch(const Spec& s, Operation operation) {
    const auto masks=[&]<mean_chart::CoordinateKind Kind, const Layout* Fixed, std::size_t K>() {
#define MASK(M) case M: return operation.template operator()<Kind,M,Fixed,K>();
        switch(s.mask) { MASK(1) MASK(2) MASK(3) MASK(4) MASK(5) MASK(6) MASK(7) }
#undef MASK
        return invalid_dispatch<decltype(operation.template operator()<Kind,1,Fixed,K>())>();
    };
    const auto kinds=[&]<const Layout* Fixed, std::size_t K>() {
        if (s.kind==0) return masks.template operator()<mean_chart::CoordinateKind::anchored,Fixed,K>();
        return masks.template operator()<mean_chart::CoordinateKind::residual,Fixed,K>();
    };
    const auto counts=[&]<const Layout* Fixed>() {
        if constexpr (Fixed != nullptr) {
            if (s.k==1) return kinds.template operator()<Fixed,1>();
            if (s.k==2) return kinds.template operator()<Fixed,2>();
            if (s.k==3) return kinds.template operator()<Fixed,3>();
            return invalid_dispatch<decltype(kinds.template operator()<Fixed,1>())>();
        } else return kinds.template operator()<nullptr,mean::dynamic>();
    };
    if (s.fixed) {
#define FIXED(P,Q,S) if(s.p==P && s.q==Q && s.start==S && s.tail==2) \
        return counts.template operator()<&fixed_layout<P,Q,S>>();
        FIXED(0,0,1) FIXED(1,0,1) FIXED(0,1,1)
        FIXED(4,0,4) FIXED(3,2,3) FIXED(7,5,7)
#undef FIXED
        return invalid_dispatch<decltype(counts.template operator()<nullptr>())>();
    }
    return counts.template operator()<nullptr>();
}
inline Layout supplied(const Spec& s) {
    auto l=make_layout(s.p,s.q,s.start,s.tail);
    if(s.mutation==1) l={};
    if(s.mutation==2) l.packed=std::numeric_limits<std::size_t>::max();
    return l;
}
}

extern "C" std::size_t coordinate_006_workspace(const coordinate_006::Spec* s) {
    if(!s || !s->mask || s->mask>7 || s->kind>1) return 0;
    const auto l=coordinate_006::supplied(*s);
    return coordinate_006::dispatch(*s,[&]<auto Kind,unsigned Mask,auto Fixed,std::size_t K>() {
        using Owner=vlk::experiment::mean_chart::MeanCoordinateJet<Kind,Mask,coordinate_006::Layout,Fixed,K>;
        return Owner::workspace_bytes(l,s->k);
    });
}

extern "C" int coordinate_006_run(const coordinate_006::Spec* s,
    const double* origin, const double* z, const double* data,
    const std::size_t* anchors, const std::size_t* selected,
    void* memory, std::size_t bytes, std::size_t points,
    double* f, double* g, double* h) {
    if(!s || !s->mask || s->mask>7 || s->kind>1) return -1;
    const auto l=coordinate_006::supplied(*s);
    return coordinate_006::dispatch(*s,[&]<auto Kind,unsigned Mask,auto Fixed,std::size_t K>() {
        using namespace vlk::experiment;
        using Owner=mean_chart::MeanCoordinateJet<Kind,Mask,coordinate_006::Layout,Fixed,K>;
        const auto& effective=mean::arma_layout<coordinate_006::Layout,Fixed>(l);
        Owner owner(l,origin,data,s->n,anchors,selected,s->k);
        if(!owner.prepare(memory,bytes)) return owner.status();
        for(std::size_t point=0; point<points; ++point)
            if(!owner.map(z+point*effective.parameters,{
                f ? f+point*effective.parameters : nullptr,
                g ? g+point*effective.parameters*effective.parameters : nullptr,
                h ? h+point*effective.parameters*effective.packed : nullptr})) return owner.status();
        return 0;
    });
}
