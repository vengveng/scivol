// Theta transport only. Filter and anchored-chart obligations are not implemented here.
#include "models/power/aparch_jet.hpp"
#include "models/joint_power/aparch_jet.hpp"

namespace aparch_range_test {
namespace ap = vlk::experiment::runtime::aparch;
namespace joint = vlk::experiment::runtime::joint_aparch;
namespace powers = vlk::experiment::runtime::joint_power;
namespace gen = vlk::experiment::generated;
namespace mean = vlk::experiment::mean;

struct Spec {
    std::size_t ar, ma, p, q, columns;
    unsigned mode, fixed, density;
};

template<class Density>
bool layout(const Spec& s, ap::Layout& result) {
    if (s.mode == 0u) return ap::make_layout<Density>(s.p,s.q,result);
    return powers::make_layout<Density,true>(s.mode==1u?s.ar:0u,
        s.mode==1u?s.ma:0u,s.mode==2u?s.columns:0u,
        s.p,s.q,s.mode==2u,s.mode==1u,result);
}

template<class Operation>
auto density(unsigned code, Operation operation) {
    if (code == 0u) return operation.template operator()<gen::NormalDensity>();
    return operation.template operator()<gen::StudentTDensity>();
}

template<class Operation>
auto request(unsigned mask, Operation operation) {
#define RANGE_REQUEST(M, O, F, G, H) \
    case M: return operation.template operator()<O, F, G, H>()
    switch (mask) {
        RANGE_REQUEST(1, 0, true, false, false);
        RANGE_REQUEST(2, 1, false, true, false);
        RANGE_REQUEST(3, 1, true, true, false);
        RANGE_REQUEST(4, 2, false, false, true);
        RANGE_REQUEST(5, 2, true, false, true);
        RANGE_REQUEST(6, 2, false, true, true);
        RANGE_REQUEST(7, 2, true, true, true);
    }
#undef RANGE_REQUEST
    return decltype(operation.template operator()<0, true, false, false>()){};
}

template<class Density,class Operation>
auto orders(const Spec& s,Operation operation) {
    if (s.mode==0u) {
        if (!s.fixed) return operation.template operator()<Density,nullptr,void>();
        if (s.p==1u && s.q==1u)
            return operation.template operator()<Density,&ap::fixed_layout<1,1,Density>,void>();
        return operation.template operator()<Density,&ap::fixed_layout<3,2,Density>,void>();
    }
    if (s.mode==1u) {
        if (!s.fixed) return operation.template operator()<Density,nullptr,powers::Orders<>>();
        if (s.p==1u && s.q==1u)
            return operation.template operator()<Density,nullptr,powers::Orders<false,true,mean::dynamic,1,1,1,1>>();
        return operation.template operator()<Density,nullptr,powers::Orders<false,true,mean::dynamic,2,1,3,2>>();
    }
    if (!s.fixed) return operation.template operator()<Density,nullptr,powers::Orders<true,false>>();
    return operation.template operator()<Density,nullptr,powers::Orders<true,false,2,0,0,3,2>>();
}

template<class Density,const ap::Layout* Fixed,class Orders,int Order,bool G>
std::size_t workspace(const ap::Layout& l) {
    if constexpr (std::is_void_v<Orders>)
        return ap::core_workspace_bytes<Order,Fixed>(l,G && std::is_same_v<Density,gen::NormalDensity>);
    else {
        constexpr auto* fixed=Orders::fixed?&joint::fixed_layout<Orders,Density,true>:nullptr;
        return joint::core_workspace_bytes<Order,true,fixed>(l,G && std::is_same_v<Density,gen::NormalDensity>);
    }
}

template<class Density,const ap::Layout* Fixed,class Orders,int Order,bool F,bool G,bool H>
bool evaluate(const ap::Layout& l,const double* theta,const double* data,
    const double* features,std::size_t n,double h0,void* memory,std::size_t bytes,
    double* f,double* g,double* h,unsigned* execution) {
    if constexpr (std::is_void_v<Orders>) {
        if constexpr (Fixed != nullptr)
            if (Fixed->p!=l.p || Fixed->q!=l.q || Fixed->parameters!=l.parameters) return false;
        const bool valid=ap::evaluate_order<Order,F,G,H,Density,Fixed>(
            l,theta,data,n,0u,h0,memory,bytes,f,g,h);
        if (valid && execution) *execution=Fixed?ap::execution_fixed:ap::execution_runtime;
        return valid;
    } else {
        const std::size_t begin=Orders::linear?0u:l.maximum_lag;
        return joint::evaluate_theta_order<Order,true,Orders,Density,F,G,H>(
            l,theta,data,n,h0,{begin,begin,{}},memory,bytes,f,g,h,
            features,nullptr,nullptr,static_cast<void*>(nullptr),
            static_cast<void*>(nullptr),static_cast<void*>(nullptr),execution);
    }
}
} // namespace aparch_range_test

extern "C" std::size_t aparch_range_workspace(const aparch_range_test::Spec* s,unsigned mask) {
    using namespace aparch_range_test;
    if (!s || mask==0u || mask>7u || s->mode>2u || s->density>1u) return 0u;
    return density(s->density,[&]<class D>() {
        ap::Layout l;
        if (!layout<D>(*s,l)) return std::size_t{0u};
        return orders<D>(*s,[&]<class Density,const ap::Layout* Fixed,class Orders>() {
            return request(mask,[&]<int O,bool F,bool G,bool H>() {
                return workspace<Density,Fixed,Orders,O,G>(l);
            });
        });
    });
}

extern "C" int aparch_range_evaluate(const aparch_range_test::Spec* s,unsigned mask,
    const double* theta,const double* data,const double* features,std::size_t n,
    double h0,void* memory,std::size_t bytes,double* f,double* g,double* h,
    unsigned* execution) {
    using namespace aparch_range_test;
    if (!s || mask==0u || mask>7u || s->mode>2u || s->density>1u) return -1;
    return density(s->density,[&]<class D>() {
        ap::Layout l;
        if (!layout<D>(*s,l) || !theta || !ap::valid_theta<D>(l,theta)) return -2;
        if (!data || !n || !(h0>0.) || !std::isfinite(h0)
            || (s->mode==1u && n<=l.maximum_lag)
            || (s->mode==2u && s->columns && !features)) return -1;
        for (std::size_t i=0;i<n;++i) if (!std::isfinite(data[i])) return -1;
        return orders<D>(*s,[&]<class Density,const ap::Layout* Fixed,class Orders>() {
            return request(mask,[&]<int O,bool F,bool G,bool H>() {
                return evaluate<Density,Fixed,Orders,O,F,G,H>(l,theta,data,features,n,h0,
                    memory,bytes,f,g,h,execution)?0:-3;
            });
        });
    });
}
