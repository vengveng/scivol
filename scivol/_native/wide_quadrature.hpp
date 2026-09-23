// SCIVOL-NATIVE: CORE fixed-width Gauss-Kronrod fallback; density integrands are external.
#pragma once
#include "wide_scalar.hpp"
#include <array>

namespace vlk::experiment::wide {
// https://www.netlib.org/quadpack/dqk31.f
template<std::size_t N>
inline std::array<Real,2*N> concatenate(const std::array<Real,N>& first,
    const std::array<Real,N>& second) {
    std::array<Real,2*N> result;
    for (std::size_t i=0;i<N;++i) {result[i]=first[i];result[N+i]=second[i];}
    return result;
}

struct Kronrod31 {
// Parsed once at module initialization, never at a parameter point.
inline static const std::array<Real,8> nodes_first {
    Real("0.998002298693397060285172840152271"),
    Real("0.987992518020485428489565718586613"),
    Real("0.967739075679139134257347978784337"),
    Real("0.937273392400705904307758947710209"),
    Real("0.897264532344081900882509656454496"),
    Real("0.848206583410427216200648320774217"),
    Real("0.790418501442465932967649294817947"),
    Real("0.724417731360170047416186054613938")};
inline static const std::array<Real,8> nodes_second {
    Real("0.650996741297416970533735895313275"),
    Real("0.570972172608538847537226737253911"),
    Real("0.485081863640239680693655740232351"),
    Real("0.394151347077563369897207370981045"),
    Real("0.299180007153168812166780024266389"),
    Real("0.201194093997434522300628303394596"),
    Real("0.101142066918717499027074231447392"),Real(0)};
inline static const auto nodes=concatenate(nodes_first,nodes_second);
inline static const std::array<Real,8> weights_first {
    Real("0.005377479872923348987792051430128"),
    Real("0.015007947329316122538374763075807"),
    Real("0.025460847326715320186874001019653"),
    Real("0.035346360791375846222037948478360"),
    Real("0.044589751324764876608227299373280"),
    Real("0.053481524690928087265343147239430"),
    Real("0.062009567800670640285139230960803"),
    Real("0.069854121318728258709520077099147")};
inline static const std::array<Real,8> weights_second {
    Real("0.076849680757720378894432777482659"),
    Real("0.083080502823133021038289247286104"),
    Real("0.088564443056211770647275443693774"),
    Real("0.093126598170825321225486872747346"),
    Real("0.096642726983623678505179907627589"),
    Real("0.099173598721791959332393173484603"),
    Real("0.100769845523875595044946662617570"),
    Real("0.101330007014791549017374792767493")};
inline static const auto kronrod=concatenate(weights_first,weights_second);
inline static const std::array<Real,8> gauss {
    Real("0.030753241996117268354628393577204"),
    Real("0.070366047488108124709267416450667"),
    Real("0.107159220467171935011869546685869"),
    Real("0.139570677926154314447804794511028"),
    Real("0.166269205816993933553200860481209"),
    Real("0.186161000015562211026800561866423"),
    Real("0.198431485327111576456118326443839"),
    Real("0.202578241925561272880620199967519")};
};

struct Integral {
    Real value=0, error=0;
    unsigned evaluations=0;
    bool valid=true;
};

template<class Rule,class Function>
Integral integrate(const Function& function, Real tolerance, unsigned budget) {
    constexpr auto pairs=Rule::nodes.size()-1;
    constexpr auto evaluations=2*pairs+1;
    struct Interval { Real left, right, tolerance; unsigned depth; };
    // Depth-first subdivision needs only one pending sibling per depth.
    std::array<Interval, 97> pending;
    pending[0]={Real(0),Real(1),tolerance,0};
    std::size_t count=1;
    Integral result;
    while (count) {
        const auto interval=pending[--count];
        if (result.evaluations+evaluations>budget) return {0,0,result.evaluations,false};
        const auto middle=(interval.left+interval.right)/2;
        const auto half=(interval.right-interval.left)/2;
        const auto center=function(middle);
        Real fine=center*Rule::kronrod.back(), coarse=center*Rule::gauss.back();
        for (std::size_t j=0;j<pairs;++j) {
            const auto offset=half*Rule::nodes[j];
            const auto pair=function(middle-offset)+function(middle+offset);
            fine+=Rule::kronrod[j]*pair;
            if (j%2) coarse+=Rule::gauss[j/2]*pair;
        }
        result.evaluations+=evaluations;
        fine*=half;
        const auto error=fabs(fine-coarse*half);
        if (!finite(fine) || !finite(error))
            return {0,0,result.evaluations,false};
        if (error<=interval.tolerance) {
            result.value+=fine;
            result.error+=error;
        } else {
            if (interval.depth==96 || middle==interval.left || middle==interval.right)
                return {0,0,result.evaluations,false};
            pending[count++]={middle,interval.right,interval.tolerance/2,interval.depth+1};
            pending[count++]={interval.left,middle,interval.tolerance/2,interval.depth+1};
        }
    }
    return result;
}

} // namespace vlk::experiment::wide
