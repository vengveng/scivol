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

// QUADPACK dqk61, Piessens/de Doncker; Fullerton's table (1981).
// https://www.netlib.org/quadpack/dqk61.f
struct Kronrod61 {
inline static const std::array<Real,31> nodes {
    Real("0.999484410050490637571325895705811"),
    Real("0.996893484074649540271630050918695"),
    Real("0.991630996870404594858628366109486"),
    Real("0.983668123279747209970032581605663"),
    Real("0.973116322501126268374693868423707"),
    Real("0.960021864968307512216871025581798"),
    Real("0.944374444748559979415831324037439"),
    Real("0.926200047429274325879324277080474"),
    Real("0.905573307699907798546522558925958"),
    Real("0.882560535792052681543116462530226"),
    Real("0.857205233546061098958658510658944"),
    Real("0.829565762382768397442898119732502"),
    Real("0.799727835821839083013668942322683"),
    Real("0.767777432104826194917977340974503"),
    Real("0.733790062453226804726171131369528"),
    Real("0.697850494793315796932292388026640"),
    Real("0.660061064126626961370053668149271"),
    Real("0.620526182989242861140477556431189"),
    Real("0.579345235826361691756024932172540"),
    Real("0.536624148142019899264169793311073"),
    Real("0.492480467861778574993693061207709"),
    Real("0.447033769538089176780609900322854"),
    Real("0.400401254830394392535476211542661"),
    Real("0.352704725530878113471037207089374"),
    Real("0.304073202273625077372677107199257"),
    Real("0.254636926167889846439805129817805"),
    Real("0.204525116682309891438957671002025"),
    Real("0.153869913608583546963794672743256"),
    Real("0.102806937966737030147096751318001"),
    Real("0.051471842555317695833025213166723"),Real(0)};
inline static const std::array<Real,31> kronrod {
    Real("0.001389013698677007624551591226760"),
    Real("0.003890461127099884051267201844516"),
    Real("0.006630703915931292173319826369750"),
    Real("0.009273279659517763428441146892024"),
    Real("0.011823015253496341742232898853251"),
    Real("0.014369729507045804812451432443580"),
    Real("0.016920889189053272627572289420322"),
    Real("0.019414141193942381173408951050128"),
    Real("0.021828035821609192297167485738339"),
    Real("0.024191162078080601365686370725232"),
    Real("0.026509954882333101610601709335075"),
    Real("0.028754048765041292843978785354334"),
    Real("0.030907257562387762472884252943092"),
    Real("0.032981447057483726031814191016854"),
    Real("0.034979338028060024137499670731468"),
    Real("0.036882364651821229223911065617136"),
    Real("0.038678945624727592950348651532281"),
    Real("0.040374538951535959111995279752468"),
    Real("0.041969810215164246147147541285970"),
    Real("0.043452539701356069316831728117073"),
    Real("0.044814800133162663192355551616723"),
    Real("0.046059238271006988116271735559374"),
    Real("0.047185546569299153945261478181099"),
    Real("0.048185861757087129140779492298305"),
    Real("0.049055434555029778887528165367238"),
    Real("0.049795683427074206357811569379942"),
    Real("0.050405921402782346840893085653585"),
    Real("0.050881795898749606492297473049805"),
    Real("0.051221547849258772170656282604944"),
    Real("0.051426128537459025933862879215781"),
    Real("0.051494729429451567558340433647099")};
// Gauss-30 has no center node. The zero adapts the shared integrator's interface.
inline static const std::array<Real,16> gauss {
    Real("0.007968192496166605615465883474674"),
    Real("0.018466468311090959142302131912047"),
    Real("0.028784707883323369349719179611292"),
    Real("0.038799192569627049596801936446348"),
    Real("0.048402672830594052902938140422808"),
    Real("0.057493156217619066481721689402056"),
    Real("0.065974229882180495128128515115962"),
    Real("0.073755974737705206268243850022191"),
    Real("0.080755895229420215354694938460530"),
    Real("0.086899787201082979802387530715126"),
    Real("0.092122522237786128717632707087619"),
    Real("0.096368737174644259639468626351810"),
    Real("0.099593420586795267062780282103569"),
    Real("0.101762389748405504596428952168554"),
    Real("0.102852652893558840341285636705415"),Real(0)};
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
