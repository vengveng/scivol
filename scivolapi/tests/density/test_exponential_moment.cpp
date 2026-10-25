#include "density/exponential_moment.hpp"
#include "density/domain.hpp"
#include "density/ged.hpp"
#include "density/gh.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

namespace {
bool watching = false;
std::size_t allocations = 0;
}
void* operator new(std::size_t size) {
    if (watching) ++allocations;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
using Status = ExpectationStatus;
constexpr const char* names[] = {"Normal", "Student-t", "Skew-t", "GED", "GH", "NIG", "Hyperbolic", "VG"};
constexpr double sentinel = 12345.0;
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
const double shapes[8][3] = {{}, {8}, {8,.3}, {1.5}, {.2,1.3,.8}, {.2,1.3}, {.2,1.3}, {.2,2.3}};
int failures = 0;
struct Reference {
    int kind;
    double shape[3], a, b, expected;
    double relative_tolerance = 1e-12;
};
// Independent mpmath references at 50 decimal digits; inputs promoted from binary64.
constexpr Reference references[] = {
    {0, {}, 0, 0, 1},
    {0, {}, 0.20000000000000001, 0, 1.1819230635865643},
    {0, {}, 0.20000000000000001, -0.29999999999999999, 1.2460084036339638},
    {0, {}, 0, 0.69999999999999996, 1.2776213132048866},
    {0, {}, -0.20000000000000001, 0.10000000000000001, 0.86215408263079074},
    {0, {}, -10, 3, 0.086402896935797696},
    {0, {}, -40, -5, 0.020250081298545917},
    {0, {}, 10, -3, 4.9875093266256084e+36},
    {0, {}, 37, 0, 3.7635947880871667e+297},
    {1, {2.0499999999999998}, -0.29999999999999999, 0.10000000000000001, 0.94244411181002441},
    {1, {2.0499999999999998}, -0.20000000000000001, -0.20000000000000001, 0.96237960423116164},
    {1, {3}, -0.29999999999999999, 0.10000000000000001, 0.84423027722089328},
    {1, {3}, -0.20000000000000001, -0.20000000000000001, 0.89998958666189055},
    {1, {8}, -0.29999999999999999, 0.10000000000000001, 0.81138361235647039},
    {1, {8}, -0.20000000000000001, -0.20000000000000001, 0.87902199593105368},
    {1, {50}, -0.29999999999999999, 0.10000000000000001, 0.80366272279323736},
    {1, {50}, -0.20000000000000001, -0.20000000000000001, 0.87402173288654783},
    {2, {2.1000000000000001,0.90000000000000002}, -0.29999999999999999, 0.10000000000000001, 0.92992382741418111},
    {2, {2.1000000000000001,0.90000000000000002}, -0.20000000000000001, -0.20000000000000001, 0.96141802369367901},
    {2, {2.1000000000000001,0.90000000000000002}, -0.5, 0, 0.89383596904789042},
    {2, {3,-0.90000000000000002}, -0.29999999999999999, 0.10000000000000001, 0.85914096046263422},
    {2, {3,-0.90000000000000002}, -0.20000000000000001, -0.20000000000000001, 0.89290936817363442},
    {2, {3,-0.90000000000000002}, -0.5, 0, 0.77110033808714817},
    {2, {8,0.29999999999999999}, -0.29999999999999999, 0.10000000000000001, 0.80769252711757134},
    {2, {8,0.29999999999999999}, -0.20000000000000001, -0.20000000000000001, 0.88303290854150274},
    {2, {8,0.29999999999999999}, -0.5, 0, 0.71164044932413528},
    {2, {30,0}, -0.29999999999999999, 0.10000000000000001, 0.80447517252976719},
    {2, {30,0}, -0.20000000000000001, -0.20000000000000001, 0.87454932626579185},
    {2, {30,0}, -0.5, 0, 0.7022723917013507},
    {3, {0.69999999999999996}, -0.29999999999999999, 0.10000000000000001, 0.84772359046358059},
    {3, {0.69999999999999996}, -0.20000000000000001, -0.20000000000000001, 0.90268133553176155},
    {3, {0.69999999999999996}, -0.5, 0, 0.76966491706504958},
    {3, {1}, -0.29999999999999999, 0.10000000000000001, 0.82780972384458107},
    {3, {1}, -0.20000000000000001, -0.20000000000000001, 0.88975939539422877},
    {3, {1}, -0.5, 0, 0.73879612503625858},
    {3, {1}, 0.20000000000000001, 0, 1.1647156696299077},
    {3, {1}, 0, 0.40000000000000002, 1.0869565217391304},
    {3, {1}, 0.20000000000000001, -0.29999999999999999, 1.2404386390733579},
    {3, {1.5}, -0.29999999999999999, 0.10000000000000001, 0.81106135487234743},
    {3, {1.5}, -0.20000000000000001, -0.20000000000000001, 0.87885316862949447},
    {3, {1.5}, -0.5, 0, 0.71264324980461269},
    {3, {1.5}, 0.20000000000000001, 0, 1.1760900551250939},
    {3, {1.5}, 0, 0.40000000000000002, 1.0841711888514682},
    {3, {1.5}, 0.20000000000000001, -0.29999999999999999, 1.2430829885119552},
    {3, {3}, -0.29999999999999999, 0.10000000000000001, 0.79423317602355092},
    {3, {3}, -0.20000000000000001, -0.20000000000000001, 0.86786474144531844},
    {3, {3}, -0.5, 0, 0.68620627836857895},
    {3, {3}, 0.20000000000000001, 0, 1.1875871080472584},
    {3, {3}, 0, 0.40000000000000002, 1.0826288676216336},
    {3, {3}, 0.20000000000000001, -0.29999999999999999, 1.2494009036602962},
    {3, {1}, 1.2, 0.20000000000000001, 51.455844122715597},
    {3, {1.5}, 2, -1, 1330.5550540946199},
    {3, {1.5}, 0, 2, 12.650246035423345},
    {4, {0.20000000000000001,1.3,0.80000000000000004}, 0.051804304333935368, -0.034536202889290248, 1.0405819174637774},
    {4, {0.20000000000000001,1.3,0.80000000000000004}, -0.29999999999999999, 0.10000000000000001, 0.81382400533010224},
    {4, {0.20000000000000001,1.3,0.80000000000000004}, 0.034536202889290248, 0, 1.0264752886985227},
    {4, {0.20000000000000001,1.3,0.80000000000000004}, 0, 0.069072405778580495, 1.0024205944707707},
    {4, {0.20000000000000001,1.3,0.80000000000000004}, 0.61503333372801916, 0.76503333372801907, 258.36829526889784},
    {4, {0.90000000000000002,0.029999999999999999,-0.5}, 0.0034185213307280468, -0.002279014220485365, 1.0010511033327878},
    {4, {0.90000000000000002,0.029999999999999999,-0.5}, -0.29999999999999999, 0.10000000000000001, 0.92119800144778163},
    {4, {0.90000000000000002,0.029999999999999999,-0.5}, 0.002279014220485365, 0, 1.0007024427091906},
    {4, {0.90000000000000002,0.029999999999999999,-0.5}, 0, 0.00455802844097073, 1.0000106417969195},
    {4, {0.90000000000000002,0.029999999999999999,-0.5}, -0.02946529587470241, 0.12053470412529758, 1.0062136635097616},
    {4, {-0.90000000000000002,0.050000000000000003,-2}, 0.00030779788672993536, -0.00020519859115329027, 1.0002158149850995},
    {4, {-0.90000000000000002,0.050000000000000003,-2}, -0.29999999999999999, 0.10000000000000001, 0.82907219014965272},
    {4, {-0.90000000000000002,0.050000000000000003,-2}, 0.00020519859115329027, 0, 1.0001438479709615},
    {4, {-0.90000000000000002,0.050000000000000003,-2}, 0, 0.00041039718230658054, 1.0000000841935024},
    {4, {-0.90000000000000002,0.050000000000000003,-2}, -0.070900132148757261, -0.079099867851242733, 0.95492147215923162},
    {4, {0.90000000000000002,40,2}, 0.13187381222226424, -0.087915874814842837, 1.1171378730307941},
    {4, {0.90000000000000002,40,2}, -0.29999999999999999, 0.10000000000000001, 0.8017126196137242},
    {4, {0.90000000000000002,40,2}, 0.087915874814842837, 0, 1.073836121524167},
    {4, {0.90000000000000002,40,2}, 0, 0.17583174962968567, 1.0159791932239461},
    {4, {0.90000000000000002,40,2}, 1.6815591788005597, 1.8315591788005596, 1662556446863.551, 1e-11}, // Prepared GH rate rounding is amplified at this tail boundary.
    {4, {-0.90000000000000002,60,0.25}, 0.14999999999999999, -0.10000000000000001, 1.139860410503079},
    {4, {-0.90000000000000002,60,0.25}, -0.29999999999999999, 0.10000000000000001, 0.80485599221718074},
    {4, {-0.90000000000000002,60,0.25}, 0.10000000000000001, 0, 1.0847797948753566},
    {4, {-0.90000000000000002,60,0.25}, 0, 0.20000000000000001, 1.0197422421934708},
    {4, {-0.90000000000000002,60,0.25}, 1.9842142627348787, -2.1342142627348788, 67265281626370.523},
    {4, {0,1,-1}, 0.031363183582486368, -0.020908789054990912, 1.0236560876414995},
    {4, {0,1,-1}, -0.29999999999999999, 0.10000000000000001, 0.82090700476523382},
    {4, {0,1,-1}, 0.020908789054990912, 0, 1.0155015617263197},
    {4, {0,1,-1}, 0, 0.041817578109981823, 1.0008751370494324},
    {4, {0,1,-1}, 0.34275760531871841, 0.49275760531871843, 1.7997691951764616},
    {5, {0.20000000000000001,1.3}, 0.035630482034348064, -0.023753654689565376, 1.0271897136929053},
    {5, {0.20000000000000001,1.3}, -0.29999999999999999, 0.10000000000000001, 0.81676298382710821},
    {5, {0.20000000000000001,1.3}, 0.023753654689565376, 0, 1.0178481671640542},
    {5, {0.20000000000000001,1.3}, 0, 0.047507309379130752, 1.0011391061549892},
    {5, {0.20000000000000001,1.3}, 0.39959802069751615, 0.54959802069751618, 2.9729274976828415},
    {5, {0.90000000000000002,0.029999999999999999}, 0.0034185213307280468, -0.002279014220485365, 1.0010511033327878},
    {5, {0.90000000000000002,0.029999999999999999}, -0.29999999999999999, 0.10000000000000001, 0.92119800144778163},
    {5, {0.90000000000000002,0.029999999999999999}, 0.002279014220485365, 0, 1.0007024427091906},
    {5, {0.90000000000000002,0.029999999999999999}, 0, 0.00455802844097073, 1.0000106417969195},
    {5, {0.90000000000000002,0.029999999999999999}, -0.02946529587470241, 0.12053470412529758, 1.0062136635097616},
    {5, {-0.90000000000000002,0.050000000000000003}, 0.0044132920608548482, -0.0029421947072365657, 1.0016177750647779},
    {5, {-0.90000000000000002,0.050000000000000003}, -0.29999999999999999, 0.10000000000000001, 0.92399130786474659},
    {5, {-0.90000000000000002,0.050000000000000003}, 0.0029421947072365657, 0, 1.0010653304390484},
    {5, {-0.90000000000000002,0.050000000000000003}, 0, 0.0058843894144731314, 1.0000169153992251},
    {5, {-0.90000000000000002,0.050000000000000003}, -0.01621494974941342, -0.13378505025058657, 1.0194265871662349},
    {5, {0.90000000000000002,40}, 0.12482674974348865, -0.083217833162325783, 1.1101403056789061},
    {5, {0.90000000000000002,40}, -0.29999999999999999, 0.10000000000000001, 0.80175007137153509},
    {5, {0.90000000000000002,40}, 0.083217833162325783, 0, 1.0696570965981289},
    {5, {0.90000000000000002,40}, 0, 0.16643566632465157, 1.0142901285368782},
    {5, {0.90000000000000002,40}, 1.5876923065832689, 1.7376923065832688, 386963148.88178211},
    {6, {0.20000000000000001,1.3}, 0.054504112326729026, -0.036336074884486022, 1.0429283819233472},
    {6, {0.20000000000000001,1.3}, -0.29999999999999999, 0.10000000000000001, 0.81317300114826352},
    {6, {0.20000000000000001,1.3}, 0.036336074884486022, 0, 1.0279750402614773},
    {6, {0.20000000000000001,1.3}, 0, 0.072672149768972044, 1.002681156965791},
    {6, {0.20000000000000001,1.3}, 0.65099477619203072, 0.80099477619203063, 752.19380049638835},
    {6, {0.90000000000000002,0.029999999999999999}, 0.037558858563206128, -0.02503923904213742, 1.0282207078848435},
    {6, {0.90000000000000002,0.029999999999999999}, -0.29999999999999999, 0.10000000000000001, 0.81069048546902578},
    {6, {0.90000000000000002,0.029999999999999999}, 0.02503923904213742, 0, 1.0187424385109494},
    {6, {0.90000000000000002,0.029999999999999999}, 0, 0.05007847808427484, 1.0012980625549288},
    {6, {0.90000000000000002,0.029999999999999999}, 0.42528399606190553, 0.5752839960619055, 369.06480930261165},
    {6, {-0.90000000000000002,0.050000000000000003}, 0.037569852327631131, -0.025046568218420759, 1.0291998402396056},
    {6, {-0.90000000000000002,0.050000000000000003}, -0.29999999999999999, 0.10000000000000001, 0.82629929703160943},
    {6, {-0.90000000000000002,0.050000000000000003}, 0.025046568218420759, 0, 1.0187488627035988},
    {6, {-0.90000000000000002,0.050000000000000003}, 0, 0.050093136436841518, 1.0012152264270959},
    {6, {-0.90000000000000002,0.050000000000000003}, 0.42543043300404665, -0.57543043300404662, 369.25259228890746},
    {6, {-0.90000000000000002,60}, 0.14999999999999999, -0.10000000000000001, 1.1398598114249452},
    {6, {-0.90000000000000002,60}, -0.29999999999999999, 0.10000000000000001, 0.80484436667268866},
    {6, {-0.90000000000000002,60}, 0.10000000000000001, 0, 1.0847827366677247},
    {6, {-0.90000000000000002,60}, 0, 0.20000000000000001, 1.019743297566692},
    {6, {-0.90000000000000002,60}, 2.0072031090844877, -2.1572031090844881, 742797486179584.5},
    {7, {0.20000000000000001,2.2999999999999998}, 0.068351115572461568, -0.045567410381641055, 1.0550353785347488},
    {7, {0.20000000000000001,2.2999999999999998}, -0.29999999999999999, 0.10000000000000001, 0.81081543877228945},
    {7, {0.20000000000000001,2.2999999999999998}, 0.045567410381641055, 0, 1.0356719025719026},
    {7, {0.20000000000000001,2.2999999999999998}, 0, 0.091134820763282109, 1.0042335876842965},
    {7, {0.20000000000000001,2.2999999999999998}, 0.83543685942518831, 0.98543685942518822, 1141619.7339123513},
    {7, {0.90000000000000002,1.5001}, 0.045993033804993311, -0.030662022536662212, 1.035726622552787},
    {7, {0.90000000000000002,1.5001}, -0.29999999999999999, 0.10000000000000001, 0.80671526806209282},
    {7, {0.90000000000000002,1.5001}, 0.030662022536662212, 0, 1.0236399322793615},
    {7, {0.90000000000000002,1.5001}, 0, 0.061324045073324424, 1.0019471493060641},
    {7, {0.90000000000000002,1.5001}, 0.53762721028251093, 0.68762721028251084, 7085.1915527241754},
    {7, {-0.90000000000000002,2}, 0.053106410712133061, -0.035404273808088714, 1.0435686641732318},
    {7, {-0.90000000000000002,2}, -0.29999999999999999, 0.10000000000000001, 0.81672633054098953},
    {7, {-0.90000000000000002,2}, 0.035404273808088714, 0, 1.0277460305191188},
    {7, {-0.90000000000000002,2}, 0, 0.070808547616177428, 1.0024295556410641},
    {7, {-0.90000000000000002,2}, 0.63237739068561249, -0.78237739068561241, 135969.22416883812},
    {7, {0.90000000000000002,20}, 0.14999999999999999, -0.10000000000000001, 1.1354141948702605},
    {7, {0.90000000000000002,20}, -0.29999999999999999, 0.10000000000000001, 0.80133709327724778},
    {7, {0.90000000000000002,20}, 0.10000000000000001, 0, 1.0847445915039098},
    {7, {0.90000000000000002,20}, 0, 0.20000000000000001, 1.0208283004825722},
    {7, {0.90000000000000002,20}, 2.1619237198733119, 2.3119237198733122, 2.1597207297712055e+51},
};


void check(bool ok, const char* message) {
    if (!ok) { ++failures; std::printf("FAIL %s\n", message); }
}

Status evaluate(DensityKind kind, const double* shape, double a, double b, double& value) {
    watching = true;
    const auto status = exponential_moment(kind,shape,a,b,value);
    watching = false;
    if (status != Status::ok) check(value == sentinel,"failure preserves output");
    return status;
}

void reference_tests() {
    double worst[8]{};
    double ordinary_worst = 0.0;
    for (const auto& r : references) {
        double value = sentinel;
        const auto status = evaluate(static_cast<DensityKind>(r.kind),r.shape,r.a,r.b,value);
        if (status != Status::ok) {
            std::printf("REF REJECT %s shape=%g,%g,%g a=%.17g b=%.17g status=%d\n",
                names[r.kind],r.shape[0],r.shape[1],r.shape[2],r.a,r.b,static_cast<int>(status));
            ++failures;
            continue;
        }
        const double error = std::abs(value-r.expected)/std::abs(r.expected);
        worst[r.kind] = std::max(worst[r.kind],error);
        if (r.relative_tolerance == 1e-12) ordinary_worst = std::max(ordinary_worst,error);
        if (error > r.relative_tolerance) {
            std::printf("REF ERROR %s shape=%g,%g,%g a=%g b=%g actual=%.17g expected=%.17g relative=%.3g\n",
                names[r.kind],r.shape[0],r.shape[1],r.shape[2],r.a,r.b,value,r.expected,error);
            ++failures;
        }
    }
    for (int k=0;k<8;++k) std::printf("worst %s relative=%.3g\n",names[k],worst[k]);
    std::printf("worst excluding conditioned GH boundary=%.3g\n",ordinary_worst);
    std::printf("reference points=%zu\n",std::size(references));
}

void require_status(DensityKind kind, const double* shape, double a, double b, Status expected) {
    double value = sentinel;
    const auto status = evaluate(kind,shape,a,b,value);
    if (status != expected) {
        std::printf("STATUS %s shape=%g,%g,%g a=%.17g b=%.17g actual=%d expected=%d\n",
            names[static_cast<int>(kind)],shape ? shape[0]:0,shape ? shape[1]:0,shape ? shape[2]:0,
            a,b,static_cast<int>(status),static_cast<int>(expected));
        ++failures;
    }
}

void polynomial_domains() {
    const double ged_heavy[3] = {.7};
    for (int k : {1,2,3}) {
        const auto kind = static_cast<DensityKind>(k);
        const double* s = k==3 ? ged_heavy : shapes[k];
        for (auto pair : {std::pair{.1,0.},std::pair{0.,.1},std::pair{0.,-.1},
                         std::pair{-.1,.2},std::pair{-.1,-.2}})
            require_status(kind,s,pair.first,pair.second,Status::infinite);
        for (auto pair : {std::pair{0.,0.},std::pair{-.2,.2},std::pair{-.2,-.2},std::pair{-.3,.1}})
            require_status(kind,s,pair.first,pair.second,Status::ok);
    }
    const double laplace[3] = {1.};
    GedDensity::Prepared p;
    check(GedDensity::prepare(laplace,0,p),"Laplace prepare");
    const double rate = std::exp(-p.kappa);
    for (double b : {rate,-rate,std::nextafter(rate,inf),std::nextafter(-rate,-inf)})
        require_status(DensityKind::ged,laplace,0,b,Status::infinite);
    require_status(DensityKind::ged,laplace,rate,0,Status::infinite);
    require_status(DensityKind::ged,laplace,std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max(),Status::infinite);
    require_status(DensityKind::ged,laplace,0,std::nextafter(rate,0.),Status::ok);
    require_status(DensityKind::ged,laplace,-1e100,0,Status::ok);
}

template<class Member>
void gh_domains(DensityKind kind, const double* s, double lambda) {
    typename GhFamilyDensity<Member>::Prepared p;
    check(GhFamilyDensity<Member>::prepare(s,0,p),"GH prepare");
    require_status(kind,s,0,std::nextafter(p.minus,inf),Status::infinite);
    require_status(kind,s,0,-std::nextafter(p.plus,inf),Status::infinite);
    require_status(kind,s,p.minus,0,p.minus>p.plus ? Status::infinite
        : lambda < 0 ? Status::ok : Status::infinite);
    const auto edge = lambda < 0 ? Status::ok : Status::infinite;
    require_status(kind,s,0,p.minus,edge);
    require_status(kind,s,0,-p.plus,edge);
    // Each equality remains the growing tail even with a negative coefficient on |z|.
    require_status(kind,s,-p.minus,2*p.minus,edge);
    require_status(kind,s,-p.plus,-2*p.plus,edge);
    require_status(kind,s,-p.minus,3*p.minus,Status::infinite);
    require_status(kind,s,-p.plus,-3*p.plus,Status::infinite);
    require_status(kind,s,0,.999*p.minus,Status::ok);
    require_status(kind,s,0,-.999*p.plus,Status::ok);
}

void gh_existence() {
    gh_domains<GeneralizedHyperbolic>(DensityKind::generalized_hyperbolic,shapes[4],.8);
    const double negative[3] = {.2,1.3,-.5};
    const double symmetric_negative[3] = {0,1.3,-.5};
    const double zero[3] = {.2,1.3,0};
    gh_domains<GeneralizedHyperbolic>(DensityKind::generalized_hyperbolic,negative,-.5);
    gh_domains<GeneralizedHyperbolic>(DensityKind::generalized_hyperbolic,symmetric_negative,-.5);
    gh_domains<GeneralizedHyperbolic>(DensityKind::generalized_hyperbolic,zero,0);
    gh_domains<NormalInverseGaussian>(DensityKind::nig,shapes[5],-.5);
    gh_domains<Hyperbolic>(DensityKind::hyperbolic,shapes[6],1);
    gh_domains<VarianceGamma>(DensityKind::variance_gamma,shapes[7],2.3);
    GhFamilyDensity<NormalInverseGaussian>::Prepared nig;
    check(GhFamilyDensity<NormalInverseGaussian>::prepare(shapes[5],0,nig),"NIG prepare");
    for (double b : {nig.minus,-nig.plus}) {
        double value = sentinel;
        check(evaluate(DensityKind::nig,shapes[5],0,b,value)==Status::ok,"NIG equality value");
        // The NIG MGF at |beta+b|=alpha is exp(mu*b + delta*gamma).
        const double expected = std::exp(nig.mu*b+std::sqrt(nig.d2*nig.minus*nig.plus));
        check(std::abs(value/expected-1)<1e-12,"NIG equality closed MGF");
    }
    // An integrable but unresolved algebraic equality tail must fail explicitly.
    const double slow[3] = {0,1.,-1e-6};
    GhFamilyDensity<GeneralizedHyperbolic>::Prepared p;
    check(GhFamilyDensity<GeneralizedHyperbolic>::prepare(slow,0,p),"slow-tail prepare");
    require_status(DensityKind::generalized_hyperbolic,slow,0,p.minus,Status::numerical_failure);
    const double tiny_index[3] = {0,1.,-1e-20};
    check(GhFamilyDensity<GeneralizedHyperbolic>::prepare(tiny_index,0,p),"tiny-index prepare");
    require_status(DensityKind::generalized_hyperbolic,tiny_index,0,p.minus,Status::numerical_failure);
}

void invalid_inputs() {
    for (int k=0;k<8;++k) {
        const auto kind = static_cast<DensityKind>(k);
        for (double x : {nan,inf,-inf}) {
            require_status(kind,shapes[k],x,0,Status::numerical_failure);
            require_status(kind,shapes[k],0,x,Status::numerical_failure);
        }
        if (k) require_status(kind,nullptr,0,0,Status::numerical_failure);
        const auto domain = shape_domains(kind);
        for (std::size_t i=0;i<domain.size();++i) {
            for (double x : {domain[i].lower,domain[i].upper,nan,inf,-inf}) {
                double s[3];
                std::copy_n(shapes[k],3,s);
                s[i] = x;
                require_status(kind,s,0,0,Status::numerical_failure);
            }
        }
        double value = sentinel;
        check(evaluate(kind,shapes[k],0,0,value)==Status::ok && value==1,"M(0,0)=1 exactly");
    }
    double value = sentinel;
    check(evaluate(static_cast<DensityKind>(-1),nullptr,0,0,value)==Status::numerical_failure,"negative density kind");
    check(evaluate(static_cast<DensityKind>(99),nullptr,0,0,value)==Status::numerical_failure,"unknown density kind");
    require_status(DensityKind::normal,nullptr,100,0,Status::numerical_failure);
    require_status(DensityKind::normal,nullptr,0,1e300,Status::numerical_failure);
    require_status(DensityKind::ged,shapes[3],100,0,Status::numerical_failure);
}

void symmetry_and_limits() {
    for (int k=0;k<8;++k) {
        const auto kind = static_cast<DensityKind>(k);
        double mirrored[3];
        std::copy_n(shapes[k],3,mirrored);
        if (k==2) mirrored[1] *= -1;
        else if (k>=4) mirrored[0] *= -1;
        double first=sentinel, second=sentinel;
        check(evaluate(kind,shapes[k],-.3,.1,first)==Status::ok,"reflection first");
        check(evaluate(kind,mirrored,-.3,-.1,second)==Status::ok,"reflection second");
        check(std::abs(first-second)<1e-12*first,"reflection identity");
        check(first>0 && first<=1,"nonpositive exponent bound");
    }
    double value=sentinel;
    check(evaluate(DensityKind::normal,nullptr,-1e300,0,value)==Status::ok,"large negative Normal exponent");
    check(std::abs(value/7.978845608028654e-301-1)<1e-12,"Normal asymptotic limit");
    for (double b : {std::numeric_limits<double>::max(),-std::numeric_limits<double>::max()}) {
        value=sentinel;
        check(evaluate(DensityKind::normal,nullptr,-std::numeric_limits<double>::max(),b,value)==Status::ok,
              "Normal negative sum overflow remains representable");
        check(value==.5,"Normal half-mass limit");
    }
}

void timing() {
    for (int k=0;k<8;++k) {
        const int repeats = k==0 ? 10000 : 10;
        double value=sentinel;
        bool ok = true;
        const auto start=std::chrono::steady_clock::now();
        for (int i=0;i<repeats;++i)
            ok &= evaluate(static_cast<DensityKind>(k),shapes[k],-.3,.1,value)==Status::ok;
        const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
        check(ok,"timing convergence");
        std::printf("%s time=%.3f us value=%.17g\n",names[k],us,value);
    }
}
}

int main() {
    reference_tests();
    polynomial_domains();
    gh_existence();
    invalid_inputs();
    symmetry_and_limits();
    timing();
    check(allocations==0,"no heap allocations");
    return failures ? 1 : 0;
}
