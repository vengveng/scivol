#include "models/joint_power/aparch_jet.hpp"
#include "models/power/aparch_equation_program.hpp"
namespace subset_test {
namespace ap=vlk::experiment::runtime::joint_aparch;
namespace gen=vlk::experiment::generated;
namespace eq=vlk::experiment::model::aparch_equation;
using CommonOrders=vlk::experiment::runtime::joint_power::Orders<false,true,
    vlk::experiment::mean::dynamic,1,1,1,1>;
}
extern "C" int seed_primitive(const double* theta, double h0, double* output) {
    using namespace subset_test;
    if (!theta || !output || !(h0 > 0.) || !std::isfinite(h0)) return -1;
    eq::PowerUnits<2> units;
    units.prepare(theta[0], theta[3], h0);
    double seed_g[4], seed_h[10], first_g[4], first_h[10], log_g[4], log_h[10];
    eq::State seed{0., seed_g, seed_h}, first{0., first_g, first_h}, logs{0., log_g, log_h};
    if (!eq::initialize_power<2, 4>(seed, h0, theta, 3, 4, units.log_scale)) return -3;
    const auto record = [&](const eq::State& state, std::size_t offset) {
        if (!eq::to_log_variance<2, 4, true>(state, theta, 3, 4, logs, units.log_scale)) return false;
        output[offset] = logs.value;
        std::copy_n(log_g, 4, output + offset + 1);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j)
                output[offset + 5 + 4*i + j] = log_h[eq::packed_index(4, i, j)];
        return true;
    };
    if (!record(seed, 0)) return -3;
    units.start<4>(first, theta, 0, 3, 4);
    eq::add_persistence<2, 4>(first, seed, theta, 1, 4);
    eq::add_persistence<2, 4>(first, seed, theta, 2, 4);
    if (!record(first, 21)) return -3;
    const double trace[]{first.value, first.value_low, units.log_scale, seed.value,
        seed_g[3], seed_h[9], units.omega_first, units.delta_first,
        units.omega_delta, units.delta_second};
    std::copy_n(trace, 10, output + 42);
    return 0;
}

extern "C" int subset_log_metadata_contract() {
    using namespace subset_test;
    const double theta[] = {.1,.2,.3,1.6};
    for (const double value : {1.,1e-310}) {
        double gradient[4], hessian[10], left_g[4], left_h[10], right_g[4], right_h[10];
        for (std::size_t i=0;i<4;++i) gradient[i]=value*(i+1);
        for (std::size_t i=0;i<10;++i) hessian[i]=value*(i+1);
        eq::State power{value,gradient,hessian}, fixed{0.,left_g,left_h}, general{0.,right_g,right_h};
        if (!eq::to_log_variance<2,4,true,3>(power,theta,eq::dynamic_index,
                eq::dynamic_index,fixed,.2)
            || !eq::to_log_variance<2,0,true>(power,theta,3,4,general,.2)) return 0;
        if (fixed.value!=general.value) return 0;
        for (std::size_t i=0;i<4;++i)
            if (!std::isfinite(left_g[i]) || left_g[i]!=right_g[i]) return 0;
        for (std::size_t i=0;i<10;++i)
            if (!std::isfinite(left_h[i]) || left_h[i]!=right_h[i]) return 0;
    }
    namespace transform=vlk::experiment::runtime::aparch_transform;
    constexpr auto* single=&ap::fixed_layout<CommonOrders,gen::StudentTDensity,true>;
    constexpr auto* multiple=&ap::fixed_layout<vlk::experiment::runtime::joint_power::Orders<false,true,vlk::experiment::mean::dynamic,2,1,3,2>,gen::StudentTDensity,true>;
    static_assert(transform::moment_lag<single>(eq::dynamic_index)==0u);
    static_assert(transform::moment_lag<multiple>(2u)==2u);
    static_assert(transform::moment_lag<nullptr>(2u)==2u);
    return 2;
}
