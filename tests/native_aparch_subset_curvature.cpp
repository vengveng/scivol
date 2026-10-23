// SCIVOL-NATIVE: TEST_ONLY; current mean-refinement owners, no solver or extension.
#include "aparch_mean_refinement.hpp"
#include <bit>
#include <tuple>

namespace subset_test {
namespace ap = vlk::experiment::runtime::aparch;
namespace rt = vlk::experiment::runtime;
namespace gen = vlk::experiment::generated;
namespace mf = ap::mean_refinement;
namespace eq = vlk::experiment::model::aparch_equation;
using Density = gen::NormalDensity;

struct Spec { std::size_t ar, ma, p, q, fixed; };
using CommonOrders = ap::FixedJointOrders<1, 1, 1, 1>;
using NoMaOrders = ap::FixedJointOrders<1, 0, 1, 2>;

static_assert(mf::mean_only_curvature<Density>);
static_assert(!mf::mean_only_curvature<gen::StudentTDensity>);
static_assert(ap::fixed_layout<CommonOrders, Density, true>.mean_parameters == 3u);
static_assert(ap::fixed_layout<NoMaOrders, Density, true>.mean_parameters == 2u);
static_assert(ap::fixed_layout<CommonOrders, Density, true>.parameters == 8u);
static_assert(ap::fixed_layout<NoMaOrders, Density, true>.parameters == 8u);

auto layout_fields(const ap::Layout& layout) {
    // Adding a layout field requires updating this all-field comparison.
    const auto& [ar, ma, p, q, lag, mean, model, width, packed, ar_start,
        ma_start, omega, alpha, gamma, beta, delta, joint, valid] = layout;
    return std::tuple{ar, ma, p, q, lag, mean, model, width, packed, ar_start,
        ma_start, omega, alpha, gamma, beta, delta, joint, valid};
}

auto extent_fields(const mf::Plan& plan) {
    const auto& [bytes, core, chart, primal, direction, width, derivative_width,
        active, square, reduced_square, centers, trials, seen, neighbors, roots, cache] = plan;
    return std::tuple{bytes, core, chart, primal, direction, width, derivative_width,
        active, square, reduced_square, centers, trials, seen, neighbors, roots, cache};
}

template<class D, class Orders, mf::ActiveSet Active>
bool check_refinement_extents(const ap::Layout& runtime) {
    const auto expected = mf::detail::order_extents<D, Active>(runtime);
    constexpr auto fixed = mf::detail::fixed_extents<D, Orders, Active>;
    return extent_fields(expected) == extent_fields(fixed);
}

template<class D, class Orders, bool Joint>
bool check_fixed_layout() {
    constexpr auto fixed = ap::fixed_layout<Orders, D, Joint>;
    static_assert(fixed.valid);
    // Keep one builder genuinely runtime under production optimization.
    volatile std::size_t inputs[] = {Orders::p_ar, Orders::q_ma, Orders::p,
        Orders::q, Orders::columns};
    ap::Layout runtime;
    bool valid;
    if constexpr (Orders::linear)
        valid = ap::make_linear_layout<D>(inputs[4], inputs[2], inputs[3], runtime);
    else
        valid = ap::make_layout<D>(inputs[0], inputs[1], inputs[2], inputs[3], Joint, runtime);
    if (!valid || layout_fields(fixed) != layout_fields(runtime)) return false;
    if constexpr (Joint)
        return check_refinement_extents<D, Orders, mf::ActiveSet::mean>(runtime)
            && check_refinement_extents<D, Orders, mf::ActiveSet::joint>(runtime);
    return true;
}

template<class D>
bool check_density_layouts() {
    namespace mean = vlk::experiment::mean;
    return check_fixed_layout<D, ap::FixedVolatilityOrders<1,1>, false>()
        && check_fixed_layout<D, ap::FixedVolatilityOrders<1,2>, false>()
        && check_fixed_layout<D, ap::FixedVolatilityOrders<2,1>, false>()
        && check_fixed_layout<D, ap::FixedVolatilityOrders<2,2>, false>()
        && check_fixed_layout<D, ap::FixedVolatilityOrders<3,3>, false>()
        && check_fixed_layout<D, ap::FixedJointOrders<1,1,1,1>, true>()
        && check_fixed_layout<D, ap::FixedJointOrders<0,0,3,2>, true>()
        && check_fixed_layout<D, ap::FixedJointOrders<1,0,1,2>, true>()
        && check_fixed_layout<D, ap::FixedJointOrders<0,5,2,1>, true>()
        && check_fixed_layout<D, ap::FixedJointOrders<5,4,3,2>, true>()
        && check_fixed_layout<D, mean::LinearVolatilityOrders<0,1,1>, true>()
        && check_fixed_layout<D, mean::LinearVolatilityOrders<4,1,1>, true>()
        && check_fixed_layout<D, mean::LinearVolatilityOrders<7,3,2>, true>()
        && check_fixed_layout<D, mean::LinearVolatilityOrders<11,4,5>, true>();
}

bool make_layout(const Spec* s, ap::Layout& l) {
    if (!s || !ap::make_layout<Density>(s->ar, s->ma, s->p, s->q, true, l)) return false;
    if (s->fixed == 0u) return true;
    if (s->fixed == 1u) return ap::matches_orders<CommonOrders>(l);
    if (s->fixed == 2u) return ap::matches_orders<NoMaOrders>(l);
    return false;
}

template<int Order, class Orders, bool Reduced>
using Chart = std::conditional_t<Reduced, mf::Chart<Order, Density, Orders>,
    vlk::experiment::mean::AnchoredArmaChart<Order, ap::Layout,
        mf::chart_layout<Density, Orders>, false>>;

template<class Orders, class Operation>
auto request(unsigned mask, Operation operation) {
#define REQUEST(M, O, F, G, H) \
    case M: return operation.template operator()<O, Orders, F, G, H>()
    switch (mask) {
        REQUEST(1, 0, true, false, false);
        REQUEST(2, 1, false, true, false);
        REQUEST(3, 1, true, true, false);
        REQUEST(4, 2, false, false, true);
        REQUEST(5, 2, true, false, true);
        REQUEST(6, 2, false, true, true);
        REQUEST(7, 2, true, true, true);
    }
#undef REQUEST
    return decltype(operation.template operator()<0, Orders, true, false, false>()){};
}

template<class Operation>
auto dispatch(const Spec& s, unsigned mask, bool reduced, Operation operation) {
    const auto by_orders = [&]<class Orders>() {
        return request<Orders>(mask, [&]<int O, class T, bool F, bool G, bool H>() {
            if (reduced) return operation.template operator()<O, T, F, G, H, true>();
            return operation.template operator()<O, T, F, G, H, false>();
        });
    };
    if (s.fixed == 1u) return by_orders.template operator()<CommonOrders>();
    if (s.fixed == 2u) return by_orders.template operator()<NoMaOrders>();
    return by_orders.template operator()<ap::RuntimeOrders>();
}

template<int Order, class Orders, bool Gradient, bool Reduced>
struct Workspace {
    std::size_t core, chart, bytes;
    explicit Workspace(const ap::Layout& l)
        : core(ap::core_workspace_bytes<Order, true, mf::chart_layout<Density, Orders>,
               Reduced && Order != 0>(l, Gradient)),
          chart(Order ? Chart<Order ? Order : 1, Orders, Reduced>::workspace_bytes(l) : 0u),
          bytes(0u) {
        rt::WorkspaceSizer size;
        if (core && size.add<double>(mf::detail::double_count(core))
            && size.add<double>(mf::detail::double_count(chart))) bytes = size.size();
    }
};

template<int Order, class Orders, bool F, bool G, bool H, bool Reduced>
int evaluate(const ap::Layout& l, const double* theta, const double* data,
    std::size_t n, double h0, std::size_t pivot, void* memory, std::size_t bytes,
    double* f, double* g, double* h, double* anchor) {
    const Workspace<Order, Orders, G, Reduced> w(l);
    if (!ap::matches_orders<Orders>(l) || !w.bytes || !memory || bytes < w.bytes) return -1;
    rt::WorkspaceArena arena(memory, bytes);
    void* core = arena.take<double>(mf::detail::double_count(w.core));
    if constexpr (Order == 0) {
        return ap::evaluate_theta_order<0, true, Orders, Density, F, G, H>(
            l, theta, data, n, h0, core, w.core, f, g, h) ? 0 : -3;
    } else {
        using MeanPlan = Chart<Order, Orders, Reduced>;
        MeanPlan plan(l, theta, data, n, pivot);
        void* chart = arena.take<double>(mf::detail::double_count(w.chart));
        if (!plan.prepare(chart, w.chart)) return -2;
        const bool ok = ap::evaluate_theta_order<Order, true, Orders, Density,
            F, G, H, void, false, MeanPlan, Reduced>(
                l, theta, data, n, h0, core, w.core, f, g, h,
                nullptr, nullptr, nullptr, nullptr, &plan);
        if (ok) {
            const auto error = plan.anchor_error(), basis = plan.intercept_basis();
            const auto residual = plan.pivot_value();
            const double evidence[] = {error.high, error.low, basis.high, basis.low,
                                       residual.high, residual.low};
            std::copy_n(evidence, 6u, anchor);
        }
        return ok ? 0 : -3;
    }
}

template<class Operation>
auto seed_dispatch(const Spec& s, unsigned mask, Operation operation) {
    if (s.fixed == 2u) return request<NoMaOrders>(mask, operation);
    return request<ap::RuntimeOrders>(mask, operation);
}

template<int Order, class Orders, bool Gradient>
std::size_t seed_bytes(const ap::Layout& l) {
    constexpr const ap::Layout* fixed = Orders::fixed
        ? &ap::fixed_layout<Orders, Density, true> : nullptr;
    return ap::core_workspace_bytes<Order, true, fixed>(l, Gradient);
}
} // namespace subset_test

extern "C" std::size_t subset_workspace(const subset_test::Spec* s, unsigned mask, bool reduced) {
    using namespace subset_test;
    ap::Layout l;
    if (!ap::valid_request(mask) || !make_layout(s, l)) return 0u;
    return dispatch(*s, mask, reduced, [&]<int O, class T, bool F, bool G, bool H, bool R>() {
        return Workspace<O, T, G, R>(l).bytes;
    });
}

extern "C" std::size_t subset_width(const subset_test::Spec* s, bool reduced) {
    using namespace subset_test;
    ap::Layout l;
    if (!make_layout(s, l)) return 0u;
    return dispatch(*s, 7u, reduced, [&]<int O, class T, bool F, bool G, bool H, bool R>() {
        const auto& selected = ap::evaluation_layout<T, Density, true>(l);
        return R ? selected.mean_parameters : selected.parameters;
    });
}

extern "C" int subset_evaluate(const subset_test::Spec* s, unsigned mask, bool reduced,
    const double* theta, const double* data, std::size_t n, double h0, std::size_t pivot,
    void* memory, std::size_t bytes, double* f, double* g, double* h, double* anchor) {
    using namespace subset_test;
    ap::Layout l;
    if (!subset_workspace(s, mask, reduced) || !theta || !data
        || !make_layout(s, l) || ((mask & 6u) && !anchor)
        || !ap::valid_data<true>(l, data, n, h0)
        || pivot < l.maximum_lag || pivot >= n
        || !std::all_of(theta, theta + l.parameters, [](double x) { return std::isfinite(x); })
        || (mask & 1u && !f) || (mask & 2u && !g) || (mask & 4u && !h)) return -1;
    return dispatch(*s, mask, reduced, [&]<int O, class T, bool F, bool G, bool H, bool R>() {
        return evaluate<O, T, F, G, H, R>(l, theta, data, n, h0, pivot,
                                        memory, bytes, f, g, h, anchor);
    });
}

// Internal theta equations, not a checked ABI or persistence-domain qualification.
extern "C" std::size_t seed_workspace(const subset_test::Spec* s, unsigned mask) {
    using namespace subset_test;
    ap::Layout l;
    if (!ap::valid_request(mask) || !make_layout(s, l)
        || (s->fixed != 0u && s->fixed != 2u)) return 0u;
    return seed_dispatch(*s, mask, [&]<int O, class T, bool F, bool G, bool H>() {
        return seed_bytes<O, T, G>(l);
    });
}

extern "C" int seed_evaluate(const subset_test::Spec* s, unsigned mask,
    const double* theta, const double* data, std::size_t n, double h0,
    void* memory, std::size_t bytes, double* f, double* g, double* h) {
    using namespace subset_test;
    ap::Layout l;
    const auto required = seed_workspace(s, mask);
    if (!required || !memory || bytes < required || !theta || !make_layout(s, l)
        || !ap::valid_data<true>(l, data, n, h0)
        || !std::all_of(theta, theta + l.parameters, [](double x) { return std::isfinite(x); })
        || (mask & 1u && !f) || (mask & 2u && !g) || (mask & 4u && !h)) return -1;
    return seed_dispatch(*s, mask, [&]<int O, class T, bool F, bool G, bool H>() {
        return ap::evaluate_theta_order<O, true, T, Density, F, G, H>(
            l, theta, data, n, h0, memory, bytes, f, g, h) ? 0 : -3;
    });
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

extern "C" int subset_layout_contract(unsigned density) {
    using namespace subset_test;
#define CHECK_LAYOUTS(CODE, TYPE) case CODE: return check_density_layouts<gen::TYPE>() ? 14 : 0
    switch (density) {
        CHECK_LAYOUTS(0, NormalDensity);
        CHECK_LAYOUTS(1, StudentTDensity);
        CHECK_LAYOUTS(2, SkewTDensity);
        CHECK_LAYOUTS(3, GedDensity);
        CHECK_LAYOUTS(4, GeneralizedHyperbolicDensity);
        CHECK_LAYOUTS(5, NormalInverseGaussianDensity);
        CHECK_LAYOUTS(6, HyperbolicDensity);
        CHECK_LAYOUTS(7, VarianceGammaDensity);
        default: return -1;
    }
#undef CHECK_LAYOUTS
}

extern "C" int subset_slope_contract() {
    using namespace subset_test;
    const double gradient[] = {1e20, -.3, 5e9, -1e8, .03, -.0005, 8., -7., .12, 13., -6.};
    const double origin[] = {.03, .1, -.02, .08, .08, .1, .6, 1.6, .8, .7, .3};
    const double trial[] = {.03, .10000000000000002, -.020000000000000004,
        .09, .07, .1, .6000000000000001, 1.6, .7, .8, .4};
    const double evidence[] = {0., 0., 1e-35, -1e-52, 0.};
    const auto check = [&]<std::size_t Width>() {
        const auto fixed = mf::detail::actual_slope<Width>(gradient, origin, trial,
            evidence, std::numeric_limits<std::size_t>::max());
        const auto general = mf::detail::actual_slope(gradient, origin, trial, evidence, Width);
        return std::isfinite(fixed) && std::bit_cast<std::uint64_t>(fixed)
            == std::bit_cast<std::uint64_t>(general);
    };
    return check.template operator()<3>() && check.template operator()<8>()
        && check.template operator()<11>() ? 3 : 0;
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
    constexpr auto* multiple=&ap::fixed_layout<ap::FixedJointOrders<2,1,3,2>,gen::StudentTDensity,true>;
    static_assert(transform::moment_lag<single>(eq::dynamic_index)==0u);
    static_assert(transform::moment_lag<multiple>(2u)==2u);
    static_assert(transform::moment_lag<nullptr>(2u)==2u);
    return 2;
}
