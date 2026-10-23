// SCIVOL-NATIVE: TEST_ONLY; canonical jets and filters, no optimizer or extension.
#include "aparch_runtime_jet.hpp"
#include "anchored_mean_chart.hpp"

namespace aparch_range_test {
namespace ap = vlk::experiment::runtime::aparch;
namespace gen = vlk::experiment::generated;
namespace mean = vlk::experiment::mean;

struct Spec {
    std::size_t ar, ma, p, q, columns;
    unsigned mode, fixed, density;
};

template<class Density>
bool layout(const Spec& s, ap::Layout& result) {
    if (s.mode == 2u) return ap::make_linear_layout<Density>(s.columns, s.p, s.q, result);
    return ap::make_layout<Density>(s.ar, s.ma, s.p, s.q, s.mode == 1u, result);
}

template<class Operation>
auto density(unsigned code, Operation operation) {
    if (code == 0u) return operation.template operator()<gen::NormalDensity>();
    return operation.template operator()<gen::StudentTDensity>();
}

template<class Density, class Operation>
auto orders(const Spec& s, Operation operation) {
    if (s.mode == 0u) {
        if (!s.fixed) return operation.template operator()<Density, ap::RuntimeOrders, false>();
        if (s.p == 1u && s.q == 1u)
            return operation.template operator()<Density, ap::FixedVolatilityOrders<1,1>, false>();
        return operation.template operator()<Density, ap::FixedVolatilityOrders<3,2>, false>();
    }
    if (s.mode == 1u) {
        if (!s.fixed) return operation.template operator()<Density, ap::RuntimeOrders, true>();
        if (s.p == 1u && s.q == 1u)
            return operation.template operator()<Density, ap::FixedJointOrders<1,1,1,1>, true>();
        return operation.template operator()<Density, ap::FixedJointOrders<2,1,3,2>, true>();
    }
    if (!s.fixed) return operation.template operator()<Density, mean::LinearVolatilityOrders<>, true>();
    return operation.template operator()<Density, mean::LinearVolatilityOrders<2,3,2>, true>();
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

template<class Orders, bool Reduced, int Order, bool F, bool G, bool H>
std::size_t chart_bytes(const ap::Layout& l) {
    if constexpr (Order == 0) return 0u;
    else {
        constexpr const auto* fixed = Orders::fixed ? &ap::fixed_layout<Orders, gen::NormalDensity, true> : nullptr;
        using Chart = mean::AnchoredArmaChart<Order, ap::Layout, fixed, Reduced>;
        const auto core = ap::core_workspace_bytes<Order, true, fixed, Reduced, false>(l, G);
        vlk::experiment::runtime::WorkspaceSizer sizer;
        return sizer.add<double>((core + 7u) / 8u)
            && sizer.add<double>((Chart::workspace_bytes(l) + 7u) / 8u) ? sizer.size() : 0u;
    }
}

template<class Operation>
auto chart_orders(bool fixed, Operation operation) {
    if (fixed) return operation.template operator()<ap::FixedJointOrders<1,1,1,1>>();
    return operation.template operator()<ap::RuntimeOrders>();
}

template<class Orders, bool Reduced, int Order, bool F, bool G, bool H>
int chart_evaluate(const ap::Layout& l, const double* theta, const double* data,
    std::size_t n, double h0, void* memory, std::size_t bytes, double* f, double* g, double* h) {
    if constexpr (Order == 0) return -1;
    else {
        constexpr const auto* fixed = Orders::fixed ? &ap::fixed_layout<Orders, gen::NormalDensity, true> : nullptr;
        using Chart = mean::AnchoredArmaChart<Order, ap::Layout, fixed, Reduced>;
        const auto core_bytes = ap::core_workspace_bytes<Order, true, fixed, Reduced, false>(l, G);
        const auto plan_bytes = Chart::workspace_bytes(l);
        vlk::experiment::runtime::WorkspaceArena arena(memory, bytes);
        void* core = arena.take<double>((core_bytes + 7u) / 8u);
        void* plan_memory = arena.take<double>((plan_bytes + 7u) / 8u);
        Chart chart(l, theta, data, n, l.maximum_lag);
        if (!chart.prepare(plan_memory, plan_bytes)) return -2;
        return ap::evaluate_theta_order<Order, true, Orders, gen::NormalDensity, F, G, H,
            void, false, Chart, Reduced>(l, theta, data, n, h0, core, core_bytes,
                f, g, h, nullptr, nullptr, nullptr, nullptr, &chart) ? 0 : -3;
    }
}
} // namespace aparch_range_test

extern "C" std::size_t aparch_range_workspace(const aparch_range_test::Spec* s, unsigned mask) {
    using namespace aparch_range_test;
    if (!s || !ap::valid_request(mask)) return 0u;
    return density(s->density, [&]<class D>() {
        ap::Layout l;
        if (!layout<D>(*s, l)) return std::size_t{0u};
        return orders<D>(*s, [&]<class Density, class Orders, bool Joint>() {
            constexpr const ap::Layout* fixed = Orders::fixed
                ? &ap::fixed_layout<Orders, Density, Joint> : nullptr;
            return request(mask, [&]<int O, bool F, bool G, bool H>() {
                return ap::core_workspace_bytes<O, Joint, fixed>(l, G);
            });
        });
    });
}

extern "C" int aparch_range_evaluate(const aparch_range_test::Spec* s, unsigned mask,
    const double* theta, const double* data, const double* features, std::size_t n,
    double h0, void* memory, std::size_t bytes, double* f, double* g, double* h,
    unsigned* execution) {
    using namespace aparch_range_test;
    if (!s || !ap::valid_request(mask)) return -1;
    return density(s->density, [&]<class D>() {
        ap::Layout l;
        if (!layout<D>(*s, l) || !ap::valid_theta<D>(l, theta)) return -2;
        return orders<D>(*s, [&]<class Density, class Orders, bool Joint>() {
            if (!ap::valid_data<Joint>(l, data, n, h0)) return -1;
            return request(mask, [&]<int O, bool F, bool G, bool H>() {
                return ap::evaluate_theta_order<O, Joint, Orders, Density, F, G, H,
                    void, false, void, false, void>(
                    l, theta, data, n, h0, memory, bytes, f, g, h,
                    features, nullptr, nullptr, nullptr, nullptr, nullptr, execution) ? 0 : -3;
            });
        });
    });
}

extern "C" int aparch_range_filter(const aparch_range_test::Spec* s,
    const double* theta, const double* data, std::size_t n, double h0,
    void* memory, std::size_t bytes, double* residual, double* variance) {
    using namespace aparch_range_test;
    if (!s || s->mode == 2u) return -1;
    return density(s->density, [&]<class D>() {
        ap::Layout l;
        if (!layout<D>(*s, l)) return -1;
        return orders<D>(*s, [&]<class Density, class Orders, bool Joint>() {
            return ap::filter_state<Joint, Density, Orders>(l, theta, data, n,
                h0, memory, bytes, residual, variance);
        });
    });
}

extern "C" std::size_t aparch_range_chart_workspace(bool fixed, bool reduced, unsigned mask) {
    using namespace aparch_range_test;
    ap::Layout l;
    if (!ap::make_layout<gen::NormalDensity>(1, 1, 1, 1, true, l)) return 0u;
    return chart_orders(fixed, [&]<class T>() {
            return request(mask, [&]<int O, bool F, bool G, bool H>() {
                return reduced ? chart_bytes<T, true, O, F, G, H>(l)
                    : chart_bytes<T, false, O, F, G, H>(l);
            });
        });
}

extern "C" int aparch_range_chart_evaluate(bool fixed, bool reduced, unsigned mask,
    const double* theta, const double* data, std::size_t n, double h0,
    void* memory, std::size_t bytes, double* f, double* g, double* h) {
    using namespace aparch_range_test;
    ap::Layout l;
    if (!ap::make_layout<gen::NormalDensity>(1, 1, 1, 1, true, l)) return -1;
    return chart_orders(fixed, [&]<class T>() {
            return request(mask, [&]<int O, bool F, bool G, bool H>() {
                return reduced ? chart_evaluate<T, true, O, F, G, H>(l, theta, data, n, h0,
                    memory, bytes, f, g, h) : chart_evaluate<T, false, O, F, G, H>(l,
                    theta, data, n, h0, memory, bytes, f, g, h);
            });
        });
}
