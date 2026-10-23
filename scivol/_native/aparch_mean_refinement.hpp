// SCIVOL-NATIVE: integration candidate; no default routing or fit certificate.
#ifndef SCIVOL_NATIVE_APARCH_MEAN_REFINEMENT_HPP
#define SCIVOL_NATIVE_APARCH_MEAN_REFINEMENT_HPP

#include "anchored_mean_chart.hpp"
#include "arma_residual_precision.hpp"
#include "aparch_runtime_jet.hpp"
#include "bounded_direction.hpp"
#include "runtime_reflection.hpp"

#include <cstring>
#include <span>

namespace vlk::experiment::runtime::aparch::mean_refinement {

namespace bounded = vlk::experiment::bounded_direction;
using Real = wide::Real;

template<class Density, class Orders>
inline constexpr const Layout* chart_layout = Orders::fixed
    ? &fixed_layout<Orders, Density, true> : nullptr;

enum class ActiveSet { mean, joint };

// Other densities retain full preparation until frozen-shape intentions are qualified.
template<class Density, ActiveSet Active = ActiveSet::mean>
inline constexpr bool mean_only_curvature =
    Active == ActiveSet::mean && std::is_same_v<Density, generated::NormalDensity>;

template<int Order, class Density, class Orders, class PrimalPlan = void,
         ActiveSet Active = ActiveSet::mean>
using Chart = mean::AnchoredArmaChart<Order, Layout,
    chart_layout<Density, Orders>, mean_only_curvature<Density, Active>, PrimalPlan>;

struct Plan {
    std::size_t bytes = 0, core = 0, chart = 0, primal = 0, direction = 0;
    std::size_t k = 0, width = 0, d = 0, square = 0, reduced_square = 0;
    std::size_t centers = 0, trials = 0, seen = 0, neighbors = 0, root = 0, mean_cache = 0;
};

namespace detail {

template<class Density, ActiveSet Active>
constexpr Plan order_extents(const Layout& l) noexcept {
    if (!l.valid) return {};
    Plan p;
    const auto active_width = Active == ActiveSet::mean ? l.mean_parameters : l.parameters;
    if (!active_width) return {};
    p.k = l.parameters; p.d = active_width - 1u; p.root = std::max(l.p_ar, l.q_ma);
    p.width = mean_only_curvature<Density, Active> ? l.mean_parameters : l.parameters;
    if (!checked_product(p.width, p.width, p.square)
        || !checked_product(p.d, p.d, p.reduced_square)
        || !checked_product(5u, active_width, p.centers)
        || !checked_sum(p.centers, 5u, p.centers)
        || !checked_product(p.centers, 8u, p.trials)
        || !checked_product(p.trials, p.k, p.seen)
        || !checked_product(8u, p.k, p.neighbors)) return {};
    if constexpr (Active == ActiveSet::joint)
        if (!checked_product(8u, l.mean_parameters, p.mean_cache)) return {};
    return p;
}

template<class Density, class Orders, ActiveSet Active>
inline constexpr Plan fixed_extents = order_extents<Density, Active>(
    fixed_layout<Orders, Density, true>);

// Only order-derived extents live here; arena sizes remain in the checked plan.
template<class Density, class Orders, ActiveSet Active>
const Plan& evaluation_extents(const Plan& runtime) noexcept {
    if constexpr (Orders::fixed) return fixed_extents<Density, Orders, Active>;
    else return runtime;
}

template<class Density, class Orders>
using Primal = mean::ArmaResidualPrecision<Layout, chart_layout<Density, Orders>>;

template<class Provider>
struct ValuePlan {
    Provider& primal;
    bool advance(std::size_t time, State& residual) noexcept {
        residual.value = static_cast<double>(primal.advance(time));
        return std::isfinite(residual.value);
    }
};

template<class Density, class Orders>
bool valid_layout(const Layout& supplied) noexcept {
    if constexpr (Orders::linear) return false;
    Layout canonical;
    if (!supplied.valid || !supplied.joint || !supplied.p_ar
        || !matches_orders<Orders>(supplied)
        || !make_layout<Density>(supplied.p_ar, supplied.q_ma,
            supplied.p, supplied.q, true, canonical)) return false;
    return supplied.maximum_lag == canonical.maximum_lag
        && supplied.mean_parameters == canonical.mean_parameters
        && supplied.model_parameters == canonical.model_parameters
        && supplied.parameters == canonical.parameters && supplied.packed == canonical.packed
        && supplied.ar_start == canonical.ar_start && supplied.ma_start == canonical.ma_start
        && supplied.omega_index == canonical.omega_index && supplied.alpha_start == canonical.alpha_start
        && supplied.gamma_start == canonical.gamma_start && supplied.beta_start == canonical.beta_start
        && supplied.delta_index == canonical.delta_index;
}

inline std::size_t double_count(std::size_t bytes) noexcept {
    return bytes / sizeof(double) + (bytes % sizeof(double) != 0u);
}

// All declared buffers are disjoint, including unused output capacity.
template<std::size_t N>
bool disjoint_buffers(const void* const (&pointers)[N],
    const std::size_t (&bytes)[N]) noexcept {
    bounded::Range ranges[N];
    for (std::size_t i = 0; i < N; ++i) {
        if (reinterpret_cast<std::uintptr_t>(pointers[i]) % alignof(double)
            || !bounded::memory_range(pointers[i], bytes[i], ranges[i])) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (bounded::intersects(ranges[i], ranges[j])) return false;
    }
    return true;
}

struct Neighbor {
    double intercept = 0., coefficient = 0.;
    Real residual{}, rho{};
    std::ptrdiff_t offset = 0;
    double distance = INFINITY;
};

// The caller has checked buffers, layout, immutable data and the pivot range.
template<class Density, class Orders, class PrimalPlan = void>
int neighbors_prevalidated(const Layout& supplied, Real target, Real error,
    const double* center, const double* data, std::size_t n,
    std::size_t pivot, std::size_t column, std::size_t radius,
    void* memory, std::size_t bytes, double* points, double* evidence,
    PrimalPlan* primal = nullptr) noexcept {
    const auto& l = evaluation_layout<Orders, Density, true>(supplied);
    if (!wide::finite(target) || !wide::finite(error)
        || !std::all_of(center, center + l.parameters,
            [](double x) { return std::isfinite(x); })) return -1;
    Chart<1, Density, Orders, PrimalPlan> current(l, center, data, n, pivot, primal);
    if (!current.prepare(memory, bytes)) return -1;
    const Real residual = current.pivot_value(), b = current.intercept_basis();
    const Real a = current.prepared_coefficient_basis(column);
    Neighbor nearest[2][4];
    const auto retain = [&](Neighbor item, std::size_t rank) noexcept {
        const Real difference = rank ? item.residual : item.residual - target;
        item.distance = std::abs(static_cast<double>(difference));
        for (std::size_t j = 0; j < 4; ++j)
            if (item.distance < nearest[rank][j].distance) std::swap(item, nearest[rank][j]);
    };
    const auto consider = [&](double value, std::ptrdiff_t offset) noexcept {
        const Real displacement = Real(value) - Real(center[column]);
        const Real ideal = Real(center[0]) + (residual - target - a * displacement) / b;
        const double intercept = static_cast<double>(ideal);
        if (!std::isfinite(intercept)) return;
        const Real actual = residual - b * (Real(intercept) - Real(center[0])) - a * displacement;
        Neighbor item{intercept, value, actual, actual + b * error, offset};
        retain(item, 0u); retain(item, 1u);
    };
    consider(center[column], 0);
    double lower = center[column], upper = lower;
    for (std::size_t i = 1; i <= radius; ++i) {
        lower = std::nextafter(lower, -INFINITY); upper = std::nextafter(upper, INFINITY);
        consider(lower, -std::ptrdiff_t(i)); consider(upper, std::ptrdiff_t(i));
    }
    std::size_t count = 0;
    for (const auto& ranking : nearest) for (const auto& item : ranking) {
        if (!std::isfinite(item.distance)) continue;
        bool duplicate = false;
        for (std::size_t i = 0; i < count; ++i)
            duplicate |= points[i * l.parameters] == item.intercept
                && points[i * l.parameters + column] == item.coefficient;
        if (duplicate) continue;
        auto* point = points + count * l.parameters;
        std::copy_n(center, l.parameters, point);
        point[0] = item.intercept; point[column] = item.coefficient;
        auto* row = evidence + count * 5u;
        row[0] = item.residual.high; row[1] = item.residual.low;
        row[2] = item.rho.high; row[3] = item.rho.low; row[4] = double(item.offset);
        ++count;
    }
    return int(count);
}

struct Scratch {
    void* core = nullptr; void* chart = nullptr; void* primal = nullptr; void* direction_memory = nullptr;
    double *current = nullptr, *best = nullptr, *selected = nullptr, *center = nullptr;
    double *gradient = nullptr, *origin = nullptr, *hessian = nullptr, *reduced_h = nullptr;
    double *reduced_g = nullptr, *direction = nullptr, *neighbors = nullptr, *evidence = nullptr;
    double *cached_mean = nullptr, *cached_evidence = nullptr;
    double *seen = nullptr, *root_current = nullptr, *root_next = nullptr;
    bool bind(const Plan& p, void* memory, std::size_t bytes) noexcept {
        if (!p.bytes || !memory || bytes < p.bytes) return false;
        WorkspaceArena arena(memory, bytes);
        core = arena.take<double>(double_count(p.core));
        chart = arena.take<double>(double_count(p.chart));
        primal = arena.take<double>(double_count(p.primal));
        direction_memory = arena.take<double>(double_count(p.direction));
        current = arena.take<double>(p.k); best = arena.take<double>(p.k);
        selected = arena.take<double>(p.k); center = arena.take<double>(p.k);
        gradient = arena.take<double>(p.width); origin = arena.take<double>(p.k);
        hessian = arena.take<double>(p.square); reduced_h = arena.take<double>(p.reduced_square);
        reduced_g = arena.take<double>(p.d); direction = arena.take<double>(p.d);
        neighbors = arena.take<double>(p.neighbors); evidence = arena.take<double>(40u);
        if (p.mean_cache) {
            cached_mean = arena.take<double>(p.mean_cache);
            cached_evidence = arena.take<double>(40u);
        }
        seen = arena.take<double>(p.seen);
        root_current = arena.take<double>(p.root); root_next = arena.take<double>(p.root);
        return root_next != nullptr;
    }
};

template<class Density, class Orders, ActiveSet Active = ActiveSet::mean>
class Objective {
    const Layout& l_;
    const double* data_; std::size_t n_; double h0_;
    const Plan& p_; Scratch& s_;
    mean_screen::ObservationSummary sample_;
    using ScalarPlan = Primal<Density, Orders>;
    static constexpr const Layout* fixed = chart_layout<Density, Orders>;
    const Layout& layout() const noexcept {
        return evaluation_layout<Orders, Density, true>(l_);
    }
public:
    Objective(const Layout& l, const double* data, std::size_t n, double h0,
        const Plan& p, Scratch& s) noexcept : l_(l), data_(data), n_(n), h0_(h0), p_(p), s_(s),
            sample_(mean_screen::ObservationSummary::prepare(data, n)) {
        request_accounting::prepare(request_accounting::Preparation::sample, 0u);
    }
    bool mean_feasible(const double* theta) const noexcept {
        const auto& l = layout();
        for (std::size_t i = 0; i < l.mean_parameters; ++i)
            if (!std::isfinite(theta[i])) return false;
        if constexpr (Active == ActiveSet::mean) {
            for (std::size_t i = l.mean_parameters; i < l.parameters; ++i)
                if (theta[i] != s_.origin[i]) return false;
        } else if (!valid_theta<Density, fixed>(l, theta)) return false;
        return (!l.p_ar || reflection_inverse_values(theta + l.ar_start, l.p_ar,
                1., nullptr, s_.root_current, s_.root_next))
            && (!l.q_ma || reflection_inverse_values(theta + l.ma_start, l.q_ma,
                -1., nullptr, s_.root_current, s_.root_next));
    }
    bool initial_feasible(const double* theta) const noexcept {
        const auto& l = layout();
        return std::all_of(theta, theta + l.parameters, [](double v) { return std::isfinite(v); })
            && valid_theta<Density, fixed>(l, theta) && mean_feasible(theta);
    }
    bool value(const double* theta, double& f) const noexcept {
        request_accounting::PhysicalRequest accounting(1u, false);
        ScalarPlan primal(layout(), theta, sample_);
        request_accounting::prepare(request_accounting::Preparation::mean_primal, 0u);
        if (!primal.prepare(s_.primal, p_.primal)) return accounting.finish_bool(false);
        ValuePlan plan{primal};
        return accounting.finish_bool(evaluate_theta_order<0, true, Orders, Density, true, false, false,
            void, false, decltype(plan)>(layout(), theta, data_, n_, h0_, s_.core, p_.core,
                &f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &plan) && std::isfinite(f));
    }
    std::size_t pivot(ScalarPlan& primal) const noexcept {
        const auto& l = layout();
        std::size_t index = n_; double minimum = INFINITY;
        for (std::size_t t = maximum_lag<Orders>(l); t < n_; ++t) {
            const double residual = static_cast<double>(primal.advance(t));
            if (!std::isfinite(residual)) return n_;
            if (std::abs(residual) < minimum) { minimum = std::abs(residual); index = t; }
        }
        return index;
    }
    bool curvature(const double* theta, std::size_t& selected_pivot, double& f,
        Real& target, Real& error, std::size_t& requests) const noexcept {
        request_accounting::PhysicalRequest accounting(7u, false);
        using MeanPlan = Chart<2, Density, Orders, ScalarPlan, Active>;
        const auto& l = layout();
        ScalarPlan primal(l, theta, sample_);
        request_accounting::prepare(request_accounting::Preparation::mean_primal, 0u);
        if (!primal.prepare(s_.primal, p_.primal)) return accounting.finish_bool(false);
        selected_pivot = pivot(primal);
        if (selected_pivot == n_) return accounting.finish_bool(false);
        primal.rewind();
        MeanPlan chart(l, theta, data_, n_, selected_pivot, &primal);
        request_accounting::prepare(request_accounting::Preparation::residual_chart,2u);
        if (!chart.prepare(s_.chart, p_.chart)) return accounting.finish_bool(false);
        target = chart.pivot_value(); error = chart.anchor_error();
        ++requests;
        return accounting.finish_bool(evaluate_theta_order<2, true, Orders, Density, true, true, true,
            void, false, MeanPlan, mean_only_curvature<Density, Active>>(
            l, theta, data_, n_, h0_, s_.core, p_.core, &f, s_.gradient, s_.hessian,
            nullptr, nullptr, nullptr, nullptr, &chart));
    }
    int neighbors(const double* center, Real target, Real error, std::size_t pivot,
        std::size_t column, std::size_t radius) const noexcept {
        ScalarPlan primal(layout(), center, sample_);
        request_accounting::prepare(request_accounting::Preparation::mean_primal, 0u);
        if (!primal.prepare(s_.primal, p_.primal)) return -1;
        return neighbors_prevalidated<Density, Orders, ScalarPlan>(layout(), target, error,
            center, data_, n_, pivot, column, radius, s_.chart, p_.chart, s_.neighbors,
            s_.evidence, &primal);
    }
};

template<std::size_t FixedWidth = 0u>
inline double actual_slope(const double* gradient, const double* origin,
    const double* trial, const double* evidence, std::size_t supplied_width) noexcept {
    const auto width = FixedWidth ? FixedWidth : supplied_width;
    Real rho(evidence[2]); rho.low = evidence[3];
    Real sum = Real(gradient[0]) * rho;
    for (std::size_t i = 1; i < width; ++i)
        sum += Real(gradient[i]) * (Real(trial[i]) - Real(origin[i]));
    return static_cast<double>(sum);
}

} // namespace detail

template<class Density, class Orders, ActiveSet Active = ActiveSet::mean>
Plan workspace(const Layout& supplied) noexcept {
    if (!detail::valid_layout<Density, Orders>(supplied)) return {};
    const auto& l = evaluation_layout<Orders, Density, true>(supplied);
    Plan p = detail::order_extents<Density, Active>(l);
    if (!p.k) return {};
    p.core = core_workspace_bytes<2, true, chart_layout<Density, Orders>, mean_only_curvature<Density, Active>, false>(
        l, std::is_same_v<Density, generated::NormalDensity>);
    p.chart = Chart<2, Density, Orders, void, Active>::workspace_bytes(l);
    p.primal = detail::Primal<Density, Orders>::workspace_bytes(l);
    p.direction = bounded::make_layout(p.d).bytes;
    if (!p.core || !p.chart || (l.q_ma && !p.primal) || !p.direction) return {};
    WorkspaceSizer s;
    for (const auto bytes : {p.core, p.chart, p.primal, p.direction})
        if (!s.add<double>(detail::double_count(bytes))) return {};
    for (unsigned i = 0; i < 5; ++i) if (!s.add<double>(p.k)) return {};
    if (!s.add<double>(p.width)) return {};
    if (!s.add<double>(p.square) || !s.add<double>(p.reduced_square)
        || !s.add<double>(p.d) || !s.add<double>(p.d)
        || !s.add<double>(p.neighbors) || !s.add<double>(40u)) return {};
    if (p.mean_cache && (!s.add<double>(p.mean_cache) || !s.add<double>(40u))) return {};
    if (!s.add<double>(p.seen) || !s.add<double>(p.root) || !s.add<double>(p.root)) return {};
    p.bytes = s.size(); return p;
}

// At most eight full-theta points, with five evidence doubles per point.
// Only AR columns are qualified. The original chart's target and error stay fixed.
template<class Density, class Orders, class PrimalPlan = void>
int rounded_neighbors_prepared(const Layout& supplied, Real target, Real error,
    std::span<const double> center, std::span<const double> data,
    std::size_t pivot, std::size_t column, std::size_t radius,
    void* memory, std::size_t bytes, std::span<double> points, std::span<double> evidence,
    PrimalPlan* primal = nullptr) noexcept {
    if (!detail::valid_layout<Density, Orders>(supplied)) return -1;
    const auto& l = evaluation_layout<Orders, Density, true>(supplied);
    const auto required = Chart<1, Density, Orders, PrimalPlan>::workspace_bytes(l);
    std::size_t point_count = 0, center_bytes = 0, data_bytes = 0, point_bytes = 0, evidence_bytes = 0;
    if (center.size() != l.parameters || data.size() <= l.maximum_lag
        || pivot < l.maximum_lag || pivot >= data.size()
        || column < l.ar_start || column >= l.ma_start
        || radius > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / 2u
        || !required || bytes < required || !wide::finite(target) || !wide::finite(error)
        || !checked_product(8u, l.parameters, point_count) || points.size() < point_count
        || evidence.size() < 40u
        || !checked_product(center.size(), sizeof(double), center_bytes)
        || !checked_product(data.size(), sizeof(double), data_bytes)
        || !checked_product(points.size(), sizeof(double), point_bytes)
        || !checked_product(evidence.size(), sizeof(double), evidence_bytes)) return -1;
    const void* pointers[] = {center.data(), data.data(), memory, points.data(), evidence.data()};
    const std::size_t sizes[] = {center_bytes, data_bytes, bytes, point_bytes, evidence_bytes};
    if (!detail::disjoint_buffers(pointers, sizes)
        || !std::all_of(data.begin(), data.end(), [](double x) { return std::isfinite(x); })) return -1;
    return detail::neighbors_prevalidated<Density, Orders, PrimalPlan>(l, target, error,
        center.data(), data.data(), data.size(), pivot, column, radius, memory, bytes,
        points.data(), evidence.data(), primal);
}

struct Step {
    std::size_t iteration, pivot;
    double value, selected_value, selected_slope, best_value;
    std::size_t proposals, source, scale;
};
struct NoObserver {
    void operator()(const Step&, std::span<const double>) const noexcept {}
};

// The optional noexcept observer reads an ephemeral continuation candidate.
// It adds no trace storage or C ABI. Invalid inputs preserve both outputs.
// A finite initial estimate is retained even when later derivatives fail.
template<class Density, class Orders, class Observer = NoObserver,
         ActiveSet Active = ActiveSet::mean>
Result refine(const Layout& supplied, std::span<const double> initial,
    std::span<const double> data, double h0, std::size_t maximum_steps, std::size_t radius,
    void* memory, std::size_t bytes, std::span<double> continuation,
    std::span<double> best, Observer observe = {}, double maximum_step = 1e-3) noexcept {
    request_accounting::PhaseScope accounting_phase(request_accounting::Phase::refinement);
    static_assert(std::is_nothrow_invocable_v<Observer, const Step&, std::span<const double>>);
    Result result;
    const auto p = workspace<Density, Orders, Active>(supplied);
    std::size_t maximum_proposals = 0, maximum_lattice = 0, maximum_values = 0;
    std::size_t initial_bytes = 0, data_bytes = 0, continuation_bytes = 0, best_bytes = 0;
    const double step_scale = maximum_step / 1e-3;
    if (!std::isfinite(step_scale) || !(step_scale > 0.)
        || !p.bytes || initial.size() != p.k || continuation.size() < p.k || best.size() < p.k
        || bytes < p.bytes || !maximum_steps
        || radius > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / 2u
        || !checked_product(maximum_steps, p.trials, maximum_proposals)
        || !checked_sum(maximum_proposals, 1u, maximum_values)
        || !checked_product(maximum_steps, p.centers, maximum_lattice)
        || !checked_product(maximum_lattice, 2u * radius + 1u, maximum_lattice)
        || !checked_product(initial.size(), sizeof(double), initial_bytes)
        || !checked_product(data.size(), sizeof(double), data_bytes)
        || !checked_product(continuation.size(), sizeof(double), continuation_bytes)
        || !checked_product(best.size(), sizeof(double), best_bytes)) return result;
    const void* pointers[] = {initial.data(), data.data(), memory, continuation.data(), best.data()};
    const std::size_t sizes[] = {initial_bytes, data_bytes, bytes, continuation_bytes, best_bytes};
    if (!detail::disjoint_buffers(pointers, sizes)
        || !valid_data<true>(supplied, data.data(), data.size(), h0)) return result;
    const auto& l = evaluation_layout<Orders, Density, true>(supplied);
    detail::Scratch s;
    if (!s.bind(p, memory, bytes)) return result;
    const auto& e = detail::evaluation_extents<Density, Orders, Active>(p);
    std::copy_n(initial.data(), e.k, s.origin); std::copy_n(initial.data(), e.k, s.current);
    std::copy_n(initial.data(), e.k, s.best);
    detail::Objective<Density, Orders, Active> objective(l, data.data(), data.size(), h0, p, s);
    if (!objective.initial_feasible(initial.data())) return result;
    ++result.values;
    if (!objective.value(initial.data(), result.value)) { result.stop = Stop::initial_unavailable; return result; }
    result.continuation_value = result.value; result.stop = Stop::budget;
    for (std::size_t iteration = 0; iteration < maximum_steps; ++iteration) {
        std::size_t pivot = data.size();
        Real target, error; double value = INFINITY;
        if (!objective.curvature(s.current, pivot, value, target, error, result.curvatures)) {
            result.stop = Stop::numerical_failure; break;
        }
        ++result.iterations;
        // rho participates in the slope; the proposal metric uses the other active axes.
        for (std::size_t i = 0; i < e.d; ++i) {
            s.reduced_g[i] = s.gradient[i + 1];
            for (std::size_t j = 0; j < e.d; ++j)
                s.reduced_h[i * e.d + j] = s.hessian[(i + 1) * e.width + j + 1];
        }
        constexpr auto fixed_dimension = Orders::fixed ? (Active == ActiveSet::mean
            ? fixed_layout<Orders, Density, true>.mean_parameters
            : fixed_layout<Orders, Density, true>.parameters) - 1u : 0u;
        const auto direction = bounded::propose<fixed_dimension>(e.d,
            {s.reduced_g, e.d}, {s.reduced_h, e.reduced_square}, maximum_step,
            {s.direction, e.d}, s.direction_memory, p.direction);
        if (direction.status != bounded::Status::ok) { result.stop = Stop::numerical_failure; break; }
        std::size_t seen = 0; double selected_value = INFINITY, selected_slope = 0.;
        std::size_t selected_source = 0, selected_scale = 0;
        // The anchor and target are fixed in this iteration; volatility-only centers
        // have identical mean projections. Keep their distinct likelihood trials.
        int cached_mean_count = -1;
        const auto consider_center = [&](std::size_t source, std::size_t scale) noexcept {
            const auto& e = detail::evaluation_extents<Density, Orders, Active>(p);
            const auto& l = evaluation_layout<Orders, Density, true>(supplied);
            int count = -1;
            bool same_mean = false;
            if constexpr (Active == ActiveSet::joint) {
                same_mean = std::memcmp(s.center, s.current,
                    l.mean_parameters * sizeof(double)) == 0;
                if (same_mean && cached_mean_count >= 0) {
                    if (!wide::finite(target) || !wide::finite(error)
                        || !std::all_of(s.center, s.center + e.k,
                            [](double v) { return std::isfinite(v); })) return false;
                    count = cached_mean_count;
                    for (int row = 0; row < count; ++row) {
                        auto* point = s.neighbors + std::size_t(row) * e.k;
                        std::copy_n(s.cached_mean + std::size_t(row) * l.mean_parameters,
                            l.mean_parameters, point);
                        std::copy_n(s.center + l.mean_parameters,
                            e.k - l.mean_parameters, point + l.mean_parameters);
                    }
                    std::copy_n(s.cached_evidence, std::size_t(count) * 5u, s.evidence);
                }
            }
            if (count < 0) {
                count = objective.neighbors(s.center, target, error, pivot, 1u, radius);
                if (count < 0) return false;
                result.lattice += 2u * radius + 1u;
                if constexpr (Active == ActiveSet::joint) {
                    if (same_mean) {
                        cached_mean_count = count;
                        for (int row = 0; row < count; ++row)
                            std::copy_n(s.neighbors + std::size_t(row) * e.k,
                                l.mean_parameters, s.cached_mean + std::size_t(row) * l.mean_parameters);
                        std::copy_n(s.evidence, std::size_t(count) * 5u, s.cached_evidence);
                    }
                }
            }
            for (int row = 0; row < count; ++row) {
                const auto* trial = s.neighbors + std::size_t(row) * e.k;
                bool duplicate = false;
                for (std::size_t i = 0; i < seen; ++i)
                    if (std::memcmp(trial, s.seen + i * e.k, e.k * sizeof(double)) == 0) {
                        duplicate = true; break;
                    }
                if (duplicate) continue;
                if (seen >= e.trials) return false;
                std::copy_n(trial, e.k, s.seen + seen++ * e.k); ++result.proposals;
                if (!objective.mean_feasible(trial)) { ++result.rejected; continue; }
                double trial_value = INFINITY; ++result.values;
                if (!objective.value(trial, trial_value)) { ++result.rejected; continue; }
                if (trial_value < result.value) { result.value = trial_value; std::copy_n(trial, e.k, s.best); }
                const double slope = detail::actual_slope<Orders::fixed ? fixed_dimension + 1u : 0u>(s.gradient, s.current,
                    trial, s.evidence + 5u * row, e.d + 1u);
                volatile double increment = 1e-4 * slope;
                if (std::isfinite(slope) && slope < 0. && trial_value <= value + increment
                    && trial_value < selected_value) {
                    selected_value = trial_value; selected_slope = slope;
                    selected_source = source; selected_scale = scale;
                    std::copy_n(trial, e.k, s.selected);
                }
            }
            return true;
        };
        bool good = true;
        for (std::size_t scale = 0; scale < 5u && good; ++scale) {
            std::copy_n(s.current, e.k, s.center);
            const double fraction = std::ldexp(1., -int(scale));
            for (std::size_t i = 0; i < e.d; ++i) {
                volatile double displacement = fraction * s.direction[i];
                s.center[i + 1] += displacement;
            }
            good = consider_center(0u, scale);
        }
        double sum = 0.;
        for (std::size_t i = 0; i < e.d; ++i) {
            volatile double square = s.reduced_g[i] * s.reduced_g[i]; sum += square;
        }
        const double length = std::sqrt(sum);
        constexpr double scales[] = {1e-4, 1e-5, 1e-6, 1e-7, 1e-8};
        if (length > 0. && std::isfinite(length))
            for (std::size_t source = 0; source <= e.d && good; ++source)
                for (std::size_t scale = 0; scale < 5u && good; ++scale) {
                    std::copy_n(s.current, e.k, s.center);
                    for (std::size_t i = 0; i < e.d; ++i) {
                        const double d = source == 0u ? -s.reduced_g[i] / length
                            : source == i + 1u ? (s.reduced_g[i] > 0. ? -1. : s.reduced_g[i] < 0. ? 1. : 0.) : 0.;
                        volatile double displacement = (scales[scale] * step_scale) * d;
                        s.center[i + 1] += displacement;
                    }
                    good = consider_center(source + 1u, scale);
                }
        const bool accepted = std::isfinite(selected_value);
        observe(Step{iteration, pivot, value, selected_value, selected_slope,
            result.value, seen, selected_source, selected_scale},
            std::span<const double>(accepted ? s.selected : s.current, e.k));
        if (!good) { result.stop = Stop::numerical_failure; break; }
        if (!accepted) { result.stop = Stop::no_accepted_step; break; }
        ++result.accepted; result.continuation_value = selected_value;
        std::copy_n(s.selected, e.k, s.current);
    }
    std::copy_n(s.current, e.k, continuation.data()); std::copy_n(s.best, e.k, best.data());
    return result;
}

} // namespace vlk::experiment::runtime::aparch::mean_refinement
#endif
