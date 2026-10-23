#pragma once

#include <scivolapi/detail/types.hpp>
#include "../math/runtime_workspace.hpp"
#include "garch_observation_assembly.hpp"
#include <tlfloat/tlmath.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>

namespace scivolapi::detail::unit_map {

// Role metadata binds the model's existing full-theta layout, not a second
// parameter-order builder. Every axis not named here has identity units.
inline constexpr auto no_axis = std::numeric_limits<std::size_t>::max();
enum class ScaleCoordinate { absent, variance, power, log_variance };
struct Block {
    std::size_t begin = 0, count = 0;
    constexpr bool contains(std::size_t i) const noexcept {
        return i >= begin && i - begin < count;
    }
    constexpr bool operator==(const Block&) const = default;
};
struct LayoutSpec {
    std::size_t parameters = 0, intercept = no_axis;
    Block exogenous;
    ScaleCoordinate scale = ScaleCoordinate::absent;
    std::size_t scale_axis = no_axis, delta = no_axis;
    Block beta;
    constexpr bool operator==(const LayoutSpec&) const = default;
};
struct Layout {
    LayoutSpec roles;
    std::size_t hessian_elements = 0;
    bool valid = false;
    constexpr bool operator==(const Layout&) const = default;
};

[[nodiscard]] constexpr Layout make_layout(LayoutSpec s) noexcept {
    Layout result{s};
    const auto block_valid = [s](Block b) {
        return b.begin <= s.parameters && b.count <= s.parameters - b.begin;
    };
    const auto axis_valid = [s](std::size_t i) { return i == no_axis || i < s.parameters; };
    if (!block_valid(s.exogenous) || !block_valid(s.beta) || !axis_valid(s.intercept)
        || !axis_valid(s.scale_axis) || !axis_valid(s.delta)
        || (s.scale == ScaleCoordinate::absent) != (s.scale_axis == no_axis)
        || (s.scale == ScaleCoordinate::power) != (s.delta != no_axis)
        || (s.scale != ScaleCoordinate::log_variance && s.beta.count)
        || (s.scale != ScaleCoordinate::absent && s.scale != ScaleCoordinate::variance
            && s.scale != ScaleCoordinate::power && s.scale != ScaleCoordinate::log_variance))
        return result;
    for (auto axis : {s.intercept, s.scale_axis, s.delta}) {
        if (axis != no_axis && (s.exogenous.contains(axis) || s.beta.contains(axis))) return result;
    }
    if ((s.intercept != no_axis && (s.intercept == s.scale_axis || s.intercept == s.delta))
        || (s.delta != no_axis && s.delta == s.scale_axis)
        || (s.exogenous.count && s.beta.count
            && (s.exogenous.contains(s.beta.begin) || s.beta.contains(s.exogenous.begin)))
        || !vlk::experiment::runtime::checked_product(s.parameters, s.parameters, result.hessian_elements))
        return result;
    result.valid = true;
    return result;
}
struct RuntimeLayout {
    Layout value;
    constexpr Layout get() const noexcept { return value; }
};
template<LayoutSpec Spec> struct FixedLayout {
    static constexpr auto value = make_layout(Spec);
    static_assert(value.valid);
    constexpr Layout get() const noexcept { return value; }
};

// Public scale=s belongs to the sample adapter. These inputs are already in
// fitting/reporting units, not original observation units and not solver metrics.
struct UnitRequest {
    double response_divisor = 1;
    std::span<const double> exogenous_divisors;
    std::size_t n_eff = 0;
};
struct RecoveryLog {
    tlfloat::Quad quad;
    tlfloat::Octuple octuple;
};
template<class LayoutOwner = RuntimeLayout> class Plan {
public:
    const LayoutOwner& layout() const noexcept { return layout_; }
    const UnitRequest& units() const noexcept { return units_; }
    long double log_response() const noexcept { return log_response_; }
    unsigned recovery_order() const noexcept { return recovery_order_; }
    const std::optional<RecoveryLog>& derivative_log() const noexcept { return derivative_log_; }
private:
    LayoutOwner layout_{};
    UnitRequest units_{};
    long double log_response_ = 0;
    unsigned recovery_order_ = 0;
    std::optional<RecoveryLog> derivative_log_;
    template<class Owner> friend Status make_plan(Owner, UnitRequest, Plan<Owner>&, JetRequest) noexcept;
    template<class To, class From> friend Status rebind_plan(To, const Plan<From>&, Plan<To>&) noexcept;
};
enum class Direction { to_internal, to_fitting };
enum class StateCoordinate { response, variance, power, log_variance };

namespace implementation {
using vlk::experiment::runtime::WorkspaceArena;
using vlk::experiment::runtime::WorkspaceSizer;
using vlk::experiment::runtime::checked_product;
using vlk::experiment::model::scaled_derivative_product;
using Quad = tlfloat::Quad;
using Octuple = tlfloat::Octuple;

struct Range { std::uintptr_t begin = 0, end = 0; };
template<class T> inline bool range(std::span<T> s, Range& r) noexcept {
    r = {};
    if (s.empty()) return true;
    const auto begin = reinterpret_cast<std::uintptr_t>(s.data());
    std::size_t bytes = 0;
    if (!begin || begin % alignof(T) || !checked_product(s.size(), sizeof(T), bytes)
        || bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
    r = {begin, begin + bytes};
    return true;
}
inline bool overlaps(Range a, Range b) noexcept { return a.begin < b.end && b.begin < a.end; }
template<class T> inline std::span<const T> object(const T& x) noexcept { return {&x, 1}; }
template<std::size_t N> inline bool disjoint(const std::array<Range, N>& ranges) noexcept {
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (overlaps(ranges[i], ranges[j])) return false;
    return true;
}
inline Status narrow(long double x, double& result) noexcept {
    const double value = static_cast<double>(x);
    if (!std::isfinite(x) || !std::isfinite(value) || (x != 0 && value == 0))
        return Status::numerical_range;
    result = value;
    return Status::ok;
}
inline Status scaled(double x, double numerator, double denominator, double& result) noexcept {
    if (x == 0 || numerator == denominator) { result = x; return Status::ok; }
    const double value = scaled_derivative_product(x, numerator, denominator);
    if (!std::isfinite(value) || (x != 0 && value == 0)) return Status::numerical_range;
    result = value;
    return Status::ok;
}
inline Status squared(double x, double r, Direction direction, double& result) noexcept {
    if (x == 0 || r == 1) { result = x; return Status::ok; }
    const double value = direction == Direction::to_internal
        ? scaled_derivative_product(x, 1, 1, r, r)
        : scaled_derivative_product(x, r, r, 1, 1);
    if (!std::isfinite(value) || (x != 0 && value == 0)) return Status::numerical_range;
    result = value;
    return Status::ok;
}
struct PowerScale { long double exponent = 0, factor = 1; };
inline Status power_value(double x, PowerScale scale, double& out) noexcept {
    if (x == 0) { out = x; return Status::ok; }
    const long double product = static_cast<long double>(x) * scale.factor;
    if (std::isnormal(scale.factor) && std::isfinite(product) && product != 0)
        return narrow(product, out);
    // Recover representable products when r^-delta itself loses range. This
    // fallback changes arithmetic, not the model or its floating support bounds.
    const long double recovered = std::copysign(std::exp(std::log(std::abs(static_cast<long double>(x)))
        + scale.exponent), static_cast<long double>(x));
    if (recovered == 0) return Status::numerical_range;
    return narrow(recovered, out);
}
template<class Number> inline bool finite_number(const Number& x) noexcept {
    if constexpr (std::is_arithmetic_v<Number>) return std::isfinite(x);
    else return finite(x);
}
template<class Number> inline Number absolute(const Number& x) noexcept {
    using std::fabs;
    return fabs(x);
}
template<class Number> inline Number epsilon() noexcept {
    if constexpr (std::is_floating_point_v<Number>) return std::numeric_limits<Number>::epsilon();
    else if constexpr (std::is_same_v<Number, Quad>) return ldexp(Number(1), -112);
    else return ldexp(Number(1), -236);
}
template<class Number> inline Status narrow_number(const Number& x, double& result) noexcept {
    const double value = static_cast<double>(x);
    if (!finite_number(x) || !std::isfinite(value) || (x != 0 && value == 0))
        return Status::numerical_range;
    result = value;
    return Status::ok;
}
inline bool ordinary_contract(double coefficient, double row, double column, double& out) noexcept {
    if (!std::isfinite(coefficient) || !std::isnormal(row) || !std::isnormal(column)) return false;
    const double value = scaled_derivative_product(coefficient, row, column, 1, 1);
    if (coefficient != 0 && !std::isnormal(value)) return false;
    out = value;
    return true;
}
template<class Number, std::size_t N> inline Number sum(std::array<Number, N> terms) noexcept {
    std::sort(terms.begin(), terms.end(), [](const auto& a, const auto& b) {
        return absolute(a) > absolute(b);
    });
    Number result(0);
    for (const auto& x : terms) result += x;
    return result;
}
template<unsigned Order, class Number> struct Coefficients;
template<class Number> struct Coefficients<1, Number> {
    std::span<const Number> diagonal;
    Number factor = 1, coupling = 0, preparation_error = 0, log_response = 0;
};
template<class Number> struct Curvature { Number omega_delta = 0, delta_delta = 0; };
template<class Number> struct Coefficients<2, Number> : Coefficients<1, Number> {
    std::optional<Curvature<Number>> curvature;
};
template<class Number> inline void prepare_curvature(LayoutSpec roles,
    Coefficients<2, Number>& out, double omega) noexcept {
    if (roles.scale != ScaleCoordinate::power || out.curvature) return;
    out.curvature.emplace();
    out.curvature->omega_delta = -out.log_response;
    out.curvature->delta_delta = Number(omega)*out.log_response*out.log_response;
}

// One coefficient owner for all arithmetic intentions. Ordinary preparation
// consumes the value pass's log/exp; only lazy recovery calls vendor log/exp.
template<unsigned Order, class Number> inline Coefficients<Order, Number> coefficients(
    LayoutSpec roles, Number log_response, Number power, double omega, double delta,
    std::span<const Number> diagonal, bool second = true) noexcept {
    Coefficients<Order, Number> out;
    out.diagonal = diagonal;
    out.log_response = log_response;
    if (roles.scale == ScaleCoordinate::power) {
        out.factor = power;
        out.coupling = -Number(omega)*log_response;
    } else if (roles.scale == ScaleCoordinate::log_variance) {
        out.factor = 2*log_response;
        out.coupling = 1;
    }
    // Conservative libm/roundoff admission estimate, not a changed test or model
    // tolerance. Its assumptions require the production-flag MP qualification.
    out.preparation_error = (Number(4) + (roles.scale == ScaleCoordinate::power
        ? 2*absolute(Number(delta)*log_response) : Number(0)))*epsilon<Number>();
    if constexpr (Order == 2) if (second) prepare_curvature(roles,out,omega);
    return out;
}
template<class Number> inline Number diagonal_value(LayoutSpec roles, std::size_t i,
    double response, std::span<const double> divisors) noexcept {
    const Number r(response);
    if (i == roles.intercept) return Number(1)/r;
    if (roles.exogenous.contains(i)) return Number(divisors[i-roles.exogenous.begin])/r;
    if (i == roles.scale_axis && roles.scale == ScaleCoordinate::variance) return (Number(1)/r)/r;
    return Number(1);
}
inline bool coupled_axis(LayoutSpec s, double response, double omega, std::size_t i) noexcept {
    return response != 1 && ((s.scale == ScaleCoordinate::power && i == s.delta && omega != 0)
        || (s.scale == ScaleCoordinate::log_variance && s.beta.contains(i)));
}
inline bool curved_pair(LayoutSpec s, double response, double omega,
    std::size_t a, std::size_t b) noexcept {
    return s.scale == ScaleCoordinate::power && response != 1
        && ((a == s.scale_axis && b == s.delta) || (b == s.scale_axis && a == s.delta)
            || (a == s.delta && b == s.delta && omega != 0));
}
template<class Number> struct SeedEntry {
    Number coefficient = 0;
    unsigned degree = 0;
    bool active = false;
};
template<unsigned Order, class Number> inline SeedEntry<Number> first_entry(
    LayoutSpec s, double response, double omega, const Coefficients<Order, Number>& c,
    std::size_t row, std::size_t column) noexcept {
    if (row == column)
        return {c.diagonal[row], unsigned(s.scale == ScaleCoordinate::power && row == s.scale_axis), true};
    if (row == s.scale_axis && coupled_axis(s, response, omega, column))
        return {c.coupling, 1, true};
    return {};
}
template<class Number> inline SeedEntry<Number> second_entry(
    LayoutSpec s, double response, double omega, const Coefficients<2, Number>& c,
    std::size_t row, std::size_t a, std::size_t b) noexcept {
    if (row != s.scale_axis || !curved_pair(s, response, omega, a, b)) return {};
    if (!c.curvature) return {Number(0), 1, true};
    return {a == s.delta && b == s.delta ? c.curvature->delta_delta : c.curvature->omega_delta, 1, true};
}
struct PullbackCounts {
    std::size_t ordinary = 0, quad = 0, octuple = 0;
    std::size_t quad_preparations = 0, octuple_preparations = 0;
};
inline bool valid_request(JetRequest r) noexcept {
    const auto bits = static_cast<unsigned>(r);
    return bits > 0 && bits <= 7;
}
inline unsigned derivative_order(JetRequest r) noexcept {
    return (static_cast<unsigned>(r) & 4) ? 2 : (static_cast<unsigned>(r) & 2) ? 1 : 0;
}
template<class Owner> inline bool valid_plan(const Plan<Owner>& p) noexcept {
    const auto l = p.layout().get();
    Range r;
    if (!l.valid || l != make_layout(l.roles) || !std::isfinite(p.units().response_divisor)
        || p.units().response_divisor <= 0 || !std::isfinite(p.log_response())
        || p.units().exogenous_divisors.size() != l.roles.exogenous.count
        || !range(p.units().exogenous_divisors, r) || p.recovery_order() > 2) return false;
    for (double d : p.units().exogenous_divisors) if (!std::isfinite(d) || d <= 0) return false;
    const bool needs_log = p.recovery_order() && p.units().response_divisor != 1
        && (l.roles.scale == ScaleCoordinate::power || l.roles.scale == ScaleCoordinate::log_variance);
    if (needs_log != p.derivative_log().has_value()
        || (p.derivative_log() && (!finite_number(p.derivative_log()->quad)
            || !finite_number(p.derivative_log()->octuple)))) return false;
    return true;
}
template<class Owner> inline Status theta_value(const Plan<Owner>& p,
    std::span<const double> theta, std::size_t i, Direction direction, double& out,
    const PowerScale* prepared_power = nullptr) noexcept {
    const auto s = p.layout().get().roles;
    const auto r = p.units().response_divisor;
    if (!std::isfinite(theta[i])) return Status::invalid_argument;
    if (i == s.intercept)
        return direction == Direction::to_internal ? scaled(theta[i], 1, r, out)
                                                   : scaled(theta[i], r, 1, out);
    if (s.exogenous.contains(i)) {
        const double d = p.units().exogenous_divisors[i - s.exogenous.begin];
        return direction == Direction::to_internal ? scaled(theta[i], d, r, out)
                                                   : scaled(theta[i], r, d, out);
    }
    if (i != s.scale_axis || r == 1) { out = theta[i]; return Status::ok; }
    if (s.scale == ScaleCoordinate::variance) return squared(theta[i], r, direction, out);
    if (s.scale == ScaleCoordinate::power) {
        if (!std::isfinite(theta[s.delta])) return Status::invalid_argument;
        if (theta[s.delta] <= 0) return Status::mathematical_domain;
        if (theta[i] == 0) { out = theta[i]; return Status::ok; }
        PowerScale power;
        if (prepared_power) power = *prepared_power;
        else {
            power.exponent = (direction == Direction::to_internal ? -1 : 1)
                * static_cast<long double>(theta[s.delta]) * p.log_response();
            power.factor = std::exp(power.exponent);
        }
        return power_value(theta[i], power, out);
    }
    // Accumulate 1-sum(beta) with the existing compensation owner. No second
    // log-variance recursion or density equation is introduced by the unit map.
    vlk::experiment::generated::NormalDensity::ValueAccumulator complement;
    complement.add(1);
    for (std::size_t j = 0; j < s.beta.count; ++j) {
        const double b = theta[s.beta.begin + j];
        if (!std::isfinite(b)) return Status::invalid_argument;
        complement.add(-b);
    }
    const long double difference = static_cast<long double>(complement.sum) + complement.correction;
    return narrow(static_cast<long double>(theta[i])
        + (direction == Direction::to_internal ? -2 : 2) * p.log_response() * difference, out);
}
} // namespace implementation

// Borrowed exogenous divisors must remain immutable for the plan and its points.
template<class Owner> [[nodiscard]] inline Status make_plan(Owner owner, UnitRequest units,
    Plan<Owner>& out, JetRequest planned_requests = JetRequest::FGH) noexcept {
    if (!implementation::valid_request(planned_requests)) return Status::invalid_argument;
    Plan<Owner> candidate;
    candidate.layout_ = owner;
    candidate.units_ = units;
    if (!implementation::valid_plan(candidate)) return Status::invalid_argument;
    implementation::Range scales, destination;
    if (!implementation::range(units.exogenous_divisors, scales)
        || !implementation::range(implementation::object(out), destination)
        || implementation::overlaps(scales, destination)) return Status::invalid_argument;
    candidate.log_response_ = std::log(static_cast<long double>(units.response_divisor));
    candidate.recovery_order_ = implementation::derivative_order(planned_requests);
    const auto scale = owner.get().roles.scale;
    if (candidate.recovery_order_ && units.response_divisor != 1
        && (scale == ScaleCoordinate::power || scale == ScaleCoordinate::log_variance)) {
        const auto logarithm = tlfloat::log(tlfloat::Octuple(units.response_divisor));
        candidate.derivative_log_ = RecoveryLog{tlfloat::Quad(logarithm),logarithm};
    }
    if (candidate.derivative_log_ && (!implementation::finite_number(candidate.derivative_log_->quad)
        || !implementation::finite_number(candidate.derivative_log_->octuple)))
        return Status::numerical_range;
    // Own the recovery log by value. A point copies this admitted plan and may
    // outlive its original Plan object without a borrowed lazy-cache pointer.
    out = candidate;
    return Status::ok;
}

// Rebind admitted metadata without repeating sample-lifetime logarithms. The
// complete role layout must match; a claimed fixed dispatch cannot fall back.
template<class To, class From> [[nodiscard]] inline Status rebind_plan(
    To owner, const Plan<From>& source, Plan<To>& out) noexcept {
    if (owner.get() != source.layout().get() || !implementation::valid_plan(source))
        return Status::invalid_argument;
    implementation::Range divisors, destination;
    if (!implementation::range(source.units().exogenous_divisors, divisors)
        || !implementation::range(implementation::object(out), destination)
        || implementation::overlaps(divisors, destination)) return Status::invalid_argument;
    Plan<To> candidate;
    candidate.layout_ = owner;
    candidate.units_ = source.units();
    candidate.log_response_ = source.log_response();
    candidate.recovery_order_ = source.recovery_order();
    candidate.derivative_log_ = source.derivative_log();
    out = candidate;
    return Status::ok;
}

template<unsigned Order> struct MapDerivatives;
template<> struct MapDerivatives<0> {};
template<unsigned Order> struct DerivativeCache {
    // Normal coefficients are prepared once. Unsafe materializations are never
    // statistical zeros: recovery rebuilds them from the immutable raw units.
    std::span<const long double> diagonal;
    std::span<implementation::Quad> quad_diagonal;
    std::span<implementation::Octuple> octuple_diagonal;
    implementation::Coefficients<Order, long double> ordinary;
    double original_omega = 0, original_delta = 0;
    mutable std::optional<implementation::Coefficients<Order, implementation::Quad>> quad;
    mutable std::optional<implementation::Coefficients<Order, implementation::Octuple>> octuple;
};
template<> struct MapDerivatives<1> : DerivativeCache<1> {};
template<> struct MapDerivatives<2> : DerivativeCache<2> {};
struct InternalJet;
struct FittingJet;
template<unsigned Order, class Owner = RuntimeLayout, class StorageOwner = void> class PreparedPoint;
template<unsigned Order, class Owner> class PreparedPoint<Order, Owner, void> {
public:
    static_assert(Order <= 2);
    const Plan<Owner>& plan() const noexcept { return plan_; }
    std::span<const double> internal_theta() const noexcept { return internal_theta_; }
    const MapDerivatives<Order>& derivatives() const noexcept { return derivatives_; }
    bool valid() const noexcept { return valid_; }
    const PreparedPoint& storage() const noexcept { return *this; }
private:
    Plan<Owner> plan_;
    std::span<const double> internal_theta_;
    MapDerivatives<Order> derivatives_;
    bool valid_ = false;
    template<unsigned O, class L, class S> friend Status prepare(const Plan<L>&, std::span<const double>,
        std::span<std::byte>, PreparedPoint<O, S>&) noexcept;
};

// A lexical view of one existing point, not a second prepared state or cache.
// The source slot and its arena must remain alive and unchanged while borrowed.
template<unsigned Order, class Owner, class StorageOwner> class PreparedPoint {
public:
    static_assert(Order <= 2 && !std::is_void_v<StorageOwner>);
    const Plan<Owner>& plan() const noexcept { return plan_; }
    std::span<const double> internal_theta() const noexcept { return storage_->internal_theta(); }
    const MapDerivatives<Order>& derivatives() const noexcept { return storage_->derivatives(); }
    bool valid() const noexcept { return storage_->valid(); }
    const PreparedPoint<Order, StorageOwner>& storage() const noexcept { return *storage_; }
private:
    PreparedPoint(Plan<Owner> plan, const PreparedPoint<Order, StorageOwner>& storage) noexcept
        : plan_(plan), storage_(&storage) {}
    Plan<Owner> plan_;
    const PreparedPoint<Order, StorageOwner>* storage_;
    template<class To, unsigned O, class From>
    friend std::optional<PreparedPoint<O, To, From>> rebind_point(
        To, const PreparedPoint<O, From>&) noexcept;
};

template<class To, unsigned Order, class From>
[[nodiscard]] inline std::optional<PreparedPoint<Order, To, From>> rebind_point(
    To owner, const PreparedPoint<Order, From>& source) noexcept {
    if (!source.valid()) return std::nullopt;
    Plan<To> plan;
    if (rebind_plan(owner, source.plan(), plan) != Status::ok) return std::nullopt;
    return PreparedPoint<Order, To, From>(plan, source);
}
template<class To, unsigned Order, class From>
std::optional<PreparedPoint<Order, To, From>> rebind_point(
    To, const PreparedPoint<Order, From>&&) = delete;
struct Requirements {
    Status status = Status::invalid_argument;
    std::size_t bytes = 0, alignment = 1;
    unsigned order = 0;
};
template<class Owner> [[nodiscard]] inline Requirements requirements(const Plan<Owner>& p,
                                                                    JetRequest request) noexcept {
    Requirements out;
    if (!implementation::valid_request(request) || !implementation::valid_plan(p)) return out;
    out.order = implementation::derivative_order(request);
    if (out.order > p.recovery_order()) { out.status = Status::unsupported_operation; return out; }
    implementation::WorkspaceSizer s;
    const auto n = p.layout().get().roles.parameters;
    if (!s.add<double>(n) || (out.order && (!s.add<long double>(n)
        || !s.add<implementation::Quad>(n) || !s.add<implementation::Octuple>(n)))) {
        out.status = Status::insufficient_capacity;
        return out;
    }
    out.bytes = s.size();
    out.alignment = out.order ? std::max({alignof(long double), alignof(implementation::Quad),
        alignof(implementation::Octuple)}) : alignof(double);
    out.status = Status::ok;
    return out;
}

template<unsigned Order, class Owner, class StorageOwner> [[nodiscard]] inline Status prepare(const Plan<Owner>& p,
    std::span<const double> fitting_theta, std::span<std::byte> workspace,
    PreparedPoint<Order, StorageOwner>& out) noexcept {
    static_assert(Order <= 2);
    const auto needs = requirements(p, Order == 2 ? JetRequest::H : Order == 1 ? JetRequest::G : JetRequest::F);
    if (needs.status != Status::ok) return needs.status;
    const auto roles = p.layout().get().roles;
    const auto n = roles.parameters;
    if (fitting_theta.size() != n) return Status::invalid_argument;
    if (workspace.size() < needs.bytes) return Status::insufficient_capacity;
    std::array<implementation::Range, 3> protected_ranges;
    implementation::Range scratch;
    if (!implementation::range(fitting_theta, protected_ranges[0])
        || !implementation::range(p.units().exogenous_divisors, protected_ranges[1])
        || !implementation::range(implementation::object(p), protected_ranges[2])
        || !implementation::range(workspace.first(needs.bytes), scratch)
        || (needs.bytes && scratch.begin % needs.alignment)) return Status::invalid_argument;
    implementation::Range output_object;
    if (!implementation::range(implementation::object(out), output_object)
        || implementation::overlaps(scratch, output_object)) return Status::invalid_argument;
    for (auto r : protected_ranges)
        if (implementation::overlaps(r, scratch) || implementation::overlaps(r, output_object))
            return Status::invalid_argument;
    // Admission is complete. A failed new trial must not leave this slot marked
    // valid after its caller arena has been partly overwritten.
    out.valid_ = false;
    for (double t : fitting_theta) if (!std::isfinite(t)) return Status::invalid_argument;
    if (roles.scale == ScaleCoordinate::power && fitting_theta[roles.delta] <= 0)
        return Status::mathematical_domain;
    implementation::WorkspaceArena arena(workspace.data(), needs.bytes);
    double* theta = n ? arena.take<double>(n) : nullptr;
    if (n && !theta) return Status::insufficient_capacity;
    PreparedPoint<Order, StorageOwner> candidate;
    if constexpr (std::is_same_v<Owner, StorageOwner>) candidate.plan_ = p;
    else {
        const auto storage_owner = [&] {
            if constexpr (std::is_same_v<StorageOwner, RuntimeLayout>) return RuntimeLayout{p.layout().get()};
            else return StorageOwner{};
        }();
        if (rebind_plan(storage_owner, p, candidate.plan_) != Status::ok) return Status::invalid_argument;
    }
    candidate.internal_theta_ = {theta, n};
    implementation::PowerScale power;
    if (roles.scale == ScaleCoordinate::power && p.units().response_divisor != 1
        && (Order != 0 || fitting_theta[roles.scale_axis] != 0)) {
        power.exponent = -static_cast<long double>(fitting_theta[roles.delta]) * p.log_response();
        power.factor = std::exp(power.exponent);
    }
    for (std::size_t i = 0; i < n; ++i) {
        const auto status = implementation::theta_value(p, fitting_theta, i, Direction::to_internal, theta[i], &power);
        if (status != Status::ok) return status;
    }
    if constexpr (Order >= 1) {
        long double* diagonal = n ? arena.take<long double>(n) : nullptr;
        auto* quad = n ? arena.take<implementation::Quad>(n) : nullptr;
        auto* octuple = n ? arena.take<implementation::Octuple>(n) : nullptr;
        if (n && (!diagonal || !quad || !octuple)) return Status::insufficient_capacity;
        for (std::size_t i = 0; i < n; ++i)
            diagonal[i] = implementation::diagonal_value<long double>(roles, i,
                p.units().response_divisor, p.units().exogenous_divisors);
        candidate.derivatives_.diagonal = {diagonal, n};
        candidate.derivatives_.quad_diagonal = {quad, n};
        candidate.derivatives_.octuple_diagonal = {octuple, n};
        if (roles.scale == ScaleCoordinate::power) {
            candidate.derivatives_.original_omega = fitting_theta[roles.scale_axis];
            candidate.derivatives_.original_delta = fitting_theta[roles.delta];
        }
        candidate.derivatives_.ordinary = implementation::coefficients<Order>(roles,
            p.log_response(), power.factor, candidate.derivatives_.original_omega,
            candidate.derivatives_.original_delta, candidate.derivatives_.diagonal);
    }
    candidate.valid_ = true;
    out = candidate;
    return Status::ok;
}

// Prevalidated point access. Both full pullback and selected consumers borrow
// this cache; log/exp preparation has no seed/observation/pair lifetime.
template<class Number, bool NeedCurvature, unsigned Order, class Owner, class StorageOwner>
inline const implementation::Coefficients<Order, Number>& coefficient_view(
    const PreparedPoint<Order, Owner, StorageOwner>& point,
    implementation::PullbackCounts* counts = nullptr) noexcept {
    static_assert(Order >= 1 && (!NeedCurvature || Order == 2));
    static_assert(std::is_same_v<Number, long double> || std::is_same_v<Number, implementation::Quad>
        || std::is_same_v<Number, implementation::Octuple>);
    const auto& derivatives = point.derivatives();
    if constexpr (std::is_same_v<Number, long double>) return derivatives.ordinary;
    else {
        const auto& plan = point.plan();
        const auto s = plan.layout().get().roles;
        auto& cache = [&]() -> auto& {
            if constexpr (std::is_same_v<Number, implementation::Quad>) return derivatives.quad;
            else return derivatives.octuple;
        }();
        if (!cache) {
            auto buffer = [&]() {
                if constexpr (std::is_same_v<Number, implementation::Quad>) return derivatives.quad_diagonal;
                else return derivatives.octuple_diagonal;
            }();
            for (std::size_t k = 0; k < s.parameters; ++k)
                std::construct_at(buffer.data() + k, implementation::diagonal_value<Number>(s, k,
                    plan.units().response_divisor, plan.units().exogenous_divisors));
            Number l(0), a(1);
            if (plan.derivative_log()) {
                if constexpr (std::is_same_v<Number, implementation::Quad>) l = plan.derivative_log()->quad;
                else l = plan.derivative_log()->octuple;
            }
            if (s.scale == ScaleCoordinate::power)
                a = tlfloat::exp(-Number(derivatives.original_delta) * l);
            cache = implementation::coefficients<Order>(s, l, a, derivatives.original_omega,
                derivatives.original_delta, std::span<const Number>(buffer), NeedCurvature);
            if (counts) {
                if constexpr (std::is_same_v<Number, implementation::Quad>) ++counts->quad_preparations;
                else ++counts->octuple_preparations;
            }
        }
        if constexpr (Order == 2 && NeedCurvature)
            implementation::prepare_curvature(s, *cache, derivatives.original_omega);
        return *cache;
    }
}

// H-only public requests need the internal omega gradient for curved APARCH.
template<class Owner> [[nodiscard]] inline JetRequest internal_request(const Plan<Owner>& p,
                                                                      JetRequest request) noexcept {
    auto bits = static_cast<unsigned>(request);
    if ((bits & 4) && p.layout().get().roles.scale == ScaleCoordinate::power && p.units().response_divisor != 1)
        bits |= 2;
    return static_cast<JetRequest>(bits);
}
template<class Owner> [[nodiscard]] inline Status restore_likelihood(const Plan<Owner>& p,
    double internal, double& fitting) noexcept {
    return implementation::narrow(static_cast<long double>(internal)
        + static_cast<long double>(p.units().n_eff) * p.log_response(), fitting);
}
struct InternalJet {
    std::span<const double> value, gradient, hessian; // dense row-major H
    JetValidity validity;
    Status reason(unsigned bit) const noexcept {
        if (validity.valid_mask & bit) return Status::ok;
        const auto status = bit == 1 ? validity.f_reason
            : bit == 2 ? validity.g_reason : validity.h_reason;
        return status != Status::ok ? status : validity.status != Status::ok
            ? validity.status : Status::unsupported_operation;
    }
};
struct FittingJet {
    std::span<double> value, gradient, hessian;
};
enum class ContractionAdmission { invalid, insufficient_precision, ready };
template<class Gradient, class Hessian> struct ContractionInput {
    Gradient gradient;
    Hessian hessian;
    static constexpr bool binary64_inputs = std::is_same_v<
        std::remove_cvref_t<decltype(std::declval<Gradient>()(std::size_t{}))>, double>
        && std::is_same_v<std::remove_cvref_t<decltype(std::declval<Hessian>()(std::size_t{}, std::size_t{}))>, double>;
};
template<class Gradient, class Hessian> inline auto contraction_input(Gradient gradient, Hessian hessian) noexcept {
    return ContractionInput<Gradient, Hessian>{gradient, hessian};
}

namespace implementation {
template<class Number> struct Evaluation {
    Number value = 0, magnitude = 0;
    bool valid = true, uses_preparation = true;
};
inline bool identity_diagonal(LayoutSpec s, std::size_t i, double r,
                              std::span<const double> divisors) noexcept {
    if (s.exogenous.contains(i)) return divisors[i-s.exogenous.begin] == r;
    if (i == s.intercept || (i == s.scale_axis && s.scale == ScaleCoordinate::variance)) return r == 1;
    return true;
}

template<class Number, class Coefficient> inline Number contraction_product(Evaluation<Number>& out,
    const Coefficient& coefficient, Number row, Number column,
    Number row_unit = Number(1), Number column_unit = Number(1)) noexcept {
    if (!finite_number(coefficient)) { out.valid = false; return Number(0); }
    if (coefficient == 0) return Number(0);
    if (!finite_number(row) || !finite_number(column) || row == 0 || column == 0) {
        out.valid = false; return Number(0);
    }
    const auto first = Number(coefficient) * row;
    auto value = first * column;
    if (!finite_number(first) || !finite_number(value) || first == 0 || value == 0) out.valid = false;
    if constexpr (std::is_floating_point_v<Number>)
        if (!std::isnormal(row) || !std::isnormal(column) || !std::isnormal(first)
            || !std::isnormal(value)) out.valid = false;
    for (const auto& unit : {row_unit, column_unit}) {
        if (!finite_number(unit) || unit <= 0) { out.valid = false; return Number(0); }
        if (unit == 1) continue;
        value /= unit;
        if (!finite_number(value) || value == 0) out.valid = false;
        if constexpr (std::is_floating_point_v<Number>)
            if (!std::isnormal(value)) out.valid = false;
    }
    return value;
}
template<bool Binary64Inputs, unsigned Order, class Number, std::size_t N> inline Evaluation<Number> reduce_terms(
    Evaluation<Number> out, const Coefficients<Order, Number>& c,
    std::array<std::array<Number, N>, 3> terms,
    const std::array<std::array<bool, N>, 3>& exact) noexcept {
    if (!out.valid) return out;
    std::array<Number, 3> coefficient{}, amplitude{};
    for (unsigned p = 0; p < 3; ++p) {
        for (std::size_t a = 0; a < N; ++a) if (exact[p][a] && terms[p][a] != 0)
            for (std::size_t b = a + 1; b < N; ++b)
                if (exact[p][b] && terms[p][a] == -terms[p][b]) {
                    terms[p][a] = 0; terms[p][b] = 0; break;
                }
        coefficient[p] = sum(terms[p]);
        for (const auto& term : terms[p]) amplitude[p] += absolute(term);
    }
    if (coefficient[1] == 0 && coefficient[2] == 0 && amplitude[1] == 0 && amplitude[2] == 0) {
        out.value = coefficient[0]; out.magnitude = amplitude[0];
        out.uses_preparation = false; return out;
    }
    if (!finite_number(c.factor)) { out.valid = false; return out; }
    if (c.factor == 0) {
        if constexpr (!Binary64Inputs || std::is_floating_point_v<Number>) {
            out.valid = false; return out;
        }
        // This range proof belongs only to the original finite-binary64 map
        // inputs, not to arbitrary already-wide recursive sensitivities.
        out.valid = coefficient[0] != 0;
        out.value = coefficient[0]; out.magnitude = amplitude[0];
        out.uses_preparation = false; return out;
    }
    std::array<Number, 3> weighted{coefficient[0], 0, 0};
    out.magnitude = amplitude[0];
    for (unsigned p = 1; p < 3; ++p) {
        weighted[p] = coefficient[p];
        Number bound = amplitude[p];
        for (unsigned k = 0; k < p; ++k) {
            weighted[p] *= c.factor;
            bound *= absolute(c.factor);
        }
        if (!finite_number(weighted[p]) || !finite_number(bound)
            || (coefficient[p] != 0 && weighted[p] == 0)) out.valid = false;
        if constexpr (std::is_floating_point_v<Number>)
            if (coefficient[p] != 0 && !std::isnormal(weighted[p])) out.valid = false;
        out.magnitude += bound;
    }
    if (out.valid) out.value = sum(weighted);
    out.valid = out.valid && finite_number(out.value) && finite_number(out.magnitude);
    return out;
}

// APARCH uses a=r^-delta; EGARCH uses a=2*log(r). Both sparse maps are
// degree-two polynomials in a. The same formula serves every arithmetic tier.
template<bool Hessian, unsigned Order, class Number, class Input> inline Evaluation<Number> contract(
    LayoutSpec s, double response, std::span<const double> divisors,
    double original_omega, const Coefficients<Order, Number>& c,
    Input input, std::size_t i, std::size_t j = 0) noexcept {
    Evaluation<Number> out;
    std::array<std::array<Number, 5>, 3> terms{};
    std::array<std::array<bool, 5>, 3> exact{};
    const auto w = s.scale_axis;
    const bool power = s.scale == ScaleCoordinate::power;
    const auto degree = [s, response, original_omega, &c](std::size_t k) {
        return first_entry(s, response, original_omega, c, k, k).degree;
    };
    const auto coupled = [s, response, original_omega](std::size_t k) {
        return coupled_axis(s, response, original_omega, k);
    };
    const auto product = [&out](const auto& coefficient, Number row, Number column) {
        return contraction_product(out, coefficient, row, column);
    };
    const auto di = first_entry(s, response, original_omega, c, i, i).coefficient;
    const auto ci = first_entry(s, response, original_omega, c, w, i).coefficient;
    const bool idi = Input::binary64_inputs && identity_diagonal(s, i, response, divisors);
    if constexpr (!Hessian) {
        terms[degree(i)][0] = product(input.gradient(i), di, Number(1));
        exact[degree(i)][0] = idi;
        if (coupled(i)) {
            terms[1][1] = product(input.gradient(w), ci, Number(1));
            exact[1][1] = Input::binary64_inputs && !power;
        }
    } else {
        static_assert(Order == 2);
        const auto dj = first_entry(s, response, original_omega, c, j, j).coefficient;
        const auto cj = first_entry(s, response, original_omega, c, w, j).coefficient;
        const bool idj = Input::binary64_inputs && identity_diagonal(s, j, response, divisors);
        terms[degree(i)+degree(j)][0] = product(input.hessian(i, j), di, dj);
        exact[degree(i)+degree(j)][0] = idi && idj;
        if (coupled(i)) {
            terms[1+degree(j)][1] = product(input.hessian(w, j), ci, dj);
            exact[1+degree(j)][1] = !power && idj;
        }
        if (coupled(j)) {
            terms[degree(i)+1][2] = product(input.hessian(i, w), di, cj);
            exact[degree(i)+1][2] = !power && idi;
        }
        if (coupled(i) && coupled(j)) {
            terms[2][3] = product(input.hessian(w, w), ci, cj);
            exact[2][3] = Input::binary64_inputs && !power;
        }
        if (power && response != 1) {
            const bool needs_curvature = curved_pair(s, response, original_omega, i, j);
            if (needs_curvature && !c.curvature) { out.valid = false; return out; }
            if (needs_curvature)
                terms[1][4] = product(input.gradient(w),
                    second_entry(s, response, original_omega, c, w, i, j).coefficient, Number(1));
        }
    }
    return reduce_terms<Input::binary64_inputs>(out, c, terms, exact);
}
template<class Number, unsigned Order> inline bool sufficient_precision(
    const Evaluation<Number>& e, const Coefficients<Order, Number>& c, std::size_t operations = 32) noexcept {
    if (!e.valid) return false;
    if (e.magnitude == 0) return e.value == 0;
    const auto unit = epsilon<Number>();
    const Number steps(std::max(operations, std::size_t(32)));
    if (!finite_number(steps) || steps * unit >= 1) return false;
    const auto gamma = steps * unit / (Number(1) - steps * unit);
    const auto error = (gamma+(e.uses_preparation ? 4*c.preparation_error : Number(0)))*e.magnitude;
    // An internal admission budget, stricter than the unchanged 2e-13 MP gate.
    // Do not increase it to force a faster tier through a cancellation case.
    const Number budget(64*std::numeric_limits<double>::epsilon());
    return finite_number(error) && error <= budget*absolute(e.value);
}
template<class Number, unsigned Order> inline ContractionAdmission admission(
    const Evaluation<Number>& e, const Coefficients<Order, Number>& c, std::size_t operations = 32) noexcept {
    if (!e.valid || !finite_number(e.value) || !finite_number(e.magnitude) || e.magnitude < 0)
        return ContractionAdmission::invalid;
    return sufficient_precision(e, c, operations) ? ContractionAdmission::ready
        : ContractionAdmission::insufficient_precision;
}
} // namespace implementation

// Enclose a round-to-nearest nonnegative expansion estimate once at the output
// boundary. depth bounds rounding-factor multiplicity, not just unique DAG ops.
// Subnormal/underflowed bound arithmetic must already invalidate its attempt.
template<class Number> [[nodiscard]] inline ContractionAdmission enclose_magnitude(
    implementation::Evaluation<Number>& e, std::size_t depth) noexcept {
    static_assert(std::is_same_v<Number, long double> || std::is_same_v<Number, implementation::Quad>
        || std::is_same_v<Number, implementation::Octuple>);
    if (!e.valid || !implementation::finite_number(e.value)
        || !implementation::finite_number(e.magnitude) || e.magnitude < 0)
        return ContractionAdmission::invalid;
    if (e.magnitude == 0)
        return e.value == 0 ? ContractionAdmission::ready : ContractionAdmission::invalid;
    if constexpr (std::is_floating_point_v<Number>) {
        if (!std::isnormal(e.magnitude)) return ContractionAdmission::invalid;
    } else if (e.magnitude < Number::flt_min()) return ContractionAdmission::invalid;
    if (depth == 0) return ContractionAdmission::ready;
    using std::nextafter;
    const Number infinity(std::numeric_limits<double>::infinity());
    const auto up = [&infinity](Number x) { return nextafter(x, infinity); };
    const Number steps = up(Number(depth));
    const Number relative = up(steps * implementation::epsilon<Number>());
    if (!implementation::finite_number(relative) || relative >= 1)
        return ContractionAdmission::insufficient_precision;
    const Number denominator = nextafter(Number(1) - relative, Number(0));
    if (!(denominator > 0)) return ContractionAdmission::insufficient_precision;
    const Number enclosed = up(e.magnitude / denominator);
    if (!implementation::finite_number(enclosed)) return ContractionAdmission::invalid;
    e.magnitude = enclosed;
    return ContractionAdmission::ready;
}

// Borrowed typed actions. Their values can remain in caller-owned derivative
// rings; finish is exclusively the completed public-output boundary.
template<class Number, bool NeedCurvature, unsigned Order, class Owner = RuntimeLayout,
    class StorageOwner = void> class MapActions {
public:
    using Evaluation = implementation::Evaluation<Number>;
    MapActions(const PreparedPoint<Order, Owner, StorageOwner>& point,
        const implementation::Coefficients<Order, Number>& coefficients) noexcept
        : point_(&point), coefficients_(&coefficients) {}
    const auto& coefficients() const noexcept { return *coefficients_; }
    template<class Payload> Evaluation first(std::size_t row, std::size_t column,
        const Payload& payload, Number unit = Number(1)) const noexcept {
        if (!axis(row) || !axis(column)) return invalid();
        return action(implementation::first_entry(roles(), response(), omega(), coefficients(), row, column),
            {Number(1), 0, true}, payload, unit, Number(1));
    }
    template<class Payload> Evaluation second(std::size_t row, std::size_t a, std::size_t b,
        const Payload& payload, Number unit = Number(1)) const noexcept {
        static_assert(NeedCurvature && Order == 2);
        if (!axis(row) || !axis(a) || !axis(b)) return invalid();
        return action(implementation::second_entry(roles(), response(), omega(), coefficients(), row, a, b),
            {Number(1), 0, true}, payload, unit, Number(1));
    }
    template<class Payload> Evaluation pair(std::size_t row, std::size_t a,
        std::size_t column, std::size_t b, const Payload& payload,
        Number u = Number(1), Number v = Number(1)) const noexcept {
        if (!axis(row) || !axis(a) || !axis(column) || !axis(b)) return invalid();
        return action(implementation::first_entry(roles(), response(), omega(), coefficients(), row, a),
            implementation::first_entry(roles(), response(), omega(), coefficients(), column, b), payload, u, v);
    }
    template<class Input> Evaluation gradient(Input input, std::size_t i) const noexcept {
        if (!axis(i)) return invalid();
        return implementation::contract<false>(roles(), response(), point_->plan().units().exogenous_divisors,
            omega(), coefficients(), input, i);
    }
    template<class Input> Evaluation hessian(Input input, std::size_t i, std::size_t j) const noexcept {
        static_assert(NeedCurvature && Order == 2);
        if (!axis(i) || !axis(j)) return invalid();
        return implementation::contract<true>(roles(), response(), point_->plan().units().exogenous_divisors,
            omega(), coefficients(), input, i, j);
    }
    ContractionAdmission admission(const Evaluation& value, std::size_t operations = 32) const noexcept {
        return implementation::admission(value, coefficients(), operations);
    }
    Status finish(const Evaluation& value, double& out, std::size_t operations = 32) const noexcept {
        if (admission(value, operations) != ContractionAdmission::ready) return Status::numerical_range;
        return implementation::narrow_number(value.value, out);
    }
private:
    LayoutSpec roles() const noexcept { return point_->plan().layout().get().roles; }
    double response() const noexcept { return point_->plan().units().response_divisor; }
    double omega() const noexcept { return point_->derivatives().original_omega; }
    bool axis(std::size_t i) const noexcept { return i < roles().parameters; }
    static Evaluation invalid() noexcept { Evaluation out; out.valid = false; return out; }
    template<class Payload> Evaluation action(implementation::SeedEntry<Number> row,
        implementation::SeedEntry<Number> column, const Payload& payload, Number u, Number v) const noexcept {
        Evaluation out;
        if (!row.active || !column.active) { out.uses_preparation = false; return out; }
        if (!implementation::finite_number(u) || !implementation::finite_number(v) || u <= 0 || v <= 0)
            return invalid();
        std::array<std::array<Number, 1>, 3> terms{};
        std::array<std::array<bool, 1>, 3> exact{};
        terms[row.degree + column.degree][0] = implementation::contraction_product(
            out, payload, row.coefficient, column.coefficient, u, v);
        return implementation::reduce_terms<false>(out, coefficients(), terms, exact);
    }
    const PreparedPoint<Order, Owner, StorageOwner>* point_;
    const implementation::Coefficients<Order, Number>* coefficients_;
};

// Checked once outside traversal; views borrow the point and its worker arena.
template<class Number, bool NeedCurvature, unsigned Order, class Owner, class StorageOwner>
[[nodiscard]] inline std::optional<MapActions<Number, NeedCurvature, Order, Owner, StorageOwner>> map_actions(
    const PreparedPoint<Order, Owner, StorageOwner>& point,
    implementation::PullbackCounts* counts = nullptr) noexcept {
    static_assert(Order >= 1 && (!NeedCurvature || Order == 2));
    const auto n = point.plan().layout().get().roles.parameters;
    if (!point.valid() || !implementation::valid_plan(point.plan()) || point.internal_theta().size() != n
        || point.derivatives().diagonal.size() != n || point.derivatives().quad_diagonal.size() != n
        || point.derivatives().octuple_diagonal.size() != n) return std::nullopt;
    if (counts) {
        implementation::Range accounting;
        std::array<implementation::Range, 7> sources;
        if (!implementation::range(std::span(counts, 1), accounting)
            || !implementation::range(implementation::object(point), sources[0])
            || !implementation::range(point.internal_theta(), sources[1])
            || !implementation::range(point.plan().units().exogenous_divisors, sources[2])
            || !implementation::range(point.derivatives().diagonal, sources[3])
            || !implementation::range(point.derivatives().quad_diagonal, sources[4])
            || !implementation::range(point.derivatives().octuple_diagonal, sources[5])
            || !implementation::range(implementation::object(point.storage()), sources[6])) return std::nullopt;
        for (auto source : sources) if (implementation::overlaps(accounting, source)) return std::nullopt;
    }
    return MapActions<Number, NeedCurvature, Order, Owner, StorageOwner>(point,
        coefficient_view<Number, NeedCurvature>(point, counts));
}

// Prevalidated binary64 payload boundary. Span/alias admission stays with the
// full jet or selected consumer; wide recurrences use MapActions without narrowing.
template<bool Hessian, unsigned Order, class Owner, class Input, class StorageOwner>
[[nodiscard]] inline Status contract_to_double(const PreparedPoint<Order, Owner, StorageOwner>& point,
    Input input, std::size_t i, std::size_t j, double& out,
    implementation::PullbackCounts* counts = nullptr) noexcept {
    static_assert(Input::binary64_inputs && Order >= 1 && (!Hessian || Order == 2));
    const auto& derivatives = point.derivatives();
    const auto s = point.plan().layout().get().roles;
    const double response = point.plan().units().response_divisor;
    const auto divisors = point.plan().units().exogenous_divisors;
    if (!point.valid() || i >= s.parameters || (Hessian && j >= s.parameters)) return Status::invalid_argument;
    const auto coupled = [s, response, &derivatives](std::size_t k) {
        return implementation::coupled_axis(s, response, derivatives.original_omega, k);
    };
    const bool curvature = Hessian
        && implementation::curved_pair(s, response, derivatives.original_omega, i, j);
    const auto ordinary_diagonal = [&derivatives, s](std::size_t k) {
        return static_cast<double>(s.scale == ScaleCoordinate::power && k == s.scale_axis
            ? derivatives.ordinary.factor : derivatives.diagonal[k]);
    };
    const bool uses_power = s.scale == ScaleCoordinate::power
        && (i == s.scale_axis || (Hessian && j == s.scale_axis));
    if (!coupled(i) && (!Hessian || !coupled(j)) && !curvature
        && (!uses_power || derivatives.ordinary.preparation_error <= 8 * std::numeric_limits<double>::epsilon())
        && implementation::ordinary_contract(Hessian ? input.hessian(i, j) : input.gradient(i),
            ordinary_diagonal(i), Hessian ? ordinary_diagonal(j) : 1, out)) {
        if (counts) ++counts->ordinary;
        return Status::ok;
    }
    const auto attempt = [&]<class Number>() {
        const auto& c = coefficient_view<Number, Hessian>(point, counts);
        const MapActions<Number, Hessian, Order, Owner, StorageOwner> actions(point, c);
        const auto e = [&] {
            if constexpr (Hessian) return actions.hessian(input, i, j);
            else return actions.gradient(input, i);
        }();
        const auto result = actions.admission(e);
        if (result == ContractionAdmission::invalid) return -1;
        if (result == ContractionAdmission::insufficient_precision) return 0;
        return implementation::narrow_number(e.value, out) == Status::ok ? 1 : -1;
    };
    if (attempt.template operator()<long double>() == 1) {
        if (counts) ++counts->ordinary;
        return Status::ok;
    }
    const int quad = attempt.template operator()<implementation::Quad>();
    if (quad == 1) { if (counts) ++counts->quad; return Status::ok; }
    // Preserve the finite-binary64 contraction range proof and precision-only
    // Octuple escalation. It does not certify arbitrary recursive state range.
    if (quad < 0) return Status::numerical_range;
    if (attempt.template operator()<implementation::Octuple>() == 1) {
        if (counts) ++counts->octuple;
        return Status::ok;
    }
    return Status::numerical_range;
}

template<unsigned Order, class Owner, class StorageOwner> [[nodiscard]] inline JetValidity pullback(
    const PreparedPoint<Order, Owner, StorageOwner>& point, JetRequest request, InternalJet input, FittingJet output,
    implementation::PullbackCounts* counts = nullptr) noexcept {
    JetValidity result;
    result.requested = request;
    const auto fail = [&result, request](Status s) {
        const auto mask = static_cast<unsigned>(request);
        result.status = s;
        result.f_reason = (mask & 1) ? s : Status::unsupported_operation;
        result.g_reason = (mask & 2) ? s : Status::unsupported_operation;
        result.h_reason = (mask & 4) ? s : Status::unsupported_operation;
        return result;
    };
    if (!implementation::valid_request(request) || !point.valid()
        || !implementation::valid_plan(point.plan()) || implementation::derivative_order(request) > Order)
        return fail(Status::invalid_argument);
    const auto layout = point.plan().layout().get();
    const auto s = layout.roles;
    const auto n = s.parameters;
    const unsigned bits = static_cast<unsigned>(request);
    if (point.internal_theta().size() != n) return fail(Status::invalid_argument);
    if constexpr (Order >= 1)
        if (point.derivatives().diagonal.size() != n || point.derivatives().quad_diagonal.size() != n
            || point.derivatives().octuple_diagonal.size() != n) return fail(Status::invalid_argument);
    if (((bits & 1) && (output.value.empty() || (input.reason(1) == Status::ok && input.value.empty())))
        || ((bits & 2) && (output.gradient.size() < n
            || (input.reason(2) == Status::ok && input.gradient.size() < n)))
        || ((bits & 4) && (output.hessian.size() < layout.hessian_elements
            || (input.reason(4) == Status::ok && input.hessian.size() < layout.hessian_elements)))
        || ((static_cast<unsigned>(internal_request(point.plan(), request)) & 2)
            && input.reason(2) == Status::ok && input.gradient.size() < n))
        return fail(Status::insufficient_capacity);
    // Admit every supplied output, including unrequested storage: an alias must
    // not let a requested output overwrite an unrequested one or an input.
    std::array<implementation::Range, 3> destinations;
    std::array<implementation::Range, 10> sources;
    if (!implementation::range(output.value, destinations[0])
        || !implementation::range(output.gradient, destinations[1])
        || !implementation::range(output.hessian, destinations[2])
        || !implementation::disjoint(destinations)
        || !implementation::range(input.value, sources[0])
        || !implementation::range(input.gradient, sources[1])
        || !implementation::range(input.hessian, sources[2])
        || !implementation::range(point.internal_theta(), sources[3])
        || !implementation::range(point.plan().units().exogenous_divisors, sources[4])
        || !implementation::range(implementation::object(point), sources[5])
        || !implementation::range(implementation::object(point.storage()), sources[9])) return fail(Status::invalid_argument);
    sources[6] = {}; sources[7] = {}; sources[8] = {};
    if constexpr (Order >= 1)
        if (!implementation::range(point.derivatives().diagonal, sources[6])
            || !implementation::range(point.derivatives().quad_diagonal, sources[7])
            || !implementation::range(point.derivatives().octuple_diagonal, sources[8])) return fail(Status::invalid_argument);
    for (auto a : destinations) for (auto b : sources)
        if (implementation::overlaps(a, b)) return fail(Status::invalid_argument);
    if (counts) {
        implementation::Range accounting;
        if (!implementation::range(std::span(counts, 1), accounting)) return fail(Status::invalid_argument);
        for (auto a : destinations) if (implementation::overlaps(a, accounting)) return fail(Status::invalid_argument);
        for (auto b : sources) if (implementation::overlaps(b, accounting)) return fail(Status::invalid_argument);
    }
    result.status = Status::ok;
    const auto record = [&result](unsigned bit, Status reason) {
        if (bit == 1) result.f_reason = reason;
        if (bit == 2) result.g_reason = reason;
        if (bit == 4) result.h_reason = reason;
        if (reason == Status::ok) result.valid_mask |= bit;
        else if (result.status == Status::ok) result.status = reason;
    };
    if (bits & 1) {
        Status reason = input.reason(1);
        if (reason == Status::ok)
            reason = !std::isfinite(input.value[0]) ? Status::numerical_range
                : restore_likelihood(point.plan(), input.value[0], output.value[0]);
        record(1, reason);
    }
    if constexpr (Order >= 1) {
        const auto payload = contraction_input(
            [input](std::size_t i) { return input.gradient[i]; },
            [input, n](std::size_t i, std::size_t j) { return input.hessian[i * n + j]; });
        const auto evaluate = [&]<bool Hessian>(std::size_t i, std::size_t j, double& out) -> Status {
            return contract_to_double<Hessian>(point, payload, i, j, out, counts);
        };
        if (bits & 2) {
            Status reason = input.reason(2);
            for (std::size_t i = 0; reason == Status::ok && i < n; ++i)
                reason = evaluate.template operator()<false>(i, 0, output.gradient[i]);
            record(2, reason);
        }
        if constexpr (Order == 2) {
            if (bits & 4) {
                const bool curved = s.scale == ScaleCoordinate::power && point.plan().units().response_divisor != 1;
                Status reason = input.reason(4);
                if (reason == Status::ok && curved) reason = input.reason(2);
                for (std::size_t i = 0; reason == Status::ok && i < n; ++i)
                    for (std::size_t j = 0; reason == Status::ok && j < n; ++j)
                        reason = evaluate.template operator()<true>(i, j, output.hessian[i*n+j]);
                record(4, reason);
            }
        }
    }
    return result;
}

template<class Owner> [[nodiscard]] inline Status restore_theta(const Plan<Owner>& p,
    std::span<const double> internal, std::span<double> fitting) noexcept {
    if (!implementation::valid_plan(p) || internal.size() != p.layout().get().roles.parameters)
        return Status::invalid_argument;
    if (fitting.size() < internal.size()) return Status::insufficient_capacity;
    std::array<implementation::Range, 3> inputs;
    implementation::Range output;
    if (!implementation::range(internal, inputs[0])
        || !implementation::range(p.units().exogenous_divisors, inputs[1])
        || !implementation::range(implementation::object(p), inputs[2])
        || !implementation::range(fitting.first(internal.size()), output)) return Status::invalid_argument;
    for (auto r : inputs) if (implementation::overlaps(r, output)) return Status::invalid_argument;
    if (p.layout().get().roles.scale == ScaleCoordinate::power && internal[p.layout().get().roles.delta] <= 0)
        return Status::mathematical_domain;
    for (std::size_t i = 0; i < internal.size(); ++i) {
        const auto status = implementation::theta_value(p, internal, i, Direction::to_fitting, fitting[i]);
        if (status != Status::ok) return status;
    }
    return Status::ok;
}

// Point-aware theta restoration protects derivative caches and all additional
// finalizer destinations, including unrequested G/H storage.
template<unsigned Order, class Owner, class StorageOwner> [[nodiscard]] inline Status restore_theta(
    const PreparedPoint<Order, Owner, StorageOwner>& point, std::span<double> fitting,
    FittingJet other_outputs = {}) noexcept {
    if (!point.valid()) return Status::invalid_argument;
    std::array<implementation::Range, 4> writes;
    std::array<implementation::Range, 7> reads;
    if (!implementation::range(fitting, writes[0])
        || !implementation::range(other_outputs.value, writes[1])
        || !implementation::range(other_outputs.gradient, writes[2])
        || !implementation::range(other_outputs.hessian, writes[3])
        || !implementation::disjoint(writes)
        || !implementation::range(point.internal_theta(), reads[0])
        || !implementation::range(point.plan().units().exogenous_divisors, reads[1])
        || !implementation::range(implementation::object(point), reads[2])
        || !implementation::range(implementation::object(point.storage()), reads[6])) return Status::invalid_argument;
    reads[3] = {}; reads[4] = {}; reads[5] = {};
    if constexpr (Order > 0)
        if (!implementation::range(point.derivatives().diagonal, reads[3])
            || !implementation::range(point.derivatives().quad_diagonal, reads[4])
            || !implementation::range(point.derivatives().octuple_diagonal, reads[5])) return Status::invalid_argument;
    for (auto w : writes) for (auto r : reads)
        if (implementation::overlaps(w, r)) return Status::invalid_argument;
    return restore_theta(point.plan(), point.internal_theta(), fitting);
}

template<class Owner> [[nodiscard]] inline Status admit_scalar(const Plan<Owner>& p, double& out) noexcept {
    implementation::Range destination, plan, divisors;
    if (!implementation::range(implementation::object(out), destination)
        || !implementation::range(implementation::object(p), plan)
        || !implementation::range(p.units().exogenous_divisors, divisors)
        || implementation::overlaps(destination, plan)
        || implementation::overlaps(destination, divisors)) return Status::invalid_argument;
    return Status::ok;
}

// Scalar history/path intentions are shared with sample preparation. Supplied
// raw histories first receive s or s^2 outside this owner. No seed policy here.
template<class Owner> [[nodiscard]] inline Status convert_state(const Plan<Owner>& p, double value,
    StateCoordinate coordinate, Direction direction, double& out, double delta = 2) noexcept {
    if (!implementation::valid_plan(p) || !std::isfinite(value)) return Status::invalid_argument;
    if (admit_scalar(p, out) != Status::ok) return Status::invalid_argument;
    const double r = p.units().response_divisor;
    switch (coordinate) {
    case StateCoordinate::response:
        return direction == Direction::to_internal ? implementation::scaled(value, 1, r, out)
                                                   : implementation::scaled(value, r, 1, out);
    case StateCoordinate::variance: return implementation::squared(value, r, direction, out);
    case StateCoordinate::log_variance:
        if (r == 1) { out = value; return Status::ok; }
        return implementation::narrow(static_cast<long double>(value)
            + (direction == Direction::to_internal ? -2 : 2) * p.log_response(), out);
    case StateCoordinate::power:
        if (!std::isfinite(delta)) return Status::invalid_argument;
        if (delta <= 0) return Status::mathematical_domain;
        if (value == 0 || r == 1) { out = value; return Status::ok; }
        {
            const long double exponent = (direction == Direction::to_internal ? -1 : 1)
                * static_cast<long double>(delta) * p.log_response();
            return implementation::power_value(value, {exponent, std::exp(exponent)}, out);
        }
    }
    return Status::invalid_argument;
}
template<class Owner> [[nodiscard]] inline Status convert_exogenous(const Plan<Owner>& p,
    std::size_t column, double value, Direction direction, double& out) noexcept {
    if (!implementation::valid_plan(p) || column >= p.units().exogenous_divisors.size()
        || !std::isfinite(value)) return Status::invalid_argument;
    if (admit_scalar(p, out) != Status::ok) return Status::invalid_argument;
    const double d = p.units().exogenous_divisors[column];
    return direction == Direction::to_internal ? implementation::scaled(value, 1, d, out)
                                               : implementation::scaled(value, d, 1, out);
}

} // namespace scivolapi::detail::unit_map
