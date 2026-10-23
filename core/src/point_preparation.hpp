#pragma once

#include <scivolapi/jet.hpp>
#include "preparation_banks.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

// Standalone identity wrapper used by metadata tests. Production shares Bank
// through DensityPreparation and keeps identity in WorkspaceState.
namespace scivolapi::detail::point_preparation {

enum class Component : std::uint32_t {
    density = 1, value_constant = 2, map = 4, domain = 8, moments = 16, profile = 32
};
constexpr Components bit(Component component) noexcept { return static_cast<Components>(component); }
struct Need {
    Components components = 0;
    std::uint64_t numerical_plan_id = 0;
};
struct Key {
    ObjectiveKey objective{};
    LayoutId full_theta_layout = 0;
    std::uint64_t point_id = 0;
    bool operator==(const Key&) const = default;
};
struct PointInput { std::span<const double> theta, original_z; };
struct PointWidths { std::size_t theta = 0, original_z = 0; };
struct StorageRequirements {
    std::size_t bytes = 0, alignment = alignof(double);
    bool operator==(const StorageRequirements&) const = default;
};
struct State {
    Components ready = 0;
    std::uint64_t numerical_plan_id = 0;
    Need last_attempt{};
    Status last_status = Status::unsupported_operation;
};

class Plan {
public:
    Status status() const noexcept { return status_; }
    std::size_t persistent_bytes() const noexcept { return persistent_bytes_; }
    std::size_t persistent_alignment() const noexcept { return persistent_alignment_; }
    std::size_t staging_bytes() const noexcept { return staging_.bytes; }
    std::size_t staging_alignment() const noexcept { return staging_.alignment; }
private:
    template<class, class, class> friend class Cache;
    Status status_ = Status::invalid_argument;
    PointWidths widths_{};
    Coordinates coordinates_ = Coordinates::theta;
    JetRequest maximum_ = JetRequest::F;
    std::array<StorageRequirements, 3> data_{};
    StorageRequirements staging_{};
    std::size_t persistent_bytes_ = 0, persistent_alignment_ = alignof(double);
    storage::Region theta_, z_;
    std::array<std::array<storage::BankLayout, 2>, 3> banks_{};
    std::array<bool, 3> enabled_{};
};

template<class Payload> struct BuildContext {
    const Key& key;
    PointInput point;
    Need requested;
    Components required;
    const Payload* previous;
    State previous_state;
    Arena& staging;
};

// Each order has two final-address banks. A commit switches banks, never moves a
// payload. The caller keeps both arenas alive through destruction of this cache.
template<class Value, class First, class Second> class Cache {
    template<unsigned Order> using Payload = std::tuple_element_t<Order, std::tuple<Value, First, Second>>;
    struct Slot { State state; Bank bank; };
    struct Preparing {
        bool& active;
        explicit Preparing(bool& value) noexcept : active(value) { active = true; }
        ~Preparing() { active = false; }
    };
    static_assert(std::is_nothrow_destructible_v<Value> && std::is_nothrow_destructible_v<First>
        && std::is_nothrow_destructible_v<Second>);
public:
    Cache() noexcept = default;
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    Cache(Cache&&) = delete;
    Cache& operator=(Cache&&) = delete;
    ~Cache() { clear_point(); }

    static Plan requirements(PointWidths widths, Coordinates coordinates, JetRequest maximum,
        std::array<StorageRequirements, 3> data = {}, StorageRequirements staging = {}) noexcept {
        Plan plan;
        const auto mask = static_cast<unsigned>(maximum);
        if (!mask || mask > 7 || !storage::alignment(staging.alignment)
            || staging.bytes > max_span_bytes
            || (coordinates != Coordinates::theta && coordinates != Coordinates::original_chart)
            || (coordinates == Coordinates::theta && widths.original_z)
            || (coordinates == Coordinates::original_chart && widths.theta && !widths.original_z)) return plan;
        for (auto requirement : data)
            if (!storage::alignment(requirement.alignment)) return plan;
        plan.widths_ = widths; plan.coordinates_ = coordinates; plan.maximum_ = maximum;
        plan.data_ = data; plan.staging_ = staging;
        plan.enabled_ = {true, bool(mask & 2), bool(mask & 4)};
        storage::LayoutBuilder layout;
        std::size_t theta_bytes, z_bytes;
        if (!storage::native::checked_product(widths.theta, sizeof(double), theta_bytes)
            || !storage::native::checked_product(widths.original_z, sizeof(double), z_bytes)
            || !layout.append(theta_bytes, alignof(double), plan.theta_)
            || !layout.append(z_bytes, alignof(double), plan.z_)
            || !append_banks<0>(plan, layout) || !append_banks<1>(plan, layout)
            || !append_banks<2>(plan, layout) || layout.bytes() > max_span_bytes) return Plan{};
        plan.persistent_bytes_ = layout.bytes();
        plan.status_ = Status::ok;
        return plan;
    }

    Status bind(const Plan& plan, std::span<std::byte> persistent, std::span<std::byte> staging) noexcept {
        if (preparing_ || plan.status_ != Status::ok) return Status::invalid_argument;
        const auto expected = requirements(plan.widths_, plan.coordinates_, plan.maximum_, plan.data_, plan.staging_);
        if (expected.status_ != Status::ok || expected.banks_ != plan.banks_
            || expected.persistent_bytes_ != plan.persistent_bytes_
            || expected.persistent_alignment_ != plan.persistent_alignment_) return Status::invalid_argument;
        if (persistent.size() < plan.persistent_bytes_ || staging.size() < plan.staging_.bytes)
            return Status::insufficient_capacity;
        storage::Range p, s, metadata;
        if (!storage::range(persistent, p) || !storage::range(staging, s)
            || !storage::range(this, sizeof(*this), metadata)
            || (p.begin && p.begin % plan.persistent_alignment_)
            || (s.begin && s.begin % plan.staging_.alignment)
            || storage::overlaps(p, s) || storage::overlaps(p, metadata)
            || storage::overlaps(s, metadata)) return Status::invalid_argument;
        clear_point();
        plan_ = plan;
        persistent_ = persistent.first(plan.persistent_bytes_);
        staging_ = staging.first(plan.staging_.bytes);
        theta_ = construct_snapshot(plan.theta_, plan.widths_.theta);
        z_ = construct_snapshot(plan.z_, plan.widths_.original_z);
        if (bind_bank<0>()!=Status::ok || bind_bank<1>()!=Status::ok || bind_bank<2>()!=Status::ok) {
            for (auto& slot:slots_) slot.bank.release();
            bound_=false;return Status::invalid_argument;
        }
        bound_ = true;
        return Status::ok;
    }

    // Admits representation and identity only; does not certify map/domain math.
    Status bind_point(Key key, PointInput input) noexcept {
        if (preparing_ || !bound_ || !key.point_id || !key.full_theta_layout
            || !key.objective.problem || !key.objective.derivative_layout
            || (key.objective.chart != 0) != (plan_.coordinates_ == Coordinates::original_chart)
            || input.theta.size() != theta_.size() || input.original_z.size() != z_.size()
            || !admit_input(input.theta, theta_) || !admit_input(input.original_z, z_))
            return Status::invalid_argument;
        if (has_point_ && key_ == key && identical(theta_, input.theta) && identical(z_, input.original_z))
            return Status::ok;
        clear_point();
        if (!theta_.empty()) std::memmove(theta_.data(), input.theta.data(), input.theta.size_bytes());
        if (!z_.empty()) std::memmove(z_.data(), input.original_z.data(), input.original_z.size_bytes());
        key_ = key;
        has_point_ = true;
        return Status::ok;
    }

    Status clear_point() noexcept {
        if (preparing_) return Status::invalid_argument;
        for (auto& slot:slots_) {slot.bank.clear();slot.state={};}
        key_ = {};
        has_point_ = false;
        return Status::ok;
    }
    bool has_point() const noexcept { return has_point_; }
    const Key* key() const noexcept { return has_point_ ? &key_ : nullptr; }
    PointInput point() const noexcept { return has_point_ ? PointInput{theta_, z_} : PointInput{}; }
    template<unsigned Order> const State& state() const noexcept {
        static_assert(Order < 3);
        return slots_[Order].state;
    }
    template<unsigned Order> const Payload<Order>* get(Need need) const noexcept {
        static_assert(Order < 3);
        const auto& slot = slots_[Order];
        return has_point_ && valid_components(need.components) && slot.bank.template get<Payload<Order>>()
            && slot.state.numerical_plan_id == need.numerical_plan_id
            && (slot.state.ready & need.components) == need.components ? active_payload<Order>() : nullptr;
    }

    // Producer must retain no staging/previous-payload pointers. Candidate data,
    // cache point snapshots and immutable problem leases are stable dependencies.
    // A successful same-plan upgrade must reconstruct every already-ready component.
    template<unsigned Order, class Builder> Status ensure(Need need, Builder&& builder) noexcept {
        static_assert(Order < 3);
        using T = Payload<Order>;
        static_assert(std::is_nothrow_invocable_r_v<Status, Builder&, Candidate<T>&, const BuildContext<T>&>);
        if (preparing_ || !bound_ || !has_point_ || !valid_components(need.components)) return Status::invalid_argument;
        if (!plan_.enabled_[Order]) return Status::insufficient_capacity;
        if (get<Order>(need)) return Status::ok;
        Preparing activity(preparing_);
        auto& slot = slots_[Order];
        const auto previous = slot.state;
        const auto* previous_payload=active_payload<Order>();
        const auto required = need.components | (previous_payload
            && previous.numerical_plan_id == need.numerical_plan_id ? previous.ready : 0u);
        Components committed_ready=0;
        const auto status=slot.bank.template replace<T>([&](Candidate<T>& candidate) noexcept {
            Arena scratch(staging_);
            const BuildContext<T> context{key_, point(), need, required, previous_payload, previous, scratch};
            auto result=builder(candidate,context);
            if (result==Status::ok && (!candidate.payload_ || (candidate.ready_&required)!=required))
                result=Status::internal_error;
            committed_ready=candidate.ready_;
            return result;
        });
        slot.state.last_attempt = need;
        slot.state.last_status = status;
        if (status != Status::ok) return status;
        slot.state.ready = committed_ready;
        slot.state.numerical_plan_id = need.numerical_plan_id;
        return Status::ok;
    }

private:
    static constexpr auto max_span_bytes = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    template<unsigned Order> static bool append_banks(Plan& plan, storage::LayoutBuilder& layout) noexcept {
        if (!plan.enabled_[Order]) return true;
        using T = Payload<Order>;
        const auto data = plan.data_[Order];
        plan.persistent_alignment_ = std::max({plan.persistent_alignment_, alignof(T), data.alignment});
        return storage::append_banks<T>(layout,plan.banks_[Order],data.bytes,data.alignment);
    }
    template<unsigned Order> Status bind_bank() noexcept {
        auto& bank=slots_[Order].bank;
        return plan_.enabled_[Order]
            ? bank.template bind<Payload<Order>>(persistent_,plan_.banks_[Order]):bank.release();
    }
    std::span<double> construct_snapshot(storage::Region r, std::size_t count) noexcept {
        if (!count) return {};
        auto* values = reinterpret_cast<double*>(persistent_.data() + r.offset);
        for (std::size_t i = 0; i < count; ++i) std::construct_at(values + i);
        return {values, count};
    }
    static bool identical(std::span<const double> a, std::span<const double> b) noexcept {
        return a.empty() || std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
    }
    bool admit_input(std::span<const double> values, std::span<const double> snapshot) const noexcept {
        storage::Range input, persistent, staging, metadata;
        if (!storage::range(values, input) || (input.begin && input.begin % alignof(double))
            || !storage::range(persistent_, persistent) || !storage::range(staging_, staging)
            || !storage::range(this, sizeof(*this), metadata)
            || storage::overlaps(input, staging) || storage::overlaps(input, metadata)
            || (storage::overlaps(input, persistent) && (values.data() != snapshot.data()
                || values.size() != snapshot.size()))) return false;
        return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
    }
    template<unsigned Order> Payload<Order>* active_payload() const noexcept {
        return slots_[Order].bank.template get<Payload<Order>>();
    }
    Plan plan_;
    std::span<std::byte> persistent_, staging_;
    std::span<double> theta_, z_;
    std::array<Slot, 3> slots_{};
    Key key_{};
    bool bound_ = false, has_point_ = false, preparing_ = false;
};

} // namespace scivolapi::detail::point_preparation
