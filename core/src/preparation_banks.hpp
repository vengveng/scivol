#pragma once

#include <scivolapi/detail/types.hpp>
#include "math/runtime_workspace.hpp"
#include <array>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace scivolapi::detail::point_preparation {
using Components = std::uint32_t;
inline constexpr Components all_components = 63;
constexpr bool valid_components(Components components) noexcept {
    return components && !(components & ~all_components);
}
namespace storage {
namespace native = vlk::experiment::runtime;
struct Range { std::uintptr_t begin = 0, end = 0; };
inline bool range(const void* pointer, std::size_t bytes, Range& result) noexcept {
    result = {};
    if (!bytes) return true;
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (!pointer || bytes > std::numeric_limits<std::uintptr_t>::max() - address) return false;
    result = {address, address + bytes};
    return true;
}
template<class T> bool range(std::span<T> values, Range& result) noexcept {
    std::size_t bytes;
    return native::checked_product(values.size(), sizeof(T), bytes)
        && range(values.data(), bytes, result);
}
inline bool overlaps(Range a, Range b) noexcept { return a.begin < b.end && b.begin < a.end; }
inline bool alignment(std::size_t value) noexcept { return value && !(value & (value - 1)); }
struct Region {
    std::size_t offset = 0, bytes = 0;
    bool operator==(const Region&) const = default;
};
struct BankLayout {
    Region object, data;
    bool operator==(const BankLayout&) const = default;
};
class LayoutBuilder {
public:
    bool append(std::size_t bytes, std::size_t align, Region& result) noexcept {
        if (!alignment(align)) return false;
        const auto padding = (align - size_.size() % align) % align;
        if (!size_.add<std::byte>(padding)) return false;
        result = {size_.size(), bytes};
        return size_.add<std::byte>(bytes);
    }
    std::size_t bytes() const noexcept { return size_.size(); }
private:
    native::WorkspaceSizer size_;
};
template<class Payload>
bool append_banks(LayoutBuilder& layout, std::array<BankLayout,2>& banks,
    std::size_t data_bytes=0,std::size_t data_alignment=alignof(double)) noexcept {
    for (auto& bank:banks)
        if (!layout.append(sizeof(Payload),alignof(Payload),bank.object)
            || !layout.append(data_bytes,data_alignment,bank.data)) return false;
    return true;
}
} // namespace storage

// Arrays obtained here have started lifetimes. Their destructors must be trivial.
class Arena {
public:
    explicit Arena(std::span<std::byte> memory) noexcept : arena_(memory.data(), memory.size()) {}
    template<class T> std::span<T> take(std::size_t count) noexcept {
        static_assert(std::is_nothrow_default_constructible_v<T> && std::is_trivially_destructible_v<T>);
        if (!count) return {};
        auto* values = arena_.take<T>(count);
        if (!values) return {};
        for (std::size_t i = 0; i < count; ++i) std::construct_at(values + i);
        return {values, count};
    }
private:
    storage::native::WorkspaceArena arena_;
};

class Bank;
template<class Value, class First, class Second> class Cache;
template<class Payload> class Candidate {
public:
    Candidate(const Candidate&) = delete;
    Candidate& operator=(const Candidate&) = delete;
    ~Candidate() { if (payload_) std::destroy_at(payload_); }
    template<class... Args> Payload* emplace(Args&&... args) noexcept {
        static_assert(std::is_nothrow_constructible_v<Payload, Args...>);
        if (payload_) return nullptr;
        payload_ = std::construct_at(reinterpret_cast<Payload*>(object_), std::forward<Args>(args)...);
        return payload_;
    }
    template<class Factory> Payload* emplace_from(Factory&& factory) noexcept {
        static_assert(std::is_same_v<std::invoke_result_t<Factory&>,Payload>);
        if (payload_) return nullptr;
        // Guaranteed elision permits nonmovable, self-referencing math payloads.
        payload_ = ::new (object_) Payload(factory());
        return payload_;
    }
    Arena& data() noexcept { return data_; }
    bool mark_ready(Components components) noexcept {
        if (!valid_components(components)) return false;
        ready_ |= components;
        return true;
    }
private:
    friend class Bank;
    template<class, class, class> friend class Cache;
    Candidate(std::byte* object, std::span<std::byte> data) noexcept : object_(object), data_(data) {}
    Payload* release() noexcept { return std::exchange(payload_, nullptr); }
    std::byte* object_;
    Arena data_;
    Payload* payload_ = nullptr;
    Components ready_ = 0;
};

// Identity-free final-address storage. The checked owner selects one Payload type
// per binding and must use that same type for get/replace until release/rebind.
// Candidate data may be retained; staging and previous-payload pointers may not.
class Bank {
    std::array<std::span<std::byte>,2> objects_{}, data_{};
    void (*destroy_)(std::byte*) noexcept = nullptr;
    int active_ = -1;
    bool replacing_ = false;
    void destroy_active() noexcept {
        if (active_>=0) destroy_(objects_[active_].data());
        active_=-1;
    }
public:
    Bank() noexcept=default;
    Bank(const Bank&)=delete;
    Bank& operator=(const Bank&)=delete;
    ~Bank() { destroy_active(); }
    template<class Payload>
    Status bind(std::span<std::byte> memory,const std::array<storage::BankLayout,2>& layout) noexcept {
        static_assert(std::is_nothrow_destructible_v<Payload>);
        if (replacing_) return Status::invalid_argument;
        std::array<std::span<std::byte>,2> objects{},data{};
        std::array<storage::Range,4> ranges{};
        storage::Range metadata, arena;
        if (!storage::range(this,sizeof(*this),metadata) || !storage::range(memory,arena)) return Status::invalid_argument;
        for (unsigned i=0;i<2;++i) {
            const auto b=layout[i];
            if (b.object.offset>memory.size() || b.object.bytes>memory.size()-b.object.offset
                || b.data.offset>memory.size() || b.data.bytes>memory.size()-b.data.offset
                || b.object.bytes!=sizeof(Payload)) return Status::insufficient_capacity;
            objects[i]=memory.subspan(b.object.offset,b.object.bytes);
            data[i]=memory.subspan(b.data.offset,b.data.bytes);
            if (!storage::range(objects[i],ranges[2*i]) || !storage::range(data[i],ranges[2*i+1])
                || ranges[2*i].begin%alignof(Payload)) return Status::invalid_argument;
        }
        for (unsigned i=0;i<4;++i) {
            if (storage::overlaps(ranges[i],metadata)) return Status::invalid_argument;
            for (unsigned j=0;j<i;++j)
                if (storage::overlaps(ranges[i],ranges[j])) return Status::invalid_argument;
        }
        destroy_active();objects_=objects;data_=data;
        destroy_=[](std::byte* p) noexcept {std::destroy_at(std::launder(reinterpret_cast<Payload*>(p)));};
        return Status::ok;
    }
    bool bound() const noexcept { return destroy_!=nullptr; }
    template<class Payload> Payload* get() const noexcept {
        return active_<0 ? nullptr:std::launder(reinterpret_cast<Payload*>(objects_[active_].data()));
    }
    Status clear() noexcept {
        if (replacing_) return Status::invalid_argument;
        destroy_active();return Status::ok;
    }
    Status release() noexcept {
        if (replacing_) return Status::invalid_argument;
        destroy_active();objects_={};data_={};destroy_=nullptr;return Status::ok;
    }
    template<class Payload,class Builder> Status replace(Builder&& builder) noexcept {
        static_assert(std::is_nothrow_invocable_r_v<Status,Builder&,Candidate<Payload>&>);
        if (!bound() || replacing_) return Status::invalid_argument;
        const int target=active_<0 ? 0:1-active_;
        replacing_=true;
        struct Restore {bool& busy;~Restore() {busy=false;}} restore{replacing_};
        Candidate<Payload> candidate(objects_[target].data(),data_[target]);
        auto status=builder(candidate);
        if (status==Status::ok && !candidate.payload_) status=Status::internal_error;
        if (status==Status::ok) {
            destroy_active();candidate.release();active_=target;
        }
        return status;
    }
};
} // namespace scivolapi::detail::point_preparation
