#pragma once
#include "../preparation_banks.hpp"
#include "../math/local_density_axes.hpp"
#include <array>
#include <cstddef>
#include <type_traits>

namespace scivolapi::detail {
enum class MomentProducer : unsigned {negative_second,absolute,wide_absolute,power_batch,precision,wide_absolute_value,count};
inline constexpr unsigned moment_producers=unsigned(MomentProducer::count);
struct MomentReservation {std::size_t object=0,data=0,alignment=alignof(double);};
using MomentReservations=std::array<std::array<MomentReservation,moment_producers>,3>;
using MomentLayouts=std::array<std::array<std::array<point_preparation::storage::BankLayout,2>,moment_producers>,3>;

// The token identifies a C++ payload type, never a parameter point. WorkspaceState
// alone owns point admission and calls clear when that point changes.
class MomentPreparation {
    struct Slot {
        point_preparation::Bank bank;
        std::array<point_preparation::storage::BankLayout,2> layout{};
        MomentReservation capacity{};
        const void* type=nullptr;
        unsigned shapes=0;
    };
    std::array<std::array<Slot,moment_producers>,3> slots_{};
    std::span<std::byte> memory_{};
    template<class Payload> static inline const char payload_type=0;
public:
    std::array<std::array<std::size_t,moment_producers>,3>* builds=nullptr;
    MomentPreparation() noexcept=default;
    MomentPreparation(const MomentPreparation&)=delete;
    MomentPreparation& operator=(const MomentPreparation&)=delete;
    static bool append(point_preparation::storage::LayoutBuilder& size,
        const MomentReservations& reservations,MomentLayouts& layouts) noexcept {
        for(unsigned order=0;order<3;++order) for(unsigned kind=0;kind<moment_producers;++kind) {
            const auto r=reservations[order][kind];
            if (!r.object) continue;
            for(auto& bank:layouts[order][kind])
                if (!size.append(r.object,r.alignment,bank.object)
                    || !size.append(r.data,alignof(std::max_align_t),bank.data)) return false;
        }
        return true;
    }
    Status bind(std::span<std::byte> memory,const MomentReservations& reservations,
        const MomentLayouts& layouts) noexcept {
        point_preparation::storage::Range arena,metadata;
        if (!point_preparation::storage::range(memory,arena)
            || !point_preparation::storage::range(this,sizeof(*this),metadata)
            || point_preparation::storage::overlaps(arena,metadata)) return Status::invalid_argument;
        std::array<point_preparation::storage::Range,12*moment_producers> ranges{};
        std::size_t used=0;
        for(unsigned order=0;order<3;++order) for(unsigned kind=0;kind<moment_producers;++kind) {
            const auto r=reservations[order][kind];
            for(const auto& bank:layouts[order][kind]) {
                if (bank.object.offset>memory.size() || bank.object.bytes>memory.size()-bank.object.offset
                    || bank.data.offset>memory.size() || bank.data.bytes>memory.size()-bank.data.offset)
                    return Status::insufficient_capacity;
                if (r.object && (!point_preparation::storage::alignment(r.alignment)
                    || reinterpret_cast<std::uintptr_t>(memory.data()+bank.object.offset)%r.alignment))
                    return Status::invalid_argument;
                if (!r.object) continue;
                for(const auto region:{bank.object,bank.data}) {
                    auto& range=ranges[used++];
                    if (!point_preparation::storage::range(memory.data()+region.offset,region.bytes,range))
                        return Status::invalid_argument;
                    for(std::size_t i=0;i+1<used;++i)
                        if (point_preparation::storage::overlaps(range,ranges[i])) return Status::invalid_argument;
                }
            }
        }
        release();memory_=memory;
        for(unsigned order=0;order<3;++order) for(unsigned kind=0;kind<moment_producers;++kind) {
            slots_[order][kind].capacity=reservations[order][kind];
            slots_[order][kind].layout=layouts[order][kind];
        }
        return Status::ok;
    }
    void clear() noexcept {for(auto& order:slots_) for(auto& slot:order) slot.bank.clear();}
    void release() noexcept {
        for(auto& order:slots_) for(auto& slot:order) {
            slot.bank.release();slot.type=nullptr;slot.capacity={};slot.layout={};slot.shapes=0;
        }
        memory_={};
    }
    template<unsigned Order,MomentProducer Kind,class Payload,class Builder>
    Payload* get(vlk::experiment::generated::LocalDensityAxes axes,Builder&& build) noexcept {
        static_assert(Order<=2 && unsigned(Kind)<moment_producers);
        auto& slot=slots_[Order][unsigned(Kind)];
        if (sizeof(Payload)>slot.capacity.object || alignof(Payload)>slot.capacity.alignment
            || memory_.empty() || (slot.type && slot.type!=&payload_type<Payload>)) return nullptr;
        if (!slot.type) {
            auto layout=slot.layout;
            for(auto& bank:layout) bank.object.bytes=sizeof(Payload);
            if (slot.bank.template bind<Payload>(memory_,layout)!=Status::ok) return nullptr;
            slot.type=&payload_type<Payload>;
        }
        if (auto* value=slot.bank.template get<Payload>())
            if (slot.shapes==axes.shapes) return value;
        if (builds) ++(*builds)[Order][unsigned(Kind)];
        if (slot.bank.template replace<Payload>(std::forward<Builder>(build))!=Status::ok) return nullptr;
        slot.shapes=axes.shapes;
        return slot.bank.template get<Payload>();
    }
};
} // namespace scivolapi::detail
