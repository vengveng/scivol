// SCIVOL-NATIVE: integration candidate; exact request reuse, no solver policy.
#pragma once
#include "runtime_workspace.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace vlk::experiment::request_reuse {
namespace work = vlk::experiment::runtime;
struct Values { const double *value, *gradient, *hessian; };
struct Outputs { double *value, *gradient, *hessian; };
constexpr std::size_t ways=8;
struct Entry { std::uint64_t hash=0, age=0; std::uint32_t mask=0, coordinate=0; };
struct Layout {
    std::size_t width=0, square=0, outputs=0, stride=0, buckets=0, entries=0, bytes=0;
    std::uint32_t supported=7u;
};
constexpr Layout layout(std::size_t width,std::size_t buckets,bool curvature=true) noexcept {
    Layout l; l.width=width; l.buckets=buckets;
    l.supported=curvature?7u:3u;
    std::size_t storage=0;
    if (!width || !buckets || (buckets&(buckets-1u))
        || !work::checked_product(width,width,l.square)
        || !work::checked_sum(curvature?l.square:0u,width,l.outputs)
        || !work::checked_sum(l.outputs,1u,l.outputs)
        || !work::checked_sum(l.outputs,width,l.stride)
        || !work::checked_product(buckets,ways,l.entries)
        || !work::checked_product(l.entries,l.stride,storage)) return {};
    work::WorkspaceSizer s;
    s.add<Entry>(l.entries); s.add<double>(storage);
    l.bytes=s.size(); return l;
}

// Binding defines one immutable model/sample/request-plan lifetime. Rebinding
// clears all entries; the caller may not reuse a binding after changing inputs.
class Cache {
    Layout l_;
    Entry* entries_=nullptr;
    double* storage_=nullptr;
    std::uint64_t clock_=0;
    std::uint64_t key(const double* point,std::uint32_t mask,std::uint32_t coordinate) const noexcept {
        std::uint64_t h=14695981039346656037ull ^ (std::uint64_t(mask)<<32u) ^ coordinate;
        for (std::size_t i=0;i<l_.width;++i) {
            h^=std::bit_cast<std::uint64_t>(point[i]); h*=1099511628211ull;
            h^=h>>32u;
        }
        return h;
    }
    void tick() noexcept {
        if (++clock_==0) { std::fill_n(entries_,l_.entries,Entry{}); clock_=1; }
    }
    double* point(std::size_t entry) const noexcept { return storage_+entry*l_.stride; }
    void restore(Outputs out,const double* source,std::uint32_t mask) const noexcept {
        if (mask&1u) std::memcpy(out.value,source,sizeof(double));
        if (mask&2u) std::memcpy(out.gradient,source+1u,l_.width*sizeof(double));
        if (mask&4u) std::memcpy(out.hessian,source+1u+l_.width,l_.square*sizeof(double));
    }
    void store(double* target,Values out,std::uint32_t mask) const noexcept {
        if (mask&1u) std::memcpy(target,out.value,sizeof(double));
        if (mask&2u) std::memcpy(target+1u,out.gradient,l_.width*sizeof(double));
        if (mask&4u) std::memcpy(target+1u+l_.width,out.hessian,l_.square*sizeof(double));
    }
public:
    bool get(const double* input,std::uint32_t mask,std::uint32_t coordinate,double* packed) noexcept {
        if (!storage_ || !packed) return false;
        return get(input,mask,coordinate,Outputs{packed,packed+1u,packed+1u+l_.width});
    }
    void put(const double* input,std::uint32_t mask,std::uint32_t coordinate,const double* packed,
        std::int32_t status,std::uint32_t valid) noexcept {
        if (!storage_ || !packed) return;
        put(input,mask,coordinate,Values{packed,packed+1u,packed+1u+l_.width},status,valid);
    }
    bool bind(Layout l,void* memory,std::size_t bytes) noexcept {
        *this = Cache{};
        if (!l.bytes || !memory || bytes<l.bytes
            || reinterpret_cast<std::uintptr_t>(memory)%alignof(Entry)) return false;
        work::WorkspaceArena arena(memory,bytes);
        entries_=arena.take<Entry>(l.entries);
        storage_=arena.take<double>(l.entries*l.stride);
        if (!storage_) return false;
        l_=l; clock_=0; std::fill_n(entries_,l.entries,Entry{}); return true;
    }
    bool get(const double* input,std::uint32_t mask,std::uint32_t coordinate,Outputs outputs) noexcept {
        if (!storage_ || !mask || (mask&~l_.supported) || coordinate>1u) return false;
        const auto h=key(input,mask,coordinate);
        const auto first=(h&(l_.buckets-1u))*ways;
        tick();
        for (std::size_t i=first;i<first+ways;++i) {
            auto& e=entries_[i];
            if (e.age && e.hash==h && e.mask==mask && e.coordinate==coordinate
                && std::memcmp(point(i),input,l_.width*sizeof(double))==0) {
                e.age=clock_; restore(outputs,point(i)+l_.width,mask); return true;
            }
        }
        return false;
    }
    void put(const double* input,std::uint32_t mask,std::uint32_t coordinate,
        Values outputs,std::int32_t status,std::uint32_t valid) noexcept {
        if (!storage_ || status || valid!=mask || !mask || (mask&~l_.supported) || coordinate>1u) return;
        for (std::size_t i=0;i<l_.width;++i) if (!std::isfinite(input[i])) return;
        if ((mask&1u) && !std::isfinite(*outputs.value)) return;
        if (mask&2u) for (std::size_t i=0;i<l_.width;++i)
            if (!std::isfinite(outputs.gradient[i])) return;
        if (mask&4u) for (std::size_t i=0;i<l_.square;++i)
            if (!std::isfinite(outputs.hessian[i])) return;
        const auto h=key(input,mask,coordinate);
        const auto first=(h&(l_.buckets-1u))*ways;
        auto slot=first;
        for (std::size_t i=first+1;i<first+ways;++i)
            if (entries_[i].age<entries_[slot].age) slot=i;
        tick();
        std::memcpy(point(slot),input,l_.width*sizeof(double));
        store(point(slot)+l_.width,outputs,mask);
        entries_[slot]={h,clock_,mask,coordinate};
    }
};

class Bank {
    Cache first_,curvature_;
    std::size_t queries_=0,hits_=0;
public:
    static std::size_t bytes(std::size_t width,std::size_t buckets) noexcept {
        const auto first=layout(width,buckets,false), curvature=layout(width,1u,true);
        std::size_t result=0;
        return first.bytes && curvature.bytes && work::checked_sum(first.bytes,curvature.bytes,result)
            ? result : 0u;
    }
    bool bind(std::size_t width,std::size_t buckets,void* memory,std::size_t size) noexcept {
        *this = Bank{};
        const auto required=bytes(width,buckets), first=layout(width,buckets,false).bytes;
        const bool valid=required && memory && size>=required
            && first_.bind(layout(width,buckets,false),memory,first)
            && curvature_.bind(layout(width,1u,true),static_cast<std::byte*>(memory)+first,size-first);
        if (valid) queries_=hits_=0;
        return valid;
    }
    Cache& for_mask(std::uint32_t mask) noexcept { return mask&4u?curvature_:first_; }
    bool get(const double* point,std::uint32_t mask,std::uint32_t coordinate,Outputs out) noexcept {
        ++queries_;
        const bool hit=for_mask(mask).get(point,mask,coordinate,out);
        hits_+=hit; return hit;
    }
    void put(const double* point,std::uint32_t mask,std::uint32_t coordinate,Values out,
        std::int32_t status,std::uint32_t valid) noexcept {
        for_mask(mask).put(point,mask,coordinate,out,status,valid);
    }
    std::size_t queries() const noexcept { return queries_; }
    std::size_t hits() const noexcept { return hits_; }
};

} // namespace vlk::experiment::request_reuse
