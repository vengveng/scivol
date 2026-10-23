#pragma once

#include "garch_extended_density_policy.hpp"
#include "gh_point_preparation.hpp"
#include "moment_preparation.hpp"
#include <scivolapi/detail/types.hpp>
#include "../preparation_banks.hpp"
#include "../math/local_density_status.hpp"
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace scivolapi::detail {
using LocalDensityAxes=vlk::experiment::generated::LocalDensityAxes;
enum class PreparedDensity : unsigned {student=1,skew=2,ged=3,gh=4,nig=5,hyperbolic=6,vg=7};
struct DensityPreparationOptions {
    PreparedDensity density=PreparedDensity::student;
    unsigned residual_orders=0;
    MomentReservations moments{};
    // Immutable per-order producer basis; individual consumers may demand less.
    std::array<unsigned,3> shape_axes{~0u,~0u,~0u};
};
template<class D> inline constexpr bool elementary_density =
    std::is_same_v<D,vlk::experiment::generated::GedDensity>
    || std::is_same_v<D,vlk::experiment::generated::SkewTDensity>;
template<class D> inline constexpr auto elementary_kind =
    std::is_same_v<D,vlk::experiment::generated::GedDensity> ? PreparedDensity::ged:PreparedDensity::skew;
template<class D> inline constexpr PreparedDensity gh_kind=[] {
    using namespace vlk::experiment::generated;
    if constexpr (std::is_same_v<D,GeneralizedHyperbolicDensity>) return PreparedDensity::gh;
    else if constexpr (std::is_same_v<D,HyperbolicDensity>) return PreparedDensity::hyperbolic;
    else if constexpr (std::is_same_v<D,NormalInverseGaussianDensity>) return PreparedDensity::nig;
    else {static_assert(std::is_same_v<D,VarianceGammaDensity>);return PreparedDensity::vg;}
}();
template<class Consumer>
decltype(auto) with_prepared_density(PreparedDensity kind,Consumer&& consume) noexcept {
    using namespace vlk::experiment::generated;
    switch(kind) {
    case PreparedDensity::student:return consume.template operator()<StudentTDensity>();
    case PreparedDensity::skew:return consume.template operator()<SkewTDensity>();
    case PreparedDensity::ged:return consume.template operator()<GedDensity>();
    case PreparedDensity::gh:return consume.template operator()<GeneralizedHyperbolicDensity>();
    case PreparedDensity::hyperbolic:return consume.template operator()<HyperbolicDensity>();
    case PreparedDensity::nig:return consume.template operator()<NormalInverseGaussianDensity>();
    case PreparedDensity::vg:return consume.template operator()<VarianceGammaDensity>();
    }
    return consume.template operator()<StudentTDensity>();
}

// Optional test-owned sink. No global counters or optimizer output caching.
struct PreparationCounts {
    std::array<std::size_t, 8> producer_masks{};
    std::array<std::size_t, 8> consumer_masks{};
    std::size_t value_calls=0, first_calls=0, second_calls=0;
    std::size_t scaled_replays=0;
    std::array<std::array<std::size_t,3>,8> density_builds{};
    std::array<std::array<std::size_t,8>,8> density_requests{};
    std::array<std::array<std::size_t,moment_producers>,3> moment_builds{};
    vlk::experiment::generated::GhSecondPreparationCounts gh_second{};
    bool cold_density=false;
};
template<bool F,bool G,bool H>
constexpr unsigned student_normalizer_demand(vlk::experiment::generated::LocalDensityAxes axes) noexcept {
    return unsigned(F) | (unsigned(G && axes.shape(0))<<1) | (unsigned(H && axes.shape(0))<<2);
}
template<bool F, bool G, bool H>
auto prepare_student_point(const double* shape, std::size_t scored, PreparationCounts* counts,
    vlk::experiment::generated::LocalDensityAxes axes = {}) noexcept {
    if (counts) {
        ++counts->producer_masks[unsigned(F) | (unsigned(G)<<1) | (unsigned(H)<<2)];
        counts->value_calls+=F; counts->first_calls+=G; counts->second_calls+=H;
    }
    return vlk::experiment::generated::StudentTDensity::template prepare<F,G,H>(shape,scored,axes);
}
inline auto prepare_selected_student_point(unsigned mask,const double* shape,std::size_t scored,
    PreparationCounts* counts,vlk::experiment::generated::LocalDensityAxes axes) noexcept {
    switch (mask) {
#define SCIVOL_STUDENT_PREPARE(M) case M: return prepare_student_point<bool(M&1),bool(M&2),bool(M&4)>(shape,scored,counts,axes)
        SCIVOL_STUDENT_PREPARE(0); SCIVOL_STUDENT_PREPARE(1);
        SCIVOL_STUDENT_PREPARE(2); SCIVOL_STUDENT_PREPARE(3);
        SCIVOL_STUDENT_PREPARE(4); SCIVOL_STUDENT_PREPARE(5);
        SCIVOL_STUDENT_PREPARE(6); SCIVOL_STUDENT_PREPARE(7);
#undef SCIVOL_STUDENT_PREPARE
    }
    return vlk::experiment::generated::StudentTDensity::Prepared{};
}

// Mathematical payload only. WorkspaceState owns the single point identity and
// snapshots shared with unit preparation. This owner neither hashes nor binds points.
class DensityPreparation {
    using Student=vlk::experiment::generated::StudentTDensity;
    using Prepared=Student::Prepared;
    struct Slot {
        point_preparation::Bank bank;
        unsigned ready=0;
        bool base_ready=false;
    };
    std::array<Slot,3> slots_{};
    Prepared* staging_=nullptr;
    DensityPreparationOptions options_{};
    std::span<const double> theta_;
    std::size_t scored_=0;
    using Layout=std::array<std::array<point_preparation::storage::BankLayout,2>,3>;
    static DensityPreparationOptions canonical_options(DensityPreparationOptions options) noexcept {
        const auto shapes=with_prepared_density(options.density,[]<class D>() {
            return (1u<<D::shape_parameters)-1u;
        });
        for(unsigned order=0;order<3;++order) {
            options.shape_axes[order]&=shapes;
            if (options.moments[order][unsigned(MomentProducer::negative_second)].object
                || options.moments[order][unsigned(MomentProducer::power_batch)].object)
                options.shape_axes[order]|=shapes;
        }
        return options;
    }
    template<unsigned Order,class D>
    bool admitted_axes(LocalDensityAxes axes) const noexcept {
        constexpr unsigned shapes=(1u<<D::shape_parameters)-1u;
        return !(axes.shapes&shapes&~options_.shape_axes[Order]);
    }
    template<unsigned Order>
    LocalDensityAxes producer_axes(LocalDensityAxes axes) const noexcept {
        axes.shapes=options_.shape_axes[Order];
        return axes;
    }
    template<class D,unsigned Order,bool Residual>
    static bool append_elementary(point_preparation::storage::LayoutBuilder& size,Layout& banks) noexcept {
        using Value=DensityPayloadT<D,Order,Residual>;
        static_assert(alignof(Value)<=alignof(double));
        return point_preparation::storage::append_banks<Value>(size,banks[Order]);
    }
    template<class D,unsigned Order>
    static bool append_elementary(point_preparation::storage::LayoutBuilder& size,Layout& banks,unsigned residuals) noexcept {
        return residuals&(1u<<Order) ? append_elementary<D,Order,true>(size,banks)
            : append_elementary<D,Order,false>(size,banks);
    }
    static std::size_t layout(unsigned mask,Layout& banks,const DensityPreparationOptions& options,
        MomentLayouts* moment_layouts=nullptr) noexcept {
        if (!mask || mask>7 || options.residual_orders>7
            || unsigned(options.density)<1 || unsigned(options.density)>7) return 0;
        for(unsigned order=0;order<3;++order)
            if (options.moments[order][unsigned(MomentProducer::negative_second)].object
                || options.moments[order][unsigned(MomentProducer::power_batch)].object) mask|=1u<<order;
        point_preparation::storage::LayoutBuilder size;
        if (options.density==PreparedDensity::student) {
            for (unsigned order=0;order<3;++order)
                if ((mask&(1u<<order)) && !point_preparation::storage::append_banks<Prepared>(size,banks[order]))
                    return 0;
        } else {
            const auto append=[&]<class D>() {
                return (!(mask&1) || append_elementary<D,0>(size,banks,options.residual_orders))
                    && (!(mask&2) || append_elementary<D,1>(size,banks,options.residual_orders))
                    && (!(mask&4) || append_elementary<D,2>(size,banks,options.residual_orders));
            };
            const bool ok=with_prepared_density(options.density,[&]<class D>() {
                if constexpr (std::is_same_v<D,Student>) return false;
                else return append.template operator()<D>();
            });
            if (!ok) return 0;
        }
        MomentLayouts moments{};
        if (!MomentPreparation::append(size,options.moments,moments)) return 0;
        if (moment_layouts) *moment_layouts=moments;
        return size.bytes();
    }
    template<class D,unsigned Order,bool Residual>
    Status bind_elementary(std::span<std::byte> memory,const Layout& banks) noexcept {
        return slots_[Order].bank.template bind<DensityPayloadT<D,Order,Residual>>(memory,banks[Order]);
    }
    template<class D,unsigned Order>
    Status bind_elementary(std::span<std::byte> memory,const Layout& banks,unsigned residuals) noexcept {
        return residuals&(1u<<Order) ? bind_elementary<D,Order,true>(memory,banks)
            : bind_elementary<D,Order,false>(memory,banks);
    }
    template<class D> Status bind_elementary(std::span<std::byte> memory,const Layout& banks,unsigned mask) noexcept {
        if ((mask&1) && bind_elementary<D,0>(memory,banks,options_.residual_orders)!=Status::ok) return Status::invalid_argument;
        if ((mask&2) && bind_elementary<D,1>(memory,banks,options_.residual_orders)!=Status::ok) return Status::invalid_argument;
        if ((mask&4) && bind_elementary<D,2>(memory,banks,options_.residual_orders)!=Status::ok) return Status::invalid_argument;
        return Status::ok;
    }
    Prepared produce(unsigned mask, const double* shape,
        vlk::experiment::generated::LocalDensityAxes axes) noexcept {
        return prepare_selected_student_point(mask,shape,scored_,counts,axes);
    }
public:
    struct Requirements {
        std::size_t persistent=0,staging=sizeof(Prepared);
        DensityPreparationOptions options{};
    };
    PreparationCounts* counts=nullptr;
    MomentPreparation moments;
    DensityPreparation() noexcept=default;
    DensityPreparation(const DensityPreparation&)=delete;
    DensityPreparation& operator=(const DensityPreparation&)=delete;
    ~DensityPreparation() { release(); }
    static Requirements requirements(unsigned mask,DensityPreparationOptions options = {}) noexcept {
        options=canonical_options(options);
        Layout banks{};
        return {layout(mask,banks,options),options.density==PreparedDensity::student ? sizeof(Prepared):0,options};
    }
    Status bind(unsigned mask, std::span<std::byte> persistent, std::span<std::byte> staging,
        DensityPreparationOptions options = {}) noexcept {
        const auto r=requirements(mask,options);
        options=r.options;
        if (!r.persistent || persistent.size()<r.persistent || staging.size()<r.staging)
            return Status::insufficient_capacity;
        const auto p=reinterpret_cast<std::uintptr_t>(persistent.data());
        const auto s=reinterpret_cast<std::uintptr_t>(staging.data());
        const auto m=reinterpret_cast<std::uintptr_t>(this);
        const auto end=[](std::uintptr_t begin,std::size_t size) {return size<=UINTPTR_MAX-begin;};
        if (!p || (r.staging && !s) || p%alignof(Prepared) || s%alignof(Prepared)
            || !end(p,persistent.size()) || !end(s,staging.size()) || !end(m,sizeof(*this)))
            return Status::invalid_argument;
        const auto overlaps=[](auto a,auto n,auto b,auto k) {return n && k && a<b+k && b<a+n;};
        if (overlaps(p,persistent.size(),s,staging.size())
            || overlaps(p,persistent.size(),m,sizeof(*this)) || overlaps(s,staging.size(),m,sizeof(*this)))
            return Status::invalid_argument;
        release();
        options_=options;
        for(unsigned order=0;order<3;++order)
            if (options.moments[order][unsigned(MomentProducer::negative_second)].object
                || options.moments[order][unsigned(MomentProducer::power_batch)].object) mask|=1u<<order;
        Layout banks{};MomentLayouts moment_layouts{};
        layout(mask,banks,options,&moment_layouts);
        const auto moment_status=moments.bind(persistent,options.moments,moment_layouts);
        if (moment_status!=Status::ok) {release();return moment_status;}
        if (options.density!=PreparedDensity::student) {
            const auto status=with_prepared_density(options.density,[&]<class D>() {
                if constexpr (std::is_same_v<D,Student>) return Status::invalid_argument;
                else return bind_elementary<D>(persistent,banks,mask);
            });
            if (status!=Status::ok) release();
            return status;
        }
        for (unsigned order=0;order<3;++order) {
            if (!(mask&(1u<<order))) continue;
            const auto status=slots_[order].bank.bind<Prepared>(persistent,banks[order]);
            if (status!=Status::ok) {release();return status;}
        }
        staging_=reinterpret_cast<Prepared*>(staging.data());
        return Status::ok;
    }
    void release() noexcept {
        moments.release();
        for (auto& slot:slots_) {slot.bank.release();slot.ready=0;slot.base_ready=false;}
        staging_=nullptr;theta_={};scored_=0;
    }
    // Called only by the shared identity reset, after its immutable snapshots copy.
    void reset(std::span<const double> theta, std::size_t scored) noexcept {
        moments.clear();
        for (auto& slot:slots_) {slot.bank.clear();slot.ready=0;slot.base_ready=false;}
        theta_=theta; scored_=scored;
    }
    template<class D> const double* stable_shape(const double* shape) const noexcept {
        if constexpr (gh_density<D>) {if (options_.density!=gh_kind<D>) return nullptr;}
        else if constexpr (elementary_density<D>) {if (options_.density!=elementary_kind<D>) return nullptr;}
        else if constexpr (std::is_same_v<D,Student>) {if (options_.density!=PreparedDensity::student) return nullptr;}
        else return nullptr;
        if (theta_.size()<D::shape_parameters || !D::valid_shape(shape)) return nullptr;
        const auto* stable=theta_.data()+theta_.size()-D::shape_parameters;
        return std::memcmp(shape,stable,D::shape_parameters*sizeof(double)) ? nullptr:stable;
    }
    std::span<const double> point_block(std::size_t begin,std::size_t count) const noexcept {
        return begin>theta_.size() || count>theta_.size()-begin ? std::span<const double>{}:theta_.subspan(begin,count);
    }
    template<class D,unsigned Order>
    const auto* moment_density(const double* shape,LocalDensityAxes axes = {}) noexcept {
        using Value=typename GhPreparedPoint<D,Order,false>::Companion;
        if (options_.residual_orders&(1u<<Order)) {
            auto* payload=gh<D,Order,true>(shape,scored_,axes);
            return payload ? payload->companion(shape,axes):static_cast<const Value*>(nullptr);
        }
        auto* payload=gh<D,Order,false>(shape,scored_,axes);
        return payload ? payload->companion(shape,axes):static_cast<const Value*>(nullptr);
    }
    template<unsigned Order, bool F, bool G, bool H>
    const Prepared* student(const double* shape, std::size_t scored,
        vlk::experiment::generated::LocalDensityAxes axes = {}) noexcept {
        static_assert(Order==(H ? 2u : G ? 1u : 0u));
        auto* slot=&slots_[Order];
        if (options_.density!=PreparedDensity::student || !slot->bank.bound() || !staging_ || theta_.empty() || scored!=scored_ || !Student::valid_shape(shape)
            || std::memcmp(shape,&theta_.back(),sizeof(double))) return nullptr;
        auto* value=slot->bank.get<Prepared>();
        const unsigned need=student_normalizer_demand<F,G,H>(axes);
        const unsigned missing=need & ~slot->ready;
        if (!missing && slot->base_ready) {
            value->axes=axes;
            return value;
        }
        auto* candidate=std::construct_at(staging_,produce(missing,shape,axes));
        bool valid=candidate->valid && std::isfinite(candidate->nu)
            && std::isfinite(candidate->inverse_nu_minus_two)
            && (!(missing&1) || std::isfinite(candidate->constant))
            && (!(missing&2) || std::isfinite(candidate->normalizer_first))
            && (!(missing&4) || std::isfinite(candidate->normalizer_second));
        if (valid && !slot->base_ready) {
            valid=slot->bank.replace<Prepared>([&](auto& target) noexcept {
                target.emplace(*candidate);
                return Status::ok;
            })==Status::ok;
            value=slot->bank.get<Prepared>();
        }
        if (valid) {
            // Student's prepared state is scalar-only. Complete independent
            // components; never copy another order or a pointer-bearing payload.
            value->nu=candidate->nu;
            value->inverse_nu_minus_two=candidate->inverse_nu_minus_two;
            value->axes=axes;
            slot->base_ready=true;
            if (missing&1) value->constant=candidate->constant;
            if (missing&2) value->normalizer_first=candidate->normalizer_first;
            if (missing&4) value->normalizer_second=candidate->normalizer_second;
            slot->ready|=missing;
        }
        std::destroy_at(candidate);
        return valid ? value : nullptr;
    }
    template<class D,unsigned Order,bool F,bool G,bool H,bool Residual,class Factory>
    const auto* elementary(const double* shape,std::size_t scored,LocalDensityAxes axes,Factory&& factory) noexcept {
        static_assert(elementary_density<D> && Order==(H ? 2u:G ? 1u:0u));
        using Value=typename D::template Prepared<Order,Residual>;
        auto& slot=slots_[Order];
        const auto reject=[]() -> const Value* {return nullptr;};
        if (options_.density!=elementary_kind<D> || !slot.bank.bound()
            || bool(options_.residual_orders&(1u<<Order))!=Residual || theta_.size()<D::shape_parameters
            || scored!=scored_ || !D::valid_shape(shape)
            || !admitted_axes<Order,D>(axes)
            || std::memcmp(shape,theta_.data()+theta_.size()-D::shape_parameters,D::shape_parameters*sizeof(double)))
            return reject();
        const auto prepared_axes=producer_axes<Order>(axes);
        auto* value=slot.bank.template get<Value>();
        if (value) {
            if (!D::template prepared_valid<false>(*value,axes)) return reject();
            if constexpr (F)
                if (!(slot.ready&1) && !D::template prepared_valid<true>(*value,axes)) return reject();
        } else {
            if (counts) ++counts->density_builds[unsigned(options_.density)][Order];
            const auto status=slot.bank.template replace<Value>([&](auto& target) noexcept {
                const auto* candidate=target.emplace_from([&] {
                    if constexpr (std::is_invocable_v<Factory&,LocalDensityAxes>) return factory(prepared_axes);
                    else {
                        constexpr unsigned shapes=(1u<<D::shape_parameters)-1u;
                        if ((axes.shapes&shapes)==prepared_axes.shapes) return factory();
                        return D::template prepare<F,G,H,Residual>(shape,scored,prepared_axes);
                    }
                });
                return (candidate->axes.shapes&((1u<<D::shape_parameters)-1u))==prepared_axes.shapes
                    && vlk::experiment::density_status::accept(*candidate,nullptr)
                    && D::template prepared_valid<F>(*candidate,axes) ? Status::ok:Status::numerical_range;
            });
            if (status!=Status::ok) return reject();
            value=slot.bank.template get<Value>();
        }
        value->axes=axes;slot.base_ready=true;
        slot.ready|=unsigned(F)|(unsigned(G)<<1)|(unsigned(H)<<2);
        return static_cast<const Value*>(value);
    }

    template<class D,unsigned Order,bool Residual>
    GhPreparedPoint<D,Order,Residual>* gh(const double* shape,std::size_t scored,LocalDensityAxes axes) noexcept {
        static_assert(gh_density<D> && Order<=2);
        using Value=GhPreparedPoint<D,Order,Residual>;
        auto& slot=slots_[Order];
        if (options_.density!=gh_kind<D> || !slot.bank.bound()
            || bool(options_.residual_orders&(1u<<Order))!=Residual
            || theta_.size()<D::shape_parameters || scored!=scored_ || !D::valid_shape(shape)
            || !admitted_axes<Order,D>(axes)
            || std::memcmp(shape,theta_.data()+theta_.size()-D::shape_parameters,D::shape_parameters*sizeof(double)))
            return nullptr;
        const auto prepared_axes=producer_axes<Order>(axes);
        auto* value=slot.bank.template get<Value>();
        if (value) {
            if (value->shape_axes()!=prepared_axes.shapes) return nullptr;
        } else {
            if (counts) ++counts->density_builds[unsigned(options_.density)][Order];
            const auto status=slot.bank.template replace<Value>([&](auto& target) noexcept {
                const auto* candidate=target.emplace(shape,prepared_axes,counts ? &counts->gh_second:nullptr);
                return candidate->get().valid && candidate->retained_plan_is_local()
                    ? Status::ok:Status::numerical_range;
            });
            if (status!=Status::ok) return nullptr;
            value=slot.bank.template get<Value>();
        }
        value->refresh(axes);slot.base_ready=true;
        return value;
    }
};

template<class Density, bool F, bool G, bool H, bool Residual=false>
class DensityLease {
    using Prepared=decltype(Density::template prepare<F,G,H,Residual>(nullptr,0));
    Prepared cold_;
public:
    template<class Factory>
    DensityLease(DensityPreparation*, const double*, std::size_t,
        vlk::experiment::generated::LocalDensityAxes, Factory&& factory) noexcept
        : cold_(factory()) {}
    template<class Factory>
    DensityLease(DensityPreparation* cache, const double* shape, std::size_t scored, Factory&& factory) noexcept
        : DensityLease(cache,shape,scored,vlk::experiment::generated::LocalDensityAxes{},
            std::forward<Factory>(factory)) {}
    DensityLease(DensityPreparation* cache, const double* shape, std::size_t scored) noexcept
        : DensityLease(cache,shape,scored,[&] {return Density::template prepare<F,G,H,Residual>(shape,scored);}) {}
    DensityLease(DensityPreparation* cache, const double* shape, std::size_t scored,
        vlk::experiment::generated::LocalDensityAxes axes) noexcept
        : DensityLease(cache,shape,scored,axes,[&] {return Density::template prepare<F,G,H,Residual>(shape,scored,axes);}) {}
    DensityLease(const DensityLease&)=delete;
    DensityLease& operator=(const DensityLease&)=delete;
    bool valid() const noexcept {return cold_.valid;}
    bool accept(vlk::experiment::density_status::Report* report) const noexcept {
        return vlk::experiment::density_status::accept(cold_,report);
    }
    const Prepared& get() const noexcept {return cold_;}
};

template<bool F, bool G, bool H, bool Residual>
class DensityLease<vlk::experiment::generated::StudentTDensity,F,G,H,Residual> {
    using Prepared=vlk::experiment::generated::StudentTDensity::Prepared;
    std::optional<Prepared> cold_;
    const Prepared* prepared_=nullptr;
public:
    DensityLease(DensityPreparation* cache, const double* shape, std::size_t scored,
        vlk::experiment::generated::LocalDensityAxes axes = {}) noexcept {
        if (cache && cache->counts)
            ++cache->counts->consumer_masks[unsigned(F) | (unsigned(G)<<1) | (unsigned(H)<<2)];
        if (cache && !(cache->counts && cache->counts->cold_density)) {
            prepared_=cache->template student<H ? 2 : G ? 1 : 0,F,G,H>(shape,scored,axes);
            return;
        }
        // Cold and persistent paths use the same selected normalizer producer.
        const unsigned need=student_normalizer_demand<F,G,H>(axes);
        cold_.emplace(prepare_selected_student_point(need,shape,scored,cache ? cache->counts : nullptr,axes));
        prepared_=&*cold_;
    }
    template<class Factory>
    DensityLease(DensityPreparation* cache, const double* shape, std::size_t scored, Factory&&) noexcept
        : DensityLease(cache,shape,scored) {}
    template<class Factory>
    DensityLease(DensityPreparation* cache, const double* shape, std::size_t scored,
        vlk::experiment::generated::LocalDensityAxes axes, Factory&&) noexcept
        : DensityLease(cache,shape,scored,axes) {}
    DensityLease(const DensityLease&)=delete;
    DensityLease& operator=(const DensityLease&)=delete;
    bool valid() const noexcept {return prepared_ && prepared_->valid;}
    bool accept(vlk::experiment::density_status::Report* report) const noexcept {
        if (prepared_) return vlk::experiment::density_status::accept(*prepared_,report);
        if (report) report->reject(vlk::experiment::density_status::Code::numerical_failure);
        return false;
    }
    const Prepared& get() const noexcept {return *prepared_;}
};
template<class Density,bool F,bool G,bool H,bool Residual> requires elementary_density<Density>
class DensityLease<Density,F,G,H,Residual> {
    static constexpr unsigned Order=H ? 2:G ? 1:0;
    using Prepared=typename Density::template Prepared<Order,Residual>;
    std::optional<Prepared> cold_;
    const Prepared* prepared_=nullptr;
public:
    template<class Factory>
    DensityLease(DensityPreparation* cache,const double* shape,std::size_t scored,
        LocalDensityAxes axes,Factory&& factory) noexcept {
        if (cache && cache->counts)
            ++cache->counts->density_requests[unsigned(elementary_kind<Density>)][unsigned(F)|(unsigned(G)<<1)|(unsigned(H)<<2)];
        if (cache && !(cache->counts && cache->counts->cold_density)) {
            prepared_=cache->template elementary<Density,Order,F,G,H,Residual>(shape,scored,axes,std::forward<Factory>(factory));
        } else {
            if (cache && cache->counts) ++cache->counts->density_builds[unsigned(elementary_kind<Density>)][Order];
            cold_.emplace(factory());prepared_=&*cold_;
        }
    }
    DensityLease(DensityPreparation* cache,const double* shape,std::size_t scored,LocalDensityAxes axes = {}) noexcept
        : DensityLease(cache,shape,scored,axes,[&] {return Density::template prepare<F,G,H,Residual>(shape,scored,axes);}) {}
    template<class Factory> requires (!std::is_same_v<std::remove_cvref_t<Factory>,LocalDensityAxes>)
    DensityLease(DensityPreparation* cache,const double* shape,std::size_t scored,Factory&& factory) noexcept
        : DensityLease(cache,shape,scored,LocalDensityAxes{},std::forward<Factory>(factory)) {}
    DensityLease(const DensityLease&)=delete;
    DensityLease& operator=(const DensityLease&)=delete;
    bool valid() const noexcept {return prepared_ && prepared_->valid;}
    bool accept(vlk::experiment::density_status::Report* report) const noexcept {
        if (prepared_) return vlk::experiment::density_status::accept(*prepared_,report);
        if (report) report->reject(vlk::experiment::density_status::Code::numerical_failure);
        return false;
    }
    const Prepared& get() const noexcept {return *prepared_;}
};
template<class Density,bool F,bool G,bool H,bool Residual> requires gh_density<Density>
class DensityLease<Density,F,G,H,Residual> {
    static constexpr unsigned Order=H ? 2:G ? 1:0;
    using Payload=GhPreparedPoint<Density,Order,Residual>;
    using Prepared=typename Payload::Prepared;
    std::optional<Payload> cold_;
    Payload* payload_=nullptr;
    const double* shape_=nullptr;
    LocalDensityAxes axes_{};
public:
    DensityLease(DensityPreparation* cache,const double* shape,std::size_t scored,
        LocalDensityAxes axes = {}) noexcept :shape_(shape),axes_(axes) {
        if (cache && cache->counts)
            ++cache->counts->density_requests[unsigned(gh_kind<Density>)][unsigned(F)|(unsigned(G)<<1)|(unsigned(H)<<2)];
        if (cache && !(cache->counts && cache->counts->cold_density)) {
            shape_=cache->template stable_shape<Density>(shape);
            if (shape_) payload_=cache->template gh<Density,Order,Residual>(shape_,scored,axes);
        } else {
            if (cache && cache->counts) ++cache->counts->density_builds[unsigned(gh_kind<Density>)][Order];
            // Direct member construction also keeps the cold self-reference local.
            cold_.emplace(shape,axes,cache && cache->counts ? &cache->counts->gh_second:nullptr);payload_=&*cold_;
        }
    }
    template<class Factory> requires (!std::is_same_v<std::remove_cvref_t<Factory>,LocalDensityAxes>)
    DensityLease(DensityPreparation* cache,const double* shape,std::size_t scored,Factory&&) noexcept
        :DensityLease(cache,shape,scored) {}
    template<class Factory>
    DensityLease(DensityPreparation* cache,const double* shape,std::size_t scored,
        LocalDensityAxes axes,Factory&&) noexcept :DensityLease(cache,shape,scored,axes) {}
    DensityLease(const DensityLease&)=delete;
    DensityLease& operator=(const DensityLease&)=delete;
    bool valid() const noexcept {return payload_ && payload_->get().valid;}
    bool accept(vlk::experiment::density_status::Report* report) const noexcept {
        if (payload_) return vlk::experiment::density_status::accept(payload_->get(),report);
        if (report) report->reject(vlk::experiment::density_status::Code::numerical_failure);
        return false;
    }
    const Prepared& get() const noexcept {return payload_->get();}
    const auto* moment_prepared() noexcept {
        return payload_ ? payload_->companion(shape_,axes_):nullptr;
    }
};
inline void note_preparation_replay(DensityPreparation* cache) noexcept {
    if (cache && cache->counts) ++cache->counts->scaled_replays;
}
} // namespace scivolapi::detail
