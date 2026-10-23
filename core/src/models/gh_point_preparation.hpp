#pragma once
#include "garch_extended_density_policy.hpp"
#include <optional>
#include <type_traits>
#include <variant>

namespace scivolapi::detail {
template<class Density> struct GhPreparationTraits { static constexpr bool enabled=false; };
template<class Family>
struct GhPreparationTraits<vlk::experiment::generated::GhFamilyDensity<Family>> {
    static constexpr bool enabled=true;
    using family=Family;
};
template<class Density> inline constexpr bool gh_density=GhPreparationTraits<Density>::enabled;

// The sharing plan borrows its first cache. Both objects start their lifetimes
// here, at the committed bank address, before a companion can borrow the plan.
template<class Density,unsigned Order,bool Residual>
class GhPreparedPoint {
    using Family=typename GhPreparationTraits<Density>::family;
public:
    using Prepared=decltype(Density::template prepare<true,Order>=1,Order>=2,Residual>(nullptr,0));
    using Companion=typename vlk::experiment::generated::GhMomentBorrowType<
        decltype(Density::template prepare<true,Order>=1,Order>=2,false>(nullptr,0))>::type;
private:
    static constexpr bool has_companion=Order==2 || (Residual && Order==1);
    unsigned shape_axes_=0;
    [[no_unique_address]] std::conditional_t<Order==1,
        vlk::experiment::gh_first::FirstPreparation,std::monostate> preparation_{};
    Prepared prepared_;
    [[no_unique_address]] std::conditional_t<has_companion,
        std::optional<Companion>,std::monostate> companion_;
public:
    GhPreparedPoint(const double* shape,vlk::experiment::generated::LocalDensityAxes axes,
        vlk::experiment::generated::GhSecondPreparationCounts* counts=nullptr) noexcept
        :shape_axes_(axes.shapes),prepared_([&]() -> Prepared {
            if constexpr (Order==1) return Prepared(shape,&preparation_,axes);
            else if constexpr (Order==2) return Prepared(shape,axes,counts);
            else return Prepared(shape,axes);
        }()) {}
    GhPreparedPoint(const GhPreparedPoint&)=delete;
    GhPreparedPoint& operator=(const GhPreparedPoint&)=delete;
    GhPreparedPoint(GhPreparedPoint&&)=delete;
    GhPreparedPoint& operator=(GhPreparedPoint&&)=delete;
    const Prepared& get() const noexcept {return prepared_;}
    unsigned shape_axes() const noexcept {return shape_axes_;}
    auto axes() const noexcept {
        if constexpr (Order==0) return prepared_.axes;
        else return prepared_.cache.axes;
    }
    void refresh(vlk::experiment::generated::LocalDensityAxes axes) noexcept {
        if constexpr (Order==0) prepared_.axes=axes;
        else prepared_.cache.axes=axes;
        if constexpr (Residual && Order==1)
            if (companion_) companion_->cache.axes=axes;
    }
    const Companion* companion(const double* shape,
        vlk::experiment::generated::LocalDensityAxes axes) noexcept {
        if constexpr (!has_companion) return &prepared_;
        else {
            if (!companion_) {
                if constexpr (Order==1) {
                    axes.shapes=shape_axes_;
                    companion_.emplace(shape,&preparation_,axes);
                }
                else companion_.emplace(prepared_);
                if (!companion_->valid) {companion_.reset();return nullptr;}
            }
            return &*companion_;
        }
    }
    bool retained_plan_is_local() const noexcept {
        if constexpr (Order==1)
            return !preparation_.retained_plan
                || preparation_.retained_plan==&prepared_.cache.observation_bessel;
        else return true;
    }
};

template<class Density,unsigned Order,bool Residual,bool Gh=gh_density<Density>>
struct DensityPayload {using type=typename Density::template Prepared<Order,Residual>;};
template<class Density,unsigned Order,bool Residual>
struct DensityPayload<Density,Order,Residual,true> {using type=GhPreparedPoint<Density,Order,Residual>;};
template<class Density,unsigned Order,bool Residual>
using DensityPayloadT=typename DensityPayload<Density,Order,Residual>::type;
} // namespace scivolapi::detail
