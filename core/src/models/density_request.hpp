#pragma once

#include "density_preparation.hpp"

namespace scivolapi::detail {
struct NoDensityBorrow {
    template<class Prepared> const Prepared* get() const noexcept {return nullptr;}
};
template<class Prepared> struct DensityBorrow {
    const Prepared* value;
    template<class Target> const Target* get() const noexcept {
        if constexpr(std::is_same_v<Target,Prepared>)return value;
        else return nullptr;
    }
};
template<class Prepared> DensityBorrow<Prepared> borrow_density(const Prepared& value) noexcept {return {&value};}

// A replay borrows the request's existing lease. The owning form still constructs
// through that lease's exact factory, at its final address, with no payload copy.
template<class Density,bool F,bool G,bool H,bool Residual>
class DensityRequest {
    using Lease=DensityLease<Density,F,G,H,Residual>;
    using Prepared=decltype(Density::template prepare<F,G,H,Residual>(nullptr,0));
    std::optional<Lease> owner_;
    const Prepared* prepared_=nullptr;
public:
    template<class Factory,class Borrowed=NoDensityBorrow>
    DensityRequest(DensityPreparation* cache,const double* shape,std::size_t scored,
        vlk::experiment::generated::LocalDensityAxes axes,Factory&& factory,Borrowed borrowed={}) noexcept {
        prepared_=borrowed.template get<Prepared>();
        if(!prepared_) {
            owner_.emplace(cache,shape,scored,axes,std::forward<Factory>(factory));
            if(owner_->valid())prepared_=&owner_->get();
        }
    }
    bool valid() const noexcept {return prepared_ && prepared_->valid;}
    bool accept(vlk::experiment::density_status::Report* report) const noexcept {
        if(prepared_)return vlk::experiment::density_status::accept(*prepared_,report);
        return owner_ && owner_->accept(report);
    }
    const Prepared& get() const noexcept {return *prepared_;}
};
} // namespace scivolapi::detail
