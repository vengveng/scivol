#pragma once
#include "density_preparation.hpp"
#include "gjr_moment_policy.hpp"
#include "power/egarch_precision_state.hpp"
#include "power/aparch_moment_policy.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace scivolapi::detail {
namespace moment_gen=vlk::experiment::generated;
enum class MomentUse {none,negative_second,absolute,wide_absolute,power};

template<unsigned Order,class Density>
class RetainedPowerBatch {
    using Values=moment_gen::PowerMomentValues<Density>;
    using Moment=moment_gen::AsymmetricPowerMoment<Density::shape_parameters>;
    std::span<typename Values::Entry> entries_;
    Values values_;
    moment_gen::PowerMomentBatchState<Order,Density> state_;
    const double* shape_;
    double delta_;
    const moment_gen::PowerMomentPrepared<Order,Density>* density_;
    std::span<Moment> results_;
    std::span<unsigned char> ready_;
public:
    RetainedPowerBatch(const double* shape,double delta,std::size_t lags,
        point_preparation::Arena& arena,const moment_gen::PowerMomentPrepared<Order,Density>* density=nullptr) noexcept
        :entries_(arena.template take<typename Values::Entry>(lags)),
         values_{entries_.data(),entries_.size()},shape_(shape),delta_(delta),density_(density),
         results_(arena.template take<Moment>(lags)),
         ready_(arena.template take<unsigned char>(lags)) {}
    RetainedPowerBatch(const RetainedPowerBatch&)=delete;
    RetainedPowerBatch& operator=(const RetainedPowerBatch&)=delete;
    bool valid() const noexcept {return !entries_.empty() && entries_.size()==results_.size() && results_.size()==ready_.size();}
    template<class Lags>
    Moment moment(double gamma,std::size_t slot,Lags lags) noexcept {
        if (!valid() || std::size_t(lags)!=results_.size() || slot>=results_.size()) return {};
        if (!ready_[slot]) {
            moment_gen::PowerMomentBatch<Order,Density,Lags> batch(shape_,delta_,lags,&values_,density_,&state_);
            const auto result=batch(gamma,slot);
            if (!result.valid) return result;
            results_[slot]=result;ready_[slot]=1;
        }
        return results_[slot];
    }
    template<class Lags>
    double value_error_scale(double gamma,double value,Lags lags) noexcept {
        moment_gen::PowerMomentBatch<Order,Density,Lags> batch(shape_,delta_,lags,&values_,density_,&state_);
        return batch.value_error_scale(gamma,value);
    }
};

template<class Density,unsigned Order>
MomentReservation power_reservation(std::size_t lags) noexcept {
    vlk::experiment::runtime::WorkspaceSizer size;
    if (!lags || !size.template add<typename moment_gen::PowerMomentValues<Density>::Entry>(lags)
        || !size.template add<moment_gen::AsymmetricPowerMoment<Density::shape_parameters>>(lags)
        || !size.template add<unsigned char>(lags)) return {SIZE_MAX,0,alignof(double)};
    using Payload=RetainedPowerBatch<Order,Density>;
    return {sizeof(Payload),size.size(),alignof(Payload)};
}
template<class Density>
MomentReservations moment_reservations(unsigned mask,MomentUse use,std::size_t lags=0) noexcept {
    MomentReservations result{};
    const auto reserve=[&]<unsigned Order>() {
        if (!(mask&(1u<<Order))) return;
        using Jet=moment_gen::AbsoluteMoment<Density::shape_parameters>;
        using Wide=vlk::experiment::runtime::arma_egarch::WideAbsoluteMoment<Density::shape_parameters>;
        if (use==MomentUse::negative_second)
            result[Order][unsigned(MomentProducer::negative_second)]={sizeof(Jet),0,alignof(Jet)};
        if (use==MomentUse::absolute)
            result[Order][unsigned(MomentProducer::absolute)]={sizeof(Jet),0,alignof(Jet)};
        if (use==MomentUse::wide_absolute)
            result[Order][unsigned(MomentProducer::wide_absolute)]={sizeof(Wide),0,alignof(Wide)};
        if (use==MomentUse::power) {
            result[Order][unsigned(MomentProducer::power_batch)]=power_reservation<Density,Order>(lags);
            using Precision=moment_gen::PowerMomentPrecisionCache;
            if constexpr (Order==0)
                result[Order][unsigned(MomentProducer::precision)]={sizeof(Precision),0,alignof(Precision)};
        }
    };
    reserve.template operator()<0>();reserve.template operator()<1>();reserve.template operator()<2>();
    if (mask && use==MomentUse::wide_absolute) {
        using Value=vlk::experiment::wide::Integral;
        result[0][unsigned(MomentProducer::wide_absolute_value)]={sizeof(Value),0,alignof(Value)};
    }
    return result;
}

template<unsigned Order,MomentProducer Kind,class Density,class Factory>
auto prepared_moment(DensityPreparation* cache,const double* shape,
    LocalDensityAxes axes,Factory&& factory) noexcept -> std::invoke_result_t<Factory&> {
    using Value=std::invoke_result_t<Factory&>;
    if (!cache || (cache->counts && cache->counts->cold_density)) return factory();
    if (!cache->template stable_shape<Density>(shape)) return {};
    axes.shapes&=(1u<<Density::shape_parameters)-1u;
    cache->moments.builds=cache->counts ? &cache->counts->moment_builds:nullptr;
    Value failure{};
    const auto* value=cache->moments.template get<Order,Kind,Value>(axes,[&](auto& target) noexcept {
        const auto* candidate=target.emplace_from(factory);
        if (!candidate->valid) failure=*candidate;
        return candidate->valid ? Status::ok:Status::numerical_range;
    });
    return value ? *value:failure;
}

template<unsigned Order,class Density,class Lease>
auto prepared_absolute(DensityPreparation* cache,const double* shape,LocalDensityAxes axes,Lease& lease) noexcept {
    return prepared_moment<Order,MomentProducer::absolute,Density>(cache,shape,axes,[&] {
        if constexpr (gh_density<Density> && !std::is_same_v<Density,moment_gen::VarianceGammaDensity>) {
            if constexpr (Order>=1)
                if (!(axes.shapes&((1u<<Density::shape_parameters)-1u)))
                    return moment_gen::absolute_moment<Order,Density>(shape,axes);
            const auto* prepared=lease.moment_prepared();
            return prepared ? moment_gen::egarch_moment_detail::integrate_prepared<Order,Density>(shape,*prepared,axes)
                :moment_gen::AbsoluteMoment<Density::shape_parameters>{};
        } else return moment_gen::absolute_moment<Order,Density>(shape,axes);
    });
}
template<unsigned Order,class Density>
auto prepared_negative_second(DensityPreparation* cache,const double* shape) noexcept {
    return prepared_moment<Order,MomentProducer::negative_second,Density>(cache,shape,{},[&] {
        if (!Density::valid_shape(shape)) return moment_gen::NegativeSecondMoment<Density::shape_parameters>{};
        return moment_gen::negative_second_moment_prepared<Order,Density>(shape,[&] {
            if constexpr (gh_density<Density>) {
                if (cache && !(cache->counts && cache->counts->cold_density)) {
                    const auto* prepared=cache->template moment_density<Density,Order>(shape);
                    return prepared ? moment_gen::gjr_moment_detail::integrate_prepared<Order,Density>(shape,*prepared)
                        :moment_gen::NegativeSecondMoment<Density::shape_parameters>{};
                }
            }
            return moment_gen::gjr_moment_detail::integrate<Order,Density>(shape);
        });
    });
}
template<unsigned Order,class Density,class Lease,class PointState>
auto prepared_wide_absolute(DensityPreparation* cache,const double* shape,LocalDensityAxes axes,
    Lease& lease,const PointState& point) noexcept {
    return prepared_moment<Order,MomentProducer::wide_absolute,Density>(cache,shape,axes,[&] {
        const auto value=[&] {
            if constexpr (requires {point.prepare_absolute_value(shape);}) {
                LocalDensityAxes value_axes{};value_axes.shapes=0;
                return prepared_moment<0,MomentProducer::wide_absolute_value,Density>(cache,shape,value_axes,
                    [&] {return point.prepare_absolute_value(shape);});
            } else return vlk::experiment::wide::Integral{0,0,0,false};
        };
        if constexpr (gh_density<Density> && requires {point.template absolute_moment_prepared_with_value<Order,Density>(shape,*lease.moment_prepared(),axes,value);}) {
            const auto* prepared=lease.moment_prepared();
            using Value=decltype(point.template absolute_moment_prepared_with_value<Order,Density>(shape,*prepared,axes,value));
            return prepared ? point.template absolute_moment_prepared_with_value<Order,Density>(shape,*prepared,axes,value):Value{};
        } else if constexpr (requires {point.prepare_absolute_value(shape);}) {
            return point.template absolute_moment_with_value<Order,Density>(shape,axes,value);
        } else return point.template absolute_moment<Order,Density>(shape,axes);
    });
}

struct PowerMomentPointLayout {std::size_t gamma=SIZE_MAX,delta=SIZE_MAX;};
template<unsigned Order,class Density,class Lags>
class PowerBatchLease {
    using Payload=RetainedPowerBatch<Order,Density>;
    moment_gen::PowerMomentBatch<Order,Density,Lags> cold_;
    Payload* prepared_=nullptr;
    bool cached_=false;
    [[no_unique_address]] Lags lags_;
    std::span<const double> gamma_;
public:
    PowerBatchLease(DensityPreparation* cache,const double* shape,double delta,Lags lags,
        moment_gen::PowerMomentValues<Density>* values=nullptr,PowerMomentPointLayout layout={}) noexcept
        :cold_(shape,delta,lags,values),cached_(cache && !(cache->counts && cache->counts->cold_density)),lags_(lags) {
        if (!cached_) return;
        const auto* stable=cache->template stable_shape<Density>(shape);
        const auto power=cache->point_block(layout.delta,1);
        gamma_=cache->point_block(layout.gamma,std::size_t(lags));
        if (!stable || power.empty() || std::memcmp(power.data(),&delta,sizeof(double))
            || gamma_.size()!=std::size_t(lags)) return;
        const moment_gen::PowerMomentPrepared<Order,Density>* density=nullptr;
        if constexpr (gh_density<Density>) {
            density=cache->template moment_density<Density,Order>(stable);
            if (!density) return;
        }
        cache->moments.builds=cache->counts ? &cache->counts->moment_builds:nullptr;
        prepared_=cache->moments.template get<Order,MomentProducer::power_batch,Payload>({},[&](auto& target) noexcept {
            const auto* candidate=target.emplace(stable,delta,lags,target.data(),density);
            return candidate->valid() ? Status::ok:Status::insufficient_capacity;
        });
    }
    auto operator()(double gamma,std::size_t lag) noexcept {
        using Value=moment_gen::AsymmetricPowerMoment<Density::shape_parameters>;
        if (!cached_) return cold_(gamma,lag);
        if (!prepared_ || lag>=gamma_.size() || std::memcmp(gamma_.data()+lag,&gamma,sizeof(double))) return Value{};
        return prepared_->moment(gamma,lag,lags_);
    }
    double value_error_scale(double gamma,double value) const noexcept {
        return prepared_ ? prepared_->value_error_scale(gamma,value,lags_):cold_.value_error_scale(gamma,value);
    }
};
template<class Density,class Layout>
auto prepared_power_persistence(DensityPreparation* cache,const Layout& layout,const double* theta,
    moment_gen::PowerMomentValues<Density>* memo=nullptr,std::span<const double> prepared_values={}) noexcept {
    if (!cache || (cache->counts && cache->counts->cold_density))
        return moment_gen::power_persistence<Density>(layout,theta,memo,prepared_values);
    if (!theta || !cache->template stable_shape<Density>(theta+layout.model_parameters))
        return moment_gen::PowerPersistence{};
    PowerBatchLease<0,Density,std::remove_cvref_t<decltype(layout.p)>> batch(
        cache,theta+layout.model_parameters,theta[layout.delta_index],layout.p,memo,{layout.gamma_start,layout.delta_index});
    using Precision=moment_gen::PowerMomentPrecisionCache;
    cache->moments.builds=cache->counts ? &cache->counts->moment_builds:nullptr;
    return moment_gen::power_persistence_prepared<Density>(layout,theta,batch,[&](auto&) noexcept {
        return cache->moments.template get<0,MomentProducer::precision,Precision>({},[](auto& target) noexcept {
            target.emplace();return Status::ok;
        });
    },prepared_values);
}
} // namespace scivolapi::detail
