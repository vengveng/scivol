// SCIVOL-NATIVE: solver-owned exact-point EGARCH value-state reuse.
#pragma once
#include "linear_power_program.hpp"
#include <cstring>
#include <span>

namespace vlk::experiment::runtime {

template<class Scalar>
struct EgarchRetainedPath {
    struct Observation {Scalar residual,log_variance,standardized,inverse_scale;};
    std::span<Observation> observations;
    std::span<double> theta,input;
    const double* y=nullptr;
    const double* x=nullptr;
    double h0=0;
    bool transformed=false,complete=false;

    bool ready(std::size_t parameters,std::size_t n) const noexcept {
        return observations.data() && observations.size()==n
            && theta.data() && theta.size()==parameters
            && input.data() && input.size()==parameters;
    }
    void begin(const arma_egarch::Layout& layout,const double* parameters,
        const double* data,std::size_t n,double initial_variance,const double* features) noexcept {
        complete=false;
        std::copy_n(parameters,layout.parameters,theta.data());
        y=data;x=features;h0=initial_variance;
    }
    bool matches(const arma_egarch::Layout& layout,const double* parameters,
        const double* data,std::size_t n,double initial_variance,const double* features) const noexcept {
        // A solve owns one immutable sample and layout. Both theta and the
        // optimizer input are checked: rounded coordinate maps need not be injective.
        return complete && y==data && x==features && h0==initial_variance
            && ready(layout.parameters,n)
            && std::memcmp(theta.data(),parameters,theta.size_bytes())==0;
    }
    bool matches_input(const double* point,bool coordinate_space) const noexcept {
        return complete && transformed==coordinate_space
            && std::memcmp(input.data(),point,input.size_bytes())==0;
    }
};

template<class Prepared>
struct EgarchMomentReference {
    using Scalar=typename Prepared::Scalar;
    const Prepared* moment;
    template<int Order,class Policy>
    auto absolute_moment(const double* shape) const noexcept {
        return moment->template absolute_moment<Order,Policy>(shape);
    }
    template<int Order,class Policy>
    auto density_preparation() const noexcept
        requires requires {moment->template density_preparation<Order,Policy>();} {
        return moment->template density_preparation<Order,Policy>();
    }
    template<class Policy,bool Value,bool Gradient,bool Hessian,bool Mean,class Preparation>
    auto prepare_density(const double* shape,std::size_t scored,Preparation& preparation) const noexcept {
        return moment->template prepare_density<Policy,Value,Gradient,Hessian,Mean>(shape,scored,preparation);
    }
    template<int Order,class Policy,class Preparation>
    auto absolute_moment(const double* shape,Preparation& preparation) const noexcept {
        return moment->template absolute_moment<Order,Policy>(shape,preparation);
    }
};

template<class Prepared>
struct EgarchRecordingState:EgarchMomentReference<Prepared> {
    using Scalar=typename Prepared::Scalar;
    EgarchRetainedPath<Scalar>* path;
    void begin_recursion(const arma_egarch::Layout& layout,const double* theta,
        const double* y,std::size_t n,double h0,const double* x) const noexcept {
        path->begin(layout,theta,y,n,h0,x);
    }
    void observe_recursion(std::size_t time,Scalar residual,Scalar log_variance,
        Scalar standardized,Scalar inverse_scale) const noexcept {
        path->observations[time]={residual,log_variance,standardized,inverse_scale};
    }
};

template<class Prepared>
struct EgarchReplayState:EgarchMomentReference<Prepared> {
    using Scalar=typename Prepared::Scalar;
    const EgarchRetainedPath<Scalar>* path;
    bool valid_recursion(const arma_egarch::Layout& layout,const double* theta,
        const double* y,std::size_t n,double h0,const double* x) const noexcept {
        return path->matches(layout,theta,y,n,h0,x);
    }
    void restore_recursion(std::size_t time,Scalar& residual,Scalar& log_variance,
        Scalar& standardized,Scalar& inverse_scale) const noexcept {
        const auto& state=path->observations[time];
        residual=state.residual;log_variance=state.log_variance;
        standardized=state.standardized;inverse_scale=state.inverse_scale;
    }
};

template<class Base>
struct RetainedEgarchProgram:Base {
    using Prepared=typename Base::point_state_type;
    using Path=EgarchRetainedPath<typename Prepared::Scalar>;
    static_assert(!Base::aparch && !std::is_same_v<typename Prepared::Scalar,double>);
    mutable Path paths[2];
    mutable unsigned accepted=0,trial=1;

    bool plan_retention(WorkspaceSizer& size,std::size_t n) const noexcept {
        if (!n) return false;
        for (unsigned slot=0;slot<2;++slot)
            if (!size.add<typename Path::Observation>(n)
                || !size.add<double>(this->layout.parameters)
                || !size.add<double>(this->layout.parameters)) return false;
        return true;
    }
    bool bind_retention(WorkspaceArena& arena,std::size_t n) noexcept {
        accepted=0;trial=1;
        for (auto& path:paths) {
            path={};
            auto* observations=arena.take<typename Path::Observation>(n);
            auto* theta=arena.take<double>(this->layout.parameters);
            auto* input=arena.take<double>(this->layout.parameters);
            if (!n || !observations || !theta || !input) return false;
            path.observations={observations,n};
            path.theta={theta,this->layout.parameters};
            path.input={input,this->layout.parameters};
        }
        return true;
    }

    template<class Orders>
    int evaluate(const double* theta,const double* y,const double* x,std::size_t n,
        double h0,unsigned mask,bool transformed,void* work,std::size_t bytes,
        double* f,double* g,double* h,double* residuals,double* variances,
        bool checked=true) const noexcept {
        if ((mask!=1 && mask!=2) || checked || residuals || variances
            || !paths[0].ready(this->layout.parameters,n)
            || !paths[1].ready(this->layout.parameters,n))
            return Base::template evaluate<Orders>(theta,y,x,n,h0,mask,transformed,
                work,bytes,f,g,h,residuals,variances,checked);
        const auto run=[&](const auto& state) {
            unsigned valid=0;
            return arma_egarch::evaluate<typename Base::density_type,
                typename Base::Shape,Orders,false>(this->layout,theta,
                this->layout.parameters,y,n,h0,transformed,mask,work,bytes,
                f,g,h,&valid,x,nullptr,nullptr,state);
        };
        const EgarchMomentReference<Prepared> state{&this->egarch_point_state};
        if (mask==1) {
            auto& path=paths[trial];
            path.complete=false;
            const auto status=run(EgarchRecordingState<Prepared>{state,&path});
            if (status==VLK_JET_OK) {
                std::copy_n(theta,this->layout.parameters,path.input.data());
                path.transformed=transformed;
                path.complete=true;
            }
            return status;
        }
        for (unsigned slot:{trial,accepted}) {
            if (!theta || !paths[slot].matches_input(theta,transformed)) continue;
            const auto status=run(EgarchReplayState<Prepared>{state,&paths[slot]});
            if (status==VLK_JET_OK && slot==trial) std::swap(accepted,trial);
            return status;
        }
        return Base::template evaluate<Orders>(theta,y,x,n,h0,mask,transformed,
            work,bytes,f,g,h,residuals,variances,false);
    }
};

} // namespace vlk::experiment::runtime
