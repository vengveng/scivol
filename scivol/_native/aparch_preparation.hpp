// SCIVOL-NATIVE: PROTOTYPE; optional preparation, never a final-fit certificate.
#pragma once
#include "aparch_parameter_rows.hpp"
#include "bounded_direction.hpp"
#include "optimizer_controller.hpp"
#include "observed_scaled_objective.hpp"
#include "exact_request_reuse.hpp"

namespace vlk::experiment::runtime::aparch::preparation {
namespace range=vlk::experiment::bounded_direction;
namespace at=aparch_transform;
using Narrow=FixedJointOrders<1,1,1,1>;
using Wide=FixedJointOrders<2,1,3,2>;

enum class Strategy : std::size_t { kraft_equality_or_box=0, box_full_controller_reuse=1 };
inline constexpr std::size_t controller_reuse_buckets=128;

template<class Inner> concept ControllerObjective = requires(
    const Inner& inner,const double* point,double* value,double* gradient,double* hessian,
    request_reuse::Bank* cache) {
    inner.with_request_reuse(cache);
    inner.scaled(1.0);
    inner.evaluate_curvature(point,value,gradient,hessian);
    inner.scientific_kkt(point,gradient,hessian,1e-6);
    inner.boundary_reached(point,gradient,1e-6);
};

struct Request {
    std::size_t ar,ma,p,q,density,force_runtime_map;
    std::size_t width,count;
    double initial_variance,power,accuracy;
    std::int64_t iterations;
    Strategy strategy=Strategy::kraft_equality_or_box;
};
struct Result {
    double continuation_value=std::numeric_limits<double>::infinity();
    double best_value=std::numeric_limits<double>::infinity();
    double equality_error=std::numeric_limits<double>::infinity();
    std::int64_t mode=-1,iterations=0,objective_dispatch=-1;
    std::size_t values=0,gradients=0,hessians=0,constraints=0,jacobians=0;
    std::size_t map_dispatch=0,independent_power=0,eligible=0,retained_feasible_trial=0;
    Strategy strategy=Strategy::kraft_equality_or_box;
    kraft::Result solver;
    double unrestricted_coordinate_kkt=std::numeric_limits<double>::infinity();
    std::size_t refinement_restarts=0,curvature_steps=0,scientific_projection_steps=0;
    std::size_t reuse_queries=0,reuse_hits=0;
};

template<class D,class S,class Call> auto orders(const Request& request,Call call) noexcept {
    if (!request.force_runtime_map) {
        if (request.ar==1 && request.ma==1 && request.p==1 && request.q==1)
            return call.template operator()<D,S,Narrow>();
        if (request.ar==2 && request.ma==1 && request.p==3 && request.q==2)
            return call.template operator()<D,S,Wide>();
    }
    return call.template operator()<D,S,RuntimeOrders>();
}
template<class Call> auto dispatch(const Request& request,Call call) noexcept {
    using Return=decltype(orders<generated::NormalDensity,generated::NoShapeTransform>(request,call));
#define VLK_PREPARATION_DENSITY(CODE,DENSITY,SHAPE) \
    case CODE: return orders<generated::DENSITY,generated::SHAPE>(request,call)
    switch (request.density) {
    VLK_PREPARATION_DENSITY(VLK_GARCH_NORMAL,NormalDensity,NoShapeTransform);
    VLK_PREPARATION_DENSITY(VLK_GARCH_STUDENTT,StudentTDensity,StudentTShapeTransform);
    VLK_PREPARATION_DENSITY(VLK_GARCH_SKEWT,SkewTDensity,SkewTShapeTransform);
    VLK_PREPARATION_DENSITY(VLK_GARCH_GED,GedDensity,GedShapeTransform);
    VLK_PREPARATION_DENSITY(VLK_GARCH_GH,GeneralizedHyperbolicDensity,GhShapeTransform);
    VLK_PREPARATION_DENSITY(VLK_GARCH_NIG,NormalInverseGaussianDensity,FixedLambdaGhShapeTransform);
    VLK_PREPARATION_DENSITY(VLK_GARCH_HYPERBOLIC,HyperbolicDensity,FixedLambdaGhShapeTransform);
    VLK_PREPARATION_DENSITY(VLK_GARCH_VARIANCE_GAMMA,VarianceGammaDensity,VarianceGammaShapeTransform);
    default:
        if constexpr (std::is_signed_v<Return>) return Return{-1};
        else return Return{};
    }
#undef VLK_PREPARATION_DENSITY
}
inline bool layout(const Request* request,Layout& result) noexcept {
    range::Range typed_request;
    if (!request || reinterpret_cast<std::uintptr_t>(request)%alignof(Request)
        || !range::memory_range(request,sizeof(*request),typed_request)
        || request->density>VLK_GARCH_VARIANCE_GAMMA || request->force_runtime_map>1
        || static_cast<std::size_t>(request->strategy)>1
        || (request->strategy==Strategy::box_full_controller_reuse
            && (request->density==VLK_GARCH_STUDENTT || request->density==VLK_GARCH_SKEWT))) return false;
    return dispatch(*request,[&]<class D,class S,class O>() {
        return make_layout<D>(request->ar,request->ma,request->p,request->q,true,result);
    });
}
template<class D,class S,class O> std::size_t workspace(const Layout& supplied,Strategy strategy) noexcept {
    constexpr auto fixed=O::fixed ? &fixed_layout<O,D,true> : nullptr;
    const auto& l=at::evaluation_layout<fixed>(supplied);
    constexpr std::size_t rows=at::tail_coupled_delta<D>?1:0;
    const bool full=strategy==Strategy::box_full_controller_reuse;
    const auto solver=full ? controller::workspace_bytes(l.parameters)
                          : runtime_kraft_workspace_bytes(l.parameters,rows,rows);
    const auto map=at::ParameterRows<D,S,O>::bytes(l,true);
    WorkspaceSizer size;
    if (!solver || !map || !size.add<std::byte>(solver) || !size.add<std::byte>(map)) return 0;
    for (unsigned i=0;i<3;++i) if (!size.add<double>(l.parameters)) return 0;
    if (full) {
        const auto cache=request_reuse::Bank::bytes(l.parameters,controller_reuse_buckets);
        if (!cache || !size.add<std::byte>(cache)) return 0;
    }
    return size.size();
}
inline std::size_t workspace_bytes(const Request* request) noexcept {
    Layout l;
    return layout(request,l) ? dispatch(*request,[&]<class D,class S,class O>() {return workspace<D,S,O>(l,request->strategy);}) : 0;
}

struct RetainedValues {
    std::size_t width;
    double* best;
    Result& result;
    void request(bool f,bool g,bool h) const noexcept {
        result.values+=f;result.gradients+=g;result.hessians+=h;
    }
    void retain(const double* point,double value) const noexcept {
        if (value<result.best_value) {
            result.best_value=value;
            std::copy_n(point,width,best);
        }
    }
};

// Inner objectives return the additive canonical NLL. Scaling is owned here.
template<class Inner,class Map> struct RetainedObjective {
    const Inner& inner;
    double scale;
    double* best;
    Result& result;
    Map& map;
    std::size_t selected;
    mutable double additive_value=std::numeric_limits<double>::infinity();
    std::size_t dimension() const noexcept {return map.dimension();}
    bool evaluate(kraft::ObjectiveRequest request,const double* point,double* value,double* gradient,bool same) const noexcept {
        const bool f=request!=kraft::ObjectiveRequest::gradient;
        const bool g=request!=kraft::ObjectiveRequest::value;
        result.values+=f;result.gradients+=g;
        const bool valid=inner.evaluate(request,point,value,gradient,same);
        if (!valid) return false;
        (void)map.prepare(point,false,&selected,1,{});
        if (f) {
            if (!std::isfinite(*value)) return false;
            additive_value=*value;
            RetainedValues{dimension(),best,result}.retain(point,*value);
            *value*=scale;
        }
        if (g) for (std::size_t i=0;i<dimension();++i) gradient[i]*=scale;
        return true;
    }
};
template<class Inner> struct CountedConstraints {
    const Inner& inner;
    Result& result;
    std::size_t dimension() const noexcept {return inner.dimension();}
    std::size_t count() const noexcept {return inner.count();}
    std::size_t equalities() const noexcept {return inner.equalities();}
    bool evaluate(kraft::ConstraintRequest request,const double* point,double* values,double* jacobian,
                  ConstraintColumns columns={}) const noexcept {
        result.constraints+=request!=kraft::ConstraintRequest::jacobian;
        result.jacobians+=request!=kraft::ConstraintRequest::value;
        return inner.evaluate(request,point,values,jacobian,columns);
    }
};
struct NoEqualities {
    std::size_t width;
    std::size_t dimension() const noexcept {return width;}
    std::size_t count() const noexcept {return 0;}
    std::size_t equalities() const noexcept {return 0;}
    bool evaluate(kraft::ConstraintRequest,const double*,double*,double*,ConstraintColumns={}) const noexcept {return true;}
};

template<class D,class S,class O,class Inner>
int run(const Request& request,const Layout& supplied,const Inner& inner,
    const double* incoming,const double* lower,const double* upper,
    void* memory,std::size_t bytes,double* continuation,double* best,Result& result) noexcept {
    constexpr auto fixed=O::fixed ? &fixed_layout<O,D,true> : nullptr;
    constexpr bool coupled=at::tail_coupled_delta<D>;
    const auto& l=at::evaluation_layout<fixed>(supplied);
    const bool full=request.strategy==Strategy::box_full_controller_reuse;
    if (full && (coupled || !ControllerObjective<Inner>)) return -1;
    const auto solver_bytes=full ? controller::workspace_bytes(l.parameters)
        : runtime_kraft_workspace_bytes(l.parameters,coupled?1:0,coupled?1:0);
    const auto map_bytes=at::ParameterRows<D,S,O>::bytes(l,true);
    WorkspaceArena arena(memory,bytes);
    auto* solver=arena.take<std::byte>(solver_bytes);
    auto* map_memory=arena.take<std::byte>(map_bytes);
    auto* point=arena.take<double>(l.parameters);
    auto* lo=arena.take<double>(l.parameters);
    auto* hi=arena.take<double>(l.parameters);
    if (!solver || !map_memory || !point || !lo || !hi) return -1;
    result={};
    result.strategy=request.strategy;
    result.map_dispatch=O::fixed ? (O::p_ar==1?1:2) : 0;
    result.independent_power=!coupled;
    std::copy_n(incoming,l.parameters,point);
    std::copy_n(lower,l.parameters,lo);std::copy_n(upper,l.parameters,hi);
    at::ParameterRows<D,S,O> map(l,map_memory,map_bytes);
    const std::size_t selected=l.delta_index;
    const double scale=1./static_cast<double>(request.count-l.maximum_lag);
    const MappedParameterEqualities equality{map,&selected,&request.power,std::size_t{1}};
    const auto execute=[&](const auto& objective,const auto& solve) {
        double value=0.;
        (void)objective.evaluate(kraft::ObjectiveRequest::value,point,&value,nullptr,false);
        if (!at::ParameterRows<D,S,O>::retarget_power(l,point,request.power)) {result.mode=-2;return -2;}
        for (std::size_t i=0;i<l.parameters;++i)
            if (point[i]<lo[i] || point[i]>hi[i]) {result.mode=-2;return -2;}
        // Every density except Student-t/Skew-t has an independent power coordinate.
        if constexpr (!coupled) lo[l.delta_index]=hi[l.delta_index]=point[l.delta_index];
        (void)objective.evaluate(kraft::ObjectiveRequest::value,point,&value,nullptr,false);
        const auto solved=solve(objective);
        result.solver=solved;
        // Unrestricted diagnostics control policy, not a held-power certificate.
        result.mode=full ? kraft::retained_point_mode : solved.mode;
        result.iterations=solved.iterations;
        result.retained_feasible_trial=solved.retained_feasible_trial;
        if (objective.evaluate(kraft::ObjectiveRequest::value,point,&value,nullptr,false)) {
            result.continuation_value=objective.additive_value;
            std::copy_n(point,l.parameters,continuation);
            double error=0.;
            const bool mapped=[&] {
                if constexpr (coupled) return CountedConstraints{equality,result}.evaluate(kraft::ConstraintRequest::value,point,&error,nullptr);
                return map.prepare(point,false,&selected,1,{}) && (error=map.value(selected)-request.power,std::isfinite(error));
            }();
            if (mapped) result.equality_error=std::abs(error);
            result.eligible=mapped && result.equality_error<=1e-10 && solved.maximum_violation<=1e-10;
        }
        return std::isfinite(result.best_value)?0:-2;
    };
    if constexpr (!coupled && ControllerObjective<Inner>) {
        if (full) {
            const auto cache_bytes=request_reuse::Bank::bytes(l.parameters,controller_reuse_buckets);
            auto* cache_memory=arena.take<std::byte>(cache_bytes);
            request_reuse::Bank cache;
            if (!cache_memory || !cache.bind(l.parameters,controller_reuse_buckets,cache_memory,cache_bytes)) return -1;
            const controller::ObservedScaledObjective objective{
                inner.with_request_reuse(&cache),scale,RetainedValues{l.parameters,best,result}};
            const int status=execute(objective,[&](const auto& observed) {
                return controller::minimize_objective(observed,point,lo,hi,
                    request.accuracy,request.iterations,solver,solver_bytes,
                    result.unrestricted_coordinate_kkt,result.refinement_restarts,
                    result.curvature_steps,result.scientific_projection_steps);
            });
            result.reuse_queries=cache.queries();result.reuse_hits=cache.hits();
            return status;
        }
    }
    RuntimeKraftWorkspace scratch;
    if (!bind_runtime_kraft_workspace(l.parameters,coupled?1:0,solver,solver_bytes,scratch,coupled?1:0)) return -1;
    const RetainedObjective objective{inner,scale,best,result,map,selected};
    return execute(objective,[&](const auto& observed) {
        const auto solve=[&](const auto& constraints) {
            return minimize_runtime_kraft(observed,constraints,point,{lo,hi},
                {request.accuracy,request.iterations,false},scratch);
        };
        if constexpr (coupled) return solve(CountedConstraints{equality,result});
        else return solve(NoEqualities{l.parameters});
    });
}

// Validates all borrowed ranges before reading the sample or writing any output.
template<class Inner,class Validate,class JetBytes>
int run_checked(const Request* request,const Inner& inner,Validate validate,
    const double* incoming,const double* observations,const double* lower,const double* upper,
    void* memory,std::size_t bytes,void* jet,std::size_t jet_bytes,JetBytes required_jet,
    double* continuation,std::size_t continuation_count,double* best,std::size_t best_count,Result* output,
    const void* provider=nullptr,std::size_t provider_bytes=0) noexcept {
    Layout l;
    const auto needed=workspace_bytes(request);
    if (!needed || !layout(request,l) || request->width!=l.parameters || inner.dimension()!=l.parameters
        || request->count<=l.maximum_lag || !std::isfinite(request->initial_variance) || request->initial_variance<=0.
        || !std::isfinite(request->power) || request->power<=0. || !std::isfinite(request->accuracy)
        || request->accuracy<=0. || request->iterations<=0 || bytes<needed
        || continuation_count!=l.parameters || best_count!=l.parameters
        || (request->strategy==Strategy::box_full_controller_reuse && !ControllerObjective<Inner>)
        || reinterpret_cast<std::uintptr_t>(memory)%16 || reinterpret_cast<std::uintptr_t>(jet)%16) return -1;
    std::size_t point_bytes=0,data_bytes=0;
    if (!checked_product(l.parameters,sizeof(double),point_bytes) || !checked_product(request->count,sizeof(double),data_bytes)) return -1;
    const void* pointers[]={request,incoming,observations,lower,upper,memory,jet,continuation,best,output,provider};
    const std::size_t sizes[]={sizeof(*request),point_bytes,data_bytes,point_bytes,point_bytes,bytes,jet_bytes,point_bytes,point_bytes,sizeof(*output),provider_bytes};
    range::Range ranges[11];
    for (std::size_t i=0;i<(provider_bytes?11:10);++i) {
        if (reinterpret_cast<std::uintptr_t>(pointers[i])%alignof(double) || !range::memory_range(pointers[i],sizes[i],ranges[i])) return -1;
        for (std::size_t j=0;j<i;++j) if (range::intersects(ranges[i],ranges[j])) return -1;
    }
    const auto needed_jet=required_jet();
    if (!needed_jet || jet_bytes<needed_jet) return -1;
    if (!runtime_kraft_detail::valid_bounds(l.parameters,{lower,upper})) return -1;
    for (std::size_t i=0;i<l.parameters;++i)
        if (!std::isfinite(incoming[i]) || incoming[i]<lower[i] || incoming[i]>upper[i]) return -1;
    if (!validate()) return -1;
    return dispatch(*request,[&]<class D,class S,class O>() {
        return run<D,S,O>(*request,l,inner,incoming,lower,upper,memory,bytes,continuation,best,*output);
    });
}

} // namespace vlk::experiment::runtime::aparch::preparation
