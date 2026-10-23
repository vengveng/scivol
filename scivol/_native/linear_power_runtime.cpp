// SCIVOL-NATIVE: linear EGARCH/APARCH program and typed solver; promotion pending.
#include "linear_joint_api.hpp"
#include "linear_power_program.hpp"
#include "egarch_precision_state.hpp"
#include "egarch_retained_program.hpp"
#include "gjr_transform_policy.hpp"
#include "mean_solver.hpp"
#include "affine_optimizer_constraints.hpp"
#include "power_solver_constraints.hpp"
#include "power_constraint_curvature.hpp"
#include "native_active_face_phase.hpp"
#include "restricted_optimizer_coordinates.hpp"

namespace {
using namespace vlk::experiment;
using runtime::PowerConstraints;

template<class Density>
struct EgarchPointState {using type=runtime::arma_egarch::WideMomentPointState<Density>;};
template<>
struct EgarchPointState<generated::NormalDensity> {
    using type=runtime::arma_egarch::Binary64PointState;
};
template<class Family>
struct EgarchPointState<generated::GhFamilyDensity<Family>> {
    using type=runtime::arma_egarch::GhFamilyPointState<Family>;
};
template<>
struct EgarchPointState<generated::NormalInverseGaussianDensity> {
    using type=runtime::arma_egarch::NormalInverseGaussianPointState;
};
template<>
struct EgarchPointState<generated::VarianceGammaDensity> {
    using type=runtime::arma_egarch::VarianceGammaPointState;
};

template<class Density,bool Aparch>
using Program=runtime::LinearPowerProgram<Density,Aparch,
    std::conditional_t<Aparch,runtime::arma_egarch::Binary64PointState,
        typename EgarchPointState<Density>::type>>;

template<class Function>
auto dispatch(bool aparch,unsigned density,Function function)
{
    using namespace generated;
    switch(density) {
#define D(C,T) case C: return aparch ? function(Program<T,true>{}) : function(Program<T,false>{})
        D(0,NormalDensity);D(1,StudentTDensity);D(2,SkewTDensity);D(3,GedDensity);
        D(4,GeneralizedHyperbolicDensity);D(5,NormalInverseGaussianDensity);
        D(6,HyperbolicDensity);D(7,VarianceGammaDensity);
#undef D
    }
    return decltype(function(Program<NormalDensity,true>{})){};
}

template<class Function>
auto dispatch_solve(bool aparch,unsigned density,Function function)
{
    return dispatch(aparch,density,[&](auto program) {
        using Base=decltype(program);
        // Binary64 Normal keeps its measured streaming plan. Standalone jets
        // and APARCH do not acquire sample-length retention buffers.
        if constexpr (!Base::aparch
            && !std::is_same_v<typename Base::point_state_type::Scalar,double>)
            return function(runtime::RetainedEgarchProgram<Base>{});
        else return function(program);
    });
}

std::size_t power_bytes(bool aparch,unsigned density,std::size_t columns,
    std::size_t p,std::size_t q,unsigned mask,bool transformed)
{
    return dispatch(aparch,density,[&](auto program)->std::size_t {
        return program.prepare(columns,p,q) ? program.bytes(mask,transformed) : 0u;
    });
}

int power_evaluate(bool aparch,unsigned density,std::size_t columns,
    std::size_t p,std::size_t q,const double* theta,const double* y,const double* x,
    std::size_t n,double h0,unsigned mask,bool transformed,bool fixed,void* work,
    std::size_t size,double* f,double* g,double* h,double* residuals,double* variances)
{
    if (density>=8) return VLK_JET_INVALID_ARGUMENT;
    return dispatch(aparch,density,[&](auto program)->int {
        if (!program.prepare(columns,p,q)) return VLK_JET_INVALID_ARGUMENT;
        const auto run=[&]<class Orders>() {
            return program.template evaluate<Orders>(theta,y,x,n,h0,mask,transformed,
                work,size,f,g,h,residuals,variances,false);
        };
        if (columns==0)
            return run.template operator()<mean::LinearVolatilityOrders<0>>();
        if (fixed && columns==4 && p==1 && q==1)
            return run.template operator()<mean::LinearVolatilityOrders<4,1,1>>();
        if (fixed && columns==7 && p==3 && q==2)
            return run.template operator()<mean::LinearVolatilityOrders<7,3,2>>();
        return run.template operator()<mean::LinearVolatilityOrders<>>();
    });
}

bool power_map(bool aparch,unsigned density,std::size_t columns,
    std::size_t p,std::size_t q,const double* source,double* destination,
    bool forward,bool fixed,void* work,std::size_t size)
{
    return dispatch(aparch,density,[&](auto program)->bool {
        if (!program.prepare(columns,p,q)) return false;
        const auto run=[&]<class Orders>() {
            return program.template map<Orders>(source,destination,forward,work,size);
        };
        if (fixed && columns==4 && p==1 && q==1)
            return run.template operator()<mean::LinearVolatilityOrders<4,1,1>>();
        if (fixed && columns==7 && p==3 && q==2)
            return run.template operator()<mean::LinearVolatilityOrders<7,3,2>>();
        return run.template operator()<mean::LinearVolatilityOrders<>>();
    });
}

template<class Program,class Orders>
struct JointObjective {
    static constexpr auto fixed_layout=Program::template fixed_layout<Orders>;
    Program program;
    const double* y;
    const double* x;
    std::size_t n;
    double h0;
    void* work;
    std::size_t bytes;
    mean::PolynomialConstraintPlan mean;
    double* values;
    void* domain_work;
    std::size_t domain_bytes;
    bool fused_trials;
    bool transformed;
    mutable std::int64_t requests[3]{};
    std::size_t dimension() const noexcept
    {
        if constexpr (fixed_layout==nullptr) return program.layout.parameters;
        else return fixed_layout->parameters;
    }
    bool speculate_trial_gradient(std::int64_t) const noexcept {return fused_trials;}
    bool evaluate(kraft::ObjectiveRequest request,const double* z,double* f,double* g,bool) const noexcept
    {
        unsigned mask=request==kraft::ObjectiveRequest::value ? 1u :
            request==kraft::ObjectiveRequest::gradient ? 2u : 3u;
        ++requests[mask-1];
        if (mean.domain.order && !mean.domain.evaluate(z+mean.offset,values,nullptr,
            domain_work,domain_bytes,true)) return false;
        if (program.template evaluate<Orders>(z,y,x,n,h0,mask,transformed,work,bytes,
            f,g,nullptr,nullptr,nullptr,false)!=VLK_JET_OK) return false;
        if (mask&1) *f/=static_cast<double>(n);
        if (mask&2) for (std::size_t i=0;i<dimension();++i) g[i]/=static_cast<double>(n);
        return true;
    }
};



template<class Program>
std::size_t solve_bytes(const Program& program, mean::PolynomialDomain domain,
                        bool transformed, std::size_t affine = 0u,std::size_t n=0u)
{
    const auto k=program.layout.parameters;
    const auto kernel=program.bytes(3u,transformed);
    std::size_t rows,jacobian,full_jacobian;
    if (!runtime::checked_product(domain.order,2u,rows)
        || !runtime::checked_product(rows,domain.parameters,jacobian)
        || !runtime::checked_sum(rows,transformed?0u:program.theta_constraints(),rows)
        || !runtime::checked_product(rows,k,full_jacobian)) return 0;
    std::size_t solver_rows;
    if (!runtime::checked_sum(rows,affine,solver_rows)) return 0;
    const auto solver=runtime::runtime_kraft_workspace_bytes(k,solver_rows,affine);
    const auto restrictions=std::max(domain.order?domain.workspace_bytes(true):0u,
        mean::PolynomialDomain{program.q(),program.q(),nullptr}.workspace_bytes(true));
    if (!kernel || !solver || (domain.order && !restrictions)) return 0;
    runtime::WorkspaceSizer size;
    if (!(size.add<std::byte>(kernel) && size.add<std::byte>(solver)
        && size.add<double>(k) && size.add<double>(jacobian) && size.add<double>(rows)
        && size.add<std::byte>(restrictions) && size.add<double>(full_jacobian)
        && size.add<double>(rows))) return 0;
    if constexpr (Program::reuse_moments)
        if (!size.add<typename Program::MomentValues::Entry>(program.layout.p)) return 0;
    if constexpr (requires {program.plan_retention(size,n);})
        if (!program.plan_retention(size,n)) return 0;
    return size.size();
}
template<class Program,class Solver=runtime::FullCoordinateKraft>
bool fit_program(const scivol::linear_joint::Plan& p,
    const scivol::linear_joint::Solve& s,Program program,const Solver& solve={})
{
    if (!program.prepare(p.columns,p.p,p.q)) return false;
    if (!s.workspace || reinterpret_cast<std::uintptr_t>(s.workspace)
        %alignof(std::max_align_t)) return false;
    const runtime::AffineEquality affine{s.mean_normal,s.mean_normal?p.columns:0u,s.mean_target,
        s.mean_rows,s.mean_targets};
    if (!affine.valid(program.layout.parameters)) return false;
    const mean::PolynomialConstraintPlan polynomial{
        {s.mean.order,s.mean.width,s.mean.basis},s.mean.offset};
    const auto required=solve_bytes(program,polynomial.domain,s.transformed,affine.count(),p.n);
    if (!required || s.bytes<required) return false;
    const auto k=program.layout.parameters;
    const auto rows=2*s.mean.order+(s.transformed?0u:program.theta_constraints());
    const auto kernel_bytes=program.bytes(3u,s.transformed);
    const auto solver_rows=rows+affine.count();
    const auto solver_bytes=runtime::runtime_kraft_workspace_bytes(k,solver_rows,affine.count());
    const auto domain_bytes=std::max(s.mean.order?polynomial.domain.workspace_bytes(true):0u,
        mean::PolynomialDomain{p.q,p.q,nullptr}.workspace_bytes(true));
    runtime::WorkspaceArena arena(s.workspace,s.bytes);
    void* kernel=arena.take<std::byte>(kernel_bytes);
    void* solver=arena.take<std::byte>(solver_bytes);
    double* scaled_point=arena.take<double>(k);
    double* jacobian=arena.take<double>(2*s.mean.order*s.mean.width);
    double* values=arena.take<double>(rows);
    void* domain=arena.take<std::byte>(domain_bytes);
    double* local_jacobian=arena.take<double>(rows*k);
    double* local_values=arena.take<double>(rows);
    typename decltype(program)::MomentValues moments;
    if constexpr (decltype(program)::reuse_moments) {
        moments.entries=arena.take<typename decltype(program)::MomentValues::Entry>(p.p);
        moments.slots=p.p;
        if (!moments.reset()) return false;
        program.moment_values=&moments;
    }
    if constexpr (requires {program.bind_retention(arena,p.n);})
        if (!program.bind_retention(arena,p.n)) return false;
    runtime::RuntimeKraftWorkspace state;
    if (!runtime::bind_runtime_kraft_workspace(solve.dimension(k),solver_rows,solver,solver_bytes,state,affine.count())) return false;
    const auto run=[&]<class Orders>() {
        using Objective=JointObjective<Program,Orders>;
        const Objective objective{program,s.y,s.x,p.n,
            s.h0,kernel,kernel_bytes,polynomial,values,domain,domain_bytes,s.speculate,s.transformed};
        const PowerConstraints<Program,Objective::fixed_layout> constraints{program,
            mean::PolynomialConstraints{objective.dimension(),{polynomial,{}},
                jacobian,domain,domain_bytes},
            s.transformed,local_values,local_jacobian};
        const runtime::AffineEqualityPrefix restricted{constraints,affine};
        const auto result=solve(
            runtime::DiagonalObjective{objective,s.scales,scaled_point},
            runtime::DiagonalConstraints{restricted,s.scales,scaled_point},s.point,
            {s.lower,s.upper},{s.accuracy,s.iterations,s.curvature_reset,
                s.minimum_invalid_step?kraft::InvalidTrialPolicy::minimum_step
                                      :kraft::InvalidTrialPolicy::interpolate},state);
        if (s.reset_count) *s.reset_count=result.constraint_curvature_resets;
        for (std::size_t j=0;j<k;++j) s.point[j]*=s.scales[j];
        s.stats[0]=result.mode;s.stats[1]=result.iterations;
        s.stats[2]=result.function_evaluations;s.stats[3]=result.gradient_evaluations;
        s.stats[4]=result.constraint_evaluations;
        s.stats[5]=Orders::fixed && decltype(constraints)::fixed
            ?2:(Orders::columns!=mean::dynamic?1:0);
        const bool mapped=s.transformed
            ? program.template map<Orders>(s.point,s.theta,true,kernel,kernel_bytes)
            : (std::copy_n(s.point,k,s.theta),true);
        if (s.requests) std::copy_n(objective.requests,3,s.requests);
        if (!mapped) return false;
        ++s.stats[2];++s.stats[3];
        if (s.requests) ++s.requests[2];
        return program.template evaluate<Orders>(s.theta,s.y,s.x,p.n,s.h0,
            3u,false,kernel,kernel_bytes,s.value,s.gradient,nullptr,nullptr,nullptr,false)==VLK_JET_OK;
    };
    if (p.columns==0)
        return run.template operator()<mean::LinearVolatilityOrders<0>>();
    if (p.fixed && p.columns==4 && p.p==1 && p.q==1)
        return run.template operator()<mean::LinearVolatilityOrders<4,1,1>>();
    if (p.fixed && p.columns==7 && p.p==3 && p.q==2)
        return run.template operator()<mean::LinearVolatilityOrders<7,3,2>>();
    return run.template operator()<mean::LinearVolatilityOrders<>>();
}

struct RefinementWorkspacePlan {
    std::size_t kernel=0,mean_jacobian=0,rows=0,jacobian=0;
    std::size_t domain=0,curvature=0,phase=0,total=0;
};

template<class Program>
RefinementWorkspacePlan refinement_plan(const Program& program,
    const scivol::linear_joint::Domain& mean_domain,std::size_t columns)
{
    RefinementWorkspacePlan plan;
    if (mean_domain.offset>columns || mean_domain.width>columns-mean_domain.offset
        || (mean_domain.order && (!mean_domain.width || !mean_domain.basis))) return plan;
    const mean::PolynomialConstraintPlan polynomial{
        {mean_domain.order,mean_domain.width,mean_domain.basis},mean_domain.offset};
    const auto k=program.layout.parameters;
    if (!runtime::checked_product(mean_domain.order,2u,plan.rows)
        || !runtime::checked_product(plan.rows,mean_domain.width,plan.mean_jacobian)
        || !runtime::checked_sum(plan.rows,program.theta_constraints(),plan.rows)
        || !runtime::checked_product(plan.rows,k,plan.jacobian)) return {};
    const PowerConstraints<Program> domain{program,
        mean::PolynomialConstraints{k,{polynomial,{}},nullptr,nullptr,0},false,nullptr,nullptr};
    plan.curvature=runtime::PowerConstraintCurvature<Program>{domain}.workspace_bytes();
    plan.kernel=program.bytes(7u,false);
    plan.domain=std::max(mean_domain.order?polynomial.domain.workspace_bytes(true):0u,
        Program::aparch?0u:mean::PolynomialDomain{program.q(),program.q(),nullptr}.workspace_bytes(true));
    plan.phase=curvature::refinement_workspace_bytes(k,plan.rows);
    if (!plan.kernel || !plan.curvature || !plan.phase
        || ((mean_domain.order || !Program::aparch) && !plan.domain)) return {};
    runtime::WorkspaceSizer size;
    size.add<std::byte>(plan.kernel);size.add<double>(plan.mean_jacobian);
    size.add<double>(plan.rows);size.add<double>(plan.jacobian);
    if constexpr (Program::reuse_moments)
        size.add<typename Program::MomentValues::Entry>(program.layout.p);
    for (const auto bytes : {plan.domain,plan.curvature,plan.phase}) {
        size.add<std::max_align_t>(0);size.add<std::byte>(bytes);
    }
    plan.total=size.size();
    return plan;
}

template<class Program,class Orders>
struct RefinementModel {
    static constexpr auto fixed_layout=Program::template fixed_layout<Orders>;
    Program program;
    const PowerConstraints<Program,fixed_layout>& domain;
    const scivol::linear_joint::Plan& plan;
    const scivol::linear_joint::Refinement& input;
    void *kernel,*curvature_memory;
    std::size_t kernel_bytes,curvature_bytes;
    std::size_t dimension() const noexcept {return domain.dimension();}
    std::size_t count() const noexcept {return domain.count();}
    double objective_scale() const noexcept {return 1.0/plan.n;}
    bool objective(unsigned mask,const double* theta,double* f,double* g,double* h) const noexcept
    {
        return program.template evaluate<Orders>(theta,input.y,input.x,plan.n,input.h0,
            mask,false,kernel,kernel_bytes,f,g,h,nullptr,nullptr,false)==VLK_JET_OK;
    }
    bool constraints(kraft::ConstraintRequest request,const double* theta,
                     double* values,double* jacobian) const noexcept
    {return domain.evaluate(request,theta,values,jacobian);}
    bool feasible(const double*,const double* values) const noexcept
    {return domain.feasible_values(values);}
    bool parameter_units(const double* theta,double* units) const noexcept
    {return program.template parameter_units<fixed_layout>(theta,input.units,input.h0,units);}
    bool constraint_curvature(const double* theta,const double* weights,double* h) const noexcept
    {
        return runtime::PowerConstraintCurvature<Program,fixed_layout>{domain}.weighted_hessian(
            theta,weights,h,curvature_memory,curvature_bytes);
    }
};

template<class Program>
int refine_program(const scivol::linear_joint::Plan& p,
                   const scivol::linear_joint::Refinement& s,Program program)
{
    if (!program.prepare(p.columns,p.p,p.q) || !p.n) return 0;
    const auto plan=refinement_plan(program,s.mean,p.columns);
    if (!plan.total || !s.workspace || s.bytes<plan.total
        || reinterpret_cast<std::uintptr_t>(s.workspace)%alignof(std::max_align_t)
        || !s.point || !s.best || s.point==s.best || !s.lower || !s.upper || !s.units
        || !s.evidence || !s.counts) return 0;
    runtime::WorkspaceArena arena(s.workspace,s.bytes);
    void* kernel=arena.take<std::byte>(plan.kernel);
    double* mean_jacobian=arena.take<double>(plan.mean_jacobian);
    double* values=arena.take<double>(plan.rows);
    double* jacobian=arena.take<double>(plan.jacobian);
    typename Program::MomentValues moments;
    if constexpr (Program::reuse_moments) {
        moments.entries=arena.take<typename Program::MomentValues::Entry>(p.p);
        moments.slots=p.p;
        if (!moments.reset()) return 0;
        program.moment_values=&moments;
    }
    const auto scratch=[&](std::size_t bytes)->void* {
        return arena.take<std::max_align_t>(0)?arena.take<std::byte>(bytes):nullptr;
    };
    void* domain_memory=scratch(plan.domain);
    void* curvature_memory=scratch(plan.curvature);
    void* phase_memory=scratch(plan.phase);
    if (!phase_memory) return 0;
    const mean::PolynomialConstraintPlan polynomial{
        {s.mean.order,s.mean.width,s.mean.basis},s.mean.offset};
    const auto run=[&]<class Orders>() {
        constexpr auto fixed_layout=Program::template fixed_layout<Orders>;
        const auto width=[&] {
            if constexpr (fixed_layout==nullptr) return program.layout.parameters;
            else return fixed_layout->parameters;
        }();
        const PowerConstraints<Program,fixed_layout> domain{program,
            mean::PolynomialConstraints{width,{polynomial,{}},
                mean_jacobian,domain_memory,plan.domain},false,values,jacobian};
        s.counts[15]=Orders::fixed && decltype(domain)::fixed
            ?2:(Orders::columns!=mean::dynamic?1:0);
        return curvature::refine_active_face(RefinementModel<Program,Orders>{
            program,domain,p,s,kernel,curvature_memory,plan.kernel,plan.curvature},
            s.point,s.best,s.lower,s.upper,{s.backtracks,4,s.target},phase_memory,plan.phase);
    };
    const auto result=p.columns==0
        ? run.template operator()<mean::LinearVolatilityOrders<0>>()
        : p.fixed && p.columns==4 && p.p==1 && p.q==1
            ? run.template operator()<mean::LinearVolatilityOrders<4,1,1>>()
            : p.fixed && p.columns==7 && p.p==3 && p.q==2
                ? run.template operator()<mean::LinearVolatilityOrders<7,3,2>>()
                : run.template operator()<mean::LinearVolatilityOrders<>>();
    s.evidence[0]=result.value;s.evidence[1]=result.bound;
    s.evidence[2]=result.best_value;s.evidence[3]=result.best_bound;
    s.evidence[4]=result.best_valid;s.evidence[5]=result.best_is_accepted;
    std::copy_n(result.objective_requests,8,s.counts);
    std::copy_n(result.constraint_requests,3,s.counts+8);
    s.counts[11]=result.curvature_requests;s.counts[12]=result.projections;
    s.counts[13]=result.trials;s.counts[14]=result.corrections;
    return 1+static_cast<int>(result.status);
}

} // namespace

namespace scivol::linear_joint {
std::size_t power_workspace(const Plan& p,unsigned mask,bool transformed)
{
    return power_bytes(p.family==3,p.density,p.columns,p.p,p.q,mask,transformed);
}
std::size_t power_solve_workspace(const Plan& p,const Domain& d,bool transformed,std::size_t affine)
{
    return dispatch_solve(p.family==3,p.density,[&](auto program)->std::size_t {
        return program.prepare(p.columns,p.p,p.q)
            ? solve_bytes(program,{d.order,d.width,d.basis},transformed,affine,p.n) : 0;
    });
}
bool power_evaluate(const Plan& p,const Evaluation& e)
{
    return ::power_evaluate(p.family==3,p.density,p.columns,p.p,p.q,e.theta,e.y,e.x,
        p.n,e.h0,e.request,e.transformed,p.fixed,e.workspace,e.bytes,e.value,
        e.gradient,e.hessian,e.residuals,e.variances)==VLK_JET_OK;
}
bool power_inverse(const Plan& p,const double* theta,double* z,void* work,std::size_t bytes)
{
    return power_map(p.family==3,p.density,p.columns,p.p,p.q,theta,z,false,p.fixed,work,bytes);
}
bool power_fit(const Plan& p,const Solve& s)
{
    return dispatch_solve(p.family==3,p.density,[&](auto program)->bool {
        return fit_program(p,s,program);
    });
}
std::size_t power_refinement_workspace(const Plan& p,const Domain& d)
{
    if (p.family<2 || p.family>3) return 0;
    return dispatch(p.family==3,p.density,[&](auto program)->std::size_t {
        return program.prepare(p.columns,p.p,p.q)?refinement_plan(program,d,p.columns).total:0u;
    });
}
int power_refine(const Plan& p,const Refinement& s)
{
    if (p.family<2 || p.family>3) return 0;
    return dispatch(p.family==3,p.density,[&](auto program)->int {
        return refine_program(p,s,program);
    });
}
std::size_t power_restriction_workspace(const Plan& p,const Domain& d,
                                      bool transformed,std::size_t affine)
{
    return dispatch_solve(p.family==3,p.density,[&](auto program)->std::size_t {
        if (!program.prepare(p.columns,p.p,p.q)) return 0;
        std::size_t rows;
        if (!runtime::checked_product(d.order,2u,rows)
            || !runtime::checked_sum(rows,transformed?0u:program.theta_constraints(),rows)
            || !runtime::checked_sum(rows,affine,rows)) return 0;
        return runtime::CoordinateRestriction::bytes(program.layout.parameters,rows);
    });
}
bool power_fit_restricted(const Plan& p,const Solve& s,const CoordinateSubset& subset)
{
    return dispatch_solve(p.family==3,p.density,[&](auto program)->bool {
        if (!program.prepare(p.columns,p.p,p.q)) return false;
        std::size_t rows;
        if (!runtime::checked_product(s.mean.order,2u,rows)
            || !runtime::checked_sum(rows,s.transformed?0u:program.theta_constraints(),rows)
            || !runtime::checked_sum(rows,s.mean_normal?s.mean_rows:0u,rows)) return false;
        runtime::CoordinateRestriction map;
        if (!map.bind(program.layout.parameters,subset.count,rows,subset.indices,s.point,
            {s.lower,s.upper},subset.workspace,subset.bytes)) return false;
        return fit_program(p,s,program,runtime::RestrictedCoordinateKraft{map});
    });
}
bool power_constraints(const Plan& p,const Domain& d,const double* theta,double* values,
                       double* jacobian,void* work,std::size_t bytes)
{
    return dispatch(p.family==3,p.density,[&](auto program)->bool {
        if (!program.prepare(p.columns,p.p,p.q)) return false;
        const mean::PolynomialConstraintPlan plan{{d.order,d.width,d.basis},d.offset};
        const auto required=solve_bytes(program,plan.domain,false);
        if (!required || bytes<required) return false;
        const auto k=program.layout.parameters,rows=2*d.order+program.theta_constraints();
        const auto domain_bytes=std::max(d.order?plan.domain.workspace_bytes(true):0u,
            mean::PolynomialDomain{p.q,p.q,nullptr}.workspace_bytes(true));
        runtime::WorkspaceArena arena(work,bytes);
        double* mean_jacobian=arena.take<double>(2*d.order*d.width);
        void* domain=arena.take<std::byte>(domain_bytes);
        double* local_jacobian=arena.take<double>(rows*k);
        double* local_values=arena.take<double>(rows);
        const PowerConstraints constraints{program,
            mean::PolynomialConstraints{k,{plan,{}},mean_jacobian,domain,domain_bytes},
            false,local_values,local_jacobian};
        return constraints.evaluate(jacobian?kraft::ConstraintRequest::value_jacobian
            :kraft::ConstraintRequest::value,theta,values,jacobian);
    });
}
} // namespace scivol::linear_joint
