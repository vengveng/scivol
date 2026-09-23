// SCIVOL-NATIVE: linear EGARCH/APARCH program and typed solver; promotion pending.
#include "linear_joint_api.hpp"
#include "linear_power_program.hpp"
#include "egarch_precision_state.hpp"
#include "gjr_transform_policy.hpp"
#include "mean_solver.hpp"
#include "affine_optimizer_constraints.hpp"
#include "power_solver_constraints.hpp"
#include "restricted_optimizer_coordinates.hpp"

namespace {
using namespace vlk::experiment;
using runtime::PowerConstraints;

template<class Density,bool Aparch>
using Program=runtime::LinearPowerProgram<Density,Aparch,
    std::conditional_t<!Aparch && std::is_same_v<Density,generated::VarianceGammaDensity>,
        runtime::arma_egarch::VarianceGammaPointState,runtime::arma_egarch::Binary64PointState>>;

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
    bool forward,void* work,std::size_t size)
{
    return dispatch(aparch,density,[&](auto program)->bool {
        return program.prepare(columns,p,q) && program.map(source,destination,forward,work,size);
    });
}

template<class Program,class Orders>
struct JointObjective {
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
    std::size_t dimension() const noexcept {return program.layout.parameters;}
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
                        bool transformed, bool affine = false)
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
    return size.size();
}
template<class Program,template<class> class Constraints=PowerConstraints,
         class Solver=runtime::FullCoordinateKraft>
bool fit_program(const scivol::linear_joint::Plan& p,
    const scivol::linear_joint::Solve& s,Program program,const Solver& solve={})
{
    if (!program.prepare(p.columns,p.p,p.q)) return false;
    const runtime::AffineEquality affine{s.mean_normal,s.mean_normal?p.columns:0u,s.mean_target};
    if (!affine.valid(program.layout.parameters)) return false;
    const mean::PolynomialConstraintPlan polynomial{
        {s.mean.order,s.mean.width,s.mean.basis},s.mean.offset};
    const auto required=solve_bytes(program,polynomial.domain,s.transformed,affine.active());
    if (!required || s.bytes<required) return false;
    const auto k=program.layout.parameters;
    const auto rows=2*s.mean.order+(s.transformed?0u:program.theta_constraints());
    const auto kernel_bytes=program.bytes(3u,s.transformed);
    const auto solver_rows=rows+affine.active();
    const auto solver_bytes=runtime::runtime_kraft_workspace_bytes(k,solver_rows,affine.active());
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
    runtime::RuntimeKraftWorkspace state;
    if (!runtime::bind_runtime_kraft_workspace(solve.dimension(k),solver_rows,solver,solver_bytes,state,affine.active())) return false;
    const auto run=[&]<class Orders>() {
        const JointObjective<decltype(program),Orders> objective{program,s.y,s.x,p.n,
            s.h0,kernel,kernel_bytes,polynomial,values,domain,domain_bytes,s.speculate,s.transformed};
        const Constraints<Program> constraints{program,
            mean::PolynomialConstraints{k,{polynomial,{}},jacobian,domain,domain_bytes},
            s.transformed,local_values,local_jacobian};
        const runtime::AffineEqualityPrefix restricted{constraints,affine};
        const auto result=solve(
            runtime::DiagonalObjective{objective,s.scales,scaled_point},
            runtime::DiagonalConstraints{restricted,s.scales,scaled_point},s.point,
            {s.lower,s.upper},{s.accuracy,s.iterations,s.curvature_reset},state);
        if (s.reset_count) *s.reset_count=result.constraint_curvature_resets;
        for (std::size_t j=0;j<k;++j) s.point[j]*=s.scales[j];
        s.stats[0]=result.mode;s.stats[1]=result.iterations;
        s.stats[2]=result.function_evaluations;s.stats[3]=result.gradient_evaluations;
        s.stats[4]=result.constraint_evaluations;
        s.stats[5]=Orders::fixed?2:(Orders::columns!=mean::dynamic?1:0);
        const bool mapped=s.transformed
            ? program.map(s.point,s.theta,true,kernel,kernel_bytes)
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

} // namespace

namespace scivol::linear_joint {
std::size_t power_workspace(const Plan& p,unsigned mask,bool transformed)
{
    return power_bytes(p.family==3,p.density,p.columns,p.p,p.q,mask,transformed);
}
std::size_t power_solve_workspace(const Plan& p,const Domain& d,bool transformed,bool affine)
{
    return dispatch(p.family==3,p.density,[&](auto program)->std::size_t {
        return program.prepare(p.columns,p.p,p.q)
            ? solve_bytes(program,{d.order,d.width,d.basis},transformed,affine) : 0;
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
    return power_map(p.family==3,p.density,p.columns,p.p,p.q,theta,z,false,work,bytes);
}
bool power_fit(const Plan& p,const Solve& s)
{
    return dispatch(p.family==3,p.density,[&](auto program)->bool {
        return fit_program(p,s,program);
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
