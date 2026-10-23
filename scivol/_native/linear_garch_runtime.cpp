// SCIVOL-NATIVE: linear GARCH/GJR program and typed solver; promotion pending.
#include "linear_joint_api.hpp"
#include "joint_garch_program.hpp"
#include "gjr_moment_policy.hpp"
#include "gjr_transform_policy.hpp"
#include "mean_solver.hpp"
#include "garch_solver_constraints.hpp"
#include <cmath>

namespace {
using namespace vlk::experiment;

using runtime::joint_garch::linear_layout;
constexpr std::size_t joint_density_shapes[] {0,1,2,1,3,2,2,2};

std::size_t jm_bytes(std::size_t columns,std::size_t p,std::size_t q,
                               bool leverage,unsigned request)
{ return runtime::joint_garch::workspace_bytes<true>(linear_layout(columns,p,q,leverage),request); }

template<class Density,bool Leverage,unsigned Request,bool Fixed>
bool evaluate(const runtime::ArmaGarchLayout& layout,const double* theta,
    const double* y,const double* x,std::size_t n,double h0,void* memory,
    std::size_t bytes,double* f,double* g,double* h,double* e,double* v)
{
    using namespace runtime::joint_garch;
    runtime::WorkspaceArena arena(memory,bytes);
    Scratch scratch;
    if (!bind_scratch<true>(arena,layout,Request,scratch)) return false;
    const auto run=[&]<std::size_t Columns=mean::dynamic,
                      std::size_t P=mean::dynamic,std::size_t Q=mean::dynamic,
                      unsigned Interleave=4>() {
        constexpr auto Width=Columns==mean::dynamic || P==mean::dynamic || Q==mean::dynamic
            ? mean::dynamic : Columns+1+P*(Leverage?2:1)+Q;
        return runtime::joint_garch::evaluate<Density,Leverage,
            (Request&1)!=0,(Request&2)!=0,(Request&4)!=0,
            mean::LinearDesign<Columns,Interleave>,P,Q,Width,
            Columns==mean::dynamic && Interleave==4>(
                layout,theta,y,n,h0,scratch,f,g,h,{x,layout.mean_parameters},e,v);
    };
    if constexpr (Fixed) {
        if (layout.p_vol==1 && layout.q_vol==1) {
            if (layout.mean_parameters==4) return run.template operator()<4,1,1>();
            if (layout.mean_parameters==7) return run.template operator()<7,1,1>();
            if (layout.mean_parameters<16)
                return run.template operator()<mean::dynamic,1,1,2>();
            return run.template operator()<mean::dynamic,1,1>();
        }
        if (layout.mean_parameters==4) return run.template operator()<4>();
        if (layout.mean_parameters==7) return run.template operator()<7>();
    }
    if (layout.mean_parameters<16)
        return run.template operator()<mean::dynamic,mean::dynamic,mean::dynamic,2>();
    return run.template operator()<>();
}

template<class Density,bool Leverage,bool Fixed>
bool request(const runtime::ArmaGarchLayout& layout,const double* theta,
    const double* y,const double* x,std::size_t n,double h0,unsigned mask,
    void* memory,std::size_t bytes,double* f,double* g,double* h,double* e,double* v)
{
    const auto k=layout.parameter_count;
    for (std::size_t j=0;j<k+Density::shape_parameters;++j)
        if (!std::isfinite(theta[j])) return false;
    if (!(theta[layout.omega_index]>0) || !Density::valid_shape(theta+k)) return false;
    if constexpr (Leverage) {
        if (!generated::valid_gjr_coefficients_and_persistence<Density>(theta,
            layout.p_vol,layout.q_vol,layout.alpha_start,layout.gamma_start,
            layout.beta_start,theta+k)) return false;
    } else {
        double persistence=0;
        for (std::size_t j=layout.alpha_start;j<k;++j) {
            if (theta[j]<0) return false;
            persistence+=theta[j];
        }
        if (!(persistence<1)) return false;
    }
    bool valid=false;
    switch(mask) {
#define MASK(R) case R: valid=evaluate<Density,Leverage,R,Fixed>(layout,theta,y,x,n,h0,memory,bytes,f,g,h,e,v);break
        MASK(1);MASK(2);MASK(3);MASK(4);MASK(5);MASK(6);MASK(7);
#undef MASK
    }
    if (!valid) return false;
    if ((mask&1) && !std::isfinite(*f)) return false;
    if (mask&2) for (std::size_t j=0;j<k+Density::shape_parameters;++j)
        if (!std::isfinite(g[j])) return false;
    if (mask&4) for (std::size_t j=0;j<(k+Density::shape_parameters)*(k+Density::shape_parameters);++j)
        if (!std::isfinite(h[j])) return false;
    return true;
}

template<class Density, bool Leverage, bool Fixed>
bool transformed_request(unsigned density,const runtime::ArmaGarchLayout& layout,
    const double* z,const double* y,const double* x,std::size_t n,double h0,
    unsigned mask,void* memory,std::size_t bytes,double* f,double* g,double* h,
    double* theta_out=nullptr,double* gradient_out=nullptr)
{
    const auto kernel=[x,theta_out,gradient_out](const auto& plan,const double* theta,std::size_t total,
        const double* observations,std::size_t count,double initial,unsigned r,
        void* work,std::size_t size,double* value,double* gradient,double* hessian,
        unsigned* valid) noexcept {
        const bool ok=request<Density,Leverage,Fixed>(plan,theta,observations,x,
            count,initial,r,work,size,value,gradient,hessian,nullptr,nullptr);
        if (ok && theta_out) std::copy_n(theta,total,theta_out);
        if (ok && gradient_out) std::copy_n(gradient,total,gradient_out);
        *valid=ok?r:0u;
        return ok?VLK_JET_OK:VLK_JET_DOMAIN_ERROR;
    };
    unsigned valid=0;
    return runtime::joint_garch::evaluate_transformed<Density,
        generated::DensityShapeTransformT<Density>,Leverage,true>(
            layout,density,z,layout.parameter_count+Density::shape_parameters,
            y,n,h0,mask,memory,bytes,f,g,h,&valid,false,kernel)==VLK_JET_OK;
}

template<bool Leverage, bool Fixed>
bool transformed_density_request(unsigned density,const runtime::ArmaGarchLayout& layout,
    const double* z,const double* y,const double* x,std::size_t n,double h0,
    unsigned mask,void* memory,std::size_t bytes,double* f,double* g,double* h)
{
    using namespace generated;
    switch(density) {
#define D(C,T) case C:return transformed_request<T,Leverage,Fixed>(density,layout,z,y,x,n,h0,mask,memory,bytes,f,g,h)
        D(0,NormalDensity);D(1,StudentTDensity);D(2,SkewTDensity);D(3,GedDensity);
        D(4,GeneralizedHyperbolicDensity);D(5,NormalInverseGaussianDensity);
        D(6,HyperbolicDensity);D(7,VarianceGammaDensity);
#undef D
    }
    return false;
}

std::size_t jm_coordinate_bytes(std::size_t columns,std::size_t p,
    std::size_t q,bool leverage,unsigned density,unsigned mask)
{
    const auto layout=linear_layout(columns,p,q,leverage);
    using namespace generated;
    switch(density) {
#define D(C,T) case C:return runtime::joint_garch::transformed_workspace_bytes<T,true>(layout,density,mask)
        D(0,NormalDensity);D(1,StudentTDensity);D(2,SkewTDensity);D(3,GedDensity);
        D(4,GeneralizedHyperbolicDensity);D(5,NormalInverseGaussianDensity);
        D(6,HyperbolicDensity);D(7,VarianceGammaDensity);
#undef D
    }
    return 0;
}

template<class Shape>
bool coordinate_inverse(const runtime::ArmaGarchLayout& layout,unsigned density,
    const double* theta,double* z,void* memory,std::size_t bytes)
{
    if (layout.leverage)
        return runtime::arma_gjr_garch_coordinate_inverse(layout,density,theta,memory,bytes,z);
    return runtime::arma_garch_coordinate_inverse(layout,theta,memory,bytes,z)
        && Shape::unpack(theta+layout.parameter_count,z+layout.parameter_count);
}

bool jm_coordinate_inverse(std::size_t columns,std::size_t p,
    std::size_t q,bool leverage,unsigned density,const double* theta,double* z,
    void* memory,std::size_t bytes)
{
    const auto layout=linear_layout(columns,p,q,leverage);
    using namespace generated;
    switch(density) {
#define D(C,T) case C:return coordinate_inverse<T>(layout,density,theta,z,memory,bytes)
        D(0,NoShapeTransform);D(1,StudentTShapeTransform);D(2,SkewTShapeTransform);
        D(3,GedShapeTransform);D(4,GhShapeTransform);D(5,FixedLambdaGhShapeTransform);
        D(6,FixedLambdaGhShapeTransform);D(7,VarianceGammaShapeTransform);
#undef D
    }
    return false;
}

bool jm_coordinate_evaluate(const double* z,const double* y,const double* x,
    std::size_t n,std::size_t columns,std::size_t p,std::size_t q,bool leverage,
    unsigned density,unsigned mask,bool fixed,double h0,void* memory,std::size_t bytes,
    double* f,double* g,double* h)
{
    const auto layout=linear_layout(columns,p,q,leverage);
    if (!layout.valid || !z || !y || !x || !n || !(h0>0) || !std::isfinite(h0))
        return false;
    const bool use_fixed=fixed;
#define CALL(L,F) transformed_density_request<L,F>(density,layout,z,y,x,n,h0,mask,memory,bytes,f,g,h)
    return leverage ? (use_fixed?CALL(true,true):CALL(true,false))
                    : (use_fixed?CALL(false,true):CALL(false,false));
#undef CALL
}

std::size_t jm_solve_bytes(std::size_t columns,std::size_t p,
    std::size_t q,bool leverage,unsigned density,std::size_t order,
    std::size_t ar_parameters,const double* basis,bool transformed)
{
    const auto layout=linear_layout(columns,p,q,leverage);
    if (!layout.valid || density>=8 || ar_parameters>columns) return 0;
    const auto k=layout.parameter_count+joint_density_shapes[density];
    const mean::PolynomialDomain domain {order,ar_parameters,basis};
    const auto kernel=transformed?jm_coordinate_bytes(columns,p,q,leverage,density,3):jm_bytes(columns,p,q,leverage,3);
    const auto rows=2*order+(transformed?0u:(leverage?p:0u)+1u);
    const auto solver=runtime::runtime_kraft_workspace_bytes(k,rows);
    const auto domain_bytes=order?domain.workspace_bytes(true):0;
    std::size_t jacobian;
    if (!kernel || !solver || (order && !domain_bytes)
        || !runtime::checked_product(2*order,ar_parameters,jacobian)) return 0;
    runtime::WorkspaceSizer size;
    size.add<std::byte>(kernel);
    size.add<std::byte>(solver);
    size.add<double>(k);
    if (!transformed) size.add<double>(rows*k);
    if (order) {
        size.add<std::byte>(domain_bytes);
        size.add<double>(jacobian);
        size.add<double>(2*order);
    }
    return size.size();
}

template<class Density,bool Leverage,bool Fixed,bool Transformed>
struct JointObjective {
    runtime::ArmaGarchLayout plan;
    unsigned density;
    const double* y;
    const double* x;
    std::size_t n;
    double h0;
    void* memory;
    std::size_t bytes;
    mean::PolynomialConstraintPlan mean_domain;
    void* domain_memory;
    std::size_t domain_bytes;
    double* domain_values;
    bool speculate;
    mutable std::int64_t requests[3]{};
    std::size_t dimension() const noexcept
    { return plan.parameter_count+Density::shape_parameters; }
    bool speculate_trial_gradient(std::int64_t) const noexcept {return speculate;}
    bool evaluate(kraft::ObjectiveRequest requested,const double* z,double* f,
        double* g,bool) const noexcept
    {
        const unsigned mask=requested==kraft::ObjectiveRequest::value?1u:
            requested==kraft::ObjectiveRequest::gradient?2u:3u;
        ++requests[mask-1];
        if (mean_domain.domain.order && !mean_domain.domain.evaluate(z+mean_domain.offset,
            domain_values,nullptr,domain_memory,domain_bytes,true)) return false;
        if constexpr (Transformed) {
            if (!transformed_request<Density,Leverage,Fixed>(density,plan,z,y,x,n,h0,
                mask,memory,bytes,f,g,nullptr)) return false;
        } else if (!request<Density,Leverage,Fixed>(plan,z,y,x,n,h0,mask,
            memory,bytes,f,g,nullptr,nullptr,nullptr)) return false;
        if (mask&1) *f/=static_cast<double>(n);
        if (mask&2) for (std::size_t j=0;j<dimension();++j) g[j]/=static_cast<double>(n);
        return true;
    }
};

struct JointConstraints {
    mean::PolynomialConstraints mean;
    runtime::GarchThetaConstraints volatility;
    bool transformed;
    double* local_jacobian;
    std::size_t dimension() const noexcept {return mean.dimension();}
    std::size_t count() const noexcept {return mean.count()+(transformed?0u:volatility.count());}
    bool evaluate(kraft::ConstraintRequest requested,const double* theta,double* values,double* jacobian) const noexcept
    {
        if (transformed) return mean.evaluate(requested,theta,values,jacobian);
        const bool derivatives=requested!=kraft::ConstraintRequest::value;
        if (!mean.evaluate(requested,theta,values,derivatives?local_jacobian:nullptr)) return false;
        if (derivatives) for (std::size_t column=0;column<dimension();++column)
            std::copy_n(local_jacobian+column*mean.count(),mean.count(),jacobian+column*count());
        if (!volatility.evaluate(requested,theta,values+mean.count(),derivatives?local_jacobian:nullptr)) return false;
        if (derivatives) for (std::size_t column=0;column<dimension();++column)
            std::copy_n(local_jacobian+column*volatility.count(),volatility.count(),jacobian+column*count()+mean.count());
        return true;
    }
};

template<class Density,bool Leverage,bool Fixed,bool Transformed>
bool solve_joint(const runtime::ArmaGarchLayout& layout,unsigned density,
    double* z,const double* y,const double* x,std::size_t n,double h0,
    mean::PolynomialConstraintPlan polynomial,const double* lower,const double* upper,const double* scales,
    void* memory,std::size_t bytes,double* theta,double* gradient,
    std::int64_t* stats,double* value,double accuracy,std::int64_t iterations,bool speculate,
    std::int64_t* requests,bool curvature_reset,std::int64_t* reset_count,bool minimum_invalid_step)
{
    const auto k=layout.parameter_count+Density::shape_parameters;
    const auto mean_rows=2*polynomial.domain.order;
    const auto constraints=mean_rows+(Transformed?0u:(Leverage?layout.p_vol:0u)+1u);
    runtime::WorkspaceArena arena(memory,bytes);
    const auto kernel_bytes=Transformed?jm_coordinate_bytes(layout.mean_parameters,layout.p_vol,
        layout.q_vol,Leverage,density,3):jm_bytes(layout.mean_parameters,layout.p_vol,layout.q_vol,Leverage,3);
    void* kernel=arena.take<std::byte>(kernel_bytes);
    const auto solver_bytes=runtime::runtime_kraft_workspace_bytes(k,constraints);
    void* solver=arena.take<std::byte>(solver_bytes);
    double* scaled_point=arena.take<double>(k);
    double* combined_jacobian=Transformed?nullptr:arena.take<double>(constraints*k);
    const auto domain_bytes=mean_rows?polynomial.domain.workspace_bytes(true):0;
    void* domain=mean_rows?arena.take<std::byte>(domain_bytes):nullptr;
    double* jacobian=mean_rows?arena.take<double>(mean_rows*polynomial.domain.parameters):nullptr;
    double* values=mean_rows?arena.take<double>(mean_rows):nullptr;
    runtime::RuntimeKraftWorkspace state;
    if (!kernel || !solver || (mean_rows && (!domain || !jacobian || !values))
        || !runtime::bind_runtime_kraft_workspace(k,constraints,solver,solver_bytes,state))
        return false;
    const JointObjective<Density,Leverage,Fixed,Transformed> objective {
        layout,density,y,x,n,h0,kernel,kernel_bytes,polynomial,domain,domain_bytes,values,speculate};
    const JointConstraints restrictions {{k,{polynomial,{}},jacobian,domain,domain_bytes},
        {k,layout.mean_parameters,layout.p_vol,layout.q_vol,density,Leverage},Transformed,combined_jacobian};
    const auto result=runtime::minimize_runtime_kraft(
        runtime::DiagonalObjective{objective,scales,scaled_point},
        runtime::DiagonalConstraints{restrictions,scales,scaled_point},z,
        {lower,upper},{accuracy,iterations,curvature_reset,
            minimum_invalid_step?kraft::InvalidTrialPolicy::minimum_step
                                :kraft::InvalidTrialPolicy::interpolate},state);
    if (reset_count) *reset_count=result.constraint_curvature_resets;
    for (std::size_t j=0;j<k;++j) z[j]*=scales[j];
    stats[0]=result.mode;stats[1]=result.iterations;
    stats[2]=result.function_evaluations;stats[3]=result.gradient_evaluations;
    stats[4]=result.constraint_evaluations;
    stats[5]=!Fixed?0:((layout.p_vol==1 && layout.q_vol==1)
        ?((layout.mean_parameters==4 || layout.mean_parameters==7)?2:1)
        :((layout.mean_parameters==4 || layout.mean_parameters==7)?1:0));
    if (requests) {
        std::copy_n(objective.requests,3,requests);
        ++requests[2];
    }
    bool valid;
    if constexpr (Transformed)
        valid=transformed_request<Density,Leverage,Fixed>(density,layout,z,y,x,n,h0,
            3,kernel,kernel_bytes,value,state.gradient,nullptr,theta,gradient);
    else {
        std::copy_n(z,k,theta);
        valid=request<Density,Leverage,Fixed>(layout,theta,y,x,n,h0,3,kernel,kernel_bytes,value,gradient,nullptr,nullptr,nullptr);
    }
    ++stats[2];++stats[3];
    return valid;
}

template<bool Leverage,bool Fixed,bool Transformed>
bool solve_density(const runtime::ArmaGarchLayout& layout,unsigned density,
    double* z,const double* y,const double* x,std::size_t n,double h0,
    mean::PolynomialConstraintPlan polynomial,const double* lower,const double* upper,const double* scales,
    void* memory,std::size_t bytes,double* theta,double* gradient,
    std::int64_t* stats,double* value,double accuracy,std::int64_t iterations,bool speculate,
    std::int64_t* requests,bool curvature_reset,std::int64_t* reset_count,bool minimum_invalid_step)
{
    using namespace generated;
    switch(density) {
#define D(C,T) case C:return solve_joint<T,Leverage,Fixed,Transformed>(layout,density,z,y,x,n,h0,polynomial,lower,upper,scales,memory,bytes,theta,gradient,stats,value,accuracy,iterations,speculate,requests,curvature_reset,reset_count,minimum_invalid_step)
        D(0,NormalDensity);D(1,StudentTDensity);D(2,SkewTDensity);D(3,GedDensity);
        D(4,GeneralizedHyperbolicDensity);D(5,NormalInverseGaussianDensity);
        D(6,HyperbolicDensity);D(7,VarianceGammaDensity);
#undef D
    }
    return false;
}

bool jm_solve(double* z,const double* y,const double* x,std::size_t n,
    std::size_t columns,std::size_t p,std::size_t q,bool leverage,unsigned density,bool fixed,bool transformed,
    double h0,std::size_t order,std::size_t ar_parameters,std::size_t offset,const double* basis,
    const double* lower,const double* upper,const double* scales,void* memory,std::size_t bytes,
    double* theta,double* gradient,std::int64_t* stats,double* value,double accuracy,std::int64_t iterations,bool speculate,
    std::int64_t* requests,bool curvature_reset,std::int64_t* reset_count,bool minimum_invalid_step)
{
    const auto layout=linear_layout(columns,p,q,leverage);
    const auto required=jm_solve_bytes(columns,p,q,leverage,density,order,ar_parameters,basis,transformed);
    if (!required || !memory || bytes<required || !n || !z || !y || !x || !lower || !upper
        || !scales || !theta || !gradient || !stats || !value || !(h0>0) || !std::isfinite(h0)
        || offset>columns || ar_parameters>columns-offset || !(accuracy>0) || iterations<=0)
        return false;
    std::size_t features;
    if (!runtime::checked_product(n,columns,features)) return false;
    for (std::size_t i=0;i<n;++i) if (!std::isfinite(y[i])) return false;
    for (std::size_t i=0;i<features;++i) if (!std::isfinite(x[i])) return false;
    const mean::PolynomialConstraintPlan polynomial {{order,ar_parameters,basis},offset};
    const bool use_fixed=fixed;
#define CALL(L,F) (transformed?solve_density<L,F,true>(layout,density,z,y,x,n,h0,polynomial,lower,upper,scales,memory,bytes,theta,gradient,stats,value,accuracy,iterations,speculate,requests,curvature_reset,reset_count,minimum_invalid_step):solve_density<L,F,false>(layout,density,z,y,x,n,h0,polynomial,lower,upper,scales,memory,bytes,theta,gradient,stats,value,accuracy,iterations,speculate,requests,curvature_reset,reset_count,minimum_invalid_step))
    return leverage?(use_fixed?CALL(true,true):CALL(true,false)):
                    (use_fixed?CALL(false,true):CALL(false,false));
#undef CALL
}

template<bool Leverage,bool Fixed>
bool density_request(unsigned density,const runtime::ArmaGarchLayout& layout,
    const double* theta,const double* y,const double* x,std::size_t n,double h0,
    unsigned mask,void* memory,std::size_t bytes,double* f,double* g,double* h,double* e,double* v)
{
    using namespace generated;
    switch(density) {
#define DENSITY(C,T) case C:return request<T,Leverage,Fixed>(layout,theta,y,x,n,h0,mask,memory,bytes,f,g,h,e,v)
        DENSITY(0,NormalDensity);DENSITY(1,StudentTDensity);DENSITY(2,SkewTDensity);
        DENSITY(3,GedDensity);DENSITY(4,GeneralizedHyperbolicDensity);
        DENSITY(5,NormalInverseGaussianDensity);DENSITY(6,HyperbolicDensity);
        DENSITY(7,VarianceGammaDensity);
#undef DENSITY
    }
    return false;
}

bool joint_entry(const double* theta,const double* y,const double* x,
    std::size_t n,std::size_t columns,std::size_t p,std::size_t q,bool leverage,
    unsigned density,unsigned mask,bool fixed,double h0,void* memory,
    std::size_t bytes,double* f,double* g,double* h,double* e,double* v,bool validate_data)
{
    const auto layout=linear_layout(columns,p,q,leverage);
    const auto required=runtime::joint_garch::workspace_bytes<true>(layout,mask);
    std::size_t features;
    if (!layout.valid || !required || bytes<required || !memory || !theta || !y
        || !n || !(h0>0) || !std::isfinite(h0)
        || !runtime::checked_product(n,columns,features) || (features && !x)
        || ((mask&1)&&!f) || ((mask&2)&&!g) || ((mask&4)&&!h)) return false;
    if (validate_data) {
        for (std::size_t j=0;j<n;++j) if (!std::isfinite(y[j])) return false;
        for (std::size_t j=0;j<features;++j) if (!std::isfinite(x[j])) return false;
    }
    if (leverage) {
        if (fixed) return density_request<true,true>(density,layout,theta,y,x,n,h0,mask,memory,bytes,f,g,h,e,v);
        return density_request<true,false>(density,layout,theta,y,x,n,h0,mask,memory,bytes,f,g,h,e,v);
    }
    if (fixed) return density_request<false,true>(density,layout,theta,y,x,n,h0,mask,memory,bytes,f,g,h,e,v);
    return density_request<false,false>(density,layout,theta,y,x,n,h0,mask,memory,bytes,f,g,h,e,v);
}

bool jm_evaluate(const double* theta,const double* y,const double* x,
    std::size_t n,std::size_t columns,std::size_t p,std::size_t q,bool leverage,
    unsigned density,unsigned mask,bool fixed,double h0,void* memory,
    std::size_t bytes,double* f,double* g,double* h,double* e,double* v)
{ return joint_entry(theta,y,x,n,columns,p,q,leverage,density,mask,fixed,h0,memory,bytes,f,g,h,e,v,true); }


} // namespace
namespace scivol::linear_joint {
std::size_t garch_workspace(const Plan& p,unsigned mask,bool transformed)
{
    return transformed?jm_coordinate_bytes(p.columns,p.p,p.q,p.family==1,p.density,mask)
        :jm_bytes(p.columns,p.p,p.q,p.family==1,mask);
}
std::size_t garch_solve_workspace(const Plan& p,const Domain& d,bool transformed)
{
    return jm_solve_bytes(p.columns,p.p,p.q,p.family==1,p.density,d.order,d.width,d.basis,transformed);
}
bool garch_evaluate(const Plan& p,const Evaluation& e)
{
    if (e.transformed)
        return jm_coordinate_evaluate(e.theta,e.y,e.x,p.n,p.columns,p.p,p.q,p.family==1,
            p.density,e.request,p.fixed,e.h0,e.workspace,e.bytes,e.value,e.gradient,e.hessian);
    return joint_entry(e.theta,e.y,e.x,p.n,p.columns,p.p,p.q,p.family==1,p.density,
        e.request,p.fixed,e.h0,e.workspace,e.bytes,e.value,e.gradient,e.hessian,e.residuals,e.variances,false);
}
bool garch_inverse(const Plan& p,const double* theta,double* z,void* work,std::size_t bytes)
{
    return jm_coordinate_inverse(p.columns,p.p,p.q,p.family==1,p.density,theta,z,work,bytes);
}
bool garch_fit(const Plan& p,const Solve& s)
{
    return jm_solve(s.point,s.y,s.x,p.n,p.columns,p.p,p.q,p.family==1,p.density,p.fixed,
        s.transformed,s.h0,s.mean.order,s.mean.width,s.mean.offset,s.mean.basis,
        s.lower,s.upper,s.scales,s.workspace,s.bytes,s.theta,s.gradient,s.stats,s.value,
        s.accuracy,s.iterations,s.speculate,s.requests,s.curvature_reset,s.reset_count,s.minimum_invalid_step);
}
bool garch_constraints(const Plan& p,const Domain& d,const double* theta,double* values,
                       double* jacobian,void* work,std::size_t bytes)
{
    const auto required=garch_solve_workspace(p,d,false);
    if (!required || bytes<required) return false;
    const auto layout=linear_layout(p.columns,p.p,p.q,p.family==1);
    const auto k=layout.parameter_count+joint_density_shapes[p.density];
    const mean::PolynomialConstraintPlan plan{{d.order,d.width,d.basis},d.offset};
    const runtime::GarchThetaConstraints volatility{k,p.columns,p.p,p.q,p.density,p.family==1};
    const auto rows=2*d.order+volatility.count();
    const auto domain_bytes=d.order?plan.domain.workspace_bytes(true):0u;
    runtime::WorkspaceArena arena(work,bytes);
    void* domain=arena.take<std::byte>(domain_bytes);
    double* mean_jacobian=arena.take<double>(2*d.order*d.width);
    double* local_jacobian=arena.take<double>(rows*k);
    const JointConstraints constraints{
        {k,{plan,{}},mean_jacobian,domain,domain_bytes},volatility,false,local_jacobian};
    return constraints.evaluate(jacobian?kraft::ConstraintRequest::value_jacobian
        :kraft::ConstraintRequest::value,theta,values,jacobian);
}
} // namespace scivol::linear_joint
