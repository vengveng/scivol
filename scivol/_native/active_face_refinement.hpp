// SCIVOL-NATIVE: CORE; model-independent active-face algebra and stationarity.
#pragma once
#include "runtime_kraft_slsqp.hpp"

extern "C" {
void vlk_dgeqr2(CBLAS_INT*,CBLAS_INT*,double*,CBLAS_INT*,double*,double*,CBLAS_INT*);
void vlk_dorm2r(const char*,const char*,CBLAS_INT*,CBLAS_INT*,CBLAS_INT*,double*,
    CBLAS_INT*,double*,double*,CBLAS_INT*,double*,CBLAS_INT*);
}

namespace vlk::experiment::curvature {

kraft::Result direct_projection(std::size_t,std::size_t,const double*,const double*,
    const double*,double*,double*,double*,runtime::RuntimeKraftWorkspace&) noexcept;

// A and eigenvectors are column-major. Scaling protects the rotations from
// overflow; convergence is checked before any eigenvalue can select a direction.
inline bool symmetric_eigensystem(double* a,std::size_t n,double* values,
    double* vectors,unsigned maximum_sweeps=64) noexcept
{
    std::size_t square;
    if (!n || !a || !values || !vectors || !maximum_sweeps
        || !runtime::checked_product(n,n,square)) return false;
    double scale=0;
    for (std::size_t i=0;i<square;++i) {
        if (!std::isfinite(a[i])) return false;
        scale=std::max(scale,std::abs(a[i]));
    }
    const double epsilon=std::numeric_limits<double>::epsilon();
    std::fill_n(vectors,square,0.0);
    for (std::size_t i=0;i<n;++i) vectors[i+i*n]=1.0;
    if (!scale) {std::fill_n(values,n,0.0);return true;}
    for (std::size_t j=0;j<n;++j) for (std::size_t i=0;i<=j;++i) {
        const double left=a[i+j*n]/scale,right=a[j+i*n]/scale;
        if (std::abs(left-right)>8*n*epsilon) return false;
        a[i+j*n]=a[j+i*n]=0.5*left+0.5*right;
    }
    bool converged=false;
    for (unsigned sweep=0;sweep<maximum_sweeps;++sweep) {
        double maximum=0;
        for (std::size_t q=1;q<n;++q) for (std::size_t p=0;p<q;++p) {
            const double off=a[p+q*n];
            maximum=std::max(maximum,std::abs(off));
            if (std::abs(off)<=epsilon) continue;
            const double theta=(a[q+q*n]-a[p+p*n])/(2*off);
            const double t=std::copysign(1.0,theta)/(std::abs(theta)+std::hypot(1.0,theta));
            const double c=1/std::hypot(1.0,t),s=t*c;
            for (std::size_t i=0;i<n;++i) if (i!=p && i!=q) {
                const double left=a[i+p*n],right=a[i+q*n];
                a[i+p*n]=a[p+i*n]=c*left-s*right;
                a[i+q*n]=a[q+i*n]=s*left+c*right;
            }
            a[p+p*n]-=t*off;a[q+q*n]+=t*off;
            a[p+q*n]=a[q+p*n]=0;
            for (std::size_t i=0;i<n;++i) {
                const double left=vectors[i+p*n],right=vectors[i+q*n];
                vectors[i+p*n]=c*left-s*right;
                vectors[i+q*n]=s*left+c*right;
            }
        }
        if (maximum<=epsilon) {converged=true;break;}
    }
    if (!converged) {
        double remaining=0;
        for (std::size_t j=1;j<n;++j) for (std::size_t i=0;i<j;++i)
            remaining=std::max(remaining,std::abs(a[i+j*n]));
        if (remaining>epsilon) return false;
    }
    for (std::size_t i=0;i<n;++i) {
        values[i]=a[i+i*n]*scale;
        if (!std::isfinite(values[i])) return false;
    }
    for (std::size_t i=0;i<n;++i) {
        auto smallest=i;
        for (std::size_t j=i+1;j<n;++j) if (values[j]<values[smallest]) smallest=j;
        if (smallest!=i) {
            std::swap(values[i],values[smallest]);
            for (std::size_t row=0;row<n;++row)
                std::swap(vectors[row+i*n],vectors[row+smallest*n]);
        }
    }
    return true;
}

inline std::size_t active_face_workspace_bytes(std::size_t n) noexcept
{
    std::size_t square;
    if (!n || n>static_cast<std::size_t>(CBLAS_INT_MAX)
        || !runtime::checked_product(n,n,square)) return 0;
    runtime::WorkspaceSizer size;
    for (unsigned i=0;i<5;++i) size.add<double>(square);
    for (unsigned i=0;i<3;++i) size.add<double>(n);
    return size.size();
}

inline bool normal_space_basis(std::size_t n,std::size_t active,const double* normals,
    double* qr,double* q,double* tau,double* work,bool complete,double* row_scales=nullptr) noexcept
{
    for (std::size_t j=0;j<active;++j) {
        double norm=0;
        for (std::size_t i=0;i<n;++i) {
            const double value=normals[j*n+i];
            if (!std::isfinite(value)) return false;
            norm=std::max(norm,std::abs(value));
        }
        if (!norm) return false;
        if (row_scales) row_scales[j]=norm;
        for (std::size_t i=0;i<n;++i) qr[i+j*n]=normals[j*n+i]/norm;
    }
    const auto columns=complete?n:active;
    std::fill_n(q,n*columns,0.0);
    for (std::size_t i=0;i<columns;++i) q[i+i*n]=1.0;
    CBLAS_INT width=static_cast<CBLAS_INT>(n),count=static_cast<CBLAS_INT>(active);
    CBLAS_INT output_columns=static_cast<CBLAS_INT>(columns),info=0;
    if (!active) return true;
    vlk_dgeqr2(&width,&count,qr,&width,tau,work,&info);
    if (info) return false;
    const double tolerance=64*std::numeric_limits<double>::epsilon();
    for (std::size_t i=0;i<active;++i)
        if (!std::isfinite(qr[i+i*n]) || !(std::abs(qr[i+i*n])>tolerance)) return false;
    vlk_dorm2r("L","N",&width,&output_columns,&count,qr,&width,tau,q,&width,work,&info);
    return !info;
}

// Inputs are dimensionless gradient/curvature and row-major active normals.
// Negative and unresolved curvature cannot propose a Newton step. They are not
// discarded from the later full-domain stationarity assessment.
inline bool active_face_direction(std::size_t n,std::size_t active,
    const double* normals,const double* hessian,const double* gradient,
    double* direction,double* eigenvalues,std::size_t& resolved,
    void* memory,std::size_t bytes,const double* second_gradient=nullptr,
    double* second_direction=nullptr) noexcept
{
    resolved=0;
    const auto required=active_face_workspace_bytes(n);
    if (!required || active>n || (active && !normals) || !hessian || !gradient
        || !direction || !eigenvalues || !memory || bytes<required
        || bool(second_gradient)!=bool(second_direction)) return false;
    runtime::WorkspaceArena arena(memory,bytes);
    double* qr=arena.take<double>(n*n);
    double* q=arena.take<double>(n*n);
    double* product=arena.take<double>(n*n);
    double* reduced=arena.take<double>(n*n);
    double* vectors=arena.take<double>(n*n);
    double* tau=arena.take<double>(n);
    double* work=arena.take<double>(n);
    double* projected=arena.take<double>(n);
    if (!projected) return false;
    double hessian_scale=0;
    for (std::size_t i=0;i<n*n;++i) {
        if (!std::isfinite(hessian[i])) return false;
        hessian_scale=std::max(hessian_scale,std::abs(hessian[i]));
    }
    if (hessian_scale) for (std::size_t i=0;i<n;++i) for (std::size_t j=i+1;j<n;++j)
        if (std::abs(hessian[i*n+j]/hessian_scale-hessian[j*n+i]/hessian_scale)
            >8*n*std::numeric_limits<double>::epsilon()) return false;
    for (std::size_t i=0;i<n;++i) if (!std::isfinite(gradient[i])) return false;
    if (second_gradient) for (std::size_t i=0;i<n;++i)
        if (!std::isfinite(second_gradient[i])) return false;
    if (!normal_space_basis(n,active,normals,qr,q,tau,work,true)) return false;
    const auto free=n-active;
    std::fill_n(direction,n,0.0);
    if (second_direction) std::fill_n(second_direction,n,0.0);
    if (!free) return true;
    const double* tangent=q+active*n;
    for (std::size_t j=0;j<free;++j) {
        projected[j]=0;
        for (std::size_t row=0;row<n;++row) {
            double value=0;
            for (std::size_t col=0;col<n;++col)
                value+=hessian[row*n+col]*tangent[col+j*n];
            product[row+j*n]=value;
            projected[j]+=tangent[row+j*n]*gradient[row];
        }
    }
    for (std::size_t j=0;j<free;++j) for (std::size_t i=0;i<=j;++i) {
        double value=0;
        for (std::size_t row=0;row<n;++row) value+=tangent[row+i*n]*product[row+j*n];
        reduced[i+j*free]=reduced[j+i*free]=value;
    }
    if (!symmetric_eigensystem(reduced,free,eigenvalues,vectors)) return false;
    double magnitude=0;
    for (std::size_t j=0;j<free;++j) magnitude=std::max(magnitude,std::abs(eigenvalues[j]));
    const double resolution=n*std::numeric_limits<double>::epsilon()*magnitude;
    for (std::size_t j=0;j<free;++j) resolved+=eigenvalues[j]>resolution;
    const auto assemble=[&](double* output) {
        std::fill_n(work,free,0.0);
        for (std::size_t j=0;j<free;++j) if (eigenvalues[j]>resolution) {
            double coefficient=0;
            for (std::size_t i=0;i<free;++i) coefficient+=vectors[i+j*free]*projected[i];
            coefficient/=-eigenvalues[j];
            for (std::size_t i=0;i<free;++i) work[i]+=vectors[i+j*free]*coefficient;
        }
        for (std::size_t row=0;row<n;++row) {
            for (std::size_t j=0;j<free;++j) output[row]+=tangent[row+j*n]*work[j];
            if (!std::isfinite(output[row])) return false;
        }
        return true;
    };
    if (!assemble(direction)) return false;
    if (!second_gradient) return true;
    for (std::size_t j=0;j<free;++j) {
        projected[j]=0;
        for (std::size_t row=0;row<n;++row) projected[j]+=tangent[row+j*n]*second_gradient[row];
    }
    return assemble(second_direction);
}

// Same power-of-two curvature heuristic as the main SQP adapter. Assessment
// units remain separate; only this local quadratic's search coordinates change.
inline bool condition_curvature(std::size_t n,std::size_t rows,
    const double* hessian,const double* gradient,const double* normals,double* units,
    double* conditioned_hessian,double* conditioned_gradient,double* conditioned_normals,
    double* exponents) noexcept
{
    const auto representable=[](double before,double after) {
        return std::isfinite(after) && (before==0 || after!=0);
    };
    for (std::size_t i=0;i<n;++i) {
        if (!std::isfinite(hessian[i*n+i])) return false;
        exponents[i]=std::nearbyint(-0.5*std::log2(std::max(
            std::abs(hessian[i*n+i]),std::numeric_limits<double>::epsilon())));
        conditioned_gradient[i]=std::scalbn(gradient[i],int(exponents[i]));
        if (!representable(gradient[i],conditioned_gradient[i])
            || !representable(units[i],std::scalbn(units[i],int(exponents[i])))) return false;
    }
    for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j) {
        const auto index=i*n+j;
        conditioned_hessian[index]=std::scalbn(hessian[index],int(exponents[i]+exponents[j]));
        if (!representable(hessian[index],conditioned_hessian[index])) return false;
    }
    for (std::size_t row=0;row<rows;++row) for (std::size_t i=0;i<n;++i) {
        const auto index=row*n+i;
        conditioned_normals[index]=std::scalbn(normals[index],int(exponents[i]));
        if (!representable(normals[index],conditioned_normals[index])) return false;
    }
    for (std::size_t i=0;i<n;++i) units[i]=std::scalbn(units[i],int(exponents[i]));
    return true;
}

// Minimum-norm correction J*d = discrepancy. The nonlinear evaluator owns the
// target slack; this operation neither invents margins nor changes constraints.
inline bool normal_correction(std::size_t n,std::size_t rows,const double* jacobian,
    const double* discrepancy,double* correction,void* memory,std::size_t bytes) noexcept
{
    const auto required=active_face_workspace_bytes(n);
    if (!required || rows>n || (rows && (!jacobian || !discrepancy))
        || !correction || !memory || bytes<required) return false;
    runtime::WorkspaceArena arena(memory,bytes);
    double* qr=arena.take<double>(n*n);
    double* q=arena.take<double>(n*n);
    double* tau=arena.take<double>(n);
    double* work=arena.take<double>(n);
    double* scales=arena.take<double>(n);
    if (!scales) return false;
    for (std::size_t i=0;i<rows;++i) if (!std::isfinite(discrepancy[i])) return false;
    if (!normal_space_basis(n,rows,jacobian,qr,q,tau,work,false,scales)) return false;
    // R' z = discrepancy, followed by d = Q z.
    for (std::size_t i=0;i<rows;++i) {
        double value=discrepancy[i]/scales[i];
        for (std::size_t j=0;j<i;++j) value-=qr[j+i*n]*work[j];
        work[i]=value/qr[i+i*n];
    }
    for (std::size_t i=0;i<n;++i) {
        correction[i]=0;
        for (std::size_t j=0;j<rows;++j) correction[i]+=q[i+j*n]*work[j];
        if (!std::isfinite(correction[i])) return false;
    }
    return true;
}

inline std::size_t projection_workspace_bytes(std::size_t n,std::size_t m) noexcept
{
    const auto inner=runtime::runtime_kraft_workspace_bytes(n,m);
    if (!inner) return 0;
    runtime::WorkspaceSizer size;
    size.add<double>(n);size.add<double>(n);size.add<double>(n);
    size.add<double>(m);size.add<std::byte>(inner);
    return size.size();
}

// Working-set evidence only. The returned point/multipliers require an independent
// primal-dual certificate; a successful SQP exit is not that certificate.
inline kraft::Result projection_multipliers(std::size_t n,std::size_t m,
    const double* gradient,const double* normals,const double* gaps,
    double* point,double* dual,void* memory,std::size_t bytes) noexcept
{
    const auto required=projection_workspace_bytes(n,m);
    std::size_t matrix;
    if (!required || !gradient || (m && (!normals || !gaps || !dual))
        || !point || !memory || bytes<required || !runtime::checked_product(n,m,matrix)) return {};
    for (std::size_t i=0;i<n;++i) if (!std::isfinite(gradient[i])) return {};
    for (std::size_t i=0;i<matrix;++i) if (!std::isfinite(normals[i])) return {};
    for (std::size_t i=0;i<m;++i) if (!std::isfinite(gaps[i]) || gaps[i]<0) return {};
    runtime::WorkspaceArena arena(memory,bytes);
    double* lower=arena.take<double>(n);
    double* upper=arena.take<double>(n);
    double* scaled_gradient=arena.take<double>(n);
    double* scaled_gaps=arena.take<double>(m);
    const auto inner=runtime::runtime_kraft_workspace_bytes(n,m);
    void* state_memory=arena.take<std::byte>(inner);
    runtime::RuntimeKraftWorkspace state;
    if (!state_memory || !runtime::bind_runtime_kraft_workspace(n,m,state_memory,inner,state)) return {};
    // x=2^exponent*y and f(x)/2^(2*exponent) preserve the projection and its
    // nonzero gaps. Restore both primal and dual units before certification.
    double magnitude=1.0;
    for (std::size_t i=0;i<n;++i) magnitude=std::max(magnitude,std::abs(gradient[i]));
    const int exponent=std::ilogb(magnitude);
    for (std::size_t i=0;i<n;++i) {
        scaled_gradient[i]=std::scalbn(gradient[i],-exponent);
        if (gradient[i]!=0 && scaled_gradient[i]==0) return {};
    }
    for (std::size_t row=0;row<m;++row) {
        scaled_gaps[row]=std::scalbn(gaps[row],-exponent);
        if (gaps[row]!=0 && scaled_gaps[row]==0) return {};
    }
    std::fill_n(lower,n,std::numeric_limits<double>::quiet_NaN());
    std::fill_n(upper,n,std::numeric_limits<double>::quiet_NaN());
    std::fill_n(point,n,0.0);
    if (m) std::fill_n(dual,m,0.0);
    auto result=direct_projection(n,m,scaled_gradient,normals,scaled_gaps,
        lower,upper,point,state);
    for (std::size_t i=0;i<n;++i) {
        point[i]=std::scalbn(point[i],exponent);
        if (!std::isfinite(point[i])) result.mode=-101;
    }
    if (result.mode==0) for (std::size_t row=0;row<m;++row) {
        dual[row]=std::scalbn(state.multipliers[row],exponent);
        if (!std::isfinite(dual[row])) result.mode=-101;
    }
    result.value=std::scalbn(result.value,2*exponent);
    result.maximum_violation=std::scalbn(result.maximum_violation,exponent);
    if (!std::isfinite(result.value) || !std::isfinite(result.maximum_violation)) result.mode=-101;
    return result;
}

// Bounds the infinity norm of the exact projection for the supplied floating-
// point linearized problem. The primal and dual need not be optimal. Invalid
// proposals never become zero certificates; an infeasible primal is retracted.
inline double primal_dual_bound(std::size_t n,std::size_t m,const double* gradient,
    const double* normals,const double* gaps,const double* point,const double* dual) noexcept
{
    constexpr double infinity=std::numeric_limits<double>::infinity();
    std::size_t matrix;
    if (!n || !gradient || !point || (m && (!normals || !gaps || !dual))
        || !runtime::checked_product(n,m,matrix)) return infinity;
    for (std::size_t i=0;i<n;++i)
        if (!std::isfinite(gradient[i]) || !std::isfinite(point[i])) return infinity;
    for (std::size_t i=0;i<matrix;++i) if (!std::isfinite(normals[i])) return infinity;
    for (std::size_t i=0;i<m;++i)
        if (!std::isfinite(gaps[i]) || gaps[i]<0 || !std::isfinite(dual[i]) || dual[i]<0)
            return infinity;
    using Extended=long double;
    const auto movement=[&](std::size_t row,Extended fraction,Extended* error=nullptr) {
        Extended value=0,magnitude=0;
        for (std::size_t i=0;i<n;++i) {
            const Extended term=Extended(normals[row*n+i])*(Extended(point[i])*fraction);
            value+=term;magnitude+=std::abs(term);
        }
        if (error) *error=8*(1+Extended(n))*(std::numeric_limits<Extended>::epsilon()
            *(magnitude+Extended(gaps[row]))
            +std::numeric_limits<Extended>::denorm_min());
        return value;
    };
    double fraction=1;
    bool retracted=false;
    for (std::size_t row=0;row<m;++row) {
        Extended error=0;
        const auto value=movement(row,1,&error);
        // Reserve one dot-error allowance for the subsequent feasibility check.
        const auto change=value-2*error;
        if (!std::isfinite(change)) return infinity;
        if (change+Extended(gaps[row])<0) {
            fraction=std::min(fraction,static_cast<double>(Extended(gaps[row])/-change));
            retracted=true;
        }
    }
    if (retracted) fraction=std::nextafter(std::max(0.0,fraction),0.0);
    for (std::size_t row=0;row<m;++row) {
        Extended error=0;
        const auto change=movement(row,fraction,&error);
        if (change-error+Extended(gaps[row])<0) {fraction=0;break;}
    }
    Extended complementarity=0;
    for (std::size_t row=0;row<m;++row) {
        Extended error=0;
        const auto change=movement(row,fraction,&error);
        const auto upper_slack=std::nextafter(change+Extended(gaps[row])+error,
            std::numeric_limits<Extended>::infinity());
        complementarity+=Extended(dual[row])*upper_slack;
    }
    // Slack error enters the gap before its square root, not as an O(eps)
    // allowance afterward. Sum only nonnegative upper products.
    complementarity=complementarity*(1+8*Extended(m)*std::numeric_limits<Extended>::epsilon())
        +8*Extended(m)*std::numeric_limits<Extended>::denorm_min();
    if (!(complementarity>=0) || !std::isfinite(complementarity)) return infinity;
    Extended residual_norm=0,force_scale=0,gradient_norm=0,point_norm=0;
    for (std::size_t i=0;i<n;++i) {
        Extended force=0,absolute_force=0;
        for (std::size_t row=0;row<m;++row) {
            const Extended value=Extended(normals[row*n+i])*Extended(dual[row]);
            force+=value;absolute_force+=std::abs(value);
        }
        const Extended primal=Extended(point[i])*Extended(fraction);
        residual_norm=std::hypot(residual_norm,primal+Extended(gradient[i])-force);
        force_scale=std::hypot(force_scale,std::abs(Extended(gradient[i]))+absolute_force);
        gradient_norm=std::hypot(gradient_norm,Extended(gradient[i]));
        point_norm=std::max(point_norm,std::abs(primal));
    }
    const Extended slack_error=std::sqrt(2*complementarity);
    const Extended roundoff=8*std::numeric_limits<Extended>::epsilon()*(1+Extended(m)+Extended(n))
        *(force_scale+slack_error)
        +8*(1+Extended(m)+Extended(n))*std::numeric_limits<Extended>::denorm_min();
    const Extended bound=point_norm+std::hypot(residual_norm,slack_error)+roundoff;
    if (!std::isfinite(bound) || !std::isfinite(gradient_norm)) return infinity;
    // Relative allowances alone vanish when iterative hypot rounds in the subnormal range.
    const Extended gradient_cap=gradient_norm*(1+8*n*std::numeric_limits<Extended>::epsilon())
        +8*Extended(n)*std::numeric_limits<Extended>::denorm_min();
    const double upper=std::nextafter(static_cast<double>(std::min(bound,gradient_cap)),infinity);
    return std::isfinite(upper)?upper:infinity;
}

struct StationarityEvidence {
    double bound=std::numeric_limits<double>::infinity();
    bool projected=false;
    kraft::Result projection{};
};

// Borrowed evidence, valid until this arena is reused. Row identities refer to
// original constraints, then lower bounds, then upper bounds, without compaction.
struct ProjectionGeometry {
    std::size_t dimension=0,rows=0;
    const double *normals=nullptr,*gaps=nullptr,*scales=nullptr,*dual=nullptr;
    const std::size_t* identities=nullptr;
    const double* point=nullptr;
};

inline std::size_t stationarity_workspace_bytes(std::size_t n,std::size_t constraints) noexcept
{
    std::size_t twice,rows,matrix;
    if (!runtime::checked_product(n,2,twice)
        || !runtime::checked_sum(constraints,twice,rows)
        || !runtime::checked_product(rows,n,matrix)) return 0;
    const auto projection=projection_workspace_bytes(n,rows);
    if (!projection) return 0;
    runtime::WorkspaceSizer size;
    size.add<double>(matrix);
    size.add<double>(rows);
    size.add<double>(rows);
    size.add<double>(rows);
    size.add<std::size_t>(rows);
    size.add<double>(n);
    size.add<std::byte>(projection);
    return size.size();
}

// Units and sample scaling are supplied by the model-owned assessment adapter.
// Jacobians use the native column-major constraint contract. Infinite step bounds
// are allowed; NaNs, infeasible origins and unavailable derivatives are not.
inline StationarityEvidence stationarity_bound(std::size_t n,std::size_t constraints,
    const double* gradient,const double* lower_step,const double* upper_step,
    const double* jacobian,const double* slack,void* memory,std::size_t bytes,
    ProjectionGeometry* prepared=nullptr) noexcept
{
    if (prepared) *prepared={};
    StationarityEvidence result;
    const auto required=stationarity_workspace_bytes(n,constraints);
    if (!required || !gradient || !lower_step || !upper_step || !memory || bytes<required
        || (constraints && (!jacobian || !slack))) return result;
    for (std::size_t i=0;i<n;++i)
        if (!std::isfinite(gradient[i]) || std::isnan(lower_step[i]) || std::isnan(upper_step[i])
            || lower_step[i]>0 || upper_step[i]<0) return result;
    for (std::size_t i=0;i<n*constraints;++i) if (!std::isfinite(jacobian[i])) return result;
    for (std::size_t i=0;i<constraints;++i)
        if (!std::isfinite(slack[i]) || slack[i]<0) return result;
    double box_bound=0;
    for (std::size_t i=0;i<n;++i) {
        const bool supported=(lower_step[i]==0 && gradient[i]>0)
            || (upper_step[i]==0 && gradient[i]<0);
        if (!supported) box_bound=std::hypot(box_bound,gradient[i]);
    }
    if (!box_bound) {result.bound=0;return result;}
    constexpr double infinity=std::numeric_limits<double>::infinity();
    box_bound=std::nextafter(box_bound*(1+8*n*std::numeric_limits<double>::epsilon()),infinity);
    result.bound=box_bound;
    if (!std::isfinite(box_bound)) return result;
    const auto maximum_rows=constraints+2*n;
    runtime::WorkspaceArena arena(memory,bytes);
    double* normals=arena.take<double>(maximum_rows*n);
    double* gaps=arena.take<double>(maximum_rows);
    double* dual=arena.take<double>(maximum_rows);
    double* scales=arena.take<double>(maximum_rows);
    auto* identities=arena.take<std::size_t>(maximum_rows);
    double* point=arena.take<double>(n);
    const auto projection_bytes=projection_workspace_bytes(n,maximum_rows);
    void* projection_memory=arena.take<std::byte>(projection_bytes);
    if (!projection_memory) return StationarityEvidence{};
    std::size_t rows=0;
    for (std::size_t row=0;row<constraints;++row) {
        double scale=0;
        for (std::size_t i=0;i<n;++i) scale=std::max(scale,std::abs(jacobian[row+i*constraints]));
        if (!scale) continue;
        scales[rows]=scale;
        identities[rows]=row;
        gaps[rows]=slack[row]/scale;
        if (!std::isfinite(gaps[rows]) || (slack[row]!=0 && gaps[rows]==0)) return result;
        for (std::size_t i=0;i<n;++i) {
            const double value=jacobian[row+i*constraints];
            normals[rows*n+i]=value/scale;
            if (value!=0 && normals[rows*n+i]==0) return result;
        }
        ++rows;
    }
    for (bool lower : {true,false}) for (std::size_t i=0;i<n;++i) {
        const double step=lower?lower_step[i]:upper_step[i];
        if (!std::isfinite(step)) continue;
        std::fill_n(normals+rows*n,n,0.0);
        normals[rows*n+i]=lower?1.0:-1.0;
        gaps[rows]=lower?-step:step;
        scales[rows]=1.0;
        identities[rows]=constraints+(lower?0:n)+i;
        ++rows;
    }
    if (!rows) {
        if (prepared) {
            for (std::size_t i=0;i<n;++i) point[i]=-gradient[i];
            *prepared={n,0,normals,gaps,scales,dual,identities,point};
        }
        return result;
    }
    result.projected=true;
    result.projection=projection_multipliers(n,rows,gradient,normals,gaps,point,dual,
        projection_memory,projection_bytes);
    if (!result.projection.success()) return result;
    if (prepared) *prepared={n,rows,normals,gaps,scales,dual,identities,point};
    const double certificate=primal_dual_bound(n,rows,gradient,normals,gaps,point,dual);
    if (std::isfinite(certificate)) result.bound=std::min(box_bound,certificate);
    return result;
}

} // namespace vlk::experiment::curvature
