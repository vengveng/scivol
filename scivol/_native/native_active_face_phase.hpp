// SCIVOL-NATIVE: CORE; bounded typed inequality/bound refinement.
// Model adapters own raw jets, strict feasibility, units and constraint curvature.
#pragma once
#include "active_face_refinement.hpp"

namespace vlk::experiment::curvature {

enum class RefinementStatus {
    invalid_input,invalid_start,derivative_unavailable,already_certified,
    curvature_unavailable,no_direction,line_search_exhausted,accepted
};

struct RefinementOptions {
    unsigned backtracks=16,corrections=4;
    double stationarity_target=1e-6;
};

struct RefinementResult {
    RefinementStatus status=RefinementStatus::invalid_input;
    double value=std::numeric_limits<double>::infinity();
    double bound=std::numeric_limits<double>::infinity();
    double best_value=std::numeric_limits<double>::infinity();
    double best_bound=std::numeric_limits<double>::infinity();
    bool best_valid=false,best_is_accepted=false;
    std::size_t objective_requests[8]{},constraint_requests[3]{};
    std::size_t curvature_requests=0,projections=0,projection_iterations=0;
    std::size_t trials=0,corrections=0,active=0,resolved=0;
};

inline std::size_t refinement_workspace_bytes(std::size_t n,std::size_t m) noexcept
{
    std::size_t square,matrix;
    const auto projection=stationarity_workspace_bytes(n,m),algebra=active_face_workspace_bytes(n);
    if (!projection || !algebra || !runtime::checked_product(n,n,square)
        || !runtime::checked_product(n,m,matrix)) return 0;
    runtime::WorkspaceSizer size;
    for (unsigned i=0;i<13;++i) size.add<double>(n);
    for (unsigned i=0;i<2;++i) size.add<double>(m);
    for (unsigned i=0;i<2;++i) size.add<double>(matrix);
    for (unsigned i=0;i<4;++i) size.add<double>(square);
    size.add<std::size_t>(n);size.add<std::size_t>(n);size.add<unsigned char>(n);
    size.add<double>(0);size.add<std::byte>(projection);
    size.add<double>(0);size.add<std::byte>(algebra);
    return size.size();
}

template<class Model>
RefinementResult refine_active_face(const Model& model,double* point,double* best,
    const double* lower,const double* upper,const RefinementOptions& options,
    void* memory,std::size_t bytes) noexcept
{
    RefinementResult result;
    const auto n=model.dimension(),m=model.count();
    const auto required=refinement_workspace_bytes(n,m);
    const double scale=model.objective_scale();
    if (!required || !point || !best || point==best || !lower || !upper || !memory
        || bytes<required || !std::isfinite(scale) || !(scale>0)
        || !std::isfinite(options.stationarity_target) || options.stationarity_target<0
        || options.backtracks>1024) return result;
    for (std::size_t i=0;i<n;++i)
        if (!std::isfinite(point[i]) || std::isnan(lower[i]) || std::isnan(upper[i])
            || lower[i]>upper[i]) return result;
    runtime::WorkspaceArena arena(memory,bytes);
    double* initial=arena.take<double>(n);
    double* trial=arena.take<double>(n);
    double* gradient=arena.take<double>(n);
    double* units=arena.take<double>(n);
    double* base_units=arena.take<double>(n);
    double* scaled_gradient=arena.take<double>(n);
    double* low_step=arena.take<double>(n);
    double* high_step=arena.take<double>(n);
    double* direction=arena.take<double>(n);
    double* affine_direction=arena.take<double>(n);
    double* quadratic_direction=arena.take<double>(n);
    double* eigenvalues=arena.take<double>(n);
    double* correction=arena.take<double>(n);
    double* slack=arena.take<double>(m);
    double* original_slack=arena.take<double>(m);
    double* jacobian=arena.take<double>(m*n);
    double* scaled_jacobian=arena.take<double>(m*n);
    double* hessian=arena.take<double>(n*n);
    double* constraint_hessian=arena.take<double>(n*n);
    double* active_normals=arena.take<double>(n*n);
    double* correction_matrix=arena.take<double>(n*n);
    auto* nonlinear=arena.take<std::size_t>(n);
    auto* free_indices=arena.take<std::size_t>(n);
    auto* bound_face=arena.take<unsigned char>(n);
    const auto projection_bytes=stationarity_workspace_bytes(n,m),algebra_bytes=active_face_workspace_bytes(n);
    if (!arena.take<double>(0)) return result;
    void* projection_memory=arena.take<std::byte>(projection_bytes);
    if (!arena.take<double>(0)) return result;
    void* algebra_memory=arena.take<std::byte>(algebra_bytes);
    if (!algebra_memory) return result;
    std::copy_n(point,n,initial);
    const auto objective=[&](unsigned mask,const double* theta,double* value) {
        ++result.objective_requests[mask];
        if (!model.objective(mask,theta,value,mask&2?gradient:nullptr,mask&4?hessian:nullptr)) return false;
        if ((mask&1) && !std::isfinite(*value)) return false;
        if (mask&2) for (std::size_t i=0;i<n;++i) if (!std::isfinite(gradient[i])) return false;
        if (mask&4) for (std::size_t i=0;i<n*n;++i) if (!std::isfinite(hessian[i])) return false;
        return true;
    };
    const auto constraints=[&](kraft::ConstraintRequest request,const double* theta) {
        const auto index=request==kraft::ConstraintRequest::value?0u:
            request==kraft::ConstraintRequest::jacobian?1u:2u;
        ++result.constraint_requests[index];
        if (!model.constraints(request,theta,slack,jacobian)) return false;
        if (index!=1) for (std::size_t i=0;i<m;++i) if (!std::isfinite(slack[i])) return false;
        if (index!=0) for (std::size_t i=0;i<m*n;++i) if (!std::isfinite(jacobian[i])) return false;
        return true;
    };
    const auto feasible=[&](const double* theta) {
        for (std::size_t i=0;i<n;++i)
            if (!std::isfinite(theta[i]) || theta[i]<lower[i] || theta[i]>upper[i]) return false;
        return model.feasible(theta,slack);
    };
    const auto assess=[&](const double* theta,ProjectionGeometry* geometry=nullptr) {
        if (geometry) *geometry={};
        if (!constraints(kraft::ConstraintRequest::jacobian,theta) || !model.parameter_units(theta,units))
            return std::numeric_limits<double>::infinity();
        for (std::size_t i=0;i<n;++i) {
            if (!std::isfinite(units[i]) || !(units[i]>0)) return std::numeric_limits<double>::infinity();
            scaled_gradient[i]=gradient[i]*units[i]*scale;
            low_step[i]=(lower[i]-theta[i])/units[i];
            high_step[i]=(upper[i]-theta[i])/units[i];
            for (std::size_t row=0;row<m;++row) scaled_jacobian[row+i*m]=jacobian[row+i*m]*units[i];
        }
        const auto evidence=stationarity_bound(n,m,scaled_gradient,low_step,high_step,
            scaled_jacobian,slack,projection_memory,projection_bytes,geometry);
        result.projections+=evidence.projected;
        result.projection_iterations+=std::max(std::int64_t(0),evidence.projection.iterations);
        return evidence.bound;
    };
    if (!constraints(kraft::ConstraintRequest::value,initial) || !feasible(initial)
        || !objective(1,initial,&result.value)) {
        result.status=RefinementStatus::invalid_start;return result;
    }
    result.best_valid=true;result.best_value=result.value;std::copy_n(initial,n,best);
    result.best_is_accepted=true;
    if (!objective(2,initial,nullptr)) {
        result.status=RefinementStatus::derivative_unavailable;return result;
    }
    ProjectionGeometry geometry;
    result.bound=assess(initial,&geometry);result.best_bound=result.bound;
    if (result.bound<=options.stationarity_target) {
        result.status=RefinementStatus::already_certified;return result;
    }
    if (!options.backtracks) {result.status=RefinementStatus::line_search_exhausted;return result;}
    if (!geometry.dimension) {result.status=RefinementStatus::no_direction;return result;}
    if (!objective(4,initial,nullptr)) {result.status=RefinementStatus::curvature_unavailable;return result;}
    std::copy_n(units,n,base_units);std::copy_n(slack,m,original_slack);
    std::fill_n(bound_face,n,0);
    // Reuse the just-certified projection. Only selected rows survive arena reuse.
    std::fill_n(slack,m,0.0);
    std::size_t nonlinear_count=0,free_count=0;
    bool normal_shift=false;
    // Unit bound normals first give exact coordinate eliminations in QR. Tiny
    // leaked bound components can otherwise amplify large cross-curvature.
    for (bool bounds_first : {true,false})
    for (std::size_t row=0;row<geometry.rows;++row)
    if (geometry.dual[row]>0 && (geometry.identities[row]>=m)==bounds_first) {
        if (result.active==n) {result.status=RefinementStatus::no_direction;return result;}
        std::copy_n(geometry.normals+row*n,n,active_normals+result.active*n);
        const auto identity=geometry.identities[row];
        eigenvalues[result.active++]=-geometry.gaps[row];
        normal_shift|=eigenvalues[result.active-1]!=0;
        if (identity<m) {
            nonlinear[nonlinear_count++]=identity;
            slack[identity]=geometry.dual[row]/geometry.scales[row];
        } else bound_face[(identity-m)%n]=identity<m+n?1:2;
    }
    ++result.curvature_requests;
    if (!model.constraint_curvature(initial,slack,constraint_hessian)) {
        result.status=RefinementStatus::curvature_unavailable;return result;
    }
    for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j)
        hessian[i*n+j]=(hessian[i*n+j]*scale-constraint_hessian[i*n+j])*units[i]*units[j];
    const bool conditioned=condition_curvature(n,result.active,hessian,scaled_gradient,
        active_normals,base_units,correction_matrix,low_step,constraint_hessian,high_step);
    const double* search_hessian=conditioned?correction_matrix:hessian;
    const double* search_gradient=conditioned?low_step:scaled_gradient;
    const double* search_normals=conditioned?constraint_hessian:active_normals;
    // Projection-active rows can be away from the current point. First reach
    // their linearized face, then minimize in its tangent using g + H*d_normal.
    if (normal_shift) {
        if (!normal_correction(n,result.active,search_normals,eigenvalues,correction,
            algebra_memory,algebra_bytes)) {
            result.status=RefinementStatus::no_direction;return result;
        }
        for (std::size_t i=0;i<n;++i) {
            high_step[i]=search_gradient[i];
            for (std::size_t j=0;j<n;++j) high_step[i]+=search_hessian[i*n+j]*correction[j];
        }
    } else std::fill_n(correction,n,0.0);
    if (!active_face_direction(n,result.active,search_normals,search_hessian,search_gradient,
        direction,eigenvalues,result.resolved,algebra_memory,algebra_bytes,
        normal_shift?high_step:nullptr,normal_shift?affine_direction:nullptr)) {
        result.status=RefinementStatus::no_direction;return result;
    }
    double slopes[3]={};
    for (std::size_t i=0;i<n;++i) {
        if (bound_face[i]) {
            direction[i]=0;
            if (normal_shift) affine_direction[i]=((bound_face[i]==1?lower[i]:upper[i])-initial[i])/base_units[i];
        } else {
            if (normal_shift) affine_direction[i]+=correction[i];
            free_indices[free_count++]=i;
        }
        slopes[0]+=search_gradient[i]*direction[i];
        if (normal_shift) slopes[1]+=search_gradient[i]*affine_direction[i];
    }
    if (nonlinear_count>free_count) {
        result.status=RefinementStatus::no_direction;return result;
    }
    // A positive-curvature tangent can omit useful descent. The alternative
    // solves the full linearized domain in the already conditioned coordinates.
    bool quadratic_plan=false;
    if (conditioned && result.resolved<n-result.active) {
        for (std::size_t i=0;i<n;++i) {
            scaled_gradient[i]=gradient[i]*base_units[i]*scale;
            low_step[i]=(lower[i]-initial[i])/base_units[i];
            high_step[i]=(upper[i]-initial[i])/base_units[i];
            for (std::size_t row=0;row<m;++row)
                scaled_jacobian[row+i*m]=jacobian[row+i*m]*base_units[i];
        }
        ProjectionGeometry quadratic;
        const auto proposal=stationarity_bound(n,m,scaled_gradient,low_step,high_step,
            scaled_jacobian,original_slack,projection_memory,projection_bytes,&quadratic);
        result.projections+=proposal.projected;
        result.projection_iterations+=std::max(std::int64_t(0),proposal.projection.iterations);
        if (quadratic.dimension && quadratic.point) {
            std::copy_n(quadratic.point,n,quadratic_direction);
            for (std::size_t i=0;i<n;++i) slopes[2]+=scaled_gradient[i]*quadratic_direction[i];
            quadratic_plan=std::isfinite(slopes[2]) && slopes[2]<0;
        }
    }
    result.status=RefinementStatus::no_direction;
    // All proposals share point preparation and the requested model Hessian.
    for (unsigned plan=0;plan<3;++plan) {
    if ((plan==1 && !normal_shift) || (plan==2 && !quadratic_plan)) continue;
    if (!std::isfinite(slopes[plan]) || !(slopes[plan]<0)) continue;
    if (result.status==RefinementStatus::no_direction) result.status=RefinementStatus::line_search_exhausted;
    const double* step=plan==2?quadratic_direction:(plan?affine_direction:direction);
    for (unsigned line=0;line<options.backtracks;++line) {
        ++result.trials;
        const double fraction=std::ldexp(1.0,-int(line));
        for (std::size_t i=0;i<n;++i) {
            if (plan!=2 && bound_face[i]) {
                const double target=plan?(bound_face[i]==1?lower[i]:upper[i]):initial[i];
                trial[i]=line?initial[i]+fraction*(target-initial[i]):target;
            } else trial[i]=initial[i]+base_units[i]*step[i]*fraction;
            trial[i]=std::clamp(trial[i],lower[i],upper[i]);
        }
        bool valid=true;
        for (unsigned iteration=0;plan!=2 && iteration<options.corrections && nonlinear_count;++iteration) {
            if (!constraints(kraft::ConstraintRequest::value_jacobian,trial)) {valid=false;break;}
            bool discrepancy=false;
            for (std::size_t row=0;row<nonlinear_count;++row) {
                const double target=original_slack[nonlinear[row]]*(plan==1?1-fraction:1);
                eigenvalues[row]=target-slack[nonlinear[row]];
                discrepancy|=eigenvalues[row]!=0;
                for (std::size_t j=0;j<free_count;++j)
                    correction_matrix[row*free_count+j]=jacobian[nonlinear[row]+free_indices[j]*m]*base_units[free_indices[j]];
            }
            if (!discrepancy) break;
            ++result.corrections;
            if (!free_count || !normal_correction(free_count,nonlinear_count,correction_matrix,
                eigenvalues,correction,algebra_memory,algebra_bytes)) {valid=false;break;}
            for (std::size_t j=0;j<free_count;++j) {
                const auto i=free_indices[j];
                trial[i]=std::clamp(trial[i]+base_units[i]*correction[j],lower[i],upper[i]);
            }
        }
        if (!valid || !constraints(kraft::ConstraintRequest::value,trial) || !feasible(trial)) continue;
        double value;
        if (!objective(1,trial,&value)) continue;
        const bool improved=value<result.best_value;
        if (improved) {
            result.best_value=value;std::copy_n(trial,n,best);
            result.best_bound=std::numeric_limits<double>::infinity();result.best_is_accepted=false;
        }
        if (value>result.value || !objective(2,trial,nullptr)) continue;
        const double bound=assess(trial);
        if (improved) result.best_bound=bound;
        if (bound<result.bound) {
            std::copy_n(trial,n,point);result.value=value;result.bound=bound;
            result.status=RefinementStatus::accepted;
            if (value<=result.best_value) {
                std::copy_n(trial,n,best);result.best_value=value;result.best_bound=bound;
                result.best_is_accepted=true;
            }
            break;
        }
    }
    if (result.status==RefinementStatus::accepted && result.bound<=options.stationarity_target) break;
    }
    return result;
}

} // namespace vlk::experiment::curvature
