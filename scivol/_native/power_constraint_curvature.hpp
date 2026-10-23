// SCIVOL-NATIVE: shared model-owned weighted EGARCH/APARCH constraint curvature.
#pragma once

#include "power_solver_constraints.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vlk::experiment::runtime {

// Sum weights[row] * Hessian(c_row), in row-major coordinates of constraints.
// Objective scaling and the Lagrangian's multiplier sign belong to the caller.
template<class Program,auto StaticLayout=nullptr>
struct PowerConstraintCurvature {
    const PowerConstraints<Program,StaticLayout>& constraints;

private:
    struct WorkspacePlan {
        std::size_t square=0, matrix=1, scratch=0, rows=0;
    };

    static long double weighted_product(double weight,double value,double coefficient=1.) noexcept
    {
        // Do not round weight*coefficient to zero before multiplying a large
        // derivative. long double has binary64 range on some supported hosts.
        int w=0,v=0,c=0;
        const auto wm=std::frexp(weight,&w),vm=std::frexp(value,&v);
        const auto cm=std::frexp(coefficient,&c);
        return std::scalbn(static_cast<long double>(wm)*vm*cm,w+v+c);
    }

    bool workspace_plan(WorkspacePlan& plan) const noexcept
    {
        const auto k=constraints.dimension();
        if (!k || constraints.mean.dimension()!=k || !constraints.layout().valid
            || !checked_product(k,k,plan.square)) return false;
        const auto fits=[k](std::size_t offset,std::size_t width) {
            return offset<=k && width<=k-offset;
        };
        const auto include=[&](const mean::PolynomialDomain& domain,std::size_t offset) {
            std::size_t rows,square;
            if (!fits(offset,domain.parameters)
                || !checked_product(2u,domain.order,rows)
                || !checked_sum(plan.rows,rows,plan.rows)) return false;
            if (!domain.order) return true;
            const auto scratch=domain.curvature_workspace_bytes();
            if (!scratch || !checked_product(domain.parameters,domain.parameters,square))
                return false;
            plan.matrix=std::max(plan.matrix,square);
            plan.scratch=std::max(plan.scratch,scratch);
            return true;
        };
        for (const auto& block : {constraints.mean.plan.first,constraints.mean.plan.second})
            if (!include(block.domain,block.offset)) return false;
        if (!constraints.transformed) {
            const auto& layout=constraints.layout();
            if constexpr (Program::aparch) {
                using Density=typename Program::density_type;
                if (!fits(layout.alpha_start,layout.p) || !fits(layout.gamma_start,layout.p)
                    || !fits(layout.beta_start,layout.q) || !fits(layout.delta_index,1u)
                    || !fits(layout.model_parameters,Density::shape_parameters)
                    || !checked_sum(plan.rows,1u+aparch_transform::tail_coupled_delta<Density>,
                                    plan.rows)) return false;
            } else if (!include({layout.q_vol,layout.q_vol,nullptr},layout.beta_start))
                return false;
        }
        return true;
    }

    static std::size_t workspace_bytes(const WorkspacePlan& plan) noexcept
    {
        WorkspaceSizer size;
        size.add<long double>(plan.square);
        size.add<double>(plan.matrix);
        size.add<std::byte>(plan.scratch);
        size.add<std::byte>(alignof(long double)-1u);
        return size.size();
    }

    bool add_polynomial(const mean::PolynomialDomain& domain,std::size_t offset,
        std::size_t row,const double* theta,const double* weights,long double* sum,
        double* local,void* scratch,std::size_t bytes) const noexcept
    {
        bool active=false;
        for (std::size_t i=0;i<2u*domain.order;++i) active|=weights[row+i]!=0.;
        if (!active) return true;
        if (!domain.weighted_curvature(theta+offset,weights+row,local,scratch,bytes))
            return false;
        const auto k=constraints.dimension();
        for (std::size_t i=0;i<domain.parameters;++i)
            for (std::size_t j=0;j<domain.parameters;++j)
                sum[(offset+i)*k+offset+j]+=local[i*domain.parameters+j];
        return true;
    }

    template<class Density>
    bool add_persistence(const double* theta,double weight,long double* sum) const noexcept
    {
        if (weight==0.) return true;
        const auto& layout=constraints.layout();
        const auto k=constraints.dimension(),d=layout.delta_index;
        const auto* shape=theta+layout.model_parameters;
        constexpr auto width=Density::shape_parameters+2u;
        auto first=generated::make_power_moment_batch<1,Density>(
            shape,theta[d],constraints.template field<&aparch::Layout::p>(),constraints.program.moment_values);
        auto second=generated::make_power_moment_batch<2,Density>(
            shape,theta[d],constraints.template field<&aparch::Layout::p>(),constraints.program.moment_values);
        for (std::size_t lag=0;lag<layout.p;++lag) {
            const auto a=layout.alpha_start+lag,g=layout.gamma_start+lag;
            const double alpha=theta[a];
            std::array<std::size_t,width> indices{g,d};
            for (std::size_t s=2;s<width;++s) indices[s]=layout.model_parameters+s-2u;
            // Zero alpha removes alpha*M'', but never the mixed alpha*M' block.
            const auto raw=alpha==0. ? first(theta[g],lag) : second(theta[g],lag);
            auto gradient=raw;
            double gradient_scale=-weight;
            if (!gradient.valid) {
                if (alpha!=0.) gradient=first(theta[g],lag);
                if (!gradient.valid) {
                    gradient=generated::scaled_asymmetric_power_moment<1,Density>(
                        shape,theta[g],theta[d],std::log(std::abs(weight)));
                    gradient_scale=-std::copysign(1.,weight);
                }
            }
            if (!gradient.valid) return false;
            for (std::size_t i=0;i<width;++i) {
                const long double mixed=weighted_product(gradient_scale,gradient.gradient[i]);
                sum[a*k+indices[i]]+=mixed;
                sum[indices[i]*k+a]+=mixed;
            }
            if (alpha==0.) continue;
            auto hessian=raw;
            double hessian_scale=-weight,hessian_coefficient=alpha;
            if (!hessian.valid) {
                // The multiplier is constant in the local moment jet; alpha's
                // derivatives are the mixed block assembled above.
                hessian=generated::scaled_asymmetric_power_moment<2,Density>(
                    shape,theta[g],theta[d],std::log(std::abs(weight))+std::log(std::abs(alpha)));
                hessian_scale=-(std::copysign(1.,weight)*std::copysign(1.,alpha));
                hessian_coefficient=1.;
            }
            if (!hessian.valid) return false;
            for (std::size_t i=0;i<width;++i) for (std::size_t j=i;j<width;++j) {
                const auto left=indices[i],right=indices[j];
                const long double value=weighted_product(hessian_scale,hessian.hessian[
                    model::packed_upper_index(width,i,j)],hessian_coefficient);
                sum[left*k+right]+=value;
                if (left!=right) sum[right*k+left]+=value;
            }
        }
        // Persistence's beta terms and the tail-coupled nu-delta row are affine.
        return true;
    }

public:
    std::size_t workspace_bytes() const noexcept
    {
        WorkspacePlan plan;
        return workspace_plan(plan) ? workspace_bytes(plan) : 0u;
    }

    // False means invalid inputs/arena or unavailable representable curvature.
    // On failure, output is unchanged. Memory must not alias inputs or output.
    bool weighted_hessian(const double* theta,const double* weights,double* output,
        void* memory,std::size_t bytes) const noexcept
    {
        WorkspacePlan plan;
        if (!workspace_plan(plan)) return false;
        const auto required=workspace_bytes(plan),k=constraints.dimension();
        if (!required || !theta || (plan.rows && !weights) || !output || !memory
            || bytes<required) return false;
        const auto square=k*k; // workspace_plan already checked the product.
        for (std::size_t i=0;i<k;++i) if (!std::isfinite(theta[i])) return false;
        for (std::size_t i=0;i<plan.rows;++i) if (!std::isfinite(weights[i])) return false;
        WorkspaceArena arena(memory,bytes);
        auto* sum=arena.take<long double>(square);
        auto* local=arena.take<double>(plan.matrix);
        auto* scratch=arena.take<std::byte>(plan.scratch);
        if (!sum || !local || !scratch) return false;
        std::fill_n(sum,square,0.L);
        std::size_t row=0;
        for (const auto& block : {constraints.mean.plan.first,constraints.mean.plan.second}) {
            if (!add_polynomial(block.domain,block.offset,row,theta,weights,sum,
                                local,scratch,plan.scratch)) return false;
            row+=2u*block.domain.order;
        }
        if (!constraints.transformed) {
            if constexpr (Program::aparch) {
                if (!add_persistence<typename Program::density_type>(theta,weights[row],sum))
                    return false;
            } else {
                const auto& layout=constraints.layout();
                if (!add_polynomial({layout.q_vol,layout.q_vol,nullptr},layout.beta_start,
                    row,theta,weights,sum,local,scratch,plan.scratch)) return false;
            }
        }
        for (std::size_t i=0;i<square;++i)
            if (!std::isfinite(sum[i]) || std::abs(sum[i])>std::numeric_limits<double>::max())
                return false;
        for (std::size_t i=0;i<square;++i) output[i]=static_cast<double>(sum[i]);
        return true;
    }
};

} // namespace vlk::experiment::runtime
