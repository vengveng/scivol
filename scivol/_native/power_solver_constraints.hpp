// SCIVOL-NATIVE: CORE; shared linear EGARCH/APARCH solver constraints.
#pragma once

#include "aparch_transform_policy.hpp"
#include "mean_solver.hpp"

namespace vlk::experiment::runtime {

template<class Program,auto StaticLayout=nullptr>
struct PowerConstraints {
    static constexpr bool fixed=StaticLayout!=nullptr;
    Program program;
    mean::PolynomialConstraints mean;
    bool transformed;
    double* local_values;
    double* local_jacobian;
    const auto& layout() const noexcept
    {
        if constexpr (StaticLayout==nullptr) return program.layout;
        else {
            static_assert(StaticLayout->valid);
            return *StaticLayout;
        }
    }
    template<auto Field>
    constexpr auto field() const noexcept
    {
        if constexpr (StaticLayout==nullptr) return program.layout.*Field;
        else return std::integral_constant<std::size_t,((*StaticLayout).*Field)>{};
    }
    std::size_t dimension() const noexcept {return layout().parameters;}
    std::size_t count() const noexcept
    {
        if (transformed) return mean.count();
        if constexpr (Program::aparch)
            return mean.count()+1u+runtime::aparch_transform::tail_coupled_delta<typename Program::density_type>;
        else return mean.count()+2u*layout().q_vol;
    }
    bool feasible_values(const double* values) const noexcept
    {
        if (count() && !values) return false;
        for (std::size_t i=0;i<count();++i)
            if (!std::isfinite(values[i]) || !(values[i]>0)) return false;
        return true;
    }
    bool evaluate(kraft::ConstraintRequest request,const double* theta,
        double* values,double* jacobian,ConstraintColumns columns={}) const noexcept
    {
        const auto rows=count(),k=dimension(),m=mean.count();
        const bool derivatives=request!=kraft::ConstraintRequest::value;
        if (transformed) return mean.evaluate(request,theta,values,jacobian);
        if (derivatives) std::fill_n(jacobian,rows*k,0.0);
        if (!mean.evaluate(request,theta,local_values,local_jacobian)) return false;
        if (request!=kraft::ConstraintRequest::jacobian) std::copy_n(local_values,m,values);
        if (derivatives)
            for (std::size_t j=0;j<k;++j)
                for (std::size_t i=0;i<m;++i) jacobian[i+j*rows]=local_jacobian[i+j*m];
        const auto& layout=this->layout();
        if constexpr (Program::aparch) {
            using Density=typename Program::density_type;
            constexpr auto shapes=Density::shape_parameters;
            if (request!=kraft::ConstraintRequest::jacobian) {
                const auto persistence=generated::power_persistence<Density>(
                    aparch_transform::policy_layout<StaticLayout>(layout),theta,program.moment_values);
                if (!persistence.valid) return false;
                values[m]=persistence.slack;
            }
            if (derivatives) {
                const auto d=layout.delta_index;
                const auto lags=field<&aparch::Layout::p>();
                auto scalar=generated::make_power_moment_batch<0,Density>(theta+layout.model_parameters,
                    theta[d],lags,program.moment_values);
                auto first=generated::make_power_moment_batch<1,Density>(theta+layout.model_parameters,
                    theta[d],lags,program.moment_values);
                bool shared_derivatives=columns.contains(d);
                for (std::size_t s=0;s<shapes;++s)
                    shared_derivatives|=columns.contains(layout.model_parameters+s);
                for (std::size_t lag=0;lag<layout.p;++lag) {
                    const auto a=layout.alpha_start+lag,g=layout.gamma_start+lag;
                    if (!generated::valid_power_moment_domain<Density>(
                            theta+layout.model_parameters,theta[g],theta[d])) return false;
                    if (columns.contains(a)) {
                        const auto value=scalar(theta[g],lag);
                        if (!value.valid) return false;
                        jacobian[m+a*rows]=-value.value;
                    }
                    if (theta[a]==0.0) {
                        if (columns.contains(g)) jacobian[m+g*rows]=-0.0;
                        continue;
                    }
                    if (!shared_derivatives && !columns.contains(g)) continue;
                    auto moment=first(theta[g],lag);
                    double gradient_multiplier=theta[a];
                    if (!moment.valid && std::isfinite(theta[a]) && theta[a]!=0.0) {
                        // Include |alpha| before exponentiation when M'
                        // overflows but the requested alpha*M' is finite.
                        moment=generated::scaled_asymmetric_power_moment<1,Density>(
                            theta+layout.model_parameters,theta[g],theta[d],std::log(std::abs(theta[a])));
                        gradient_multiplier=std::copysign(1.0,theta[a]);
                    }
                    if (!moment.valid) return false;
                    if (columns.contains(g)) jacobian[m+g*rows]=-gradient_multiplier*moment.gradient[0];
                    if (columns.contains(d)) jacobian[m+d*rows]-=gradient_multiplier*moment.gradient[1];
                    for (std::size_t s=0;s<shapes;++s)
                        if (columns.contains(layout.model_parameters+s))
                            jacobian[m+(layout.model_parameters+s)*rows]-=gradient_multiplier*moment.gradient[2+s];
                }
            }
            for (std::size_t lag=0;lag<layout.q;++lag) {
                if (derivatives) jacobian[m+(layout.beta_start+lag)*rows]=-1.0;
            }
            if constexpr (runtime::aparch_transform::tail_coupled_delta<Density>) {
                if (request!=kraft::ConstraintRequest::jacobian)
                    values[m+1]=theta[layout.model_parameters]-theta[layout.delta_index];
                if (derivatives) {
                    jacobian[m+1+layout.model_parameters*rows]=1.0;
                    jacobian[m+1+layout.delta_index*rows]=-1.0;
                }
            }
        } else {
            const auto q=layout.q_vol;
            const mean::PolynomialDomain polynomial{q,q,nullptr};
            if (!polynomial.evaluate(theta+layout.beta_start,local_values,
                derivatives?local_jacobian:nullptr,mean.workspace,mean.bytes,false)) return false;
            if (request!=kraft::ConstraintRequest::jacobian) std::copy_n(local_values,2*q,values+m);
            if (derivatives)
                for (std::size_t j=0;j<q;++j)
                    for (std::size_t i=0;i<2*q;++i)
                        jacobian[m+i+(layout.beta_start+j)*rows]=local_jacobian[i*q+j];
        }
        return true;
    }
};

} // namespace vlk::experiment::runtime
