// SCIVOL-NATIVE: CORE; shared linear EGARCH/APARCH solver constraints.
#pragma once

#include "aparch_transform_policy.hpp"
#include "mean_solver.hpp"

namespace vlk::experiment::runtime {

template<class Program>
struct PowerConstraints {
    Program program;
    mean::PolynomialConstraints mean;
    bool transformed;
    double* local_values;
    double* local_jacobian;
    std::size_t dimension() const noexcept {return program.layout.parameters;}
    std::size_t count() const noexcept
    {
        if (transformed) return mean.count();
        if constexpr (Program::aparch)
            return mean.count()+1u+runtime::aparch_transform::tail_coupled_delta<typename Program::density_type>;
        else return mean.count()+2u*program.layout.q_vol;
    }
    bool feasible_values(const double* values) const noexcept
    {
        if (count() && !values) return false;
        for (std::size_t i=0;i<count();++i)
            if (!std::isfinite(values[i]) || !(values[i]>0)) return false;
        return true;
    }
    bool evaluate(kraft::ConstraintRequest request,const double* theta,
        double* values,double* jacobian) const noexcept
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
        const auto& layout=program.layout;
        if constexpr (Program::aparch) {
            using Density=typename Program::density_type;
            constexpr auto shapes=Density::shape_parameters;
            double persistence=0.0;
            for (std::size_t lag=0;lag<layout.p;++lag) {
                const auto a=layout.alpha_start+lag,g=layout.gamma_start+lag,d=layout.delta_index;
                if (!derivatives && theta[a]==0.0) continue;
                // d(alpha*M)/dalpha still needs M at alpha=0. Other first
                // derivatives are exactly zero and need no moment derivatives.
                auto moment=derivatives && theta[a]!=0.0
                    ? generated::asymmetric_power_moment<1,Density>(theta+layout.model_parameters,theta[g],theta[d])
                    : generated::asymmetric_power_moment<0,Density>(theta+layout.model_parameters,theta[g],theta[d],program.moment_values,lag);
                double moment_value=moment.value,gradient_multiplier=theta[a];
                if constexpr (!generated::quadrature_power_moment<Density>) {
                    if (!moment.valid && derivatives && std::isfinite(theta[a]) && theta[a]!=0.0) {
                        const auto value=generated::asymmetric_power_moment<0,Density>(
                            theta+layout.model_parameters,theta[g],theta[d]);
                        if (!value.valid) return false;
                        // The requested constraint needs alpha*M', not M'.
                        // Include |alpha| before differentiating the exponential.
                        moment=generated::aparch_moment_detail::symmetric_closed_form<1,Density>(
                            theta+layout.model_parameters,theta[g],theta[d],std::log(std::abs(theta[a])));
                        moment_value=value.value;
                        gradient_multiplier=std::copysign(1.0,theta[a]);
                    }
                }
                if (!moment.valid) return false;
                persistence+=theta[a]*moment_value;
                if (derivatives) {
                    jacobian[m+a*rows]=-moment_value;
                    jacobian[m+g*rows]=-gradient_multiplier*moment.gradient[0];
                    jacobian[m+d*rows]-=gradient_multiplier*moment.gradient[1];
                    for (std::size_t s=0;s<shapes;++s)
                        jacobian[m+(layout.model_parameters+s)*rows]-=gradient_multiplier*moment.gradient[2+s];
                }
            }
            for (std::size_t lag=0;lag<layout.q;++lag) {
                persistence+=theta[layout.beta_start+lag];
                if (derivatives) jacobian[m+(layout.beta_start+lag)*rows]=-1.0;
            }
            if (request!=kraft::ConstraintRequest::jacobian) values[m]=1.0-persistence;
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
