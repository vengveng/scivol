// SCIVOL-NATIVE: CORE; objective decoration, no optimizer or model equations.
#pragma once
#include "kraft_slsqp_driver.hpp"
#include <cmath>
#include <limits>

namespace vlk::experiment::controller {

// Inner supplies additive F/G/H and scaled(factor) for scientific diagnostics.
// Inheritance preserves optional controller policies, including speculation.
template<class Inner,class Observer> struct ObservedScaledObjective : Inner {
    double scale;
    Observer observer;
    mutable double additive_value=std::numeric_limits<double>::infinity();

    ObservedScaledObjective(const Inner& inner,double factor,Observer observed) noexcept
        :Inner(inner),scale(factor),observer(observed) {}

    bool evaluate(kraft::ObjectiveRequest request,const double* point,
                  double* value,double* gradient,bool same) const noexcept {
        const bool f=request!=kraft::ObjectiveRequest::gradient;
        const bool g=request!=kraft::ObjectiveRequest::value;
        observer.request(f,g,false);
        if (!Inner::evaluate(request,point,value,gradient,same)) return false;
        if (f) {
            if (!std::isfinite(*value)) return false;
            additive_value=*value;
            observer.retain(point,*value);
            *value*=scale;
        }
        if (g) for (std::size_t i=0;i<this->dimension();++i) gradient[i]*=scale;
        return true;
    }

    bool evaluate_curvature(const double* point,double* value,
                            double* gradient,double* hessian) const noexcept {
        observer.request(true,true,true);
        if (!Inner::evaluate_curvature(point,value,gradient,hessian)
            || !std::isfinite(*value)) return false;
        additive_value=*value;
        observer.retain(point,*value);
        *value*=scale;
        for (std::size_t row=0;row<this->dimension();++row) {
            gradient[row]*=scale;
            for (std::size_t column=0;column<this->dimension();++column)
                hessian[row*this->dimension()+column]*=scale;
        }
        return true;
    }

    double scientific_kkt(const double* point,double* theta,double* gradient,
                          double tolerance) const noexcept {
        observer.request(false,true,false);
        return Inner::scaled(scale).scientific_kkt(point,theta,gradient,tolerance);
    }
};

} // namespace vlk::experiment::controller
