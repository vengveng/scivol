// SCIVOL-NATIVE: exact diagonal variable scaling, independent of model mathematics.
#ifndef SCIVOL_DIAGONAL_OPTIMIZER_SCALING_HPP
#define SCIVOL_DIAGONAL_OPTIMIZER_SCALING_HPP
#include "runtime_kraft_slsqp.hpp"

namespace vlk::experiment::runtime {
template<class Objective>
struct DiagonalObjective {
    const Objective& inner;
    const double* scales;
    double* point;
    std::size_t dimension() const noexcept { return inner.dimension(); }
    bool speculate_trial_gradient(std::int64_t iteration) const noexcept
        requires requires (const Objective& objective) {
            objective.speculate_trial_gradient(iteration);
        }
    { return inner.speculate_trial_gradient(iteration); }
    bool evaluate(kraft::ObjectiveRequest request, const double* x,
                  double* f, double* g, bool accepted) const noexcept
    {
        for (std::size_t i=0; i<dimension(); ++i)
            point[i] = x[i]*scales[i];
        if (!inner.evaluate(request,point,f,g,accepted))
            return false;
        if (request != kraft::ObjectiveRequest::value)
            for (std::size_t i=0; i<dimension(); ++i)
                g[i] *= scales[i];
        return true;
    }
};

template<class Constraints>
struct DiagonalConstraints {
    const Constraints& inner;
    const double* scales;
    double* point;
    std::size_t dimension() const noexcept { return inner.dimension(); }
    std::size_t count() const noexcept { return inner.count(); }
    std::size_t equalities() const noexcept
    { return runtime_kraft_detail::equality_count(inner); }
    bool evaluate(kraft::ConstraintRequest request, const double* x,
                  double* values, double* jacobian, ConstraintColumns columns={}) const noexcept
    {
        for (std::size_t i=0; i<dimension(); ++i)
            point[i] = x[i]*scales[i];
        if (!runtime_kraft_detail::invoke_constraints(inner,request,point,values,jacobian,columns))
            return false;
        if (request != kraft::ConstraintRequest::value)
            for (std::size_t j=0; j<dimension(); ++j)
                if (columns.contains(j))
                    for (std::size_t i=0; i<count(); ++i)
                        jacobian[i+j*count()] *= scales[j];
        return true;
    }
};
}
#endif
