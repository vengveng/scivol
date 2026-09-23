// SCIVOL-NATIVE: PROTOTYPE; affine coordinate restriction, no model equations.
#pragma once
#include "runtime_kraft_slsqp.hpp"

namespace vlk::experiment::runtime {

struct CoordinateRestriction {
    std::size_t full=0,free=0,rows=0;
    const std::size_t* indices=nullptr;
    const double* origin=nullptr;
    RuntimeBounds original_bounds{};
    double *expanded=nullptr,*gradient=nullptr,*jacobian=nullptr;
    double *point=nullptr,*lower=nullptr,*upper=nullptr,*scales=nullptr;

    static std::size_t bytes(std::size_t full,std::size_t rows) noexcept
    {
        std::size_t matrix;
        WorkspaceSizer size;
        if (!full || !checked_product(full,rows,matrix)) return 0;
        for (unsigned i=0;i<7;++i) if (!size.add<double>(full)) return 0;
        return size.add<double>(matrix)?size.size():0;
    }

    bool bind(std::size_t width,std::size_t count,std::size_t constraint_rows,
        const std::size_t* selected,const double* initial,RuntimeBounds bounds,
        void* work,std::size_t size,const double* units=nullptr) noexcept
    {
        *this={};
        const auto required=bytes(width,constraint_rows);
        if (!required || !count || count>width || !selected || !initial || !work
            || size<required || !runtime_kraft_detail::valid_bounds(width,bounds)) return false;
        for (std::size_t i=0;i<width;++i)
            if (!std::isfinite(initial[i]) || initial[i]<bounds.lower[i]
                || initial[i]>bounds.upper[i]) return false;
        for (std::size_t j=0;j<count;++j) {
            const auto i=selected[j];
            if (i>=width || (j && selected[j-1]>=i)) return false;
            const double scale=units?units[i]:1.0;
            if (!std::isfinite(scale) || !(scale>0)) return false;
            if ((!std::isnan(bounds.lower[i]) && !std::isfinite((bounds.lower[i]-initial[i])/scale))
                || (!std::isnan(bounds.upper[i]) && !std::isfinite((bounds.upper[i]-initial[i])/scale))) return false;
        }
        full=width;free=count;rows=constraint_rows;indices=selected;
        original_bounds=bounds;
        WorkspaceArena arena(work,size);
        double* saved=arena.take<double>(full);
        expanded=arena.take<double>(full);gradient=arena.take<double>(full);
        point=arena.take<double>(full);lower=arena.take<double>(full);upper=arena.take<double>(full);
        scales=arena.take<double>(full);
        jacobian=arena.take<double>(full*rows);
        if (!saved || !expanded || !gradient || !point || !lower || !upper
            || !scales || (rows && !jacobian)) {*this={};return false;}
        std::copy_n(initial,full,saved);origin=saved;
        std::copy_n(origin,full,expanded);
        std::fill_n(point,free,0.0);
        for (std::size_t j=0;j<free;++j) {
            scales[j]=units?units[indices[j]]:1.0;
            lower[j]=(bounds.lower[indices[j]]-origin[indices[j]])/scales[j];
            upper[j]=(bounds.upper[indices[j]]-origin[indices[j]])/scales[j];
        }
        return true;
    }

    bool expand(const double* x) const noexcept
    {
        if (!x || !expanded) return false;
        for (std::size_t j=0;j<free;++j) {
            const auto i=indices[j];
            const double value=x[j]==0.0?origin[i]:origin[i]+x[j]*scales[j];
            if (!std::isfinite(value) || value<original_bounds.lower[i]
                || value>original_bounds.upper[i]) return false;
            expanded[i]=value;
        }
        return true;
    }
};

template<class Objective> struct RestrictedObjective {
    const Objective& inner;
    const CoordinateRestriction& map;
    std::size_t dimension() const noexcept {return map.free;}
    bool speculate_trial_gradient(std::int64_t iteration) const noexcept {
        if constexpr (requires {inner.speculate_trial_gradient(iteration);})
            return inner.speculate_trial_gradient(iteration);
        return false;
    }
    bool evaluate(kraft::ObjectiveRequest request,const double* x,double* f,
                  double* g,bool same_point) const noexcept {
        if (inner.dimension()!=map.full || !map.expand(x)
            || !runtime_kraft_detail::evaluate_objective(inner,request,
                map.expanded,f,map.gradient,same_point)) return false;
        if (request!=kraft::ObjectiveRequest::value)
            for (std::size_t j=0;j<map.free;++j) g[j]=map.gradient[map.indices[j]]*map.scales[j];
        return true;
    }
};

template<class Constraints> struct RestrictedConstraints {
    const Constraints& inner;
    const CoordinateRestriction& map;
    std::size_t dimension() const noexcept {return map.free;}
    std::size_t count() const noexcept {return inner.count();}
    std::size_t equalities() const noexcept {return runtime_kraft_detail::equality_count(inner);}
    bool evaluate(kraft::ConstraintRequest request,const double* x,double* values,
                  double* jacobian) const noexcept {
        if (inner.dimension()!=map.full || count()!=map.rows || !map.expand(x)
            || !runtime_kraft_detail::evaluate_constraints(inner,request,
                map.expanded,values,map.jacobian)) return false;
        if (request!=kraft::ConstraintRequest::value)
            for (std::size_t j=0;j<map.free;++j)
                for (std::size_t i=0;i<map.rows;++i)
                    jacobian[i+j*map.rows]=map.jacobian[i+map.indices[j]*map.rows]*map.scales[j];
        return true;
    }
};

struct FullCoordinateKraft {
    std::size_t dimension(std::size_t full) const noexcept {return full;}
    template<class Objective,class Constraints>
    kraft::Result operator()(const Objective& objective,const Constraints& constraints,
        double* point,RuntimeBounds bounds,const kraft::Options& options,
        RuntimeKraftWorkspace& state) const noexcept {
        return minimize_runtime_kraft(objective,constraints,point,bounds,options,state);
    }
};

struct RestrictedCoordinateKraft {
    CoordinateRestriction& map;
    std::size_t dimension(std::size_t full) const noexcept {return full==map.full?map.free:0;}
    template<class Objective,class Constraints>
    kraft::Result operator()(const Objective& objective,const Constraints& constraints,
        double* point,RuntimeBounds,const kraft::Options& options,
        RuntimeKraftWorkspace& state) const noexcept {
        auto result=minimize_runtime_kraft(RestrictedObjective{objective,map},
            RestrictedConstraints{constraints,map},map.point,{map.lower,map.upper},options,state);
        if (!map.expand(map.point)) return {};
        std::copy_n(map.expanded,map.full,point);
        return result;
    }
};

} // namespace vlk::experiment::runtime
