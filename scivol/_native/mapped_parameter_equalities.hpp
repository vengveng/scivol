// SCIVOL-NATIVE: PROTOTYPE; parameter rows over a model-owned map.
#pragma once
#include "runtime_kraft_slsqp.hpp"

namespace vlk::experiment::runtime {

template<class Map> struct MappedParameterEqualities {
    Map& map;
    const std::size_t* selected;
    const double* targets;
    std::size_t rows;

    std::size_t dimension() const noexcept { return map.dimension(); }
    std::size_t count() const noexcept { return rows; }
    std::size_t equalities() const noexcept { return rows; }
    bool valid() const noexcept {
        if (!rows || rows>dimension() || !selected || !targets) return false;
        for (std::size_t i=0;i<rows;++i)
            if (selected[i]>=dimension() || !std::isfinite(targets[i])
                || (i && selected[i-1]>=selected[i])) return false;
        return true;
    }
    bool evaluate(kraft::ConstraintRequest request,const double* point,
        double* values,double* jacobian,ConstraintColumns columns={}) const noexcept {
        const bool value=request!=kraft::ConstraintRequest::jacobian;
        const bool derivative=request!=kraft::ConstraintRequest::value;
        if (!valid() || !point || (value && !values) || (derivative && !jacobian)
            || columns.size>dimension() || (columns.size && !columns.indices)) return false;
        for (std::size_t i=0;i<columns.size;++i)
            if (columns.indices[i]>=dimension() || (i && columns.indices[i-1]>=columns.indices[i])) return false;
        const bool prepared=[&] {
            if constexpr (requires {map.prepare(point,derivative,selected,rows,columns);})
                return map.prepare(point,derivative,selected,rows,columns);
            return map.prepare(point,derivative);
        }();
        if (!prepared) return false;
        for (std::size_t i=0;i<rows;++i) {
            if (value && !std::isfinite(map.value(selected[i])-targets[i])) return false;
            if (derivative)
                for (std::size_t j=0;j<dimension();++j)
                    if (columns.contains(j) && !std::isfinite(map.derivative(selected[i],j))) return false;
        }
        if (value)
            for (std::size_t i=0;i<rows;++i) values[i]=map.value(selected[i])-targets[i];
        if (derivative)
            for (std::size_t j=0;j<dimension();++j)
                if (columns.contains(j))
                    for (std::size_t i=0;i<rows;++i) jacobian[j*rows+i]=map.derivative(selected[i],j);
        return true;
    }
};

} // namespace vlk::experiment::runtime
