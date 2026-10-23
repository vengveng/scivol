// SCIVOL-NATIVE: PINNED SLSQP TRANSLATION UNIT
#if defined(_MSC_VER)
#  define restrict __restrict
#else
#  define restrict __restrict__
#endif
#define BLAS_SYMBOL_PREFIX vlk_
#define NO_APPEND_FORTRAN

extern "C" {
#include "third_party/scipy_slsqp/slsqp.c"
}

#include "active_face_refinement.hpp"

namespace vlk::experiment::curvature {

kraft::Result direct_projection(std::size_t n,std::size_t m,
    const double* gradient,const double* normals,const double* gaps,
    double* lower,double* upper,double* point,
    runtime::RuntimeKraftWorkspace& state) noexcept
{
    // The caller validated dimensions and bound this exact runtime workspace.
    // Use the same LSQ offset as SLSQP, retaining its conservative scratch bound.
    const auto triangular=n*(n+1)/2;
    const auto offset=triangular+n+m+3*(n+1);
    double* factor=state.buffer;
    std::fill_n(factor,triangular,0.0);
    for (std::size_t i=0,index=0;i<n;++i) {
        factor[index]=1.0;
        index+=n-i;
    }
    std::copy_n(gradient,n,state.gradient);
    for (std::size_t row=0;row<m;++row) {
        state.constraint_values[row]=gaps[row];
        for (std::size_t i=0;i<n;++i)
            state.jacobian[row+i*m]=normals[row*n+i];
    }
    std::int64_t mode=0;
    lsq(static_cast<CBLAS_INT>(m),0,static_cast<CBLAS_INT>(n),0,0.0,
        factor,state.gradient,state.jacobian,state.constraint_values,
        lower,upper,point,state.multipliers,state.buffer+offset,state.indices,&mode,state.nnls);
    kraft::Result result;
    // LSQ mode 1 means a QP solution, not a stationarity certificate.
    result.mode=mode==1?0:(mode==0?-101:mode);
    result.iterations=1;
    result.value=0.0;
    result.maximum_violation=0.0;
    for (std::size_t i=0;i<n;++i) {
        if (!std::isfinite(point[i])) result.mode=-101;
        result.value+=(0.5*point[i]+gradient[i])*point[i];
    }
    for (std::size_t row=0;row<m;++row) {
        double slack=gaps[row];
        for (std::size_t i=0;i<n;++i) slack+=normals[row*n+i]*point[i];
        if (!std::isfinite(slack)) result.mode=-101;
        result.maximum_violation=std::max(result.maximum_violation,-slack);
    }
    return result;
}

} // namespace vlk::experiment::curvature
