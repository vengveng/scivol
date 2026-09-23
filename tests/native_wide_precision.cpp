// TEST_ONLY: production scalar, moment cache and cold-call allocation contracts.
#include "egarch_precision_state.hpp"
#include <cassert>
#include <new>

namespace wide=vlk::experiment::wide;
using wide::Real;

static void export_real(Real value,double* output) {
    for (unsigned j=0;j<3;++j) {
        output[j]=static_cast<double>(value);
        value-=Real(output[j]);
    }
}

extern "C" int probe_scalar_binary(unsigned op,const double* a,const double* b,double* out) {
    const Real left=Real(a[0])+Real(a[1]),right=Real(b[0])+Real(b[1]);
    Real value;
    switch (op) {
    case 0:value=left+right;break;
    case 1:value=left-right;break;
    case 2:value=left*right;break;
    case 3:value=left/right;break;
    default:return -2;
    }
    if (!wide::finite(value)) return -3;
    export_real(value,out);
    return 0;
}

extern "C" int probe_scalar_fma(double a,double b,double c,double* out) {
    const auto value=wide::fma(Real(a),Real(b),Real(c));
    if (!wide::finite(value)) return -3;
    export_real(value,out);
    return 0;
}

extern "C" int probe_portable_math(unsigned op,double x,double* out) {
    if (!std::isfinite(x) || ((op==1 || op==4) && x<=0) || (op==2 && x<0))
        return -1;
    Real value;
    switch (op) {
    case 0:value=wide::exp(Real(x));break;
    case 1:value=wide::log(Real(x));break;
    case 2:value=wide::sqrt(Real(x));break;
    case 3:value=wide::erf(Real(x));break;
    case 4:value=wide::lgamma(Real(x));break;
    default:return -2;
    }
    if (!wide::finite(value)) return -3;
    export_real(value,out);
    return 0;
}

extern "C" int probe_moment(double rho,double lambda,double* out) {
    const auto moment=wide::variance_gamma_absolute_value(rho,lambda);
    if (!moment.valid) return -1;
    export_real(moment.value,out);
    return 0;
}

static std::size_t allocations=0;
void* operator new(std::size_t bytes) {
    ++allocations;
    if (void* pointer=std::malloc(bytes?bytes:1)) return pointer;
    std::abort();
}
void* operator new[](std::size_t bytes) {return ::operator new(bytes);}
void operator delete(void* pointer) noexcept {std::free(pointer);}
void operator delete[](void* pointer) noexcept {std::free(pointer);}
void operator delete(void* pointer,std::size_t) noexcept {std::free(pointer);}
void operator delete[](void* pointer,std::size_t) noexcept {std::free(pointer);}

int main() {
    using State=vlk::experiment::runtime::arma_egarch::VarianceGammaPointState;
    using Density=vlk::experiment::generated::VarianceGammaDensity;
    State first,second;
    for (const std::array<double,2> shape : {
            std::array<double,2>{.1,2.5},
            std::array<double,2>{.08590362405556805,3.6453348340489273},
            std::array<double,2>{.5,10.1},
            std::array<double,2>{.9949999999999999,50.0}}) {
        const auto before=allocations;
        const auto value=first.absolute_moment<0,Density>(shape.data());
        assert(value.valid && first.prepared && first.last_shape==shape);
        const auto cached=first.absolute_moment<0,Density>(shape.data());
        assert(cached.valid && cached.value==value.value);
        const auto independent=second.absolute_moment<0,Density>(shape.data());
        assert(independent.valid && independent.value==value.value);
        assert(allocations==before);
    }
    const std::array<double,2> invalid{1,2};
    assert((!first.absolute_moment<0,Density>(invalid.data()).valid));
    const std::array<double,2> restored{.1,2.5};
    assert((first.absolute_moment<0,Density>(restored.data()).valid));
    assert(first.last_shape==restored);
}
