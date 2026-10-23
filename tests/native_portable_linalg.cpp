// TEST_ONLY: BLAS zero coefficients must not read mathematically unused inputs.
#define BLAS_SYMBOL_PREFIX vlk_
#define NO_APPEND_FORTRAN
extern "C" {
#include "third_party/scipy_slsqp/blaslapack_declarations.h"
}
#include <array>
#include <cassert>
#include <cmath>

void verify_gemv(char transpose, int increment) {
    int m=2, n=3, lda=3, incx=increment, incy=-increment;
    double alpha=2, beta=0;
    // Column-major, with padding that must not be consumed.
    double a[]{1,2,NAN, 3,4,NAN, 5,6,NAN};
    std::array<double,5> x, y;
    x.fill(NAN); y.fill(NAN);
    const bool trans=transpose!='N';
    const int input_size=trans?m:n, output_size=trans?n:m;
    const int x_start=incx<0 ? (1-input_size)*incx : 0;
    const int y_start=incy<0 ? (1-output_size)*incy : 0;
    for (int i=0;i<input_size;++i) x[x_start+i*incx]=i+1;
    BLAS_FUNC(dgemv)(&transpose,&m,&n,&alpha,a,&lda,x.data(),&incx,
                     &beta,y.data(),&incy);
    for (int i=0;i<output_size;++i) {
        const double expected=trans ? 10+12*i : 44+12*i;
        assert(y[y_start+i*incy]==expected);
    }
    for (int i=0;i<5;++i) {
        bool active=false;
        for (int j=0;j<output_size;++j) active|=i==y_start+j*incy;
        if (!active) assert(std::isnan(y[i]));
    }
    // An alpha-zero request only scales y; A and x are not referenced.
    alpha=0; beta=-2;
    BLAS_FUNC(dgemv)(&transpose,&m,&n,&alpha,nullptr,&lda,nullptr,&incx,
                     &beta,y.data(),&incy);
    for (int i=0;i<output_size;++i) {
        const double expected=trans ? 10+12*i : 44+12*i;
        assert(y[y_start+i*incy]==-2*expected);
    }
    beta=0;
    y.fill(NAN);
    BLAS_FUNC(dgemv)(&transpose,&m,&n,&alpha,nullptr,&lda,nullptr,&incx,
                     &beta,y.data(),&incy);
    for (int i=0;i<output_size;++i) assert(y[y_start+i*incy]==0);
}

void verify_range_factorizations() {
    for (double scale : {1e-310,1e-100,1.0,1e308}) {
        for (double sign : {-1.,1.}) {
            int n=2, stride=1, m=2, lda=2, info=0;
            double lwork=8;
            double alpha=sign*scale, x=scale, tau=0;
            BLAS_FUNC(dlarfgp)(&n,&alpha,&x,&stride,&tau);
            const double expected_tau=1.-sign/std::sqrt(2.);
            assert(std::isfinite(alpha) && std::abs(tau-expected_tau)<3e-15);
            assert(std::abs(x+1./(std::sqrt(2.)*expected_tau))<3e-15);
            double matrix[]{sign*scale,scale},work[8];
            BLAS_FUNC(dgeqrf)(&m,&stride,matrix,&lda,&tau,work,&lwork,&info);
            assert(info==0 && std::isfinite(matrix[0]));
            assert(std::abs(tau-(1.+1./std::sqrt(2.)))<3e-15);
            assert(std::abs(matrix[1]-sign/(std::sqrt(2.)+1.))<3e-15);
        }
    }
    for (double scale : {1e-100,1e-12,1.,1e100}) {
        int m=2,n=2,nrhs=1,lda=3,ldb=2,rank=0,info=0,lwork=32,pivots[2]{};
        double matrix[]{scale,0.,NAN,0.,scale,NAN};
        double rhs[]{scale,2*scale},rcond=1e-10,work[32];
        BLAS_FUNC(dgelsy)(&m,&n,&nrhs,matrix,&lda,rhs,&ldb,pivots,&rcond,&rank,work,&lwork,&info);
        assert(info==0 && rank==2);
        assert(std::abs(rhs[0]-1.)<2e-15 && std::abs(rhs[1]-2.)<4e-15);
    }
}

int main() {
    verify_range_factorizations();
    for (char trans : {'N','T','C'})
        for (int increment : {-2,-1,1,2}) verify_gemv(trans,increment);
    int one=1, zero=0, two=2;
    double alpha=1, beta=0, sentinel=7;
    for (char trans : {'N','T'}) {
        BLAS_FUNC(dgemv)(&trans,&zero,&two,&alpha,nullptr,&one,nullptr,&one,
                         &beta,&sentinel,&one);
        BLAS_FUNC(dgemv)(&trans,&two,&zero,&alpha,nullptr,&two,nullptr,&one,
                         &beta,&sentinel,&one);
        assert(sentinel==7);
    }
    alpha=0; beta=1;
    BLAS_FUNC(dgemv)("N",&two,&two,&alpha,nullptr,&two,nullptr,&one,
                     &beta,nullptr,&one);
    BLAS_FUNC(daxpy)(&two,&alpha,nullptr,&one,nullptr,&one);
    alpha=1;
    BLAS_FUNC(daxpy)(&zero,&alpha,nullptr,&one,nullptr,&one);
}
