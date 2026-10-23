// SCIVOL-NATIVE: TEST_ONLY; rectangular sensitivity propagation contract.
#include "math/mean/runtime_reflection.hpp"
#include "models/mean/stability_domain.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

using namespace vlk::experiment::runtime;

void verify_domain_curvature()
{
    using vlk::experiment::mean::PolynomialDomain;
    for (bool composed : {false,true}) for (double sign : {-1.0,1.0}) {
        const std::size_t width=composed?3:2;
        std::array<double,6> basis{.5,-.2,.1,.1,.3,-.1};
        for (double& value:basis) value*=sign;
        const std::array<double,3> theta{.15,-.2,.1};
        const std::array<double,4> weights{.25,.75,-.1,.2};
        const PolynomialDomain domain{2,width,composed?basis.data():nullptr,sign};
        const auto bytes=domain.curvature_workspace_bytes();
        std::vector<double> arena((bytes+7)/8+2,917.0);
        std::array<double,11> output;
        output.fill(811.0);
        assert(bytes>domain.workspace_bytes(true));
        assert(domain.weighted_curvature(theta.data(),weights.data(),output.data()+1,
            arena.data()+1,bytes));
        std::array<double,2> coefficients{};
        std::array<double,6> jacobian{};
        domain.prepare_coefficients(theta.data(),coefficients.data(),jacobian.data());
        const double denominator=1-coefficients[1];
        const double cross=(weights[1]-weights[0])/(denominator*denominator);
        const double second=2*coefficients[0]*cross/denominator;
        for (std::size_t i=0;i<width;++i) for (std::size_t j=0;j<width;++j) {
            const double expected=cross*(jacobian[i]*jacobian[width+j]
                +jacobian[width+i]*jacobian[j])+second*jacobian[width+i]*jacobian[width+j];
            assert(std::abs(output[1+i*width+j]-expected)<2e-15);
        }
        assert(output[0]==811.0 && output[width*width+1]==811.0);
        assert(arena.front()==917.0 && arena.back()==917.0);
        output.fill(811.0);
        assert(!domain.weighted_curvature(theta.data(),weights.data(),output.data()+1,
            arena.data()+1,bytes-1));
        for (double value:output) assert(value==811.0);
    }
    const PolynomialDomain oversized{std::numeric_limits<std::size_t>::max(),1,nullptr};
    assert(oversized.curvature_workspace_bytes()==0);
}

int main()
{
    verify_domain_curvature();
    {
        const std::array<double,2> coefficients{.7,-1.0};
        std::array<double,2> values,current,next;
        std::array<double,4> jacobian;
        std::array<double,12> scratch;
        assert(reflection_inverse_values(coefficients.data(),2,1.0,values.data(),
            current.data(),next.data(),false));
        assert(values[0]==.35 && values[1]==-1.0);
        assert(!reflection_inverse_values(coefficients.data(),2,1.0,values.data(),
            current.data(),next.data(),true));
        assert(reflection_inverse_unrestricted(coefficients.data(),2,values.data(),
            jacobian.data(),scratch.data()));
        assert(jacobian[0]==.5 && jacobian[1]==.175
            && jacobian[2]==0 && jacobian[3]==1);
        assert(!reflection_inverse(coefficients.data(),2,values.data(),
            jacobian.data(),scratch.data()));
        const std::array<double,3> singular{.2,.1,-1.0};
        std::array<double,3> v3,c3,n3;
        assert(!reflection_inverse_values(singular.data(),3,1.0,v3.data(),
            c3.data(),n3.data(),false));
    }
    constexpr std::size_t order=5, width=3;
    const std::array<double,width> beta {.2,-.1,.15};
    std::array<double,order*width> basis;
    std::array<double,order> coefficients {}, dense_values {}, tied_values {};
    std::array<double,order*order> dense_jacobian;
    std::array<double,order*width> tied_jacobian;
    std::array<double,2*order+2*order*order> dense_work;
    std::array<double,2*order+2*order*width> tied_work;
    for (std::size_t row=0; row<order; ++row) {
        for (std::size_t col=0; col<width; ++col) {
            basis[row*width+col]=.1*double(row+col+1);
            coefficients[row]+=basis[row*width+col]*beta[col];
        }
    }
    if (reflection_inverse_scratch_doubles(order)!=dense_work.size()
        || reflection_inverse_scratch_doubles(order,width)!=tied_work.size()
        || reflection_inverse_scratch_doubles(std::numeric_limits<std::size_t>::max(),width)!=0)
        return 1;
    if (!reflection_inverse(coefficients.data(),order,dense_values.data(),
            dense_jacobian.data(),dense_work.data())
        || !reflection_inverse_composed(coefficients.data(),order,basis.data(),width,
            tied_values.data(),tied_jacobian.data(),tied_work.data()))
        return 2;
    for (std::size_t row=0; row<order; ++row) {
        if (std::abs(dense_values[row]-tied_values[row])>1e-14)
            return 3;
        for (std::size_t col=0; col<width; ++col) {
            double expected=0;
            for (std::size_t inner=0; inner<order; ++inner)
                expected+=dense_jacobian[row*order+inner]*basis[inner*width+col];
            if (std::abs(expected-tied_jacobian[row*width+col])>1e-13)
                return 4;
        }
    }
    coefficients[0]=1.2;
    coefficients[1]=-.5;
    if (!reflection_inverse(coefficients.data(),2,dense_values.data(),
            dense_jacobian.data(),dense_work.data()))
        return 5;
    coefficients[0]=.9;
    coefficients[1]=.9;
    if (reflection_inverse(coefficients.data(),2,dense_values.data(),
            dense_jacobian.data(),dense_work.data()))
        return 6;
    if (!reflection_inverse_unrestricted(coefficients.data(),2,dense_values.data(),
            dense_jacobian.data(),dense_work.data()))
        return 7;
    return 0;
}
