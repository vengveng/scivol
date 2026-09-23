// SCIVOL-NATIVE: TEST_ONLY; shared assembly and structural-zero regression.
#include "location_scale_observation_assembly.hpp"
#include <array>
#include <cmath>
#include <span>

using namespace vlk::experiment;

struct Density {
    static constexpr std::size_t shape_parameters = 2;
    using ValueAccumulator = double;
    static void accumulate_value(double& value, const generated::DensityObservation<2>& obs)
    {
        value += obs.value;
    }
};

template<unsigned Request, bool Dense, bool Constant, bool Affine, std::size_t Static>
bool verify()
{
    constexpr bool F = Request & 1;
    constexpr bool G = Request & 2;
    constexpr bool H = Request & 4;
    constexpr std::size_t k = 5, width = 7, packed = 15;
    generated::DensityObservation<2> obs;
    obs.value = .7;
    obs.residual_gradient = .4;
    obs.variance_gradient = -.6;
    obs.residual_hessian = .8;
    obs.residual_variance_hessian = -.3;
    obs.variance_hessian = .2;
    obs.shape_gradient = {.1, .9};
    obs.residual_shape_hessian = {.6, .2};
    obs.variance_shape_hessian = {-.8, .4};
    obs.shape_hessian = {.3, -.2, .5};
    std::array<double,k> e, v;
    std::array<double,packed> ee, vv;
    for (std::size_t i=0; i<k; ++i) {
        e[i] = Constant && i==k-1 ? 0 : .1*static_cast<double>(i+1);
        v[i] = Constant ? double(i==k-1) : -.2*static_cast<double>(i+1);
    }
    for (std::size_t i=0; i<packed; ++i) {
        ee[i] = Affine ? 0 : .02*static_cast<double>(i+1);
        vv[i] = Constant ? 0 : -.03*static_cast<double>(i+1);
    }
    if constexpr (Constant) {
        for (std::size_t i=0; i<k; ++i)
            ee[model::packed_upper_index(k,i,k-1)] = 0;
    }
    double value = 0;
    std::array<double,width> gradient {};
    std::array<double,width*width> hessian {};
    if constexpr (H) {
        model::accumulate_location_scale_second<F,G,Dense,Static,Density,Constant,Affine>(
            k,obs,e.data(),Constant ? nullptr : v.data(),
            Affine ? nullptr : ee.data(),Constant ? nullptr : vv.data(),
            value,G ? gradient.data() : nullptr,hessian.data());
    } else if constexpr (G) {
        model::accumulate_location_scale_first<F,Static,Density,Constant>(
            k,obs,e.data(),Constant ? nullptr : v.data(),value,gradient.data());
    } else if constexpr (F) {
        Density::accumulate_value(value,obs);
    }
    const auto equal = [](double a,double b) { return std::abs(a-b)<1e-13; };
    if (!equal(value,F ? obs.value : 0)) return false;
    for (std::size_t i=0; i<width; ++i) {
        if constexpr (G) {
            const double expected = i<k ? obs.residual_gradient*e[i]+obs.variance_gradient*v[i]
                                        : obs.shape_gradient[i-k];
            if (!equal(gradient[i],expected)) return false;
        }
        if constexpr (H) {
            for (std::size_t j=i; j<width; ++j) {
                double expected;
                if (j<k) {
                    const auto packed_index = model::packed_upper_index(k,i,j);
                    expected = obs.residual_hessian*e[i]*e[j]
                        + obs.residual_variance_hessian*(e[i]*v[j]+v[i]*e[j])
                        + obs.variance_hessian*v[i]*v[j]
                        + obs.residual_gradient*ee[packed_index]
                        + obs.variance_gradient*vv[packed_index];
                } else if (i<k) {
                    expected = obs.residual_shape_hessian[j-k]*e[i]
                        + obs.variance_shape_hessian[j-k]*v[i];
                } else {
                    expected = obs.shape_hessian[model::packed_upper_index(2,i-k,j-k)];
                }
                const auto index = Dense ? i*width+j : model::packed_upper_index(width,i,j);
                if (!equal(hessian[index],expected)) return false;
            }
        }
    }
    return true;
}

template<unsigned Request>
bool masks()
{
    return verify<Request,true,false,false,0>()
        && verify<Request,false,false,false,5>()
        && verify<Request,true,true,false,0>()
        && verify<Request,false,true,false,5>()
        && verify<Request,true,true,true,0>()
        && verify<Request,false,true,true,5>();
}

template<bool Value, std::size_t Static>
bool verify_residual_view()
{
    constexpr std::size_t k=5;
    const std::array<double,k> residual {1.,-2.,3.,-4.,5.};
    const std::array<double,k> variance {.2,.4,.6,.8,1.};
    generated::DensityObservation<2> obs;
    obs.value=.7;obs.residual_gradient=.4;obs.variance_gradient=-.6;
    obs.shape_gradient={.1,.9};
    for (const std::size_t active : {0u,3u,5u}) {
        double value=0.;
        std::array<double,k+2> gradient {};
        model::accumulate_location_scale_first<Value,Static,Density>(
            k,obs,std::span(residual.data(),active),variance.data(),value,gradient.data());
        if (std::abs(value-(Value?obs.value:0.))>1e-13) return false;
        for (std::size_t j=0;j<k;++j) {
            const double expected=obs.variance_gradient*variance[j]
                +(j<active?obs.residual_gradient*residual[j]:0.);
            if (std::abs(gradient[j]-expected)>1e-13) return false;
        }
        for (std::size_t j=0;j<2;++j)
            if (gradient[k+j]!=obs.shape_gradient[j]) return false;
    }
    return true;
}

int main()
{
    return masks<1>() && masks<2>() && masks<3>() && masks<4>()
        && masks<5>() && masks<6>() && masks<7>()
        && verify_residual_view<false,0>() && verify_residual_view<true,0>()
        && verify_residual_view<false,5>() && verify_residual_view<true,5>() ? 0 : 1;
}
