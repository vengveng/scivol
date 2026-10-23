// SCIVOL-NATIVE: TEST_ONLY; shared assembly and structural-zero regression.
#include "location_scale_observation_assembly.hpp"
#include "egarch_observation_assembly.hpp"
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

template<class Base>
struct ObservedDensity : Base {
    inline static unsigned gradient_requests = 0;

    template<bool F, bool G, bool H, bool Residual = false, class Prepared>
    static auto observe(const Prepared& prepared, double variance, double residual,
                        double squared)
    {
        gradient_requests += G;
        auto result = Base::template observe<F,G,H,Residual>(prepared,variance,residual,squared);
        if constexpr (!G) result.shape_gradient.fill(NAN);
        return result;
    }

    template<bool F, bool G, bool H, bool Residual, class Prepared>
    static auto observe_log_variance(const Prepared& prepared, double log_variance,
        double residual, double squared, bool require_residual = true)
        requires requires {
            Base::template observe_log_variance<F,G,H,Residual>(
                prepared,log_variance,residual,squared,require_residual);
        }
    {
        gradient_requests += G;
        return Base::template observe_log_variance<F,G,H,Residual>(
            prepared,log_variance,residual,squared,require_residual);
    }
};

template<std::size_t Width>
struct ObservationState {
    double value = 0.;
    std::array<double,Width> gradient {};
    std::array<double,Width*(Width+1)/2> hessian {};
};

template<class Base, unsigned Request, bool Joint>
bool verify_power_observation_intention(double logh)
{
    using Instrumented = ObservedDensity<Base>;
    constexpr bool F = Request & 1, G = Request & 2, H = Request & 4;
    constexpr int order = H ? 2 : G ? 1 : 0;
    constexpr std::size_t width = 1+Base::shape_parameters;
    const double shape[] = {Base::shape_parameters==1 ? 1.7 : 7., .2};
    const auto prepared = Base::template prepare<F,G,H,Joint && (order>=1)>(shape,1);
    ObservationState<width> residual, log_variance, objective;
    const double scale = std::exp(logh/2);
    residual.value = .3*scale;
    residual.gradient[0] = .2*scale;
    residual.hessian[0] = -.1*scale;
    log_variance.value = logh;
    log_variance.gradient[0] = -.4;
    log_variance.hessian[0] = .7;
    if constexpr (!G) objective.gradient.fill(NAN);
    if constexpr (!H) objective.hessian.fill(NAN);
    auto value = Base::initial_value(prepared);
    Instrumented::gradient_requests = 0;
    bool valid;
    if constexpr (Joint)
        valid = model::egarch_observation::accumulate_joint<F,G,H,order,width,Instrumented>(
            prepared,residual,log_variance,value,objective,1,width);
    else
        valid = model::egarch_observation::accumulate_density<F,G,H,order,width,Instrumented>(
            prepared,residual.value,residual.value*residual.value,
            log_variance,value,objective,1,width);
    const bool minimal = Joint && std::is_same_v<Base,generated::SkewTDensity>
        && !model::egarch_observation::rescale_local_variance(logh);
    if (!valid || Instrumented::gradient_requests != unsigned(minimal ? G : order>=1))
        return false;
    for (double x : objective.gradient)
        if (G ? !std::isfinite(x) : !std::isnan(x)) return false;
    for (double x : objective.hessian)
        if (H ? !std::isfinite(x) : !std::isnan(x)) return false;
    if constexpr (H) {
        // Local slopes are still needed by the second-order chain rule.
        if (objective.hessian[0] == 0.) return false;
        ObservationState<width> full;
        auto full_value = Base::initial_value(prepared);
        bool full_valid;
        if constexpr (Joint)
            full_valid = model::egarch_observation::accumulate_joint<F,true,true,2,width,Base>(
                prepared,residual,log_variance,full_value,full,1,width);
        else
            full_valid = model::egarch_observation::accumulate_density<F,true,true,2,width,Base>(
                prepared,residual.value,residual.value*residual.value,
                log_variance,full_value,full,1,width);
        if (!full_valid) return false;
        for (std::size_t i=0; i<objective.hessian.size(); ++i)
            if (std::abs(objective.hessian[i]-full.hessian[i])
                > 1e-12*(1.+std::abs(full.hessian[i]))) return false;
    }
    return true;
}

template<unsigned Request>
bool power_observation_intentions()
{
    for (double logh : {.3, -400., 400.}) {
        if (!verify_power_observation_intention<generated::SkewTDensity,Request,true>(logh)
            || !verify_power_observation_intention<generated::SkewTDensity,Request,false>(logh)
            || !verify_power_observation_intention<generated::GedDensity,Request,true>(logh)
            || !verify_power_observation_intention<generated::GedDensity,Request,false>(logh))
            return false;
    }
    return true;
}

bool verify_normal_value_accumulation()
{
    using Normal = generated::NormalDensity;
    const auto evaluate = [](std::span<const double> terms, double constant) {
        auto prepared = Normal::prepare<true,false,false>(nullptr, terms.size());
        prepared.value_constant = constant;
        auto sum = Normal::initial_value(prepared);
        for (double term : terms) {
            generated::DensityObservation<0> observation;
            observation.value = term;
            Normal::accumulate_value(sum, observation);
        }
        return Normal::finalize_value(sum, prepared);
    };
    const std::array cancellation {1e16, 1., -1e16};
    const std::array reversed {-1e16, 1., 1e16};
    const std::array final_positive {1e16, 1.};
    const std::array final_negative {-1e16, -1.};
    return evaluate(cancellation, 0.) == .5
        && evaluate(reversed, 0.) == .5
        && evaluate(final_positive, -5e15) == .5
        && evaluate(final_negative, 5e15) == -.5
        && evaluate({}, 2.) == 2.;
}

int main()
{
    return masks<1>() && masks<2>() && masks<3>() && masks<4>()
        && masks<5>() && masks<6>() && masks<7>()
        && verify_residual_view<false,0>() && verify_residual_view<true,0>()
        && verify_residual_view<false,5>() && verify_residual_view<true,5>()
        && power_observation_intentions<1>() && power_observation_intentions<2>()
        && power_observation_intentions<3>() && power_observation_intentions<4>()
        && power_observation_intentions<5>() && power_observation_intentions<6>()
        && power_observation_intentions<7>()
        && verify_normal_value_accumulation() ? 0 : 1;
}
