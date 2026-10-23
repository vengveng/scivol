#include "garch_generated_jet.hpp"
#include "gjr_transform_policy.hpp"
#include "aparch_runtime_jet.hpp"
#include "joint_garch_program.hpp"
#include "arma_garch_normal_jet.hpp"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <vector>

volatile std::size_t exponential_calls = 0u;

#if defined(__APPLE__)
extern "C" __attribute__((noinline)) double exp(double value)
#else
extern "C" __attribute__((noinline)) double exp(double value) noexcept
#endif
{
    static const auto original = reinterpret_cast<double(*)(double)>(dlsym(RTLD_NEXT,"exp"));
    if (original == nullptr) std::abort();
    exponential_calls = exponential_calls + 1u;
    return original(value);
}

namespace generated = vlk::experiment::generated;
namespace runtime = vlk::experiment::runtime;
namespace power = runtime::aparch;

constexpr double data[] = {.3, -.7, 1.1, -.2, .9, -.4, .8, -.6};
constexpr std::size_t observations = sizeof(data) / sizeof(data[0]);

struct Outputs {
    std::vector<double> theta, gradient, hessian, theta_gradient, theta_hessian;
    double value = 123., theta_value = 0.;
    std::size_t map_exp_calls = 0u;
    explicit Outputs(std::size_t width)
        : theta(width), gradient(width, 456.), hessian(width * width, 789.),
          theta_gradient(width), theta_hessian(width * width) {}
};

template<std::size_t P, std::size_t Q, bool Leverage>
bool fixed_garch(const double* z, unsigned request, Outputs& result)
{
    using Density = generated::NormalDensity;
    using Recursion = std::conditional_t<Leverage,
        generated::GjrRecursion<P,Q>, generated::GarchRecursion<P,Q>>;
    using Transform = std::conditional_t<Leverage,
        generated::GjrTransform<P,Q,Density>, generated::GarchNormalTransform<P,Q>>;
    using Theta = generated::GarchJet<P,Q,Density,Recursion>;
    using Transformed = generated::TransformedGarchJet<P,Q,Density,Transform,Recursion>;
    double squared[observations];
    for (std::size_t i=0; i<observations; ++i) squared[i] = data[i] * data[i];
    generated::GarchSample sample {data, squared};
    Transform::pack(z, result.theta.data());
    if (!Transformed::valid_inputs(z, sample, observations, 1.)) return false;
    Theta::evaluate_planned(result.theta.data(), sample, observations, 1., 7u,
        &result.theta_value, result.theta_gradient.data(), result.theta_hessian.data());
    const std::size_t before = exponential_calls;
    Transformed::evaluate_planned(z, sample, observations, 1., request,
        &result.value, result.gradient.data(), result.hessian.data());
    result.map_exp_calls = exponential_calls - before;
    return true;
}

template<class Orders, int Order>
bool aparch(const power::Layout& layout, const double* z, unsigned request, Outputs& result)
{
    using Density = generated::NormalDensity;
    using Shape = generated::NoShapeTransform;
    constexpr const power::Layout* fixed = Orders::fixed
        ? &power::fixed_layout<Orders,Density,false> : nullptr;
    const std::size_t size = power::workspace_bytes<Density,Shape,Orders,false>(layout,true,7u);
    std::vector<std::byte> workspace(size);
    runtime::aparch_transform::Scratch scratch;
    const std::size_t core = power::theta_workspace_bytes<Density,Shape,Orders,false>(layout,7u);
    const std::size_t before = exponential_calls;
    if (!runtime::aparch_transform::bind_scratch<fixed>(layout,core,request,
            workspace.data(),workspace.size(),scratch)
        || !runtime::aparch_transform::map_coordinates<Order,Density,Shape,fixed>(layout,z,scratch))
        return false;
    result.map_exp_calls = exponential_calls - before;
    std::copy_n(scratch.theta, layout.parameters, result.theta.data());
    // Isolate the mathematical consumer from the separate persistence checker.
    if (!power::evaluate_prepared_theta<2,false,Orders,Density,true,true,true>(
            layout,result.theta.data(),data,observations,1.,workspace.data(),core,
            &result.theta_value,result.theta_gradient.data(),result.theta_hessian.data(),
            nullptr,nullptr,nullptr,static_cast<void*>(nullptr),static_cast<void*>(nullptr)))
        return false;
    if constexpr (Order >= 1)
        std::copy(result.theta_gradient.begin(),result.theta_gradient.end(),scratch.theta_gradient);
    if constexpr (Order >= 2)
        std::copy(result.theta_hessian.begin(),result.theta_hessian.end(),scratch.theta_hessian);
    if (request & 1u) result.value=result.theta_value;
    if (request & 2u) runtime::aparch_transform::pullback_gradient<fixed>(
        layout,scratch,scratch.theta_gradient,result.gradient.data());
    return !(request & 4u) || runtime::aparch_transform::pullback_hessian<Density,Shape,fixed>(
        layout,z,scratch,result.hessian.data());
}

template<class Orders>
bool aparch_requests(const power::Layout& layout, const double* z, unsigned request, Outputs& result)
{
    if (request & 4u) return aparch<Orders,2>(layout,z,request,result);
    if (request & 2u) return aparch<Orders,1>(layout,z,request,result);
    return aparch<Orders,0>(layout,z,request,result);
}

runtime::ArmaGarchLayout joint_layout(std::size_t p, std::size_t q)
{
    auto layout = runtime::joint_garch::linear_layout(3u,p,q,false);
    layout.p_ar=1u; layout.q_ma=1u; layout.ma_start=2u;
    layout.maximum_lag=std::max({p,q,std::size_t{1u}});
    return layout;
}

template<bool Fixed, std::size_t P=1u, std::size_t Q=1u>
bool joint(const runtime::ArmaGarchLayout& layout, const double* z,
    unsigned request, Outputs& result)
{
    using Density = generated::NormalDensity;
    using Shape = generated::NoShapeTransform;
    auto theta_kernel = [&](const runtime::ArmaGarchLayout& current,
        const double* theta, std::size_t, const double* y, std::size_t n, double h0,
        unsigned mask, void* memory, std::size_t bytes, double* f, double* g, double* h,
        unsigned* valid) {
        bool ok = false;
        if constexpr (Fixed) {
            ok = generated::ArmaGarchJet<1u,1u,P,Q,Density>::evaluate_planned(
                theta,y,n,h0,mask,f,g,h);
        } else {
            runtime::WorkspaceArena arena(memory,bytes);
            runtime::joint_garch::Scratch scratch;
            if (!runtime::joint_garch::bind_scratch(arena,current,mask,scratch)) return 1;
#define REQUEST(MASK,F,G,H) case MASK: ok=runtime::joint_garch::evaluate<Density,false,F,G,H>(current,theta,y,n,h0,scratch,f,g,h); break
            switch (mask) {
                REQUEST(1u,true,false,false);
                REQUEST(2u,false,true,false);
                REQUEST(3u,true,true,false);
                REQUEST(4u,false,false,true);
                REQUEST(5u,true,false,true);
                REQUEST(6u,false,true,true);
                REQUEST(7u,true,true,true);
            }
#undef REQUEST
        }
        *valid=ok?mask:0u;
        return ok?0:1;
    };
    std::vector<std::byte> map_memory(runtime::arma_garch_map_workspace_bytes(layout));
    if (!runtime::arma_garch_coordinate_forward(layout,z,false,
            map_memory.data(),map_memory.size(),result.theta.data(),nullptr)) return false;
    std::vector<std::byte> theta_memory(runtime::joint_garch::workspace_bytes(layout,7u));
    unsigned valid=0;
    if (theta_kernel(layout,result.theta.data(),layout.parameter_count,data,observations,1.,
        7u,theta_memory.data(),theta_memory.size(),&result.theta_value,
        result.theta_gradient.data(),result.theta_hessian.data(),&valid)) return false;
    std::vector<std::byte> workspace(runtime::joint_garch::transformed_workspace_bytes<Density>(
        layout,0u,request,Fixed));
    const std::size_t before = exponential_calls;
    const bool ok = runtime::joint_garch::evaluate_transformed<Density,Shape,false,false>(
        layout,0u,z,layout.parameter_count,data,observations,1.,request,
        workspace.data(),workspace.size(),&result.value,result.gradient.data(),
        result.hessian.data(),&valid,Fixed,theta_kernel)==0 && valid==request;
    result.map_exp_calls = exponential_calls - before;
    return ok;
}

void print(double value) { std::printf("%a ",value); }
void print(const std::vector<double>& values) { for (double value:values) print(value); }

int main(int argc, char** argv)
{
    if (argc==3) {
        const std::size_t count=std::strtoul(argv[1],nullptr,10);
        const double dominant=std::strtod(argv[2],nullptr);
        std::vector<double> weights(count);
        const generated::SimplexTailPoint simplex(count,
            [&](std::size_t index) {return index==0u?dominant:-2.-.125*double(index);},weights.data());
        const std::size_t prepared=exponential_calls;
        double checksum=0.;
        for (std::size_t output=0;output<count;++output) {
            const double value=simplex.contribution(output,1.);
            for (std::size_t row=0;row<count;++row) {
                checksum+=simplex.first(output,row,value,simplex.weight(row));
                for (std::size_t column=0;column<count;++column)
                    checksum+=simplex.second(output,row,column,value,
                        simplex.weight(row),simplex.weight(column));
            }
        }
        std::printf("%zu %zu %a\n",prepared,std::size_t(exponential_calls),checksum);
        return 0;
    }
    unsigned family,p,q,fixed,request;
    while (std::scanf("%u %u %u %u %u",&family,&p,&q,&fixed,&request)==5) {
        if (q != 1u || (p != 1u && p != 3u)) return 2;
        const std::size_t width = family==0u ? 1u+p+q : family==1u ? 1u+2u*p+q
            : family==2u ? 2u+2u*p+q : 4u+p+q;
        std::vector<double> z(width);
        char token[128];
        for (double& value:z) {
            if (std::scanf("%127s",token)!=1) return 3;
            value=std::strtod(token,nullptr);
        }
        Outputs result(width);
        bool ok=false;
        if (family==0u) ok=p==1u?fixed_garch<1u,1u,false>(z.data(),request,result)
            :fixed_garch<3u,1u,false>(z.data(),request,result);
        if (family==1u) ok=p==1u?fixed_garch<1u,1u,true>(z.data(),request,result)
            :fixed_garch<3u,1u,true>(z.data(),request,result);
        if (family==2u) {
            power::Layout layout;
            if (!power::make_layout<generated::NormalDensity>(0u,0u,p,q,false,layout)) return 4;
            if (!fixed) ok=aparch_requests<power::RuntimeOrders>(layout,z.data(),request,result);
            else if (p==1u) ok=aparch_requests<power::FixedVolatilityOrders<1u,1u>>(layout,z.data(),request,result);
            else ok=aparch_requests<power::FixedVolatilityOrders<3u,1u>>(layout,z.data(),request,result);
        }
        if (family==3u) {
            const auto layout=joint_layout(p,q);
            if (!fixed) ok=joint<false>(layout,z.data(),request,result);
            else if (p==1u) ok=joint<true,1u,1u>(layout,z.data(),request,result);
            else ok=joint<true,3u,1u>(layout,z.data(),request,result);
        }
        if (!ok) return 5;
        print(result.theta); print(result.value); print(result.gradient); print(result.hessian);
        print(result.theta_value); print(result.theta_gradient); print(result.theta_hessian);
        print(double(result.map_exp_calls));
        std::printf("\n");
    }
}
