#include "models/garch_generated_jet.hpp"
#include "models/gjr_transform_policy.hpp"
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
        if (family>1u) return 6; // Untransferred family, never success.
        if (!ok) return 5;
        print(result.theta); print(result.value); print(result.gradient); print(result.hessian);
        print(result.theta_value); print(result.theta_gradient); print(result.theta_hessian);
        print(double(result.map_exp_calls));
        std::printf("\n");
    }
}
