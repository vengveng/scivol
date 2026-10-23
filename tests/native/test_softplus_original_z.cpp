#include "backend.hpp"
#include "joint_backend.hpp"
#include "models/gjr_transform_policy.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace generated=vlk::experiment::generated;
namespace ad=vlk::experiment::ad;
namespace zero=scivolapi::detail::backend;
namespace joint=scivolapi::detail::joint_backend;
using scivolapi::JetRequest;
using scivolapi::Status;
namespace {
constexpr double sentinel=-1234567.25,weight=1e300;
std::size_t checks=0;
void check(bool good,int line) {
    ++checks;
    if(!good) {std::fprintf(stderr,"original-z softplus line %d\n",line);std::exit(1);}
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
bool close(double a,double b) {
    return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=2e-14*std::abs(b);
}
// Independent 100-digit Decimal exp, exact binary64 z and weight as inputs.
// The first two z values map to the same positive binary64 omega.
struct Endpoint {double z,first_weighted,second_weighted;};
constexpr std::array endpoints{
    Endpoint{-744.,7.67194470417998e-24,7.67194470417998e-24},
    Endpoint{-743.9,8.478810172144383e-24,8.478810172144383e-24},
    Endpoint{-720.,2.0322308024242932e-13,2.0322308024242932e-13},
    Endpoint{-36.,2.319522830243569e284,2.3195228302435686e284},
    Endpoint{50.,1e300,1.9287498479639177e278},
    Endpoint{744.,1e300,7.67194470417998e-24}
};

void scalar(const Endpoint& e) {
    auto first=ad::Differential<1,1>::variable(e.z,0);first.gradient[0]=weight;
    const auto g=ad::softplus(first);
    CHECK(g.value==generated::softplus(e.z));
    CHECK(close(g.gradient[0],e.first_weighted));
    auto second=ad::Differential<2,1>::variable(e.z,0);second.gradient[0]=1e150;
    const auto h=ad::softplus(second);
    CHECK(close(h.hessian[0],e.second_weighted));
}

template<std::size_t P,std::size_t Q,bool Leverage>
void zero_map(const Endpoint& e) {
    using Transform=std::conditional_t<Leverage,
        generated::GjrTransform<P,Q,generated::NormalDensity>,
        generated::GarchTransform<P,Q,generated::NoShapeTransform>>;
    constexpr auto k=Transform::parameters;
    std::array<double,k> z,theta{},g{},gz{};
    std::array<double,k*k> h{},hz{},jacobian{};
    z.fill(-4.);z[0]=e.z;g[0]=weight;h[1]=h[k]=weight;
    Transform::pack(z.data(),theta.data());
    CHECK(theta[0]>0&&theta[0]==generated::softplus(e.z));
    Transform::fill_jacobian(theta.data(),z.data(),jacobian.data());
    Transform::transform_gradient(theta.data(),z.data(),g.data(),gz.data());
    Transform::transform_hessian(theta.data(),z.data(),g.data(),h.data(),hz.data());
    CHECK(close(gz[0],e.first_weighted));
    CHECK(close(hz[0],e.second_weighted));
    for(std::size_t j=1;j<k;++j) {
        const double expected=e.first_weighted*jacobian[k+j];
        CHECK(close(hz[j],expected));CHECK(close(hz[j*k],expected));
    }
    const zero::ModelConfig config{P,Q,0,Leverage};
    for(unsigned mask:{2u,4u,6u}) {
        const auto request=static_cast<JetRequest>(mask);
        const auto bytes=zero::chart_pullback_bytes(config,request);
        std::vector<double> memory((bytes+7)/8);
        std::array<double,k> rg;rg.fill(sentinel);
        std::array<double,k*k> rh;rh.fill(sentinel);
        CHECK(zero::chart_pullback(config,z,theta,request,std::as_writable_bytes(std::span(memory)),
            {{},g,h,{}},{{},rg,rh})==Status::ok);
        for(std::size_t j=0;j<k;++j) CHECK(mask&2?close(rg[j],gz[j]):rg[j]==sentinel);
        for(std::size_t j=0;j<k*k;++j) CHECK(mask&4?close(rh[j],hz[j]):rh[j]==sentinel);
    }
}

void joint_map(const Endpoint& e,joint::ModelConfig config) {
    const auto k=joint::parameter_count(config);
    const auto omega=k-(1+(config.leverage?2:1)*config.p_vol+config.q_vol);
    std::vector<double> z(k,-4.),theta(k),g(k),h(k*k);
    std::fill(z.begin(),z.begin()+omega,0.01);z[omega]=e.z;
    g[omega]=weight;h[omega*k]=h[omega]=weight;
    const auto need=joint::map_requirements(config);
    CHECK(need.status==Status::ok&&need.dispatch==(config.fixed?1:0));
    std::vector<double> map_memory((need.bytes+7)/8);
    CHECK(joint::map(config,z,theta,std::as_writable_bytes(std::span(map_memory)))==Status::ok);
    const std::array<double,4> y{0.3,-0.7,0.4,0.9};
    const joint::SampleView sample{y,{},0,0,0.8};
    for(unsigned mask:{2u,4u,6u}) {
        const auto request=static_cast<JetRequest>(mask);
        const auto bytes=joint::chart_pullback_bytes(config,request);
        std::vector<double> memory((bytes+7)/8),gz(k,sentinel),hz(k*k,sentinel);
        CHECK(joint::chart_pullback(config,sample,z,theta,request,std::as_writable_bytes(std::span(memory)),
            {{},g,h,{}},{{},gz,hz})==Status::ok);
        if(mask&2) CHECK(close(gz[omega],e.first_weighted));
        else for(double value:gz) CHECK(value==sentinel);
        if(mask&4) {
            CHECK(close(hz[omega*k+omega],e.second_weighted));
            CHECK(close(hz[omega*k],e.first_weighted));
            CHECK(close(hz[omega],e.first_weighted));
        } else for(double value:hz) CHECK(value==sentinel);
    }
}
}
int main() {
    CHECK(generated::softplus(endpoints[0].z)==generated::softplus(endpoints[1].z));
    CHECK(endpoints[0].first_weighted!=endpoints[1].first_weighted);
    for(const auto& e:endpoints) {
        scalar(e);
        zero_map<1,1,false>(e);zero_map<1,1,true>(e);
        zero_map<3,2,false>(e);zero_map<3,2,true>(e);
        for(bool leverage:{false,true}) for(bool fixed:{false,true}) {
            joint_map(e,{joint::Kind::arma,1,1,0,1,1,0,leverage,true,fixed});
            joint_map(e,{joint::Kind::linear,0,0,7,1,1,0,leverage,false,fixed});
        }
    }
    std::printf("original-z softplus: %zu checks\n",checks);
}
