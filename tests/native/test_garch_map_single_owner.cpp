#include "models/garch_generated_jet.hpp"
#include "models/garch_runtime.hpp"
#include "models/joint/coordinate_map.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include "garch_map_hessian_capture.hpp"

namespace generated = vlk::experiment::generated;
namespace runtime = vlk::experiment::runtime;
namespace domain = vlk::experiment::domain;
namespace {
constexpr double sentinel = -1234567.25;
std::size_t checks = 0;
void check(bool condition, int line) {
    ++checks;
    if (!condition) { std::fprintf(stderr, "GARCH map owner line %d\n", line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__), __LINE__)
bool close(double a, double b, double relative = 3e-11, double absolute = 2e-12) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a-b) <= absolute + relative * std::max(std::abs(a),std::abs(b));
}

template<std::size_t P, std::size_t Q, class Shape>
void map_and_pullback(unsigned density) {
    using Fixed = generated::GarchTransform<P,Q,Shape>;
    using Map = generated::GarchCoordinateMap<Shape>;
    constexpr auto k = Fixed::parameters, c = P+Q;
    static_assert(decltype(Fixed::map_layout.coefficient_parameters)::value == c);
    static_assert(decltype(Fixed::map_layout.parameters)::value == k);
    static_assert(decltype(Fixed::map_layout.shape_start())::value == 1+c);
    const auto layout = generated::GarchMapLayout{std::size_t{c},std::size_t{k},std::size_t{0}};
    std::array<double,k> z{},theta{},other{},abi{},inverse{},g{},gz{},other_g{};
    std::array<double,c> weights{},other_weights{};
    std::array<double,k*k> j{},other_j{},h{},hz{},other_h{},intermediate{};
    generated::SimplexTailPoint fixed_point,runtime_point;
    z[0] = -0.8;
    for (std::size_t i=1;i<=c;++i) z[i] = -2.0 + 0.1*double(i);
    for (std::size_t i=0;i<Shape::parameters;++i) z[1+c+i] = 0.12*double(i+1);
    Fixed::pack(z.data(),theta.data(),&fixed_point,weights.data());
    Map::pack(layout,z.data(),other.data(),&runtime_point,other_weights.data());
    CHECK(theta == other && weights == other_weights);
    CHECK(fixed_point.mapped_admission == domain::Admission::inside);
    CHECK(runtime_point.shifted_weights == other_weights.data());
    CHECK(runtime::garch_coordinate_forward(P,Q,density,z.data(),abi.data()));
    CHECK(abi == theta);
    CHECK(Fixed::unpack(theta.data(),inverse.data()));
    for (std::size_t i=0;i<k;++i) CHECK(close(inverse[i],z[i]));
    Fixed::fill_jacobian_prepared(theta.data(),z.data(),j.data(),fixed_point);
    Map::fill_jacobian(layout,theta.data(),z.data(),other_j.data(),runtime_point);
    CHECK(j == other_j);
    for (std::size_t i=0;i<k;++i) {
        g[i] = (i%2 ? -0.3 : 0.4)*double(i+1);
        for (std::size_t t=0;t<k;++t)
            h[i*k+t] = (i==t ? 2.0 : 0.0) + 0.02*double((i+1)*(t+1));
    }
    Fixed::transform_gradient(theta.data(),z.data(),g.data(),gz.data(),&fixed_point);
    Map::transform_gradient(layout,theta.data(),z.data(),g.data(),other_g.data(),runtime_point);
    CHECK(gz == other_g);
    Fixed::transform_hessian(theta.data(),z.data(),g.data(),h.data(),hz.data(),&fixed_point);
    Map::transform_hessian(layout,theta.data(),z.data(),g.data(),h.data(),other_j.data(),
        intermediate.data(),other_h.data(),runtime_point);
    // Fixed unrolling and runtime loops need not contract to identical doubles.
    // Independent complete-map AD owns accuracy; this checks numerical parity.
    const auto same_hessian=[&] {
        for (std::size_t i=0;i<k*k;++i)
            if (!close(hz[i],other_h[i],32*std::numeric_limits<double>::epsilon(),0)) return false;
        return true;
    }();
    if (!same_hessian)
        capture_hessian<P,Q,Shape>(density,z,theta,g,h,j,other_j,hz,other_h,fixed_point,runtime_point);
    CHECK(same_hessian);
    for (unsigned mask : {2u,4u,6u}) {
        const auto bytes = runtime::garch_coordinate_pullback_bytes(P,Q,density,false,mask);
        CHECK(bytes > 0);
        std::vector<double> scratch((bytes+7)/8);
        other_g.fill(sentinel); other_h.fill(sentinel);
        CHECK(runtime::garch_coordinate_pullback(P,Q,density,false,z.data(),theta.data(),g.data(),
            h.data(),mask,scratch.data(),bytes,other_g.data(),other_h.data()));
        for (std::size_t i=0;i<k;++i) CHECK(mask&2 ? close(other_g[i],gz[i]) : other_g[i]==sentinel);
        for (std::size_t i=0;i<k*k;++i) CHECK(mask&4 ? close(other_h[i],hz[i]) : other_h[i]==sentinel);
    }
    Fixed::pack(z.data(),other.data(),&runtime_point);
    CHECK(runtime_point.shifted_weights == nullptr);
    Fixed::fill_jacobian_prepared(other.data(),z.data(),other_j.data(),runtime_point);
    CHECK(other_j == j);
    Fixed::transform_gradient(other.data(),z.data(),g.data(),other_g.data(),&runtime_point);
    Fixed::transform_hessian(other.data(),z.data(),g.data(),h.data(),other_h.data(),&runtime_point);
    CHECK(other_g == gz && other_h == hz);
    weights.fill(sentinel);
    Fixed::pack(z.data(),other.data(),nullptr,weights.data());
    for (double value:weights) CHECK(value==sentinel);
    Fixed::pack(z.data(),other.data(),&runtime_point,other.data()+2);
    CHECK(runtime_point.shifted_weights == nullptr);
    CHECK(runtime_point.mapped_admission == domain::Admission::numerical_range);
    for (double value:other) CHECK(std::isnan(value));
    auto mutable_z=z;
    Fixed::pack(mutable_z.data(),other.data(),&runtime_point,mutable_z.data()+1);
    CHECK(mutable_z==z && runtime_point.shifted_weights==nullptr);
    for (double value:other) CHECK(std::isnan(value));
}

template<std::size_t P, std::size_t Q, class Density, class Shape>
void complete_requests(unsigned density, double omega=-0.8, double dominant=-2.0,
    bool accepted=true) {
    using Transform = generated::GarchTransform<P,Q,Shape>;
    using Jet = generated::TransformedGarchJet<P,Q,Density,Transform>;
    constexpr auto k=Transform::parameters;
    std::array<double,k> z{},full_g{};
    std::array<double,k*k> full_h{};
    z[0]=omega;
    for (std::size_t i=1;i<=P+Q;++i) z[i]=-1.0;
    z[1]=dominant;
    if constexpr (Shape::parameters>0) z[1+P+Q]=0.24;
    const std::array<double,8> raw{0.37,-0.81,1.23,-0.19,0.62,-1.41,0.08,-0.54};
    std::array<double,8> squared{};
    for (std::size_t i=0;i<raw.size();++i) squared[i]=raw[i]*raw[i];
    const auto square_plan=scivolapi::detail::squared_shocks::prepare(
        raw.data(),squared.data(),raw.size());
    double full_f=0;
    for (unsigned mask : {7u,1u,2u,3u,4u,5u,6u}) {
        double f=sentinel,rf=sentinel;
        std::array<double,k> g,rg; std::array<double,k*k> h,rh;
        g.fill(sentinel);rg.fill(sentinel);h.fill(sentinel);rh.fill(sentinel);
        unsigned valid=123,rv=123;
        const auto bytes=vlk_runtime_garch_sample_jet_workspace_bytes(P,Q,density,1,mask);
        CHECK(bytes>0);
        std::vector<double> scratch((bytes+7)/8);
        const auto fixed_status=generated::evaluate_prevalidated_sample<Jet>(z.data(),
            {raw.data(),squared.data(),2,{},0.0,square_plan},raw.size(),0.93,mask,&f,g.data(),h.data(),&valid);
        const auto runtime_status=vlk_garch_selected_sample_jet_evaluate(false,true,z.data(),
            raw.data(),squared.data(),raw.size(),2,0.93,P,Q,density,1,mask,
            scratch.data(),bytes,&rf,rg.data(),rh.data(),&rv);
        CHECK(fixed_status==runtime_status);
        CHECK(fixed_status==(accepted ? VLK_JET_OK : VLK_JET_DOMAIN_ERROR));
        CHECK(valid==(accepted ? mask : 0) && rv==valid);
        if (!accepted) {
            CHECK(f==sentinel && rf==sentinel);
            for (std::size_t i=0;i<k;++i) CHECK(g[i]==sentinel && rg[i]==sentinel);
            for (std::size_t i=0;i<k*k;++i) CHECK(h[i]==sentinel && rh[i]==sentinel);
            continue;
        }
        if (mask==7) { full_f=f;full_g=g;full_h=h; }
        CHECK(mask&1 ? close(f,rf) && close(f,full_f) : f==sentinel && rf==sentinel);
        for (std::size_t i=0;i<k;++i)
            CHECK(mask&2 ? close(g[i],rg[i]) && close(g[i],full_g[i]) : g[i]==sentinel && rg[i]==sentinel);
        for (std::size_t i=0;i<k*k;++i)
            CHECK(mask&4 ? close(h[i],rh[i]) && close(h[i],full_h[i]) : h[i]==sentinel && rh[i]==sentinel);
    }
}

void endpoint_derivatives() {
    using Transform=generated::GarchNormalTransform<1,1>;
    std::array<double,3> z{-0.8,37,0},theta{},inverse{},g{0,1,0},gz{};
    std::array<double,2> weights{}; generated::SimplexTailPoint point;
    std::array<double,9> j{},h{},hz{};
    Transform::pack(z.data(),theta.data(),&point,weights.data());
    CHECK(theta[1]==0x1.ffffffffffffep-1 && point.mapped_admission==domain::Admission::inside);
    CHECK(Transform::unpack(theta.data(),inverse.data()) && inverse[1]!=z[1]);
    Transform::fill_jacobian_prepared(theta.data(),z.data(),j.data(),point);
    Transform::transform_gradient(theta.data(),z.data(),g.data(),gz.data(),&point);
    Transform::transform_hessian(theta.data(),z.data(),g.data(),h.data(),hz.data(),&point);
    CHECK(close(j[4],1.7066095251488127e-16,4e-15,0));
    CHECK(close(gz[1],j[4],4e-15,0));
    CHECK(close(hz[4],-1.7066095251488122e-16,4e-15,0));
    // Weighted subnormal slopes cannot be validated with an absolute tolerance.
    z={-745,-2,-1}; g={0x1p700,0,0}; h.fill(0);
    h[0]=0x1p1000;h[1]=h[3]=0x1p700;
    Transform::pack(z.data(),theta.data(),&point,weights.data());
    Transform::transform_gradient(theta.data(),z.data(),g.data(),gz.data(),&point);
    Transform::transform_hessian(theta.data(),z.data(),g.data(),h.data(),hz.data(),&point);
    // Evaluate the weighted exponential before its binary64 underflow plateau.
    const double weighted=std::exp(-745.0+700.0*std::log(2.0));
    CHECK(close(gz[0],weighted,5e-13,0));
    CHECK(close(hz[0],weighted,5e-13,0));
    Transform::fill_jacobian_prepared(theta.data(),z.data(),j.data(),point);
    CHECK(close(hz[1],weighted*j[4],5e-13,0));
    CHECK(hz[1]==hz[3] && hz[1]!=0);
    theta[2]=0; CHECK(!Transform::unpack(theta.data(),inverse.data()));
}

void joint_coefficient_block() {
    constexpr auto* fixed=&runtime::fixed_arma_garch_layout<3,2,3,2,false,false>;
    constexpr auto k=fixed->parameter_count,omega=fixed->omega_index,c=fixed->p_vol+fixed->q_vol;
    const auto tag=runtime::joint_map::plain_garch_map_layout<fixed>(*fixed);
    static_assert(decltype(tag.parameters)::value==k);
    static_assert(decltype(tag.omega_index)::value==omega);
    static_assert(decltype(tag.coefficient_parameters)::value==c);
    std::array<double,k> z{},theta{},other{},inverse{},g{};
    std::array<double,k*k> j{},other_j{},curvature{};
    for (std::size_t i=0;i<omega;++i) z[i]=0.02*double(i+1);
    z[omega]=-0.8;
    for (std::size_t i=omega+1;i<k;++i) z[i]=-2;
    const auto bytes=runtime::joint_map::garch_map_workspace_bytes<fixed>(*fixed);
    std::vector<double> memory((bytes+7)/8),other_memory((bytes+7)/8);
    runtime::ArmaGarchLayout poison{}; poison.parameter_count=poison.p_vol=999;
    std::int32_t status=123;
    CHECK(runtime::joint_map::garch_forward<fixed>(poison,z.data(),true,memory.data(),bytes,
        theta.data(),j.data(),&status) && status==VLK_JET_OK);
    CHECK(runtime::joint_map::garch_forward(*fixed,z.data(),true,other_memory.data(),bytes,
        other.data(),other_j.data(),&status) && status==VLK_JET_OK);
    CHECK(theta==other && j==other_j);
    using Standalone=generated::GarchNormalTransform<3,2>;
    std::array<double,1+c> sub{},sg{};std::array<double,c> weights{};
    std::array<double,(1+c)*(1+c)> sub_j{},sub_h{};
    generated::SimplexTailPoint simplex;
    Standalone::pack(z.data()+omega,sub.data(),&simplex,weights.data());
    Standalone::fill_jacobian_prepared(sub.data(),z.data()+omega,sub_j.data(),simplex);
    for (std::size_t i=0;i<1+c;++i) {
        CHECK(theta[omega+i]==sub[i]);
        g[omega+i]=sg[i]=double(i+1)*0.2;
        for (std::size_t t=0;t<1+c;++t) CHECK(j[(omega+i)*k+omega+t]==sub_j[i*(1+c)+t]);
    }
    const auto cb=runtime::joint_map::garch_curvature_workspace_bytes<fixed>(*fixed);
    std::vector<double> scratch((cb+7)/8);
    CHECK(runtime::joint_map::add_garch_curvature<fixed>(poison,z.data(),theta.data(),g.data(),
        scratch.data(),cb,curvature.data(),memory.data(),bytes));
    generated::GarchCoefficientMap::add_curvature<true>(Standalone::map_layout,sub.data(),
        z.data()+omega,sg.data(),sub_h.data(),simplex);
    for (std::size_t i=0;i<k;++i) for (std::size_t t=0;t<k;++t)
        CHECK(i>=omega && t>=omega ? curvature[i*k+t]==sub_h[(i-omega)*(1+c)+t-omega]
            : curvature[i*k+t]==0);
    CHECK(runtime::joint_map::garch_inverse<fixed>(poison,theta.data(),memory.data(),bytes,inverse.data()));
    for (std::size_t i=0;i<k;++i) CHECK(close(inverse[i],z[i]));
}
}

int main() {
#define MAP_CASES(S,C) map_and_pullback<1,1,generated::S>(C); map_and_pullback<3,2,generated::S>(C)
    MAP_CASES(NoShapeTransform,0);MAP_CASES(StudentTShapeTransform,1);
    MAP_CASES(SkewTShapeTransform,2);MAP_CASES(GedShapeTransform,3);
    MAP_CASES(GhShapeTransform,4);MAP_CASES(FixedLambdaGhShapeTransform,5);
    MAP_CASES(FixedLambdaGhShapeTransform,6);MAP_CASES(VarianceGammaShapeTransform,7);
#undef MAP_CASES
#define COMPLETE(D,S,C) complete_requests<1,1,generated::D,generated::S>(C); complete_requests<3,2,generated::D,generated::S>(C)
    COMPLETE(NormalDensity,NoShapeTransform,0);
    COMPLETE(StudentTDensity,StudentTShapeTransform,1);
    COMPLETE(SkewTDensity,SkewTShapeTransform,2);
    COMPLETE(GedDensity,GedShapeTransform,3);
#undef COMPLETE
    complete_requests<1,1,generated::NormalDensity,generated::NoShapeTransform>(0,-0.8,37);
    complete_requests<1,1,generated::NormalDensity,generated::NoShapeTransform>(0,-0.8,40,false);
    complete_requests<1,1,generated::NormalDensity,generated::NoShapeTransform>(0,-720);
    complete_requests<1,1,generated::NormalDensity,generated::NoShapeTransform>(0,50);
    CHECK(vlk_garch_dispatch_kind(1,1,0,1)==1);
    CHECK(vlk_garch_dispatch_kind(3,2,0,1)==0);
    endpoint_derivatives();joint_coefficient_block();
    std::printf("GARCH single-owner prototype: %zu checks\n",checks);
}
