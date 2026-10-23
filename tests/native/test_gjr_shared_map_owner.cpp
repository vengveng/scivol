#include "models/gjr_transform_policy.hpp"
#include "models/garch_generated_jet.hpp"
#include "models/garch_runtime.hpp"
#include "models/joint/coordinate_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace generated=vlk::experiment::generated;
namespace native=vlk::experiment::runtime;
namespace owner=vlk::experiment::runtime::gjr_coordinate;
namespace {
constexpr double sentinel=-1234567.25;
std::size_t checks=0;
std::array<unsigned,3> moment_calls{};
int failed_order=-1;
struct ProbeDensity : generated::NormalDensity {};

void check(bool good,int line) {
    ++checks;
    if (!good) {std::fprintf(stderr,"GJR shared map owner line %d\n",line);std::exit(1);}
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
bool close(double actual,double expected) {
    return std::isfinite(actual) && std::isfinite(expected)
        && (actual==expected || std::abs(actual-expected)<=2e-12*std::abs(expected));
}
void compare(const auto& actual,const auto& expected) {
    CHECK(actual.size()==expected.size());
    for (std::size_t i=0;i<actual.size();++i) CHECK(close(actual[i],expected[i]));
}
template<int Order> generated::NegativeSecondMoment<0> probe_moment() {
    ++moment_calls[Order];
    generated::NegativeSecondMoment<0> result;
    result.value=0.5;result.valid=failed_order!=Order;
    return result;
}
}
namespace vlk::experiment::generated {
template<> struct DensityShapeTransform<ProbeDensity> {using type=NoShapeTransform;};
template<> inline constexpr bool exact_half_moment<ProbeDensity> = true;
template<> inline NegativeSecondMoment<0>
negative_second_moment<0,ProbeDensity>(const double*) noexcept {return probe_moment<0>();}
template<> inline NegativeSecondMoment<0>
negative_second_moment<1,ProbeDensity>(const double*) noexcept {return probe_moment<1>();}
template<> inline NegativeSecondMoment<0>
negative_second_moment<2,ProbeDensity>(const double*) noexcept {return probe_moment<2>();}
}
namespace {
template<std::size_t P,std::size_t Q,class Density> void map_contract(unsigned density) {
    using Transform=generated::GjrTransform<P,Q,Density>;
    using Shape=generated::DensityShapeTransformT<Density>;
    constexpr auto fixed=Transform::layout;
    constexpr auto runtime=owner::map_layout<Density>(P,Q);
    static_assert(fixed.p==runtime.p && fixed.q==runtime.q && fixed.shape==runtime.shape);
    static_assert(fixed.variance_parameters==runtime.variance_parameters);
    static_assert(fixed.parameters==runtime.parameters && fixed.alpha_start==runtime.alpha_start);
    static_assert(fixed.gamma_start==runtime.gamma_start && fixed.beta_start==runtime.beta_start);
    static_assert(decltype(fixed.p)::value==P && decltype(fixed.q)::value==Q);
    static_assert(decltype(fixed.parameters)::value==Transform::parameters);
    constexpr auto k=Transform::parameters,count=Transform::contribution_parameters;
    std::array<double,k> z,theta{},reference{},g{},gz{},expected_g{};
    std::array<double,count> weights{},other_weights{};
    std::array<double,k*k> j{},expected_j{},h{},hz{},expected_h{},scratch{};
    z.fill(-4.);z[0]=-0.8;
    std::fill(z.begin()+Transform::variance_parameters,z.end(),0.0);
    generated::SimplexTailPoint point,other;
    Transform::pack(z.data(),theta.data(),&point,weights.data());
    CHECK((owner::pack_gjr_coordinates<Density,Shape>(
        runtime,z.data(),reference.data(),&other,other_weights.data())));
    compare(theta,reference);
    const auto saved_theta=theta;
    std::array<double,k> inverse{},runtime_inverse{},public_inverse{},repacked{};
    CHECK(Transform::unpack(theta.data(),inverse.data()));
    CHECK((owner::unpack_gjr_coordinates<Density,Shape>(
        runtime,theta.data(),runtime_inverse.data())));
    CHECK(native::gjr_coordinate_inverse(P,Q,density,theta.data(),public_inverse.data()));
    compare(inverse,runtime_inverse);compare(inverse,public_inverse);
    Transform::pack(inverse.data(),repacked.data());
    for (std::size_t i=0;i<k;++i)
        CHECK(std::abs(repacked[i]-theta[i])<=4e-12*(1.0+std::abs(theta[i])));
    CHECK(theta==saved_theta);
    if constexpr(Density::shape_parameters!=0) {
        auto invalid=theta;
        invalid[Transform::variance_parameters]=std::numeric_limits<double>::quiet_NaN();
        CHECK(!Transform::unpack(invalid.data(),inverse.data()));
        CHECK((!owner::unpack_gjr_coordinates<Density,Shape>(
            runtime,invalid.data(),runtime_inverse.data())));
        CHECK(!native::gjr_coordinate_inverse(P,Q,density,invalid.data(),public_inverse.data()));
    }
    CHECK(point.shifted_weights==weights.data());
    CHECK(point.mapped_admission==other.mapped_admission);
    CHECK(point.negative_moment.value==other.negative_moment.value);
    CHECK(point.negative_moment.valid==other.negative_moment.valid);
    CHECK(point.negative_moment.exact_half==other.negative_moment.exact_half);
    const auto retained=weights;
    Transform::fill_jacobian_prepared(theta.data(),z.data(),j.data(),point);
    CHECK((owner::fill_gjr_transform_jacobian<Density,Shape>(
        runtime,z.data(),reference.data(),expected_j.data(),&other)));
    compare(j,expected_j);
    for (std::size_t i=0;i<k;++i) {
        g[i]=double(i+1)/double(k);
        for (std::size_t column=0;column<k;++column)
            h[i*k+column]=(i==column?1.:0.)+double(i+1)*double(column+1)/double(k*k);
    }
    Transform::transform_gradient(theta.data(),z.data(),g.data(),gz.data(),&point);
    CHECK((owner::pullback_gjr_gradient<Density,Shape>(
        runtime,z.data(),reference.data(),g.data(),expected_j.data(),expected_g.data(),&other)));
    compare(gz,expected_g);
    Transform::transform_hessian(theta.data(),z.data(),g.data(),h.data(),hz.data(),&point);
    CHECK((owner::pullback_gjr_hessian<Density,Shape>(runtime,z.data(),reference.data(),
        g.data(),h.data(),expected_j.data(),scratch.data(),expected_h.data(),&other)));
    compare(hz,expected_h);CHECK(weights==retained);
    Transform::pack(z.data(),reference.data(),&other);
    compare(theta,reference);CHECK(other.shifted_weights==nullptr);
    Transform::fill_jacobian(reference.data(),z.data(),expected_j.data());compare(j,expected_j);
    Transform::transform_gradient(reference.data(),z.data(),g.data(),expected_g.data());compare(gz,expected_g);
    Transform::transform_hessian(reference.data(),z.data(),g.data(),h.data(),expected_h.data());compare(hz,expected_h);
    Transform::pack(z.data(),reference.data(),&other,reference.data()+1);
    compare(theta,reference);CHECK(other.shifted_weights==nullptr);
    Transform::pack(z.data(),reference.data(),&other,reference.data()+2);
    CHECK(other.shifted_weights==nullptr && !other.negative_moment.valid);
    for (double value:reference) CHECK(std::isnan(value));
    const auto saved_z=z;
    Transform::pack(z.data(),reference.data(),&other,z.data()+1);
    CHECK(z==saved_z && other.shifted_weights==nullptr);
    for (double value:reference) CHECK(std::isnan(value));
}

void preparation_and_failures() {
    using Transform=generated::GjrTransform<2,3,ProbeDensity>;
    constexpr auto k=Transform::parameters;
    std::array<double,k> z,theta{},g{},gz{},inverse{};
    std::array<double,Transform::contribution_parameters> weights;
    std::array<double,k*k> j{},h{},hz{};
    z.fill(-4.);z[0]=-0.8;g.fill(0.25);
    generated::SimplexTailPoint point;
    moment_calls.fill(0);
    Transform::pack(z.data(),theta.data(),&point,weights.data());
    CHECK((moment_calls==std::array<unsigned,3>{1,0,0}));
    CHECK(point.negative_moment.valid && point.negative_moment.exact_half);
    moment_calls.fill(0);
    CHECK(Transform::unpack(theta.data(),inverse.data()));
    CHECK((moment_calls==std::array<unsigned,3>{1,0,0}));
    moment_calls.fill(0);
    Transform::transform_gradient(theta.data(),z.data(),g.data(),gz.data(),&point);
    CHECK((moment_calls==std::array<unsigned,3>{0,1,0}));
    moment_calls.fill(0);
    Transform::transform_hessian(theta.data(),z.data(),g.data(),h.data(),hz.data(),&point);
    CHECK((moment_calls==std::array<unsigned,3>{0,1,1}));
    failed_order=1;
    Transform::fill_jacobian_prepared(theta.data(),z.data(),j.data(),point);
    for (double value:j) CHECK(std::isnan(value));
    Transform::transform_gradient(theta.data(),z.data(),g.data(),gz.data(),&point);
    for (double value:gz) CHECK(std::isnan(value));
    failed_order=2;
    Transform::transform_hessian(theta.data(),z.data(),g.data(),h.data(),hz.data(),&point);
    for (double value:hz) CHECK(std::isnan(value));
    failed_order=0;
    CHECK(!Transform::unpack(theta.data(),inverse.data()));
    Transform::pack(z.data(),theta.data(),&point,weights.data());
    CHECK(!point.negative_moment.valid && point.shifted_weights==nullptr);
    for (double value:theta) CHECK(std::isnan(value));
    failed_order=-1;
    z[0]=-1000.;
    Transform::pack(z.data(),theta.data(),&point,weights.data());
    CHECK(point.mapped_admission==vlk::experiment::domain::Admission::numerical_range);
    CHECK(point.negative_moment.valid && point.negative_moment.exact_half);
    CHECK(point.shifted_weights==weights.data() && theta[0]==0.0);
    for (double value:theta) CHECK(std::isfinite(value));
    z.fill(-1000.);z[0]=-0.8;z[Transform::alpha_start]=1000.;
    Transform::pack(z.data(),theta.data(),&point,weights.data());
    CHECK(point.mapped_admission==vlk::experiment::domain::Admission::mathematical_domain);
    CHECK(point.negative_moment.valid && point.negative_moment.exact_half);
    for (double value:theta) CHECK(std::isfinite(value));
}

void inverse_rejections() {
    using Density=generated::NormalDensity;
    using Shape=generated::NoShapeTransform;
    using Transform=generated::GjrTransform<1,1,Density>;
    constexpr auto runtime=owner::map_layout<Density>(std::size_t{1},std::size_t{1});
    // Exact contributions 1/8,1/8,1/4 and slack 1/2 give these log ratios.
    const std::array<double,4> canonical_theta{0.6931471805599453,0.25,0.0,0.25};
    const std::array<double,4> canonical_z{0.0,-1.3862943611198906,-1.3862943611198906,-0.6931471805599453};
    std::array<double,4> actual_z{};
    CHECK(Transform::unpack(canonical_theta.data(),actual_z.data()));
    for (std::size_t i=0;i<actual_z.size();++i)
        CHECK(std::isfinite(actual_z[i]) && std::abs(actual_z[i]-canonical_z[i])<=2e-14);
    const std::array<double,4> interior{0.3,0.2,-0.05,0.4};
    const auto reject=[&](const std::array<double,4>& theta) {
        const auto saved=theta;
        std::array<double,4> fixed_z{},runtime_z{},public_z{};
        CHECK(!Transform::unpack(theta.data(),fixed_z.data()));
        CHECK((!owner::unpack_gjr_coordinates<Density,Shape>(
            runtime,theta.data(),runtime_z.data())));
        CHECK(!native::gjr_coordinate_inverse(1,1,VLK_GARCH_NORMAL,theta.data(),public_z.data()));
        CHECK(theta==saved);
    };
    for (std::size_t axis:{0u,1u,3u}) {
        auto theta=interior;theta[axis]=0.0;
        if (axis==1u) theta[2]=0.05;
        reject(theta);
        theta[axis]=-0.1;reject(theta);
        theta[axis]=std::numeric_limits<double>::infinity();reject(theta);
    }
    auto theta=interior;theta[2]=-theta[1];reject(theta);
    theta[2]=-theta[1]-0.1;reject(theta);
    reject({0.3,0.5,0.0,0.5});
}

void weighted_tails() {
    using Transform=generated::GjrTransform<2,3,generated::NormalDensity>;
    constexpr auto k=Transform::parameters;
    double plateau_inverse=0.0;
    // Independent literals already used by test_softplus_original_z.
    for (auto [coordinate,first,second]:std::array{
        std::array{-744.,7.67194470417998e-24,7.67194470417998e-24},
        std::array{-743.9,8.478810172144383e-24,8.478810172144383e-24},
        std::array{50.,1e300,1.9287498479639177e278},
        std::array{744.,1e300,7.67194470417998e-24}}) {
        std::array<double,k> z,theta{},g{},gz{};
        std::array<double,k*k> h{},hz{},j{};
        z.fill(-4.);z[0]=coordinate;g[0]=1e300;h[1]=h[k]=1e300;
        Transform::pack(z.data(),theta.data());
        Transform::fill_jacobian(theta.data(),z.data(),j.data());
        Transform::transform_gradient(theta.data(),z.data(),g.data(),gz.data());
        Transform::transform_hessian(theta.data(),z.data(),g.data(),h.data(),hz.data());
        CHECK(close(gz[0],first));CHECK(close(hz[0],second));
        for (std::size_t column=1;column<k;++column) {
            CHECK(close(hz[column],first*j[k+column]));
            CHECK(close(hz[column*k],first*j[k+column]));
        }
        std::array<double,k> inverse{},repacked{};
        CHECK(Transform::unpack(theta.data(),inverse.data()));
        Transform::pack(inverse.data(),repacked.data());
        CHECK(repacked[0]==theta[0]);
        if (coordinate==-744.) plateau_inverse=inverse[0];
        if (coordinate==-743.9) CHECK(inverse[0]==plateau_inverse);
    }
}

template<class Density> void joint_inverse(unsigned density) {
    constexpr auto* fixed=&native::fixed_arma_garch_layout<3,2,2,3,true,false>;
    constexpr auto mean=fixed->mean_parameters,k=fixed->parameter_count+Density::shape_parameters;
    using Transform=generated::GjrTransform<2,3,Density>;
    std::array<double,k> z{},theta{},inverse{},runtime_inverse{},repacked{};
    std::array<double,Transform::parameters> standalone_inverse{};
    for (std::size_t i=0;i<mean;++i) z[i]=0.01*double(i+1);
    z[mean]=-0.8;
    std::fill(z.begin()+mean+1,z.begin()+fixed->parameter_count,-4.0);
    const auto bytes=native::joint_map::gjr_map_workspace_bytes<Density,fixed>(*fixed,density);
    CHECK(bytes>0);
    std::vector<double> memory((bytes+7)/8);
    std::int32_t status=123;
    CHECK((native::joint_map::gjr_forward<Density,fixed>(*fixed,density,z.data(),false,
        memory.data(),bytes,theta.data(),nullptr,&status)) && status==VLK_JET_OK);
    const auto saved=theta;
    native::ArmaGarchLayout poison{};poison.parameter_count=poison.p_vol=999;
    CHECK((native::joint_map::gjr_inverse<Density,fixed>(poison,density,theta.data(),
        memory.data(),bytes,inverse.data())));
    CHECK((native::joint_map::gjr_inverse<Density>(*fixed,density,theta.data(),
        memory.data(),bytes,runtime_inverse.data())));
    compare(inverse,runtime_inverse);
    CHECK(Transform::unpack(theta.data()+mean,standalone_inverse.data()));
    for (std::size_t i=0;i<standalone_inverse.size();++i)
        CHECK(close(inverse[mean+i],standalone_inverse[i]));
    CHECK((native::joint_map::gjr_forward<Density,fixed>(*fixed,density,inverse.data(),false,
        memory.data(),bytes,repacked.data(),nullptr,&status)) && status==VLK_JET_OK);
    for (std::size_t i=0;i<k;++i)
        CHECK(std::abs(repacked[i]-theta[i])<=4e-12*(1.0+std::abs(theta[i])));
    CHECK(theta==saved);
    CHECK((!native::joint_map::gjr_inverse<Density,fixed>(poison,density,theta.data(),
        memory.data(),bytes-1,inverse.data())));
}

template<std::size_t P,std::size_t Q> void complete_masks() {
    using Density=generated::NormalDensity;
    using Transform=generated::GjrTransform<P,Q,Density>;
    using Jet=generated::TransformedGarchJet<P,Q,Density,Transform,generated::GjrRecursion<P,Q>>;
    constexpr auto k=Transform::parameters;
    const std::array<double,8> raw{0.3,-0.7,0.4,0.9,-0.5,0.2,-0.6,0.8};
    std::array<double,8> squares;
    for (std::size_t i=0;i<raw.size();++i) squares[i]=raw[i]*raw[i];
    std::array<double,k> z;z.fill(-4.);z[0]=-0.8;
    std::array<double,k> full_g{};
    std::array<double,k*k> full_h{};double full_f=0;
    for (unsigned mask:{7u,1u,2u,3u,4u,5u,6u}) {
        std::array<double,k> g,rg;
        std::array<double,k*k> h,rh;
        g.fill(sentinel);rg.fill(sentinel);h.fill(sentinel);rh.fill(sentinel);
        double f=sentinel,rf=sentinel;unsigned valid=0,rv=0;
        CHECK((generated::evaluate_checked_sample<Jet>(z.data(),
            {raw.data(),squares.data()},raw.size(),0.8,mask,&f,g.data(),h.data(),&valid)==VLK_JET_OK));
        const auto bytes=vlk_runtime_gjr_garch_sample_jet_workspace_bytes(P,Q,VLK_GARCH_NORMAL,1,mask);
        std::vector<double> work((bytes+7)/8);
        CHECK(vlk_runtime_gjr_garch_sample_jet_evaluate(z.data(),raw.data(),squares.data(),
            raw.size(),0.8,P,Q,VLK_GARCH_NORMAL,1,mask,work.data(),bytes,&rf,rg.data(),rh.data(),&rv)==VLK_JET_OK);
        CHECK(valid==mask && rv==mask);
        CHECK(mask&1 ? close(f,rf) : f==sentinel && rf==sentinel);
        for (std::size_t i=0;i<k;++i) CHECK(mask&2 ? close(g[i],rg[i]) : g[i]==sentinel && rg[i]==sentinel);
        for (std::size_t i=0;i<k*k;++i) CHECK(mask&4 ? close(h[i],rh[i]) : h[i]==sentinel && rh[i]==sentinel);
        if (mask==7) {full_f=f;full_g=g;full_h=h;}
        else {
            if (mask&1) CHECK(close(f,full_f));
            if (mask&2) compare(g,full_g);
            if (mask&4) compare(h,full_h);
        }
    }
}
}
int main() {
    map_contract<1,1,generated::NormalDensity>(VLK_GARCH_NORMAL);
    map_contract<1,1,generated::StudentTDensity>(VLK_GARCH_STUDENTT);
    map_contract<1,1,generated::SkewTDensity>(VLK_GARCH_SKEWT);
    map_contract<1,1,generated::GedDensity>(VLK_GARCH_GED);
    map_contract<1,1,generated::GeneralizedHyperbolicDensity>(VLK_GARCH_GH);
    map_contract<1,1,generated::NormalInverseGaussianDensity>(VLK_GARCH_NIG);
    map_contract<1,1,generated::HyperbolicDensity>(VLK_GARCH_HYPERBOLIC);
    map_contract<1,1,generated::VarianceGammaDensity>(VLK_GARCH_VARIANCE_GAMMA);
    map_contract<3,2,generated::NormalDensity>(VLK_GARCH_NORMAL);
    map_contract<2,3,generated::SkewTDensity>(VLK_GARCH_SKEWT);
    preparation_and_failures();inverse_rejections();weighted_tails();
    joint_inverse<generated::NormalDensity>(VLK_GARCH_NORMAL);
    joint_inverse<generated::SkewTDensity>(VLK_GARCH_SKEWT);
    complete_masks<1,1>();complete_masks<3,2>();
    std::printf("GJR shared map owner: %zu checks\n",checks);
}
