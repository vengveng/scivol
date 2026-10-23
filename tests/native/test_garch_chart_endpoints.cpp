#include "models/garch_generated_jet.hpp"
#include "models/gjr_transform_policy.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#if defined(SCIVOLAPI_LINKED_ENDPOINT_TEST)
#include "backend.hpp"
#include "joint_backend.hpp"
#include "mean_backend.hpp"
#include "power_backend.hpp"
#include <vector>
#endif

namespace generated=vlk::experiment::generated;
namespace domain=vlk::experiment::domain;
namespace {
std::size_t checks=0;
constexpr double sentinel=-1234567.25;
void check(bool valid,const char* expression,int line) {
    ++checks;
    if(!valid) { std::fprintf(stderr,"chart endpoint line %d: %s\n",line,expression); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__),#__VA_ARGS__,__LINE__)
bool close(double a,double b,double relative=2e-12,double absolute=2e-13) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=absolute+relative*std::max(std::abs(a),std::abs(b));
}
using Transform=generated::GarchTransform<1,1,generated::NoShapeTransform>;
using Theta=generated::GarchJet<1,1,generated::NormalDensity>;
using Chart=generated::TransformedGarchJet<1,1,generated::NormalDensity,Transform>;
constexpr std::array<double,8> raw{0.3,-0.7,1.1,-0.2,0.5,-1.3,0.9,0.4};
std::array<double,8> squares() {
    std::array<double,8> result{};
    for(std::size_t i=0;i<raw.size();++i) result[i]=raw[i]*raw[i];
    return result;
}

void exact_theta() {
    const std::array<double,3> inside{0.2,0x1.fffffffffffffp-1,0x1p-54};
    CHECK(inside[1]+inside[2]==1.0);
    CHECK(Theta::parameter_status(inside.data())==VLK_JET_OK);
    const std::array<double,3> equal{0.2,0.5,0.5};
    CHECK(Theta::parameter_status(equal.data())==VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    const std::array<double,3> outside{0.2,1.0,std::numeric_limits<double>::denorm_min()};
    CHECK(Theta::parameter_status(outside.data())==VLK_JET_MATHEMATICAL_DOMAIN_ERROR);
    const auto r2=squares();
    const auto square_plan=scivolapi::detail::squared_shocks::prepare(raw.data(),r2.data(),raw.size());
    for(unsigned mask=1;mask<=7;++mask) {
        double f=sentinel; std::array<double,3> g; std::array<double,9> h;
        g.fill(sentinel); h.fill(sentinel); unsigned valid=0;
        CHECK(generated::evaluate_prevalidated_sample<Theta>(inside.data(),{raw.data(),r2.data(),2,{},0.0,square_plan},
            raw.size(),0.8,mask,&f,g.data(),h.data(),&valid)==VLK_JET_OK);
        CHECK(valid==mask);
        if(!(mask&1)) CHECK(f==sentinel);
        if(!(mask&2)) for(double x:g) CHECK(x==sentinel);
        if(!(mask&4)) for(double x:h) CHECK(x==sentinel);
    }
}
void tail_map() {
    const std::array<double,3> z{-0.8,37,0};
    std::array<double,3> theta{}; std::array<double,2> weights{};
    generated::SimplexTailPoint point;
    Transform::pack(z.data(),theta.data(),&point,weights.data());
    CHECK(theta[1]==0x1.ffffffffffffep-1);
    CHECK(theta[2]>0 && point.mapped_admission==domain::Admission::inside);
    CHECK(close(point.first(0,0,theta[1],point.weight(0)),1.7066095251488127e-16,3e-15,0));
    CHECK(close(point.second(0,0,0,theta[1],point.weight(0),point.weight(0)),
        -1.7066095251488122e-16,3e-15,0));
    const auto r2=squares();
    const auto square_plan=scivolapi::detail::squared_shocks::prepare(raw.data(),r2.data(),raw.size());
    std::array<double,3> gt{},gz{}; std::array<double,9> ht{},hz{},j{};
    double ft=0,fz=0; unsigned mask=0;
    CHECK(generated::evaluate_prevalidated_sample<Theta>(theta.data(),{raw.data(),r2.data(),2,{},0.0,square_plan},8,0.8,7,
        &ft,gt.data(),ht.data(),&mask)==VLK_JET_OK);
    CHECK(generated::evaluate_prevalidated_sample<Chart>(z.data(),{raw.data(),r2.data(),2,{},0.0,square_plan},8,0.8,7,
        &fz,gz.data(),hz.data(),&mask)==VLK_JET_OK);
    CHECK(close(ft,fz));
    Transform::fill_jacobian(theta.data(),z.data(),j.data());
    for(std::size_t a=0;a<3;++a) {
        double expected=0;
        for(std::size_t i=0;i<3;++i) expected+=j[i*3+a]*gt[i];
        CHECK(close(gz[a],expected));
    }
    // Independent two-logit curvature, including the original-z dominant tail.
    const double r=std::exp(-37),a=1.0-2*r/(1+2*r),b=r/(1+2*r);
    for(std::size_t left=0;left<3;++left) for(std::size_t right=0;right<3;++right) {
        double expected=0;
        for(std::size_t i=0;i<3;++i) for(std::size_t k=0;k<3;++k)
            expected+=j[i*3+left]*ht[i*3+k]*j[k*3+right];
        if(left==0 && right==0) expected+=gt[0]*generated::softplus_second_from_positive(generated::softplus(z[0]));
        if(left && right) {
            const double aa=-a*(2*r/(1+2*r))*(2*a-1),ab=a*b*(2*a-1),abb=-a*b*(1-2*b);
            const double baa=-a*b*(1-2*a),bab=-a*b*(1-2*b),bb=b*(1-b)*(1-2*b);
            const bool diagonal=left==right;
            expected+=gt[1]*(diagonal ? (left==1 ? aa : abb) : ab)
                +gt[2]*(diagonal ? (left==1 ? baa : bb) : bab);
        }
        CHECK(close(hz[left*3+right],expected));
    }
    for(unsigned request=1;request<=7;++request) {
        double f=sentinel; std::array<double,3> g; std::array<double,9> h; unsigned valid=0;
        g.fill(sentinel);h.fill(sentinel);
        CHECK(generated::evaluate_prevalidated_sample<Chart>(z.data(),{raw.data(),r2.data(),2,{},0.0,square_plan},8,0.8,request,
            &f,g.data(),h.data(),&valid,false)==VLK_JET_OK && valid==request);
        if(request&1) CHECK(close(f,fz)); else CHECK(f==sentinel);
        for(std::size_t i=0;i<3;++i) CHECK(request&2 ? close(g[i],gz[i]) : g[i]==sentinel);
        for(std::size_t i=0;i<9;++i) CHECK(request&4 ? close(h[i],hz[i]) : h[i]==sentinel);
    }
    const std::array<double,3> endpoint{-0.8,40,0};
    for(unsigned request=1;request<=7;++request) {
        double f=sentinel; gz.fill(sentinel); hz.fill(sentinel); mask=123;
        CHECK(generated::evaluate_prevalidated_sample<Chart>(endpoint.data(),{raw.data(),r2.data(),2,{},0.0,square_plan},8,0.8,
            request,&f,gz.data(),hz.data(),&mask,false)==VLK_JET_DOMAIN_ERROR);
        CHECK(mask==0 && f==sentinel);
        for(double x:gz) CHECK(x==sentinel);
        for(double x:hz) CHECK(x==sentinel);
    }
    std::array<double,3> zero{0.2,0.1,0},inverse{};
    CHECK(!Transform::unpack(zero.data(),inverse.data()));
    // The supplied z, not a rounded-theta inverse, owns the derivative slope.
    CHECK(Transform::unpack(theta.data(),inverse.data()));
    CHECK(inverse[1]!=z[1]);
}
void offset_maps() {
    const std::array<double,3> z{-720,-2,-1};
    std::array<double,3> theta{};
    Transform::pack(z.data(),theta.data());
    CHECK(theta[0]>0 && theta[0]<std::numeric_limits<double>::min());
    CHECK(theta[0]==generated::softplus(z[0]));
    std::array<double,1> shape{};
    const double student=-35.2;
    generated::StudentTShapeTransform::pack(&student,shape.data());
    CHECK(shape[0]==std::nextafter(2.0,std::numeric_limits<double>::infinity()));
    CHECK(generated::StudentTDensity::valid_shape(shape.data()));
    using StudentTheta=generated::GarchJet<1,1,generated::StudentTDensity>;
    using StudentMap=generated::GarchTransform<1,1,generated::StudentTShapeTransform>;
    using StudentChart=generated::TransformedGarchJet<1,1,generated::StudentTDensity,StudentMap>;
    const std::array<double,4> student_z{-0.8,-2,-1,student};
    std::array<double,4> student_theta{},gradient{}; std::array<double,16> hessian{};
    StudentMap::pack(student_z.data(),student_theta.data());
    CHECK(StudentTheta::parameter_status(student_theta.data())==VLK_JET_OK);
    const auto r2=squares();
    const auto square_plan=scivolapi::detail::squared_shocks::prepare(raw.data(),r2.data(),raw.size()); unsigned valid=0;double value=0;
    CHECK(generated::evaluate_prevalidated_sample<StudentChart>(student_z.data(),{raw.data(),r2.data(),2,{},0.0,square_plan},8,0.8,7,
        &value,gradient.data(),hessian.data(),&valid)==VLK_JET_OK && valid==7);
    const double ged=-300;
    generated::GedShapeTransform::pack(&ged,shape.data());
    CHECK(shape[0]>0 && shape[0]<0x1p-330);
    CHECK(generated::GedDensity::valid_shape(shape.data()));
    using GedTheta=generated::GarchJet<1,1,generated::GedDensity>;
    using GedMap=generated::GarchTransform<1,1,generated::GedShapeTransform>;
    using GedChart=generated::TransformedGarchJet<1,1,generated::GedDensity,GedMap>;
    const std::array<double,4> ged_z{-0.8,-2,-1,ged};
    std::array<double,4> ged_theta{};
    GedMap::pack(ged_z.data(),ged_theta.data());
    CHECK(GedTheta::parameter_status(ged_theta.data())==VLK_JET_OK);
    for(unsigned request=1;request<=7;++request) {
        valid=123;
        const auto status=generated::evaluate_prevalidated_sample<GedChart>(ged_z.data(),{raw.data(),r2.data(),2,{},0.0,square_plan},8,0.8,
            request,&value,gradient.data(),hessian.data(),&valid);
        CHECK(status==VLK_JET_OK || status==VLK_JET_DOMAIN_ERROR);
        CHECK(valid==(status==VLK_JET_OK ? request : 0));
    }
    const double collapsed=-750;
    generated::GedShapeTransform::pack(&collapsed,shape.data());
    CHECK(shape[0]==0);
    const std::array<double,3> omega_collapse{-750,-2,-1};
    generated::SimplexTailPoint point; std::array<double,2> weights;
    Transform::pack(omega_collapse.data(),theta.data(),&point,weights.data());
    CHECK(point.mapped_admission==domain::Admission::numerical_range);
}
void moment_certificates() {
    const double alpha=1.2,gamma=-0.8,beta=0.05;
    const auto admit=[&](const domain::MomentWitness* witness) {
        return domain::gjr_coefficients(1,1,[&](std::size_t){return alpha;},
            [&](std::size_t){return gamma;},[&](std::size_t){return beta;},witness);
    };
    const domain::MomentWitness approximate{0.5,true,false},exact{0.5,true,true};
    CHECK(admit(&approximate)==domain::Admission::uncertified_moment);
    CHECK(domain::admission_status(admit(&approximate),true)==VLK_JET_UNCERTIFIED_DOMAIN);
    CHECK(admit(&exact)==domain::Admission::inside);
    using GjrTransform=generated::GjrTransform<1,1,generated::NormalDensity>;
    using GjrChart=generated::TransformedGarchJet<1,1,generated::NormalDensity,GjrTransform,generated::GjrRecursion<1,1>>;
    const std::array<double,4> z{-0.8,37,0,0}; const auto r2=squares();
    const auto square_plan=scivolapi::detail::squared_shocks::prepare(raw.data(),r2.data(),raw.size());
    std::array<double,4> theta,weights{}; generated::SimplexTailPoint prepared;
    GjrTransform::pack(z.data(),theta.data(),&prepared,weights.data());
    CHECK(prepared.negative_moment.valid && prepared.negative_moment.exact_half);
    const auto status=domain::admission_status(prepared.mapped_admission,true);
    for(unsigned mask=1;mask<=7;++mask) {
        double f=sentinel; std::array<double,4> g{}; std::array<double,16> h{}; unsigned valid=0;
        CHECK(generated::evaluate_prevalidated_sample<GjrChart>(z.data(),{raw.data(),r2.data(),2,{},0.0,square_plan},8,0.8,mask,
            &f,g.data(),h.data(),&valid,false)==status);
        CHECK(status==VLK_JET_OK ? valid==mask : valid==0);
    }
}
void retained_weight_lifetime() {
    const double* dummy=reinterpret_cast<const double*>(std::uintptr_t{64});
    CHECK(!generated::map_storage_overlap(dummy,0,dummy,1));
    CHECK(!generated::map_storage_overlap(dummy,1,dummy,0));
    const std::array<double,3> z{-0.8,37,0};
    std::array<double,3> theta{};std::array<double,9> expected{},actual{};
    generated::SimplexTailPoint scalar;
    Transform::pack(z.data(),theta.data(),&scalar);
    CHECK(scalar.shifted_weights==nullptr && scalar.mapped_admission==domain::Admission::inside);
    Transform::fill_jacobian(theta.data(),z.data(),expected.data());
    Transform::fill_jacobian_prepared(theta.data(),z.data(),actual.data(),scalar);
    CHECK(expected==actual);
    std::array<double,2> weights{};
    Transform::pack(z.data(),theta.data(),&scalar,weights.data());
    CHECK(scalar.shifted_weights==weights.data());
    Transform::pack(z.data(),theta.data(),&scalar,theta.data()+1);
    CHECK(scalar.shifted_weights==nullptr);
    using Gjr=generated::GjrTransform<1,1,generated::NormalDensity>;
    const std::array<double,4> gjr_z{-0.8,-2,-3,-1};
    std::array<double,4> gjr_theta{};std::array<double,16> gjr_expected{},gjr_actual{};
    Gjr::pack(gjr_z.data(),gjr_theta.data(),&scalar);
    CHECK(scalar.shifted_weights==nullptr && scalar.mapped_admission==domain::Admission::inside);
    Gjr::fill_jacobian(gjr_theta.data(),gjr_z.data(),gjr_expected.data());
    Gjr::fill_jacobian_prepared(gjr_theta.data(),gjr_z.data(),gjr_actual.data(),scalar);
    CHECK(gjr_expected==gjr_actual);
    std::array<double,3> gjr_weights{};
    Gjr::pack(gjr_z.data(),gjr_theta.data(),&scalar,gjr_weights.data());
    CHECK(scalar.shifted_weights==gjr_weights.data());
    Gjr::pack(gjr_z.data(),gjr_theta.data(),&scalar,gjr_theta.data()+1);
    CHECK(scalar.shifted_weights==nullptr);
    Gjr::pack(gjr_z.data(),gjr_theta.data(),&scalar,gjr_theta.data()+2);
    CHECK(scalar.shifted_weights==nullptr && scalar.mapped_admission==domain::Admission::numerical_range);
    CHECK(std::isnan(gjr_theta[0]));
}
} // namespace

struct CountingDensity : generated::NormalDensity {};
std::size_t moment_preparations=0;
namespace vlk::experiment::generated {
template<> struct DensityShapeTransform<CountingDensity> { using type=NoShapeTransform; };
template<> inline NegativeSecondMoment<0> negative_second_moment<0,CountingDensity>(const double*) noexcept {
    ++moment_preparations; NegativeSecondMoment<0> result; result.value=0.5; result.valid=true; return result;
}
}
void one_moment_preparation() {
    using Transform=generated::GjrTransform<1,1,CountingDensity>;
    using Jet=generated::TransformedGarchJet<1,1,CountingDensity,Transform,generated::GjrRecursion<1,1>>;
    const std::array<double,4> z{-0.8,0,0,0}; const auto r2=squares();
    const auto square_plan=scivolapi::detail::squared_shocks::prepare(raw.data(),r2.data(),raw.size());
    for(bool parameters_validated:{false,true}) {
        const auto before=moment_preparations; unsigned mask=0; double value=0;
        CHECK(generated::evaluate_prevalidated_sample<Jet>(z.data(),{raw.data(),r2.data(),2,{},0.0,square_plan},8,0.8,1,
            &value,nullptr,nullptr,&mask,!parameters_validated)==VLK_JET_OK);
        CHECK(mask==1 && moment_preparations-before==1);
    }
}
#if defined(SCIVOLAPI_LINKED_ENDPOINT_TEST)
void linked_entries() {
    using namespace scivolapi;
    namespace backend=scivolapi::detail::backend;
    namespace joint=scivolapi::detail::joint_backend;
    const auto r2=squares();
    const auto square_plan=scivolapi::detail::squared_shocks::prepare(raw.data(),r2.data(),raw.size());
    for(std::size_t p:{1u,3u}) for(unsigned request=1;request<=7;++request) {
        const std::size_t q=p==1 ? 1 : 2,width=1+p+q;
        std::vector<double> theta(width,0),g(width,sentinel),h(width*width,sentinel);
        theta[0]=0.2; theta[1]=0x1.fffffffffffffp-1; theta[1+p]=0x1p-54;
        const backend::ModelConfig config{p,q,0,false};
        CHECK(backend::parameter_status(config,theta.data())==Status::ok);
        const auto bytes=vlk_runtime_garch_sample_jet_workspace_bytes(p,q,0,0,request);
        std::vector<double> memory((bytes+7)/8);
        double f=sentinel;unsigned valid=0;
        CHECK(vlk_garch_selected_sample_jet_evaluate(false,true,theta.data(),raw.data(),r2.data(),8,2,0.8,
            p,q,0,0,request,memory.data(),bytes,&f,g.data(),h.data(),&valid)==VLK_JET_OK && valid==request);
        if(p==1) {
            std::array<double,3> z{-0.8,40,0};
            const auto transformed_bytes=vlk_runtime_garch_sample_jet_workspace_bytes(1,1,0,1,request);
            memory.resize((transformed_bytes+7)/8);
            CHECK(vlk_garch_selected_sample_jet_evaluate(false,true,z.data(),raw.data(),r2.data(),8,2,0.8,
                1,1,0,1,request,memory.data(),transformed_bytes,&f,g.data(),h.data(),&valid)==VLK_JET_DOMAIN_ERROR && valid==0);
        }
    }
    for(bool leverage:{false,true}) {
        joint::ModelConfig config{joint::Kind::arma,1,1,0,1,1,0,leverage,true,true};
        const auto width=joint::parameter_count(config);
        std::vector<double> z(width,0); z[0]=0.2; z[3]=-0.8;
        const auto map_need=joint::map_requirements(config,true);
        std::vector<double> scratch((map_need.bytes+7)/8),theta(width),jacobian(width*width);
        auto work=std::as_writable_bytes(std::span<double>(scratch)).first(map_need.bytes);
        CHECK(joint::map(config,z,theta,work,jacobian)==Status::ok);
        joint::SampleView sample{raw,{},1,3,0.8};
        for(unsigned request=1;request<=7;++request) {
            const auto need=joint::requirements(config,Coordinates::original_chart,static_cast<JetRequest>(request));
            scratch.resize((need.bytes+7)/8); work=std::as_writable_bytes(std::span<double>(scratch)).first(need.bytes);
            double f=sentinel;std::vector<double> g(width,sentinel),h(width*width,sentinel);
            const auto result=joint::evaluate(config,sample,z,Coordinates::original_chart,
                static_cast<JetRequest>(request),work,{{&f,1},g,h});
            CHECK(result.status==Status::ok && result.valid_mask==request);
        }
        z[4]=40; // Original-z admission happens before the recurrence on both map paths.
        scratch.resize((map_need.bytes+7)/8); work=std::as_writable_bytes(std::span<double>(scratch)).first(map_need.bytes);
        const auto status=joint::map(config,z,theta,work,jacobian);
        CHECK(status==Status::numerical_range || status==Status::unsupported_operation);
    }
}
void shared_shape_consumers() {
    using namespace scivolapi;
    namespace mean=scivolapi::detail::mean_backend;
    namespace power=scivolapi::detail::power_backend;
    for(unsigned density:{1u,2u,3u}) {
        const double shape_z=density==3 ? -300.0 : -35.2;
        const double expected=density==3 ? generated::softplus(shape_z)
            : 2.0+generated::softplus(shape_z);
        const mean::ModelConfig config{mean::Kind::linear,0,0,0,density,true,true};
        const auto need=mean::map_requirements(config);
        std::vector<double> z(need.parameters,0),theta(need.parameters),memory((need.bytes+7)/8);
        z[1]=-720;z[2]=shape_z;
        const auto workspace=std::as_writable_bytes(std::span<double>(memory)).first(need.bytes);
        CHECK(mean::map(config,z,theta,workspace)==Status::ok);
        CHECK(theta[1]==generated::softplus(-720.0) && theta[2]==expected);
        for(auto kind:{power::Kind::egarch,power::Kind::aparch}) {
            const power::ModelConfig config{kind,1,1,density,true};
            const auto need=power::map_requirements(config);
            const auto shape=need.parameters-(density==2 ? 2u : 1u);
            z.assign(need.parameters,0);theta.resize(need.parameters);memory.resize((need.bytes+7)/8);
            z[shape]=shape_z;
            const auto workspace=std::as_writable_bytes(std::span<double>(memory)).first(need.bytes);
            const auto status=power::map(config,z,theta,workspace);
            CHECK(status==Status::ok || status==Status::numerical_range);
            if(status==Status::ok) CHECK(theta[shape]==expected);
        }
    }
}

void checked_student_shape_maps() {
    using namespace scivolapi;
    namespace backend=scivolapi::detail::backend;
    namespace joint=scivolapi::detail::joint_backend;
    for(unsigned density:{1u,2u}) for(std::size_t p:{1u,3u}) for(bool leverage:{false,true}) {
        const std::size_t q=p==1 ? 1 : 2,shape_count=density==1 ? 1 : 2;
        const backend::ModelConfig config{p,q,density,leverage};
        const auto width=backend::parameter_count(config),nu=width-shape_count;
        std::vector<double> z(width,-2),theta(width);
        z[0]=-0.7;z[nu]=5;
        if(density==2) z[nu+1]=0;
        CHECK(backend::map(config,z.data(),theta.data())==Status::ok);
        theta[nu]=2;
        CHECK(backend::parameter_status(config,theta.data())==Status::mathematical_domain);
        for(double endpoint:{-40.,-41.}) {
            z[nu]=endpoint;
            CHECK(backend::map(config,z.data(),theta.data())==Status::numerical_range);
        }
        for(std::size_t design:{0u,3u,6u}) for(bool derivatives:{false,true}) {
            const bool linear=design!=0;
            joint::ModelConfig c{linear ? joint::Kind::linear : joint::Kind::arma,
                linear ? 0u : 1u,linear ? 0u : 1u,design,p,q,density,leverage,true,true};
            const auto need=joint::map_requirements(c,derivatives);
            CHECK(need.status==Status::ok && need.dispatch==(p==1 ? 1 : 0));
            std::vector<double> coordinates(need.parameters,-2),mapped(need.parameters),
                jacobian(derivatives ? need.parameters*need.parameters : 0),memory((need.bytes+7)/8);
            const auto index=need.parameters-shape_count;
            std::fill_n(coordinates.begin(),linear ? design+1 : 3,0.05);
            coordinates[index]=5;
            if(density==2) coordinates[index+1]=0;
            auto work=std::as_writable_bytes(std::span<double>(memory)).first(need.bytes);
            CHECK(joint::map(c,coordinates,mapped,work,jacobian)==Status::ok);
            for(double endpoint:{-40.,-41.}) {
                coordinates[index]=endpoint;
                CHECK(joint::map(c,coordinates,mapped,work,jacobian)==Status::numerical_range);
            }
            if(density==2) for(double endpoint:{-40.,40.}) {
                coordinates[index]=5;coordinates[index+1]=endpoint;
                CHECK(joint::map(c,coordinates,mapped,work,jacobian)==Status::numerical_range);
            }
        }
    }
}
#endif
int main() {
    exact_theta(); tail_map(); offset_maps(); moment_certificates(); retained_weight_lifetime(); one_moment_preparation();
#if defined(SCIVOLAPI_LINKED_ENDPOINT_TEST)
    linked_entries(); shared_shape_consumers(); checked_student_shape_maps();
#endif
    std::printf("garch chart endpoint prototype: %zu checks\n",checks);
}
