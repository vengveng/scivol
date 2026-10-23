#include "models/gjr_transform_policy.hpp"
#include "models/gjr_runtime_map.hpp"
#include "models/garch_generated_jet.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace generated=vlk::experiment::generated;
namespace domain=vlk::experiment::domain;
namespace owner=vlk::experiment::runtime::gjr_coordinate;
namespace {
std::size_t checks=0;
void check(bool good,int line) {
    ++checks;
    if(!good) {std::fprintf(stderr,"GJR half identity line %d\n",line); std::exit(1);}
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
constexpr double tiny=std::numeric_limits<double>::denorm_min();
static_assert(std::bit_cast<std::uint64_t>(tiny)==1);
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits==53);

template<std::size_t P,std::size_t Q>
void fixed_theta_admission() {
    using Density=generated::NormalDensity;
    using Recursion=generated::GjrRecursion<P,Q>;
    using Jet=generated::GarchJet<P,Q,Density,Recursion>;
    std::array<double,Jet::K> theta{};
    theta[0]=0.125; theta[Recursion::alpha_start]=0.25;
    theta[Recursion::gamma_start]=0.5; theta[Recursion::beta_start]=0.375;
    CHECK(Jet::parameter_status(theta.data())==VLK_JET_OK);
    CHECK(Recursion::template valid_parameters<Density>(theta.data()));
    theta.fill(0.); theta[0]=0.125; theta[Recursion::alpha_start]=1.;
    for(double gamma : {-tiny,0.,tiny}) {
        theta[Recursion::gamma_start]=gamma;
        const bool inside=gamma<0.;
        CHECK(Jet::parameter_status(theta.data())
            ==(inside ? VLK_JET_OK : VLK_JET_MATHEMATICAL_DOMAIN_ERROR));
        CHECK(Recursion::template valid_parameters<Density>(theta.data())==inside);
    }
}

template<int Order>
void unconditional_half_values(const double* student,const double* ged) {
    const auto normal=generated::negative_second_moment<Order,generated::NormalDensity>(nullptr);
    const auto t=generated::negative_second_moment<Order,generated::StudentTDensity>(student);
    const auto ged_moment=generated::negative_second_moment<Order,generated::GedDensity>(ged);
    CHECK(normal.valid && t.valid && ged_moment.valid);
    CHECK(normal.value==0.5 && t.value==normal.value && ged_moment.value==normal.value);
}

template<class Density,int Order>
void print_moment(const double* shape) {
    const auto moment=generated::negative_second_moment<Order,Density>(shape);
    std::printf("%d %d %.17g",Order,int(moment.valid),moment.value);
    for(double value : moment.gradient) std::printf(" %.17g",value);
    for(double value : moment.hessian) std::printf(" %.17g",value);
    std::putchar('\n');
}

template<class Density>
int print_moment_orders(int argc,char** argv) {
    constexpr std::size_t D=Density::shape_parameters;
    if(argc!=3+int(D)) return 2;
    std::array<double,D> shape;
    for(std::size_t i=0;i<D;++i) {
        char* end=nullptr;
        shape[i]=std::strtod(argv[3+i],&end);
        if(end==argv[3+i] || *end!='\0' || !std::isfinite(shape[i])) return 2;
    }
    print_moment<Density,0>(shape.data());
    print_moment<Density,1>(shape.data());
    print_moment<Density,2>(shape.data());
    return 0;
}

template<class Density>
void identity_boundary(std::array<double,Density::shape_parameters> shape,std::size_t skew) {
    constexpr auto D=Density::shape_parameters;
    for(double value : {0.0,-0.0,tiny,-tiny}) {
        shape[skew]=value;
        const bool exact=value==0.0;
        CHECK(Density::valid_shape(shape.data()));
        CHECK(generated::exact_half_moment_at<Density>(shape.data())==exact);
        generated::NegativeSecondMoment<D> rounded;
        rounded.valid=true;
        rounded.value=0.5; // Numerically equal is not an identity witness.
        const auto witness=generated::negative_second_moment_witness<Density>(shape.data(),rounded);
        CHECK(witness.valid && witness.value==0.5 && witness.exact_half==exact);
        std::array<double,4+D> theta{};
        theta[0]=0.1; theta[1]=1.0; theta[2]=-tiny;
        std::copy(shape.begin(),shape.end(),theta.begin()+4);
        const auto expected=exact ? domain::Admission::inside : domain::Admission::uncertified_moment;
        const domain::MomentWitness stale{0.5,true,true};
        for(const auto* prepared : {static_cast<const domain::MomentWitness*>(nullptr),&stale}) {
            CHECK(generated::gjr_coefficient_admission<Density>(theta.data(),
                std::size_t{1},std::size_t{1},std::size_t{1},std::size_t{2},std::size_t{3},
                theta.data()+4,prepared)==expected);
        }
        if(exact) {
            const auto moment=generated::negative_second_moment<0,Density>(shape.data());
            CHECK(moment.valid && moment.value==0.5);
        }
    }
    shape[skew]=std::numeric_limits<double>::quiet_NaN();
    CHECK(!generated::exact_half_moment_at<Density>(shape.data()));
    CHECK(!generated::negative_second_moment<0,Density>(shape.data()).valid);
}

template<class Density,int Order>
void derivative_preservation(const double* shape) {
    const auto raw=[&] {
        if constexpr(std::is_same_v<Density,generated::SkewTDensity>)
            return generated::skew_negative_moment::evaluate<Order>(shape);
        else if constexpr(std::is_same_v<Density,generated::VarianceGammaDensity>)
            return generated::density_moment::variance_gamma_mixture<
                Order,generated::density_moment::NormalMoment::negative_second>(shape);
        else return generated::gjr_moment_detail::integrate<Order,Density>(shape);
    }();
    const auto anchored=generated::negative_second_moment<Order,Density>(shape);
    CHECK(raw.valid && anchored.valid && anchored.value==0.5);
    CHECK(raw.gradient==anchored.gradient);
    CHECK(raw.hessian==anchored.hessian);
}

template<class Density>
void map_witness(const double* shape) {
    using Transform=generated::GjrTransform<1,1,Density>;
    using Shape=generated::DensityShapeTransformT<Density>;
    std::array<double,Transform::parameters> z{},theta{},runtime_theta{};
    std::array<double,3> weights{},runtime_weights{};
    CHECK(Shape::unpack(shape,z.data()+4));
    generated::SimplexTailPoint fixed,runtime;
    Transform::pack(z.data(),theta.data(),&fixed,weights.data());
    const auto layout=owner::map_layout<Density>(std::size_t{1},std::size_t{1});
    CHECK(owner::pack_gjr_coordinates<Density,Shape>(layout,z.data(),runtime_theta.data(),
        &runtime,runtime_weights.data()));
    CHECK(fixed.negative_moment.valid && fixed.negative_moment.exact_half);
    CHECK(runtime.negative_moment.valid && runtime.negative_moment.exact_half);
    CHECK(fixed.negative_moment.value==0.5 && runtime.negative_moment.value==0.5);
    CHECK(fixed.mapped_admission==domain::Admission::inside);
    CHECK(runtime.mapped_admission==domain::Admission::inside);
    CHECK(theta==runtime_theta);
}

template<class Density>
void conditional_law(std::array<double,Density::shape_parameters> shape,std::size_t skew) {
    identity_boundary<Density>(shape,skew);
    derivative_preservation<Density,1>(shape.data());
    derivative_preservation<Density,2>(shape.data());
    map_witness<Density>(shape.data());
}
}
int main(int argc,char** argv) {
    if(argc>=3 && std::strcmp(argv[1],"--moments")==0) {
        if(std::strcmp(argv[2],"skewt")==0) return print_moment_orders<generated::SkewTDensity>(argc,argv);
        if(std::strcmp(argv[2],"gh")==0) return print_moment_orders<generated::GeneralizedHyperbolicDensity>(argc,argv);
        if(std::strcmp(argv[2],"nig")==0) return print_moment_orders<generated::NormalInverseGaussianDensity>(argc,argv);
        if(std::strcmp(argv[2],"hyperbolic")==0) return print_moment_orders<generated::HyperbolicDensity>(argc,argv);
        if(std::strcmp(argv[2],"variance_gamma")==0) return print_moment_orders<generated::VarianceGammaDensity>(argc,argv);
        return 2;
    }
    if(argc!=1) return 2;
    fixed_theta_admission<1,1>();
    fixed_theta_admission<3,2>();
    const double student[]{8.0},ged[]{1.5};
    unconditional_half_values<0>(student,ged);
    unconditional_half_values<1>(student,ged);
    unconditional_half_values<2>(student,ged);
    CHECK(generated::exact_half_moment_at<generated::NormalDensity>(nullptr));
    CHECK(generated::exact_half_moment_at<generated::StudentTDensity>(student));
    CHECK(generated::exact_half_moment_at<generated::GedDensity>(ged));
    const double invalid_student[]{2.0};
    CHECK(!generated::exact_half_moment_at<generated::StudentTDensity>(invalid_student));
    conditional_law<generated::SkewTDensity>({8.0,0.0},1);
    conditional_law<generated::GeneralizedHyperbolicDensity>({0.0,1.5,0.7},0);
    conditional_law<generated::NormalInverseGaussianDensity>({0.0,1.5},0);
    conditional_law<generated::HyperbolicDensity>({0.0,1.5},0);
    conditional_law<generated::VarianceGammaDensity>({0.0,2.3},0);
    const double skew[]{8.0,0.0};
    const auto moment=generated::negative_second_moment<2,generated::SkewTDensity>(skew);
    // At nu=8, the positive Student first moment squared is 75/512.
    // Hence d m / d lambda = 8*(75/512)-3/2 = -21/64, not zero.
    CHECK(std::abs(moment.gradient[1]+21.0/64.0)<2e-14);
    CHECK(moment.hessian[vlk::experiment::ad::packed_index<2>(0,1)]!=0.0);
    std::printf("GJR half identity: %zu checks\n",checks);
}
