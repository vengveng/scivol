// SCIVOL-NATIVE: TEST_ONLY; high-precision Python reference consumes these jets.
#include "math/student_normalization.hpp"
#include "models/power/egarch_moment_policy.hpp"
#include "models/joint_power/egarch_jet.hpp"
#include "models/gjr_transform_policy.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

template<int Order>
void emit_power_moment(double power,double nu)
{
    using namespace vlk::experiment;
    using N=ad::Differential<Order,2>;
    const auto value=ad::exp(generated::student_log_absolute_power(
        N::variable(power,0),N::variable(nu,1),N(.5*std::log(2.)),N(.5*std::log(M_PI))));
    assert(ad::finite(value));
    std::printf("%.17g ",value.value);
    for (unsigned i=0;i<2;++i) {
        if constexpr (Order>=1) std::printf("%.17g ",value.gradient[i]);
        else std::printf("0 ");
    }
    for (unsigned i=0;i<3;++i) {
        if constexpr (Order>=2) std::printf("%.17g ",value.hessian[i]);
        else std::printf("0 ");
    }
}

template<class Orders, class Density=vlk::experiment::generated::StudentTDensity>
void emit_joint(double nu)
{
    using namespace vlk::experiment;
    namespace jet=runtime::joint_egarch;
    const double y[]{0.5,-0.8,1.2,-0.3,0.7,-1.1,0.2,0.9};
    const double x[]{1.,0.2,1.,-0.3,1.,0.5,1.,-0.1,1.,0.4,1.,-0.5,1.,0.3,1.,-0.2};
    const double theta[]{0.1,-0.04,-0.1,0.08,0.04,0.02,-0.05,0.01,0.03,0.3,0.2,nu,0.08728423370553595};
    constexpr auto width=11+Density::shape_parameters;
    jet::Layout layout;
    assert((runtime::joint_power::make_layout<Density,false>(0,0,2,3,2,true,false,layout)));
    assert(layout.parameters==width);
    const auto run=[&]<unsigned Mask>() {
        constexpr int order=(Mask&4)?2:(Mask&2)?1:0;
        const auto bytes=jet::core_workspace_bytes<order,double,Orders,Density>(layout);
        std::vector<unsigned char> work(bytes);
        double f=NAN;
        std::array<double,width> g;
        std::array<double,width*width> h;
        g.fill(NAN); h.fill(NAN);
        jet::SamplePlan sample{0,0,{}};
        sample.history.implicit_log_variance=0.0;
        const bool ok=jet::evaluate_theta_order<order,bool(Mask&1),bool(Mask&2),bool(Mask&4),Orders,Density>(
            layout,theta,y,8,1.1,sample,work.data(),work.size(),
            Mask&1?&f:nullptr,Mask&2?g.data():nullptr,Mask&4?h.data():nullptr,x);
        assert(ok);
        std::printf("%u %.17g ",Mask,f);
        for (double value:g) std::printf("%.17g ",value);
        for (double value:h) std::printf("%.17g ",value);
        std::printf("\n");
    };
    run.template operator()<1>(); run.template operator()<2>();
    run.template operator()<3>(); run.template operator()<4>();
    run.template operator()<5>(); run.template operator()<6>(); run.template operator()<7>();
}

void emit_skew_density(char** argv)
{
    using namespace vlk::experiment::generated;
    const double shape[]{std::strtod(argv[2],nullptr),std::strtod(argv[3],nullptr)};
    const double z=std::strtod(argv[4],nullptr);
    const auto prepared=SkewTDensity::prepare<true,true,true,true>(shape,1);
    const auto point=SkewTDensity::observe<true,true,true,true>(prepared,1,z,z*z);
    const auto moment=absolute_moment<2,SkewTDensity>(shape);
    const auto tail=student_tail<false,true,true>(shape[0],z*z,1.0/(shape[0]-2.0),
        std::log1p(z*z/(shape[0]-2.0)));
    std::printf("%.17g %.17g %.17g %.17g %.17g %.17g\n",
        prepared.observation_constant.value+point.value,
        prepared.observation_constant.gradient[0]+point.shape_gradient[0],
        prepared.observation_constant.hessian[0]+point.shape_hessian[0],
        moment.value,moment.gradient[0],moment.hessian[0]);
    std::printf("%.17g %.17g %.17g %.17g %.17g\n",2*z*tail.squared,tail.shape,
        2*tail.squared+4*z*z*tail.squared_squared,2*z*tail.squared_shape,tail.shape_shape);
    const double g[]{point.residual_gradient,point.variance_gradient,
        prepared.observation_constant.gradient[0]+point.shape_gradient[0],
        prepared.observation_constant.gradient[1]+point.shape_gradient[1]};
    const double h[]{point.residual_hessian,point.residual_variance_hessian,
        point.residual_shape_hessian[0],point.residual_shape_hessian[1],
        point.residual_variance_hessian,point.variance_hessian,
        point.variance_shape_hessian[0],point.variance_shape_hessian[1],
        point.residual_shape_hessian[0],point.variance_shape_hessian[0],
        prepared.observation_constant.hessian[0]+point.shape_hessian[0],
        prepared.observation_constant.hessian[1]+point.shape_hessian[1],
        point.residual_shape_hessian[1],point.variance_shape_hessian[1],
        prepared.observation_constant.hessian[1]+point.shape_hessian[1],
        prepared.observation_constant.hessian[2]+point.shape_hessian[2]};
    for (double value:g) std::printf("%.17g ",value);
    for (double value:h) std::printf("%.17g ",value);
    std::printf("\n");
}

template<bool F,bool G,bool H,class Full>
void check_student_request(double variance,double residual,const double* shape,const Full& full) {
    using D=vlk::experiment::generated::StudentTDensity;
    const auto prepared=D::prepare<F,G,H,true>(shape,1);
    const auto part=D::observe<F,G,H,true>(prepared,variance,residual,residual*residual);
    const auto close=[](double a,double b) {
        assert(std::isfinite(a) && std::isfinite(b));
        assert(std::abs(a-b)<=4e-14*std::abs(b)+2*std::numeric_limits<double>::denorm_min());
    };
    if constexpr(G) {
        close(part.residual_gradient,full.residual_gradient);
        close(part.variance_gradient,full.variance_gradient);
        close(part.shape_gradient[0],full.shape_gradient[0]);
    }
    if constexpr(H) {
        close(part.residual_hessian,full.residual_hessian);
        close(part.residual_variance_hessian,full.residual_variance_hessian);
        close(part.variance_hessian,full.variance_hessian);
        close(part.residual_shape_hessian[0],full.residual_shape_hessian[0]);
        close(part.variance_shape_hessian[0],full.variance_shape_hessian[0]);
        close(part.shape_hessian[0],full.shape_hessian[0]);
    }
    if constexpr(F) {
        auto a=D::initial_value(prepared),b=D::initial_value(prepared);
        D::accumulate_value(a,part);D::accumulate_value(b,full);
        close(D::finalize_value(a,prepared),D::finalize_value(b,prepared));
    }
}

int main(int argc, char** argv)
{
    if (argc>2 && std::strcmp(argv[1],"--normal-tail")==0) {
        using namespace vlk::experiment;
        for (int i=2;i<argc;++i) {
            const double t=std::strtod(argv[i],nullptr);
            const auto value=generated::density_moment::normal_negative_second_tail(
                ad::Differential<0,1>::constant(t));
            const auto gradient=generated::density_moment::normal_negative_second_tail(
                ad::Differential<1,1>::variable(t,0));
            const auto hessian=generated::density_moment::normal_negative_second_tail(
                ad::Differential<2,1>::variable(t,0));
            assert(value.value==gradient.value && value.value==hessian.value);
            assert(gradient.gradient[0]==hessian.gradient[0]);
            std::printf("%.17g %.17g %.17g\n",value.value,gradient.gradient[0],hessian.hessian[0]);
        }
        return 0;
    }
    if (argc==4 && std::strcmp(argv[1],"--tail-cancellation")==0) {
        const double nu=std::strtod(argv[2],nullptr),square=std::strtod(argv[3],nullptr);
        const auto jet=vlk::experiment::generated::student_tail<true,true,true>(
            nu,square,1./(nu-2.),std::log1p(square/(nu-2.)));
        std::printf("%.17g %.17g %.17g %.17g %.17g\n",jet.value,jet.shape,jet.shape_shape,
            vlk_student_tail_shape_first(nu,square),vlk_student_tail_shape_second(nu,square));
        return 0;
    }
    if (argc==5 && std::strcmp(argv[1],"--student-range")==0) {
        using D=vlk::experiment::generated::StudentTDensity;
        const double h=std::strtod(argv[2],nullptr), e=std::strtod(argv[3],nullptr);
        const double shape[]{std::strtod(argv[4],nullptr)};
        const auto prepared=D::prepare<true,true,true,true>(shape,1);
        const auto point=D::observe<true,true,true,true>(prepared,h,e,e*e);
        check_student_request<true,false,false>(h,e,shape,point);
        check_student_request<false,true,false>(h,e,shape,point);
        check_student_request<false,false,true>(h,e,shape,point);
        check_student_request<true,true,false>(h,e,shape,point);
        check_student_request<true,false,true>(h,e,shape,point);
        check_student_request<false,true,true>(h,e,shape,point);
        auto sum=D::initial_value(prepared);
        D::accumulate_value(sum,point);
        std::printf("%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
            D::finalize_value(sum,prepared), point.residual_gradient,point.variance_gradient,
            point.shape_gradient[0],point.residual_hessian,point.residual_variance_hessian,
            point.residual_shape_hessian[0],point.variance_hessian,
            point.variance_shape_hessian[0],point.shape_hessian[0]);
        return 0;
    }
    if (argc>3 && std::strcmp(argv[1],"--power")==0) {
        const double power=std::strtod(argv[2],nullptr);
        for (int i=3;i<argc;++i) {
            const double nu=std::strtod(argv[i],nullptr);
            emit_power_moment<0>(power,nu);
            emit_power_moment<1>(power,nu);
            emit_power_moment<2>(power,nu);
            std::printf("\n");
        }
        return 0;
    }
    if (argc==5 && std::strcmp(argv[1],"--skew")==0) {
        emit_skew_density(argv);
        return 0;
    }
    if (argc>1 && std::strcmp(argv[1],"--skew-joint")==0) {
        for (int i=2;i<argc;++i) {
            const double nu=std::strtod(argv[i],nullptr);
            using namespace vlk::experiment;
            emit_joint<runtime::joint_power::Orders<true,false,2,0,0,3,2>,generated::SkewTDensity>(nu);
            emit_joint<runtime::joint_power::Orders<true,false>,generated::SkewTDensity>(nu);
        }
        return 0;
    }
    if (argc>1 && std::strcmp(argv[1],"--joint")==0) {
        for (int i=2;i<argc;++i) {
            const double nu=std::strtod(argv[i],nullptr);
            emit_joint<vlk::experiment::runtime::joint_power::Orders<true,false,2,0,0,3,2>>(nu);
            emit_joint<vlk::experiment::runtime::joint_power::Orders<true,false>>(nu);
        }
        return 0;
    }
    const bool moment=argc>1 && std::strcmp(argv[1],"--moment")==0;
    if (moment) {
        for (double nu: std::array<double,4>{0.,2.,INFINITY,NAN}) {
            double f=31.,g=31.,h=31.;
            assert(!vlk_student_abs_moment(nu,&f,&g,&h));
            assert(f==31. && g==31. && h==31.);
        }
        const double nu=std::numeric_limits<double>::max();
        double f,g,h;
        assert(vlk_student_abs_moment(nu,&f,&g,&h));
        assert(std::abs(f-std::sqrt(2/M_PI))<4*std::numeric_limits<double>::epsilon());
        assert(g==0 && h==0);
    }
    for (int i=moment?2:1; i<argc; ++i) {
        const double nu=std::strtod(argv[i],nullptr);
        if (moment) {
            const auto emit=[&]<int Order>() {
                const auto jet=vlk::experiment::generated::absolute_moment<
                    Order,vlk::experiment::generated::StudentTDensity>(&nu);
                assert(jet.valid);
                std::printf("%.17g %.17g %.17g ",jet.value,jet.gradient[0],jet.hessian[0]);
            };
            emit.template operator()<0>();
            emit.template operator()<1>();
            emit.template operator()<2>();
            double f,h;
            assert(vlk_student_abs_moment(nu,&f,nullptr,&h));
            std::printf("%.17g %.17g\n",f,h);
            continue;
        }
        const auto jet=vlk::experiment::generated::student_normalization<true,true,true>(nu);
        std::printf("%.17g %.17g %.17g\n",jet.value,jet.first,jet.second);
    }
}
