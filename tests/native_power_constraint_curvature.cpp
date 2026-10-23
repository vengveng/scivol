// TEST_ONLY: isolated shared-header bridge, without the production extension.
#include <array>
#include <cstddef>

// Only the pytest fixture's copied moment header refers to these counters.
namespace aparch_test {
inline std::array<std::size_t,3> common_preparations{};
}

#include "linear_power_program.hpp"
#include "power_constraint_curvature.hpp"

#include <cstdlib>
#include <vector>

static std::size_t allocations=0;
void* operator new(std::size_t bytes) {
    ++allocations;
    if (void* memory=std::malloc(bytes?bytes:1u)) return memory;
    std::abort();
}
void* operator new[](std::size_t bytes) {return ::operator new(bytes);}
void operator delete(void* memory) noexcept {std::free(memory);}
void operator delete[](void* memory) noexcept {std::free(memory);}
void operator delete(void* memory,std::size_t) noexcept {std::free(memory);}
void operator delete[](void* memory,std::size_t) noexcept {std::free(memory);}

using namespace vlk::experiment;

struct CurvatureTestConfig {
    std::size_t density,aparch,transformed,columns,p,q;
    std::size_t first_order,first_parameters,first_offset;
    std::size_t second_order,second_parameters,second_offset;
    const double* first_basis;
    const double* second_basis;
    double first_scale,second_scale;
    std::size_t fixed;
};

template<class Density,bool Aparch,auto StaticLayout=nullptr,class Consumer>
auto with_curvature(const CurvatureTestConfig& config,Consumer&& consume)
{
    using Program=runtime::LinearPowerProgram<Density,Aparch>;
    Program program;
    program.prepare(config.columns,config.p,config.q);
    const auto parameters=[&] {
        if constexpr (StaticLayout==nullptr) return program.layout.parameters;
        else return StaticLayout->parameters;
    }();
    if constexpr (StaticLayout!=nullptr) {
        using Constraints=runtime::PowerConstraints<Program,StaticLayout>;
        using Width=decltype(std::declval<const Constraints&>()
            .template field<&Program::Layout::parameters>());
        static_assert(Width::value==StaticLayout->parameters);
        if (config.fixed==2u) program.layout={};
    }
    mean::MeanConstraintPlan plan{
        {{config.first_order,config.first_parameters,config.first_basis,config.first_scale},
         config.first_offset},
        {{config.second_order,config.second_parameters,config.second_basis,config.second_scale},
         config.second_offset}};
    mean::PolynomialConstraints polynomial{parameters,plan,nullptr,nullptr,0};
    runtime::PowerConstraints<Program,StaticLayout> constraints{
        program,polynomial,config.transformed!=0,nullptr,nullptr};
    runtime::PowerConstraintCurvature<Program,StaticLayout> curvature{constraints};
    return consume(curvature);
}

template<class Density,class Orders,class Consumer>
auto with_fixed_curvature(const CurvatureTestConfig& config,Consumer&& consume)
{
    if (config.aparch)
        return with_curvature<Density,true,
            runtime::LinearPowerProgram<Density,true>::template fixed_layout<Orders>>(
                config,consume);
    return with_curvature<Density,false,
        runtime::LinearPowerProgram<Density,false>::template fixed_layout<Orders>>(
            config,consume);
}

template<class Consumer>
auto dispatch(const CurvatureTestConfig& config,Consumer&& consume)
{
    const auto run=[&]<class Density>() {
        if (config.fixed) {
            if (config.columns==4u && config.p==1u && config.q==1u)
                return with_fixed_curvature<Density,mean::LinearVolatilityOrders<4,1,1>>(
                    config,consume);
            if (config.columns==4u && config.p==3u && config.q==2u)
                return with_fixed_curvature<Density,mean::LinearVolatilityOrders<4,3,2>>(
                    config,consume);
            if (config.columns==7u && config.p==3u && config.q==2u)
                return with_fixed_curvature<Density,mean::LinearVolatilityOrders<7,3,2>>(
                    config,consume);
            return decltype(with_curvature<Density,true>(config,consume)){};
        }
        return config.aparch ? with_curvature<Density,true>(config,consume)
                             : with_curvature<Density,false>(config,consume);
    };
    switch (config.density) {
    case 0: return run.template operator()<generated::NormalDensity>();
    case 1: return run.template operator()<generated::StudentTDensity>();
    case 2: return run.template operator()<generated::SkewTDensity>();
    case 3: return run.template operator()<generated::GedDensity>();
    case 4: return run.template operator()<generated::GeneralizedHyperbolicDensity>();
    case 5: return run.template operator()<generated::NormalInverseGaussianDensity>();
    case 6: return run.template operator()<generated::HyperbolicDensity>();
    case 7: return run.template operator()<generated::VarianceGammaDensity>();
    default: return decltype(run.template operator()<generated::NormalDensity>()){};
    }
}

extern "C" std::size_t curvature_test_workspace_bytes(const CurvatureTestConfig* config)
{
    return dispatch(*config,[](const auto& curvature) {return curvature.workspace_bytes();});
}

extern "C" int curvature_test_weighted_hessian(const CurvatureTestConfig* config,
    const double* theta,const double* weights,double* output,void* memory,std::size_t bytes)
{
    const auto before=allocations;
    const bool valid=dispatch(*config,[&](const auto& curvature) {
        return curvature.weighted_hessian(theta,weights,output,memory,bytes);
    });
    if (allocations!=before) return -1;
    return valid ? 1 : 0;
}

template<class Program,auto StaticLayout>
auto constraint_workspace(const runtime::PowerConstraints<Program,StaticLayout>& constraints)
{
    struct Plan {std::size_t rows,local_jacobian,scratch,bytes;};
    Plan plan{std::max(std::size_t{1},constraints.mean.count()),
        std::max(std::size_t{1},constraints.mean.plan.jacobian_size()),
        std::max(std::size_t{1},constraints.mean.plan.workspace_bytes()),0u};
    if constexpr (!Program::aparch) {
        const auto q=constraints.layout().q_vol;
        plan.rows=std::max(plan.rows,2u*q);
        const mean::PolynomialDomain polynomial{q,q,nullptr};
        plan.scratch=std::max(plan.scratch,polynomial.workspace_bytes(true));
    }
    runtime::WorkspaceSizer size;
    size.add<double>(plan.rows);
    size.add<double>(plan.rows*constraints.dimension());
    size.add<double>(plan.local_jacobian);
    size.add<std::byte>(plan.scratch);
    size.add<std::byte>(alignof(double)-1u);
    plan.bytes=size.size();
    return plan;
}

extern "C" std::size_t curvature_test_constraint_size(const CurvatureTestConfig* config,
    unsigned field)
{
    return dispatch(*config,[&](const auto& curvature) {
        const auto& constraints=curvature.constraints;
        if (field==0u) return constraints.dimension();
        if (field==1u) return constraints.count();
        return constraint_workspace(constraints).bytes;
    });
}

extern "C" int curvature_test_constraints(const CurvatureTestConfig* config,unsigned request,
    const double* theta,double* values,double* jacobian,void* memory,std::size_t bytes,
    const std::size_t* columns,std::size_t count)
{
    const auto before=allocations;
    const bool valid=dispatch(*config,[&](const auto& curvature) {
        auto constraints=curvature.constraints;
        const auto plan=constraint_workspace(constraints);
        if (request>2u || !theta || !values || !jacobian || !memory || bytes<plan.bytes)
            return false;
        runtime::WorkspaceArena arena(memory,bytes);
        constraints.local_values=arena.take<double>(plan.rows);
        constraints.local_jacobian=arena.take<double>(plan.rows*constraints.dimension());
        constraints.mean.local_jacobian=arena.take<double>(plan.local_jacobian);
        constraints.mean.workspace=arena.take<std::byte>(plan.scratch);
        constraints.mean.bytes=plan.scratch;
        if (!constraints.local_values || !constraints.local_jacobian
            || !constraints.mean.local_jacobian || !constraints.mean.workspace) return false;
        return constraints.evaluate(static_cast<kraft::ConstraintRequest>(request),
            theta,values,jacobian,runtime::ConstraintColumns{columns,count});
    });
    if (allocations!=before) return -1;
    return valid ? 1 : 0;
}

extern "C" int curvature_test_parameter_units(const CurvatureTestConfig* config,
    const double* theta,const double* base,double h0,double* output)
{
    const auto before=allocations;
    const bool valid=dispatch(*config,[&]<class Program,auto StaticLayout>(
        const runtime::PowerConstraintCurvature<Program,StaticLayout>& curvature) {
        return curvature.constraints.program.template parameter_units<StaticLayout>(
            theta,base,h0,output);
    });
    if (allocations!=before) return -1;
    return valid ? 1 : 0;
}

template<bool Aparch>
bool arbitrary_fixed_layout_contract()
{
    using Density=generated::NormalDensity;
    using Orders=mean::LinearVolatilityOrders<11,4,5>;
    using Program=runtime::LinearPowerProgram<Density,Aparch>;
    constexpr auto* fixed=[] {
        if constexpr (Aparch) return &runtime::aparch::fixed_layout<Orders,Density,true>;
        else return &runtime::arma_egarch::fixed_layout<Orders,Density>;
    }();
    static_assert(Program::template fixed_layout<Orders> == fixed);
    using Constraints=runtime::PowerConstraints<Program,fixed>;
    using Width=decltype(std::declval<const Constraints&>()
        .template field<&Program::Layout::parameters>());
    static_assert(Width::value==fixed->parameters);
    using RuntimeWidth=decltype(std::declval<const runtime::PowerConstraints<Program>&>()
        .template field<&Program::Layout::parameters>());
    static_assert(std::is_same_v<RuntimeWidth,std::size_t>);
    Program program;
    mean::PolynomialConstraints polynomial{fixed->parameters,{},nullptr,nullptr,0};
    Constraints constraints{program,polynomial,false,nullptr,nullptr};
    runtime::PowerConstraintCurvature<Program,fixed> curvature{constraints};
    return &constraints.layout()==fixed && constraints.dimension()==fixed->parameters
        && curvature.workspace_bytes()!=0u;
}

extern "C" int curvature_test_arbitrary_fixed_layout()
{
    return arbitrary_fixed_layout_contract<true>() && arbitrary_fixed_layout_contract<false>();
}

extern "C" int curvature_test_overflow_contracts()
{
    using Program=runtime::LinearPowerProgram<generated::NormalDensity,true>;
    Program program;
    if (!program.prepare(0,1,1)) return 0;
    mean::PolynomialConstraints polynomial{program.layout.parameters,{},nullptr,nullptr,0};
    runtime::PowerConstraints<Program> constraints{program,polynomial,false,nullptr,nullptr};
    runtime::PowerConstraintCurvature curvature{constraints};
    const auto good=constraints.program.layout;
    const auto maximum=std::numeric_limits<std::size_t>::max();
    constraints.program.layout.parameters=maximum;
    constraints.mean.parameters=maximum;
    if (curvature.workspace_bytes()) return 0;
    const auto large=std::size_t{1}<<(std::numeric_limits<std::size_t>::digits/2);
    constraints.program.layout.parameters=large-1;
    constraints.mean.parameters=large-1;
    if (curvature.workspace_bytes()) return 0; // square fits; its byte count does not.
    constraints.program.layout=good;
    constraints.mean.parameters=good.parameters;
    constraints.program.layout.alpha_start=maximum;
    if (curvature.workspace_bytes()) return 0;
    constraints.program.layout=good;
    constraints.program.layout.p=maximum;
    if (curvature.workspace_bytes()) return 0;
    constraints.program.layout=good;
    constraints.mean.plan.first={{maximum,1,nullptr},0};
    if (curvature.workspace_bytes()) return 0;
    constraints.mean.plan.first={{2,maximum,nullptr},0};
    if (curvature.workspace_bytes()) return 0;
    constraints.mean.plan.first={{2,1,nullptr},0};
    if (curvature.workspace_bytes()) return 0;
    constraints.mean.plan.first={};
    return curvature.workspace_bytes()!=0;
}

extern "C" int curvature_test_first_order_only_range()
{
    using Density=generated::NormalDensity;
    const auto first=generated::asymmetric_power_moment<1,Density>(nullptr,0.,298.);
    const auto second=generated::asymmetric_power_moment<2,Density>(nullptr,0.,298.);
    return first.valid && !second.valid;
}

struct CountedDensity : generated::GeneralizedHyperbolicDensity {
    static inline unsigned first_preparations=0,second_preparations=0;
    template<bool F,bool G,bool H,bool E=false>
    static auto prepare(const double* shape,std::size_t n) noexcept
    {
        if constexpr (H) ++second_preparations;
        else if constexpr (G) ++first_preparations;
        return generated::GeneralizedHyperbolicDensity::template prepare<F,G,H,E>(shape,n);
    }
};

extern "C" int curvature_test_preparation_contracts()
{
    struct Program {
        using density_type=CountedDensity;
        enum {aparch=true};
        runtime::aparch::Layout layout;
        generated::PowerMomentValues<CountedDensity>* moment_values=nullptr;
    };
    Program program;
    if (!runtime::aparch::make_linear_layout<CountedDensity>(0,5,2,program.layout)) return 0;
    mean::PolynomialConstraints polynomial{program.layout.parameters,{},nullptr,nullptr,0};
    runtime::PowerConstraints<Program> constraints{program,polynomial,false,nullptr,nullptr};
    runtime::PowerConstraintCurvature curvature{constraints};
    std::vector<std::byte> arena(curvature.workspace_bytes());
    std::array<double,17> theta{};
    std::array<double,17*17> output{};
    double weight=0.;
    const auto& layout=program.layout;
    theta[layout.omega_index]=.08;
    theta[layout.delta_index]=1.7;
    theta[layout.model_parameters]=.1;
    theta[layout.model_parameters+1]=1.3;
    theta[layout.model_parameters+2]=.4;
    for (std::size_t i=0;i<layout.p;++i) theta[layout.gamma_start+i]=.01*i;
    const auto run=[&](unsigned first,unsigned second) {
        CountedDensity::first_preparations=CountedDensity::second_preparations=0;
        const auto before=allocations;
        return curvature.weighted_hessian(theta.data(),&weight,output.data(),
            arena.data(),arena.size()) && allocations==before
            && CountedDensity::first_preparations==first
            && CountedDensity::second_preparations==second;
    };
    if (!run(0,0)) return 0;
    weight=1.;
    if (!run(1,0)) return 0;
    theta[layout.alpha_start+2]=.04;
    if (!run(1,1)) return 0;
    for (std::size_t i=0;i<layout.p;++i) theta[layout.alpha_start+i]=.04;
    return run(0,1);
}

extern "C" void curvature_test_reset_common_preparations()
{
    aparch_test::common_preparations={};
}

extern "C" void curvature_test_common_preparations(std::size_t* output)
{
    std::copy_n(aparch_test::common_preparations.data(),3u,output);
}

struct AparchJointTestConfig {
    std::size_t density,columns,fixed; // columns=0 selects ARMA(2,1).
};

template<class Consumer>
auto with_aparch_joint(const AparchJointTestConfig& config,Consumer&& consume)
{
    const auto run=[&]<class Density>() {
        if (!config.columns) {
            if (config.fixed)
                return consume.template operator()<Density,runtime::aparch::FixedJointOrders<2,1,3,2>>();
            return consume.template operator()<Density,runtime::aparch::RuntimeOrders>();
        }
        if (!config.fixed)
            return consume.template operator()<Density,mean::LinearVolatilityOrders<>>();
        if (config.columns==4u)
            return consume.template operator()<Density,mean::LinearVolatilityOrders<4,3,2>>();
        if (config.columns==7u)
            return consume.template operator()<Density,mean::LinearVolatilityOrders<7,3,2>>();
        return decltype(consume.template operator()<Density,runtime::aparch::RuntimeOrders>()){};
    };
    switch (config.density) {
    case 0: return run.template operator()<generated::NormalDensity>();
    case 1: return run.template operator()<generated::StudentTDensity>();
    case 3: return run.template operator()<generated::GedDensity>();
    default: return decltype(run.template operator()<generated::NormalDensity>()){};
    }
}

template<class Density,class Orders>
auto aparch_joint_layout(const AparchJointTestConfig& config)
{
    runtime::aparch::Layout layout;
    if constexpr (Orders::linear)
        runtime::aparch::make_linear_layout<Density>(config.columns,3u,2u,layout);
    else runtime::aparch::make_layout<Density>(2u,1u,3u,2u,true,layout);
    return layout;
}

extern "C" std::size_t curvature_test_joint_workspace(const AparchJointTestConfig* config,
    unsigned transformed,unsigned request,unsigned map_only)
{
    return with_aparch_joint(*config,[&]<class Density,class Orders>() {
        using Shape=generated::DensityShapeTransformT<Density>;
        const auto layout=aparch_joint_layout<Density,Orders>(*config);
        constexpr auto fixed=Orders::fixed ? &runtime::aparch::fixed_layout<Orders,Density,true> : nullptr;
        return map_only ? runtime::aparch_transform::workspace_bytes<fixed>(layout,8u,request)
            : runtime::aparch::workspace_bytes<Density,Shape,Orders,true>(layout,transformed,request);
    });
}

extern "C" int curvature_test_joint_jet(const AparchJointTestConfig* config,
    unsigned transformed,unsigned request,const double* point,const double* y,const double* x,
    std::size_t n,double h0,void* memory,std::size_t bytes,double* f,double* g,double* h)
{
    const auto before=allocations;
    const int status=with_aparch_joint(*config,[&]<class Density,class Orders>() {
        const auto layout=aparch_joint_layout<Density,Orders>(*config);
        if constexpr (Orders::linear) {
            runtime::LinearPowerProgram<Density,true> program;
            program.layout=layout;
            return program.template evaluate<Orders>(point,y,x,n,h0,request,transformed,
                memory,bytes,f,g,h,nullptr,nullptr);
        } else {
            using Shape=generated::DensityShapeTransformT<Density>;
            unsigned valid=0;
            const int result=runtime::aparch::evaluate<true,Density,Shape,Orders,true>(layout,
                point,layout.parameters,y,n,h0,transformed,request,memory,bytes,f,g,h,&valid);
            return result==0 && valid!=request ? -999 : result;
        }
    });
    return allocations==before ? status : -998;
}

template<int Order,class Density,class Shape,auto Fixed>
bool aparch_joint_map(const runtime::aparch::Layout& layout,const double* z,unsigned request,
    const double* weights,void* memory,std::size_t bytes,double* theta,double* jacobian,double* hessian)
{
    namespace tr=runtime::aparch_transform;
    tr::Scratch scratch;
    if (!tr::bind_scratch<Fixed>(layout,8u,request,memory,bytes,scratch)
        || !tr::map_coordinates<Order,Density,Shape,Fixed>(layout,z,scratch)) return false;
    const auto k=layout.parameters;
    if (request&1u) std::copy_n(scratch.theta,k,theta);
    if (request&2u) std::copy_n(scratch.jacobian,k*k,jacobian);
    if constexpr (Order>=2) {
        std::fill_n(scratch.theta_hessian,k*k,0.);
        std::copy_n(weights,k,scratch.theta_gradient);
        if (!tr::pullback_hessian<Density,Shape,Fixed>(layout,z,scratch,hessian)) return false;
    }
    return true;
}

extern "C" int curvature_test_joint_map(const AparchJointTestConfig* config,unsigned request,
    const double* z,const double* weights,void* memory,std::size_t bytes,
    double* theta,double* jacobian,double* hessian)
{
    const auto before=allocations;
    const bool valid=with_aparch_joint(*config,[&]<class Density,class Orders>() {
        using Shape=generated::DensityShapeTransformT<Density>;
        const auto layout=aparch_joint_layout<Density,Orders>(*config);
        constexpr auto fixed=Orders::fixed ? &runtime::aparch::fixed_layout<Orders,Density,true> : nullptr;
        if (request&4u) return aparch_joint_map<2,Density,Shape,fixed>(
            layout,z,request,weights,memory,bytes,theta,jacobian,hessian);
        if (request&2u) return aparch_joint_map<1,Density,Shape,fixed>(
            layout,z,request,weights,memory,bytes,theta,jacobian,hessian);
        return aparch_joint_map<0,Density,Shape,fixed>(
            layout,z,request,weights,memory,bytes,theta,jacobian,hessian);
    });
    if (allocations!=before) return -1;
    return valid ? 1 : 0;
}
