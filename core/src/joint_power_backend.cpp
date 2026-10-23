#include "joint_power_backend.hpp"
#include "models/unit_layout_roles.hpp"
#include "chart_contract.hpp"
#include "power_backend.hpp"
#include "models/density_domain.hpp"
#include "models/joint_power/program.hpp"
#include "models/mean/stability_domain.hpp"
#include "models/mean/generated_ar_chart.hpp"

#include <array>
#include <cmath>

namespace scivolapi::detail::joint_power_backend {
namespace {
namespace native = vlk::experiment::runtime;

Status config_status(ModelConfig c) noexcept {
    if (c.density>=8 || !c.p_vol || !c.q_vol
        || (c.mean!=MeanKind::arma && c.mean!=MeanKind::linear)
        || (c.kind!=Kind::egarch && c.kind!=Kind::aparch)
        || (c.mean==MeanKind::arma && (c.design_width || c.linear_ar_order))
        || (c.mean==MeanKind::linear && (c.p_ar || c.q_ma || c.linear_ar_order>c.design_width)))
        return Status::invalid_argument;
    return Status::ok;
}

power_backend::ModelConfig power_config(ModelConfig c) noexcept {
    return {c.kind==Kind::aparch ? power_backend::Kind::aparch : power_backend::Kind::egarch,
        c.p_vol,c.q_vol,c.density,false};
}

template<class Function>
decltype(auto) dispatch_generated_ar_layout(ModelConfig c,Function&& function) noexcept {
    if (!c.linear_ar_order || chart_dispatch_kind(c)<=0)
        return function.template operator()<nullptr>();
    return program::density(c.density,[&]<class Density,class Shape>() {
        const auto select=[&]<bool Aparch>() {
            using Orders=native::joint_power::Orders<true,true,4,0,0,1,1>;
            constexpr auto Inner=program::selected_layout<Orders,Density,Aparch>;
            return generated_ar_chart::dispatch_common<Inner->parameters>(true,function);
        };
        return c.kind==Kind::aparch ? select.template operator()<true>() : select.template operator()<false>();
    });
}

std::size_t domain_bytes(ModelConfig c) noexcept {
    const auto power=power_backend::requirements(power_config(c),Coordinates::theta,JetRequest::F);
    native::WorkspaceSizer size;
    const auto order=std::max({c.p_ar,c.q_ma,c.linear_ar_order});
    if (power.status!=Status::ok || !size.add<double>(order) || !size.add<double>(order)
        || !size.add<std::byte>(power.bytes)) return 0;
    return size.size();
}

native::joint_power::SamplePlan sample_plan(ModelConfig c,SampleView sample) noexcept {
    const auto begin=sample.recursion_begin==default_position ? default_recursion_begin(c) : sample.recursion_begin;
    return {begin,sample.first_scored==default_position ? begin : sample.first_scored,
        {sample.observation_history,sample.innovation_history,begin,sample.implicit_log_variance}};
}

struct Range { std::uintptr_t begin=0,end=0; bool valid=true; };
template<class T> Range range(std::span<T> values) noexcept {
    if (values.empty()) return {};
    const auto begin=reinterpret_cast<std::uintptr_t>(values.data());
    std::size_t bytes;
    if (!begin || !native::checked_product(values.size(),sizeof(T),bytes)
        || bytes>std::numeric_limits<std::uintptr_t>::max()-begin) return {0,0,false};
    return {begin,begin+bytes,true};
}
bool overlaps(Range a,Range b) noexcept {
    return a.begin<a.end && b.begin<b.end && a.begin<b.end && b.begin<a.end;
}
bool admitted(std::span<const double> input) noexcept {
    const auto r=range(input);
    return r.valid && (!r.begin || r.begin%alignof(double)==0);
}

Status sample_admission(ModelConfig c,SampleView sample,bool scan) noexcept {
    if (config_status(c)!=Status::ok || !parameter_count(c)) return Status::invalid_argument;
    if (!sample.variance_history.empty()
        || (!c.p_ar && !sample.observation_history.empty())) return Status::unsupported_operation;
    const auto history=sample_plan(c,sample).history;
    if (!history.shape(vlk::experiment::conditioning::required_lags(c.p_ar,c.q_ma,c.p_vol)) || (scan && !history.finite()))
        return Status::invalid_argument;
    if (!admitted(sample.observations) || !admitted(sample.design)) return Status::invalid_argument;
    const auto n=sample.observations.size();
    const auto plan=sample_plan(c,sample);
    if (!n || plan.recursion_begin>plan.first_scored || plan.first_scored>=n
        || (c.mean==MeanKind::arma && plan.recursion_begin<default_recursion_begin(c))
        || !std::isfinite(sample.variance_seed) || !(sample.variance_seed>0)) return Status::invalid_argument;
    if (c.mean==MeanKind::linear && plan.recursion_begin!=0) return Status::unsupported_operation;
    std::size_t design_count;
    if (!native::checked_product(n,c.design_width,design_count) || sample.design.size()!=design_count)
        return Status::invalid_argument;
    if (scan) {
        for (double y:sample.observations) if (!std::isfinite(y)) return Status::invalid_argument;
        for (double x:sample.design) if (!std::isfinite(x)) return Status::invalid_argument;
    }
    return Status::ok;
}

Status output_admission(std::span<const double> parameters,SampleView sample,
    std::span<std::byte> work,Buffers output,unsigned mask,std::size_t width,std::size_t required) noexcept {
    std::size_t square;
    if (!native::checked_product(width,width,square)) return Status::invalid_argument;
    if (((mask&1) && output.value.size()<1) || ((mask&2) && output.gradient.size()<width)
        || ((mask&4) && output.hessian.size()<square) || work.size()<required)
        return Status::insufficient_capacity;
    const auto memory=range(work);
    const std::array<Range,5> inputs{range(parameters),range(sample.observations),range(sample.design),
        range(sample.observation_history),range(sample.innovation_history)};
    const std::array<Range,3> outputs{
        range((mask&1) ? output.value.first(1) : std::span<double>{}),
        range((mask&2) ? output.gradient.first(width) : std::span<double>{}),
        range((mask&4) ? output.hessian.first(square) : std::span<double>{})};
    if (!memory.valid || (required && (!memory.begin || memory.begin%alignof(double)))) return Status::invalid_argument;
    for (auto input:inputs)
        if (!input.valid || (input.begin && input.begin%alignof(double)) || overlaps(memory,input))
            return Status::invalid_argument;
    for (std::size_t i=0;i<outputs.size();++i) {
        if (!outputs[i].valid || overlaps(memory,outputs[i])
            || (outputs[i].begin && outputs[i].begin%alignof(double))) return Status::invalid_argument;
        for (auto input:inputs) if (overlaps(input,outputs[i])) return Status::invalid_argument;
        for (std::size_t j=i+1;j<outputs.size();++j)
            if (overlaps(outputs[i],outputs[j])) return Status::invalid_argument;
    }
    return Status::ok;
}

template<class Layout>
Status point_status(ModelConfig c,const Layout& layout,const double* theta,
    std::span<std::byte> memory) noexcept {
    if (!layout.valid || !theta) return Status::invalid_argument;
    for (std::size_t i=0;i<layout.parameters;++i)
        if (!std::isfinite(theta[i])) return Status::mathematical_domain;
    const auto shape=density_shape_status(c.density,theta+layout.model_parameters);
    if (shape!=Status::ok) return shape;
    const auto required=domain_bytes(c);
    if (!required || memory.size()<required) return Status::insufficient_capacity;
    native::WorkspaceArena arena(memory.data(),memory.size());
    const auto order=std::max({c.p_ar,c.q_ma,c.linear_ar_order});
    auto* scratch=arena.take<double>(2*order);
    namespace mean=vlk::experiment::mean;
    for (const auto [start,count] : {std::pair{c.mean==MeanKind::linear ? std::size_t(c.intercept) : layout.ar_start,
                                            c.mean==MeanKind::linear ? c.linear_ar_order : c.p_ar},
                                    std::pair{layout.ma_start,c.q_ma}}) {
        if (!count) continue;
        const auto status=mean::polynomial_status({theta+start,count},
            c.q_ma && start==layout.ma_start ? -1.0 : 1.0,{scratch,2*order});
        if (status!=Status::ok) return status;
    }
    const auto offset=2*order*sizeof(double);
    return power_backend::parameter_status(power_config(c),
        {theta+layout.omega_index,layout.parameters-layout.omega_index},memory.subspan(offset));
}

template<unsigned Mask,class Orders,class Density,class Shape,bool Aparch>
Status execute(ModelConfig c,const program::Layout<Aparch>& layout,SampleView sample,
    std::span<const double> parameters,Coordinates coordinates,std::span<std::byte> memory,Buffers output, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept {
    const auto domain=domain_bytes(c);
    const auto kernel=memory.subspan(domain);
    const auto status=program::evaluate<Mask,Orders,Density,Shape,Aparch>(layout,sample,sample_plan(c,sample),
        parameters.data(),coordinates==Coordinates::original_chart,kernel.data(),kernel.size(),
        (Mask&1) ? output.value.data() : nullptr,(Mask&2) ? output.gradient.data() : nullptr,
        (Mask&4) ? output.hessian.data() : nullptr,[&](const double* theta) {
            return point_status(c,layout,theta,memory.first(domain));
        },persistent);
    if (status!=Status::ok) return status;
    if constexpr (Mask&1) if (!std::isfinite(output.value[0])) return Status::numerical_range;
    if constexpr (Mask&2) for (std::size_t i=0;i<layout.parameters;++i)
        if (!std::isfinite(output.gradient[i])) return Status::numerical_range;
    if constexpr (Mask&4) for (std::size_t i=0;i<layout.parameters*layout.parameters;++i)
        if (!std::isfinite(output.hessian[i])) return Status::numerical_range;
    return Status::ok;
}

Status execute_mask(ModelConfig c,SampleView sample,std::span<const double> parameters,
    Coordinates coordinates,unsigned mask,std::span<std::byte> work,Buffers output, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept {
    return program::density(c.density,[&]<class Density,class Shape>() {
        const auto family=[&]<bool Aparch>() {
            const auto layout=program::layout_for<Density,Aparch>(c);
            return program::orders(c,[&]<class Orders>() {
                switch(mask) {
#define SCIVOL_JOINT_POWER_MASK(M) case M: return execute<M,Orders,Density,Shape,Aparch>(c,layout,sample,parameters,coordinates,work,output,persistent)
                SCIVOL_JOINT_POWER_MASK(1); SCIVOL_JOINT_POWER_MASK(2); SCIVOL_JOINT_POWER_MASK(3);
                SCIVOL_JOINT_POWER_MASK(4); SCIVOL_JOINT_POWER_MASK(5); SCIVOL_JOINT_POWER_MASK(6);
                SCIVOL_JOINT_POWER_MASK(7);
#undef SCIVOL_JOINT_POWER_MASK
                }
                return Status::invalid_argument;
            });
        };
        return c.kind==Kind::aparch ? family.template operator()<true>() : family.template operator()<false>();
    });
}

JetValidity evaluate_impl(ModelConfig c,SampleView sample,std::span<const double> parameters,
    Coordinates coordinates,JetRequest request,std::span<std::byte> memory,Buffers output,bool scan, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept {
    JetValidity result;
    result.requested=request;
    const auto mask=static_cast<unsigned>(request);
    const auto needed=requirements(c,coordinates,request);
    auto status=needed.status;
    if (status==Status::ok) status=sample_admission(c,sample,scan);
    if (status==Status::ok && (parameters.size()!=needed.parameters || !admitted(parameters)))
        status=Status::invalid_argument;
    if (status==Status::ok) for (double value:parameters)
        if (!std::isfinite(value)) {status=Status::invalid_argument;break;}
    if (status==Status::ok) status=output_admission(parameters,sample,memory,output,mask,needed.parameters,needed.bytes);
    if (status!=Status::ok) {
        result.status=status;
        if (mask&1) result.f_reason=status;
        if (mask&2) result.g_reason=status;
        if (mask&4) result.h_reason=status;
        return result;
    }
    if (coordinates==Coordinates::original_chart && c.linear_ar_order) {
        auto inner=c; inner.linear_ar_order=0;
        const auto inner_request=static_cast<JetRequest>(generated_ar_chart::inner_mask(mask));
        const auto inner_bytes=requirements(inner,coordinates,inner_request).bytes;
        return dispatch_generated_ar_layout(c,[&]<const generated_ar_chart::Layout* Fixed>() {
            return generated_ar_chart::evaluate<Fixed>(parameters,{std::size_t(c.intercept),c.linear_ar_order},
                request,inner_bytes,memory,output,[&](auto point,JetRequest selected,auto work,Buffers buffers) {
                    return evaluate_impl(inner,sample,point,coordinates,selected,work,buffers,false,persistent);
                });
        });
    }
    status=execute_mask(c,sample,parameters,coordinates,mask,memory,output,persistent);
    if (status==Status::ok) result.valid_mask=mask;
    else if ((mask&(mask-1))!=0) {
        for (unsigned bit:{1u,2u,4u}) if (mask&bit) {
            const auto individual=execute_mask(c,sample,parameters,coordinates,bit,memory,output,persistent);
            if (individual==Status::ok) result.valid_mask|=bit;
            if (bit==1) result.f_reason=individual;
            if (bit==2) result.g_reason=individual;
            if (bit==4) result.h_reason=individual;
        }
    }
    result.status=result.valid_mask==mask ? Status::ok : status;
    if (mask&1) result.f_reason=(result.valid_mask&1) ? Status::ok : result.f_reason==Status::unsupported_operation ? status : result.f_reason;
    if (mask&2) result.g_reason=(result.valid_mask&2) ? Status::ok : result.g_reason==Status::unsupported_operation ? status : result.g_reason;
    if (mask&4) result.h_reason=(result.valid_mask&4) ? Status::ok : result.h_reason==Status::unsupported_operation ? status : result.h_reason;
    return vlk::experiment::density_status::complete_validity(result);
}
} // namespace

std::size_t parameter_count(ModelConfig c) noexcept {
    if (config_status(c)!=Status::ok) return 0;
    return program::density(c.density,[&]<class Density,class Shape>() {
        if (c.kind==Kind::aparch) {const auto layout=program::layout_for<Density,true>(c); return layout.valid ? layout.parameters : 0;}
        const auto layout=program::layout_for<Density,false>(c); return layout.valid ? layout.parameters : 0;
    });
}
std::size_t default_recursion_begin(ModelConfig c) noexcept {
    return c.mean==MeanKind::linear ? 0 : std::max({c.p_ar,c.q_ma,c.p_vol,c.q_vol});
}
int dispatch_kind(ModelConfig c) noexcept {
    if (!parameter_count(c)) return -1;
    if (!c.fixed || c.p_vol!=1 || c.q_vol!=1) return 0;
    if (c.mean==MeanKind::arma) return c.intercept && c.p_ar==1 && c.q_ma==1 ? 1 : 0;
    std::size_t columns;
    return native::checked_sum(c.design_width,std::size_t(c.intercept),columns) && columns==4 ? 2 : 0;
}
int chart_dispatch_kind(ModelConfig c) noexcept {
    const auto selected=dispatch_kind(c);
    if (selected<0 || !c.linear_ar_order) return selected;
    return generated_ar_chart::common_candidate(selected,c.intercept,c.design_width,c.linear_ar_order) ? selected : 0;
}
Requirements requirements(ModelConfig c,Coordinates coordinates,JetRequest request) noexcept {
    Requirements result;
    const auto mask=static_cast<unsigned>(request);
    if (config_status(c)!=Status::ok || mask<1 || mask>7 || (coordinates!=Coordinates::theta
        && coordinates!=Coordinates::original_chart) || !parameter_count(c)) return result;
    if (coordinates==Coordinates::original_chart && c.linear_ar_order) {
        auto inner=c; inner.linear_ar_order=0;
        result=requirements(inner,coordinates,static_cast<JetRequest>(generated_ar_chart::inner_mask(mask)));
        if (result.status!=Status::ok) return result;
        result.bytes=dispatch_generated_ar_layout(c,[&]<const generated_ar_chart::Layout* Fixed>() {
            return generated_ar_chart::workspace_bytes<Fixed>(result.parameters,
                {std::size_t(c.intercept),c.linear_ar_order},mask,result.bytes);
        });
        result.dispatch=chart_dispatch_kind(c);
        result.status=result.bytes ? Status::ok : Status::invalid_argument;
        return result;
    }
    return program::density(c.density,[&]<class Density,class Shape>() {
        const auto family=[&]<bool Aparch>() {
            Requirements need;
            const auto layout=program::layout_for<Density,Aparch>(c);
            const auto kernel=program::orders(c,[&]<class Orders>() {
                return program::bytes<Orders,Density,Aparch>(layout,mask,coordinates==Coordinates::original_chart);
            });
            const auto domain=domain_bytes(c);
            native::WorkspaceSizer size;
            if (!layout.valid || !kernel || !domain || !size.add<std::byte>(domain) || !size.add<std::byte>(kernel)) return need;
            need.status=Status::ok; need.bytes=size.size(); need.parameters=layout.parameters;
            need.dispatch=dispatch_kind(c);
            return need;
        };
        return c.kind==Kind::aparch ? family.template operator()<true>() : family.template operator()<false>();
    });
}
Status sample_status(ModelConfig c,SampleView sample) noexcept { return sample_admission(c,sample,true); }
Requirements map_requirements(ModelConfig c,bool jacobian) noexcept {
    if (c.linear_ar_order && config_status(c)==Status::ok) {
        auto inner=c; inner.linear_ar_order=0;
        auto result=map_requirements(inner,jacobian);
        if (result.status!=Status::ok) return result;
        result.bytes=dispatch_generated_ar_layout(c,[&]<const generated_ar_chart::Layout* Fixed>() {
            return generated_ar_chart::workspace_bytes<Fixed>(result.parameters,
                {std::size_t(c.intercept),c.linear_ar_order},jacobian ? 2u : 1u,result.bytes);
        });
        result.dispatch=chart_dispatch_kind(c);
        result.status=result.bytes ? Status::ok : Status::invalid_argument;
        return result;
    }
    // Map scratch shares the exact request-order planner; no objective is evaluated.
    return requirements(c,Coordinates::original_chart,jacobian ? JetRequest::G : JetRequest::F);
}
Status map(ModelConfig c,std::span<const double> z,std::span<double> theta,std::span<std::byte> memory,
    std::span<double> jacobian) noexcept {
    const auto need=map_requirements(c,!jacobian.empty());
    if (need.status!=Status::ok) return need.status;
    if (z.size()!=need.parameters || theta.size()!=need.parameters) return Status::invalid_argument;
    const auto status=output_admission(z,{},memory,{{},theta,jacobian},jacobian.empty() ? 2u : 6u,need.parameters,need.bytes);
    if (status!=Status::ok) return status;
    for (double value:z) if (!std::isfinite(value)) return Status::invalid_argument;
    if (c.linear_ar_order) {
        auto inner=c; inner.linear_ar_order=0;
        return dispatch_generated_ar_layout(c,[&]<const generated_ar_chart::Layout* Fixed>() {
            return generated_ar_chart::map<Fixed>(z,{std::size_t(c.intercept),c.linear_ar_order},theta,jacobian,
                map_requirements(inner,!jacobian.empty()).bytes,memory,[&](auto point,auto work) {
                    return map(inner,point,theta,work,jacobian);
                });
        });
    }
    const auto kernel=memory.subspan(domain_bytes(c));
    const bool mapped=program::density(c.density,[&]<class Density,class Shape>() {
        const auto family=[&]<bool Aparch>() {
            const auto layout=program::layout_for<Density,Aparch>(c);
            return program::orders(c,[&]<class Orders>() {
                if (jacobian.empty()) return program::map_point<0,Orders,Density,Shape,Aparch>(layout,z.data(),kernel.data(),kernel.size(),theta.data());
                return program::map_point<1,Orders,Density,Shape,Aparch>(layout,z.data(),kernel.data(),kernel.size(),theta.data(),jacobian.data());
            });
        };
        return c.kind==Kind::aparch ? family.template operator()<true>() : family.template operator()<false>();
    });
    if (!mapped) return Status::numerical_range;
    for (double value:theta) if (!std::isfinite(value)) return Status::numerical_range;
    if (!jacobian.empty()) for (std::size_t i=0;i<need.parameters*need.parameters;++i)
        if (!std::isfinite(jacobian[i])) return Status::numerical_range;
    return mapped_student_shape_status(c.density,theta);
}
Status parameter_status(ModelConfig c,std::span<const double> theta,std::span<std::byte> memory) noexcept {
    if (!parameter_count(c) || theta.size()!=parameter_count(c)) return Status::invalid_argument;
    const auto needed=domain_bytes(c);
    const auto status=output_admission(theta,{},memory,{},0,theta.size(),needed);
    if (status!=Status::ok) return status;
    return program::density(c.density,[&]<class Density,class Shape>() {
        return c.kind==Kind::aparch ? point_status(c,program::layout_for<Density,true>(c),theta.data(),memory)
            : point_status(c,program::layout_for<Density,false>(c),theta.data(),memory);
    });
}
JetValidity evaluate(ModelConfig c,SampleView sample,std::span<const double> parameters,Coordinates coordinates,
    JetRequest request,std::span<std::byte> memory,Buffers output) noexcept {
    return evaluate_impl(c,sample,parameters,coordinates,request,memory,output,true);
}
JetValidity evaluate_prevalidated(ModelConfig c,SampleView sample,std::span<const double> parameters,Coordinates coordinates,
    JetRequest request,std::span<std::byte> memory,Buffers output,DensityPreparation* persistent) noexcept {
    return evaluate_impl(c,sample,parameters,coordinates,request,memory,output,false,persistent);
}
unit_map::Layout unit_layout(ModelConfig c) noexcept {
    if (config_status(c) != Status::ok) return {};
    return program::density(c.density, [&]<class Density, class Shape>() {
        const auto exog = c.mean == MeanKind::linear
            ? unit_map::Block{std::size_t(c.intercept), c.design_width} : unit_map::Block{};
        if (c.kind == Kind::aparch) {
            const auto l = program::layout_for<Density, true>(c);
            return unit_layout_roles::power<true>(l,c.intercept,exog);
        }
        const auto l = program::layout_for<Density, false>(c);
        return unit_layout_roles::power<false>(l,c.intercept,exog);
    });
}

std::size_t chart_pullback_bytes(ModelConfig c, JetRequest r) noexcept {
    return requirements(c,Coordinates::original_chart,r).bytes;
}
Status chart_pullback(ModelConfig c, SampleView sample, std::span<const double> z,
    std::span<const double> theta, JetRequest request, std::span<std::byte> work,
    unit_map::InternalJet input, KernelBuffers output) noexcept {
    const auto n=parameter_count(c);
    const auto mask=static_cast<unsigned>(request);
    if (const auto s=admit_chart(z,theta,request,work,chart_pullback_bytes(c,request),n,input,output); s!=Status::ok) return s;
    if (c.linear_ar_order) {
        auto inner=c; inner.linear_ar_order=0;
        const auto inner_request=static_cast<JetRequest>(generated_ar_chart::inner_mask(mask));
        return dispatch_generated_ar_layout(c,[&]<const generated_ar_chart::Layout* Fixed>() {
            return generated_ar_chart::chart_pullback<Fixed>(z,{std::size_t(c.intercept),c.linear_ar_order},request,
                chart_pullback_bytes(inner,inner_request),work,output,
                [&](auto point,JetRequest selected,auto memory,Buffers buffers) {
                    return chart_pullback(inner,sample,point,theta,selected,memory,input,buffers);
                });
        });
    }
    return program::density(c.density,[&]<class Density,class Shape>() {
        return program::orders(c,[&]<class Orders>() {
            const auto run=[&]<bool Aparch>() {
                constexpr auto Fixed=program::selected_layout<Orders,Density,Aparch>;
                const auto supplied=program::layout_for<Density,Aparch>(c);
                const auto& l=[&]() -> const program::Layout<Aparch>& {
                    if constexpr (Fixed!=nullptr) return *Fixed; else return supplied;
                }();
                const auto core=(mask&4) ? program::core_bytes<2,Orders,Density,Aparch>(l,mask|2)
                    : program::core_bytes<1,Orders,Density,Aparch>(l,mask);
                if constexpr (Aparch) {
                    native::aparch_transform::Scratch scratch;
                    if (!native::aparch_transform::bind_scratch<Fixed>(l,core,mask,work.data(),work.size(),scratch))
                        return Status::insufficient_capacity;
                    const bool mapped=(mask&4)
                        ? native::aparch_transform::map_coordinates<2,Density,Shape,Fixed>(l,z.data(),scratch)
                        : native::aparch_transform::map_coordinates<1,Density,Shape,Fixed>(l,z.data(),scratch);
                    if (!mapped) return Status::numerical_range;
                    std::copy_n(theta.data(),n,scratch.theta);
                    std::copy_n(input.gradient.data(),n,scratch.theta_gradient);
                    if (mask&2) native::aparch_transform::pullback_gradient<Fixed>(l,scratch,scratch.theta_gradient,output.gradient.data());
                    if (mask&4) {
                        std::copy_n(input.hessian.data(),n*n,scratch.theta_hessian);
                        if (!native::aparch_transform::pullback_hessian<Density,Shape,Fixed>(l,z.data(),scratch,output.hessian.data()))
                            return Status::numerical_range;
                    }
                } else {
                    native::joint_power::EgarchChartScratch scratch;
                    if (!native::joint_power::bind_egarch_chart<Fixed>(l,core,mask,work.data(),work.size(),scratch))
                        return Status::insufficient_capacity;
                    if (!native::joint_power::map_egarch_chart<Shape,Fixed>(l,z.data(),true,scratch)) return Status::numerical_range;
                    std::copy_n(theta.data(),n,scratch.power.theta);
                    std::copy_n(input.gradient.data(),n,scratch.power.theta_gradient);
                    if (mask&2) native::joint_power::egarch_gradient<Fixed>(l,scratch,output.gradient.data());
                    if (mask&4) {
                        std::copy_n(input.hessian.data(),n*n,scratch.power.theta_hessian);
                        if (!native::joint_power::egarch_hessian<Shape,Fixed>(l,z.data(),scratch,output.hessian.data()))
                            return Status::numerical_range;
                    }
                }
                return Status::ok;
            };
            return c.kind==Kind::aparch ? run.template operator()<true>() : run.template operator()<false>();
        });
    });
}

} // namespace scivolapi::detail::joint_power_backend
