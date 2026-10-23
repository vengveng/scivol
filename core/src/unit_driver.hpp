#pragma once
#include "internal.hpp"
#include "objective_projection.hpp"
#include "unit_layout_dispatch.hpp"
#include "preparation_plan.hpp"
#include <cstring>

namespace scivolapi::detail {
inline bool selected_restriction(const ObjectiveView& objective) noexcept {
    const auto& state=Access::get(objective);
    return !state.profile_config && state.projection && state.projection->restricted()
        && !state.projection->indices().empty()
        && program::supports_active_basis(Access::get(objective.problem().model()).program);
}
template<class Callback>
decltype(auto) with_unit_plan(const ObjectiveView& objective,const unit_map::Plan<>& source,
    Callback&& callback) noexcept {
    const auto run=[&](auto owner) -> decltype(auto) {
        using Owner=decltype(owner);
        unit_map::Plan<Owner> plan;
        const auto status=unit_map::rebind_plan(owner,source,plan);
        return callback(status,plan);
    };
    if(Access::get(objective).profile_config)
        return unit_layout_dispatch::profiled(*Access::get(objective).profile_config,
            source.layout().get(),run);
    return unit_layout_dispatch::dispatch(Access::get(objective.problem().model()).program,source.layout().get(),run);
}
template<class Config>
constexpr bool prepares_student_density(const Config& config) noexcept {
    if constexpr (std::is_same_v<Config,backend::ModelConfig>
        || std::is_same_v<Config,mean_backend::ModelConfig>
        || std::is_same_v<Config,power_backend::ModelConfig>
        || std::is_same_v<Config,joint_backend::ModelConfig>
        || std::is_same_v<Config,joint_power_backend::ModelConfig>)
        return config.density==1;
    else return false;
}
inline bool prepares_student_density(const ObjectiveView& objective) noexcept {
    // Reduced profile inputs and nuisance reconstruction have a separate owner.
    if (objective.key().target==DerivativeTarget::exact_profile
        || Access::get(objective).profile_config) return false;
    const auto& model=Access::get(objective.problem().model());
    return std::visit([](const auto& config) {return prepares_student_density(config);},model.program);
}
inline bool prepares_density(const ObjectiveView& objective) noexcept {
    if(objective.key().target==DerivativeTarget::exact_profile || Access::get(objective).profile_config)return false;
    return preparation_plan::enabled(Access::get(objective.problem().model()).program);
}
inline bool uses_preparation_workspace(const ObjectiveView& objective) noexcept {
    return !Access::get(objective.problem()).identity_units || prepares_density(objective);
}
struct UnitWorkPlan {
    Status status=Status::invalid_argument;
    std::array<std::size_t,3> cache_bytes{};
    std::size_t bytes=0,kernel_bytes=0,chart_bytes=0;
    unsigned internal_mask=0,maximum=0,density_mask=0;
    bool selected=false;
    DensityPreparation::Requirements density{};
};
inline JetValidity unit_failure(Status s,JetRequest request) noexcept {
    const auto m=static_cast<unsigned>(request);
    return {s,request,0,(m&1) ? s : Status::unsupported_operation,
        (m&2) ? s : Status::unsupported_operation,(m&4) ? s : Status::unsupported_operation};
}
inline unsigned unit_inner_mask(const ObjectiveView& objective,bool chart,unsigned mask) noexcept {
    if (chart && (mask&4)) mask|=2;
    return with_unit_plan(objective,Access::get(objective.problem()).unit_plan,
        [&](Status status,const auto& plan) {return status==Status::ok
            ?static_cast<unsigned>(unit_map::internal_request(plan,static_cast<JetRequest>(mask))):0u;});
}
inline std::size_t unit_parameter_count(const ObjectiveView& objective) noexcept {
    return Access::get(objective).profile_config ? objective.derivative_layout().axes.size()
        : objective.problem().model().layout().full_theta_width;
}
inline UnitWorkPlan unit_work_plan(const ObjectiveView& objective,JetRequest maximum) noexcept {
    const auto& p=Access::get(objective.problem());
    const auto& model=Access::get(p.model);
    UnitWorkPlan plan;
    plan.maximum=static_cast<unsigned>(maximum);
    if (!uses_preparation_workspace(objective) || !valid_mask(plan.maximum)) return plan;
    const bool chart=objective.key().chart!=0;
    const auto& projection=Access::get(objective).projection;
    const bool selected=!Access::get(objective).profile_config
        && projection && projection->restricted() && !projection->indices().empty()
        && program::supports_active_basis(model.program);
    const auto axes=projection ? std::optional{projection->sensitivity_basis()} : std::nullopt;
    plan.selected=selected;
    const auto seeds=selected ? std::optional{selected_units::Point(*axes,p.unit_plan)} : std::nullopt;
    const auto& profile=Access::get(objective).profile_config;
    const auto& units=profile ? *Access::get(objective).profile_units : p.unit_plan;
    // Binding reconstructs sigma2 even when the public arena is G-only or H-only.
    const unsigned cache_mask=plan.maximum | (profile ? 1u : 0u);
    plan.internal_mask=profile ? cache_mask : selected ? plan.maximum : unit_inner_mask(objective,chart,plan.maximum);
    const auto semantics=program_mean_semantics(p);
    if (profile) {
        const auto coordinates=chart ? Coordinates::original_chart : Coordinates::theta;
        const auto requested=profile_backend::requirements(*profile,coordinates,maximum);
        const auto reconstruction=profile_backend::requirements(*profile,coordinates,JetRequest::F);
        const auto map=profile_backend::map_requirements(*profile);
        if (requested.status!=Status::ok || reconstruction.status!=Status::ok || map.status!=Status::ok)
            return plan;
        const auto domain=semantics.causal
            ? program::workspace_bytes(model.program,false,JetRequest::F,semantics) : 0;
        if (semantics.causal && !domain) return plan;
        plan.kernel_bytes=std::max({requested.bytes,reconstruction.bytes,map.bytes,domain});
    } else {
        plan.kernel_bytes=program::workspace_bytes(model.program,p.identity_units && chart,
            static_cast<JetRequest>(plan.internal_mask),semantics,
            selected ? &*axes : nullptr,selected && !p.identity_units ? &*seeds : nullptr);
        // Point maps/domain checks must not use or overwrite a prepared unit cache.
        const auto binding_bytes=program::workspace_bytes(model.program,chart,JetRequest::F,semantics);
        plan.kernel_bytes=std::max(plan.kernel_bytes,binding_bytes);
    }
    if (!profile && !p.identity_units && chart && (plan.maximum&6))
        plan.chart_bytes=program::chart_pullback_bytes(model.program,static_cast<JetRequest>(plan.maximum&6));
    if ((plan.kernel_bytes==0 && program::dispatch_kind(model.program,false)!=1)
        || (!profile && !p.identity_units && chart && (plan.maximum&6) && !plan.chart_bytes)) return plan;
    vlk::experiment::runtime::WorkspaceSizer size;
    const auto n=unit_parameter_count(objective);
    const auto square=profile ? units.layout().get().hessian_elements : model.unit_layout.hessian_elements;
    if (!size.add<double>(n) || !size.add<double>(chart ? n : 0)) return plan;
    if (prepares_density(objective)) {
        const auto enrollment=preparation_plan::make(model.program,plan.internal_mask,chart,selected?&*axes:nullptr);
        if(enrollment.status!=Status::ok)return plan;
        plan.density_mask=enrollment.mask;
        plan.density=DensityPreparation::requirements(plan.density_mask,enrollment.options);
        if (!plan.density.persistent || !size.add<std::max_align_t>(0)
            || !size.add<std::byte>(plan.density.persistent) || !size.add<std::max_align_t>(0)
            || !size.add<std::byte>(plan.density.staging)) return plan;
    }
    for (unsigned order=0;order<3;++order) {
        if (p.identity_units) break;
        if (selected ? (order==0 ? !(cache_mask&1) : order==1 ? !(cache_mask&6) : true) : !(cache_mask&(1u<<order))) continue;
        const auto r=with_unit_plan(objective,units,[&](Status status,const auto& typed_plan) {
            return status==Status::ok?unit_map::requirements(typed_plan,static_cast<JetRequest>(1u<<order))
                :unit_map::Requirements{status};
        });
        if (r.status!=Status::ok || !size.add<std::max_align_t>(0) || !size.add<std::byte>(r.bytes)) return plan;
        plan.cache_bytes[order]=r.bytes;
    }
    if (!p.identity_units && !selected && (!size.add<double>((plan.internal_mask&1) ? 1 : 0)
        || !size.add<double>((plan.internal_mask&2) ? n : 0)
        || !size.add<double>((plan.internal_mask&4) ? square : 0))) return plan;
    if (!profile && !p.identity_units && chart && (!size.add<double>((plan.maximum&1) ? 1 : 0)
        || !size.add<double>((plan.maximum&6) ? n : 0)
        || !size.add<double>((plan.maximum&4) ? square : 0))) return plan;
    if (!size.add<std::max_align_t>(0) || !size.add<std::byte>(plan.kernel_bytes)
        || !size.add<std::max_align_t>(0) || !size.add<std::byte>(plan.chart_bytes)) return plan;
    plan.bytes=size.size();
    plan.status=Status::ok;
    return plan;
}
inline Status bind_unit_workspace(WorkspaceState& state,const UnitWorkPlan& plan,
    std::span<std::byte> ambient_arena) noexcept {
    if (plan.status!=Status::ok || ambient_arena.size()<plan.bytes) return Status::insufficient_capacity;
    const auto& p=Access::get(state.objective.problem());
    const auto& profile=Access::get(state.objective).profile_config;
    const auto& units=profile ? *Access::get(state.objective).profile_units : p.unit_plan;
    const auto n=unit_parameter_count(state.objective);
    const auto square=profile ? units.layout().get().hessian_elements
        : Access::get(p.model).unit_layout.hessian_elements;
    vlk::experiment::runtime::WorkspaceArena arena(ambient_arena.data(),ambient_arena.size());
    state.unit_public_theta={arena.take<double>(n),n};
    if (!state.unit_public_theta.data()) return Status::insufficient_capacity;
    if (state.objective.key().chart)
        state.unit_original_z={arena.take<double>(n),n};
    if (state.objective.key().chart && !state.unit_original_z.data()) return Status::insufficient_capacity;
    if (plan.density.persistent) {
        if (!arena.take<std::max_align_t>(0)) return Status::insufficient_capacity;
        const std::span<std::byte> persistent{arena.take<std::byte>(plan.density.persistent),plan.density.persistent};
        if (!arena.take<std::max_align_t>(0)) return Status::insufficient_capacity;
        const std::span<std::byte> staging{arena.take<std::byte>(plan.density.staging),plan.density.staging};
        const auto status=state.density_preparation.bind(plan.density_mask,persistent,staging,plan.density.options);
        if (status!=Status::ok) return status;
    }
    for (unsigned i=0;i<3;++i) {
        if (!plan.cache_bytes[i]) continue;
        if (!arena.take<std::max_align_t>(0)) return Status::insufficient_capacity;
        auto* memory=arena.take<std::byte>(plan.cache_bytes[i]);
        if (!memory) return Status::insufficient_capacity;
        state.unit_arenas[i]={memory,plan.cache_bytes[i]};
    }
    const auto take=[&](std::size_t count) -> std::span<double> {
        if (!count) return {};
        return {arena.take<double>(count),count};
    };
    if (!p.identity_units && !plan.selected) state.internal_jet={take((plan.internal_mask&1) ? 1 : 0),
        take((plan.internal_mask&2) ? n : 0),take((plan.internal_mask&4) ? square : 0)};
    if (!profile && !p.identity_units && state.objective.key().chart)
        state.fitting_jet={take((plan.maximum&1) ? 1 : 0),
            take((plan.maximum&6) ? n : 0),take((plan.maximum&4) ? square : 0)};
    if (!arena.take<std::max_align_t>(0)) return Status::insufficient_capacity;
    state.unit_kernel_scratch={arena.take<std::byte>(plan.kernel_bytes),plan.kernel_bytes};
    if (!arena.take<std::max_align_t>(0)) return Status::insufficient_capacity;
    state.chart_scratch={arena.take<std::byte>(plan.chart_bytes),plan.chart_bytes};
    return Status::ok;
}
template<unsigned Order> inline auto& unit_slot(WorkspaceState& state) noexcept {
    if constexpr (Order==0) return state.units0;
    else if constexpr (Order==1) return state.units1;
    else return state.units2;
}
template<unsigned Order> inline JetValidity evaluate_units_order(const ObjectiveView& objective,PointView point,
    JetRequest request,WorkspaceState& state,JetBuffers outputs,
    const ActiveSensitivityBasis* basis=nullptr) noexcept {
    const auto& p=Access::get(objective.problem());
    const auto& model=Access::get(p.model);
    return with_unit_plan(objective,p.unit_plan,[&](Status plan_status,const auto& typed_plan) -> JetValidity {
    if(plan_status!=Status::ok)return unit_failure(plan_status,request);
    constexpr bool fixed=!std::is_same_v<std::remove_cvref_t<decltype(typed_plan.layout())>,unit_map::RuntimeLayout>;
    auto& slot=unit_slot<Order>(state);
    if (!state.unit_attempted[Order]) {
        state.unit_attempted[Order]=true;
        slot.emplace();
        state.unit_status[Order]=unit_map::prepare(typed_plan,point.theta(),state.unit_arenas[Order],*slot);
        if (state.unit_status[Order]!=Status::ok) slot.reset();
        if(state.unit_dispatch_counts)
            ++(fixed?state.unit_dispatch_counts->fixed_preparations:state.unit_dispatch_counts->runtime_preparations)[Order];
    }
    if (!slot) return unit_failure(state.unit_status[Order],request);
    const auto view=unit_map::rebind_point(typed_plan.layout(),*slot);
    if(!view)return unit_failure(Status::invalid_argument,request);
    const auto& prepared=*view;
    if(basis) {
        if constexpr(Order==0) {
            auto result=program::evaluate(model.program,program_sample(p),prepared.internal_theta(),false,
                request,state.unit_kernel_scratch,outputs,program_mean_semantics(p),prepared.internal_theta(),
                prepares_density(objective)?&state.density_preparation:nullptr,basis);
            if(result.valid_mask&1) {
                const auto status=unit_map::restore_likelihood(typed_plan,outputs.value[0],outputs.value[0]);
                if(status!=Status::ok){result.valid_mask&=~1u;result.f_reason=status;}
            }
            return vlk::experiment::density_status::complete_validity(result);
        } else if constexpr(Order==1) {
            const selected_units::Point selected(*basis,*slot,point.theta(),state.unit_pullback_counts,state.unit_dispatch_counts);
            return program::evaluate(model.program,program_sample(p),prepared.internal_theta(),false,request,
                state.unit_kernel_scratch,outputs,program_mean_semantics(p),prepared.internal_theta(),
                prepares_density(objective)?&state.density_preparation:nullptr,basis,&selected);
        } else return unit_failure(Status::unsupported_operation,request);
    }
    const bool chart=objective.key().chart!=0;
    const auto mask=static_cast<unsigned>(request);
    const auto theta_request=static_cast<JetRequest>(chart && (mask&4) ? mask|2 : mask);
    const auto inner_request=unit_map::internal_request(typed_plan,theta_request);
    const auto upstream=program::evaluate(model.program,program_sample(p),prepared.internal_theta(),false,
        inner_request,state.unit_kernel_scratch,
        {objective.key().derivative_layout,state.internal_jet.value,state.internal_jet.gradient,
            {state.internal_jet.hessian,model.layout.full_theta_width}}, program_mean_semantics(p),{},
        prepares_density(objective) ? &state.density_preparation : nullptr);
    const unit_map::InternalJet internal{state.internal_jet.value,state.internal_jet.gradient,
        state.internal_jet.hessian,upstream};
    if(state.unit_dispatch_counts)
        ++(fixed?state.unit_dispatch_counts->fixed_pullbacks:state.unit_dispatch_counts->runtime_pullbacks)[Order];
    if (!chart) return unit_map::pullback(prepared,request,internal,
        {outputs.value,outputs.gradient,outputs.hessian.elements},state.unit_pullback_counts);
    const auto mapped=unit_map::pullback(prepared,theta_request,internal,
        {state.fitting_jet.value,state.fitting_jet.gradient,state.fitting_jet.hessian},state.unit_pullback_counts);
    return program::chart_pullback(model.program,program_sample(p),point.original_coordinates(),point.theta(),
        request,state.chart_scratch,
        {state.fitting_jet.value,state.fitting_jet.gradient,state.fitting_jet.hessian,mapped},outputs);
    });
}
inline Status bind_preparation_point(const ObjectiveView& objective,PointView point,
    WorkspaceState& state) noexcept {
    if (!(point.objective()==state.objective.key()) || !(objective.key()==state.objective.key())
        || point.evidence().full_theta_layout!=objective.problem().model().layout().id)
        return Status::invalid_argument;
    const auto theta=point.theta().first(unit_parameter_count(objective));
    const auto z=objective.key().chart ? point.original_coordinates() : std::span<const double>{};
    if (theta.size()!=state.unit_public_theta.size() || z.size()!=state.unit_original_z.size())
        return Status::invalid_argument;
    if (state.unit_point_id!=point.evidence().point_id) {
        state.unit_point_id=point.evidence().point_id;
        std::copy(theta.begin(),theta.end(),state.unit_public_theta.begin());
        std::copy(z.begin(),z.end(),state.unit_original_z.begin());
        state.units0.reset(); state.units1.reset(); state.units2.reset();
        state.unit_attempted.fill(false);
        state.density_preparation.reset(state.unit_public_theta,Access::get(objective.problem()).sample.n_eff);
    } else if ((!theta.empty() && std::memcmp(theta.data(),state.unit_public_theta.data(),theta.size_bytes())!=0)
        || (!z.empty() && std::memcmp(z.data(),state.unit_original_z.data(),z.size_bytes())!=0))
        return Status::invalid_argument;
    return Status::ok;
}
inline JetValidity evaluate_units(const ObjectiveView& objective,PointView point,JetRequest request,
    WorkspaceState& state,JetBuffers outputs,const ActiveSensitivityBasis* basis=nullptr) noexcept {
    const auto mask=static_cast<unsigned>(request);
    if(basis) {
        JetValidity result{Status::ok,request,0};
        if(mask&1) {
            const auto value=evaluate_units_order<0>(objective,point,JetRequest::F,state,outputs,basis);
            result.valid_mask|=value.valid_mask;result.f_reason=value.f_reason;
        }
        if(mask&6) {
            const auto derivatives=evaluate_units_order<1>(objective,point,static_cast<JetRequest>(mask&6),state,outputs,basis);
            result.valid_mask|=derivatives.valid_mask;result.g_reason=derivatives.g_reason;result.h_reason=derivatives.h_reason;
        }
        return vlk::experiment::density_status::complete_validity(result);
    }
    const auto run=[&](JetRequest r) {
        const auto bits=static_cast<unsigned>(r);
        if (bits&4) return evaluate_units_order<2>(objective,point,r,state,outputs);
        if (bits&2) return evaluate_units_order<1>(objective,point,r,state,outputs);
        return evaluate_units_order<0>(objective,point,r,state,outputs);
    };
    auto result=run(request);
    // Failed tighter map preparation never consumes a compatible lower cache.
    // Recover only requested, unavailable lower projections on identical data.
    if (mask&(mask-1)) for (unsigned bit : {1u,2u}) {
        if (!(mask&bit) || (result.valid_mask&bit)) continue;
        const auto recovered=run(static_cast<JetRequest>(bit));
        if (bit==1) result.f_reason=recovered.f_reason; else result.g_reason=recovered.g_reason;
        if (recovered.valid_mask&bit) result.valid_mask|=bit;
    }
    result.requested=request;
    result.valid_mask&=mask;
    if (result.valid_mask==mask) result.status=Status::ok;
    else if ((mask&1) && !(result.valid_mask&1)) result.status=result.f_reason;
    else if ((mask&2) && !(result.valid_mask&2)) result.status=result.g_reason;
    else result.status=result.h_reason;
    return result;
}
}
