#include "backend.hpp"
#include "models/unit_layout_roles.hpp"
#include "joint_backend.hpp"
#include "chart_contract.hpp"

#include <cmath>

namespace scivolapi::detail::backend {
namespace {
joint_backend::ModelConfig common_garch(ModelConfig c) noexcept {
    joint_backend::ModelConfig out;
    out.kind=joint_backend::Kind::linear;
    out.intercept=false;out.p_vol=c.p;out.q_vol=c.q;out.density=c.density;out.leverage=c.leverage;
    return out;
}
}
namespace {
Status translate(std::int32_t status) noexcept
{
    switch (status) {
    case VLK_JET_OK: return Status::ok;
    case VLK_JET_INVALID_ARGUMENT: return Status::invalid_argument;
    case VLK_JET_INSUFFICIENT_WORKSPACE: return Status::insufficient_capacity;
    case VLK_JET_MATHEMATICAL_DOMAIN_ERROR: return Status::mathematical_domain;
    case VLK_JET_NONDIFFERENTIABLE: return Status::nondifferentiable;
    case VLK_JET_DOMAIN_ERROR: return Status::numerical_range;
    case VLK_JET_UNCERTIFIED_DOMAIN: return Status::unsupported_operation;
    default: return Status::internal_error;
    }
}
} // namespace

std::size_t parameter_count(ModelConfig config) noexcept
{
    return vlk::experiment::runtime::garch_parameter_count(
        config.p, config.q, config.density, config.leverage);
}

int dispatch_kind(ModelConfig config, bool transformed) noexcept
{
    if (parameter_count(config) == 0)
        return -1;
    return config.leverage
        ? vlk_gjr_garch_dispatch_kind(config.p, config.q, config.density, transformed)
        : vlk_garch_dispatch_kind(config.p, config.q, config.density, transformed);
}

std::size_t workspace_bytes(
    ModelConfig config, bool transformed, std::uint32_t request,const ActiveSensitivityBasis* basis,
    const selected_units::Point* unit_basis) noexcept
{
    if(basis) {
        if(transformed) return 0;
        const auto plan=joint_backend::requirements(common_garch(config),Coordinates::theta,
            static_cast<JetRequest>(request),basis,unit_basis);
        return plan.status==Status::ok ? plan.bytes : 0;
    }
    return config.leverage
        ? vlk_gjr_garch_sample_jet_workspace_bytes(
            config.p, config.q, config.density, transformed, request)
        : vlk_garch_sample_jet_workspace_bytes(
            config.p, config.q, config.density, transformed, request);
}

Status map(ModelConfig config, const double* coordinates, double* theta) noexcept
{
    const auto width = parameter_count(config);
    if (width == 0 || coordinates == nullptr || theta == nullptr)
        return Status::invalid_argument;
    for (std::size_t index = 0; index < width; ++index)
        if (!std::isfinite(coordinates[index]))
            return Status::invalid_argument;
    std::int32_t status=VLK_JET_DOMAIN_ERROR;
    const bool ok = config.leverage
        ? vlk::experiment::runtime::gjr_coordinate_forward(
            config.p, config.q, config.density, coordinates, false, theta, nullptr,nullptr,nullptr,&status)
        : vlk::experiment::runtime::garch_coordinate_forward(
            config.p, config.q, config.density, coordinates, theta,&status);
    // Map success establishes representable theta only, not a domain certificate.
    return ok ? Status::ok : translate(status);
}

Status parameter_status(ModelConfig config, const double* theta) noexcept
{
    return translate(vlk::experiment::runtime::garch_parameter_status(
        config.p, config.q, config.density, config.leverage, theta));
}

Status evaluate(
    ModelConfig config, const double* parameters, const double* raw,
    const double* squares, std::size_t n, std::size_t first_scored,
    double h0, bool transformed, std::uint32_t request,
    void* workspace, std::size_t bytes, double* F, double* G, double* H,
    std::uint32_t* valid_mask, vlk::experiment::conditioning::History history, scivolapi::detail::DensityPreparation* persistent,
    scivolapi::detail::squared_shocks::Plan square_plan,
    const ActiveSensitivityBasis* basis,const selected_units::Point* unit_basis) noexcept
{
    if(basis) {
        if(!valid_mask) return Status::invalid_argument;
        *valid_mask=0;
        if(transformed) return Status::unsupported_operation;
        // The selected common recursion must first retain the qualified raw
        // sensitivity units. Never reinterpret a prepared subnormal plan.
        if(square_plan==scivolapi::detail::squared_shocks::Plan::raw_subnormal) {
            if(request&6)return Status::unsupported_operation;
            return translate(vlk_garch_selected_sample_jet_evaluate_prevalidated(
                config.leverage,false,parameters,raw,squares,n,first_scored,h0,
                config.p,config.q,config.density,false,request,workspace,bytes,F,nullptr,nullptr,
                valid_mask,history,persistent,square_plan));
        }
        const auto width=basis->width(parameter_count(config));
        const joint_backend::SampleView sample{{raw,n},{},history.recursion_begin,first_scored,h0,
            history.observations,history.innovations,{}};
        const auto result=joint_backend::evaluate_prevalidated(common_garch(config),sample,
            {parameters,parameter_count(config)},Coordinates::theta,static_cast<JetRequest>(request),
            {static_cast<std::byte*>(workspace),bytes},
            {{F,(request&1)?1u:0u},{G,(request&2)?width:0},{H,(request&4)?width*width:0}},persistent,basis,unit_basis);
        *valid_mask=result.valid_mask;
        return result.status;
    }
    return translate(vlk_garch_selected_sample_jet_evaluate_prevalidated(
        config.leverage, false, parameters, raw, squares, n, first_scored, h0,
        config.p, config.q, config.density, transformed, request,
        workspace, bytes, F, G, H, valid_mask, history,persistent,square_plan));
}

unit_map::Layout unit_layout(ModelConfig c) noexcept {
    return unit_layout_roles::variance(parameter_count(c),0);
}

std::size_t chart_pullback_bytes(ModelConfig c, JetRequest request) noexcept {
    return vlk::experiment::runtime::garch_coordinate_pullback_bytes(
        c.p,c.q,c.density,c.leverage,static_cast<unsigned>(request));
}
Status chart_pullback(ModelConfig c,std::span<const double> z,std::span<const double> theta,
    JetRequest request,std::span<std::byte> work,unit_map::InternalJet input,KernelBuffers output) noexcept {
    const auto n=parameter_count(c);
    if (const auto s=admit_chart(z,theta,request,work,chart_pullback_bytes(c,request),n,input,output); s!=Status::ok) return s;
    const auto mask=static_cast<unsigned>(request);
    return vlk::experiment::runtime::garch_coordinate_pullback(c.p,c.q,c.density,c.leverage,z.data(),
        theta.data(),input.gradient.data(),input.hessian.data(),mask,work.data(),work.size(),
        (mask&2) ? output.gradient.data() : nullptr,(mask&4) ? output.hessian.data() : nullptr)
        ? Status::ok : Status::numerical_range;
}

} // namespace scivolapi::detail::backend
