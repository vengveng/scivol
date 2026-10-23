#pragma once
#include "../density_dispatch.hpp"

#include "joint_garch_program.hpp"
#include "../mean/design_adapters.hpp"
#include "../gjr_transform_policy.hpp"

namespace scivolapi::detail::joint_math {
namespace native = vlk::experiment::runtime;
namespace mean = vlk::experiment::mean;
namespace generated = vlk::experiment::generated;

template<std::size_t Columns, std::size_t P, std::size_t Q, bool Leverage>
inline constexpr auto fixed_linear_layout = native::joint_garch::linear_layout(Columns, P, Q, Leverage);

static_assert(fixed_linear_layout<6, 3, 2, true>
              == native::joint_garch::linear_layout(6, 3, 2, true));
static_assert(fixed_linear_layout<0, 2, 4, false>
              == native::joint_garch::linear_layout(0, 2, 4, false));

// Every static tuple uses the general builder and the same observation program.
template<class Density, bool Leverage, unsigned Request,
         class Design, std::size_t P = mean::dynamic, std::size_t Q = mean::dynamic,
         std::size_t Width = mean::dynamic, std::size_t QMa = mean::dynamic,
         const native::ArmaGarchLayout* Fixed = nullptr,
         class Basis=scivolapi::detail::IdentitySensitivityBasis,class Borrowed=scivolapi::detail::NoDensityBorrow>
bool run(const native::ArmaGarchLayout& supplied, Design design,
    const double* theta, const double* y, std::size_t n, double h0,
    native::JointSamplePlan sample, void* memory, std::size_t bytes,
    double* f, vlk::experiment::sensitivity_number_t<Basis>* g, vlk::experiment::sensitivity_number_t<Basis>* h, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr,Basis basis={},Borrowed borrowed={}) noexcept
{
    const auto& layout = mean::arma_layout<native::ArmaGarchLayout, Fixed>(supplied);
    const auto required = native::joint_garch::workspace_bytes<Design::linear>(layout, Request,basis);
    if (!required || !memory || bytes < required) return false;
    native::WorkspaceArena arena(memory, bytes);
    native::joint_garch::NumberScratch<vlk::experiment::sensitivity_number_t<Basis>> scratch;
    if (!native::joint_garch::bind_scratch<Design::linear>(arena, layout, Request, scratch,basis)) return false;
    return native::joint_garch::evaluate<Density, Leverage,
        (Request&1)!=0, (Request&2)!=0, (Request&4)!=0,
        Design, P, Q, Width, false, QMa, Fixed,Basis>(
            layout, theta, y, n, h0, scratch, f, g, h, design, nullptr, nullptr, sample, density_report,persistent,basis,borrowed);
}

template<class Density, bool Leverage, unsigned Request,
         std::size_t PAr, std::size_t QMa, std::size_t PVol, std::size_t QVol,
         bool Intercept = true,class Basis=scivolapi::detail::IdentitySensitivityBasis,class Borrowed=scivolapi::detail::NoDensityBorrow>
bool run_fixed_arma(const double* theta, const double* y, std::size_t n,
    double h0, native::JointSamplePlan sample, void* memory, std::size_t bytes,
    double* f, vlk::experiment::sensitivity_number_t<Basis>* g, vlk::experiment::sensitivity_number_t<Basis>* h, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr,Basis basis={},Borrowed borrowed={}) noexcept
{
    constexpr auto* layout = &native::fixed_arma_garch_layout<PAr,QMa,PVol,QVol,Leverage,Intercept>;
    static_assert(layout->valid);
    return run<Density,Leverage,Request,mean::ConditionalArmaDesign<PAr,Intercept>,PVol,QVol,
        layout->parameter_count,QMa,layout>(*layout,{y,PAr,sample.history},theta,y,n,h0,
            sample,memory,bytes,f,g,h, density_report,persistent,basis,borrowed);
}

template<class Density, bool Leverage, unsigned Request,
         std::size_t Columns, std::size_t PVol, std::size_t QVol, bool Intercept,
         class Basis=scivolapi::detail::IdentitySensitivityBasis,class Borrowed=scivolapi::detail::NoDensityBorrow>
bool run_fixed_linear(const double* theta, const double* y, const double* x,
    std::size_t n, double h0, native::JointSamplePlan sample,
    void* memory, std::size_t bytes, double* f, vlk::experiment::sensitivity_number_t<Basis>* g, vlk::experiment::sensitivity_number_t<Basis>* h, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr,Basis basis={},Borrowed borrowed={}) noexcept
{
    constexpr auto* layout = &fixed_linear_layout<Columns,PVol,QVol,Leverage>;
    static_assert(layout->valid && Columns >= std::size_t(Intercept));
    return run<Density,Leverage,Request,mean::ExternalLinearDesign<Columns,Intercept>,
        PVol,QVol,layout->parameter_count,0,layout>(
            *layout,{x,Columns-std::size_t(Intercept)},theta,y,n,h0,
            sample,memory,bytes,f,g,h, density_report,persistent,basis,borrowed);
}

} // namespace scivolapi::detail::joint_math
