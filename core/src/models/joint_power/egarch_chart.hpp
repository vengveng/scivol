#pragma once

#include "layout.hpp"
#include "../power/egarch_transform_policy.hpp"
#include "../mean/mean_map.hpp"

namespace vlk::experiment::runtime::joint_power {

// The power chart owns beta/shape mapping; this adapter composes AR and MA blocks.
struct EgarchPowerChart {
    std::size_t q, parameters, beta_start, model_parameters;
};
template<const EgarchLayout* Fixed>
struct FixedEgarchPowerChart {
    static_assert(Fixed != nullptr && Fixed->valid);
    static constexpr std::integral_constant<std::size_t, Fixed->q_vol> q{};
    static constexpr std::integral_constant<std::size_t, Fixed->parameters> parameters{};
    static constexpr std::integral_constant<std::size_t, Fixed->beta_start> beta_start{};
    static constexpr std::integral_constant<std::size_t, Fixed->model_parameters> model_parameters{};
};

template<const EgarchLayout* Fixed = nullptr>
constexpr auto power_chart(const EgarchLayout& layout) noexcept {
    if constexpr (Fixed != nullptr) return FixedEgarchPowerChart<Fixed>{};
    else return EgarchPowerChart{layout.q_vol,layout.parameters,layout.beta_start,layout.model_parameters};
}

template<const EgarchLayout* Fixed, std::size_t EgarchLayout::* Field>
constexpr auto chart_field(const EgarchLayout& layout) noexcept {
    if constexpr (Fixed != nullptr)
        return std::integral_constant<std::size_t, ((*Fixed).*Field)>{};
    else return layout.*Field;
}

template<const EgarchLayout* Fixed, bool Ma>
struct FixedMeanChartBlock {
    static constexpr std::integral_constant<std::size_t, Ma ? Fixed->ma_start : Fixed->ar_start> start{};
    static constexpr std::integral_constant<std::size_t, Ma ? Fixed->q_ma : Fixed->p_ar> order{};
    // Preserve compatibility until the reflection owner accepts typed blocks.
    constexpr operator mean::Autoregression() const noexcept { return {start,order}; }
};

template<const EgarchLayout* Fixed, bool Ma>
constexpr auto mean_chart_block(const EgarchLayout& layout) noexcept {
    if constexpr (Fixed != nullptr) return FixedMeanChartBlock<Fixed,Ma>{};
    else return mean::Autoregression{Ma ? layout.ma_start : layout.ar_start,Ma ? layout.q_ma : layout.p_ar};
}
struct EgarchChartScratch {
    egarch_transform::Scratch power;
    double* jacobian = nullptr;
    double* column = nullptr;
    double* mean_scratch = nullptr;
};

template<const EgarchLayout* Fixed = nullptr>
constexpr const EgarchLayout& chart_layout(const EgarchLayout& supplied) noexcept {
    if constexpr (Fixed != nullptr) return *Fixed;
    return supplied;
}

template<const EgarchLayout* Fixed = nullptr>
std::size_t egarch_chart_bytes(const EgarchLayout& supplied,std::size_t core,unsigned mask) noexcept {
    const auto& layout=chart_layout<Fixed>(supplied);
    const auto base=egarch_transform::workspace_bytes(power_chart<Fixed>(layout),core,mask);
    const auto mean=mean::map_scratch_count(std::max(layout.p_ar,layout.q_ma),mask);
    std::size_t square;
    WorkspaceSizer size;
    if (!base || !mean || !checked_product(layout.parameters,layout.parameters,square)
        || !size.add<std::byte>(base) || !size.add<double>(mean)
        || ((mask&6) && (!size.add<double>(square) || !size.add<double>(layout.parameters)))) return 0;
    return size.size();
}

template<const EgarchLayout* Fixed = nullptr>
bool bind_egarch_chart(const EgarchLayout& supplied,std::size_t core,unsigned mask,
    void* memory,std::size_t bytes,EgarchChartScratch& scratch) noexcept {
    const auto& layout=chart_layout<Fixed>(supplied);
    const auto base=egarch_transform::workspace_bytes(power_chart<Fixed>(layout),core,mask);
    const auto required=egarch_chart_bytes<Fixed>(layout,core,mask);
    if (!required || bytes<required
        || !egarch_transform::bind_scratch(power_chart<Fixed>(layout),core,mask,memory,base,scratch.power)) return false;
    WorkspaceArena arena(static_cast<std::byte*>(memory)+base,bytes-base);
    scratch.mean_scratch=arena.take<double>(mean::map_scratch_count(std::max(layout.p_ar,layout.q_ma),mask));
    if (mask&6) {
        scratch.jacobian=arena.take<double>(layout.parameters*layout.parameters);
        scratch.column=arena.take<double>(layout.parameters);
    }
    return scratch.mean_scratch && (!(mask&6) || (scratch.jacobian && scratch.column));
}

template<class Shape,const EgarchLayout* Fixed = nullptr>
bool map_egarch_chart(const EgarchLayout& supplied,const double* original_z,bool derivatives,
    EgarchChartScratch& scratch) noexcept {
    const auto& layout=chart_layout<Fixed>(supplied);
    const auto power=power_chart<Fixed>(layout);
    if (!egarch_transform::map_coordinates<Shape>(power,original_z,derivatives,scratch.power)) return false;
    if (derivatives) {
        for (std::size_t column=0;column<layout.parameters;++column) {
            egarch_transform::fill_jacobian_column<Shape>(power,scratch.power,column,scratch.column);
            for (std::size_t row=0;row<layout.parameters;++row)
                scratch.jacobian[row*layout.parameters+column]=scratch.column[row];
        }
    }
    return mean::reflection_block(original_z,scratch.power.theta,derivatives ? scratch.jacobian : nullptr,
        chart_field<Fixed,&EgarchLayout::parameters>(layout),mean_chart_block<Fixed,false>(layout),1,scratch.mean_scratch)
        && mean::reflection_block(original_z,scratch.power.theta,derivatives ? scratch.jacobian : nullptr,
        chart_field<Fixed,&EgarchLayout::parameters>(layout),mean_chart_block<Fixed,true>(layout),-1,scratch.mean_scratch);
}

template<const EgarchLayout* Fixed = nullptr>
void egarch_gradient(const EgarchLayout& supplied,const EgarchChartScratch& scratch,double* output) noexcept {
    const auto& layout=chart_layout<Fixed>(supplied);
    for (std::size_t col=0;col<layout.parameters;++col) {
        double entry=0;
        for (std::size_t row=0;row<layout.parameters;++row)
            entry+=scratch.power.theta_gradient[row]*scratch.jacobian[row*layout.parameters+col];
        output[col]=entry;
    }
}

template<class Shape,const EgarchLayout* Fixed = nullptr>
bool egarch_hessian(const EgarchLayout& supplied,const double* original_z,
    EgarchChartScratch& scratch,double* output) noexcept {
    const auto& layout=chart_layout<Fixed>(supplied);
    // Pull back the power chart with its own original-z curvature first.
    if (!egarch_transform::pullback_hessian<Shape>(power_chart<Fixed>(layout),original_z,scratch.power,output)) return false;
    auto* matrix=scratch.power.theta_hessian;
    std::copy_n(output,layout.parameters*layout.parameters,matrix);
    // Power and mean maps act on disjoint blocks. Replace power J by identity
    // for the second pullback; its curvature is already present in matrix.
    for (std::size_t row=layout.mean_parameters;row<layout.parameters;++row) {
        std::fill_n(scratch.jacobian+row*layout.parameters,layout.parameters,0.0);
        scratch.jacobian[row*layout.parameters+row]=1;
    }
    // Caller-owned dense products preserve every mean/volatility/shape cross block.
    for (std::size_t col=0;col<layout.parameters;++col) {
        for (std::size_t row=0;row<layout.parameters;++row) {
            double entry=0;
            for (std::size_t inner=0;inner<layout.parameters;++inner)
                entry+=matrix[row*layout.parameters+inner]*scratch.jacobian[inner*layout.parameters+col];
            scratch.power.hessian_product[row]=entry;
        }
        for (std::size_t row=0;row<layout.parameters;++row) {
            double entry=0;
            for (std::size_t inner=0;inner<layout.parameters;++inner)
                entry+=scratch.jacobian[inner*layout.parameters+row]*scratch.power.hessian_product[inner];
            output[row*layout.parameters+col]=entry;
        }
    }
    // Mean curvature weights are theta derivatives, unchanged by the power map.
    return (!layout.p_ar || reflection_add_contracted_curvature(original_z,
        chart_field<Fixed,&EgarchLayout::ar_start>(layout),chart_field<Fixed,&EgarchLayout::p_ar>(layout),
        1,scratch.power.theta_gradient,chart_field<Fixed,&EgarchLayout::parameters>(layout),output,scratch.mean_scratch))
        && (!layout.q_ma || reflection_add_contracted_curvature(original_z,
        chart_field<Fixed,&EgarchLayout::ma_start>(layout),chart_field<Fixed,&EgarchLayout::q_ma>(layout),
        -1,scratch.power.theta_gradient,chart_field<Fixed,&EgarchLayout::parameters>(layout),output,scratch.mean_scratch));
}
} // namespace vlk::experiment::runtime::joint_power
