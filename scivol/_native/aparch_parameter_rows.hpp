// SCIVOL-NATIVE: PROTOTYPE; adapter, no coordinate-map equations.
#pragma once
#include "aparch_runtime_jet.hpp"
#include "mapped_parameter_equalities.hpp"
#include <bit>

namespace vlk::experiment::runtime::aparch_transform {

template<class Density,class Shape,class Orders> class ParameterRows {
    static constexpr const aparch::Layout* fixed=Orders::fixed ? &aparch::fixed_layout<Orders,Density,true> : nullptr;
    static constexpr std::size_t unused_core_bytes=16;
    const aparch::Layout& layout_;
    void* memory_;
    std::size_t bytes_;
    Scratch scratch_{};
    bool intrinsic_=false;
    double omega_=0.,power_=0.,nu_=0.,omega_first_=0.,power_first_=0.,power_tail_=0.;
    double omega_coordinate_=0.,power_coordinate_=0.,tail_coordinate_=0.;
    bool omega_valid_=false,power_valid_=false,omega_first_valid_=false,power_first_valid_=false,power_tail_valid_=false;
    std::size_t value_preparations_=0,first_preparations_=0,reuses_=0;
    const aparch::Layout& metadata() const noexcept { return evaluation_layout<fixed>(layout_); }
    static bool same(double left,double right) noexcept {
        return std::bit_cast<std::uint64_t>(left)==std::bit_cast<std::uint64_t>(right);
    }
public:
    ParameterRows(const aparch::Layout& layout,void* memory,std::size_t bytes) noexcept
        :layout_(evaluation_layout<fixed>(layout)),memory_(memory),bytes_(bytes) {}
    std::size_t dimension() const noexcept { return metadata().parameters; }
    std::array<std::size_t,3> preparation_counts() const noexcept {
        return {value_preparations_,first_preparations_,reuses_};
    }
    static std::size_t bytes(const aparch::Layout& layout,bool derivative) noexcept {
        return workspace_bytes<fixed>(layout,unused_core_bytes,derivative?3u:1u);
    }
    static bool retarget_power(const aparch::Layout& supplied,double* point,double power) noexcept {
        const auto& layout=evaluation_layout<fixed>(supplied);
        using Tail=generated::ShiftedSoftplusMap<generated::StudentTShapeLowerBound>;
        double nu=0.,coordinate=0.;
        if constexpr (tail_coupled_delta<Density>) nu=Tail::forward(point[layout.model_parameters]);
        if (!PowerCoordinate<Density>::inverse(power,nu,coordinate)) return false;
        point[layout.delta_index]=coordinate;
        return true;
    }
    bool prepare(const double* point,bool derivative) noexcept {
        intrinsic_=false;
        if (!point || !memory_ || bytes_<bytes(metadata(),derivative)
            || !bind_scratch<fixed>(metadata(),unused_core_bytes,derivative?3u:1u,memory_,bytes_,scratch_)) return false;
        return derivative ? map_coordinates<1,Density,Shape,fixed>(metadata(),point,scratch_)
                          : map_coordinates<0,Density,Shape,fixed>(metadata(),point,scratch_);
    }
    bool prepare(const double* point,bool derivative,const std::size_t* selected,
                 std::size_t rows,ConstraintColumns columns) noexcept {
        if (!point || !selected || !rows) return false;
        const auto& layout=metadata();
        bool omega=false,power=false;
        for (std::size_t i=0;i<rows;++i) {
            if (selected[i]==layout.omega_index) omega=true;
            else if (selected[i]==layout.delta_index) power=true;
            else return prepare(point,derivative);
        }
        intrinsic_=true;
        if (omega) {
            if (!omega_valid_ || !same(omega_coordinate_,point[layout.omega_index])) {
                omega_valid_=omega_first_valid_=false;
                omega_coordinate_=point[layout.omega_index];
                ++value_preparations_;
                request_accounting::prepare(request_accounting::Preparation::parameter_row,0u);
                omega_=PositiveCoordinate::value(omega_coordinate_);
                if (!std::isfinite(omega_) || omega_<=0.) return false;
                omega_valid_=true;
            } else ++reuses_;
            if (derivative && columns.contains(layout.omega_index) && !omega_first_valid_) {
                ++first_preparations_;
                request_accounting::prepare(request_accounting::Preparation::parameter_row,1u);
                omega_first_=PositiveCoordinate::first(omega_);omega_first_valid_=true;
            }
        }
        if (power) {
            using Power=PowerCoordinate<Density>;
            using Tail=generated::ShiftedSoftplusMap<generated::StudentTShapeLowerBound>;
            const double tail=tail_coupled_delta<Density>?point[layout.model_parameters]:0.;
            if (!power_valid_ || !same(power_coordinate_,point[layout.delta_index]) || !same(tail_coordinate_,tail)) {
                power_valid_=power_first_valid_=power_tail_valid_=false;
                power_coordinate_=point[layout.delta_index];tail_coordinate_=tail;
                ++value_preparations_;
                request_accounting::prepare(request_accounting::Preparation::parameter_row,0u);
                nu_=tail_coupled_delta<Density>?Tail::forward(tail):0.;
                power_=Power::value(power_coordinate_,nu_);
                if (!Power::valid(power_,nu_)) return false;
                power_valid_=true;
            } else ++reuses_;
            if (derivative && columns.contains(layout.delta_index) && !power_first_valid_) {
                ++first_preparations_;
                request_accounting::prepare(request_accounting::Preparation::parameter_row,1u);
                power_first_=Power::first(power_coordinate_,power_,nu_);power_first_valid_=true;
            }
            if constexpr (tail_coupled_delta<Density>)
                if (derivative && columns.contains(layout.model_parameters) && !power_tail_valid_) {
                    ++first_preparations_;
                    request_accounting::prepare(request_accounting::Preparation::parameter_row,1u);
                    power_tail_=Power::first_tail(power_coordinate_,Tail::first_at(tail));power_tail_valid_=true;
                }
        }
        return true;
    }
    double value(std::size_t row) const noexcept {
        return intrinsic_ ? (row==metadata().omega_index?omega_:power_) : scratch_.theta[row];
    }
    double derivative(std::size_t row,std::size_t column) const noexcept {
        const auto& layout=metadata();
        if (!intrinsic_) return scratch_.jacobian[row*layout.parameters+column];
        if (row==layout.omega_index) return column==row?omega_first_:0.;
        if (column==layout.delta_index) return power_first_;
        if constexpr (tail_coupled_delta<Density>)
            if (column==layout.model_parameters) return power_tail_;
        return 0.;
    }
};

} // namespace vlk::experiment::runtime::aparch_transform
