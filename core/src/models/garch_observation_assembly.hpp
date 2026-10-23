// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_OBSERVATION_ASSEMBLY_HPP
#define SCIVOL_EXPERIMENT_GARCH_OBSERVATION_ASSEMBLY_HPP

#include "garch_density_policy.hpp"
#include "../active_sensitivity_basis.hpp"
#include "../math/sensitivity_storage.hpp"

#include <cstddef>
#include <cstdint>

namespace vlk::experiment::model {

inline auto normalized_density_derivatives(auto observation) noexcept {
    if(!observation.normalized_assembly) {
        constexpr auto shapes=std::tuple_size_v<std::remove_cvref_t<decltype(observation.shape_gradient)>>;
        observation.normalized=static_cast<const generated::DensityObservationFields<shapes>&>(observation);
        observation.residual_scale=observation.variance_scale=1;
        observation.normalized_assembly=true;
    }
    return observation;
}

inline auto absolute_density_derivatives(auto observation) noexcept {
    const auto make_absolute=[](auto& local) {
        local.composed_residual_slope=std::abs(local.composed_residual_slope);
        for(auto* x:{&local.residual_gradient,&local.variance_gradient,&local.residual_hessian,
            &local.residual_variance_hessian,&local.variance_hessian})*x=std::abs(*x);
        for(auto* values:{&local.shape_gradient,&local.residual_shape_hessian,&local.variance_shape_hessian})
            for(auto& x:*values)x=std::abs(x);
        for(auto& x:local.shape_hessian)x=std::abs(x);
    };
    make_absolute(observation);
    make_absolute(observation.normalized);
    if constexpr (requires { observation.residual_first_factors; }) {
        auto& factor=observation.residual_first_factors;
        factor.numerator=std::abs(factor.numerator);
        factor.gradient=std::abs(factor.gradient);
        factor.variance_hessian=std::abs(factor.variance_hessian);
    }
    return observation;
}

[[nodiscard]] inline bool uses_normalized_assembly(const auto& observation) noexcept
{
    if constexpr (requires { observation.normalized_assembly; })
        return observation.normalized_assembly;
    return false;
}

// Frozen units can be far from the sensitivity's units. Contract all factors
// before rounding an otherwise overflowing or underflowing intermediate.
[[nodiscard]] inline double scaled_derivative_product(
    double coefficient, double sensitivity, double unit) noexcept
{
    if (coefficient == 0.0 || sensitivity == 0.0)
        return 0.0;
    const double normalized = sensitivity / unit;
    const double product = coefficient * normalized;
    if (std::isnormal(normalized) && std::isfinite(product))
        return product;
    int ce = 0, se = 0, ue = 0;
    const double cm = std::frexp(coefficient, &ce);
    const double sm = std::frexp(sensitivity, &se);
    const double um = std::frexp(unit, &ue);
    return std::scalbn((cm * sm) / um, ce + se - ue);
}

[[nodiscard]] inline double scaled_derivative_product(
    double coefficient, double row, double column,
    double row_unit, double column_unit) noexcept
{
    if (coefficient == 0.0 || row == 0.0 || column == 0.0)
        return 0.0;
    const double normalized_row = row / row_unit;
    const double normalized_column = column / column_unit;
    const double first = coefficient * normalized_row;
    const double product = first * normalized_column;
    if (std::isnormal(normalized_row) && std::isnormal(normalized_column)
        && std::isnormal(first) && std::isfinite(product))
        return product;
    int ce = 0, re = 0, ke = 0, rue = 0, cue = 0;
    const double cm = std::frexp(coefficient, &ce);
    const double rm = std::frexp(row, &re);
    const double km = std::frexp(column, &ke);
    const double rum = std::frexp(row_unit, &rue);
    const double cum = std::frexp(column_unit, &cue);
    return std::scalbn(((cm * rm) * km) / (rum * cum),
                       ce + re + ke - rue - cue);
}

template<class Number> requires (!std::is_same_v<Number,double>
    && (std::is_floating_point_v<Number> || requires {Number::flt_min();}))
[[nodiscard]] inline Number scaled_derivative_product(
    double coefficient,const Number& sensitivity,double unit) noexcept
{
    if(coefficient==0 || sensitivity==0)return Number(0);
    const auto normalized=sensitivity/Number(unit),result=Number(coefficient)*normalized;
    if(!normal_sensitivity(normalized) || !normal_sensitivity(result) || normalized==0 || result==0)
        return Number(std::numeric_limits<double>::quiet_NaN());
    return result;
}

template<class Number> requires (!std::is_same_v<Number,double>
    && (std::is_floating_point_v<Number> || requires {Number::flt_min();}))
[[nodiscard]] inline Number scaled_derivative_product(
    double coefficient,const Number& row,const Number& column,double row_unit,double column_unit) noexcept
{
    if(coefficient==0 || row==0 || column==0)return Number(0);
    const auto a=row/Number(row_unit),b=column/Number(column_unit);
    const auto first=Number(coefficient)*a,result=first*b;
    if(!normal_sensitivity(a) || !normal_sensitivity(b) || !normal_sensitivity(first)
        || !normal_sensitivity(result) || a==0 || b==0 || first==0 || result==0)
        return Number(std::numeric_limits<double>::quiet_NaN());
    return result;
}

template<class Number>
[[nodiscard]] inline Number factored_derivative_product(
    double coefficient, const generated::ResidualFirstFactors& factor,
    Number row, Number column=Number(1), double column_unit=1.0) noexcept
{
    const auto invalid=[] {return Number(std::numeric_limits<double>::quiet_NaN());};
    if(!std::isfinite(coefficient) || !std::isfinite(factor.numerator)
        || !std::isfinite(factor.denominator) || !(factor.denominator>0)
        || !std::isfinite(column_unit) || !(column_unit>0)
        || !finite_sensitivity(row) || !finite_sensitivity(column))return invalid();
    if(coefficient==0 || factor.numerator==0 || row==0 || column==0)return Number(0);
    using std::frexp;
    using std::ldexp;
    int ce=0,ne=0,re=0,ke=0,de=0,ue=0;
    const auto cm=std::frexp(coefficient,&ce),nm=std::frexp(factor.numerator,&ne);
    const auto rm=frexp(row,&re),km=frexp(column,&ke);
    const auto dm=std::frexp(factor.denominator,&de),um=std::frexp(column_unit,&ue);
    const auto exponent=std::int64_t(ce)+ne+re+ke-de-ue;
    if(exponent<std::numeric_limits<int>::min() || exponent>std::numeric_limits<int>::max())return invalid();
    const Number mantissa=(((Number(cm)*Number(nm))*rm)*km)/Number(dm)/Number(um);
    const Number result=ldexp(mantissa,static_cast<int>(exponent));
    if(!finite_sensitivity(result) || result==0)return invalid();
    if constexpr(!std::is_same_v<Number,double>)if(!normal_sensitivity(result))return invalid();
    return result;
}

template<bool VarianceCross=false,class Number>
[[nodiscard]] inline Number residual_first_product(const auto& observation,Number first) noexcept
{
    if constexpr(requires {observation.residual_first_factors;}) {
        const auto& factor=observation.residual_first_factors;
        if(factor.numerator!=0)return factored_derivative_product(
            VarianceCross?factor.variance_hessian:factor.gradient,factor,first);
    }
    constexpr auto shapes=std::tuple_size_v<std::remove_cvref_t<decltype(observation.shape_gradient)>>;
    const bool normalized=uses_normalized_assembly(observation);
    const generated::DensityObservationFields<shapes>& local=normalized?observation.normalized:observation;
    return scaled_derivative_product(VarianceCross?local.residual_variance_hessian:local.residual_gradient,
        first,normalized?observation.residual_scale:1.0);
}

template<class Number>
[[nodiscard]] inline Number residual_variance_product(
    const auto& observation,Number residual_first,Number variance_first) noexcept
{
    if constexpr(requires {observation.residual_first_factors;}) {
        const auto& factor=observation.residual_first_factors;
        if(factor.numerator!=0)return factored_derivative_product(factor.variance_hessian,
            factor,residual_first,variance_first,observation.variance_scale);
    }
    constexpr auto shapes=std::tuple_size_v<std::remove_cvref_t<decltype(observation.shape_gradient)>>;
    const bool normalized=uses_normalized_assembly(observation);
    const generated::DensityObservationFields<shapes>& local=normalized?observation.normalized:observation;
    return scaled_derivative_product(local.residual_variance_hessian,residual_first,variance_first,
        normalized?observation.residual_scale:1.0,normalized?observation.variance_scale:1.0);
}

template<class Number>
[[nodiscard]] inline Number compose_normalized_first(
    const auto& observation, Number residual_first, Number variance_first) noexcept
{
    const auto& local = observation.normalized;
    return residual_first_product(observation,residual_first)
        + scaled_derivative_product(local.variance_gradient, variance_first,
                                    observation.variance_scale);
}

template<class Number>
[[nodiscard]] inline Number compose_normalized_second(
    const auto& observation, Number residual_row, Number variance_row,
    Number residual_column, Number variance_column,
    Number residual_second, Number variance_second) noexcept
{
    const auto& local = observation.normalized;
    return scaled_derivative_product(local.residual_hessian,
            residual_row, residual_column,
            observation.residual_scale, observation.residual_scale)
        + residual_variance_product(observation,residual_row,variance_column)
        + residual_variance_product(observation,residual_column,variance_row)
        + scaled_derivative_product(local.variance_hessian,
            variance_row, variance_column,
            observation.variance_scale, observation.variance_scale)
        + compose_normalized_first(observation, residual_second, variance_second);
}

// Contract a complete product before binary64 rounding. Zero is allowed only as
// the final rounded product; invalid factors are checked even when another is zero.
template<std::size_t N,std::size_t D>
[[nodiscard]] inline double scaled_derivative_factors(
    const std::array<double,N>& numerator,const std::array<double,D>& denominator) noexcept
{
    static_assert(N<=8 && D<=4); // Bounded mantissa products and exponent sums.
    for (double x:numerator)
        if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
    for (double x:denominator)
        if (!(x>0.0) || !std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
    double mantissa=1.0;
    int exponent=0;
    for (double x:numerator) {
        int shift=0;
        mantissa*=std::frexp(x,&shift);
        exponent+=shift;
    }
    for (double x:denominator) {
        int shift=0;
        mantissa/=std::frexp(x,&shift);
        exponent-=shift;
    }
    return std::scalbn(mantissa,exponent);
}

struct DefaultGarchCurvature { static constexpr bool affine=false; };

template<class Density>
inline constexpr bool has_log_variance_factors=requires {
    Density::log_variance_factors(1.0,1.0,1.0,true);
};

struct AffineLogCurvature {
    static constexpr bool affine=true;
    generated::LogVarianceFactors factors;
    double remainder;
    const double* remainder_first;
    double unit;
    std::size_t beta_start;

    [[nodiscard]] double compose(std::size_t column,const double* first,
                                 const double* second) const noexcept {
        const double A=first[0], Aj=second[column]; // Packed row zero starts at zero.
        const double Rj=remainder_first[column-beta_start];
        const double determinant=remainder*Aj-A*Rj;
        const double h=factors.variance, a=factors.square_left, b=factors.square_right;
        return scaled_derivative_factors(
            std::array{factors.score_constant,determinant,unit,unit},std::array{h,h})
            +scaled_derivative_factors(
                std::array{factors.score_square,a,b,determinant,unit,unit},std::array{h,h,h})
            +scaled_derivative_factors(
                std::array{factors.curvature_square,a,b,A,first[column],unit,unit},std::array{h,h,h});
    }
};

template<std::size_t StaticVarianceParameters>
[[nodiscard]] inline std::size_t garch_variance_parameter_count(
    std::size_t runtime_count)
{
    if constexpr (StaticVarianceParameters == 0u)
        return runtime_count;
    return StaticVarianceParameters;
}

[[nodiscard]] inline std::size_t packed_upper_index(
    std::size_t width, std::size_t row, std::size_t column)
{
    return row * width - row * (row + 1u) / 2u + column;
}

template<bool EmitGradient, bool EmitHessian, bool DenseHessian,
         class Density, class Prepared, class Basis = scivolapi::detail::IdentitySensitivityBasis>
inline void initialize_density_shape_derivatives(
    const Prepared& prepared,
    std::size_t model_parameters,
    sensitivity_number_t<Basis>* gradient,
    sensitivity_number_t<Basis>* hessian, Basis basis = {})
{
    if constexpr (requires {
        Density::template initialize_shape_derivatives<
            EmitGradient, EmitHessian, DenseHessian>(
                prepared, model_parameters, gradient, hessian, basis);
    }) {
        Density::template initialize_shape_derivatives<
            EmitGradient, EmitHessian, DenseHessian>(
                prepared, model_parameters, gradient, hessian, basis);
    }
}

template<bool EmitValue, std::size_t StaticVarianceParameters, class Density>
inline void accumulate_garch_first(
    std::size_t runtime_variance_parameters,
    const auto& observation,
    const double* variance_first,
    typename Density::ValueAccumulator& objective,
    double* gradient)
{
    const std::size_t variance_parameters =
        garch_variance_parameter_count<StaticVarianceParameters>(
            runtime_variance_parameters);
    if constexpr (EmitValue)
        Density::accumulate_value(objective, observation);
    for (std::size_t parameter = 0; parameter < variance_parameters;
         ++parameter) {
        if (uses_normalized_assembly(observation)) {
            gradient[parameter] += compose_normalized_first(
                observation, 0.0, variance_first[parameter]);
            continue;
        }
        gradient[parameter] += observation.variance_gradient * variance_first[parameter];
    }
    for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape) {
        gradient[variance_parameters + shape] +=
            observation.shape_gradient[shape];
    }
}

template<bool EmitValue, bool EmitGradient, bool DenseHessian,
         std::size_t StaticVarianceParameters, class Density,
         class Curvature = DefaultGarchCurvature>
inline void accumulate_garch_second(
    std::size_t runtime_variance_parameters,
    const auto& observation,
    const double* variance_first,
    const double* variance_second,
    typename Density::ValueAccumulator& objective,
    double* gradient,
    double* hessian, const Curvature& curvature = {})
{
    const std::size_t variance_parameters =
        garch_variance_parameter_count<StaticVarianceParameters>(
            runtime_variance_parameters);
    const std::size_t width =
        variance_parameters + Density::shape_parameters;
    if constexpr (EmitValue)
        Density::accumulate_value(objective, observation);
    if constexpr (EmitGradient) {
        for (std::size_t parameter = 0; parameter < variance_parameters;
             ++parameter) {
            if (uses_normalized_assembly(observation)) {
                gradient[parameter] += compose_normalized_first(
                    observation, 0.0, variance_first[parameter]);
                continue;
            }
            gradient[parameter] += observation.variance_gradient * variance_first[parameter];
        }
        for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape) {
            gradient[variance_parameters + shape] +=
                observation.shape_gradient[shape];
        }
    }

    const auto add_hessian = [&](std::size_t row, std::size_t column,
                                 double value) {
        const std::size_t index = DenseHessian
            ? row * width + column
            : packed_upper_index(width, row, column);
        hessian[index] += value;
    };
    for (std::size_t row = 0; row < variance_parameters; ++row) {
        for (std::size_t column = row; column < variance_parameters; ++column) {
            if constexpr (Curvature::affine) {
                if (row==0 && column>=curvature.beta_start) {
                    add_hessian(row,column,curvature.compose(column,variance_first,variance_second));
                    continue;
                }
            }
            if (uses_normalized_assembly(observation)) {
                add_hessian(row, column, compose_normalized_second(
                    observation, 0.0, variance_first[row], 0.0, variance_first[column],
                    0.0, variance_second[packed_upper_index(variance_parameters, row, column)]));
                continue;
            }
            add_hessian(
                row, column,
                observation.variance_hessian
                    * variance_first[row] * variance_first[column]
                + observation.variance_gradient
                    * variance_second[packed_upper_index(
                        variance_parameters, row, column)]);
        }
        for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape) {
            add_hessian(
                row, variance_parameters + shape,
                uses_normalized_assembly(observation)
                    ? scaled_derivative_product(
                        observation.normalized.variance_shape_hessian[shape],
                        variance_first[row], observation.variance_scale)
                    : observation.variance_shape_hessian[shape] * variance_first[row]);
        }
    }
    for (std::size_t row = 0; row < Density::shape_parameters; ++row) {
        for (std::size_t column = row; column < Density::shape_parameters;
             ++column) {
            const std::size_t density_index =
                packed_upper_index(Density::shape_parameters, row, column);
            add_hessian(
                variance_parameters + row,
                variance_parameters + column,
                observation.shape_hessian[density_index]);
        }
    }
}

}  // namespace vlk::experiment::model

#endif
