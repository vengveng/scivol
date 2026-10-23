#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include "runtime_workspace.hpp"

namespace vlk::experiment {
template<class Number> inline Number* take_sensitivities(runtime::WorkspaceArena& arena,std::size_t count) noexcept {
    static_assert(std::is_trivially_destructible_v<Number>);
    auto* result=arena.take<Number>(count);
    if(result) {
        if constexpr(std::is_nothrow_default_constructible_v<Number>)
            std::uninitialized_default_construct_n(result,count);
        else {
            static_assert(std::is_nothrow_copy_constructible_v<Number>
                && std::is_trivially_copyable_v<Number>);
            // The wide type's default initializer is constexpr but not noexcept.
            static constexpr Number initial{};
            std::uninitialized_fill_n(result,count,initial);
        }
    }
    return result;
}
template<class Basis,class=void> struct SensitivityNumber {using type=double;};
template<class Basis> struct SensitivityNumber<Basis,std::void_t<typename Basis::number_type>> {
    using type=typename Basis::number_type;
};
template<class Basis> using sensitivity_number_t=typename SensitivityNumber<Basis>::type;
template<class Number> inline Number sensitivity_magnitude(const Number& x) noexcept {
    using std::fabs;return fabs(x);
}
template<class Number> inline bool finite_sensitivity(const Number& x) noexcept {
    if constexpr(std::is_arithmetic_v<Number>)return std::isfinite(x);
    else return finite(x);
}
template<class Number> inline bool normal_sensitivity(const Number& x) noexcept {
    if(x==0)return true;
    if(!finite_sensitivity(x))return false;
    if constexpr(std::is_floating_point_v<Number>)return sensitivity_magnitude(x)>=std::numeric_limits<Number>::min();
    else return sensitivity_magnitude(x)>=Number::flt_min();
}
template<class Basis> inline void sensitivity_operations(Basis basis,std::size_t count) noexcept {
    if constexpr(requires {basis.record_operations(count);})basis.record_operations(count);
}
template<class Basis> inline void sensitivity_operations(Basis basis,std::size_t count,std::size_t each) noexcept {
    if constexpr(requires {basis.record_operations(count);}) {
        if(each && count>std::numeric_limits<std::size_t>::max()/each)basis.invalidate_arithmetic();
        else basis.record_operations(count*each);
    }
}
template<class Basis,class Number> inline Number sensitivity_product(Basis basis,Number a,Number b) noexcept {
    if constexpr(requires {basis.product(a,b);})return basis.product(a,b);
    else return a*b;
}
template<class Basis> inline constexpr bool curved_mean_units=[] {
    if constexpr(requires {Basis::mean_curvature;})return Basis::mean_curvature;
    return Basis::mapped;
}();
} // namespace vlk::experiment
