// SCIVOL-NATIVE: mean-model mathematics; promotion gates tracked in MODELS.md.
#ifndef SCIVOL_MEAN_EQUATION_HPP
#define SCIVOL_MEAN_EQUATION_HPP

#include "arma_garch_recursion_policy.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

// GNU lambda attributes have different noexcept placement in Clang and GCC.
#if defined(__clang__)
#define VLK_MEAN_LOOP_NOEXCEPT __attribute__((always_inline)) noexcept
#elif defined(__GNUC__)
#define VLK_MEAN_LOOP_NOEXCEPT noexcept __attribute__((always_inline))
#else
#define VLK_MEAN_LOOP_NOEXCEPT noexcept
#endif

namespace vlk::experiment::mean {

inline constexpr std::size_t dynamic = std::numeric_limits<std::size_t>::max();

template<std::size_t Columns = dynamic, std::size_t P = dynamic,
         std::size_t Q = dynamic>
struct LinearVolatilityOrders {
    static constexpr bool linear = true;
    static constexpr bool fixed = Columns != dynamic && P != dynamic && Q != dynamic;
    static constexpr std::size_t columns = Columns;
    static constexpr std::size_t p_ar = 0u, q_ma = 0u;
    static constexpr std::size_t p = P, q = Q, p_vol = P, q_vol = Q;
    static constexpr std::size_t maximum_lag = P > Q ? P : Q;
};

template<std::size_t Static>
constexpr std::size_t extent(std::size_t runtime) noexcept
{
    if constexpr (Static == dynamic)
        return runtime;
    return Static;
}

template<class Layout, const Layout* Fixed = nullptr>
constexpr const Layout& arma_layout(const Layout& supplied) noexcept
{
    if constexpr (Fixed != nullptr) return *Fixed;
    return supplied;
}

template<class Layout>
constexpr bool valid_arma_mean_layout(const Layout& layout) noexcept
{
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (!layout.valid || !layout.joint || layout.ar_start != 1u
        || layout.p_ar == maximum || layout.ma_start != 1u + layout.p_ar
        || layout.q_ma > maximum - layout.ma_start) return false;
    return layout.mean_parameters == layout.ma_start + layout.q_ma
        && layout.mean_parameters <= layout.parameters
        && layout.maximum_lag >= std::max(layout.p_ar, layout.q_ma);
}

// The caller owns conditioning and storage. Scalar selects arithmetic, not equations.
#if defined(_MSC_VER)
#define VLK_MEAN_VALUE_INLINE __forceinline
#else
#define VLK_MEAN_VALUE_INLINE inline __attribute__((always_inline))
#endif

template<class Scalar, class Layout, const Layout* Fixed = nullptr, class Lagged>
VLK_MEAN_VALUE_INLINE Scalar inverse_ma_value(
    Scalar forcing, const Layout& supplied, const double* parameters,
    const Lagged& residual) noexcept
{
    const auto& layout = arma_layout<Layout, Fixed>(supplied);
    for (std::size_t lag = 0; lag < layout.q_ma; ++lag)
        generated::arma_garch_equation::subtract_product(
            forcing, parameters[layout.ma_start + lag], residual(lag));
    return forcing;
}

template<class Scalar, class Layout, const Layout* Fixed = nullptr, class Lagged>
VLK_MEAN_VALUE_INLINE Scalar arma_residual_value(
    const Layout& supplied, const double* parameters, const double* observations,
    std::size_t time, const Lagged& residual) noexcept
{
    const auto& layout = arma_layout<Layout, Fixed>(supplied);
    Scalar value = Scalar(observations[time]) - Scalar(parameters[0]);
    for (std::size_t lag = 0; lag < layout.p_ar; ++lag)
        generated::arma_garch_equation::subtract_product(
            value, parameters[layout.ar_start + lag], observations[time - 1u - lag]);
    return inverse_ma_value<Scalar, Layout, Fixed>(value, layout, parameters, residual);
}

#undef VLK_MEAN_VALUE_INLINE

template<std::size_t P = dynamic>
struct ArmaDesign {
    static constexpr bool linear = false;
    const double* observations;
    std::size_t order;

    std::size_t columns() const noexcept { return 1u + extent<P>(order); }
    double at(std::size_t time, std::size_t column) const noexcept
    {
        return column == 0u ? 1.0 : observations[time - column];
    }
};

template<std::size_t Columns = dynamic, unsigned Interleave = 4>
struct LinearDesign {
    static constexpr bool linear = true;
    static constexpr unsigned sensitivity_interleave = Interleave;
    const double* values;
    std::size_t count;

    std::size_t columns() const noexcept { return extent<Columns>(count); }
    double at(std::size_t time, std::size_t column) const noexcept
    {
        return values[time * columns() + column];
    }
    bool zero_row(std::size_t time) const noexcept
    {
        for (std::size_t column=0;column<columns();++column)
            if (at(time,column)!=0.0) return false;
        return true;
    }
};

struct State {
    double value = 0.0;
    double* first = nullptr;
    double* second = nullptr;
    bool needs_compensation = false;
};

struct History {
    double* values = nullptr;
    double* first = nullptr;
    double* second = nullptr;
    std::size_t slots = 0;
};

template<class Design>
struct LinearSensitivity {
    static constexpr unsigned interleave = Design::sensitivity_interleave;
    Design design;
    std::size_t time;
    std::size_t size() const noexcept { return design.columns(); }
    double operator[](std::size_t column) const noexcept
    { return -design.at(time,column); }
};

template<class Scalar, class Design>
Scalar generic_compensated_residual(const Design& design, const double* parameters,
    double observed, std::size_t time) noexcept
{
    using std::fma;
    Scalar sum=observed, correction=0.0;
    for (std::size_t j=0;j<design.columns();++j) {
        const Scalar a=parameters[j], b=design.at(time,j);
        // The error transform needs a rounded product, even with cross-statement FMA.
        const Scalar product=[&]() {
            if constexpr (std::is_same_v<Scalar,double>) {
                const volatile double rounded_product=a*b;
                return double(rounded_product);
            } else return a*b;
        }();
        const Scalar error=fma(a,b,-product);
        const Scalar next=sum-product;
        const Scalar difference=next-sum;
        correction+=(sum-(next-difference))+(-product-difference)-error;
        sum=next;
    }
    return sum+correction;
}

// Numeric representations may specialize the error transform, not the mean model.
template<class Scalar>
struct ResidualArithmetic {
    template<class Design>
    static Scalar evaluate(const Design& design,const double* parameters,
        double observed,std::size_t time) noexcept {
        return generic_compensated_residual<Scalar>(design,parameters,observed,time);
    }
};

template<class Scalar = double, class Design>
Scalar compensated_residual(const Design& design,const double* parameters,
    double observed,std::size_t time) noexcept {
    return ResidualArithmetic<Scalar>::evaluate(design,parameters,observed,time);
}

template<int Order, std::size_t Q, class Design>
void prepare_linear_state(State& state, std::size_t width) noexcept
{
    if constexpr (Design::linear && Q==0 && Order>=1)
        std::fill_n(state.first,width,0.0);
    if constexpr (Design::linear && Q==0 && Order>=2)
        std::fill_n(state.second,width*(width+1u)/2u,0.0);
}

template<int Order, std::size_t Q = dynamic, std::size_t Width = dynamic,
         bool MaterializeFirst = true, bool Compensate = true, class Design>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline)) inline
#elif defined(_MSC_VER)
__forceinline
#else
inline
#endif
void advance(
    State& current, const History& history, const Design& design,
    const double* parameters, double observed, std::size_t time,
    std::size_t runtime_q, std::size_t runtime_width) noexcept
{
    static_assert(MaterializeFirst || (Order==1 && Design::linear && Q==0));
    using namespace generated::arma_garch_equation;
    const std::size_t width = extent<Width>(runtime_width);
    const std::size_t q = extent<Q>(runtime_q);
    const std::size_t packed = width * (width + 1u) / 2u;
    const std::size_t ma_start = design.columns();
    current.value = observed;
    if constexpr (Order >= 1 && MaterializeFirst && !(Design::linear && Q==0))
        std::fill_n(current.first, width, 0.0);
    if constexpr (Order >= 2 && !(Design::linear && Q==0))
        std::fill_n(current.second, packed, 0.0);
    if constexpr (Design::linear) {
        // Conditional scales can amplify small dot-product errors even away from a kink.
        current.value=compensated_residual(design,parameters,observed,time);
        if constexpr (Order>=1 && MaterializeFirst) {
            for (std::size_t column=0;column<ma_start;++column)
                add_ar_first(current.first,column,design.at(time,column));
        }
    } else {
        for (std::size_t column = 0; column < ma_start; ++column) {
            const double feature = design.at(time, column);
            subtract_product(current.value, parameters[column], feature);
            if constexpr (Order >= 1)
                add_ar_first(current.first, column, feature);
        }
    }
    for (std::size_t lag = 0; lag < q; ++lag) {
        const std::size_t slot = (time - 1u - lag) % (history.slots ? history.slots : q);
        const std::size_t index = ma_start + lag;
        subtract_product(current.value, parameters[index], history.values[slot]);
        if constexpr (Order >= 1) {
            add_ma_first(current.first, width, index, parameters[index],
                         history.values[slot], history.first + slot * width);
        }
        if constexpr (Order >= 2) {
            add_ma_second(current.second, width, index, parameters[index],
                          history.first + slot * width,
                          history.second + slot * packed);
        }
    }
}

template<int Order, std::size_t Q = dynamic, std::size_t Width = dynamic>
inline void retain(
    const State& current, History& history, std::size_t time,
    std::size_t runtime_q, std::size_t runtime_width) noexcept
{
    const std::size_t q = extent<Q>(runtime_q);
    if (q == 0u)
        return;
    const std::size_t width = extent<Width>(runtime_width);
    const std::size_t packed = width * (width + 1u) / 2u;
    const std::size_t slot = time % q;
    history.values[slot] = current.value;
    if constexpr (Order >= 1)
        std::copy_n(current.first, width, history.first + slot * width);
    if constexpr (Order >= 2)
        std::copy_n(current.second, packed, history.second + slot * packed);
}

}  // namespace vlk::experiment::mean
#endif
