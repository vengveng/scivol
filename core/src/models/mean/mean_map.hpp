// SCIVOL-NATIVE: mean-model mathematics; promotion gates tracked in MODELS.md.
#ifndef SCIVOL_MEAN_MAP_HPP
#define SCIVOL_MEAN_MAP_HPP

#include "constant_mean_jet.hpp"
#include "../garch_transform_policy.hpp"
#include "../../math/mean/runtime_reflection.hpp"
#include "../../math/softplus_pullback.hpp"
#include <type_traits>

namespace vlk::experiment::mean {

struct Autoregression {
    std::size_t start;
    std::size_t order;
};

// Tags retain fixed extents across outlined shared-map and reflection helpers.
template<auto Fixed, auto Member, std::size_t Extra = 0, class Metadata>
constexpr auto map_extent_tag(const Metadata& supplied) noexcept
{
    if constexpr (Fixed != nullptr)
        return std::integral_constant<std::size_t, ((*Fixed).*Member) + Extra> {};
    else
        return (supplied.*Member) + Extra;
}

template<const Layout* Fixed, const Autoregression* FixedAr>
inline constexpr std::size_t static_map_order = [] {
    if constexpr (Fixed != nullptr && FixedAr != nullptr)
        return std::max(Fixed->ma_order, FixedAr->order);
    else
        return dynamic;
}();

template<class Shape, const Layout* Fixed = nullptr,
         const Autoregression* FixedAr = nullptr>
void coordinate_bounds(const Layout& supplied, Autoregression supplied_ar,
                       double* lower, double* upper) noexcept
{
    static_assert((Fixed == nullptr) == (FixedAr == nullptr));
    const auto& plan = arma_layout<Layout, Fixed>(supplied);
    const auto& ar = arma_layout<Autoregression, FixedAr>(supplied_ar);
    const auto n = plan.width+Shape::parameters;
    const double inf = std::numeric_limits<double>::infinity();
    std::fill_n(lower,n,-inf);
    std::fill_n(upper,n,inf);
    const double limit = std::atanh(std::nextafter(1.0,0.0));
    const auto block_bounds = [&](auto start_tag, auto order_tag) {
        const std::size_t start = start_tag, order = order_tag;
        for (std::size_t i=0; i<order; ++i) {
            lower[start+i] = -limit;
            upper[start+i] = limit;
        }
    };
    block_bounds(map_extent_tag<FixedAr, &Autoregression::start>(ar),
                 map_extent_tag<FixedAr, &Autoregression::order>(ar));
    block_bounds(map_extent_tag<Fixed, &Layout::columns>(plan),
                 map_extent_tag<Fixed, &Layout::ma_order>(plan));
    lower[plan.width-1] = generated::inverse_softplus(std::nextafter(0.0,inf));
    Shape::coordinate_bounds(lower+plan.width,upper+plan.width);
}

template<std::size_t StaticOrder = dynamic>
inline std::size_t map_scratch_count(std::size_t supplied_order, unsigned request) noexcept
{
    const auto order = std::max(extent<StaticOrder>(supplied_order), std::size_t{1});
    const auto order_tag = [&] {
        if constexpr (StaticOrder != dynamic)
            return std::integral_constant<std::size_t, std::max(StaticOrder, std::size_t{1})> {};
        else
            return order;
    }();
    if (!(request & 6u)) {
        std::size_t total;
        return runtime::checked_product(order, 2u, total) ? total : 0u;
    }
    std::size_t block = (request & 4u)
        ? runtime::reflection_curvature_scratch_doubles(order_tag)
        : runtime::reflection_forward_scratch_doubles(order_tag);
    std::size_t base, square, total;
    if (!block || !runtime::checked_product(order, order, square)
        || !runtime::checked_sum(order, square, base)
        || !runtime::checked_sum(base, block, total))
        return 0;
    return total;
}

template<class Dimension, class Start, class Order>
inline bool reflection_block(
    const double* z, double* theta, double* jacobian,
    Dimension width_tag, Start start_tag, Order order_tag,
    double sign, double* scratch) noexcept
{
    const std::size_t width = width_tag, start = start_tag, order = order_tag;
    if (order == 0)
        return true;
    double* coefficients = scratch;
    double* local_jacobian = coefficients + order;
    double* work = jacobian
        ? local_jacobian + order * order : local_jacobian;
    for (std::size_t j = 0; j < order; ++j)
        theta[start + j] = std::tanh(z[start + j]);
    const bool valid = jacobian
        ? runtime::reflection_forward(theta + start, order_tag,
                                      coefficients, local_jacobian, work)
        : runtime::reflection_forward_values(theta + start, order_tag,
                                             coefficients, work);
    if (!valid)
        return false;
    if (jacobian) {
        for (std::size_t row = 0; row < order; ++row) {
            for (std::size_t col = 0; col < order; ++col) {
                jacobian[(start + row)*width + start + col] =
                    sign*local_jacobian[row*order + col]
                    * ad::coordinate_tanh_first(z[start + col]);
            }
        }
    }
    for (std::size_t j = 0; j < order; ++j)
        theta[start + j] = sign*coefficients[j];
    return true;
}

template<class Dimension = std::size_t, class Block = Autoregression>
inline bool reflection_block(
    const double* z, double* theta, double* jacobian,
    Dimension width, Block block, double sign, double* scratch) noexcept
{
    return reflection_block(z, theta, jacobian, width, block.start, block.order,
                            sign, scratch);
}

template<class Shape, const Layout* Fixed = nullptr,
         const Autoregression* FixedAr = nullptr>
bool map_inverse(const Layout& supplied, Autoregression supplied_ar, const double* theta,
                 double* z, double* scratch) noexcept
{
    static_assert((Fixed == nullptr) == (FixedAr == nullptr));
    const auto& plan = arma_layout<Layout, Fixed>(supplied);
    const auto& ar = arma_layout<Autoregression, FixedAr>(supplied_ar);
    if (ar.order && (ar.start > plan.columns || ar.order > plan.columns-ar.start))
        return false;
    std::copy_n(theta, plan.width + Shape::parameters, z);
    const auto inverse_block = [&](auto start_tag, auto order_tag, double sign) {
        const std::size_t start = start_tag, order = order_tag;
        if (!order)
            return true;
        if (!runtime::reflection_inverse_values(theta+start, order_tag,
                sign, z+start, scratch, scratch+order))
            return false;
        for (std::size_t i=0; i<order; ++i)
            z[start+i] = std::atanh(z[start+i]);
        return true;
    };
    if (!inverse_block(map_extent_tag<FixedAr, &Autoregression::start>(ar),
                       map_extent_tag<FixedAr, &Autoregression::order>(ar),
                       ar.start == plan.columns ? -1.0 : 1.0)
        || !inverse_block(map_extent_tag<Fixed, &Layout::columns>(plan),
                          map_extent_tag<Fixed, &Layout::ma_order>(plan), -1.0))
        return false;
    if (!(theta[plan.width-1] > 0.0))
        return false;
    z[plan.width-1] = generated::inverse_softplus(theta[plan.width-1]);
    return Shape::unpack(theta+plan.width, z+plan.width);
}

template<class Shape, const Layout* Fixed = nullptr,
         const Autoregression* FixedAr = nullptr>
bool map_forward(const Layout& supplied, Autoregression supplied_ar, const double* z,
                 double* theta, double* jacobian, double* scratch) noexcept
{
    static_assert((Fixed == nullptr) == (FixedAr == nullptr));
    const auto& plan = arma_layout<Layout, Fixed>(supplied);
    const auto& ar = arma_layout<Autoregression, FixedAr>(supplied_ar);
    const auto n_tag = map_extent_tag<Fixed, &Layout::width, Shape::parameters>(plan);
    const auto width_tag = map_extent_tag<Fixed, &Layout::width>(plan);
    const std::size_t n = n_tag;
    if (ar.order && (ar.start > plan.columns || ar.order > plan.columns-ar.start))
        return false;
    std::copy_n(z, n, theta);
    if (jacobian) {
        std::fill_n(jacobian, n*n, 0.0);
        for (std::size_t j = 0; j < n; ++j)
            jacobian[j*n+j] = 1.0;
    }
    if (!reflection_block(z, theta, jacobian, n_tag,
            map_extent_tag<FixedAr, &Autoregression::start>(ar),
            map_extent_tag<FixedAr, &Autoregression::order>(ar), 1.0, scratch)
        || !reflection_block(z, theta, jacobian, n_tag,
            map_extent_tag<Fixed, &Layout::columns>(plan),
            map_extent_tag<Fixed, &Layout::ma_order>(plan), -1.0, scratch))
        return false;
    const std::size_t scale = plan.width-1;
    theta[scale] = generated::softplus(z[scale]);
    if (!(theta[scale] > 0) || !std::isfinite(theta[scale]))
        return false;
    Shape::pack(z+plan.width, theta+plan.width);
    if (jacobian) {
        jacobian[scale*n+scale] =
            positive_coordinate::ScaledLogistic(z[scale]).value();
        Shape::fill_jacobian(z+plan.width, jacobian, n_tag, width_tag);
    }
    return true;
}

template<class Shape, const Layout* Fixed = nullptr,
         const Autoregression* FixedAr = nullptr>
bool map_pullback(
    const Layout& supplied, Autoregression supplied_ar, const double* z, const double*,
    const double* jacobian, const double* g, const double* h,
    double* gz, double* hz, double* scratch, double* intermediate) noexcept
{
    static_assert((Fixed == nullptr) == (FixedAr == nullptr));
    const auto& plan = arma_layout<Layout, Fixed>(supplied);
    const auto& ar = arma_layout<Autoregression, FixedAr>(supplied_ar);
    const auto n_tag = map_extent_tag<Fixed, &Layout::width, Shape::parameters>(plan);
    const auto width_tag = map_extent_tag<Fixed, &Layout::width>(plan);
    const std::size_t n = n_tag;
    const auto scale_tag = [&] {
        if constexpr (Fixed != nullptr)
            return std::integral_constant<std::size_t, Fixed->width-1> {};
        else return plan.width-1;
    }();
    const std::size_t scale = scale_tag;
    const positive_coordinate::ScaledLogistic scale_point(z[scale]);
    if (gz) {
        for (std::size_t col = 0; col < n; ++col) {
            double sum = 0;
            for (std::size_t row = 0; row < n; ++row)
                sum += g[row]*jacobian[row*n+col];
            gz[col] = sum;
        }
        gz[scale] = scale_point.value(g[scale]);
    }
    if (!hz)
        return true;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            double sum = 0;
            for (std::size_t j = 0; j < n; ++j)
                sum += h[row*n+j]*jacobian[j*n+col];
            intermediate[row*n+col] = sum;
        }
    }
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            double sum = 0;
            for (std::size_t j = 0; j < n; ++j)
                sum += jacobian[j*n+row]*intermediate[j*n+col];
            hz[row*n+col] = sum;
        }
    }
    positive_coordinate::softplus_hessian_row(z[scale],n_tag,scale_tag,h,jacobian,hz);
    if constexpr (FixedAr == nullptr || FixedAr->order != 0) {
        if (ar.order && !runtime::reflection_add_contracted_curvature(
                z, map_extent_tag<FixedAr, &Autoregression::start>(ar),
                map_extent_tag<FixedAr, &Autoregression::order>(ar), 1.0, g, n_tag, hz, scratch))
            return false;
    }
    if constexpr (Fixed == nullptr || Fixed->ma_order != 0) {
        if (plan.ma_order && !runtime::reflection_add_contracted_curvature(
                z, map_extent_tag<Fixed, &Layout::columns>(plan),
                map_extent_tag<Fixed, &Layout::ma_order>(plan), -1.0, g, n_tag, hz, scratch))
            return false;
    }
    hz[scale*n+scale] += scale_point.first(g[scale]);
    Shape::add_curvature(z+plan.width, g+plan.width, hz, n_tag, width_tag);
    return true;
}

}  // namespace vlk::experiment::mean
#endif
