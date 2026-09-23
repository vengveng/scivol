// SCIVOL-NATIVE: mean-model mathematics; promotion gates tracked in MODELS.md.
#ifndef SCIVOL_MEAN_MAP_HPP
#define SCIVOL_MEAN_MAP_HPP

#include "constant_mean_jet.hpp"
#include "garch_transform_policy.hpp"
#include "runtime_reflection.hpp"

namespace vlk::experiment::mean {

struct Autoregression {
    std::size_t start;
    std::size_t order;
};

template<class Shape>
void coordinate_bounds(const Layout& plan, Autoregression ar,
                       double* lower, double* upper) noexcept
{
    const auto n = plan.width+Shape::parameters;
    const double inf = std::numeric_limits<double>::infinity();
    std::fill_n(lower,n,-inf);
    std::fill_n(upper,n,inf);
    const double limit = std::atanh(std::nextafter(1.0,0.0));
    for (const auto block : {ar, Autoregression{plan.columns,plan.ma_order}})
        for (std::size_t i=0; i<block.order; ++i) {
            lower[block.start+i] = -limit;
            upper[block.start+i] = limit;
        }
    lower[plan.width-1] = generated::inverse_softplus(std::nextafter(0.0,inf));
    Shape::coordinate_bounds(lower+plan.width,upper+plan.width);
}

inline std::size_t map_scratch_count(std::size_t order, unsigned request) noexcept
{
    order = std::max(order, std::size_t{1});
    if (!(request & 6u)) {
        std::size_t total;
        return runtime::checked_product(order, 2u, total) ? total : 0u;
    }
    std::size_t block = (request & 4u)
        ? runtime::reflection_curvature_scratch_doubles(order)
        : runtime::reflection_forward_scratch_doubles(order);
    std::size_t base, square, total;
    if (!block || !runtime::checked_product(order, order, square)
        || !runtime::checked_sum(order, square, base)
        || !runtime::checked_sum(base, block, total))
        return 0;
    return total;
}

inline bool reflection_block(
    const double* z, double* theta, double* jacobian,
    std::size_t width, Autoregression block, double sign, double* scratch) noexcept
{
    if (block.order == 0)
        return true;
    double* coefficients = scratch;
    double* local_jacobian = coefficients + block.order;
    double* work = jacobian
        ? local_jacobian + block.order * block.order : local_jacobian;
    for (std::size_t j = 0; j < block.order; ++j)
        theta[block.start + j] = std::tanh(z[block.start + j]);
    const bool valid = jacobian
        ? runtime::reflection_forward(theta + block.start, block.order,
                                      coefficients, local_jacobian, work)
        : runtime::reflection_forward_values(theta + block.start, block.order,
                                             coefficients, work);
    if (!valid)
        return false;
    if (jacobian) {
        for (std::size_t row = 0; row < block.order; ++row) {
            for (std::size_t col = 0; col < block.order; ++col) {
                const double reflection = theta[block.start + col];
                jacobian[(block.start + row)*width + block.start + col] =
                    sign*local_jacobian[row*block.order + col]
                    * (1.0-reflection*reflection);
            }
        }
    }
    for (std::size_t j = 0; j < block.order; ++j)
        theta[block.start + j] = sign*coefficients[j];
    return true;
}

template<class Shape>
bool map_inverse(const Layout& plan, Autoregression ar, const double* theta,
                 double* z, double* scratch) noexcept
{
    if (ar.order && (ar.start > plan.columns || ar.order > plan.columns-ar.start))
        return false;
    std::copy_n(theta, plan.width + Shape::parameters, z);
    for (const auto block : {ar, Autoregression{plan.columns, plan.ma_order}}) {
        if (!block.order)
            continue;
        const double sign = block.start == plan.columns ? -1.0 : 1.0;
        if (!runtime::reflection_inverse_values(theta+block.start, block.order,
                sign, z+block.start, scratch, scratch+block.order))
            return false;
        for (std::size_t i=0; i<block.order; ++i)
            z[block.start+i] = std::atanh(z[block.start+i]);
    }
    if (!(theta[plan.width-1] > 0.0))
        return false;
    z[plan.width-1] = generated::inverse_softplus(theta[plan.width-1]);
    return Shape::unpack(theta+plan.width, z+plan.width);
}

template<class Shape>
bool map_forward(const Layout& plan, Autoregression ar, const double* z,
                 double* theta, double* jacobian, double* scratch) noexcept
{
    const std::size_t n = plan.width + Shape::parameters;
    if (ar.order && (ar.start > plan.columns || ar.order > plan.columns-ar.start))
        return false;
    std::copy_n(z, n, theta);
    if (jacobian) {
        std::fill_n(jacobian, n*n, 0.0);
        for (std::size_t j = 0; j < n; ++j)
            jacobian[j*n+j] = 1.0;
    }
    if (!reflection_block(z, theta, jacobian, n, ar, 1.0, scratch)
        || !reflection_block(z, theta, jacobian, n,
                             {plan.columns, plan.ma_order}, -1.0, scratch))
        return false;
    const std::size_t scale = plan.width-1;
    theta[scale] = generated::softplus(z[scale]);
    if (!(theta[scale] > 0) || !std::isfinite(theta[scale]))
        return false;
    Shape::pack(z+plan.width, theta+plan.width);
    if (jacobian) {
        jacobian[scale*n+scale] =
            generated::softplus_first_from_positive(theta[scale]);
        Shape::fill_jacobian(theta+plan.width, jacobian, n, plan.width);
    }
    return true;
}

template<class Shape>
bool map_pullback(
    const Layout& plan, Autoregression ar, const double* z, const double* theta,
    const double* jacobian, const double* g, const double* h,
    double* gz, double* hz, double* scratch, double* intermediate) noexcept
{
    const std::size_t n = plan.width + Shape::parameters;
    if (gz) {
        for (std::size_t col = 0; col < n; ++col) {
            double sum = 0;
            for (std::size_t row = 0; row < n; ++row)
                sum += g[row]*jacobian[row*n+col];
            gz[col] = sum;
        }
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
    if (ar.order && !runtime::reflection_add_contracted_curvature(
            z, ar.start, ar.order, 1.0, g, n, hz, scratch))
        return false;
    if (plan.ma_order && !runtime::reflection_add_contracted_curvature(
            z, plan.columns, plan.ma_order, -1.0, g, n, hz, scratch))
        return false;
    const std::size_t scale = plan.width-1;
    hz[scale*n+scale] += g[scale]
        * generated::softplus_second_from_positive(theta[scale]);
    Shape::add_curvature(theta+plan.width, g+plan.width, hz, n, plan.width);
    return true;
}

}  // namespace vlk::experiment::mean
#endif
