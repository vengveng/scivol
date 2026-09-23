// SCIVOL-NATIVE: mean-model mathematics; promotion gates tracked in MODELS.md.
#ifndef SCIVOL_MEAN_SOLVER_HPP
#define SCIVOL_MEAN_SOLVER_HPP

#include "mean_map.hpp"
#include "runtime_kraft_slsqp.hpp"
#include "stability_domain.hpp"
#include "diagonal_optimizer_scaling.hpp"
#include "mean_parameter_domain.hpp"

namespace vlk::experiment::mean {

struct PolynomialConstraintPlan {
    PolynomialDomain domain {0,0,nullptr};
    std::size_t offset = 0;
};

struct MeanConstraintPlan {
    PolynomialConstraintPlan first, second;
    std::size_t count() const noexcept
    { return 2*(first.domain.order+second.domain.order); }
    std::size_t max_order() const noexcept
    { return std::max(first.domain.order,second.domain.order); }
    std::size_t jacobian_size() const noexcept
    { return 2*std::max(first.domain.order*first.domain.parameters,
                        second.domain.order*second.domain.parameters); }
    std::size_t workspace_bytes() const noexcept
    { return std::max(first.domain.workspace_bytes(true),second.domain.workspace_bytes(true)); }
};

inline MeanConstraintPlan constraint_plan(const Layout& plan, Autoregression ar,
    PolynomialConstraintPlan polynomial, bool transformed) noexcept
{
    if (transformed || polynomial.domain.order) return {polynomial,{}};
    return {{{ar.order,ar.order,nullptr},ar.start},
            {{plan.ma_order,plan.ma_order,nullptr,-1.0},plan.columns}};
}

inline std::size_t solve_workspace_bytes(
    const Layout& plan, Autoregression ar, std::size_t shapes,
    PolynomialConstraintPlan polynomial = {}, bool transformed=true) noexcept
{
    std::size_t n, square;
    if (!plan.valid || !runtime::checked_sum(plan.width, shapes, n)
        || !runtime::checked_product(n, n, square))
        return 0;
    const auto map_count = map_scratch_count(std::max(ar.order, plan.ma_order), 3);
    const auto kernel_bytes = workspace_bytes(plan, 3);
    std::size_t constraints, local_jacobian, orders;
    if (!runtime::checked_sum(ar.order,plan.ma_order,orders)
        || !runtime::checked_sum(orders,polynomial.domain.order,orders)
        || !runtime::checked_product(orders,2u,constraints)
        || !runtime::checked_product(constraints,n,local_jacobian))
        return 0;
    const auto domains=constraint_plan(plan,ar,polynomial,transformed);
    constraints=domains.count();
    const auto solver_bytes = runtime::runtime_kraft_workspace_bytes(n, constraints);
    if (!map_count || !kernel_bytes || !solver_bytes)
        return 0;
    runtime::WorkspaceSizer size;
    size.add<double>(map_count);
    size.add<double>(n);
    size.add<double>(n);
    size.add<double>(square);
    size.add<double>(n);
    size.add<double>(n);
    size.add<std::byte>(kernel_bytes);
    size.add<double>(n);
    size.add<std::byte>(solver_bytes);
    if (constraints) {
        const auto domain_bytes = domains.workspace_bytes();
        if (!domain_bytes)
            return 0;
        size.add<double>(domains.jacobian_size());
        size.add<double>(2*domains.max_order());
        size.add<std::byte>(domain_bytes);
    }
    return size.size();
}

template<class Density, class Shape, class Design,
         bool Transformed=true,
         std::size_t Q = dynamic, std::size_t Width = dynamic>
struct CoordinateObjective {
    Layout plan;
    Autoregression ar;
    Design design;
    const double* observations;
    std::size_t n, first;
    double* theta;
    double* theta_gradient;
    double* jacobian;
    double* map_scratch;
    WorkingState value_state;
    WorkingState gradient_state;
    MeanConstraintPlan domains;
    void* domain_workspace;
    std::size_t domain_bytes;
    double* domain_values;

    std::size_t dimension() const noexcept
    {
        return plan.width + Shape::parameters;
    }

    template<unsigned Request>
    bool request(const double* z, double* value, double* gradient) const noexcept
    {
        constexpr bool G = (Request & 2u) != 0;
        for (const auto& block : {domains.first,domains.second}) {
            if (block.domain.order && !block.domain.evaluate(z+block.offset,
                    domain_values,nullptr,domain_workspace,domain_bytes,true))
                return false;
        }
        if constexpr (Transformed) {
            if (!map_forward<Shape>(plan,ar,z,theta,G?jacobian:nullptr,map_scratch))
                return false;
        } else std::copy_n(z,dimension(),theta);
        if (!mean::evaluate_prevalidated<Density, Request, Design, Q, Width>(
                plan, design, theta, observations, n, first,
                G ? gradient_state : value_state, value,
                G ? theta_gradient : nullptr, nullptr))
            return false;
        const double scale = 1.0/static_cast<double>(n-first);
        if constexpr (Request & 1u)
            *value *= scale;
        if constexpr (G) {
            if constexpr (Transformed) {
                if (!map_pullback<Shape>(plan, ar, z, theta, jacobian,
                    theta_gradient, nullptr, gradient, nullptr,
                    map_scratch, nullptr))
                    return false;
            } else std::copy_n(theta_gradient,dimension(),gradient);
            for (std::size_t j = 0; j < dimension(); ++j)
                gradient[j] *= scale;
        }
        return true;
    }

    bool evaluate(kraft::ObjectiveRequest requested, const double* z,
                  double* value, double* gradient, bool) const noexcept
    {
        switch (requested) {
        case kraft::ObjectiveRequest::value: return request<1>(z, value, gradient);
        case kraft::ObjectiveRequest::gradient: return request<2>(z, value, gradient);
        case kraft::ObjectiveRequest::value_gradient: return request<3>(z, value, gradient);
        }
        return false;
    }
};

struct PolynomialConstraints {
    std::size_t parameters;
    MeanConstraintPlan plan;
    double* local_jacobian;
    void* workspace;
    std::size_t bytes;
    std::size_t dimension() const noexcept { return parameters; }
    std::size_t count() const noexcept { return plan.count(); }
    bool evaluate(kraft::ConstraintRequest request, const double* theta,
                  double* values, double* jacobian) const noexcept
    {
        if (!count())
            return true;
        const bool derivatives = request != kraft::ConstraintRequest::value;
        if (derivatives) std::fill_n(jacobian,count()*parameters,0.0);
        std::size_t offset=0;
        for (const auto& block : {plan.first,plan.second}) {
            if (!block.domain.order) continue;
            if (!block.domain.evaluate(theta+block.offset,
                    request==kraft::ConstraintRequest::jacobian?nullptr:values+offset,
                    derivatives?local_jacobian:nullptr,workspace,bytes,false))
                return false;
            if (derivatives) {
                for (std::size_t row=0; row<2*block.domain.order; ++row) {
                    for (std::size_t column=0; column<block.domain.parameters; ++column)
                        jacobian[offset+row+(block.offset+column)*count()] =
                            local_jacobian[row*block.domain.parameters+column];
                }
            }
            offset+=2*block.domain.order;
        }
        return true;
    }
};

template<class Density, class Shape, class Design,
         std::size_t Q = dynamic, std::size_t Width = dynamic, bool Transformed=true>
kraft::Result solve(
    const Layout& plan, Autoregression ar, const Design& design,
    const double* observations, std::size_t n, std::size_t first,
    double* coordinates, void* memory, std::size_t bytes,
    double* theta_out, double* gradient_out, const kraft::Options& options,
    PolynomialConstraintPlan polynomial = {}, const double* scales = nullptr) noexcept
{
    const auto domains=constraint_plan(plan,ar,polynomial,Transformed);
    const auto required = solve_workspace_bytes(plan, ar, Shape::parameters, polynomial,Transformed);
    if (!required || bytes < required || !memory || !coordinates
        || !observations || !theta_out || !gradient_out || first >= n
        || first<plan.ma_order || extent<Q>(plan.ma_order)!=plan.ma_order
        || extent<Width>(plan.width)!=plan.width || design.columns()!=plan.columns)
        return {};
    if (polynomial.domain.order && (ar.order || plan.ma_order
        || polynomial.offset > plan.columns
        || polynomial.domain.parameters > plan.columns-polynomial.offset))
        return {};
    const std::size_t dimension = plan.width + Shape::parameters;
    std::fill_n(theta_out,dimension,std::numeric_limits<double>::quiet_NaN());
    std::fill_n(gradient_out,dimension,std::numeric_limits<double>::quiet_NaN());
    runtime::WorkspaceArena arena(memory, bytes);
    double* scratch = arena.take<double>(map_scratch_count(std::max(ar.order, plan.ma_order), 3));
    double* theta = arena.take<double>(dimension);
    double* theta_gradient = arena.take<double>(dimension);
    double* jacobian = arena.take<double>(dimension*dimension);
    double* lower = arena.take<double>(dimension);
    double* upper = arena.take<double>(dimension);
    const auto kernel_bytes = workspace_bytes(plan, 3);
    void* kernel_memory = arena.take<std::byte>(kernel_bytes);
    double* scaled_point = arena.take<double>(dimension);
    const auto value_state=bind_workspace(plan,1,kernel_memory,kernel_bytes);
    const auto gradient_state=bind_workspace(plan,3,kernel_memory,kernel_bytes);
    if (!value_state.valid || !gradient_state.valid)
        return {};
    const std::size_t constraint_count = domains.count();
    const auto solver_bytes = runtime::runtime_kraft_workspace_bytes(dimension, constraint_count);
    void* solver_memory = arena.take<std::byte>(solver_bytes);
    double* constraint_jacobian = constraint_count
        ? arena.take<double>(domains.jacobian_size()) : nullptr;
    double* domain_values=constraint_count?arena.take<double>(2*domains.max_order()):nullptr;
    const auto domain_bytes = constraint_count ? domains.workspace_bytes() : 0;
    void* domain_memory = constraint_count ? arena.take<std::byte>(domain_bytes) : nullptr;
    runtime::RuntimeKraftWorkspace solver_workspace;
    if (!runtime::bind_runtime_kraft_workspace(
            dimension, constraint_count, solver_memory, solver_bytes, solver_workspace))
        return {};
    if constexpr (Transformed) coordinate_bounds<Shape>(plan,ar,lower,upper);
    else parameter_bounds(plan.width,density_code<Density>(),lower,upper);
    for (std::size_t i=0; i<dimension; ++i) {
        if (!std::isfinite(lower[i]))
            lower[i] = std::numeric_limits<double>::quiet_NaN();
        if (!std::isfinite(upper[i]))
            upper[i] = std::numeric_limits<double>::quiet_NaN();
    }
    const CoordinateObjective<Density, Shape, Design, Transformed, Q, Width> objective {
        plan, ar, design, observations, n, first, theta, theta_gradient,
        jacobian, scratch, value_state, gradient_state, domains,
        domain_memory,domain_bytes,domain_values};
    const PolynomialConstraints constraints {dimension, domains,
        constraint_jacobian,domain_memory,domain_bytes};
    kraft::Result result;
    if (scales) {
        for (std::size_t i=0; i<dimension; ++i) {
            if (!(scales[i]>0.0) || !std::isfinite(scales[i]))
                return {};
            coordinates[i] /= scales[i];
            lower[i] /= scales[i];
            upper[i] /= scales[i];
        }
        result = runtime::minimize_runtime_kraft(
            runtime::DiagonalObjective{objective,scales,scaled_point},
            runtime::DiagonalConstraints{constraints,scales,scaled_point},
            coordinates,{lower,upper},options,solver_workspace);
        for (std::size_t i=0; i<dimension; ++i)
            coordinates[i] *= scales[i];
    } else {
        result = runtime::minimize_runtime_kraft(objective,constraints,coordinates,
            {lower,upper},options,solver_workspace);
    }
    double value;
    if (!objective.template request<3>(coordinates, &value, gradient_out)) {
        result.mode = -101;
        return result;
    }
    ++result.function_evaluations;
    ++result.gradient_evaluations;
    result.value = value*static_cast<double>(n-first);
    std::copy_n(theta, dimension, theta_out);
    std::copy_n(theta_gradient, dimension, gradient_out);
    return result;
}

}  // namespace vlk::experiment::mean
#endif
