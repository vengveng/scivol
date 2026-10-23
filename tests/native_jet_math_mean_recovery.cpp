// SCIVOL-NATIVE: TEST_ONLY; fresh-source mean jets and coordinate maps.
#include "mean_map.hpp"
#include <vector>

using namespace vlk::experiment;

struct RecoveryPoint {
    std::size_t columns, q, first, n;
    bool fixed;
    const double *theta, *y, *x;
    void* workspace;
    std::size_t bytes;
    double *f, *g, *h;

    template<class Density, unsigned Request>
    bool evaluate() const
    {
        const auto plan = mean::layout(columns, q);
        if (fixed) {
            if (columns == 0 && q == 0)
                return mean::evaluate<Density, Request, mean::LinearDesign<0>, 0, 1>(
                    plan, {x, columns}, theta, y, n, first, workspace, bytes, f, g, h);
            if (columns == 1 && q == 0)
                return mean::evaluate<Density, Request, mean::LinearDesign<1>, 0, 2>(
                    plan, {x, columns}, theta, y, n, first, workspace, bytes, f, g, h);
            if (columns == 4 && q == 0)
                return mean::evaluate<Density, Request, mean::LinearDesign<4>, 0, 5>(
                    plan, {x, columns}, theta, y, n, first, workspace, bytes, f, g, h);
            if (columns == 2 && q == 1)
                return mean::evaluate<Density, Request, mean::LinearDesign<2>, 1, 4>(
                    plan, {x, columns}, theta, y, n, first, workspace, bytes, f, g, h);
            if (columns == 4 && q == 3)
                return mean::evaluate<Density, Request, mean::LinearDesign<4>, 3, 8>(
                    plan, {x, columns}, theta, y, n, first, workspace, bytes, f, g, h);
            return false;
        }
        return mean::evaluate<Density, Request>(plan, mean::LinearDesign<>{x, columns},
            theta, y, n, first, workspace, bytes, f, g, h);
    }

    template<class Density>
    bool dispatch(unsigned request) const
    {
        switch (request) {
        case 1: return evaluate<Density, 1>();
        case 2: return evaluate<Density, 2>();
        case 3: return evaluate<Density, 3>();
        case 4: return evaluate<Density, 4>();
        case 5: return evaluate<Density, 5>();
        case 6: return evaluate<Density, 6>();
        case 7: return evaluate<Density, 7>();
        default: return false;
        }
    }
};

template<class Shape>
bool recovery_map(const mean::Layout& plan, mean::Autoregression ar,
    const double* z, double* theta, double* jacobian, double* curvature)
{
    const auto width = plan.width + Shape::parameters;
    std::vector<double> scratch(mean::map_scratch_count(std::max(ar.order, plan.ma_order), 7));
    if (!mean::map_forward<Shape>(plan, ar, z, theta, jacobian, scratch.data()))
        return false;
    if (!curvature)
        return true;
    std::vector<double> gradient(width, 0.0), hessian(width*width, 0.0);
    std::vector<double> intermediate(width*width);
    for (std::size_t component = 0; component < width; ++component) {
        gradient[component] = 1.0;
        if (!mean::map_pullback<Shape>(plan, ar, z, theta, jacobian,
                gradient.data(), hessian.data(), nullptr,
                curvature + component*width*width, scratch.data(), intermediate.data()))
            return false;
        gradient[component] = 0.0;
    }
    return true;
}

extern "C" int jet_math_mean_recovery_map(int density, std::size_t columns,
    std::size_t q, std::size_t ar_start, std::size_t ar_order, const double* z,
    double* theta, double* jacobian, double* curvature)
{
    const auto plan = mean::layout(columns, q);
    const mean::Autoregression ar{ar_start, ar_order};
    if (!plan.valid)
        return false;
    switch (density) {
    case 0: return recovery_map<generated::NoShapeTransform>(plan, ar, z, theta, jacobian, curvature);
    case 1: return recovery_map<generated::StudentTShapeTransform>(plan, ar, z, theta, jacobian, curvature);
    case 2: return recovery_map<generated::SkewTShapeTransform>(plan, ar, z, theta, jacobian, curvature);
    case 3: return recovery_map<generated::GedShapeTransform>(plan, ar, z, theta, jacobian, curvature);
    case 4: return recovery_map<generated::GhShapeTransform>(plan, ar, z, theta, jacobian, curvature);
    default: return false;
    }
}

template<class Density, class Shape>
bool recovery_jet(RecoveryPoint point, mean::Autoregression ar,
    unsigned request, bool coordinates)
{
    const auto plan = mean::layout(point.columns, point.q);
    const auto width = plan.width + Density::shape_parameters;
    const unsigned inner = request | ((coordinates && (request & 4u)) ? 2u : 0u);
    std::vector<std::byte> memory(mean::workspace_bytes(plan, inner));
    point.workspace = memory.data();
    point.bytes = memory.size();
    if (!coordinates)
        return point.dispatch<Density>(request);
    std::vector<double> theta(width), gradient(width), hessian(width*width);
    std::vector<double> jacobian((request & 6u) ? width*width : 0);
    std::vector<double> intermediate((request & 4u) ? width*width : 0);
    std::vector<double> scratch(mean::map_scratch_count(std::max(ar.order, point.q), request));
    const double* z = point.theta;
    if (!mean::map_forward<Shape>(plan, ar, z, theta.data(),
            (request & 6u) ? jacobian.data() : nullptr, scratch.data()))
        return false;
    point.theta = theta.data();
    double* public_g = point.g;
    double* public_h = point.h;
    point.g = gradient.data();
    point.h = hessian.data();
    if (!point.dispatch<Density>(inner))
        return false;
    return !(request & 6u) || mean::map_pullback<Shape>(plan, ar, z, theta.data(),
        jacobian.data(), gradient.data(), hessian.data(),
        (request & 2u) ? public_g : nullptr, (request & 4u) ? public_h : nullptr,
        scratch.data(), intermediate.data());
}

extern "C" int jet_math_mean_recovery_jet(int density, bool fixed,
    std::size_t columns, std::size_t q, std::size_t first, std::size_t ar_start,
    std::size_t ar_order, bool coordinates, unsigned request, const double* point,
    const double* y, const double* x, std::size_t n, double* f, double* g, double* h)
{
    const RecoveryPoint input{columns, q, first, n, fixed, point, y, x, nullptr, 0, f, g, h};
    const mean::Autoregression ar{ar_start, ar_order};
    switch (density) {
    case 0: return recovery_jet<generated::NormalDensity, generated::NoShapeTransform>(input, ar, request, coordinates);
    case 1: return recovery_jet<generated::StudentTDensity, generated::StudentTShapeTransform>(input, ar, request, coordinates);
    case 2: return recovery_jet<generated::SkewTDensity, generated::SkewTShapeTransform>(input, ar, request, coordinates);
    case 3: return recovery_jet<generated::GedDensity, generated::GedShapeTransform>(input, ar, request, coordinates);
    default: return false;
    }
}
