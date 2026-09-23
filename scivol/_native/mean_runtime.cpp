// SCIVOL-NATIVE: mean-model checked ABI and outer dispatch.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "constant_mean_jet.hpp"
#include "mean_map.hpp"
#include "mean_solver.hpp"
#include "stability_domain.hpp"
#include "jet_contract.hpp"
#include "mean_parameter_domain.hpp"

using namespace vlk::experiment;

static int mean_dispatch_kind(bool linear, bool fixed, std::size_t columns,
                              std::size_t q) noexcept
{
    if (!fixed) return 0;
    if (linear && columns==4 && q==0) return 1;
    if (!linear && columns==2 && q==1) return 2;
    return 0;
}

static bool valid_mean_sample(const double* y, const double* x, std::size_t n,
    std::size_t columns, std::size_t q, std::size_t first, bool linear) noexcept
{
    std::size_t count;
    if (!y || first>=n || first<q || (linear && columns && !x)
        || (!linear && (!columns || first<columns-1))
        || !runtime::checked_product(n,columns,count))
        return false;
    for (std::size_t t=0; t<n; ++t) {
        if (!std::isfinite(y[t]))
            return false;
        if (linear && t>=first) {
            for (std::size_t column=0; column<columns; ++column)
                if (!std::isfinite(x[t*columns+column]))
                    return false;
        }
    }
    return true;
}

static std::size_t mean_stability_workspace(
    const double* basis, std::size_t order, std::size_t parameters, int derivatives)
{
    return mean::PolynomialDomain {order,parameters,basis}.workspace_bytes(derivatives != 0);
}

static int mean_stability(
    const double* basis, std::size_t order, std::size_t parameters,
    const double* theta, double* constraints, double* jacobian,
    void* workspace, std::size_t bytes, int require_stable)
{
    return mean::PolynomialDomain {order,parameters,basis}.evaluate(
        theta,constraints,jacobian,workspace,bytes,require_stable != 0);
}

struct Evaluation {
    const double* theta;
    const double* y;
    const double* x;
    std::size_t n, columns, q, first;
    bool linear, fixed;
    void* workspace;
    std::size_t bytes;
    double* f;
    double* g;
    double* h;
    double* residual_path = nullptr;

    template<class Density, unsigned Request>
    bool run() const noexcept
    {
        const auto plan = mean::layout(columns, q);
        if (linear) {
            if (mean_dispatch_kind(linear,fixed,columns,q)==1) {
                return mean::evaluate<Density, Request,
                    mean::LinearDesign<4>, 0, 5>(
                        plan, {x, columns}, theta, y, n, first,
                        workspace, bytes, f, g, h, residual_path);
            }
            return mean::evaluate<Density, Request>(
                plan, mean::LinearDesign<> {x, columns}, theta, y, n, first,
                workspace, bytes, f, g, h, residual_path);
        }
        if (columns == 0u || first < columns - 1u)
            return false;
        if (mean_dispatch_kind(linear,fixed,columns,q)==2) {
            return mean::evaluate<Density, Request,
                mean::ArmaDesign<1>, 1, 4>(
                    plan, {y, 1}, theta, y, n, first,
                    workspace, bytes, f, g, h, residual_path);
        }
        return mean::evaluate<Density, Request>(
            plan, mean::ArmaDesign<> {y, columns - 1u}, theta, y, n, first,
            workspace, bytes, f, g, h, residual_path);
    }

    template<class Density>
    bool dispatch(unsigned request) const noexcept
    {
        switch (request) {
        case 1: return run<Density, 1>();
        case 2: return run<Density, 2>();
        case 3: return run<Density, 3>();
        case 4: return run<Density, 4>();
        case 5: return run<Density, 5>();
        case 6: return run<Density, 6>();
        case 7: return run<Density, 7>();
        default: return false;
        }
    }
};


static std::size_t mean_coordinate_workspace(
    std::size_t columns, std::size_t q, std::size_t ar_order,
    unsigned shapes, unsigned request)
{
    const auto plan = mean::layout(columns, q);
    if (!plan.valid || request == 0 || request > 7)
        return 0;
    std::size_t n, square;
    if (!runtime::checked_sum(plan.width, shapes, n)
        || !runtime::checked_product(n, n, square))
        return 0;
    const auto scratch_count = mean::map_scratch_count(std::max(ar_order, q), request);
    const auto kernel_bytes = mean::workspace_bytes(plan, request | ((request & 4u) ? 2u : 0u));
    if (!scratch_count || !kernel_bytes)
        return 0;
    runtime::WorkspaceSizer size;
    size.add<double>(scratch_count);
    size.add<double>(n);
    if (request & 6u) {
        size.add<double>(square);
        size.add<double>(n);
    }
    if (request & 4u) {
        size.add<double>(square);
        size.add<double>(square);
    }
    size.add<std::byte>(kernel_bytes);
    return size.size();
}

struct CoordinateEvaluation {
    Evaluation base;
    mean::Autoregression ar;

    template<class Density, class Shape>
    bool run(unsigned request) const noexcept
    {
        const auto plan = mean::layout(base.columns, base.q);
        const auto required = mean_coordinate_workspace(
            base.columns, base.q, ar.order, Shape::parameters, request);
        if (!required || base.bytes < required || !base.workspace || !base.theta
            || ((request & 1u) && !base.f) || ((request & 2u) && !base.g)
            || ((request & 4u) && !base.h))
            return false;
        const std::size_t n = plan.width + Shape::parameters;
        runtime::WorkspaceArena arena(base.workspace, base.bytes);
        double* scratch = arena.take<double>(
            mean::map_scratch_count(std::max(ar.order, base.q), request));
        double* theta = arena.take<double>(n);
        double* jacobian = (request & 6u) ? arena.take<double>(n*n) : nullptr;
        double* gradient = (request & 6u) ? arena.take<double>(n) : nullptr;
        double* hessian = (request & 4u) ? arena.take<double>(n*n) : nullptr;
        double* intermediate = (request & 4u) ? arena.take<double>(n*n) : nullptr;
        const unsigned inner_request = request | ((request & 4u) ? 2u : 0u);
        const auto inner_bytes = mean::workspace_bytes(plan, inner_request);
        auto inner = base;
        inner.theta = theta;
        inner.workspace = arena.take<std::byte>(inner_bytes);
        inner.bytes = inner_bytes;
        inner.g = gradient;
        inner.h = hessian;
        if (!mean::map_forward<Shape>(plan, ar, base.theta, theta, jacobian, scratch)
            || !inner.dispatch<Density>(inner_request))
            return false;
        if (!(request & 6u))
            return true;
        return mean::map_pullback<Shape>(
            plan, ar, base.theta, theta, jacobian, gradient, hessian,
            (request & 2u) ? base.g : nullptr,
            (request & 4u) ? base.h : nullptr, scratch, intermediate);
    }
};


struct Fit {
    const double* y;
    const double* x;
    std::size_t n, columns, q, first;
    mean::Autoregression ar;
    bool linear, fixed;
    double* coordinates;
    void* workspace;
    std::size_t bytes;
    double* theta;
    double* gradient;
    kraft::Options options;
    mean::PolynomialConstraintPlan polynomial {};
    const double* scales = nullptr;

    template<class Density, class Shape, bool Transformed=true>
    kraft::Result run() const noexcept
    {
        const auto plan = mean::layout(columns, q);
        if (!plan.valid || !valid_mean_sample(y,x,n,columns,q,first,linear))
            return {};
        if (linear) {
            if (mean_dispatch_kind(linear,fixed,columns,q)==1)
                return mean::solve<Density, Shape, mean::LinearDesign<4>, 0, 5,Transformed>(
                    plan, ar, {x, columns}, y, n, first, coordinates,
                    workspace, bytes, theta, gradient, options, polynomial, scales);
            if (q == 0)
                return mean::solve<Density, Shape, mean::LinearDesign<>, 0,mean::dynamic,Transformed>(
                    plan, ar, {x, columns}, y, n, first, coordinates,
                    workspace, bytes, theta, gradient, options, polynomial, scales);
            return mean::solve<Density, Shape,mean::LinearDesign<>,mean::dynamic,mean::dynamic,Transformed>(
                plan, ar, mean::LinearDesign<> {x, columns}, y, n, first,
                coordinates, workspace, bytes, theta, gradient, options, polynomial, scales);
        }
        if (mean_dispatch_kind(linear,fixed,columns,q)==2)
            return mean::solve<Density, Shape, mean::ArmaDesign<1>, 1, 4,Transformed>(
                plan, ar, {y, 1}, y, n, first, coordinates,
                workspace, bytes, theta, gradient, options, polynomial, scales);
        if (q == 0)
            return mean::solve<Density, Shape, mean::ArmaDesign<>, 0,mean::dynamic,Transformed>(
                plan, ar, {y,columns-1}, y, n, first, coordinates,
                workspace, bytes, theta, gradient, options, polynomial, scales);
        return mean::solve<Density, Shape,mean::ArmaDesign<>,mean::dynamic,mean::dynamic,Transformed>(
            plan, ar, mean::ArmaDesign<> {y, columns-1}, y, n, first,
            coordinates, workspace, bytes, theta, gradient, options, polynomial, scales);
    }
};


namespace {
// The descriptor is immutable and parsed once per ABI call, never per observation.
struct Descriptor {
    std::size_t n, columns, q, first, ar_start, ar_order;
    unsigned density;
    int linear, fixed;
    std::size_t order, horizons, offset;
    bool parse(PyObject* object)
    {
        if (!PyArg_ParseTuple(object,"nnnnnnIppnnn",&n,&columns,&q,&first,
                &ar_start,&ar_order,&density,&linear,&fixed,&order,&horizons,&offset))
            return false;
        const auto plan = mean::layout(columns,q);
        std::size_t count, square;
        constexpr auto maximum = static_cast<std::size_t>(PY_SSIZE_T_MAX);
        if (n>maximum || columns>maximum || q>maximum || order>maximum
            || horizons>maximum || offset>maximum || ar_start>maximum || ar_order>maximum
            || density >= 8 || !plan.valid || first >= n || first < q
            || (ar_order && (ar_start > columns || ar_order > columns-ar_start))
            || (order && (offset > columns || horizons > columns-offset))
            || !runtime::checked_sum(plan.width,mean::shape_count(density),count)
            || !runtime::checked_product(count,count,square)) {
            PyErr_SetString(PyExc_ValueError,"Invalid mean-model dimensions or density.");
            return false;
        }
        return true;
    }
};

template<class Function>
auto dispatch_shape(unsigned density, Function function)
{
    using namespace generated;
    switch (density) {
    case 0: return function.template operator()<NormalDensity,NoShapeTransform>();
    case 1: return function.template operator()<StudentTDensity,StudentTShapeTransform>();
    case 2: return function.template operator()<SkewTDensity,SkewTShapeTransform>();
    case 3: return function.template operator()<GedDensity,GedShapeTransform>();
    case 4: return function.template operator()<GeneralizedHyperbolicDensity,GhShapeTransform>();
    case 5: return function.template operator()<NormalInverseGaussianDensity,FixedLambdaGhShapeTransform>();
    case 6: return function.template operator()<HyperbolicDensity,FixedLambdaGhShapeTransform>();
    default: return function.template operator()<VarianceGammaDensity,VarianceGammaShapeTransform>();
    }
}

template<class T> T* address(unsigned long long value)
{
    return reinterpret_cast<T*>(static_cast<std::uintptr_t>(value));
}

PyObject* py_sizes(PyObject*, PyObject* args)
{
    PyObject* descriptor;
    unsigned long long basis;
    Descriptor d;
    if (!PyArg_ParseTuple(args,"OK",&descriptor,&basis) || !d.parse(descriptor))
        return nullptr;
    const auto plan=mean::layout(d.columns,d.q);
    const auto kernel=mean::workspace_bytes(plan,7);
    const auto coordinate=mean_coordinate_workspace(d.columns,d.q,d.ar_order,
        mean::shape_count(d.density),7);
    const auto solve=mean::solve_workspace_bytes(plan,{d.ar_start,d.ar_order},
        mean::shape_count(d.density),{{d.order,d.horizons,address<double>(basis)},d.offset});
    const auto theta_solve=mean::solve_workspace_bytes(plan,{d.ar_start,d.ar_order},
        mean::shape_count(d.density),{{d.order,d.horizons,address<double>(basis)},d.offset},false);
    return Py_BuildValue("(nnn)",kernel,coordinate,std::max(solve,theta_solve));
}

PyObject* py_dispatch(PyObject*, PyObject* args)
{
    PyObject* descriptor;
    Descriptor d;
    if (!PyArg_ParseTuple(args,"O",&descriptor) || !d.parse(descriptor))
        return nullptr;
    return PyLong_FromLong(mean_dispatch_kind(d.linear,d.fixed,d.columns,d.q));
}

PyObject* py_bounds(PyObject*, PyObject* args)
{
    PyObject* descriptor;
    unsigned long long lower,upper;
    Descriptor d;
    int coordinates = 0;
    if (!PyArg_ParseTuple(args,"OKK|p",&descriptor,&lower,&upper,&coordinates) || !d.parse(descriptor))
        return nullptr;
    if (coordinates) {
        if (!lower || !upper)
            return PyBool_FromLong(0);
        dispatch_shape(d.density,[&]<class Density,class Shape>() {
            mean::coordinate_bounds<Shape>(mean::layout(d.columns,d.q),
                {d.ar_start,d.ar_order},address<double>(lower),address<double>(upper));
            return true;
        });
        return PyBool_FromLong(1);
    }
    return PyBool_FromLong(mean::parameter_bounds(mean::layout(d.columns,d.q).width,
        d.density,address<double>(lower),address<double>(upper)));
}

PyObject* py_map(PyObject*, PyObject* args)
{
    PyObject* descriptor;
    unsigned long long source,destination,memory,jacobian=0;
    Py_ssize_t bytes;
    int forward;
    Descriptor d;
    if (!PyArg_ParseTuple(args,"OpKKKn|K",&descriptor,&forward,&source,&destination,&memory,&bytes,&jacobian)
        || !d.parse(descriptor))
        return nullptr;
    const auto plan=mean::layout(d.columns,d.q);
    const auto scratch=mean::map_scratch_count(std::max(d.ar_order,d.q),jacobian?3:1);
    if (!source || !destination || !memory || bytes<0
        || std::size_t(bytes)/sizeof(double)<scratch)
        return PyBool_FromLong(0);
    const bool valid=dispatch_shape(d.density,[&]<class Density,class Shape>() {
        return forward ? mean::map_forward<Shape>(plan,{d.ar_start,d.ar_order},
            address<double>(source),address<double>(destination),address<double>(jacobian),address<double>(memory))
            : mean::map_inverse<Shape>(plan,{d.ar_start,d.ar_order},
                address<double>(source),address<double>(destination),address<double>(memory));
    });
    return PyBool_FromLong(valid);
}

PyObject* py_jet(PyObject*, PyObject* args)
{
    PyObject* descriptor;
    unsigned long long theta,y,x,memory,f,g,h,residual;
    Py_ssize_t bytes;
    unsigned request;
    int transformed;
    int checked_inputs=1;
    Descriptor d;
    if (!PyArg_ParseTuple(args,"OKKKIpKnKKKK|p",&descriptor,&theta,&y,&x,&request,
            &transformed,&memory,&bytes,&f,&g,&h,&residual,&checked_inputs) || !d.parse(descriptor))
        return nullptr;
    if (bytes<0 || !y || (d.linear && d.columns && !x)
        || (checked_inputs && !valid_mean_sample(address<double>(y),address<double>(x),
            d.n,d.columns,d.q,d.first,d.linear)))
        return PyBool_FromLong(0);
    const Evaluation evaluation{address<double>(theta),address<double>(y),
        address<double>(x),d.n,d.columns,d.q,d.first,bool(d.linear),bool(d.fixed),
        address<void>(memory),std::size_t(bytes),address<double>(f),address<double>(g),
        address<double>(h),address<double>(residual)};
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid=dispatch_shape(d.density,[&]<class Density,class Shape>() {
        return transformed ? CoordinateEvaluation{evaluation,{d.ar_start,d.ar_order}}
            .template run<Density,Shape>(request) : evaluation.template dispatch<Density>(request);
    });
    Py_END_ALLOW_THREADS
    return PyBool_FromLong(valid);
}

PyObject* py_fit(PyObject*, PyObject* args)
{
    PyObject* descriptor;
    unsigned long long y,x,z,memory,theta,g,basis,scales;
    Py_ssize_t bytes;
    int iterations;
    int transformed=1;
    double accuracy;
    Descriptor d;
    if (!PyArg_ParseTuple(args,"OKKKKnKKKKid|p",&descriptor,&y,&x,&z,&memory,&bytes,
            &theta,&g,&basis,&scales,&iterations,&accuracy,&transformed) || !d.parse(descriptor))
        return nullptr;
    if (bytes<0 || iterations<1 || !std::isfinite(accuracy) || accuracy<=0)
        return PyErr_Format(PyExc_ValueError,"Invalid mean optimizer budget or workspace.");
    const Fit fit{address<double>(y),address<double>(x),d.n,d.columns,d.q,d.first,
        {d.ar_start,d.ar_order},bool(d.linear),bool(d.fixed),address<double>(z),
        address<void>(memory),std::size_t(bytes),address<double>(theta),address<double>(g),
        {accuracy,iterations},{{d.order,d.horizons,address<double>(basis)},d.offset},
        address<double>(scales)};
    kraft::Result result;
    Py_BEGIN_ALLOW_THREADS
    result=dispatch_shape(d.density,[&]<class Density,class Shape>() {
        return transformed?fit.template run<Density,Shape>():fit.template run<Density,Shape,false>();
    });
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(diiiid)",result.value,result.mode,result.iterations,
        result.function_evaluations,result.gradient_evaluations,result.maximum_violation);
}

PyObject* py_stability(PyObject*, PyObject* args)
{
    unsigned long long basis,theta,values,jacobian,memory;
    Py_ssize_t order,parameters,bytes;
    int require_stable;
    if (!PyArg_ParseTuple(args,"KnnKKKKnp",&basis,&order,&parameters,&theta,&values,
            &jacobian,&memory,&bytes,&require_stable))
        return nullptr;
    if (order<1 || parameters<1 || bytes<0)
        return PyBool_FromLong(0);
    return PyBool_FromLong(mean_stability(address<double>(basis),order,parameters,
        address<double>(theta),address<double>(values),address<double>(jacobian),
        address<void>(memory),bytes,require_stable));
}

PyObject* py_stability_size(PyObject*, PyObject* args)
{
    unsigned long long basis;
    Py_ssize_t order,parameters;
    if (!PyArg_ParseTuple(args,"Knn",&basis,&order,&parameters))
        return nullptr;
    if (order<1 || parameters<1)
        return PyLong_FromLong(0);
    return PyLong_FromSize_t(mean_stability_workspace(address<double>(basis),order,parameters,1));
}

PyMethodDef mean_methods[] = {
    {"_mean_dispatch_kind",py_dispatch,METH_VARARGS,"Resolved mean kernel used by evaluation and solve."},
    {"_mean_sizes",py_sizes,METH_VARARGS,"Checked mean workspaces."},
    {"_mean_bounds",py_bounds,METH_VARARGS,"Original-coordinate numerical domain."},
    {"_mean_map",py_map,METH_VARARGS,"Mean coordinate map."},
    {"_mean_jet",py_jet,METH_VARARGS,"Constant-scale mean jet."},
    {"_mean_fit",py_fit,METH_VARARGS,"Typed native mean solve."},
    {"_mean_stability",py_stability,METH_VARARGS,"Implied-polynomial stability."},
    {"_mean_stability_size",py_stability_size,METH_VARARGS,"Polynomial workspace."},
    {nullptr,nullptr,0,nullptr}
};
}

int register_mean_methods(PyObject* module)
{
    return PyModule_AddFunctions(module,mean_methods);
}
