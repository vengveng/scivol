// SCIVOL-NATIVE: Python boundary; validates sample once before the native solve.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "linear_joint_api.hpp"
#include "runtime_workspace.hpp"
#include <algorithm>
#include <cmath>

namespace {
using namespace scivol::linear_joint;
namespace runtime=vlk::experiment::runtime;

template<class T> T* address(unsigned long long p)
{ return reinterpret_cast<T*>(static_cast<std::uintptr_t>(p)); }

bool parse_plan(PyObject* object,Plan& p)
{
    Py_ssize_t columns,vp,vq,n;
    int fixed;
    if (!PyArg_ParseTuple(object,"IInnnnp",&p.family,&p.density,&columns,&vp,&vq,&n,&fixed))
        return false;
    if (p.family>3 || p.density>7 || columns<0 || vp<1 || vq<1 || n<1) {
        PyErr_SetString(PyExc_ValueError,"Invalid joint model layout.");
        return false;
    }
    p.columns=columns;p.p=vp;p.q=vq;p.n=n;p.fixed=fixed;
    return true;
}

bool parse_domain(PyObject* object,const Plan& p,Domain& d)
{
    Py_ssize_t order,width,offset;
    unsigned long long basis;
    if (!PyArg_ParseTuple(object,"nnnK",&order,&width,&offset,&basis)) return false;
    if (order<0 || width<0 || offset<0 || std::size_t(offset)>p.columns
        || std::size_t(width)>p.columns-std::size_t(offset)
        || (order && (!width || !basis))) {
        PyErr_SetString(PyExc_ValueError,"Invalid joint mean stability plan.");
        return false;
    }
    d={std::size_t(order),std::size_t(width),std::size_t(offset),address<double>(basis)};
    return true;
}

bool valid_sample(const Plan& p,const double* y,const double* x,double h0)
{
    std::size_t count;
    if (!y || !(h0>0) || !std::isfinite(h0)
        || !runtime::checked_product(p.n,p.columns,count) || (count && !x)) return false;
    for (std::size_t i=0;i<p.n;++i) if (!std::isfinite(y[i])) return false;
    for (std::size_t i=0;i<count;++i) if (!std::isfinite(x[i])) return false;
    return true;
}

PyObject* sizes(PyObject*,PyObject* args)
{
    PyObject *descriptor,*domain;
    unsigned mask;
    int transformed;Py_ssize_t affine=0;
    Plan p;Domain d;
    if (!PyArg_ParseTuple(args,"OIpO|n",&descriptor,&mask,&transformed,&domain,&affine)
        || !parse_plan(descriptor,p) || !parse_domain(domain,p,d)) return nullptr;
    if (!mask || mask>7) return PyErr_Format(PyExc_ValueError,"Invalid joint jet request.");
    if (affine<0 || (affine && (p.family<2 || !p.columns || std::size_t(affine)>p.columns)))
        return PyErr_Format(PyExc_ValueError,"Joint affine phases require a linear EGARCH/APARCH mean.");
    const auto kernel=p.family<2?garch_workspace(p,mask,transformed):power_workspace(p,mask,transformed);
    const auto solver=p.family<2?garch_solve_workspace(p,d,transformed):power_solve_workspace(p,d,transformed,affine);
    return Py_BuildValue("(KK)",static_cast<unsigned long long>(kernel),static_cast<unsigned long long>(solver));
}

PyObject* jet_entry(PyObject* args,bool check_sample)
{
    PyObject* descriptor;
    unsigned long long theta,y,x,memory,f,g,h,e,v;
    Py_ssize_t bytes;
    double h0;unsigned mask;int transformed;Plan p;
    if (!PyArg_ParseTuple(args,"OKKKdIpKnKKKKK",&descriptor,&theta,&y,&x,&h0,&mask,
        &transformed,&memory,&bytes,&f,&g,&h,&e,&v) || !parse_plan(descriptor,p)) return nullptr;
    if (bytes<0 || !theta || !memory || !mask || mask>7
        || ((mask&1)&&!f) || ((mask&2)&&!g) || ((mask&4)&&!h)
        || (transformed && (e || v))) return PyBool_FromLong(0);
    const Evaluation evaluation{address<double>(theta),address<double>(y),address<double>(x),
        h0,mask,bool(transformed),address<void>(memory),std::size_t(bytes),
        address<double>(f),address<double>(g),address<double>(h),address<double>(e),address<double>(v)};
    bool valid=false;
    Py_BEGIN_ALLOW_THREADS
    if (!check_sample || valid_sample(p,evaluation.y,evaluation.x,h0))
        valid=p.family<2?garch_evaluate(p,evaluation):power_evaluate(p,evaluation);
    Py_END_ALLOW_THREADS
    return PyBool_FromLong(valid);
}

PyObject* restriction_bytes(PyObject*,PyObject* args)
{
    PyObject *descriptor,*domain;
    int transformed=0;Py_ssize_t affine=0;
    Plan p;Domain d;
    if (!PyArg_ParseTuple(args,"OO|pn",&descriptor,&domain,&transformed,&affine)
        || !parse_plan(descriptor,p) || !parse_domain(domain,p,d)) return nullptr;
    if (p.family<2 || affine<0 || std::size_t(affine)>p.columns)
        return PyErr_Format(PyExc_ValueError,"Invalid joint EGARCH/APARCH coordinate subset plan.");
    return PyLong_FromSize_t(power_restriction_workspace(p,d,transformed,affine));
}

PyObject* jet(PyObject*,PyObject* args) { return jet_entry(args,true); }
PyObject* prepared_jet(PyObject*,PyObject* args) { return jet_entry(args,false); }

PyObject* refinement_bytes(PyObject*,PyObject* args)
{
    PyObject *descriptor,*domain;Plan p;Domain d;
    if (!PyArg_ParseTuple(args,"OO",&descriptor,&domain)
        || !parse_plan(descriptor,p) || !parse_domain(domain,p,d)) return nullptr;
    if (p.family<2)
        return PyErr_Format(PyExc_ValueError,"Active-face refinement needs an EGARCH/APARCH provider.");
    return PyLong_FromSize_t(power_refinement_workspace(p,d));
}

PyObject* refine(PyObject*,PyObject* args)
{
    PyObject *descriptor,*domain;Plan p;Domain d;
    unsigned long long point,best,y,x,lower,upper,units,memory,evidence,counts;
    Py_ssize_t bytes;double h0,target;unsigned backtracks;
    if (!PyArg_ParseTuple(args,"OOKKKKdKKKKnKKdI",&descriptor,&domain,&point,&best,&y,&x,&h0,
        &lower,&upper,&units,&memory,&bytes,&evidence,&counts,&target,&backtracks)
        || !parse_plan(descriptor,p) || !parse_domain(domain,p,d)) return nullptr;
    if (p.family<2 || bytes<0 || !point || !best || point==best || !lower || !upper
        || !units || !memory || !evidence || !counts || !std::isfinite(target)
        || target<0 || backtracks>1024)
        return PyErr_Format(PyExc_ValueError,"Invalid joint refinement buffers or options.");
    const Refinement input{address<double>(point),address<double>(best),address<double>(y),
        address<double>(x),h0,d,address<double>(lower),address<double>(upper),address<double>(units),
        address<void>(memory),std::size_t(bytes),address<double>(evidence),address<std::size_t>(counts),
        target,backtracks};
    int status=0;
    Py_BEGIN_ALLOW_THREADS
    if (valid_sample(p,input.y,input.x,h0)) status=power_refine(p,input);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(status);
}

PyObject* inverse(PyObject*,PyObject* args)
{
    PyObject* descriptor;unsigned long long theta,z,memory;Py_ssize_t bytes;Plan p;
    if (!PyArg_ParseTuple(args,"OKKKn",&descriptor,&theta,&z,&memory,&bytes)
        || !parse_plan(descriptor,p)) return nullptr;
    if (!theta || !z || !memory || bytes<0) return PyBool_FromLong(0);
    const bool valid=p.family<2?garch_inverse(p,address<double>(theta),address<double>(z),address<void>(memory),bytes)
        :power_inverse(p,address<double>(theta),address<double>(z),address<void>(memory),bytes);
    return PyBool_FromLong(valid);
}

PyObject* fit(PyObject*,PyObject* args)
{
    PyObject *descriptor,*domain;
    unsigned long long z,y,x,lower,upper,scales,memory,theta,g,f,stats,normal=0,requests=0,resets=0,targets=0;
    unsigned long long free_indices=0,restriction_memory=0;
    Py_ssize_t free_count=0,restriction_size=0;
    Py_ssize_t bytes,iterations,affine_rows=1;double h0,accuracy,target=0;int transformed,speculate=-1;Plan p;Domain d;
    int curvature_reset=0,minimum_invalid_step=0;
    if (!PyArg_ParseTuple(args,"OOKKKdKKKKnKKKKdnp|pKdKpKnKpKnKn",&descriptor,&domain,&z,&y,&x,&h0,
        &lower,&upper,&scales,&memory,&bytes,&theta,&g,&f,&stats,&accuracy,&iterations,&transformed,&speculate,&normal,&target,&requests,&curvature_reset,&resets,&affine_rows,&targets,&minimum_invalid_step,
        &free_indices,&free_count,&restriction_memory,&restriction_size)
        || !parse_plan(descriptor,p) || !parse_domain(domain,p,d)) return nullptr;
    if (bytes<0 || iterations<1 || !(accuracy>0) || !std::isfinite(accuracy)
        || !z || !lower || !upper || !scales || !memory || !theta || !g || !f || !stats)
        return PyErr_Format(PyExc_ValueError,"Invalid joint optimizer buffers or budget.");
    if (normal && (p.family<2 || !p.columns || !std::isfinite(target) || affine_rows<1
        || static_cast<std::size_t>(affine_rows)>p.columns || (affine_rows>1 && !targets)))
        return PyErr_Format(PyExc_ValueError,"Invalid joint affine phase.");
    const bool restricted=free_indices || free_count || restriction_memory || restriction_size;
    if (restricted && (p.family<2 || !free_indices || free_count<1
        || !restriction_memory || restriction_size<1))
        return PyErr_Format(PyExc_ValueError,"Invalid joint optimizer coordinate subset.");
    const Solve s{address<double>(z),address<double>(y),address<double>(x),h0,d,
        address<double>(lower),address<double>(upper),address<double>(scales),address<void>(memory),
        std::size_t(bytes),address<double>(theta),address<double>(g),address<double>(f),
        address<std::int64_t>(stats),accuracy,iterations,bool(transformed),
        speculate<0?p.family>=2:bool(speculate),
        address<double>(normal),target,address<std::int64_t>(requests),
        bool(curvature_reset),address<std::int64_t>(resets),std::size_t(affine_rows),
        address<double>(targets),bool(minimum_invalid_step)};
    if (s.requests) std::fill_n(s.requests,3,0);
    if (s.reset_count) *s.reset_count=0;
    bool valid=false;
    Py_BEGIN_ALLOW_THREADS
    if (valid_sample(p,s.y,s.x,h0)) {
        if (restricted) valid=power_fit_restricted(p,s,{address<std::size_t>(free_indices),
            std::size_t(free_count),address<void>(restriction_memory),std::size_t(restriction_size)});
        else valid=p.family<2?garch_fit(p,s):power_fit(p,s);
    }
    Py_END_ALLOW_THREADS
    return PyBool_FromLong(valid);
}

PyObject* constraints(PyObject*,PyObject* args)
{
    PyObject *descriptor,*domain;
    unsigned long long theta,values,jacobian,memory;Py_ssize_t bytes;Plan p;Domain d;
    if (!PyArg_ParseTuple(args,"OOKKKKn",&descriptor,&domain,&theta,&values,&jacobian,&memory,&bytes)
        || !parse_plan(descriptor,p) || !parse_domain(domain,p,d)) return nullptr;
    if (!theta || !values || !jacobian || !memory || bytes<0) return PyBool_FromLong(0);
    const bool valid=p.family<2
        ?garch_constraints(p,d,address<double>(theta),address<double>(values),address<double>(jacobian),address<void>(memory),bytes)
        :power_constraints(p,d,address<double>(theta),address<double>(values),address<double>(jacobian),address<void>(memory),bytes);
    return PyBool_FromLong(valid);
}

PyObject* constraint_values(PyObject*,PyObject* args)
{
    PyObject *descriptor,*domain;
    unsigned long long theta,values,memory;Py_ssize_t bytes;Plan p;Domain d;
    if (!PyArg_ParseTuple(args,"OOKKKn",&descriptor,&domain,&theta,&values,&memory,&bytes)
        || !parse_plan(descriptor,p) || !parse_domain(domain,p,d)) return nullptr;
    if (!theta || !values || !memory || bytes<0) return PyBool_FromLong(0);
    const bool valid=p.family<2
        ?garch_constraints(p,d,address<double>(theta),address<double>(values),nullptr,address<void>(memory),bytes)
        :power_constraints(p,d,address<double>(theta),address<double>(values),nullptr,address<void>(memory),bytes);
    return PyBool_FromLong(valid);
}

PyMethodDef methods[]={
    {"_linear_joint_sizes",sizes,METH_VARARGS,"Joint evaluation and solver workspace sizes."},
    {"_linear_joint_restriction_bytes",restriction_bytes,METH_VARARGS,"Temporary optimizer coordinate-subset scratch size."},
    {"_linear_joint_jet",jet,METH_VARARGS,"Complete joint linear-mean volatility jet."},
    {"_linear_joint_prevalidated",prepared_jet,METH_VARARGS,"Joint jet for an already validated immutable sample."},
    {"_linear_joint_inverse",inverse,METH_VARARGS,"Joint theta-to-coordinate map."},
    {"_linear_joint_fit",fit,METH_VARARGS,"Typed joint native SLSQP solve."},
    {"_linear_joint_refinement_bytes",refinement_bytes,METH_VARARGS,"Caller-owned active-face refinement workspace."},
    {"_linear_joint_refine",refine,METH_VARARGS,"One typed active-face proposal with separate best-point retention."},
    {"_linear_joint_constraints",constraints,METH_VARARGS,"Shared joint theta-domain values and Jacobian."},
    {"_linear_joint_constraint_values",constraint_values,METH_VARARGS,"Joint theta-domain values without a Jacobian request."},
    {nullptr,nullptr,0,nullptr}
};
}

namespace scivol::linear_joint {
namespace {
PyObject* accounting_call(PyObject* arguments, PyCFunction entry)
{
    if (!PyTuple_Check(arguments) || PyTuple_GET_SIZE(arguments) == 0)
        return PyErr_Format(PyExc_TypeError,"Linear accounting requires an argument tuple with a descriptor.");
    Plan plan;
    if (!::parse_plan(PyTuple_GET_ITEM(arguments,0),plan)) return nullptr;
    if (plan.family != 3)
        return PyErr_Format(PyExc_ValueError,"Linear accounting selectors require APARCH; other families are not qualified.");
    return entry(nullptr,arguments);
}
}

// Diagnostic-only bridges to the original METH_VARARGS bodies and parsers.
PyObject* accounting_prevalidated(PyObject*, PyObject* arguments)
{ return accounting_call(arguments,::prepared_jet); }
PyObject* accounting_fit(PyObject*, PyObject* arguments)
{ return accounting_call(arguments,::fit); }
PyObject* accounting_refine(PyObject*, PyObject* arguments)
{ return accounting_call(arguments,::refine); }
}

int register_linear_joint_methods(PyObject* module)
{ return PyModule_AddFunctions(module,methods); }
