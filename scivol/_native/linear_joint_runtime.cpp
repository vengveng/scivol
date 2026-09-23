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
    int transformed,affine=0;
    Plan p;Domain d;
    if (!PyArg_ParseTuple(args,"OIpO|p",&descriptor,&mask,&transformed,&domain,&affine)
        || !parse_plan(descriptor,p) || !parse_domain(domain,p,d)) return nullptr;
    if (!mask || mask>7) return PyErr_Format(PyExc_ValueError,"Invalid joint jet request.");
    if (affine && (p.family<2 || !p.columns))
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

PyObject* jet(PyObject*,PyObject* args) { return jet_entry(args,true); }
PyObject* prepared_jet(PyObject*,PyObject* args) { return jet_entry(args,false); }

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
    unsigned long long z,y,x,lower,upper,scales,memory,theta,g,f,stats,normal=0,requests=0,resets=0;
    Py_ssize_t bytes,iterations;double h0,accuracy,target=0;int transformed,speculate=-1;Plan p;Domain d;
    int curvature_reset=0;
    if (!PyArg_ParseTuple(args,"OOKKKdKKKKnKKKKdnp|pKdKpK",&descriptor,&domain,&z,&y,&x,&h0,
        &lower,&upper,&scales,&memory,&bytes,&theta,&g,&f,&stats,&accuracy,&iterations,&transformed,&speculate,&normal,&target,&requests,&curvature_reset,&resets)
        || !parse_plan(descriptor,p) || !parse_domain(domain,p,d)) return nullptr;
    if (bytes<0 || iterations<1 || !(accuracy>0) || !std::isfinite(accuracy)
        || !z || !lower || !upper || !scales || !memory || !theta || !g || !f || !stats)
        return PyErr_Format(PyExc_ValueError,"Invalid joint optimizer buffers or budget.");
    if (normal && (p.family<2 || !p.columns || !std::isfinite(target)))
        return PyErr_Format(PyExc_ValueError,"Invalid joint affine phase.");
    const Solve s{address<double>(z),address<double>(y),address<double>(x),h0,d,
        address<double>(lower),address<double>(upper),address<double>(scales),address<void>(memory),
        std::size_t(bytes),address<double>(theta),address<double>(g),address<double>(f),
        address<std::int64_t>(stats),accuracy,iterations,bool(transformed),
        speculate<0?p.family>=2:bool(speculate),
        address<double>(normal),target,address<std::int64_t>(requests),
        bool(curvature_reset),address<std::int64_t>(resets)};
    if (s.requests) std::fill_n(s.requests,3,0);
    if (s.reset_count) *s.reset_count=0;
    bool valid=false;
    Py_BEGIN_ALLOW_THREADS
    if (valid_sample(p,s.y,s.x,h0)) valid=p.family<2?garch_fit(p,s):power_fit(p,s);
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
    {"_linear_joint_jet",jet,METH_VARARGS,"Complete joint linear-mean volatility jet."},
    {"_linear_joint_prevalidated",prepared_jet,METH_VARARGS,"Joint jet for an already validated immutable sample."},
    {"_linear_joint_inverse",inverse,METH_VARARGS,"Joint theta-to-coordinate map."},
    {"_linear_joint_fit",fit,METH_VARARGS,"Typed joint native SLSQP solve."},
    {"_linear_joint_constraints",constraints,METH_VARARGS,"Shared joint theta-domain values and Jacobian."},
    {"_linear_joint_constraint_values",constraint_values,METH_VARARGS,"Joint theta-domain values without a Jacobian request."},
    {nullptr,nullptr,0,nullptr}
};
}

int register_linear_joint_methods(PyObject* module)
{ return PyModule_AddFunctions(module,methods); }
