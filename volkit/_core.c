// // volkit/_core.c
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "volkit_core.h"       /* prototypes */

#define BAD_ARITY(fname, want, got)                                  \
    PyErr_Format(PyExc_TypeError, fname "() takes %d positional "    \
                 "arguments but %zd were given",                     \
                 (want), (got));

                
static PyObject *
py_garch_variance_pq(PyObject *self,
                     PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 6) { BAD_ARITY("garch_variance_pq", 6, nargs); return NULL; }

    const double *theta  = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *eps2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t n  = PyLong_AsSize_t(args[3]);
    size_t p  = PyLong_AsSize_t(args[4]);
    size_t q  = PyLong_AsSize_t(args[5]);
    if (PyErr_Occurred()) return NULL;

    garch_variance_pq(theta, eps2, sigma2, n, p, q);
    Py_RETURN_NONE;
}

static PyObject *
py_garch_variance_11(PyObject *self,
                           PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("garch_variance_11", 4, nargs); return NULL; }

    const double *theta  = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *eps2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t n  = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;

    garch_variance_11(theta, eps2, sigma2, n);
    Py_RETURN_NONE;

}

static PyObject *
py_garch_ll_11_normal(PyObject *self,
                           PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("garch_ll_11_normal", 4, nargs); return NULL; }

    const double *theta  = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *eps2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t n  = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;

    double ll = garch_ll_11_normal(theta, eps2, sigma2, n);
    return PyFloat_FromDouble(ll);

}
static PyObject *
py_garch_ll_pq_normal(PyObject *self,
                           PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 6) { BAD_ARITY("garch_variance_pq", 6, nargs); return NULL; }

    const double *theta  = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *eps2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t n  = PyLong_AsSize_t(args[3]);
    size_t p  = PyLong_AsSize_t(args[4]);
    size_t q  = PyLong_AsSize_t(args[5]);
    if (PyErr_Occurred()) return NULL;

    double ll = garch_ll_pq_normal(theta, eps2, sigma2, n, p, q);
    return PyFloat_FromDouble(ll);

}




static PyObject *
py_normal_ll(PyObject *self,
                     PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3) { BAD_ARITY("normal_ll", 3, nargs); return NULL; }

    const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *eps2   = (const double *)PyLong_AsVoidPtr(args[1]);
    size_t n             = PyLong_AsSize_t(args[2]);
    if (PyErr_Occurred()) return NULL;

    double ll = normal_ll(sigma2, eps2, n);
    return PyFloat_FromDouble(ll);
}

static PyObject *
py_studentt_ll(PyObject *self,
PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("studentt_ll", 4, nargs); return NULL; }

    const double *sigma2   = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *r2os2    = (const double *)PyLong_AsVoidPtr(args[1]);
    const double nu       = (const double)      PyFloat_AsDouble(args[3]);
    size_t n = PyLong_AsSize_t(args[2]);
    if (PyErr_Occurred()) return NULL;

    double ll = studentt_ll(sigma2, r2os2, n, nu);
    return PyFloat_FromDouble(ll);

}

static PyObject *
py_garch_opg_hess_oq(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
    {
        if (nargs != 7) { BAD_ARITY("garch_opg_hess_oq", 7, nargs); return NULL; }
    
        const double *eps2   = (const double *)PyLong_AsVoidPtr(args[0]);
        const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[1]);
        double       *OPG    = (double *)      PyLong_AsVoidPtr(args[2]);
        double       *HESS   = (double *)      PyLong_AsVoidPtr(args[3]);
        size_t n  = PyLong_AsSize_t(args[4]);
        size_t p  = PyLong_AsSize_t(args[5]);
        size_t q  = PyLong_AsSize_t(args[6]);
        if (PyErr_Occurred()) return NULL;
    
        garch_opg_hess_oq(eps2, sigma2, OPG, HESS, n, p, q);
        Py_RETURN_NONE;
    
    }
    

static PyObject *
py_garch_opg_hess_11(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
    {
        if (nargs != 5) { BAD_ARITY("garch_opg_hess_11", 5, nargs); return NULL; }
    
        const double *eps2   = (const double *)PyLong_AsVoidPtr(args[0]);
        const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[1]);
        double       *OPG    = (double *)      PyLong_AsVoidPtr(args[2]);
        double       *HESS   = (double *)      PyLong_AsVoidPtr(args[3]);
        size_t n  = PyLong_AsSize_t(args[4]);
        if (PyErr_Occurred()) return NULL;
    
        garch_opg_hess_11(eps2, sigma2, OPG, HESS, n);
        Py_RETURN_NONE;
    
    }
    

/* ---- Method table & module init ----------------------------------- */
static PyMethodDef Methods[] = {
    {"_garch_variance_pq",      (PyCFunction)py_garch_variance_pq,
                               METH_FASTCALL, "Internal pointer API"},

    {"_garch_variance_11",      (PyCFunction)py_garch_variance_11,
                                METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_11_normal",     (PyCFunction)py_garch_ll_11_normal,
                                 METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_pq_normal",     (PyCFunction)py_garch_ll_pq_normal,
                                 METH_FASTCALL, "Internal pointer API"},

    {"_normal_ll",              (PyCFunction)py_normal_ll,

                               METH_FASTCALL, "Internal pointer API"},
    {"_studentt_ll",            (PyCFunction)py_studentt_ll,
                               METH_FASTCALL, "Internal pointer API"},

    {"_garch_opg_hess_oq", (PyCFunction)py_garch_opg_hess_oq,
                               METH_FASTCALL, "Internal pointer API"},

    {"_garch_opg_hess_11", (PyCFunction)py_garch_opg_hess_11,
                               METH_FASTCALL, "Internal pointer API"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "volkit._core",
    .m_doc  = "Internal pointer-level helpers",
    .m_size = -1,
    .m_methods = Methods,
};

PyMODINIT_FUNC
PyInit__core(void){ return PyModule_Create(&moddef); }