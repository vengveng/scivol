#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "volkit_core.h"       /* prototypes */

#define BAD_ARITY(fname, want, got)                                  \
    PyErr_Format(PyExc_TypeError, fname "() takes %d positional "    \
                 "arguments but %zd were given",                     \
                 (want), (got));

/* ---- 1) void garch_variance_pq ------------------------------------ */
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

/* ---- 2) double normal_likelihood ---------------------------------- */
static PyObject *
py_normal_likelihood(PyObject *self,
                     PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3) { BAD_ARITY("normal_likelihood", 3, nargs); return NULL; }

    const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *eps2   = (const double *)PyLong_AsVoidPtr(args[1]);
    size_t n             = PyLong_AsSize_t(args[2]);
    if (PyErr_Occurred()) return NULL;

    double ll = normal_likelihood(sigma2, eps2, n);
    return PyFloat_FromDouble(ll);
}

static PyObject *
py_special_garch_oo_normal(PyObject *self,
                           PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("special_garch_oo_normal", 4, nargs); return NULL; }

    const double *theta  = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *eps2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t n  = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;

    double ll = special_garch_oo_normal(theta, eps2, sigma2, n);
    return PyFloat_FromDouble(ll);

}

static PyObject *
py_special_garch_oo_normal_variance(PyObject *self,
                                    PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("special_garch_oo_normal_variance", 4, nargs); return NULL; }

    const double *theta  = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *eps2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t n  = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;

    special_garch_oo_normal_variance(theta, eps2, sigma2, n);
    Py_RETURN_NONE;

}

static PyObject *
py_general_garch_pq_std_err_robust(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
    {
        if (nargs != 7) { BAD_ARITY("general_garch_pq_std_err_robust", 7, nargs); return NULL; }
    
        const double *eps2   = (const double *)PyLong_AsVoidPtr(args[0]);
        const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[1]);
        double       *OPG    = (double *)      PyLong_AsVoidPtr(args[2]);
        double       *HESS   = (double *)      PyLong_AsVoidPtr(args[3]);
        size_t n  = PyLong_AsSize_t(args[4]);
        size_t p  = PyLong_AsSize_t(args[5]);
        size_t q  = PyLong_AsSize_t(args[6]);
        if (PyErr_Occurred()) return NULL;
    
        general_garch_pq_std_err_robust(eps2, sigma2, OPG, HESS, n, p, q);
        Py_RETURN_NONE;
    
    }
    

static PyObject *
py_special_garch_11_std_err_robust(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
    {
        if (nargs != 5) { BAD_ARITY("special_garch_11_std_err_robust", 5, nargs); return NULL; }
    
        const double *eps2   = (const double *)PyLong_AsVoidPtr(args[0]);
        const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[1]);
        double       *OPG    = (double *)      PyLong_AsVoidPtr(args[2]);
        double       *HESS   = (double *)      PyLong_AsVoidPtr(args[3]);
        size_t n  = PyLong_AsSize_t(args[4]);
        if (PyErr_Occurred()) return NULL;
    
        special_garch_11_std_err_robust(eps2, sigma2, OPG, HESS, n);
        Py_RETURN_NONE;
    
    }
    
    
static PyObject *
py_any_studentt_likelihood(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
    {
        if (nargs != 4) { BAD_ARITY("any_studentt_likelihood", 4, nargs); return NULL; }
    
        const double *sigma2   = (const double *)PyLong_AsVoidPtr(args[0]);
        const double *r2os2    = (const double *)PyLong_AsVoidPtr(args[1]);
        const double nu       = (const double)      PyFloat_AsDouble(args[3]);
        size_t n = PyLong_AsSize_t(args[2]);
        if (PyErr_Occurred()) return NULL;
    
        double ll = any_studentt_likelihood(sigma2, r2os2, n, nu);
        return PyFloat_FromDouble(ll);
    
    }

/* ---- Method table & module init ----------------------------------- */
static PyMethodDef Methods[] = {
    {"_garch_variance_pq",      (PyCFunction)py_garch_variance_pq,
                               METH_FASTCALL, "Internal pointer API"},
    {"_normal_likelihood",      (PyCFunction)py_normal_likelihood,
                               METH_FASTCALL, "Internal pointer API"},
    /* add the rest here */
    {"_special_garch_oo_normal", (PyCFunction)py_special_garch_oo_normal,
                               METH_FASTCALL, "Internal pointer API"},
    {"_special_garch_oo_normal_variance", (PyCFunction)py_special_garch_oo_normal_variance,
                               METH_FASTCALL, "Internal pointer API"},
    {"_general_garch_pq_std_err_robust", (PyCFunction)py_general_garch_pq_std_err_robust,
                               METH_FASTCALL, "Internal pointer API"},
    {"_special_garch_11_std_err_robust", (PyCFunction)py_special_garch_11_std_err_robust,
                               METH_FASTCALL, "Internal pointer API"},
    {"_any_studentt_likelihood", (PyCFunction)py_any_studentt_likelihood,
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