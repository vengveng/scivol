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
py_garch_opg_hess_pq(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
    {
        if (nargs != 7) { BAD_ARITY("garch_opg_hess_pq", 7, nargs); return NULL; }
    
        const double *eps2   = (const double *)PyLong_AsVoidPtr(args[0]);
        const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[1]);
        double       *OPG    = (double *)      PyLong_AsVoidPtr(args[2]);
        double       *HESS   = (double *)      PyLong_AsVoidPtr(args[3]);
        size_t n  = PyLong_AsSize_t(args[4]);
        size_t p  = PyLong_AsSize_t(args[5]);
        size_t q  = PyLong_AsSize_t(args[6]);
        if (PyErr_Occurred()) return NULL;
    
        garch_opg_hess_pq(eps2, sigma2, OPG, HESS, n, p, q);
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
    
// __attribute__((visibility("default"), hot, flatten))
// void garch_ll_grad_hess_pq_normal(
//         const double * __restrict params,
//         const double * __restrict resid2,
//         double       * __restrict sigma2,   /* n      */
//         double       * __restrict grad,     /* K      */
//         double       * __restrict hess,     /* K×K    */
//         double       * __restrict nll,      /* scalar */
//         size_t n,
//         size_t p,
//         size_t q) {

static PyObject *py_garch_ll_grad_hess_pq_normal(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 9) { BAD_ARITY("garch_ll_grad_hess_pq_normal", 9, nargs); return NULL; }

    const double *params   = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2   = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *grad     = (double *)      PyLong_AsVoidPtr(args[3]);
    double       *hess     = (double *)      PyLong_AsVoidPtr(args[4]);
    double       *nll      = (double *)      PyLong_AsVoidPtr(args[5]);
    size_t n  = PyLong_AsSize_t(args[6]);
    size_t p  = PyLong_AsSize_t(args[7]);
    size_t q  = PyLong_AsSize_t(args[8]);
    if (PyErr_Occurred()) return NULL;

    garch_ll_grad_hess_pq_normal(params, resid2, sigma2, grad, hess, nll, n, p, q);
    Py_RETURN_NONE;
}
        
static PyObject *py_garch_ll_grad_hess_11_normal(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
    {
        if (nargs != 7) { BAD_ARITY("garch_ll_grad_hess_11_normal", 7, nargs); return NULL; }
        
        const double *params   = (const double *)PyLong_AsVoidPtr(args[0]);
        const double *resid2   = (const double *)PyLong_AsVoidPtr(args[1]);
        double       *sigma2   = (double *)      PyLong_AsVoidPtr(args[2]);
        double       *grad     = (double *)      PyLong_AsVoidPtr(args[3]);
        double       *hess     = (double *)      PyLong_AsVoidPtr(args[4]);
        double       *nll      = (double *)      PyLong_AsVoidPtr(args[5]);
    size_t n  = PyLong_AsSize_t(args[6]);
    if (PyErr_Occurred()) return NULL;

    garch_ll_grad_hess_11_normal(params, resid2, sigma2, grad, hess, nll, n);
    Py_RETURN_NONE;
    }

// __attribute__((visibility("default"), hot, flatten))
// void garch_ll_grad_11_normal(
//         const double * __restrict params,   /* [ω, α, β]          */
//         const double * __restrict resid2,   /* ε_t², length n     */
//         double       * __restrict sigma2,   /* working buffer n   */
//         double       * __restrict grad,     /* output length 3    */
//         size_t n)

// __attribute__((visibility("default"), hot, flatten))
// void garch_ll_hess_11_normal(
//         const double * __restrict params,   /* [ω, α, β]          */
//         const double * __restrict resid2,   /* ε_t², length n     */
//         double       * __restrict sigma2,   /* working buffer n   */
//         double       * __restrict hess,     /* output 3 × 3 row-major */
//         size_t n)

static PyObject *py_garch_ll_hess_11_normal(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 5) { BAD_ARITY("garch_ll_hess_11_normal", 5, nargs); return NULL; }

    const double *params   = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2   = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *hess     = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t n  = PyLong_AsSize_t(args[4]);
    if (PyErr_Occurred()) return NULL;

    garch_ll_hess_11_normal(params, resid2, sigma2, hess, n);
    Py_RETURN_NONE;
}

static PyObject *py_garch_ll_grad_11_normal(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 5) { BAD_ARITY("garch_ll_grad_11_normal", 5, nargs); return NULL; }

    const double *params   = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2   = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *grad     = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t n  = PyLong_AsSize_t(args[4]);
    if (PyErr_Occurred()) return NULL;

    garch_ll_grad_11_normal(params, resid2, sigma2, grad, n);
    Py_RETURN_NONE;
}

// __attribute__((visibility("default"), hot, flatten))
// void garch_ll_grad_pq_normal(
//         const double * __restrict params,
//         const double * __restrict resid2,
//         double       * __restrict sigma2,
//         double       * __restrict grad,
//         size_t n,
//         size_t p,
//         size_t q)

// __attribute__((visibility("default"), hot, flatten))
// void garch_ll_hess_pq_normal(
//         const double * __restrict params,
//         const double * __restrict resid2,
//         double       * __restrict sigma2,
//         double       * __restrict hess,
//         size_t n,
//         size_t p,
//         size_t q)

static PyObject *py_garch_ll_grad_pq_normal(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 7) { BAD_ARITY("garch_ll_grad_pq_normal", 7, nargs); return NULL; }

    const double *params   = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2   = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *grad     = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t n  = PyLong_AsSize_t(args[4]);
    size_t p  = PyLong_AsSize_t(args[5]);
    size_t q  = PyLong_AsSize_t(args[6]);
    if (PyErr_Occurred()) return NULL;

    garch_ll_grad_pq_normal(params, resid2, sigma2, grad, n, p, q);
    Py_RETURN_NONE;
}

static PyObject *py_garch_ll_hess_pq_normal(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 7) { BAD_ARITY("garch_ll_hess_pq_normal", 7, nargs); return NULL; }

    const double *params   = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2   = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2   = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *hess     = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t n  = PyLong_AsSize_t(args[4]);
    size_t p  = PyLong_AsSize_t(args[5]);
    size_t q  = PyLong_AsSize_t(args[6]);
    if (PyErr_Occurred()) return NULL;

    garch_ll_hess_pq_normal(params, resid2, sigma2, hess, n, p, q);
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

    {"_garch_opg_hess_pq", (PyCFunction)py_garch_opg_hess_pq,
                               METH_FASTCALL, "Internal pointer API"},

    {"_garch_opg_hess_11", (PyCFunction)py_garch_opg_hess_11,
                               METH_FASTCALL, "Internal pointer API"},





    {"_garch_ll_grad_hess_pq_normal", (PyCFunction)py_garch_ll_grad_hess_pq_normal,
                               METH_FASTCALL, "Internal pointer API"},
                               
    {"_garch_ll_grad_hess_11_normal", (PyCFunction)py_garch_ll_grad_hess_11_normal,
                                 METH_FASTCALL, "Internal pointer API"},    

    {"_garch_ll_grad_11_normal", (PyCFunction)py_garch_ll_grad_11_normal,
                                 METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_hess_11_normal", (PyCFunction)py_garch_ll_hess_11_normal,
                                    METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_grad_pq_normal", (PyCFunction)py_garch_ll_grad_pq_normal,
                                 METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_hess_pq_normal", (PyCFunction)py_garch_ll_hess_pq_normal,
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