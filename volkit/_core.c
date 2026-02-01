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
py_skewt_ll(PyObject *self,
            PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 5) { BAD_ARITY("skewt_ll", 5, nargs); return NULL; }

    const double *resid  = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[1]);
    size_t n             = PyLong_AsSize_t(args[2]);
    const double nu      = PyFloat_AsDouble(args[3]);
    const double lam     = PyFloat_AsDouble(args[4]);
    if (PyErr_Occurred()) return NULL;

    double ll = skewt_ll(resid, sigma2, n, nu, lam);
    return PyFloat_FromDouble(ll);
}

static PyObject *
py_skewt_nll(PyObject *self,
             PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 5) { BAD_ARITY("skewt_nll", 5, nargs); return NULL; }

    const double *resid  = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[1]);
    size_t n             = PyLong_AsSize_t(args[2]);
    const double nu      = PyFloat_AsDouble(args[3]);
    const double lam     = PyFloat_AsDouble(args[4]);
    if (PyErr_Occurred()) return NULL;

    double nll = skewt_nll(resid, sigma2, n, nu, lam);
    return PyFloat_FromDouble(nll);
}

/* Skew-t NLL with gradient for GARCH(1,1) 
 * Takes returns data directly, computes h internally
 * theta = [omega, alpha, beta, nu, lam] */
static PyObject *
py_garch_ll_grad_11_skewt(PyObject *self,
                          PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("garch_ll_grad_11_skewt", 4, nargs); return NULL; }

    const double *theta  = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *grad   = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t n             = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;

    double nll = garch_ll_grad_11_skewt(theta, y, grad, n);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_garch_opg_hess_pq(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
    {
        if (nargs != 8) { BAD_ARITY("garch_opg_hess_pq", 8, nargs); return NULL; }
    
        const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
        const double *eps2   = (const double *)PyLong_AsVoidPtr(args[1]);
        const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[2]);
        double       *OPG    = (double *)      PyLong_AsVoidPtr(args[3]);
        double       *HESS   = (double *)      PyLong_AsVoidPtr(args[4]);
        size_t n  = PyLong_AsSize_t(args[5]);
        size_t p  = PyLong_AsSize_t(args[6]);
        size_t q  = PyLong_AsSize_t(args[7]);
        if (PyErr_Occurred()) return NULL;
    
        garch_opg_hess_pq(params, eps2, sigma2, OPG, HESS, n, p, q);
        Py_RETURN_NONE;
    
    }
 
static PyObject *
py_garch_opg_hess_11(PyObject *self,
    PyObject *const *args, Py_ssize_t nargs)
    {
        if (nargs != 6) { BAD_ARITY("garch_opg_hess_11", 6, nargs); return NULL; }
        
        const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
        const double *eps2   = (const double *)PyLong_AsVoidPtr(args[1]);
        const double *sigma2 = (const double *)PyLong_AsVoidPtr(args[2]);
        double       *OPG    = (double *)      PyLong_AsVoidPtr(args[3]);
        double       *HESS   = (double *)      PyLong_AsVoidPtr(args[4]);
        size_t n  = PyLong_AsSize_t(args[5]);
        if (PyErr_Occurred()) return NULL;
        
        garch_opg_hess_11(params, eps2, sigma2, OPG, HESS, n);
        Py_RETURN_NONE;
        
    }

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


/* --------------------------------------------------------------------------
 *  Student-t likelihood and derivatives
 * -------------------------------------------------------------------------*/

/* GARCH(1,1) | Student-t | log-likelihood */
static PyObject *
py_garch_ll_11_studentt(PyObject *self,
                        PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("garch_ll_11_studentt", 4, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2 = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t n             = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;

    double ll = garch_ll_11_studentt(params, resid2, sigma2, n);
    return PyFloat_FromDouble(ll);
}

/* GARCH(p,q) | Student-t | log-likelihood */
static PyObject *
py_garch_ll_pq_studentt(PyObject *self,
                        PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 6) { BAD_ARITY("garch_ll_pq_studentt", 6, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2 = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t n             = PyLong_AsSize_t(args[3]);
    size_t p             = PyLong_AsSize_t(args[4]);
    size_t q             = PyLong_AsSize_t(args[5]);
    if (PyErr_Occurred()) return NULL;

    double ll = garch_ll_pq_studentt(params, resid2, sigma2, n, p, q);
    return PyFloat_FromDouble(ll);
}

/* GARCH(1,1) | Student-t | gradient */
static PyObject *
py_garch_ll_grad_11_studentt(PyObject *self,
                             PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 5) { BAD_ARITY("garch_ll_grad_11_studentt", 5, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2 = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *grad   = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t n             = PyLong_AsSize_t(args[4]);
    if (PyErr_Occurred()) return NULL;

    garch_ll_grad_11_studentt(params, resid2, sigma2, grad, n);
    Py_RETURN_NONE;
}

/* GARCH(p,q) | Student-t | gradient */
static PyObject *
py_garch_ll_grad_pq_studentt(PyObject *self,
                             PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 7) { BAD_ARITY("garch_ll_grad_pq_studentt", 7, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2 = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *grad   = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t n             = PyLong_AsSize_t(args[4]);
    size_t p             = PyLong_AsSize_t(args[5]);
    size_t q             = PyLong_AsSize_t(args[6]);
    if (PyErr_Occurred()) return NULL;

    garch_ll_grad_pq_studentt(params, resid2, sigma2, grad, n, p, q);
    Py_RETURN_NONE;
}

/* GARCH(1,1) | Student-t | Hessian */
static PyObject *
py_garch_ll_hess_11_studentt(PyObject *self,
                             PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 5) { BAD_ARITY("garch_ll_hess_11_studentt", 5, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2 = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *hess   = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t n             = PyLong_AsSize_t(args[4]);
    if (PyErr_Occurred()) return NULL;

    garch_ll_hess_11_studentt(params, resid2, sigma2, hess, n);
    Py_RETURN_NONE;
}

/* GARCH(p,q) | Student-t | Hessian */
static PyObject *
py_garch_ll_hess_pq_studentt(PyObject *self,
                             PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 7) { BAD_ARITY("garch_ll_hess_pq_studentt", 7, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *resid2 = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *hess   = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t n             = PyLong_AsSize_t(args[4]);
    size_t p             = PyLong_AsSize_t(args[5]);
    size_t q             = PyLong_AsSize_t(args[6]);
    if (PyErr_Occurred()) return NULL;

    garch_ll_hess_pq_studentt(params, resid2, sigma2, hess, n, p, q);
    Py_RETURN_NONE;
}


/* ======================== Log-space transforms ========================== */

/* --- GARCH(1,1) specialized --- */
static PyObject *
py_pack_garch_11(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) { BAD_ARITY("pack_garch_11", 2, nargs); return NULL; }
    const double *z     = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *theta = (double *)      PyLong_AsVoidPtr(args[1]);
    if (PyErr_Occurred()) return NULL;
    pack_garch_11(z, theta);
    Py_RETURN_NONE;
}

static PyObject *
py_pack_garch_studentt_11(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) { BAD_ARITY("pack_garch_studentt_11", 2, nargs); return NULL; }
    const double *z     = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *theta = (double *)      PyLong_AsVoidPtr(args[1]);
    if (PyErr_Occurred()) return NULL;
    pack_garch_studentt_11(z, theta);
    Py_RETURN_NONE;
}

static PyObject *
py_pack_garch_skewt_11(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) { BAD_ARITY("pack_garch_skewt_11", 2, nargs); return NULL; }
    const double *z     = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *theta = (double *)      PyLong_AsVoidPtr(args[1]);
    if (PyErr_Occurred()) return NULL;
    pack_garch_skewt_11(z, theta);
    Py_RETURN_NONE;
}

static PyObject *
py_jacobian_garch_11(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) { BAD_ARITY("jacobian_garch_11", 2, nargs); return NULL; }
    const double *theta = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *J     = (double *)      PyLong_AsVoidPtr(args[1]);
    if (PyErr_Occurred()) return NULL;
    jacobian_garch_11(theta, J);
    Py_RETURN_NONE;
}

static PyObject *
py_jacobian_garch_studentt_11(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) { BAD_ARITY("jacobian_garch_studentt_11", 2, nargs); return NULL; }
    const double *theta = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *J     = (double *)      PyLong_AsVoidPtr(args[1]);
    if (PyErr_Occurred()) return NULL;
    jacobian_garch_studentt_11(theta, J);
    Py_RETURN_NONE;
}

static PyObject *
py_jacobian_garch_skewt_11(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) { BAD_ARITY("jacobian_garch_skewt_11", 2, nargs); return NULL; }
    const double *theta = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *J     = (double *)      PyLong_AsVoidPtr(args[1]);
    if (PyErr_Occurred()) return NULL;
    jacobian_garch_skewt_11(theta, J);
    Py_RETURN_NONE;
}

static PyObject *
py_transform_grad_11_normal(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3) { BAD_ARITY("transform_grad_11_normal", 3, nargs); return NULL; }
    const double *grad_theta = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *J          = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *grad_z     = (double *)      PyLong_AsVoidPtr(args[2]);
    if (PyErr_Occurred()) return NULL;
    transform_grad_11_normal(grad_theta, J, grad_z);
    Py_RETURN_NONE;
}

static PyObject *
py_transform_grad_11_studentt(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3) { BAD_ARITY("transform_grad_11_studentt", 3, nargs); return NULL; }
    const double *grad_theta = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *J          = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *grad_z     = (double *)      PyLong_AsVoidPtr(args[2]);
    if (PyErr_Occurred()) return NULL;
    transform_grad_11_studentt(grad_theta, J, grad_z);
    Py_RETURN_NONE;
}

static PyObject *
py_transform_grad_11_skewt(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3) { BAD_ARITY("transform_grad_11_skewt", 3, nargs); return NULL; }
    const double *grad_theta = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *J          = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *grad_z     = (double *)      PyLong_AsVoidPtr(args[2]);
    if (PyErr_Occurred()) return NULL;
    transform_grad_11_skewt(grad_theta, J, grad_z);
    Py_RETURN_NONE;
}

/* --- General GARCH(p,q) --- */
static PyObject *
py_pack_garch_pq(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("pack_garch_pq", 4, nargs); return NULL; }
    const double *z     = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *theta = (double *)      PyLong_AsVoidPtr(args[1]);
    size_t p = PyLong_AsSize_t(args[2]);
    size_t q = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;
    pack_garch_pq(z, theta, p, q);
    Py_RETURN_NONE;
}

static PyObject *
py_pack_garch_studentt_pq(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("pack_garch_studentt_pq", 4, nargs); return NULL; }
    const double *z     = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *theta = (double *)      PyLong_AsVoidPtr(args[1]);
    size_t p = PyLong_AsSize_t(args[2]);
    size_t q = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;
    pack_garch_studentt_pq(z, theta, p, q);
    Py_RETURN_NONE;
}

static PyObject *
py_pack_garch_skewt_pq(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("pack_garch_skewt_pq", 4, nargs); return NULL; }
    const double *z     = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *theta = (double *)      PyLong_AsVoidPtr(args[1]);
    size_t p = PyLong_AsSize_t(args[2]);
    size_t q = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;
    pack_garch_skewt_pq(z, theta, p, q);
    Py_RETURN_NONE;
}

static PyObject *
py_jacobian_garch_pq(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("jacobian_garch_pq", 4, nargs); return NULL; }
    const double *theta = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *J     = (double *)      PyLong_AsVoidPtr(args[1]);
    size_t p = PyLong_AsSize_t(args[2]);
    size_t q = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;
    jacobian_garch_pq(theta, J, p, q);
    Py_RETURN_NONE;
}

static PyObject *
py_jacobian_garch_studentt_pq(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("jacobian_garch_studentt_pq", 4, nargs); return NULL; }
    const double *theta = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *J     = (double *)      PyLong_AsVoidPtr(args[1]);
    size_t p = PyLong_AsSize_t(args[2]);
    size_t q = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;
    jacobian_garch_studentt_pq(theta, J, p, q);
    Py_RETURN_NONE;
}

static PyObject *
py_jacobian_garch_skewt_pq(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("jacobian_garch_skewt_pq", 4, nargs); return NULL; }
    const double *theta = (const double *)PyLong_AsVoidPtr(args[0]);
    double       *J     = (double *)      PyLong_AsVoidPtr(args[1]);
    size_t p = PyLong_AsSize_t(args[2]);
    size_t q = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;
    jacobian_garch_skewt_pq(theta, J, p, q);
    Py_RETURN_NONE;
}

static PyObject *
py_transform_grad_pq(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("transform_grad_pq", 4, nargs); return NULL; }
    const double *grad_theta = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *J          = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *grad_z     = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t K                 = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;
    transform_grad_pq(grad_theta, J, grad_z, K);
    Py_RETURN_NONE;
}

/* ---- Method table & module init ----------------------------------- */
/* ======================== ARMA-GARCH Wrappers ============================= */

static PyObject *
py_arma_garch_nll_11_normal(PyObject *self,
                            PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 6) { BAD_ARITY("arma_garch_nll_11_normal", 6, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[3]);
    double        h0     = PyFloat_AsDouble(args[4]);
    size_t        n      = PyLong_AsSize_t(args[5]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_garch_nll_11_normal(params, y, resid, sigma2, h0, n);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_garch_nll_grad_11_normal(PyObject *self,
                                  PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 7) { BAD_ARITY("arma_garch_nll_grad_11_normal", 7, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[3]);
    double       *grad   = (double *)      PyLong_AsVoidPtr(args[4]);
    double        h0     = PyFloat_AsDouble(args[5]);
    size_t        n      = PyLong_AsSize_t(args[6]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_garch_nll_grad_11_normal(params, y, resid, sigma2, grad, h0, n);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_garch_nll_11_studentt(PyObject *self,
                              PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 6) { BAD_ARITY("arma_garch_nll_11_studentt", 6, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[3]);
    double        h0     = PyFloat_AsDouble(args[4]);
    size_t        n      = PyLong_AsSize_t(args[5]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_garch_nll_11_studentt(params, y, resid, sigma2, h0, n);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_garch_nll_grad_11_studentt(PyObject *self,
                                   PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 7) { BAD_ARITY("arma_garch_nll_grad_11_studentt", 7, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[3]);
    double       *grad   = (double *)      PyLong_AsVoidPtr(args[4]);
    double        h0     = PyFloat_AsDouble(args[5]);
    size_t        n      = PyLong_AsSize_t(args[6]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_garch_nll_grad_11_studentt(params, y, resid, sigma2, grad, h0, n);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_garch_nll_11_skewt(PyObject *self,
                           PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 6) { BAD_ARITY("arma_garch_nll_11_skewt", 6, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[3]);
    double        h0     = PyFloat_AsDouble(args[4]);
    size_t        n      = PyLong_AsSize_t(args[5]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_garch_nll_11_skewt(params, y, resid, sigma2, h0, n);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_garch_nll_pq_normal(PyObject *self,
                            PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 11) { BAD_ARITY("arma_garch_nll_pq_normal", 11, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[3]);
    double       *e0     = (double *)      PyLong_AsVoidPtr(args[4]);
    double       *h0     = (double *)      PyLong_AsVoidPtr(args[5]);
    size_t        n      = PyLong_AsSize_t(args[6]);
    size_t        p_ar   = PyLong_AsSize_t(args[7]);
    size_t        q_ma   = PyLong_AsSize_t(args[8]);
    size_t        P_arch = PyLong_AsSize_t(args[9]);
    size_t        Q_garch= PyLong_AsSize_t(args[10]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_garch_nll_pq_normal(params, y, resid, sigma2, e0, h0, n, p_ar, q_ma, P_arch, Q_garch);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_garch_nll_pq_studentt(PyObject *self,
                              PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 11) { BAD_ARITY("arma_garch_nll_pq_studentt", 11, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[3]);
    double       *e0     = (double *)      PyLong_AsVoidPtr(args[4]);
    double       *h0     = (double *)      PyLong_AsVoidPtr(args[5]);
    size_t        n      = PyLong_AsSize_t(args[6]);
    size_t        p_ar   = PyLong_AsSize_t(args[7]);
    size_t        q_ma   = PyLong_AsSize_t(args[8]);
    size_t        P_arch = PyLong_AsSize_t(args[9]);
    size_t        Q_garch= PyLong_AsSize_t(args[10]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_garch_nll_pq_studentt(params, y, resid, sigma2, e0, h0, n, p_ar, q_ma, P_arch, Q_garch);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_garch_nll_pq_skewt(PyObject *self,
                           PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 11) { BAD_ARITY("arma_garch_nll_pq_skewt", 11, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *sigma2 = (double *)      PyLong_AsVoidPtr(args[3]);
    double       *e0     = (double *)      PyLong_AsVoidPtr(args[4]);
    double       *h0     = (double *)      PyLong_AsVoidPtr(args[5]);
    size_t        n      = PyLong_AsSize_t(args[6]);
    size_t        p_ar   = PyLong_AsSize_t(args[7]);
    size_t        q_ma   = PyLong_AsSize_t(args[8]);
    size_t        P_arch = PyLong_AsSize_t(args[9]);
    size_t        Q_garch= PyLong_AsSize_t(args[10]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_garch_nll_pq_skewt(params, y, resid, sigma2, e0, h0, n, p_ar, q_ma, P_arch, Q_garch);
    return PyFloat_FromDouble(nll);
}

/* ======================== Pure ARMA Functions ============================== */

static PyObject *
py_arma_nll_11_normal(PyObject *self,
                      PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) { BAD_ARITY("arma_nll_11_normal", 4, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    size_t        n      = PyLong_AsSize_t(args[3]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_nll_11_normal(params, y, resid, n);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_nll_grad_11_normal(PyObject *self,
                           PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 5) { BAD_ARITY("arma_nll_grad_11_normal", 5, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *grad   = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t        n      = PyLong_AsSize_t(args[4]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_nll_grad_11_normal(params, y, resid, grad, n);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_hess_11_normal(PyObject *self,
                       PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 5) { BAD_ARITY("arma_hess_11_normal", 5, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *hess   = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t        n      = PyLong_AsSize_t(args[4]);
    if (PyErr_Occurred()) return NULL;

    arma_hess_11_normal(params, y, resid, hess, n);
    Py_RETURN_NONE;
}

static PyObject *
py_arma_nll_pq_normal(PyObject *self,
                      PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 7) { BAD_ARITY("arma_nll_pq_normal", 7, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *e0     = (double *)      PyLong_AsVoidPtr(args[3]);
    size_t        n      = PyLong_AsSize_t(args[4]);
    size_t        p_ar   = PyLong_AsSize_t(args[5]);
    size_t        q_ma   = PyLong_AsSize_t(args[6]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_nll_pq_normal(params, y, resid, e0, n, p_ar, q_ma);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_nll_grad_pq_normal(PyObject *self,
                           PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 8) { BAD_ARITY("arma_nll_grad_pq_normal", 8, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *e0     = (double *)      PyLong_AsVoidPtr(args[3]);
    double       *grad   = (double *)      PyLong_AsVoidPtr(args[4]);
    size_t        n      = PyLong_AsSize_t(args[5]);
    size_t        p_ar   = PyLong_AsSize_t(args[6]);
    size_t        q_ma   = PyLong_AsSize_t(args[7]);
    if (PyErr_Occurred()) return NULL;

    double nll = arma_nll_grad_pq_normal(params, y, resid, e0, grad, n, p_ar, q_ma);
    return PyFloat_FromDouble(nll);
}

static PyObject *
py_arma_hess_pq_normal(PyObject *self,
                       PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 8) { BAD_ARITY("arma_hess_pq_normal", 8, nargs); return NULL; }

    const double *params = (const double *)PyLong_AsVoidPtr(args[0]);
    const double *y      = (const double *)PyLong_AsVoidPtr(args[1]);
    double       *resid  = (double *)      PyLong_AsVoidPtr(args[2]);
    double       *e0     = (double *)      PyLong_AsVoidPtr(args[3]);
    double       *hess   = (double *)      PyLong_AsVoidPtr(args[4]);
    size_t        n      = PyLong_AsSize_t(args[5]);
    size_t        p_ar   = PyLong_AsSize_t(args[6]);
    size_t        q_ma   = PyLong_AsSize_t(args[7]);
    if (PyErr_Occurred()) return NULL;

    arma_hess_pq_normal(params, y, resid, e0, hess, n, p_ar, q_ma);
    Py_RETURN_NONE;
}

/* ======================== Method Table ===================================== */

static PyMethodDef Methods[] = {

    // GARCH | Normal | Core
    {"_garch_variance_pq",       (PyCFunction)py_garch_variance_pq,
                                  METH_FASTCALL, "Internal pointer API"},

    {"_garch_variance_11",       (PyCFunction)py_garch_variance_11,
                                  METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_11_normal",      (PyCFunction)py_garch_ll_11_normal,
                                  METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_pq_normal",      (PyCFunction)py_garch_ll_pq_normal,
                                  METH_FASTCALL, "Internal pointer API"},

    // GARCH | Normal | Asymptotics
    {"_garch_ll_grad_11_normal", (PyCFunction)py_garch_ll_grad_11_normal,
                                  METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_hess_11_normal", (PyCFunction)py_garch_ll_hess_11_normal,
                                  METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_grad_pq_normal", (PyCFunction)py_garch_ll_grad_pq_normal,
                                  METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_hess_pq_normal", (PyCFunction)py_garch_ll_hess_pq_normal,
                                  METH_FASTCALL, "Internal pointer API"},

    // GARCH | Student-t | Core
    {"_garch_ll_11_studentt",    (PyCFunction)py_garch_ll_11_studentt,
                                  METH_FASTCALL, "Internal pointer API"},

    {"_garch_ll_pq_studentt",    (PyCFunction)py_garch_ll_pq_studentt,
                                  METH_FASTCALL, "Internal pointer API"},

    // GARCH | Student-t | Asymptotics
    {"_garch_ll_grad_11_studentt",(PyCFunction)py_garch_ll_grad_11_studentt,
                                  METH_FASTCALL, "Internal pointer API"},
    {"_garch_ll_hess_11_studentt",(PyCFunction)py_garch_ll_hess_11_studentt,
                                  METH_FASTCALL, "Internal pointer API"},
    {"_garch_ll_grad_pq_studentt",(PyCFunction)py_garch_ll_grad_pq_studentt,
                                  METH_FASTCALL, "Internal pointer API"},
    {"_garch_ll_hess_pq_studentt",(PyCFunction)py_garch_ll_hess_pq_studentt,
                                  METH_FASTCALL, "Internal pointer API"},

                                  
    // Miscellaneous
    {"_normal_ll",               (PyCFunction)py_normal_ll,

                                  METH_FASTCALL, "Internal pointer API"},
    {"_studentt_ll",             (PyCFunction)py_studentt_ll,
                                  METH_FASTCALL, "Internal pointer API"},
    {"_skewt_ll",                (PyCFunction)py_skewt_ll,
                                  METH_FASTCALL, "Hansen (1994) Skew-t log-likelihood"},
    {"_skewt_nll",               (PyCFunction)py_skewt_nll,
                                  METH_FASTCALL, "Hansen (1994) Skew-t negative log-likelihood"},
    {"_garch_ll_grad_11_skewt",  (PyCFunction)py_garch_ll_grad_11_skewt,
                                  METH_FASTCALL, "GARCH(1,1) + Skew-t NLL with gradient"},

    // OLD                            
    {"_garch_opg_hess_pq",       (PyCFunction)py_garch_opg_hess_pq,
                                  METH_FASTCALL, "Internal pointer API"},

    {"_garch_opg_hess_11",       (PyCFunction)py_garch_opg_hess_11,
                                  METH_FASTCALL, "Internal pointer API"},                                 

    // Log-space transforms | GARCH(1,1) specialized
    {"_pack_garch_11",           (PyCFunction)py_pack_garch_11,
                                  METH_FASTCALL, "z -> theta transform for GARCH(1,1)"},
    {"_pack_garch_studentt_11",  (PyCFunction)py_pack_garch_studentt_11,
                                  METH_FASTCALL, "z -> theta transform for GARCH(1,1)+StudentT"},
    {"_pack_garch_skewt_11",     (PyCFunction)py_pack_garch_skewt_11,
                                  METH_FASTCALL, "z -> theta transform for GARCH(1,1)+SkewT"},
    {"_jacobian_garch_11",       (PyCFunction)py_jacobian_garch_11,
                                  METH_FASTCALL, "Jacobian for GARCH(1,1)"},
    {"_jacobian_garch_studentt_11",(PyCFunction)py_jacobian_garch_studentt_11,
                                  METH_FASTCALL, "Jacobian for GARCH(1,1)+StudentT"},
    {"_jacobian_garch_skewt_11", (PyCFunction)py_jacobian_garch_skewt_11,
                                  METH_FASTCALL, "Jacobian for GARCH(1,1)+SkewT"},
    {"_transform_grad_11_normal",(PyCFunction)py_transform_grad_11_normal,
                                  METH_FASTCALL, "J^T @ grad for K=3"},
    {"_transform_grad_11_studentt",(PyCFunction)py_transform_grad_11_studentt,
                                  METH_FASTCALL, "J^T @ grad for K=4"},
    {"_transform_grad_11_skewt", (PyCFunction)py_transform_grad_11_skewt,
                                  METH_FASTCALL, "J^T @ grad for K=5"},

    // Log-space transforms | General GARCH(p,q)
    {"_pack_garch_pq",           (PyCFunction)py_pack_garch_pq,
                                  METH_FASTCALL, "z -> theta transform for GARCH(p,q)"},
    {"_pack_garch_studentt_pq",  (PyCFunction)py_pack_garch_studentt_pq,
                                  METH_FASTCALL, "z -> theta transform for GARCH(p,q)+StudentT"},
    {"_pack_garch_skewt_pq",     (PyCFunction)py_pack_garch_skewt_pq,
                                  METH_FASTCALL, "z -> theta transform for GARCH(p,q)+SkewT"},
    {"_jacobian_garch_pq",       (PyCFunction)py_jacobian_garch_pq,
                                  METH_FASTCALL, "Jacobian for GARCH(p,q)"},
    {"_jacobian_garch_studentt_pq",(PyCFunction)py_jacobian_garch_studentt_pq,
                                  METH_FASTCALL, "Jacobian for GARCH(p,q)+StudentT"},
    {"_jacobian_garch_skewt_pq", (PyCFunction)py_jacobian_garch_skewt_pq,
                                  METH_FASTCALL, "Jacobian for GARCH(p,q)+SkewT"},
    {"_transform_grad_pq",       (PyCFunction)py_transform_grad_pq,
                                  METH_FASTCALL, "J^T @ grad for general K"},

    // ARMA-GARCH functions
    {"_arma_garch_nll_11_normal",      (PyCFunction)py_arma_garch_nll_11_normal,
                                        METH_FASTCALL, "ARMA(1,1)-GARCH(1,1) NLL with Normal"},
    {"_arma_garch_nll_grad_11_normal", (PyCFunction)py_arma_garch_nll_grad_11_normal,
                                        METH_FASTCALL, "ARMA(1,1)-GARCH(1,1) NLL+Gradient with Normal"},
    {"_arma_garch_nll_11_studentt",    (PyCFunction)py_arma_garch_nll_11_studentt,
                                        METH_FASTCALL, "ARMA(1,1)-GARCH(1,1) NLL with Student-t"},
    {"_arma_garch_nll_grad_11_studentt", (PyCFunction)py_arma_garch_nll_grad_11_studentt,
                                        METH_FASTCALL, "ARMA(1,1)-GARCH(1,1) NLL+Gradient with Student-t"},
    {"_arma_garch_nll_11_skewt",       (PyCFunction)py_arma_garch_nll_11_skewt,
                                        METH_FASTCALL, "ARMA(1,1)-GARCH(1,1) NLL with Skew-t"},
    {"_arma_garch_nll_pq_normal",      (PyCFunction)py_arma_garch_nll_pq_normal,
                                        METH_FASTCALL, "ARMA(p,q)-GARCH(P,Q) NLL with Normal"},
    {"_arma_garch_nll_pq_studentt",    (PyCFunction)py_arma_garch_nll_pq_studentt,
                                        METH_FASTCALL, "ARMA(p,q)-GARCH(P,Q) NLL with Student-t"},
    {"_arma_garch_nll_pq_skewt",       (PyCFunction)py_arma_garch_nll_pq_skewt,
                                        METH_FASTCALL, "ARMA(p,q)-GARCH(P,Q) NLL with Skew-t"},

    // Pure ARMA functions (no volatility dynamics)
    {"_arma_nll_11_normal",           (PyCFunction)py_arma_nll_11_normal,
                                        METH_FASTCALL, "ARMA(1,1) NLL with Normal (concentrated)"},
    {"_arma_nll_grad_11_normal",      (PyCFunction)py_arma_nll_grad_11_normal,
                                        METH_FASTCALL, "ARMA(1,1) NLL+Gradient with Normal"},
    {"_arma_hess_11_normal",          (PyCFunction)py_arma_hess_11_normal,
                                        METH_FASTCALL, "ARMA(1,1) Hessian with Normal"},
    {"_arma_nll_pq_normal",           (PyCFunction)py_arma_nll_pq_normal,
                                        METH_FASTCALL, "ARMA(p,q) NLL with Normal (concentrated)"},
    {"_arma_nll_grad_pq_normal",      (PyCFunction)py_arma_nll_grad_pq_normal,
                                        METH_FASTCALL, "ARMA(p,q) NLL+Gradient with Normal"},
    {"_arma_hess_pq_normal",          (PyCFunction)py_arma_hess_pq_normal,
                                        METH_FASTCALL, "ARMA(p,q) Hessian with Normal"},

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