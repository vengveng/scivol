#ifndef SCIVOL_GH_MATH_H
#define SCIVOL_GH_MATH_H

#include <stddef.h>

#define GH_DERIV_DIM 5
#define GH_FAMILY_INTERIOR 0
#define GH_FAMILY_VARIANCE_GAMMA 1

enum {
    GH_DERIV_E = 0,
    GH_DERIV_H = 1,
    GH_DERIV_RHO = 2,
    GH_DERIV_ZETA = 3,
    GH_DERIV_LAMBDA = 4
};

typedef struct {
    double value;
    double grad[GH_DERIV_DIM];
    double hess[GH_DERIV_DIM * GH_DERIV_DIM];
    int second_order;
} gh_jet_t;

typedef struct {
    gh_jet_t rho;
    gh_jet_t zeta;
    gh_jet_t lambda;
    gh_jet_t alpha;
    gh_jet_t beta;
    gh_jet_t delta;
    gh_jet_t mu;
    gh_jet_t log_normalizer;
    gh_jet_t vg_theta;
    gh_jet_t vg_scale;
    gh_jet_t vg_argument_scale;
    int family_kind;
    int estimate_lambda;
} gh_shape_cache_t;

typedef struct {
    double value;
    double grad[GH_DERIV_DIM];
    double hess[GH_DERIV_DIM * GH_DERIV_DIM];
} gh_obs_derivs_t;

typedef struct {
    double value;
    double grad[GH_DERIV_DIM];
} gh_obs_gradient_t;

int gh_shape_cache_init(double rho, double zeta, double lambda,
                        int estimate_lambda, gh_shape_cache_t *cache);

int vg_shape_cache_init(double rho, double lambda,
                        gh_shape_cache_t *cache);

int gh_family_cache_init(const double *params, size_t shape_start,
                         double fixed_lambda, gh_shape_cache_t *cache);

int gh_obs_derivatives(double residual, double variance,
                       const gh_shape_cache_t *cache,
                       gh_obs_derivs_t *out);

int gh_obs_gradient(double residual, double variance,
                    const gh_shape_cache_t *cache,
                    gh_obs_gradient_t *out);

double gh_obs_nll_scalar(double residual, double variance,
                         const gh_shape_cache_t *cache);

int gh_abs_moment_derivatives(const gh_shape_cache_t *cache,
                              double *moment, double *grad, double *hess);

double gh_standardized_logpdf_scalar(double x, double rho, double zeta,
                                     double lambda);

double gh_standardized_nll(const double *x, size_t n, double rho,
                           double zeta, double lambda);

void gh_standardized_nll_grad_hess(const double *x, size_t n, double rho,
                                   double zeta, double lambda,
                                   int estimate_lambda,
                                   double *nll, double *grad, double *hess);

double vg_standardized_logpdf_scalar(double x, double rho, double lambda);

double vg_standardized_nll(const double *x, size_t n,
                           double rho, double lambda);

void vg_standardized_nll_grad_hess(const double *x, size_t n,
                                   double rho, double lambda,
                                   double *nll, double *grad, double *hess);

#endif
