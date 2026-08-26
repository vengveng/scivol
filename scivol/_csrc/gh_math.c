#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "gh_high_order_quadrature.h"
#include "gh_math.h"
#include "math_and_helpers.h"

#define GH_QUAD_HALF 64
#define GH_QUAD_MIDPOINT 8.0
#define GH_LOG_2 0.69314718055994530942
#define GH_LOG_PI_OVER_2 0.45158270528945486473
#define GH_HALF_PI 1.57079632679489661923
#define VG_LAMBDA_MIN 1.51
#define VG_LAMBDA_MAX 50.0

static const double GH_NODES[GH_QUAD_HALF] = {
    1.222369896061576387e-02, 3.666379096873349785e-02, 6.108196960413957155e-02, 8.546364050451549232e-02,
    1.097942311276437488e-01, 1.340591994611878002e-01, 1.582440427142249306e-01, 1.823343059853371750e-01,
    2.063155909020792100e-01, 2.301735642266599935e-01, 2.538939664226942949e-01, 2.774626201779044066e-01,
    3.008654388776771937e-01, 3.240884350244133683e-01, 3.471177285976355109e-01, 3.699395553498590061e-01,
    3.925402750332674384e-01, 4.149063795522749953e-01, 4.370245010371041627e-01, 4.588814198335521843e-01,
    4.804640724041719935e-01, 5.017595591361444507e-01, 5.227551520511755090e-01, 5.434383024128103257e-01,
    5.637966482266181245e-01, 5.838180216287630619e-01, 6.034904561585485716e-01, 6.228021939105848892e-01,
    6.417416925623075086e-01, 6.602976322726460401e-01, 6.784589224477192149e-01, 6.962147083695143790e-01,
    7.135543776835874263e-01, 7.304675667419088247e-01, 7.469441667970619925e-01, 7.629743300440947618e-01,
    7.785484755064119300e-01, 7.936572947621932661e-01, 8.082917575079137107e-01, 8.224431169556438714e-01,
    8.361029150609068372e-01, 8.492629875779689197e-01, 8.619154689395485125e-01, 8.740527969580318413e-01,
    8.856677173453971763e-01, 8.967532880491582148e-01, 9.073028834017567990e-01, 9.173101980809605926e-01,
    9.267692508789479033e-01, 9.356743882779163801e-01, 9.440202878302201839e-01, 9.518019613412643798e-01,
    9.590147578536999395e-01, 9.656543664319652276e-01, 9.717168187471365570e-01, 9.771984914639073594e-01,
    9.820961084357184845e-01, 9.864067427245861719e-01, 9.901278184917343683e-01, 9.932571129002129018e-01,
    9.957927585349811705e-01, 9.977332486255140465e-01, 9.990774599773758480e-01, 9.998248879471318684e-01,
};

static const double GH_WEIGHTS[GH_QUAD_HALF] = {
    2.444618019626242436e-02, 2.443156909784994016e-02, 2.440235563384947348e-02, 2.435855726469051200e-02,
    2.430020016797176594e-02, 2.422731922281515585e-02, 2.413995798901917914e-02, 2.403816868102394769e-02,
    2.392201213670334947e-02, 2.379155778100327037e-02, 2.364688358444753946e-02, 2.348807601653577631e-02,
    2.331522999406268962e-02, 2.312844882438691921e-02, 2.292784414368674584e-02, 2.271353585023641597e-02,
    2.248565203274485030e-02, 2.224432889379970163e-02, 2.198971066846042502e-02, 2.172194953805202389e-02,
    2.144120553920841193e-02, 2.114764646822127336e-02, 2.084144778075098375e-02, 2.052279248696007100e-02,
    2.019187104212986916e-02, 1.984888123283080549e-02, 1.949402805870655009e-02, 1.912752360995090620e-02,
    1.874958694054473102e-02, 1.836044393733124794e-02, 1.796032718500876688e-02, 1.754947582711756729e-02,
    1.712813542311122580e-02, 1.669655780158906325e-02, 1.625500090978507575e-02, 1.580372865939914592e-02,
    1.534301076886531931e-02, 1.487312260214741430e-02, 1.439434500416653280e-02, 1.390696413295178582e-02,
    1.341127128861625828e-02, 1.290756273926747452e-02, 1.239613954395058990e-02, 1.187730737273994495e-02,
    1.135137632408065277e-02, 1.081866073950285730e-02, 1.027947901583225500e-02, 9.734153415006684129e-03,
    9.183009871660277601e-03, 8.626377798616769246e-03, 8.064589890486088125e-03, 7.497981925634576716e-03,
    6.926892566898738487e-03, 6.351663161706925208e-03, 5.772637542865975864e-03, 5.190161832676730078e-03,
    4.604584256703736500e-03, 4.016254983739319523e-03, 3.425526040910785094e-03, 2.832751471459144833e-03,
    2.238288430963638644e-03, 1.642503018668153413e-03, 1.045812679344212959e-03, 4.493809602893054683e-04,
};

typedef struct {
    const double *nodes;
    const double *weights;
    size_t half_count;
} gh_quadrature_rule_t;

static gh_quadrature_rule_t quadrature_rule(double order)
{
    gh_quadrature_rule_t rule;
    if (fabs(order) > 15.0) {
        rule.nodes = GH_HIGH_NODES;
        rule.weights = GH_HIGH_WEIGHTS;
        rule.half_count = GH_HIGH_QUAD_HALF;
    } else {
        rule.nodes = GH_NODES;
        rule.weights = GH_WEIGHTS;
        rule.half_count = GH_QUAD_HALF;
    }
    return rule;
}

static gh_jet_t jet_constant(double value)
{
    gh_jet_t out;
    memset(&out, 0, sizeof(out));
    out.value = value;
    out.second_order = 0;
    return out;
}

static gh_jet_t jet_variable(double value, size_t index)
{
    gh_jet_t out = jet_constant(value);
    out.grad[index] = 1.0;
    out.second_order = 1;
    return out;
}

static gh_jet_t jet_first_variable(double value, size_t index)
{
    gh_jet_t out = jet_constant(value);
    out.grad[index] = 1.0;
    return out;
}

static gh_jet_t jet_add(gh_jet_t a, gh_jet_t b)
{
    gh_jet_t out;
    out.second_order = a.second_order || b.second_order;
    int active_gradient = 0;
    out.value = a.value + b.value;
    for (size_t i = 0; i < GH_DERIV_DIM; ++i) {
        out.grad[i] = a.grad[i] + b.grad[i];
        active_gradient = active_gradient || out.grad[i] != 0.0;
        if (out.second_order) {
            for (size_t j = 0; j < GH_DERIV_DIM; ++j) {
                out.hess[i * GH_DERIV_DIM + j] =
                    a.hess[i * GH_DERIV_DIM + j] + b.hess[i * GH_DERIV_DIM + j];
            }
        }
    }
    if (!out.second_order && !active_gradient) {
        memset(out.hess, 0, sizeof(out.hess));
    }
    return out;
}

static gh_jet_t jet_scale(gh_jet_t a, double scale)
{
    gh_jet_t out;
    out.second_order = a.second_order;
    int active_gradient = 0;
    out.value = a.value * scale;
    for (size_t i = 0; i < GH_DERIV_DIM; ++i) {
        out.grad[i] = a.grad[i] * scale;
        active_gradient = active_gradient || out.grad[i] != 0.0;
        if (out.second_order) {
            for (size_t j = 0; j < GH_DERIV_DIM; ++j) {
                out.hess[i * GH_DERIV_DIM + j] =
                    a.hess[i * GH_DERIV_DIM + j] * scale;
            }
        }
    }
    if (!out.second_order && !active_gradient) {
        memset(out.hess, 0, sizeof(out.hess));
    }
    return out;
}

static gh_jet_t jet_sub(gh_jet_t a, gh_jet_t b)
{
    return jet_add(a, jet_scale(b, -1.0));
}

static gh_jet_t jet_mul(gh_jet_t a, gh_jet_t b)
{
    gh_jet_t out;
    out.second_order = a.second_order || b.second_order;
    int active_gradient = 0;
    out.value = a.value * b.value;
    for (size_t i = 0; i < GH_DERIV_DIM; ++i) {
        out.grad[i] = a.grad[i] * b.value + a.value * b.grad[i];
        active_gradient = active_gradient || out.grad[i] != 0.0;
        if (out.second_order) {
            for (size_t j = 0; j < GH_DERIV_DIM; ++j) {
                out.hess[i * GH_DERIV_DIM + j] =
                    a.hess[i * GH_DERIV_DIM + j] * b.value
                    + a.grad[i] * b.grad[j]
                    + a.grad[j] * b.grad[i]
                    + a.value * b.hess[i * GH_DERIV_DIM + j];
            }
        }
    }
    if (!out.second_order && !active_gradient) {
        memset(out.hess, 0, sizeof(out.hess));
    }
    return out;
}

static gh_jet_t jet_unary(gh_jet_t x, double value,
                          double first, double second)
{
    gh_jet_t out;
    out.second_order = x.second_order;
    int active_gradient = 0;
    out.value = value;
    for (size_t i = 0; i < GH_DERIV_DIM; ++i) {
        out.grad[i] = first * x.grad[i];
        active_gradient = active_gradient || out.grad[i] != 0.0;
        if (out.second_order) {
            for (size_t j = 0; j < GH_DERIV_DIM; ++j) {
                out.hess[i * GH_DERIV_DIM + j] =
                    second * x.grad[i] * x.grad[j]
                    + first * x.hess[i * GH_DERIV_DIM + j];
            }
        }
    }
    if (!out.second_order && !active_gradient) {
        memset(out.hess, 0, sizeof(out.hess));
    }
    return out;
}

static gh_jet_t jet_exp(gh_jet_t x)
{
    const double value = exp(x.value);
    return jet_unary(x, value, value, value);
}

static gh_jet_t jet_log(gh_jet_t x)
{
    const double inv = 1.0 / x.value;
    return jet_unary(x, log(x.value), inv, -inv * inv);
}

static gh_jet_t jet_sqrt(gh_jet_t x)
{
    const double value = sqrt(x.value);
    const double first = 0.5 / value;
    return jet_unary(x, value, first, -0.25 / (x.value * value));
}

static gh_jet_t jet_abs(gh_jet_t x)
{
    const double sign = x.value < 0.0 ? -1.0 : 1.0;
    return jet_unary(x, fabs(x.value), sign, 0.0);
}

static gh_jet_t jet_lgamma(gh_jet_t x)
{
    return jet_unary(
        x,
        lgamma(x.value),
        digamma_approx(x.value),
        trigamma_approx(x.value)
    );
}

static gh_jet_t jet_reciprocal(gh_jet_t x)
{
    const double inv = 1.0 / x.value;
    return jet_unary(x, inv, -inv * inv, 2.0 * inv * inv * inv);
}

static gh_jet_t jet_div(gh_jet_t numerator, gh_jet_t denominator)
{
    return jet_mul(numerator, jet_reciprocal(denominator));
}

static gh_jet_t jet_log_cosh(gh_jet_t x)
{
    const double abs_x = fabs(x.value);
    const double value = abs_x + log1p(exp(-2.0 * abs_x)) - GH_LOG_2;
    const double first = tanh(x.value);
    const double second = 1.0 - first * first;
    return jet_unary(x, value, first, second);
}

static double log_cosh_scalar(double x)
{
    const double abs_x = fabs(x);
    return abs_x + log1p(exp(-2.0 * abs_x)) - GH_LOG_2;
}

#define K1_APPROX_INTERVALS 5
#define K1_APPROX_DEGREE 14

static const double K1_LOG_EDGES[K1_APPROX_INTERVALS + 1] = {
    -2.302585092994045457e+00, -9.210340371976180052e-01,
     4.605170185988094467e-01,  1.842068074395237343e+00,
     3.223619130191664350e+00,  4.605170185988091802e+00,
};

/* Power coefficients in a normalized log-argument coordinate. */
static const double K1_LOG_COEFFICIENTS[K1_APPROX_INTERVALS][K1_APPROX_DEGREE + 1] = {
    {7.339731325491116731e-01, -2.580771205535690727e-01, 2.093027002239103429e-02, 2.153294013353623370e-03, -9.809336117997837889e-05, -5.238113080602127997e-05, -2.934921781697141336e-06, 9.349317664992912687e-07, 1.637458997568277436e-07, -6.559571090508332699e-09, -4.756115442878575572e-09, -3.184713935541117579e-10, 8.517105014484763611e-11, 1.575462824499516479e-11, 3.773696184686163883e-13},
    {3.154859804371253351e-01, -1.558631438123275481e-01, 2.743668115487723005e-02, -4.874581097902679896e-04, -4.411205805825196466e-04, 1.586499455760258029e-05, 1.076840161250465336e-05, -5.022142744956296172e-07, -2.873259761464207957e-07, 1.566021640867067801e-08, 7.937120816927631507e-09, -4.861681757902438184e-10, -2.214708822165959677e-10, 1.358580424585061300e-11, 5.424254471041117015e-12},
    {1.036222793922116497e-01, -6.319716090886028093e-02, 1.706519827179850055e-02, -2.286907022153183283e-03, 3.200369019376560279e-05, 4.175115317055437915e-05, -5.530043259570215700e-06, -3.572910939462568611e-07, 1.739428495837013552e-07, -1.030960216711120116e-08, -3.209562047253827844e-09, 6.027903277917042140e-10, 1.745654952259487998e-11, -1.589097411092496818e-11, 6.028005649198247308e-13},
    {2.867909079098182601e-02, -1.909245195448371415e-02, 6.127882124399593351e-03, -1.215089655398709839e-03, 1.509054549702120373e-04, -7.676326153554602436e-06, -1.271087636673979493e-06, 3.534602155771285743e-07, -3.534632518449416401e-08, -9.037575102543527258e-10, 8.354658356088732422e-10, -1.188658218794311920e-10, 2.898436877455049975e-12, 2.123445006265137596e-12, -4.224569392998386350e-13},
    {7.408859472518876735e-03, -5.068030026549820780e-03, 1.716592164714370039e-03, -3.800635184640114520e-04, 6.058109050241543210e-05, -7.050688766209480787e-06, 5.333096119545277487e-07, -4.600734563483697350e-09, -6.281389917565724931e-09, 1.186771759716875112e-09, -1.227217218248561531e-10, 2.979780515375761602e-12, 3.578216694109912610e-13, 4.549490733851806399e-13, 1.084199009602850240e-13},
};

static int log_bessel_k1_fast(double x, double *value,
                              double *first, double *second)
{
    if (!(x >= 0.1) || !isfinite(x)) return 0;
    double correction;
    double correction_first;
    double correction_second;
    if (x > 100.0) {
        const double inv = 1.0 / x;
        const double inv2 = inv * inv;
        const double inv3 = inv2 * inv;
        const double inv4 = inv3 * inv;
        const double series = 1.0 + 0.375 * inv
            - 0.1171875 * inv2 + 0.1025390625 * inv3
            - 0.144195556640625 * inv4;
        const double series_first = -0.375 * inv2
            + 0.234375 * inv3 - 0.3076171875 * inv4
            + 0.5767822265625 * inv4 * inv;
        const double series_second = 0.75 * inv3
            - 0.703125 * inv4 + 1.23046875 * inv4 * inv
            - 2.8839111328125 * inv4 * inv2;
        correction = log(series);
        correction_first = series_first / series;
        correction_second = series_second / series
            - correction_first * correction_first;
    } else {
        const double u = log(x);
        size_t interval = 0;
        while (interval + 1 < K1_APPROX_INTERVALS
               && u > K1_LOG_EDGES[interval + 1]) {
            ++interval;
        }
        const double width = K1_LOG_EDGES[interval + 1]
                           - K1_LOG_EDGES[interval];
        const double t = 2.0 * (u - K1_LOG_EDGES[interval]) / width - 1.0;
        const double *coeff = K1_LOG_COEFFICIENTS[interval];
        double polynomial = coeff[K1_APPROX_DEGREE];
        double derivative = 0.0;
        double second_derivative = 0.0;
        for (size_t degree = K1_APPROX_DEGREE; degree-- > 0;) {
            second_derivative = second_derivative * t + 2.0 * derivative;
            derivative = derivative * t + polynomial;
            polynomial = polynomial * t + coeff[degree];
        }
        const double scale = 2.0 / width;
        const double derivative_u = derivative * scale;
        const double second_u = second_derivative * scale * scale;
        correction = polynomial;
        correction_first = derivative_u / x;
        correction_second = (second_u - derivative_u) / (x * x);
    }
    *value = -x - 0.5 * log(x) + 0.5 * GH_LOG_PI_OVER_2 + correction;
    *first = -1.0 - 0.5 / x + correction_first;
    *second = 0.5 / (x * x) + correction_second;
    return isfinite(*value) && isfinite(*first) && isfinite(*second);
}

static double log_bessel_k_scalar(double order, double argument)
{
    if (fabs(fabs(order) - 0.5) < 1e-14) {
        return 0.5 * (GH_LOG_PI_OVER_2 - log(argument)) - argument;
    }
    if (fabs(fabs(order) - 1.0) < 1e-14) {
        double value, first, second;
        if (log_bessel_k1_fast(argument, &value, &first, &second)) return value;
    }
    const gh_quadrature_rule_t rule = quadrature_rule(order);
    double max_term = -INFINITY;
    for (size_t i = 0; i < rule.half_count; ++i) {
        const double offset = GH_QUAD_MIDPOINT * rule.nodes[i];
        const double log_weight = log(GH_QUAD_MIDPOINT * rule.weights[i]);
        const double t_left = GH_QUAD_MIDPOINT - offset;
        const double t_right = GH_QUAD_MIDPOINT + offset;
        const double left = log_weight - argument * cosh(t_left)
                          + log_cosh_scalar(order * t_left);
        const double right = log_weight - argument * cosh(t_right)
                           + log_cosh_scalar(order * t_right);
        if (left > max_term) max_term = left;
        if (right > max_term) max_term = right;
    }

    double sum = 0.0;
    for (size_t i = 0; i < rule.half_count; ++i) {
        const double offset = GH_QUAD_MIDPOINT * rule.nodes[i];
        const double log_weight = log(GH_QUAD_MIDPOINT * rule.weights[i]);
        const double t_left = GH_QUAD_MIDPOINT - offset;
        const double t_right = GH_QUAD_MIDPOINT + offset;
        sum += exp(log_weight - argument * cosh(t_left)
                 + log_cosh_scalar(order * t_left) - max_term);
        sum += exp(log_weight - argument * cosh(t_right)
                 + log_cosh_scalar(order * t_right) - max_term);
    }
    return max_term + log(sum);
}

static void log_bessel_k_moments(
    double order, double argument, int second_order,
    double *value, double *d_order, double *d_argument,
    double *d_order_order, double *d_order_argument,
    double *d_argument_argument)
{
    const gh_quadrature_rule_t rule = quadrature_rule(order);
    double max_term = -INFINITY;
    for (size_t i = 0; i < rule.half_count; ++i) {
        const double offset = GH_QUAD_MIDPOINT * rule.nodes[i];
        const double log_weight = log(GH_QUAD_MIDPOINT * rule.weights[i]);
        const double t_values[2] = {
            GH_QUAD_MIDPOINT - offset,
            GH_QUAD_MIDPOINT + offset,
        };
        for (size_t side = 0; side < 2; ++side) {
            const double t = t_values[side];
            const double term = log_weight - argument * cosh(t)
                              + log_cosh_scalar(order * t);
            if (term > max_term) max_term = term;
        }
    }

    double weight_sum = 0.0;
    double order_sum = 0.0;
    double argument_sum = 0.0;
    double order_square_sum = 0.0;
    double order_curvature_sum = 0.0;
    double cross_sum = 0.0;
    double argument_square_sum = 0.0;
    for (size_t i = 0; i < rule.half_count; ++i) {
        const double offset = GH_QUAD_MIDPOINT * rule.nodes[i];
        const double log_weight = log(GH_QUAD_MIDPOINT * rule.weights[i]);
        const double t_values[2] = {
            GH_QUAD_MIDPOINT - offset,
            GH_QUAD_MIDPOINT + offset,
        };
        for (size_t side = 0; side < 2; ++side) {
            const double t = t_values[side];
            const double cosh_t = cosh(t);
            const double tanh_order_t = tanh(order * t);
            const double order_score = t * tanh_order_t;
            const double argument_score = -cosh_t;
            const double weight = exp(
                log_weight - argument * cosh_t
                + log_cosh_scalar(order * t) - max_term
            );
            weight_sum += weight;
            order_sum += weight * order_score;
            argument_sum += weight * argument_score;
            if (second_order) {
                order_square_sum += weight * order_score * order_score;
                order_curvature_sum += weight * t * t
                    * (1.0 - tanh_order_t * tanh_order_t);
                cross_sum += weight * order_score * argument_score;
                argument_square_sum += weight * argument_score * argument_score;
            }
        }
    }

    const double inv_sum = 1.0 / weight_sum;
    *value = max_term + log(weight_sum);
    *d_order = order_sum * inv_sum;
    *d_argument = argument_sum * inv_sum;
    if (second_order) {
        *d_order_order = (order_square_sum + order_curvature_sum) * inv_sum
                       - *d_order * *d_order;
        *d_order_argument = cross_sum * inv_sum
                          - *d_order * *d_argument;
        *d_argument_argument = argument_square_sum * inv_sum
                             - *d_argument * *d_argument;
    }
}

static gh_jet_t jet_log_bessel_k(gh_jet_t order, gh_jet_t argument)
{
    int fixed_order = 1;
    for (size_t i = 0; i < GH_DERIV_DIM; ++i) {
        if (order.grad[i] != 0.0) {
            fixed_order = 0;
            break;
        }
    }
    if (fixed_order && fabs(fabs(order.value) - 0.5) < 1e-14) {
        return jet_sub(
            jet_scale(
                jet_sub(jet_constant(GH_LOG_PI_OVER_2), jet_log(argument)),
                0.5
            ),
            argument
        );
    }
    if (fixed_order && fabs(fabs(order.value) - 1.0) < 1e-14) {
        double value, first, second;
        if (log_bessel_k1_fast(argument.value, &value, &first, &second)) {
            return jet_unary(argument, value, first, second);
        }
    }
    gh_jet_t out;
    out.second_order = order.second_order || argument.second_order;
    double d_order, d_argument;
    double d_order_order = 0.0;
    double d_order_argument = 0.0;
    double d_argument_argument = 0.0;
    log_bessel_k_moments(
        order.value, argument.value, out.second_order,
        &out.value, &d_order, &d_argument,
        &d_order_order, &d_order_argument, &d_argument_argument
    );
    for (size_t i = 0; i < GH_DERIV_DIM; ++i) {
        out.grad[i] = d_order * order.grad[i]
                    + d_argument * argument.grad[i];
        if (out.second_order) {
            for (size_t j = 0; j < GH_DERIV_DIM; ++j) {
                out.hess[i * GH_DERIV_DIM + j] =
                    d_order_order * order.grad[i] * order.grad[j]
                    + d_order_argument
                      * (order.grad[i] * argument.grad[j]
                         + argument.grad[i] * order.grad[j])
                    + d_argument_argument
                      * argument.grad[i] * argument.grad[j]
                    + d_order * order.hess[i * GH_DERIV_DIM + j]
                    + d_argument * argument.hess[i * GH_DERIV_DIM + j];
            }
        }
    }
    return out;
}

static int valid_shapes(double rho, double zeta, double lambda)
{
    return isfinite(rho) && isfinite(zeta) && isfinite(lambda)
        && fabs(rho) < 0.995 && zeta >= 0.1 && zeta <= 25.0
        && lambda >= -6.0 && lambda <= 6.0;
}

static int scalar_standardized_parameters(double rho, double zeta, double lambda,
                                          double *alpha, double *beta,
                                          double *delta, double *mu)
{
    if (!valid_shapes(rho, zeta, lambda)) return 0;
    const double rho2 = 1.0 - rho * rho;
    const double log_k0 = log_bessel_k_scalar(lambda, zeta);
    const double kappa = exp(log_bessel_k_scalar(lambda + 1.0, zeta) - log_k0) / zeta;
    const double kappa_next = exp(
        log_bessel_k_scalar(lambda + 2.0, zeta)
        - log_bessel_k_scalar(lambda + 1.0, zeta)
    ) / zeta;
    const double delta_kappa = kappa_next - kappa;
    const double alpha_sq = zeta * zeta * kappa
        * (1.0 + rho * rho * zeta * zeta * delta_kappa / rho2) / rho2;
    if (!(alpha_sq > 0.0) || !isfinite(alpha_sq)) return 0;
    *alpha = sqrt(alpha_sq);
    *beta = *alpha * rho;
    *delta = zeta / (*alpha * sqrt(rho2));
    *mu = -(*beta) * (*delta) * (*delta) * kappa;
    return isfinite(*alpha) && isfinite(*beta) && isfinite(*delta) && isfinite(*mu);
}

int gh_shape_cache_init(double rho, double zeta, double lambda,
                        int estimate_lambda, gh_shape_cache_t *cache)
{
    if (!cache || !valid_shapes(rho, zeta, lambda)) return 0;
    memset(cache, 0, sizeof(*cache));
    cache->family_kind = GH_FAMILY_INTERIOR;
    cache->rho = jet_variable(rho, GH_DERIV_RHO);
    cache->zeta = jet_variable(zeta, GH_DERIV_ZETA);
    cache->lambda = estimate_lambda
        ? jet_variable(lambda, GH_DERIV_LAMBDA)
        : jet_constant(lambda);
    cache->estimate_lambda = estimate_lambda;

    const gh_jet_t one = jet_constant(1.0);
    const gh_jet_t rho2 = jet_sub(one, jet_mul(cache->rho, cache->rho));
    const gh_jet_t log_k0 = jet_log_bessel_k(cache->lambda, cache->zeta);
    const gh_jet_t log_k1 = jet_log_bessel_k(
        jet_add(cache->lambda, one), cache->zeta
    );
    const gh_jet_t log_k2 = jet_log_bessel_k(
        jet_add(cache->lambda, jet_constant(2.0)), cache->zeta
    );
    const gh_jet_t kappa = jet_div(jet_exp(jet_sub(log_k1, log_k0)), cache->zeta);
    const gh_jet_t kappa_next = jet_div(jet_exp(jet_sub(log_k2, log_k1)), cache->zeta);
    const gh_jet_t delta_kappa = jet_sub(kappa_next, kappa);
    const gh_jet_t zeta2 = jet_mul(cache->zeta, cache->zeta);
    const gh_jet_t rho_sq = jet_mul(cache->rho, cache->rho);
    const gh_jet_t skew_adjustment = jet_add(
        one,
        jet_div(jet_mul(jet_mul(rho_sq, zeta2), delta_kappa), rho2)
    );
    const gh_jet_t alpha_sq = jet_div(
        jet_mul(jet_mul(zeta2, kappa), skew_adjustment), rho2
    );
    if (!(alpha_sq.value > 0.0) || !isfinite(alpha_sq.value)) return 0;
    cache->alpha = jet_sqrt(alpha_sq);
    cache->beta = jet_mul(cache->alpha, cache->rho);
    cache->delta = jet_div(cache->zeta, jet_mul(cache->alpha, jet_sqrt(rho2)));
    cache->mu = jet_scale(
        jet_mul(jet_mul(jet_mul(cache->beta, cache->delta), cache->delta), kappa),
        -1.0
    );

    const gh_jet_t gamma = jet_sqrt(jet_sub(
        jet_mul(cache->alpha, cache->alpha),
        jet_mul(cache->beta, cache->beta)
    ));
    cache->log_normalizer = jet_sub(
        jet_sub(
            jet_mul(cache->lambda, jet_sub(jet_log(gamma), jet_log(cache->delta))),
            jet_constant(0.5 * LOG_2PI)
        ),
        jet_log_bessel_k(cache->lambda, jet_mul(cache->delta, gamma))
    );
    return isfinite(cache->log_normalizer.value);
}

int vg_shape_cache_init(double rho, double lambda, gh_shape_cache_t *cache)
{
    if (!cache || !isfinite(rho) || fabs(rho) >= 0.995
        || !isfinite(lambda) || lambda < VG_LAMBDA_MIN
        || lambda > VG_LAMBDA_MAX) return 0;
    memset(cache, 0, sizeof(*cache));
    cache->family_kind = GH_FAMILY_VARIANCE_GAMMA;
    cache->rho = jet_variable(rho, GH_DERIV_RHO);
    cache->zeta = jet_variable(lambda, GH_DERIV_ZETA);

    const gh_jet_t one = jet_constant(1.0);
    const gh_jet_t rho2 = jet_sub(one, jet_mul(cache->rho, cache->rho));
    cache->vg_theta = jet_div(
        jet_scale(cache->rho, sqrt(2.0)), jet_sqrt(rho2)
    );
    const gh_jet_t theta2 = jet_mul(cache->vg_theta, cache->vg_theta);
    cache->vg_scale = jet_sqrt(
        jet_mul(cache->zeta, jet_add(one, theta2))
    );
    cache->vg_argument_scale = jet_sqrt(
        jet_add(theta2, jet_constant(2.0))
    );
    return isfinite(cache->vg_scale.value)
        && isfinite(cache->vg_argument_scale.value);
}

int gh_family_cache_init(const double *params, size_t shape_start,
                         double fixed_lambda, gh_shape_cache_t *cache)
{
    if (isinf(fixed_lambda)) {
        return vg_shape_cache_init(
            params[shape_start], params[shape_start + 1], cache
        );
    }
    const int estimate_lambda = isnan(fixed_lambda);
    const double lambda = estimate_lambda
        ? params[shape_start + 2] : fixed_lambda;
    return gh_shape_cache_init(
        params[shape_start], params[shape_start + 1], lambda,
        estimate_lambda, cache
    );
}

static gh_jet_t vg_logpdf_jet(gh_jet_t x, const gh_shape_cache_t *cache)
{
    const gh_jet_t half = jet_constant(0.5);
    const gh_jet_t order = jet_sub(cache->zeta, half);
    const gh_jet_t y = jet_add(
        jet_mul(cache->vg_scale, x),
        jet_mul(cache->vg_theta, cache->zeta)
    );
    gh_jet_t logpdf = jet_add(
        jet_sub(
            jet_add(
                jet_log(cache->vg_scale),
                jet_constant(GH_LOG_2 - 0.5 * LOG_2PI)
            ),
            jet_lgamma(cache->zeta)
        ),
        jet_mul(cache->vg_theta, y)
    );

    if (cache->vg_argument_scale.value * fabs(y.value) < 1e-4) {
        const gh_jet_t center = jet_add(
            jet_add(
                jet_scale(jet_sub(order, jet_constant(1.0)), GH_LOG_2),
                jet_lgamma(order)
            ),
            jet_scale(jet_mul(order, jet_log(cache->vg_argument_scale)), -2.0)
        );
        const gh_jet_t curvature = jet_div(
            jet_mul(cache->vg_argument_scale, cache->vg_argument_scale),
            jet_scale(jet_sub(jet_constant(1.0), order), 2.0)
        );
        return jet_add(
            jet_add(logpdf, center),
            jet_scale(jet_mul(curvature, jet_mul(y, y)), 0.5)
        );
    }

    const gh_jet_t abs_y = jet_abs(y);
    const gh_jet_t argument = jet_mul(cache->vg_argument_scale, abs_y);
    const gh_jet_t bessel_product = jet_add(
        jet_mul(
            order,
            jet_sub(jet_log(abs_y), jet_log(cache->vg_argument_scale))
        ),
        jet_log_bessel_k(order, argument)
    );
    return jet_add(logpdf, bessel_product);
}

static void shape_cache_use_first_order(gh_shape_cache_t *cache)
{
    gh_jet_t *fields[] = {
        &cache->rho, &cache->zeta, &cache->lambda,
        &cache->alpha, &cache->beta, &cache->delta,
        &cache->mu, &cache->log_normalizer, &cache->vg_theta,
        &cache->vg_scale, &cache->vg_argument_scale,
    };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        fields[i]->second_order = 0;
    }
}

static int gh_obs_jet(double residual, double variance,
                      const gh_shape_cache_t *cache,
                      int second_order, gh_jet_t *out)
{
    if (!cache || !out || !isfinite(residual) || !isfinite(variance)) return 0;
    gh_shape_cache_t first_cache;
    const gh_shape_cache_t *active_cache = cache;
    if (!second_order) {
        first_cache = *cache;
        shape_cache_use_first_order(&first_cache);
        active_cache = &first_cache;
    }
    const double h_safe = variance > H_FLOOR ? variance : H_FLOOR;
    const gh_jet_t e = second_order
        ? jet_variable(residual, GH_DERIV_E)
        : jet_first_variable(residual, GH_DERIV_E);
    const gh_jet_t h = second_order
        ? jet_variable(h_safe, GH_DERIV_H)
        : jet_first_variable(h_safe, GH_DERIV_H);
    const gh_jet_t z = jet_div(e, jet_sqrt(h));
    if (active_cache->family_kind == GH_FAMILY_VARIANCE_GAMMA) {
        const gh_jet_t nll = jet_sub(
            jet_scale(jet_log(h), 0.5), vg_logpdf_jet(z, active_cache)
        );
        if (!isfinite(nll.value)) return 0;
        *out = nll;
        return 1;
    }
    const gh_jet_t centered = jet_sub(z, active_cache->mu);
    const gh_jet_t radius = jet_sqrt(jet_add(
        jet_mul(active_cache->delta, active_cache->delta),
        jet_mul(centered, centered)
    ));
    const gh_jet_t argument = jet_mul(active_cache->alpha, radius);
    const gh_jet_t log_kernel = jet_sub(
        jet_add(
            jet_mul(active_cache->beta, centered),
            jet_log_bessel_k(
                jet_sub(active_cache->lambda, jet_constant(0.5)), argument
            )
        ),
        jet_mul(
            jet_sub(jet_constant(0.5), active_cache->lambda),
            jet_sub(jet_log(radius), jet_log(active_cache->alpha))
        )
    );
    const gh_jet_t nll = jet_sub(
        jet_scale(jet_log(h), 0.5),
        jet_add(active_cache->log_normalizer, log_kernel)
    );
    if (!isfinite(nll.value)) return 0;
    *out = nll;
    return 1;
}

int gh_obs_derivatives(double residual, double variance,
                       const gh_shape_cache_t *cache,
                       gh_obs_derivs_t *out)
{
    gh_jet_t nll;
    if (!out || !gh_obs_jet(residual, variance, cache, 1, &nll)) return 0;
    out->value = nll.value;
    memcpy(out->grad, nll.grad, sizeof(out->grad));
    memcpy(out->hess, nll.hess, sizeof(out->hess));
    return 1;
}

int gh_obs_gradient(double residual, double variance,
                    const gh_shape_cache_t *cache,
                    gh_obs_gradient_t *out)
{
    gh_jet_t nll;
    if (!out || !gh_obs_jet(residual, variance, cache, 0, &nll)) return 0;
    out->value = nll.value;
    memcpy(out->grad, nll.grad, sizeof(out->grad));
    return 1;
}

double gh_obs_nll_scalar(double residual, double variance,
                         const gh_shape_cache_t *cache)
{
    if (!cache || !isfinite(residual) || !isfinite(variance)) return INFINITY;
    const double h = variance > H_FLOOR ? variance : H_FLOOR;
    const double z = residual / sqrt(h);
    if (cache->family_kind == GH_FAMILY_VARIANCE_GAMMA) {
        return 0.5 * log(h) - vg_standardized_logpdf_scalar(
            z, cache->rho.value, cache->zeta.value
        );
    }
    const double centered = z - cache->mu.value;
    const double radius = hypot(cache->delta.value, centered);
    const double argument = cache->alpha.value * radius;
    const double log_kernel = cache->beta.value * centered
        + log_bessel_k_scalar(cache->lambda.value - 0.5, argument)
        - (0.5 - cache->lambda.value)
          * (log(radius) - log(cache->alpha.value));
    return 0.5 * log(h) - cache->log_normalizer.value - log_kernel;
}

int gh_abs_moment_derivatives(const gh_shape_cache_t *cache,
                              double *moment, double *grad, double *hess)
{
    if (!cache || !moment || !grad || !hess) return 0;
    *moment = 0.0;
    memset(grad, 0, 3 * sizeof(double));
    memset(hess, 0, 9 * sizeof(double));
    const size_t source[3] = {
        GH_DERIV_RHO, GH_DERIV_ZETA, GH_DERIV_LAMBDA
    };

    for (size_t i = 0; i < GH_QUAD_HALF; ++i) {
        const double unit_nodes[2] = {
            0.5 * (1.0 - GH_NODES[i]),
            0.5 * (1.0 + GH_NODES[i]),
        };
        for (size_t node = 0; node < 2; ++node) {
            const double x_abs = tan(GH_HALF_PI * unit_nodes[node]);
            const double jacobian = GH_HALF_PI * (1.0 + x_abs * x_abs);
            for (size_t sign = 0; sign < 2; ++sign) {
                const double x = sign == 0 ? -x_abs : x_abs;
                gh_obs_derivs_t obs;
                if (!gh_obs_derivatives(x, 1.0, cache, &obs)) return 0;
                const double contribution = 0.5 * GH_WEIGHTS[i] * jacobian
                                          * x_abs * exp(-obs.value);
                *moment += contribution;
                for (size_t s = 0; s < 3; ++s) {
                    const double d_s = obs.grad[source[s]];
                    grad[s] -= contribution * d_s;
                    for (size_t r = 0; r < 3; ++r) {
                        const double d_r = obs.grad[source[r]];
                        const double d2 = obs.hess[
                            source[s] * GH_DERIV_DIM + source[r]
                        ];
                        hess[s * 3 + r] += contribution * (d_s * d_r - d2);
                    }
                }
            }
        }
    }
    return isfinite(*moment);
}

double gh_standardized_logpdf_scalar(double x, double rho, double zeta,
                                     double lambda)
{
    double alpha, beta, delta, mu;
    if (!scalar_standardized_parameters(
            rho, zeta, lambda, &alpha, &beta, &delta, &mu)) {
        return -INFINITY;
    }
    const double centered = x - mu;
    const double radius = hypot(delta, centered);
    const double gamma = sqrt(alpha * alpha - beta * beta);
    const double argument = alpha * radius;
    return lambda * (log(gamma) - log(delta))
        - 0.5 * LOG_2PI
        - log_bessel_k_scalar(lambda, delta * gamma)
        + beta * centered
        + log_bessel_k_scalar(lambda - 0.5, argument)
        - (0.5 - lambda) * (log(radius) - log(alpha));
}

double gh_standardized_nll(const double *x, size_t n, double rho,
                           double zeta, double lambda)
{
    double nll = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double logpdf = gh_standardized_logpdf_scalar(x[i], rho, zeta, lambda);
        if (!isfinite(logpdf)) return 1e10;
        nll -= logpdf;
    }
    return nll;
}

void gh_standardized_nll_grad_hess(const double *x, size_t n, double rho,
                                   double zeta, double lambda,
                                   int estimate_lambda,
                                   double *nll, double *grad, double *hess)
{
    gh_shape_cache_t cache;
    const size_t shape_count = estimate_lambda ? 3 : 2;
    if (nll) *nll = 1e10;
    if (grad) memset(grad, 0, shape_count * sizeof(double));
    if (hess) memset(hess, 0, shape_count * shape_count * sizeof(double));
    if (!gh_shape_cache_init(rho, zeta, lambda, estimate_lambda, &cache)) return;

    double total = 0.0;
    for (size_t t = 0; t < n; ++t) {
        gh_obs_derivs_t obs;
        if (!gh_obs_derivatives(x[t], 1.0, &cache, &obs)) return;
        total += obs.value;
        if (grad) {
            grad[0] += obs.grad[GH_DERIV_RHO];
            grad[1] += obs.grad[GH_DERIV_ZETA];
            if (estimate_lambda) grad[2] += obs.grad[GH_DERIV_LAMBDA];
        }
        if (hess) {
            const size_t source[3] = {
                GH_DERIV_RHO, GH_DERIV_ZETA, GH_DERIV_LAMBDA
            };
            for (size_t i = 0; i < shape_count; ++i) {
                for (size_t j = 0; j < shape_count; ++j) {
                    hess[i * shape_count + j] +=
                        obs.hess[source[i] * GH_DERIV_DIM + source[j]];
                }
            }
        }
    }
    if (nll) *nll = total;
}

double vg_standardized_logpdf_scalar(double x, double rho, double lambda)
{
    gh_shape_cache_t cache;
    if (!vg_shape_cache_init(rho, lambda, &cache)) return -INFINITY;
    gh_jet_t x_jet = jet_constant(x);
    return vg_logpdf_jet(x_jet, &cache).value;
}

double vg_standardized_nll(const double *x, size_t n,
                           double rho, double lambda)
{
    double nll = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double logpdf = vg_standardized_logpdf_scalar(
            x[i], rho, lambda
        );
        if (!isfinite(logpdf)) return 1e10;
        nll -= logpdf;
    }
    return nll;
}

void vg_standardized_nll_grad_hess(const double *x, size_t n,
                                   double rho, double lambda,
                                   double *nll, double *grad, double *hess)
{
    gh_shape_cache_t cache;
    if (nll) *nll = 1e10;
    if (grad) memset(grad, 0, 2 * sizeof(double));
    if (hess) memset(hess, 0, 4 * sizeof(double));
    if (!vg_shape_cache_init(rho, lambda, &cache)) return;

    const size_t source[2] = {GH_DERIV_RHO, GH_DERIV_ZETA};
    double total = 0.0;
    for (size_t t = 0; t < n; ++t) {
        gh_obs_derivs_t obs;
        if (!gh_obs_derivatives(x[t], 1.0, &cache, &obs)) return;
        total += obs.value;
        if (grad) {
            for (size_t i = 0; i < 2; ++i) grad[i] += obs.grad[source[i]];
        }
        if (hess) {
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    hess[i * 2 + j] += obs.hess[
                        source[i] * GH_DERIV_DIM + source[j]
                    ];
                }
            }
        }
    }
    if (nll) *nll = total;
}
