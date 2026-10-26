#include <scivolapi/statistics.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "density/distribution.hpp"
#include "detail/numeric.hpp"
#include "finalize.hpp"

namespace scivolapi {
namespace {

constexpr double nan = std::numeric_limits<double>::quiet_NaN();

void require(bool condition, const char* message)
{
    if (!condition) throw std::invalid_argument(message);
}

void finite(std::span<const double> values)
{
    require(std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); }),
            "statistics require finite inputs, with no dropped observations");
}

void check_lags(std::size_t rows, std::size_t lags)
{
    require(rows >= 3 && lags > 0 && lags < rows, "lags must be positive and smaller than the sample size");
}

TestStatistic chi_test(double statistic, std::size_t df)
{
    TestStatistic out;
    out.df = df;
    if (df == 0 || !(statistic >= 0.0) || !std::isfinite(statistic)) return out;
    if (!detail::chi_square_log_upper(statistic, double(df), out.log_p_value)) {
        out.status = TestStatus::numerical_range;
        return out;
    }
    out.statistic = statistic;
    const double probability = std::exp(out.log_p_value);
    if (probability > 0.0 && std::numeric_limits<double>::denorm_min()/probability <= 2e-13) {
        out.p_value = probability;
        out.status = TestStatus::ok;
    } else out.status = TestStatus::probability_underflow;
    return out;
}

TestStatistic t_test(double statistic, std::size_t df)
{
    TestStatistic out;
    out.df = df;
    if (df == 0 || !std::isfinite(statistic)) return out;
    double log_upper;
    if (!detail::student_log_upper(std::abs(statistic), double(df), log_upper)) {
        out.status = TestStatus::numerical_range;
        return out;
    }
    out.statistic = statistic;
    out.log_p_value = std::log(2.0) + log_upper;
    const double probability = std::exp(out.log_p_value);
    if (probability > 0.0 && std::numeric_limits<double>::denorm_min()/probability <= 2e-13) {
        out.p_value = probability;
        out.status = TestStatus::ok;
    } else out.status = TestStatus::probability_underflow;
    return out;
}

std::vector<double> normal_scores(std::span<const double> pit)
{
    finite(pit);
    std::vector<double> scores(pit.size());
    for (std::size_t t = 0; t < pit.size(); ++t) {
        require(pit[t] > 0.0 && pit[t] < 1.0, "normal PIT scores require probabilities strictly between zero and one");
        if (detail::quantile(DensityKind::normal, nullptr, pit[t], scores[t]) != detail::ExpectationStatus::ok)
            throw std::range_error("normal PIT score is not representable");
    }
    return scores;
}

// Intercept plus centered regressors. Equilibration and inversion share the
// inference code's conditioning checks; unidentified designs are not repaired.
struct Regression {
    bool valid = false;
    std::vector<double> coefficients, inverse;
    double sse = nan, total = nan;
};

Regression regress(std::span<const double> response, std::span<const double> design, std::size_t columns)
{
    finite(response);
    const std::size_t n = response.size();
    Regression out;
    if (n <= columns + 1) return out;
    std::vector<double> means(columns, 0.0), gram(columns * columns, 0.0), cross(columns, 0.0);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t j = 0; j < columns; ++j) means[j] += design[t * columns + j] / n;
    detail::CompensatedSum response_sum;
    for (double y : response) response_sum.add(y / double(n));
    const double mean_y = response_sum.total();
    double total = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        const double y = response[t] - mean_y;
        total += y * y;
        for (std::size_t i = 0; i < columns; ++i) {
            const double x = design[t * columns + i] - means[i];
            cross[i] += x * y;
            for (std::size_t j = 0; j < columns; ++j)
                gram[i * columns + j] += x * (design[t * columns + j] - means[j]);
        }
    }
    if (!(total > 0.0) || detail::invert_information(gram, columns, out.inverse) != Inference::available) return out;
    out.coefficients.assign(columns + 1, 0.0);
    out.coefficients[0] = mean_y;
    for (std::size_t i = 0; i < columns; ++i) {
        for (std::size_t j = 0; j < columns; ++j) out.coefficients[i + 1] += out.inverse[i * columns + j] * cross[j];
        out.coefficients[0] -= means[i] * out.coefficients[i + 1];
    }
    double sse = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        double error = response[t] - out.coefficients[0];
        for (std::size_t j = 0; j < columns; ++j) error -= design[t * columns + j] * out.coefficients[j + 1];
        sse += error * error;
    }
    out.sse = sse;
    out.total = total;
    out.valid = std::isfinite(sse) && sse > 0.0;
    return out;
}

// A zero-variance loss contrast has an exact, identified ordering or is a tie.
double studentize(double mean, double se)
{
    if (se > 0.0) return mean / se;
    return mean == 0.0 ? 0.0 : std::copysign(std::numeric_limits<double>::infinity(), mean);
}

} // namespace

bool TestStatistic::rejects(double alpha) const noexcept
{
    return (status == TestStatus::ok || status == TestStatus::probability_underflow) && log_p_value < std::log(alpha);
}

double normal_survival(double x)
{
    require(std::isfinite(x), "normal tail requires a finite argument");
    double lower, upper;
    if (detail::distribution(DensityKind::normal, nullptr, x, lower, upper) != detail::ExpectationStatus::ok || upper == 0.0)
        throw std::range_error("normal upper tail is not representable accurately");
    return upper;
}

double student_survival(double x, double df)
{
    require(std::isfinite(x) && std::isfinite(df) && df > 0.0, "Student tail requires finite x and positive df");
    double upper;
    if (!detail::student_upper(x, df, upper)) throw std::range_error("Student upper tail is not representable accurately");
    return upper;
}

double chi_square_survival(double x, double df)
{
    require(std::isfinite(x) && x >= 0.0 && std::isfinite(df) && df > 0.0,
            "chi-square tail requires nonnegative finite x and positive df");
    double upper;
    if (!detail::chi_square_upper(x, df, upper)) throw std::range_error("chi-square upper tail is not representable accurately");
    return upper;
}

TestStatistic coefficient_test(double coefficient, double standard_error, std::size_t df)
{
    require(std::isfinite(coefficient) && std::isfinite(standard_error) && standard_error > 0.0,
            "coefficient tests require a finite coefficient and positive finite standard error");
    return t_test(coefficient / standard_error, df);
}

Portmanteau portmanteau(std::span<const double> series, std::size_t lags, std::size_t df_adjustment)
{
    finite(series);
    check_lags(series.size(), lags);
    const double n = double(series.size());
    detail::CompensatedSum sum;
    for (double x : series) sum.add(x / n);
    const double mean = sum.total();
    double denominator = 0.0;
    for (double x : series) denominator += (x - mean) * (x - mean);
    Portmanteau out;
    out.autocorrelations.assign(lags, nan);
    if (!(denominator > 0.0) || !std::isfinite(denominator)) return out;
    double lb = 0.0, bp = 0.0;
    for (std::size_t k = 1; k <= lags; ++k) {
        double cross = 0.0;
        for (std::size_t t = k; t < series.size(); ++t) cross += (series[t] - mean) * (series[t - k] - mean);
        const double rho = cross / denominator;
        out.autocorrelations[k - 1] = rho;
        bp += rho * rho;
        lb += rho * rho / (n - k);
    }
    const std::size_t df = lags > df_adjustment ? lags - df_adjustment : 0;
    out.ljung_box = chi_test(n * (n + 2.0) * lb, df);
    out.box_pierce = chi_test(n * bp, df);
    return out;
}

TestStatistic dgt(std::span<const double> pit, std::size_t cells)
{
    require(!pit.empty() && cells >= 2, "DGT requires observations and at least two cells");
    finite(pit);
    std::vector<std::size_t> counts(cells, 0);
    for (double u : pit) {
        require(u >= 0.0 && u <= 1.0, "PIT probabilities must belong to [0, 1]");
        ++counts[u == 1.0 ? cells - 1 : std::size_t(u * cells)];
    }
    const double expected = double(pit.size()) / cells;
    double statistic = 0.0;
    for (auto count : counts) statistic += (count - expected) * (count - expected) / expected;
    return chi_test(statistic, cells - 1);
}

SignBias sign_bias(std::span<const double> standardized, std::span<const double> residual)
{
    require(standardized.size() == residual.size() && residual.size() >= 6,
            "sign-bias tests require equal residual paths with at least six rows");
    finite(standardized);
    finite(residual);
    const std::size_t n = residual.size() - 1;
    std::vector<double> response(n), design(n * 3), column(n);
    for (std::size_t t = 0; t < n; ++t) {
        response[t] = standardized[t + 1] * standardized[t + 1];
        design[t * 3] = residual[t] < 0.0 ? 1.0 : 0.0;
        design[t * 3 + 1] = residual[t] < 0.0 ? residual[t] : 0.0;
        design[t * 3 + 2] = residual[t] >= 0.0 ? residual[t] : 0.0;
    }
    SignBias out;
    for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t t = 0; t < n; ++t) column[t] = design[t * 3 + j];
        const Regression fit = regress(response, column, 1);
        if (fit.valid) out.individual[j] = t_test(fit.coefficients[1] / std::sqrt(fit.sse / (n - 2) * fit.inverse[0]), n - 2);
    }
    const Regression fit = regress(response, design, 3);
    if (fit.valid) out.joint = chi_test(n * (1.0 - fit.sse / fit.total), 3);
    return out;
}

TestStatistic arch_lm(std::span<const double> standardized, std::size_t lags)
{
    finite(standardized);
    check_lags(standardized.size(), lags);
    const std::size_t n = standardized.size() - lags;
    std::vector<double> squared(standardized.size()), design(n * lags);
    for (std::size_t t = 0; t < squared.size(); ++t) squared[t] = standardized[t] * standardized[t];
    finite(squared);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t k = 1; k <= lags; ++k) design[t * lags + k - 1] = squared[t + lags - k];
    const auto fit = regress(std::span(squared).subspan(lags), design, lags);
    if (fit.valid) return chi_test(n * (1.0 - fit.sse / fit.total), lags);
    TestStatistic out;
    out.df = lags;
    return out;
}

Berkowitz berkowitz(std::span<const double> pit)
{
    require(pit.size() >= 5, "Berkowitz requires at least five PIT observations");
    const std::vector<double> z = normal_scores(pit);
    const std::size_t n = z.size() - 1;
    const Regression fit = regress(std::span(z).subspan(1), std::span(z).first(n), 1);
    Berkowitz out;
    if (!fit.valid) return out;
    out.mean = fit.coefficients[0];
    out.ar1 = fit.coefficients[1];
    out.variance = fit.sse / n;
    double null_sse = 0.0;
    for (std::size_t t = 1; t < z.size(); ++t) null_sse += z[t] * z[t];
    out.test = chi_test(null_sse - n * (1.0 + std::log(out.variance)), 3);
    return out;
}

namespace {

// Orthonormal tangent directions of the active face, as rows. Open boundaries
// cannot occur at a feasible point. Reorthogonalization handles redundant faces.
std::vector<double> face_basis(const Model& model, std::span<const double> theta)
{
    const std::size_t d = model.dimension();
    const auto faces = model.closed_face_normals(theta);
    std::vector<double> normals, basis, direction(d);
    auto append = [&](std::vector<double>& rows) {
        for (int pass = 0; pass < 2; ++pass)
            for (std::size_t row = 0; row < rows.size() / d; ++row) {
                double dot = 0.0;
                for (std::size_t j = 0; j < d; ++j) dot += direction[j] * rows[row * d + j];
                for (std::size_t j = 0; j < d; ++j) direction[j] -= dot * rows[row * d + j];
            }
        double length = 0.0;
        for (double x : direction) length = std::hypot(length, x);
        if (length <= 64.0 * std::numeric_limits<double>::epsilon()) return;
        for (double x : direction) rows.push_back(x / length);
    };
    for (std::size_t row = 0; row < faces.size() / d; ++row) {
        std::copy_n(faces.begin() + row * d, d, direction.begin());
        append(normals);
    }
    for (std::size_t j = 0; j < d; ++j) {
        std::fill(direction.begin(), direction.end(), 0.0);
        direction[j] = 1.0;
        for (int pass = 0; pass < 2; ++pass)
            for (std::size_t row = 0; row < normals.size() / d; ++row) {
                double dot = 0.0;
                for (std::size_t k = 0; k < d; ++k) dot += direction[k] * normals[row * d + k];
                for (std::size_t k = 0; k < d; ++k) direction[k] -= dot * normals[row * d + k];
            }
        append(basis);
    }
    return basis;
}

struct DiagnosticEvaluation {
    std::vector<double> e, h, z, sensitivity, score, information;
    std::size_t parameters = 0;
    TestStatus status = TestStatus::ok;
    bool fourth_moment = false;
};

DiagnosticEvaluation evaluate_diagnostics(Jet& jet, std::span<const double> theta,
                                          Estimator estimator, bool estimated)
{
    const Model& model = jet.model();
    const std::size_t n = jet.sample().scored(), d = model.dimension();
    require(theta.size() == d, "parameter vector has the wrong length");
    finite(theta);
    DiagnosticEvaluation out;
    out.parameters = d;
    if (!model.feasible(theta)) { out.status = TestStatus::infeasible_parameters; return out; }
    out.fourth_moment = std::isfinite(model.innovation_power_moment(theta, 0.0, 4.0));
    const bool moments = out.fourth_moment;
    out.e.resize(n); out.h.resize(n); out.z.resize(n);
    out.sensitivity.resize(n * d); out.score.resize(n * d); out.information.resize(d * d);
    JetOutput evaluated;
    evaluated.paths = {out.e, out.h, out.z};
    evaluated.log_variance_sensitivities = out.sensitivity;
    evaluated.scores = out.score;
    evaluated.hessian = out.information;
    const unsigned request = paths | (estimated && moments ? log_variance_sensitivities | scores | hessian : 0u);
    const auto result = jet.theta(theta, request, evaluated);
    if ((result.valid & request) != request) out.status = TestStatus::invalid_derivatives;
    if (!(result.valid & paths)) { out.e.clear(); out.h.clear(); out.z.clear(); }
    if (!moments) out.status = TestStatus::fourth_moment_unavailable;
    if (out.status != TestStatus::ok || !estimated) return out;
    if (estimator == Estimator::qmle) {
        const Model gaussian = model.gaussian_model();
        require(model.capabilities().density_free_recursion || gaussian.dimension() == d,
                "QMLE diagnostics need a density-free recursion");
        out.parameters = gaussian.dimension();
        out.score.resize(n * out.parameters);
        out.information.resize(out.parameters * out.parameters);
        Jet gaussian_jet(gaussian, jet.sample());
        JetOutput gaussian_out;
        gaussian_out.scores = out.score;
        gaussian_out.hessian = out.information;
        if ((gaussian_jet.theta(theta.first(out.parameters), scores | hessian, gaussian_out).valid & (scores | hessian)) != (scores | hessian))
            out.status = TestStatus::invalid_qmle_derivatives;
    }
    return out;
}

LiMak li_mak_evaluated(Jet& jet, std::span<const double> theta, std::size_t lags,
                       Estimator estimator, bool estimated, const DiagnosticEvaluation& evaluated)
{
    const Model& model = jet.model();
    const std::size_t n = jet.sample().scored(), d = model.dimension(), parameters = evaluated.parameters;
    const auto& z = evaluated.z;
    const auto& sensitivity = evaluated.sensitivity;
    const auto& score = evaluated.score;
    LiMak out;
    out.test.df = lags;
    if (evaluated.status != TestStatus::ok) { out.test.status = evaluated.status; return out; }
    std::vector<double> inverse(parameters * parameters, 0.0);
    if (estimated) {
        const Model estimation_model = estimator == Estimator::qmle ? model.gaussian_model() : model;
        const auto basis = face_basis(estimation_model, theta.first(parameters));
        const std::size_t free = basis.size() / parameters;
        std::vector<double> information(free * free, 0.0), conditional_inverse;
        // Condition on the active closed face: fix its normal coordinates and
        // drop them from both score and Hessian blocks. The restricted child is
        // precisely this model; GJR one-sign faces retain the alpha=-gamma tangent.
        for (std::size_t i = 0; i < free; ++i)
            for (std::size_t j = 0; j < free; ++j)
                for (std::size_t a = 0; a < parameters; ++a)
                    for (std::size_t b = 0; b < parameters; ++b)
                        information[i * free + j] += basis[i * parameters + a] * evaluated.information[a * parameters + b] * basis[j * parameters + b];
        if (free && detail::invert_information(information, free, conditional_inverse) != Inference::available) {
            out.test.status = TestStatus::singular_information;
            return out;
        }
        for (std::size_t a = 0; a < parameters; ++a)
            for (std::size_t b = 0; b < parameters; ++b)
                for (std::size_t i = 0; i < free; ++i)
                    for (std::size_t j = 0; j < free; ++j)
                        inverse[a * parameters + b] += basis[i * parameters + a] * conditional_inverse[i * free + j] * basis[j * parameters + b];
    }
    std::vector<double> v(n), moments(n * lags, 0.0), means(lags, 0.0), projection(lags * parameters, 0.0);
    out.derivative.assign(lags * d, 0.0);
    double gamma0 = 0.0;
    for (std::size_t t = 0; t < n; ++t) { v[t] = z[t] * z[t] - 1.0; gamma0 += v[t] * v[t] / n; }
    if (!(gamma0 > 0.0) || !std::isfinite(gamma0)) return out;
    for (std::size_t k = 1; k <= lags; ++k) {
        for (std::size_t t = k; t < n; ++t) {
            moments[t * lags + k - 1] = v[t] * v[t - k];
            means[k - 1] += moments[t * lags + k - 1] / n;
            for (std::size_t j = 0; j < d; ++j) out.derivative[(k - 1) * d + j] -= v[t - k] * sensitivity[t * d + j] / n;
        }
        if (estimated)
            for (std::size_t i = 0; i < parameters; ++i)
                for (std::size_t j = 0; j < parameters; ++j)
                    projection[(k - 1) * parameters + i] += out.derivative[(k - 1) * d + j] * inverse[j * parameters + i] * n;
    }
    std::vector<double> influence_mean(lags, 0.0);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t k = 0; k < lags; ++k) {
            if (estimated)
                for (std::size_t j = 0; j < parameters; ++j) moments[t * lags + k] -= projection[k * parameters + j] * score[t * parameters + j];
            influence_mean[k] += moments[t * lags + k] / n;
        }
    std::vector<double> covariance(lags * lags, 0.0), covariance_inverse;
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t i = 0; i < lags; ++i)
            for (std::size_t j = 0; j < lags; ++j)
                covariance[i * lags + j] += (moments[t * lags + i] - influence_mean[i]) * (moments[t * lags + j] - influence_mean[j]) / n;
    out.autocorrelations.resize(lags);
    for (std::size_t k = 0; k < lags; ++k) out.autocorrelations[k] = means[k] / gamma0;
    out.covariance.resize(lags * lags);
    for (std::size_t j = 0; j < covariance.size(); ++j) out.covariance[j] = covariance[j] / (gamma0 * gamma0);
    if (detail::invert_information(covariance, lags, covariance_inverse) != Inference::available) {
        out.test.status = TestStatus::singular_moment_covariance;
        return out;
    }
    double statistic = 0.0;
    for (std::size_t i = 0; i < lags; ++i)
        for (std::size_t j = 0; j < lags; ++j) statistic += n * means[i] * covariance_inverse[i * lags + j] * means[j];
    out.test = chi_test(statistic, lags);
    return out;
}

// Prefix horizons share the same correlations and covariance, including the
// estimation correction. Only the small leading covariance block is inverted.
TestStatistic li_mak_prefix(const LiMak& full, std::size_t n, std::size_t lags)
{
    if (lags == full.test.df) return full.test;
    TestStatistic out;
    out.df = lags;
    if (full.covariance.empty()) { out.status = full.test.status; return out; }
    const std::size_t maximum = full.autocorrelations.size();
    std::vector<double> covariance(lags * lags), inverse;
    for (std::size_t i = 0; i < lags; ++i)
        for (std::size_t j = 0; j < lags; ++j) covariance[i * lags + j] = full.covariance[i * maximum + j];
    if (detail::invert_information(covariance, lags, inverse) != Inference::available) {
        out.status = TestStatus::singular_moment_covariance;
        return out;
    }
    double statistic = 0.0;
    for (std::size_t i = 0; i < lags; ++i)
        for (std::size_t j = 0; j < lags; ++j) statistic += n * full.autocorrelations[i] * inverse[i * lags + j] * full.autocorrelations[j];
    return chi_test(statistic, lags);
}

} // namespace

LiMak li_mak(Jet& jet, std::span<const double> theta, std::size_t lags, Estimator estimator, bool estimated)
{
    check_lags(jet.sample().scored(), lags);
    const auto evaluated = evaluate_diagnostics(jet, theta, estimator, estimated);
    return li_mak_evaluated(jet, theta, lags, estimator, estimated, evaluated);
}

Adequacy adequacy(Jet& jet, std::span<const double> theta, double level, Estimator estimator)
{
    require(level > 0.0 && level < 1.0, "diagnostic level must be between zero and one");
    const std::size_t n = jet.sample().scored(), maximum = std::min<std::size_t>(20, n / 5);
    Adequacy out;
    out.level = level;
    const bool constant_variance = jet.model().spec().volatility == VolatilityKind::constant;
    if (constant_variance) out.volatility_gum = "unavailable";
    auto add = [&](std::string name, TestStatistic test, bool required = true) {
        out.checks.push_back({std::move(name), test, required, required, false});
    };
    const auto evaluated = evaluate_diagnostics(jet, theta, estimator, true);
    const bool paths_valid = !evaluated.z.empty();
    Portmanteau residual;
    LiMak squared;
    if (maximum) {
        if (paths_valid) residual = portmanteau(evaluated.z, maximum, jet.model().residual_degrees_of_freedom());
        squared = li_mak_evaluated(jet, theta, maximum, estimator, true, evaluated);
    }
    std::size_t previous = 0;
    for (const std::size_t requested : {10u, 20u}) {
        const std::size_t lag = std::min(requested, maximum);
        if (lag && lag == previous) continue;
        previous = lag;
        TestStatistic lb, lm;
        if (lag && paths_valid) {
            double sum = 0.0;
            for (std::size_t k = 1; k <= lag; ++k) sum += std::pow(residual.autocorrelations[k - 1], 2) / (n - k);
            const std::size_t adjustment = jet.model().residual_degrees_of_freedom();
            lb = chi_test(double(n) * (n + 2.0) * sum, lag > adjustment ? lag - adjustment : 0);
            lm = li_mak_prefix(squared, n, lag);
        } else if (!paths_valid) lb.status = lm.status = evaluated.status;
        add("residual_ljung_box[" + std::to_string(lag) + "]", lb);
        add("li_mak[" + std::to_string(lag) + "]", lm);
        if (!lag) break;
    }
    const bool fourth_moment = evaluated.fourth_moment;
    TestStatistic sign;
    if (!paths_valid) sign.status = evaluated.status;
    else if (!fourth_moment) sign.status = TestStatus::fourth_moment_unavailable;
    else if (n >= 6) sign = sign_bias(evaluated.z, evaluated.e).joint;
    add("sign_bias", sign);
    if (constant_variance) {
        TestStatistic arch;
        if (!paths_valid) arch.status = evaluated.status;
        else if (!fourth_moment) arch.status = TestStatus::fourth_moment_unavailable;
        else if (maximum) arch = arch_lm(evaluated.z, std::min<std::size_t>(10, maximum));
        add("arch_lm", arch);
    }
    TestStatistic berk, cell;
    if (paths_valid) {
        std::vector<double> pit(n);
        const auto law = jet.model().innovation_density(theta);
        bool valid = true, interior = true;
        for (std::size_t t = 0; t < n; ++t) {
            double upper;
            valid = valid && detail::distribution(law.density, law.shape.data(), evaluated.z[t], pit[t], upper) == detail::ExpectationStatus::ok;
            interior = interior && pit[t] > 0.0 && pit[t] < 1.0;
        }
        if (valid) {
            cell = dgt(pit, 40);
            if (interior && n >= 5) berk = berkowitz(pit).test;
            else if (!interior) berk.status = TestStatus::pit_endpoint;
        } else berk.status = cell.status = TestStatus::numerical_range;
    } else berk.status = cell.status = evaluated.status;
    add("berkowitz", berk, false);
    add("dgt", cell, false);
    std::vector<std::size_t> family;
    bool unresolved = false, rejected = false;
    for (std::size_t i = 0; i < out.checks.size(); ++i) {
        const auto& check = out.checks[i];
        if (!check.required) continue;
        family.push_back(i);
        unresolved = unresolved || (check.test.status != TestStatus::ok && check.test.status != TestStatus::probability_underflow);
    }
    auto log_p = [&](std::size_t i) {
        const auto& test = out.checks[i].test;
        return (test.status == TestStatus::ok || test.status == TestStatus::probability_underflow) ? test.log_p_value : 0.0;
    };
    std::stable_sort(family.begin(), family.end(), [&](auto a, auto b) { return log_p(a) < log_p(b); });
    for (std::size_t i = 0; i < family.size(); ++i) {
        if (log_p(family[i]) > std::log(level / double(family.size() - i))) break;
        out.checks[family[i]].holm_reject = true;
        rejected = true;
    }
    out.status = unresolved ? AdequacyStatus::adequacy_unresolved : (rejected ? AdequacyStatus::rejected : AdequacyStatus::adequate);
    return out;
}

DieboldMariano diebold_mariano(std::span<const double> loss_a, std::span<const double> loss_b,
                               std::size_t horizon, int bandwidth)
{
    require(loss_a.size() == loss_b.size() && loss_a.size() >= 3, "DM requires aligned loss paths of at least three rows");
    finite(loss_a);
    finite(loss_b);
    const std::size_t rows = loss_a.size();
    require(horizon >= 1 && horizon < rows && bandwidth >= -1, "invalid DM horizon or bandwidth");
    DieboldMariano out;
    out.bandwidth = bandwidth < 0 ? std::max(std::size_t(std::cbrt(double(rows))), horizon - 1) : std::size_t(bandwidth);
    require(out.bandwidth < rows, "DM bandwidth must be smaller than the sample size");
    std::vector<double> difference(rows);
    for (std::size_t t = 0; t < rows; ++t) difference[t] = loss_a[t] - loss_b[t];
    finite(difference);
    const double n = double(rows), h = double(horizon);
    detail::CompensatedSum sum;
    for (double x : difference) sum.add(x / n);
    const double mean = sum.total();
    for (double& x : difference) x -= mean;
    double variance = 0.0;
    for (double x : difference) variance += x * x / n;
    for (std::size_t k = 1; k <= out.bandwidth; ++k) {
        double covariance = 0.0;
        for (std::size_t t = k; t < rows; ++t) covariance += difference[t] * difference[t - k] / n;
        variance += 2.0 * (1.0 - double(k) / (out.bandwidth + 1)) * covariance;
    }
    out.long_run_variance = variance;
    if (!(variance > 0.0) || !std::isfinite(variance)) return out;
    const double correction = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n;
    out.test = t_test(mean / std::sqrt(variance / n) * std::sqrt(correction), rows - 1);
    return out;
}

std::size_t mcs_block_length(std::size_t rows, int requested)
{
    require(rows >= 3 && requested >= -1, "invalid bootstrap sample size or block length");
    const std::size_t length = requested == -1 ? std::max<std::size_t>(2, std::size_t(std::round(std::cbrt(double(rows)))))
                                              : std::size_t(requested);
    require(length >= 1 && length <= rows, "block length must be between one and the sample size");
    return length;
}

ModelConfidenceSet model_confidence_set(std::span<const double> losses, std::size_t rows,
                                        std::size_t models, std::span<const double> uniforms,
                                        std::size_t replications, std::size_t block_length,
                                        double alpha, MCSStatistic statistic)
{
    require(rows >= 3 && models >= 1 && losses.size() / rows == models && losses.size() % rows == 0,
            "MCS requires a rows by models loss matrix");
    require(replications >= 2 && block_length >= 1 && block_length <= rows && alpha > 0.0 && alpha < 1.0,
            "invalid MCS bootstrap options");
    const std::size_t blocks = (rows - 1) / block_length + 1;
    require(uniforms.size() / blocks == replications && uniforms.size() % blocks == 0,
            "MCS uniforms must have B by ceil(rows/block_length) entries");
    finite(losses);
    finite(uniforms);
    for (double u : uniforms) require(u >= 0.0 && u < 1.0, "bootstrap uniforms must belong to [0, 1)");
    if (models == 1) return {{0}, {}, {1.0}, 1.0};
    // Subtract the first model row-wise before summing, removing common losses.
    std::vector<double> relative(rows * models), means(models, 0.0), bootstrap(replications * models, 0.0);
    for (std::size_t t = 0; t < rows; ++t)
        for (std::size_t j = 0; j < models; ++j) {
            relative[t * models + j] = losses[t * models + j] - losses[t * models];
            means[j] += relative[t * models + j] / rows;
        }
    finite(relative);
    finite(means);
    for (std::size_t b = 0; b < replications; ++b)
        for (std::size_t t = 0; t < rows; ++t) {
            const std::size_t block = t / block_length, start = std::size_t(uniforms[b * blocks + block] * rows);
            const std::size_t index = (start + t % block_length) % rows;
            for (std::size_t j = 0; j < models; ++j) bootstrap[b * models + j] += (relative[index * models + j] - means[j]) / rows;
        }
    finite(bootstrap);
    ModelConfidenceSet out;
    out.p_values.assign(models, 1.0);
    std::vector<std::size_t> active(models);
    std::iota(active.begin(), active.end(), 0);
    std::vector<double> boot_stat(replications), observed(models), boot_mean(replications);
    double monotone = 0.0;
    bool set_found = false;
    while (active.size() > 1) {
        const std::size_t count = active.size();
        std::fill(observed.begin(), observed.end(), -std::numeric_limits<double>::infinity());
        std::fill(boot_stat.begin(), boot_stat.end(), -std::numeric_limits<double>::infinity());
        if (statistic == MCSStatistic::maximum) {
            double grand = 0.0;
            for (auto j : active) grand += means[j] / count;
            for (std::size_t b = 0; b < replications; ++b) {
                boot_mean[b] = 0.0;
                for (auto j : active) boot_mean[b] += bootstrap[b * models + j] / count;
            }
            for (auto j : active) {
                double center = 0.0, variance = 0.0;
                for (std::size_t b = 0; b < replications; ++b) center += (bootstrap[b * models + j] - boot_mean[b]) / replications;
                for (std::size_t b = 0; b < replications; ++b) {
                    const double gap = bootstrap[b * models + j] - boot_mean[b] - center;
                    variance += gap * gap / (replications - 1);
                }
                const double error = std::sqrt(variance);
                observed[j] = studentize(means[j] - grand, error);
                for (std::size_t b = 0; b < replications; ++b)
                    boot_stat[b] = std::max(boot_stat[b], studentize(bootstrap[b * models + j] - boot_mean[b], error));
            }
        } else {
            for (auto i : active) for (auto j : active) if (i != j) {
                double center = 0.0, variance = 0.0;
                for (std::size_t b = 0; b < replications; ++b) center += (bootstrap[b * models + i] - bootstrap[b * models + j]) / replications;
                for (std::size_t b = 0; b < replications; ++b) {
                    const double gap = bootstrap[b * models + i] - bootstrap[b * models + j] - center;
                    variance += gap * gap / (replications - 1);
                }
                const double error = std::sqrt(variance);
                observed[i] = std::max(observed[i], studentize(means[i] - means[j], error));
                for (std::size_t b = 0; b < replications; ++b)
                    boot_stat[b] = std::max(boot_stat[b], std::abs(studentize(bootstrap[b * models + i] - bootstrap[b * models + j], error)));
            }
        }
        auto worst = active.begin();
        for (auto it = active.begin(); it != active.end(); ++it) if (observed[*it] > observed[*worst]) worst = it;
        const double observed_stat = observed[*worst];
        std::size_t exceed = 0;
        for (double x : boot_stat) exceed += x >= observed_stat;
        const double p = double(1 + exceed) / double(replications + 1);
        monotone = std::max(monotone, p);
        if (!set_found && monotone >= alpha) {
            out.included = active;
            out.final_p_value = monotone;
            set_found = true;
        }
        out.p_values[*worst] = monotone;
        if (!set_found) out.eliminated.push_back(*worst);
        active.erase(worst);
    }
    if (!set_found) { out.included = active; out.final_p_value = 1.0; }
    return out;
}

InformationCriteria information_criteria(double log_likelihood, std::size_t parameters, std::size_t rows)
{
    require(rows > 1 && std::isfinite(log_likelihood), "information criteria require a finite likelihood and at least two scored observations");
    const double deviance = -2.0 * log_likelihood, k = double(parameters), n = double(rows);
    return {deviance + 2.0 * k, deviance + k * std::log(n), deviance + 2.0 * k * std::log(std::log(n))};
}

Diagnostics diagnostics(Jet& jet, std::span<const double> theta, std::size_t cells, std::size_t lags,
                         std::size_t df_adjustment, Estimator estimator, bool estimated, std::size_t pit_df_adjustment)
{
    const std::size_t n = jet.sample().scored();
    check_lags(n, lags);
    std::vector<double> e(n), h(n), z(n), moment(n);
    JetOutput paths_out;
    paths_out.paths = {e, h, z};
    require((jet.theta(theta, paths, paths_out).valid & paths) != 0u, "diagnostics require evaluable residual paths");
    Diagnostics out;
    out.pit.resize(n);
    const InnovationDensity law = jet.model().innovation_density(theta);
    bool interior = true;
    for (std::size_t t = 0; t < n; ++t) {
        double upper;
        if (detail::distribution(law.density, law.shape.data(), z[t], out.pit[t], upper) != detail::ExpectationStatus::ok)
            throw std::range_error("PIT distribution could not be evaluated accurately");
        interior = interior && out.pit[t] > 0.0 && out.pit[t] < 1.0;
    }
    out.cell = dgt(out.pit, cells);
    for (std::size_t power = 1; power <= 4; ++power) {
        for (std::size_t t = 0; t < n; ++t) moment[t] = std::pow(out.pit[t] - 0.5, int(power));
        out.pit_moments[power - 1] = portmanteau(moment, lags, pit_df_adjustment);
    }
    out.residual = portmanteau(z, lags, df_adjustment);
    if (interior) {
        out.pit_normal = portmanteau(normal_scores(out.pit), lags, pit_df_adjustment);
        if (n >= 5) out.density = berkowitz(out.pit);
    } else {
        out.density.test.status = TestStatus::pit_endpoint;
        out.pit_normal.ljung_box.status = out.pit_normal.box_pierce.status = TestStatus::pit_endpoint;
    }
    if (n >= 6) out.asymmetry = sign_bias(z, e);
    out.squared_residual = li_mak(jet, theta, lags, estimator, estimated);
    return out;
}

std::vector<double> location_scale_quantiles(double mean, double variance, const InnovationDensity& law,
                                             std::span<const double> probabilities)
{
    require(std::isfinite(mean) && std::isfinite(variance) && variance > 0.0,
            "quantiles require finite location and positive finite variance");
    std::vector<double> out(probabilities.size());
    if (detail::quantiles(law.density, law.shape.data(), probabilities.data(), probabilities.size(), out.data())
        != detail::ExpectationStatus::ok) throw std::range_error("innovation quantiles could not be evaluated accurately");
    const double sigma = std::sqrt(variance);
    for (double& x : out) x = std::fma(sigma, x, mean);
    finite(out);
    return out;
}

} // namespace scivolapi
