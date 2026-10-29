#include <scivolapi/restriction.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <numeric>
#include <queue>

#include "density/distribution.hpp"
#include "finalize.hpp"
#include <scivolapi/jet.hpp>

namespace scivolapi {
namespace {

constexpr double eps = std::numeric_limits<double>::epsilon();
constexpr double pi = std::numbers::pi;
using Matrix = std::vector<double>;

bool invert(const Matrix& a, std::size_t n, Matrix& inverse)
{
    return n == 0 || detail::invert_information(a, n, inverse) == Inference::available;
}

Matrix block(const Matrix& a, std::size_t n, const std::vector<std::size_t>& rows,
             const std::vector<std::size_t>& cols)
{
    Matrix b(rows.size() * cols.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < cols.size(); ++j) b[i * cols.size() + j] = a[rows[i] * n + cols[j]];
    return b;
}

bool symmetric(std::span<const double> a, std::size_t n)
{
    if (a.size() != n * n) return false;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (!std::isfinite(a[i * n + j])
                || std::abs(a[i * n + j] - a[j * n + i]) > 64.0 * eps
                    * std::max({1.0, std::abs(a[i * n + j]), std::abs(a[j * n + i])})) return false;
    return true;
}

double correlation(const Matrix& a, std::size_t n, std::size_t i, std::size_t j)
{
    return a[i * n + j] / std::sqrt(a[i * n + i]) / std::sqrt(a[j * n + j]);
}

double small_orthant(const Matrix& a, std::size_t n)
{
    if (n == 0) return 1.0;
    if (n == 1) return 0.5;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) sum += std::asin(correlation(a, n, i, j));
    return n == 2 ? 0.25 + sum / (2.0 * pi) : 0.125 + sum / (4.0 * pi);
}

double normal_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Monotone inversion, with exact infinite endpoints for interval propagation.
double normal_quantile(double p)
{
    if (p == 0.0) return -std::numeric_limits<double>::infinity();
    if (p == 1.0) return std::numeric_limits<double>::infinity();
    double lo = -40.0, hi = 40.0;
    for (int i = 0; i < 58; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (normal_cdf(mid) < p) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

struct Probability {
    double lower = 0.0, upper = 1.0;
    [[nodiscard]] double value() const { return 0.5 * (lower + upper); }
    [[nodiscard]] double error() const { return 0.5 * (upper - lower); }
};

// Genz's conditioning transformation. Propagate ranges of each conditional
// quantile over a cube cell, giving a conservative integral enclosure.
Probability condition_cell(const Matrix& l, std::size_t n, const Matrix& lo, const Matrix& hi)
{
    Matrix zlo(n), zhi(n);
    double lower = 1.0, upper = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        double a = 0.0, b = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            const double c = -l[i * n + j] / l[i * n + i];
            if (c > 0.0) { a += c * zlo[j]; b += c * zhi[j]; }
            if (c < 0.0) { a += c * zhi[j]; b += c * zlo[j]; }
        }
        if (std::isnan(a) || std::isnan(b)) return {0.0, upper};
        const double dl = normal_cdf(a), du = normal_cdf(b);
        lower *= normal_cdf(-b);
        upper *= normal_cdf(-a);
        if (i + 1 < n) {
            zlo[i] = normal_quantile(dl + lo[i] * (1.0 - dl));
            zhi[i] = normal_quantile(du + hi[i] * (1.0 - du));
        }
    }
    return {lower, upper};
}

Probability orthant(const Matrix& a, std::size_t n, double tolerance, std::size_t maximum_cells)
{
    if (n <= 3) {
        const double p = small_orthant(a, n);
        return {p, p};
    }
    // Independent blocks factor exactly before any integration.
    std::vector<std::size_t> component(n, n);
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) if (component[i] == n) {
        component[i] = count;
        std::vector<std::size_t> queue{i};
        for (std::size_t k = 0; k < queue.size(); ++k)
            for (std::size_t j = 0; j < n; ++j)
                if (component[j] == n && a[queue[k] * n + j] != 0.0) {
                    component[j] = count;
                    queue.push_back(j);
                }
        ++count;
    }
    if (count > 1) {
        Probability p{1.0, 1.0};
        for (std::size_t k = 0; k < count; ++k) {
            std::vector<std::size_t> indices;
            for (std::size_t i = 0; i < n; ++i) if (component[i] == k) indices.push_back(i);
            const auto q = orthant(block(a, n, indices, indices), indices.size(), tolerance / count, maximum_cells);
            p.lower *= q.lower;
            p.upper *= q.upper;
        }
        return p;
    }
    Matrix l(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double s = correlation(a, n, i, j);
            for (std::size_t k = 0; k < j; ++k) s -= l[i * n + k] * l[j * n + k];
            if (i == j) {
                if (!(s > 0.0)) return {};
                l[i * n + j] = std::sqrt(s);
            } else l[i * n + j] = s / l[j * n + j];
        }
    struct Cell {
        Matrix lo, hi;
        double volume;
        Probability p;
        [[nodiscard]] double uncertainty() const { return volume * (p.upper - p.lower); }
        bool operator<(const Cell& other) const { return uncertainty() < other.uncertainty(); }
    };
    Cell root{Matrix(n - 1, 0.0), Matrix(n - 1, 1.0), 1.0, {}};
    root.p = condition_cell(l, n, root.lo, root.hi);
    Probability total = root.p;
    std::priority_queue<Cell> cells;
    cells.push(std::move(root));
    for (std::size_t evaluations = 1; evaluations + 2 <= maximum_cells && total.error() > tolerance;
         evaluations += 2) {
        Cell cell = cells.top();
        cells.pop();
        std::size_t axis = 0;
        for (std::size_t j = 1; j + 1 < n; ++j)
            if (cell.hi[j] - cell.lo[j] > cell.hi[axis] - cell.lo[axis]) axis = j;
        const double mid = 0.5 * (cell.lo[axis] + cell.hi[axis]);
        if (mid == cell.lo[axis] || mid == cell.hi[axis]) break;
        total.lower -= cell.volume * cell.p.lower;
        total.upper -= cell.volume * cell.p.upper;
        Cell left = cell, right = cell;
        left.hi[axis] = mid;
        right.lo[axis] = mid;
        for (Cell* part : {&left, &right}) {
            part->volume *= 0.5;
            part->p = condition_cell(l, n, part->lo, part->hi);
            total.lower += part->volume * part->p.lower;
            total.upper += part->volume * part->p.upper;
            cells.push(std::move(*part));
        }
    }
    const double roundoff = 256.0 * eps * maximum_cells * n;
    total.lower = std::max(0.0, total.lower - roundoff);
    total.upper = std::min(1.0, total.upper + roundoff);
    return total;
}

LRResult invalid(LRResult result, LRStatus status, std::string message)
{
    result.status = status;
    result.message = std::move(message);
    return result;
}

// One term of the chi-bar tail: chi^2_0 is the atom at zero; a chi-square tail
// too small to represent counts as zero.
double chi_bar_term(double statistic, std::size_t degrees)
{
    if (degrees == 0) return statistic > 0.0 ? 0.0 : 1.0;
    double upper;
    return detail::chi_square_upper(statistic, static_cast<double>(degrees), upper) ? upper : 0.0;
}

}  // namespace

LRWeights chi_bar_weights(std::span<const double> information, std::size_t m, double tolerance,
                         std::size_t maximum_cells)
{
    LRWeights out;
    if (!m || !(tolerance > 0.0) || !std::isfinite(tolerance) || maximum_cells < 1
        || !symmetric(information, m)) return out;
    if (m > 12) { out.status = LRStatus::unresolved; return out; }
    const Matrix j(information.begin(), information.end());
    Matrix v;
    if (!invert(j, m, v)) { out.status = LRStatus::singular_information; return out; }
    out.weights.assign(m + 1, 0.0);
    double jnorm = 0.0, vnorm = 0.0, gap = 1.0;
    for (std::size_t i = 0; i < m; ++i) {
        double jsum = 0.0, vsum = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
            jsum += std::abs(correlation(j, m, i, k));
            vsum += std::abs(v[i * m + k]) * std::sqrt(j[i * m + i]) * std::sqrt(j[k * m + k]);
            if (i != k) {
                const double rj = correlation(j, m, i, k), rv = correlation(v, m, i, k);
                gap = std::min({gap, 1.0 - rj * rj, 1.0 - rv * rv});
            }
        }
        jnorm = std::max(jnorm, jsum);
        vnorm = std::max(vnorm, vsum);
    }
    if (!(gap > 0.0)) { out.status = LRStatus::unresolved; return out; }
    out.numerical_error = 512.0 * eps * m * jnorm * vnorm / std::sqrt(gap);
    if (m == 1) out.weights = {0.5, 0.5};
    else if (m == 2) {
        const double a = std::asin(correlation(v, m, 0, 1)) / (2.0 * pi);
        out.weights = {0.25 - a, 0.5, 0.25 + a};
    } else if (m == 3) {
        const double w0 = small_orthant(j, m), w3 = small_orthant(v, m);
        out.weights = {w0, 0.5 - w3, 0.5 - w0, w3};
    } else {
        const std::size_t subsets = std::size_t{1} << m;
        std::size_t integrals = 0;
        for (std::size_t mask = 0; mask < subsets; ++mask) {
            const auto k = static_cast<std::size_t>(std::popcount(mask));
            integrals += (k > 3) + (m - k > 3);
        }
        const std::size_t cells_per_integral = std::max<std::size_t>(1, maximum_cells / integrals);
        for (std::size_t mask = 0; mask < subsets; ++mask) {
            std::vector<std::size_t> s, r;
            for (std::size_t i = 0; i < m; ++i) ((mask >> i) & 1 ? s : r).push_back(i);
            Matrix iss, schur = block(j, m, r, r);
            if (!invert(block(j, m, s, s), s.size(), iss)) { out.status = LRStatus::singular_information; return out; }
            for (std::size_t a = 0; a < r.size(); ++a)
                for (std::size_t b = 0; b < r.size(); ++b)
                    for (std::size_t i = 0; i < s.size(); ++i)
                        for (std::size_t k = 0; k < s.size(); ++k)
                            schur[a * r.size() + b] -= j[r[a] * m + s[i]] * iss[i * s.size() + k] * j[s[k] * m + r[b]];
            const double budget = tolerance / (4.0 * subsets);
            const Probability p = orthant(iss, s.size(), budget, cells_per_integral);
            const Probability q = orthant(schur, r.size(), budget, cells_per_integral);
            const Probability product{p.lower * q.lower, p.upper * q.upper};
            out.weights[s.size()] += product.value();
            out.numerical_error += product.error();
        }
    }
    const double sum = std::accumulate(out.weights.begin(), out.weights.end(), 0.0);
    if (!std::all_of(out.weights.begin(), out.weights.end(), [](double w) { return std::isfinite(w) && w >= 0.0; })
        || std::abs(sum - 1.0) > out.numerical_error + 256.0 * eps * m) {
        out.status = LRStatus::unresolved;
        return out;
    }
    out.status = out.numerical_error <= tolerance ? LRStatus::ok : LRStatus::unresolved;
    return out;
}

LRResult restriction_test(const Model& parent, const Model& child, const Restriction& r,
                          const FitResult& pf, const FitResult& cf, std::span<const double> information,
                          double level, double tolerance, std::size_t maximum_cells)
{
    LRResult out;
    const std::size_t d = parent.dimension(), c = child.dimension();
    if (!(level > 0.0 && level < 1.0) || !(tolerance > 0.0 && tolerance <= 1e-6)
        || !std::isfinite(tolerance) || !maximum_cells)
        return invalid(out, LRStatus::invalid_argument, "Invalid decision level, accuracy, or cubature budget.");
    if (!(r.parent == parent.spec()) || !(r.child == child.spec()) || !pf.fitted_spec || !cf.fitted_spec
        || !(*pf.fitted_spec == parent.spec()) || !(*cf.fitted_spec == child.spec())
        || !pf.comparison || pf.comparison != cf.comparison || pf.scored != cf.scored
        || pf.observations != cf.observations || pf.theta.size() != d || cf.theta.size() != c)
        return invalid(out, LRStatus::incompatible, "Fits must use their declared models and the same comparison plan.");
    if (r.types.empty() || r.types.size() != d - c || r.directions.size() != r.types.size() * d
        || r.null_values.size() != r.types.size())
        return invalid(out, LRStatus::invalid_argument, "The restriction has inconsistent dimensions.");
    out.statistic = 2.0 * (pf.log_likelihood - cf.log_likelihood);
    if (r.identification == Identification::unidentified_under_null)
        return invalid(out, LRStatus::unsupported, r.condition);
    if (pf.estimator != Estimator::mle || cf.estimator != Estimator::mle)
        return invalid(out, LRStatus::unsupported, "QMLE LR calibration is not implemented.");
    if (!pf.success || !cf.success || !std::isfinite(out.statistic))
        return invalid(out, LRStatus::uncertified, "Both estimates must be certified maximum-likelihood fits.");
    if (pf.kink_point || cf.kink_point)
        return invalid(out, LRStatus::unsupported, "LR calibration at a residual kink-point MLE is not implemented.");
    const auto lifted = r.embed(cf.theta);
    if (!lifted) return invalid(out, LRStatus::incompatible, "The child estimate cannot be embedded in the parent domain.");
    const double noise = 64.0 * eps * (1.0 + std::abs(pf.log_likelihood) + std::abs(cf.log_likelihood));
    if (*lifted == pf.theta && std::abs(out.statistic) <= noise) out.statistic = 0.0;
    if (out.statistic < 0.0)
        return invalid(out, LRStatus::negative_statistic, "The child improves on the parent; refit the parent from the embedded child.");
    if (!r.identified(cf.theta))
        return invalid(out, LRStatus::unsupported, "A nonminimal or unidentified recursion under the null requires specialized calibration.");
    if (!symmetric(information, d))
        return invalid(out, LRStatus::invalid_argument, "Parent information must be finite and symmetric at the embedded child estimate.");
    Matrix inverse;
    if (!invert(Matrix(information.begin(), information.end()), d, inverse))
        return invalid(out, LRStatus::singular_information, "Parent information is singular, indefinite, or ill conditioned.");
    const auto normals = parent.closed_face_normals(*lifted);
    bool orthogonal = true;
    for (std::size_t i = 0; i < normals.size()/d; ++i) {
        // A parent face is retained when it varies along the child embedding.
        bool retained = false;
        for (std::size_t j = 0; j < c; ++j) {
            double v = 0;
            for (std::size_t a = 0; a < d; ++a) v += normals[i*d+a]*r.embedding[a*c+j];
            retained = retained || v != 0;
        }
        if (!retained) continue;
        for (std::size_t j = 0; j < r.types.size(); ++j) {
            double cross = 0, va = 0, vb = 0;
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t b = 0; b < d; ++b) {
                    const double v = inverse[a*d+b];
                    cross += r.directions[j*d+a]*v*normals[i*d+b];
                    va += r.directions[j*d+a]*v*r.directions[j*d+b];
                    vb += normals[i*d+a]*v*normals[i*d+b];
                }
            orthogonal = orthogonal && std::abs(cross) <= 512*eps*std::sqrt(va)*std::sqrt(vb);
        }
    }
    if (!orthogonal) {
        // At most three prespecified looks; their 99.99% Chernoff intervals have
        // at least 99.97% simultaneous coverage by the union bound.
        // Spend the full simulation budget only near the decision threshold.
        for (const std::size_t budget : {std::size_t{1024}, std::size_t{4096}, maximum_cells}) {
            out = cone_projection_test(information, d, normals, r.directions, out.statistic, level,
                                       std::min(budget, maximum_cells));
            if (out.status != LRStatus::unresolved || budget >= maximum_cells) return out;
        }
        return out;
    }
    std::vector<std::size_t> boundary;
    for (std::size_t i = 0; i < r.types.size(); ++i) {
        if (r.types[i] == RestrictionType::closed_boundary) boundary.push_back(i);
        else ++out.interior_df;
    }
    const std::size_t m = boundary.size();
    out.law = m ? ReferenceLaw::chi_bar_square : ReferenceLaw::chi_square;
    bool weights_resolved = true;
    if (!m) {
        out.weights = {1.0};
        out.numerical_error = 128.0 * eps * (1.0 + out.interior_df);
    } else {
        // V = A_b I^{-1} A_b', profiling nuisance and signed tested directions.
        Matrix v(m * m, 0.0), j;
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t k = 0; k < m; ++k)
                for (std::size_t a = 0; a < d; ++a)
                    for (std::size_t b = 0; b < d; ++b)
                        v[i * m + k] += r.directions[boundary[i] * d + a] * inverse[a * d + b]
                            * r.directions[boundary[k] * d + b];
        if (!invert(v, m, j)) return invalid(out, LRStatus::singular_information, "Boundary information is singular.");
        auto weights = chi_bar_weights(j, m, tolerance, maximum_cells);
        out.weights = std::move(weights.weights);
        out.numerical_error = weights.numerical_error;
        if (out.weights.empty() || (weights.status != LRStatus::ok && weights.status != LRStatus::unresolved))
            return invalid(out, weights.status, "Boundary weights could not be computed.");
        weights_resolved = weights.status == LRStatus::ok;
    }
    out.p_value = 0.0;
    for (std::size_t k = 0; k < out.weights.size(); ++k)
        out.p_value += out.weights[k] * chi_bar_term(out.statistic, out.interior_df + k);
    if (!weights_resolved || out.numerical_error > tolerance
        || (out.p_value - out.numerical_error <= level && level <= out.p_value + out.numerical_error))
        return invalid(out, LRStatus::unresolved, "The cubature accuracy or decision interval is unresolved.");
    out.status = LRStatus::ok;
    out.message = "Asymptotic correct-likelihood law, conditional on identification and regular information.";
    return out;
}

LRResult restriction_test(const Model& parent, const Model& child, const Restriction& r,
                          const FitResult& pf, const FitResult& cf, const ComparisonPlan& plan,
                          double level, double tolerance, std::size_t maximum_cells)
{
    if (pf.comparison != plan.identity() || cf.comparison != plan.identity())
        return invalid({}, LRStatus::incompatible, "Fits and information must use the same comparison plan.");
    if (pf.kink_point || cf.kink_point)
        return invalid({}, LRStatus::unsupported, "LR calibration at a residual kink-point MLE is not implemented.");
    if (r.identification == Identification::unidentified_under_null)
        return invalid({}, LRStatus::unsupported, r.condition);
    const auto theta = r.embed(cf.theta);
    if (!theta) return invalid({}, LRStatus::incompatible, "The child cannot be embedded in the parent domain.");
    const auto sample = parent.prepare(plan);
    Jet jet(parent, sample);
    const std::size_t d = parent.dimension();
    std::vector<double> row_scores(sample.scored()*d);
    JetOutput output;
    output.scores = row_scores;
    if (jet.theta(*theta, scores, output).status != Status::ok)
        return invalid({}, LRStatus::unresolved, "Parent scores are unavailable at the restricted estimate.");
    return restriction_test(parent, child, r, pf, cf,
                            detail::outer_products(row_scores, sample.scored(), d), level, tolerance, maximum_cells);
}

}  // namespace scivolapi
