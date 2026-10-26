#include <scivolapi/restriction.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <random>

#include "finalize.hpp"

namespace scivolapi {
namespace {

using Matrix = std::vector<double>;
constexpr double eps = std::numeric_limits<double>::epsilon();

struct Face {
    std::vector<std::size_t> rows;
    Matrix inverse;
};

// All independent active sets. Equality multipliers are unrestricted; the
// remaining multipliers must be nonnegative. Factorizations are point constants.
std::vector<Face> faces(const Matrix& gram, std::size_t m, std::size_t equalities)
{
    std::vector<Face> out;
    const std::size_t inequalities = m - equalities;
    for (std::size_t mask = 0; mask < (std::size_t{1} << inequalities); ++mask) {
        Face f;
        for (std::size_t i = 0; i < equalities; ++i) f.rows.push_back(i);
        for (std::size_t i = 0; i < inequalities; ++i)
            if ((mask >> i) & 1) f.rows.push_back(equalities + i);
        const std::size_t k = f.rows.size();
        Matrix block(k * k);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) block[i*k+j] = gram[f.rows[i]*m+f.rows[j]];
        if (!k || detail::invert_information(block, k, f.inverse) == Inference::available)
            out.push_back(std::move(f));
    }
    return out;
}

// min ||h-z||_I^2 subject to the given equalities/inequalities. All arithmetic
// uses the constraint covariance R I^-1 R'; no full projected path is needed.
double distance(const Matrix& gram, std::span<const double> rz, std::size_t equalities,
                const std::vector<Face>& sets, Matrix& multipliers)
{
    const std::size_t m = rz.size();
    const double tolerance = 2048 * eps * (1 + std::abs(*std::max_element(rz.begin(), rz.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); })));
    for (const auto& f : sets) {
        const std::size_t k = f.rows.size();
        bool valid = true;
        for (std::size_t i = 0; i < k; ++i) {
            double v = 0;
            for (std::size_t j = 0; j < k; ++j) v -= f.inverse[i*k+j] * rz[f.rows[j]];
            multipliers[i] = v;
            if (i >= equalities && v < -std::abs(tolerance)) valid = false;
        }
        if (!valid) continue;
        for (std::size_t i = equalities; i < m && valid; ++i) {
            double v = rz[i];
            for (std::size_t j = 0; j < k; ++j) v += gram[i*m+f.rows[j]] * multipliers[j];
            if (v < -std::abs(tolerance)) valid = false;
        }
        if (!valid) continue;
        double cost = 0;
        for (std::size_t i = 0; i < k; ++i) cost -= multipliers[i] * rz[f.rows[i]];
        return cost;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

// Explicit conversion and Box-Muller keep the sequence independent of the
// implementation-defined std::normal_distribution algorithm. Seed: 0x534349564f4c.
double uniform(std::mt19937_64& random)
{
    return (static_cast<double>(random() >> 11) + 0.5) * 0x1.0p-53;
}

// Invert the Bernoulli Chernoff bound, Pr(D(p_hat || p) >= c/n) <= 2 exp(-c).
// This has finite-sample coverage, including zero exceedances in small tails.
double simulation_error(double p, std::size_t draws)
{
    const double target = std::log(20000.0)/static_cast<double>(draws); // alpha = 1e-4
    if (p == 0 || p == 1) return -std::expm1(-target);
    auto divergence = [p](double q) {
        return p*std::log(p/q)+(1-p)*(std::log1p(-p)-std::log1p(-q));
    };
    double lower = 0, inner = p;
    for (int i = 0; i < 56; ++i) {
        const double mid = (lower+inner)/2;
        if (divergence(mid) > target) lower = mid;
        else inner = mid;
    }
    double upper = 1;
    inner = p;
    for (int i = 0; i < 56; ++i) {
        const double mid = (upper+inner)/2;
        if (divergence(mid) > target) upper = mid;
        else inner = mid;
    }
    return std::max(p-lower, upper-p);
}

} // namespace

LRResult cone_projection_test(std::span<const double> information, std::size_t d,
                             std::span<const double> normals, std::span<const double> equalities,
                             double statistic, double level, std::size_t draws)
{
    LRResult out;
    out.law = ReferenceLaw::cone_projection;
    out.statistic = statistic;
    if (!d || information.size() != d*d || normals.size()%d || equalities.empty() || equalities.size()%d
        || !(statistic >= 0) || !std::isfinite(statistic) || !(level > 0 && level < 1) || draws < 32)
        return out;
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j)
            if (!std::isfinite(information[i*d+j]) || std::abs(information[i*d+j]-information[j*d+i])
                > 64*eps*std::max({1.0, std::abs(information[i*d+j]), std::abs(information[j*d+i])})) return out;
    const std::size_t e = equalities.size()/d, b = normals.size()/d, m = e+b;
    if (b > 12) { out.status = LRStatus::unresolved; out.message = "Cone active-set budget exceeded."; return out; }
    Matrix inverse;
    if (detail::invert_information(Matrix(information.begin(), information.end()), d, inverse) != Inference::available) {
        out.status = LRStatus::singular_information;
        out.message = "Cone calibration requires regular parent information.";
        return out;
    }
    Matrix rows(equalities.begin(), equalities.end());
    rows.insert(rows.end(), normals.begin(), normals.end());
    Matrix gram(m*m, 0), chol(d*d, 0);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double v = inverse[i*d+j];
            for (std::size_t k = 0; k < j; ++k) v -= chol[i*d+k]*chol[j*d+k];
            if (i == j) {
                if (!(v > 0)) { out.status = LRStatus::singular_information; return out; }
                chol[i*d+j] = std::sqrt(v);
            } else chol[i*d+j] = v/chol[j*d+j];
        }
    Matrix whitened(m*d, 0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < d; ++j)
            for (std::size_t k = j; k < d; ++k) whitened[i*d+j] += rows[i*d+k]*chol[k*d+j];
        double norm = 0;
        for (std::size_t j = 0; j < d; ++j) norm = std::hypot(norm, whitened[i*d+j]);
        if (!(norm > 0) || !std::isfinite(norm)) return out;
        for (std::size_t j = 0; j < d; ++j) whitened[i*d+j] /= norm;
    }
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t k = 0; k < d; ++k) gram[i*m+j] += whitened[i*d+k]*whitened[j*d+k];
    // Profile out the orthogonal Gaussian complement before drawing: the law
    // depends only on the span of tested directions and active face normals.
    Matrix basis(m*d, 0), reduced(m*m, 0), v(d);
    std::size_t rank = 0;
    for (std::size_t i = 0; i < m; ++i) {
        std::copy_n(whitened.data()+i*d, d, v.data());
        for (std::size_t j = 0; j < rank; ++j) {
            double coefficient = 0;
            for (std::size_t k = 0; k < d; ++k) coefficient += v[k]*basis[j*d+k];
            reduced[i*m+j] = coefficient;
            for (std::size_t k = 0; k < d; ++k) v[k] -= coefficient*basis[j*d+k];
        }
        double norm = 0;
        for (double x : v) norm = std::hypot(norm, x);
        if (norm > 2048*eps) {
            reduced[i*m+rank] = norm;
            for (std::size_t k = 0; k < d; ++k) basis[rank*d+k] = v[k]/norm;
            ++rank;
        }
    }
    Matrix parent_gram(b*b);
    for (std::size_t i = 0; i < b; ++i)
        for (std::size_t j = 0; j < b; ++j) parent_gram[i*b+j] = gram[(e+i)*m+e+j];
    const auto parent_sets = faces(parent_gram, b, 0), child_sets = faces(gram, m, e);
    if (child_sets.empty()) { out.status = LRStatus::singular_information; return out; }
    if (statistic == 0) { out.status = LRStatus::ok; out.p_value = 1; out.numerical_error = 0; return out; }
    std::mt19937_64 random(0x534349564f4cULL);
    Matrix z(rank), rz(m), multipliers(m);
    std::size_t exceed = 0;
    for (std::size_t draw = 0; draw < draws; ++draw) {
        for (std::size_t j = 0; j < rank; j += 2) {
            const double radius = std::sqrt(-2*std::log(uniform(random)));
            const double angle = 2*std::numbers::pi*uniform(random);
            z[j] = radius*std::cos(angle);
            if (j+1 < rank) z[j+1] = radius*std::sin(angle);
        }
        for (std::size_t i = 0; i < m; ++i) {
            rz[i] = 0;
            for (std::size_t j = 0; j < rank; ++j) rz[i] += reduced[i*m+j]*z[j];
        }
        const double dp = b ? distance(parent_gram, std::span(rz).subspan(e), 0, parent_sets, multipliers) : 0;
        const double dc = distance(gram, rz, e, child_sets, multipliers);
        if (!std::isfinite(dp) || !std::isfinite(dc) || dc-dp < -4096*eps*(1+dc+dp)) {
            out.status = LRStatus::unresolved; out.message = "Cone projection QP could not be resolved."; return out;
        }
        exceed += dc-dp >= statistic;
    }
    out.p_value = static_cast<double>(exceed)/draws;
    out.numerical_error = simulation_error(out.p_value, draws);
    out.status = (out.p_value-out.numerical_error <= level && level <= out.p_value+out.numerical_error)
        ? LRStatus::unresolved : LRStatus::ok;
    out.message = "Self-Liang / Andrews cone-projection law; mt19937_64 seed 0x534349564f4c, "
        + std::to_string(draws) + " Gaussian draws; 99.99% Bernoulli Chernoff Monte Carlo interval.";
    return out;
}

} // namespace scivolapi
