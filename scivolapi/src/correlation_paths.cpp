#include <scivolapi/correlation.hpp>

#include <cmath>
#include <stdexcept>

#include "correlation/recursion.hpp"
#include "correlation/work.hpp"
#include "density/distribution.hpp"
#include "density/domain.hpp"

namespace scivolapi {

namespace {

namespace dc = detail::correlation;

void require(bool ok, const char* message)
{
    if (!ok) throw std::invalid_argument(message);
}

void check(detail::ExpectationStatus status)
{
    if (status != detail::ExpectationStatus::ok)
        throw std::domain_error("a copula or marginal probability could not be computed accurately");
}

// Upper-tail probabilities below this are inverted one at a time from the upper
// tail; the rest in one batch per series and step.
constexpr double far_upper = 1e-6;

}  // namespace

CorrelationPaths simulate_correlation(const CorrelationModel& model, std::span<const double> theta,
                                      std::span<const double> eps, std::size_t series,
                                      std::span<const double> target, std::span<const InnovationDensity> marginals,
                                      std::span<const double> normals, std::span<const double> chi2,
                                      std::size_t paths)
{
    const CorrelationSpec& spec = model.spec();
    const bool student = spec.density == CorrelationDensity::student_t;
    const CorrelationSample sample = model.prepare(eps, series, target);
    const std::size_t n = series, m = n * n, rows = sample.rows();
    require(theta.size() == model.dimension() && model.evaluable(theta),
            "theta must have the model's dimension and lie where the model is defined");
    require(marginals.size() == n, "one marginal density per series is required");
    for (const InnovationDensity& marginal : marginals)
        require(marginal.shape.size() == detail::shape_domains(marginal.density).size(),
                "each marginal needs its density's shape parameters");
    require(paths > 0 && normals.size() % (paths * n) == 0, "normals must hold paths x horizon x series draws");
    const std::size_t horizon = normals.size() / (paths * n);

    const dc::Layout layout(spec);
    const double eta = student ? theta[layout.coefficients] : 0.0;
    require(!student || eta == 0.0 || chi2.size() == paths * horizon,
            "the Student-t copula needs paths x horizon draws of chi-square divided by degrees of freedom");
    if (student && eta > 0.0)
        for (double draw : chi2) require(draw > 0.0 && std::isfinite(draw),
                                         "scaled chi-square draws must be positive and finite");
    const DensityKind copula = student ? DensityKind::student_t : DensityKind::normal;
    const double* copula_shape = student ? &eta : nullptr;
    auto identity = [&](const InnovationDensity& marginal) {  // F = G: z = e exactly
        return marginal.density == copula && (!student || marginal.shape[0] == eta);
    };

    // Filter the observed rows: Q of the last q + 1 rows, slot r % (q + 1).
    const std::size_t slots = spec.q + 1;
    std::vector<double> observed(slots * m);
    std::copy_n(sample.target.data(), m, observed.data());
    for (std::size_t t = 1; t < rows; ++t)
        dc::q_step(layout, theta.data(), sample.target.data(), n, std::min(spec.p, t), std::min(spec.q, t),
                   [&](std::size_t lag) { return sample.eps.data() + (t - lag) * n; },
                   [&](std::size_t lag) { return observed.data() + ((t - lag) % slots) * m; },
                   observed.data() + (t % slots) * m);

    CorrelationPaths out;
    out.paths = paths;
    out.horizon = horizon;
    out.series = n;
    out.innovation.assign(paths * horizon * n, 0.0);
    out.correlation.assign(paths * horizon * m, 0.0);
    // Steps outer, paths inner, so each series' quantiles at a step are one batch.
    // Path state: its Q rows (slot r % (q + 1)); its innovations are in `out`.
    std::vector<double> ring(paths * slots * m), chol(m), scale(n), draws(paths * n), u(paths), z(paths);
    std::vector<std::size_t> batch(paths);
    for (std::size_t f = 0; f < horizon; ++f) {
        const std::size_t t = rows + f;
        for (std::size_t path = 0; path < paths; ++path) {
            const double* innovation = out.innovation.data() + path * horizon * n;
            double* own = ring.data() + path * slots * m;
            double* q = own + (t % slots) * m;
            dc::q_step(layout, theta.data(), sample.target.data(), n, std::min(spec.p, t), std::min(spec.q, t),
                       [&](std::size_t lag) {
                           return t - lag < rows ? sample.eps.data() + (t - lag) * n
                                                 : innovation + (t - lag - rows) * n;
                       },
                       [&](std::size_t lag) { return (t - lag < rows ? observed.data() : own) + ((t - lag) % slots) * m; },
                       q);
            double* r = out.correlation.data() + (path * horizon + f) * m;
            for (std::size_t i = 0; i < n; ++i) scale[i] = std::sqrt(q[i * n + i]);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) r[i * n + j] = (q[i * n + j] / scale[i]) / scale[j];
            if (dc::cholesky(r, n, chol.data()) != Status::ok)
                throw std::domain_error("a simulated correlation matrix is not positive definite in double precision");
            const double* g = normals.data() + (path * horizon + f) * n;
            const double factor = student && eta > 0.0 ? std::sqrt((1.0 - 2.0 * eta) / chi2[path * horizon + f]) : 1.0;
            for (std::size_t i = 0; i < n; ++i) {
                double e = 0.0;
                for (std::size_t j = 0; j <= i; ++j) e += chol[i * n + j] * g[j];
                draws[path * n + i] = factor * e;
            }
        }
        for (std::size_t i = 0; i < n; ++i) {  // z_i = F_i^{-1}(G(e*_i))
            const InnovationDensity& marginal = marginals[i];
            const double* shape = marginal.shape.empty() ? nullptr : marginal.shape.data();
            std::size_t count = 0;
            for (std::size_t path = 0; path < paths; ++path) {
                double& slot = out.innovation[(path * horizon + f) * n + i];
                const double e = draws[path * n + i];
                if (identity(marginal)) {
                    slot = e;
                    continue;
                }
                double lower = 0.0, upper = 0.0;
                check(detail::distribution(copula, copula_shape, e, lower, upper));
                if (upper < far_upper) {
                    check(detail::quantile_upper(marginal.density, shape, upper, slot));
                } else {
                    u[count] = lower;
                    batch[count++] = path;
                }
            }
            if (count == 0) continue;
            check(detail::quantiles(marginal.density, shape, u.data(), count, z.data()));
            for (std::size_t c = 0; c < count; ++c) out.innovation[(batch[c] * horizon + f) * n + i] = z[c];
        }
        for (std::size_t path = 0; path < paths; ++path)
            if (!dc::finite(out.innovation.data() + (path * horizon + f) * n, n))
                throw std::domain_error("a simulated innovation is not representable");
    }
    return out;
}

}  // namespace scivolapi
