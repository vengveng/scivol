#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "jet.hpp"
#include "model.hpp"

namespace scivolapi {

enum class CorrelationDensity { normal, student_t };

struct CorrelationSpec {
    std::size_t p = 1, q = 1;
    CorrelationDensity density = CorrelationDensity::normal;
};

// Borrows standardized residuals (rows x series, row-major), owns Qbar.
struct CorrelationSample {
    std::span<const double> eps;
    std::size_t series = 0;
    std::vector<double> target;

    [[nodiscard]] std::size_t rows() const noexcept { return series ? eps.size() / series : 0; }
};

class CorrelationModel {
public:
    explicit CorrelationModel(const CorrelationSpec& spec);

    [[nodiscard]] const CorrelationSpec& spec() const noexcept { return spec_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return names_.size(); }
    [[nodiscard]] const std::vector<std::string>& parameter_names() const noexcept { return names_; }

    [[nodiscard]] CorrelationSample prepare(std::span<const double> eps, std::size_t series,
                                            std::span<const double> target = {}) const;
    [[nodiscard]] bool feasible(std::span<const double> theta) const;
    // Parameter-only checks; the jet also checks positive definiteness of every Q_t.
    [[nodiscard]] bool evaluable(std::span<const double> theta) const;
    void bounds(std::span<double> lower, std::span<double> upper) const;
    [[nodiscard]] std::size_t constraint_count() const noexcept;
    void constraints(std::span<const double> theta, std::span<double> values,
                     std::span<double> jacobian) const;
    [[nodiscard]] std::vector<double> start() const;
    void to_theta(std::span<const double> z, std::span<double> theta) const;
    [[nodiscard]] bool to_z(std::span<const double> theta, std::span<double> z) const;

private:
    CorrelationSpec spec_;
    std::vector<std::string> names_;
};

// Owns its workspace, borrows both model and sample. Evaluations allocate nothing.
// Q_0 = Qbar is constant; every row is scored, including row zero.
class CorrelationJet {
public:
    CorrelationJet(const CorrelationModel& model, const CorrelationSample& sample);

    [[nodiscard]] const CorrelationModel& model() const noexcept { return model_; }
    [[nodiscard]] const CorrelationSample& sample() const noexcept { return sample_; }

    // `paths` writes correlations (rows x series x series). JetOutput.paths is unused.
    JetResult theta(std::span<const double> theta, unsigned request, JetOutput& out,
                    std::span<double> correlations = {});
    // Scores and correlation paths require theta coordinates.
    JetResult z(std::span<const double> z, unsigned request, JetOutput& out);

private:
    const CorrelationModel& model_;
    const CorrelationSample& sample_;
    std::vector<double> work_;
};

// The innovation density of one marginal model: its density and shape values.
struct MarginalDensity {
    DensityKind density = DensityKind::normal;
    std::vector<double> shape;
};

// Generated innovations and correlations: paths x horizon x series and
// paths x horizon x series x series, row-major.
struct CorrelationPaths {
    std::size_t paths = 0, horizon = 0, series = 0;
    std::vector<double> innovation, correlation;
};

// Continues the correlation model at theta past the sample eps (rows x series:
// the marginal innovations it was fitted on) along `paths` paths. At each step:
// R_t from the recursion; a copula draw e* = L g with L L' = R_t (Student-t:
// e* = sqrt((nu - 2) / c) L g, c ~ chi2(nu)); and the marginal innovations
// z_i = F_i^{-1}(G(e*_i)), G the copula's standardized marginal and F_i the
// marginal densities, which drive the recursion. normals: paths x horizon x series
// standard normal draws; chi2: paths x horizon draws (Student-t only). Each
// marginal model then continues along its column of z.
CorrelationPaths simulate_correlation(const CorrelationModel& model, std::span<const double> theta,
                                      std::span<const double> eps, std::size_t series,
                                      std::span<const double> target, std::span<const MarginalDensity> marginals,
                                      std::span<const double> normals, std::span<const double> chi2,
                                      std::size_t paths);

}  // namespace scivolapi
