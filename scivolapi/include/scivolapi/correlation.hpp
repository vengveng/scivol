#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "jet.hpp"

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

}  // namespace scivolapi
