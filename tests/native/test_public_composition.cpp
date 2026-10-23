#include <scivolapi/scivolapi.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace scivolapi;

std::size_t checks = 0;
const char* current_case = "oracle self-check";
constexpr double sentinel = -0x1.23456789abcp+20;

void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        std::fprintf(stderr, "public composition [%s] line %d: %s\n",
            current_case, line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)

template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "public composition [%s] setup status %u: %s\n",
            current_case, static_cast<unsigned>(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}

void near(long double actual, long double expected, long double absolute = 2e-10L,
          long double relative = 3e-10L) {
    ++checks;
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > absolute + relative * std::max(std::abs(actual), std::abs(expected))) {
        std::fprintf(stderr, "public composition [%s]: actual %.18Lg, expected %.18Lg\n",
            current_case, actual, expected);
        std::exit(EXIT_FAILURE);
    }
}

// Independent forward two-jet of scalar equations; all variable-size state is on the heap.
struct Differential {
    long double value;
    std::vector<long double> gradient, hessian;

    Differential(long double v, std::size_t k) : value(v), gradient(k), hessian(k * k) {}
    static Differential variable(double value, std::size_t k, std::size_t axis) {
        Differential result(value, k);
        result.gradient[axis] = 1;
        return result;
    }
};

Differential operator+(const Differential& a, const Differential& b) {
    CHECK(a.gradient.size() == b.gradient.size());
    Differential result(a.value + b.value, a.gradient.size());
    for (std::size_t i = 0; i < result.gradient.size(); ++i)
        result.gradient[i] = a.gradient[i] + b.gradient[i];
    for (std::size_t i = 0; i < result.hessian.size(); ++i)
        result.hessian[i] = a.hessian[i] + b.hessian[i];
    return result;
}

Differential operator*(const Differential& a, const Differential& b) {
    CHECK(a.gradient.size() == b.gradient.size());
    const auto k = a.gradient.size();
    Differential result(a.value * b.value, k);
    for (std::size_t i = 0; i < k; ++i) {
        result.gradient[i] = a.gradient[i] * b.value + a.value * b.gradient[i];
        for (std::size_t j = 0; j < k; ++j)
            result.hessian[i * k + j] = a.hessian[i * k + j] * b.value
                + a.value * b.hessian[i * k + j]
                + a.gradient[i] * b.gradient[j] + b.gradient[i] * a.gradient[j];
    }
    return result;
}

Differential operator*(long double scale, const Differential& a) {
    Differential result(a);
    result.value *= scale;
    for (auto& x : result.gradient) x *= scale;
    for (auto& x : result.hessian) x *= scale;
    return result;
}
Differential operator-(const Differential& a, const Differential& b) { return a + (-1.L * b); }

Differential unary(const Differential& a, long double value, long double slope, long double curvature) {
    const auto k = a.gradient.size();
    Differential result(value, k);
    for (std::size_t i = 0; i < k; ++i) {
        result.gradient[i] = slope * a.gradient[i];
        for (std::size_t j = 0; j < k; ++j)
            result.hessian[i * k + j] = slope * a.hessian[i * k + j]
                + curvature * a.gradient[i] * a.gradient[j];
    }
    return result;
}
Differential logarithm(const Differential& a) {
    CHECK(a.value > 0);
    return unary(a, std::log(a.value), 1 / a.value, -1 / (a.value * a.value));
}
Differential exponential(const Differential& a) {
    const auto value = std::exp(a.value);
    return unary(a, value, value, value);
}
Differential reciprocal(const Differential& a) {
    CHECK(a.value != 0);
    const auto value = 1 / a.value;
    return unary(a, value, -value * value, 2 * value * value * value);
}
Differential operator/(const Differential& a, const Differential& b) { return a * reciprocal(b); }
Differential magnitude(const Differential& a) {
    if (a.value != 0) return a.value < 0 ? -1.L * a : a;
    // Only structurally constant presample residuals may use this zero branch.
    for (auto x : a.gradient) CHECK(x == 0);
    for (auto x : a.hessian) CHECK(x == 0);
    return a;
}

void oracle_self_check() {
    const auto x = Differential::variable(2, 2, 0);
    const auto y = Differential::variable(.3, 2, 1);
    const auto result = logarithm(x) + x * y + exponential(y) / x;
    const long double e = std::exp(static_cast<long double>(.3));
    near(result.value, std::log(2.L) + 2.L * .3 + e / 2, 1e-14L, 1e-14L);
    near(result.gradient[0], .5L + .3 - e / 4, 1e-14L, 1e-14L);
    near(result.gradient[1], 2 + e / 2, 1e-14L, 1e-14L);
    near(result.hessian[0], -.25L + e / 4, 1e-14L, 1e-14L);
    near(result.hessian[1], 1 - e / 4, 1e-14L, 1e-14L);
    near(result.hessian[2], 1 - e / 4, 1e-14L, 1e-14L);
    near(result.hessian[3], e / 2, 1e-14L, 1e-14L);
}

enum class Mean { zero, arma, linear };
enum class Volatility { constant, garch, gjr, egarch, aparch };
struct Case {
    const char* name;
    Mean mean;
    Volatility volatility;
    bool intercept;
    std::size_t ar, ma, columns, p, q;
    DispatchKind dispatch;
};

// Expectations are declared here, not obtained from model/sample/backend metadata.
constexpr Case cases[] = {
    {"zero-garch22-fixed", Mean::zero, Volatility::garch, false, 0, 0, 0, 2, 2, DispatchKind::fixed},
    {"zero-gjr32-runtime", Mean::zero, Volatility::gjr, false, 0, 0, 0, 3, 2, DispatchKind::runtime},
    {"mean-arma11-fixed", Mean::arma, Volatility::constant, true, 1, 1, 0, 0, 0, DispatchKind::fixed},
    {"mean-arma23-no-intercept-runtime", Mean::arma, Volatility::constant, false, 2, 3, 0, 0, 0, DispatchKind::runtime},
    {"mean-linear3-intercept-fixed", Mean::linear, Volatility::constant, true, 0, 0, 3, 0, 0, DispatchKind::fixed},
    {"mean-linear4-no-intercept-fixed", Mean::linear, Volatility::constant, false, 0, 0, 4, 0, 0, DispatchKind::fixed},
    {"mean-linear2-no-intercept-runtime", Mean::linear, Volatility::constant, false, 0, 0, 2, 0, 0, DispatchKind::runtime},
    {"zero-egarch11-fixed", Mean::zero, Volatility::egarch, false, 0, 0, 0, 1, 1, DispatchKind::fixed},
    {"zero-egarch32-runtime", Mean::zero, Volatility::egarch, false, 0, 0, 0, 3, 2, DispatchKind::runtime},
    {"zero-aparch11-fixed", Mean::zero, Volatility::aparch, false, 0, 0, 0, 1, 1, DispatchKind::fixed},
    {"zero-aparch23-runtime", Mean::zero, Volatility::aparch, false, 0, 0, 0, 2, 3, DispatchKind::runtime},
    {"joint-arma11-garch11-fixed", Mean::arma, Volatility::garch, true, 1, 1, 0, 1, 1, DispatchKind::fixed},
    {"joint-arma23-gjr32-no-intercept-runtime", Mean::arma, Volatility::gjr, false, 2, 3, 0, 3, 2, DispatchKind::runtime},
    {"joint-linear3-garch11-intercept-fixed", Mean::linear, Volatility::garch, true, 0, 0, 3, 1, 1, DispatchKind::fixed},
    {"joint-linear4-gjr11-no-intercept-fixed", Mean::linear, Volatility::gjr, false, 0, 0, 4, 1, 1, DispatchKind::fixed},
    {"joint-linear2-gjr23-no-intercept-runtime", Mean::linear, Volatility::gjr, false, 0, 0, 2, 2, 3, DispatchKind::runtime},
    {"joint-arma11-egarch11-fixed", Mean::arma, Volatility::egarch, true, 1, 1, 0, 1, 1, DispatchKind::fixed},
    {"joint-arma23-egarch32-no-intercept-runtime", Mean::arma, Volatility::egarch, false, 2, 3, 0, 3, 2, DispatchKind::runtime},
    {"joint-arma11-aparch11-fixed", Mean::arma, Volatility::aparch, true, 1, 1, 0, 1, 1, DispatchKind::fixed},
    {"joint-arma23-aparch23-no-intercept-runtime", Mean::arma, Volatility::aparch, false, 2, 3, 0, 2, 3, DispatchKind::runtime},
    {"joint-linear3-egarch11-intercept-fixed", Mean::linear, Volatility::egarch, true, 0, 0, 3, 1, 1, DispatchKind::fixed},
    {"joint-linear2-egarch32-no-intercept-runtime", Mean::linear, Volatility::egarch, false, 0, 0, 2, 3, 2, DispatchKind::runtime},
    {"joint-linear3-aparch11-intercept-fixed", Mean::linear, Volatility::aparch, true, 0, 0, 3, 1, 1, DispatchKind::fixed},
    {"joint-linear4-aparch11-no-intercept-fixed", Mean::linear, Volatility::aparch, false, 0, 0, 4, 1, 1, DispatchKind::fixed},
    {"joint-linear2-aparch23-no-intercept-runtime", Mean::linear, Volatility::aparch, false, 0, 0, 2, 2, 3, DispatchKind::runtime}
};

bool power(Case c) { return c.volatility == Volatility::egarch || c.volatility == Volatility::aparch; }
bool joint(Case c) { return c.mean != Mean::zero && c.volatility != Volatility::constant; }
std::size_t mean_width(Case c) {
    return c.mean == Mean::zero ? 0 : std::size_t(c.intercept) + (c.mean == Mean::arma ? c.ar + c.ma : c.columns);
}
std::size_t volatility_width(Case c) {
    return c.volatility == Volatility::constant ? 1
        : 1 + ((power(c) || c.volatility == Volatility::gjr) ? 2 : 1) * c.p + c.q
            + std::size_t(c.volatility == Volatility::aparch);
}
std::size_t recursion_begin(Case c) {
    if (c.mean != Mean::arma) return 0;
    return c.volatility == Volatility::constant ? std::max({std::size_t{1}, c.ar, c.ma})
        : std::max({c.ar, c.ma, c.p, c.q});
}

struct Fixture {
    Case config;
    bool student;
    std::size_t hold_back = 2, first, stop;
    std::vector<double> observations, design, theta;
    std::vector<std::uint64_t> rows;
    std::vector<std::string> names;

    explicit Fixture(Case c, bool student_t = false)
        : config(c), student(student_t), first(hold_back + recursion_begin(c) + 2),
          observations{.31, -.73, 1.17, -.29, .57, -1.37, .93, .43, -.67,
                       .83, -.47, 1.23, .71, -.91, .53, -1.11, 2.31} {
        stop = observations.size() - 1;
        for (std::size_t t = 0; t < observations.size(); ++t) {
            rows.push_back(101 + 3 * t + t * t);
            for (std::size_t j = 0; j < c.columns; ++j)
                design.push_back((static_cast<double>((t + 2 * j) % 7) - 3) / (j + 2));
        }
        const auto append = [&](std::string name, double value) {
            names.push_back(std::move(name)); theta.push_back(value);
        };
        if (c.intercept) append("intercept", .03);
        for (std::size_t i = 0; i < c.ar; ++i)
            append("ar[" + std::to_string(i + 1) + "]", (i % 2 ? -.09 : .09) / (i + 1));
        for (std::size_t i = 0; i < c.ma; ++i)
            append("ma[" + std::to_string(i + 1) + "]", -.13 / (i + 1));
        for (std::size_t i = 0; i < c.columns; ++i)
            append("coefficient[" + std::to_string(i + 1) + "]", (i % 2 ? -.025 : .025) / (i + 1));
        if (c.volatility == Volatility::constant) append("sigma2", .93);
        else {
            append("omega", c.volatility == Volatility::egarch ? -.11 : .11);
            for (std::size_t i = 0; i < c.p; ++i)
                append("alpha[" + std::to_string(i + 1) + "]", .085 / (i + 1));
            if (power(c) || c.volatility == Volatility::gjr)
                for (std::size_t i = 0; i < c.p; ++i)
                    append("gamma[" + std::to_string(i + 1) + "]", -.025 / (i + 1));
            for (std::size_t i = 0; i < c.q; ++i)
                append("beta[" + std::to_string(i + 1) + "]", .38 / (i + 1));
            if (c.volatility == Volatility::aparch) append("delta", 1.7);
        }
        if (student) append("nu", 8.5);
        CHECK(theta.size() == mean_width(c) + volatility_width(c) + std::size_t(student));
    }

    DataView data(std::size_t count) const {
        DataView result;
        result.observations = std::span<const double>(observations).first(count);
        result.row_ids = std::span<const std::uint64_t>(rows).first(count);
        result.row_space = 73;
        result.units.response = "returns";
        if (config.columns) result.design = {std::span<const double>(design).first(count * config.columns), count, config.columns};
        return result;
    }
};

double default_seed(const Fixture& f, std::size_t stop) {
    long double sum = 0;
    for (std::size_t t = f.hold_back; t < stop; ++t) {
        const long double y = f.observations[t];
        sum += y * y;
    }
    return static_cast<double>(sum / (stop - f.hold_back));
}

Model model_for(const Fixture& f) {
    const auto c = f.config;
    ModelSpec spec;
    if (c.mean == Mean::arma) spec.mean = ARMA{c.ar, c.ma, c.intercept};
    else if (c.mean == Mean::linear) spec.mean = LinearMean{c.intercept};
    else spec.mean = ZeroMean{};
    switch (c.volatility) {
    case Volatility::constant: spec.volatility = ConstantVariance{}; break;
    case Volatility::garch: spec.volatility = GARCH{c.p, c.q}; break;
    case Volatility::gjr: spec.volatility = GJRGARCH{c.p, c.q}; break;
    case Volatility::egarch: spec.volatility = EGARCH{c.p, c.q}; break;
    case Volatility::aparch: spec.volatility = APARCH{c.p, c.q}; break;
    }
    if (f.student) spec.density = StudentT{}; else spec.density = Normal{};
    return take(compile(spec, {c.columns}));
}

void verify_layout(const Fixture& f, const Model& model, const ObjectiveView& objective) {
    const auto& layout = model.layout();
    const auto m = mean_width(f.config), v = volatility_width(f.config), k = f.theta.size();
    CHECK(layout.full_theta_width == k && layout.entries.size() == k);
    CHECK(layout.blocks.size() == (f.config.mean == Mean::zero ? 2 : 3));
    std::size_t b = 0;
    const auto block = [&](std::string_view name, std::size_t offset, std::size_t width) {
        CHECK(layout.blocks[b].name == name);
        CHECK(layout.blocks[b].offset == offset && layout.blocks[b].width == width);
        ++b;
    };
    if (f.config.mean != Mean::zero) block("mean", 0, m);
    block("volatility", m, v);
    block("density", m + v, std::size_t(f.student));
    const auto& derivatives = objective.derivative_layout();
    CHECK(derivatives.axes.size() == k && derivatives.full_parameter_partition.size() == k);
    CHECK(derivatives.target == DerivativeTarget::full_theta);
    CHECK(objective.key().scaling == Scaling::additive_nll);
    for (std::size_t i = 0; i < k; ++i) {
        CHECK(layout.entries[i].name == f.names[i] && layout.entries[i].index == i);
        CHECK(layout.entries[i].id != 0 && derivatives.axes[i] == layout.entries[i].id);
        CHECK(derivatives.full_parameter_partition[i] == ParameterDisposition::free);
        for (std::size_t j = 0; j < i; ++j) CHECK(layout.entries[i].id != layout.entries[j].id);
    }
    CHECK(model.capabilities().request_mask == 7);
    CHECK(model.capabilities().theta_dispatch == f.config.dispatch);
}

PreparedProblem problem_for(const Fixture& f, const Model& model, std::optional<double> seed = {},
                           bool prefix = false) {
    Conditioning conditioning;
    conditioning.hold_back = f.hold_back;
    conditioning.variance_seed = seed;
    return take(prepare(model, f.data(prefix ? f.stop : f.observations.size()), conditioning, {f.first, f.stop}));
}

void verify_sample(const Fixture& f, const PreparedProblem& problem, std::optional<double> seed = {}) {
    const auto& sample = problem.sample();
    const auto n = f.observations.size(), begin = recursion_begin(f.config);
    CHECK(sample.row_space == 73 && sample.clock == ObservationClock::observation_rows);
    CHECK(sample.original_n_obs == n && sample.retained_begin_position == f.hold_back);
    CHECK(sample.retained_n_obs == n - f.hold_back);
    CHECK(sample.first_eligible_original_position == f.hold_back + begin);
    CHECK(sample.first_scored_original_position == f.first && sample.n_eff == f.stop - f.first);
    CHECK(sample.retained_rows.size() == n - f.hold_back && sample.scored_rows.size() == f.stop - f.first);
    for (std::size_t i = 0; i < sample.retained_rows.size(); ++i) {
        CHECK(sample.retained_rows[i] == f.rows[f.hold_back + i]);
        CHECK(sample.original_position(i) == f.hold_back + i);
    }
    CHECK(!sample.original_position(n - f.hold_back));
    for (std::size_t i = 0; i < sample.scored_rows.size(); ++i) CHECK(sample.scored_rows[i] == f.rows[f.first + i]);
    CHECK(std::get<UnitDescriptor>(sample.units).labels.response == "returns" && std::get<UnitDescriptor>(sample.units).public_scale == 1);
    const auto& conditioning = std::get<ConditioningDescriptor>(sample.conditioning);
    CHECK(conditioning.observation_history.empty() && conditioning.innovation_history.empty());
    if (f.config.volatility == Volatility::constant) {
        CHECK(!conditioning.variance_seed && conditioning.source_rows.empty());
    } else {
        CHECK(conditioning.variance_seed);
        near(*conditioning.variance_seed, seed.value_or(default_seed(f, n)), 2e-15L, 2e-15L);
        CHECK(conditioning.seed_origin == (seed ? SeedOrigin::supplied : SeedOrigin::model_default));
        if (seed) CHECK(conditioning.source_rows.empty());
        else {
            CHECK(conditioning.source_rows.size() == n - f.hold_back);
            for (std::size_t i = 0; i < conditioning.source_rows.size(); ++i)
                CHECK(conditioning.source_rows[i] == f.rows[f.hold_back + i]);
            CHECK(conditioning.source_rows.back() == f.rows.back());
            CHECK(std::abs(default_seed(f, n) - default_seed(f, f.stop)) > .1);
        }
    }
}

// Direct declared ARMA/linear and variance equations, with startup independent of scoring.
// The diagnostic restart deliberately discards warmup state to prove the fixtures detect that bug.
Differential oracle(const Fixture& f, double seed, bool derivatives, bool restart_at_score = false) {
    CHECK(!derivatives || !f.student);
    const auto c = f.config;
    const auto k = derivatives ? f.theta.size() : 0, m = mean_width(c);
    const auto constant = [k](long double value) { return Differential(value, k); };
    std::vector<Differential> theta;
    for (std::size_t i = 0; i < f.theta.size(); ++i)
        theta.push_back(derivatives ? Differential::variable(f.theta[i], k, i) : constant(f.theta[i]));
    const auto n = f.stop - f.hold_back;
    const auto begin = restart_at_score ? f.first - f.hold_back : recursion_begin(c);
    std::vector<Differential> residuals(n, constant(0)), log_variances(n, constant(std::log(static_cast<long double>(seed))));
    std::vector<Differential> variances(n, constant(seed)), standardized(n, constant(0));
    const auto delta = c.volatility == Volatility::aparch ? theta[m + volatility_width(c) - 1] : constant(2);
    std::vector<Differential> powers(n, exponential(.5L * std::log(static_cast<long double>(seed)) * delta));
    const auto pi = std::acos(-1.L);
    const long double nu = f.student ? f.theta.back() : 0;
    const auto moment = constant(f.student
        ? std::sqrt(nu - 2) * std::exp(std::lgamma((nu - 1) / 2) - std::lgamma(nu / 2)) / std::sqrt(pi)
        : std::sqrt(2 / pi));
    const auto beta_start = m + 1 + ((power(c) || c.volatility == Volatility::gjr) ? 2 : 1) * c.p;
    Differential total = constant(0);
    for (std::size_t t = begin; t < n; ++t) {
        auto e = constant(f.observations[f.hold_back + t]);
        if (c.intercept) e = e - theta[0];
        const auto offset = std::size_t(c.intercept);
        if (c.mean == Mean::arma) {
            for (std::size_t lag = 1; lag <= c.ar; ++lag)
                e = e - theta[offset + lag - 1] * constant(f.observations[f.hold_back + t - lag]);
            for (std::size_t lag = 1; lag <= c.ma; ++lag)
                e = e - theta[offset + c.ar + lag - 1] * residuals[t - lag];
        } else if (c.mean == Mean::linear) {
            for (std::size_t j = 0; j < c.columns; ++j)
                e = e - theta[offset + j] * constant(f.design[(f.hold_back + t) * c.columns + j]);
        }
        CHECK(std::abs(e.value) > .02L);
        residuals[t] = e;
        // ARMA joint startup recurs at begin; zero/linear row zero is the fixed seed.
        const bool fixed_seed_row = c.mean != Mean::arma && t == (restart_at_score ? begin : 0);
        if (c.volatility == Volatility::constant) {
            log_variances[t] = logarithm(theta[m]);
        } else if (!fixed_seed_row) {
            auto state = theta[m];
            for (std::size_t lag = 1; lag <= c.p && lag <= t; ++lag) {
                const auto& previous = residuals[t - lag];
                const auto& alpha = theta[m + lag];
                if (c.volatility == Volatility::egarch) {
                    const auto& z = standardized[t - lag];
                    state = state + alpha * (magnitude(z) - moment) + theta[m + c.p + lag] * z;
                } else if (c.volatility == Volatility::aparch) {
                    // Fixed presample zeros contribute exactly zero, not a log(0) derivative.
                    if (previous.value == 0) { (void)magnitude(previous); continue; }
                    const auto base = magnitude(previous) - theta[m + c.p + lag] * previous;
                    state = state + alpha * exponential(delta * logarithm(base));
                } else {
                    auto coefficient = alpha;
                    if (c.volatility == Volatility::gjr && previous.value < 0)
                        coefficient = coefficient + theta[m + c.p + lag];
                    state = state + coefficient * previous * previous;
                }
            }
            for (std::size_t lag = 1; lag <= c.q && lag <= t; ++lag) {
                const auto& previous = c.volatility == Volatility::egarch ? log_variances[t - lag]
                    : c.volatility == Volatility::aparch ? powers[t - lag] : variances[t - lag];
                state = state + theta[beta_start + lag - 1] * previous;
            }
            if (c.volatility == Volatility::egarch) log_variances[t] = state;
            else if (c.volatility == Volatility::aparch) {
                powers[t] = state;
                log_variances[t] = 2.L * logarithm(state) / delta;
            } else {
                variances[t] = state;
                log_variances[t] = logarithm(state);
            }
        }
        standardized[t] = e * exponential(-.5L * log_variances[t]);
        if (f.hold_back + t < f.first) continue;
        const auto square = e * e * exponential(-1.L * log_variances[t]);
        if (f.student) {
            const auto normalizer = constant(.5L * std::log((nu - 2) * pi)
                + std::lgamma(nu / 2) - std::lgamma((nu + 1) / 2));
            total = total + .5L * log_variances[t] + normalizer
                + ((nu + 1) / 2) * logarithm(constant(1) + (1 / (nu - 2)) * square);
        } else total = total + .5L * (constant(std::log(2 * pi)) + log_variances[t] + square);
    }
    return total;
}

struct Arena {
    std::size_t bytes, alignment;
    std::byte* storage = nullptr;

    explicit Arena(const Requirements& plan) : bytes(0), alignment(0) {
        CHECK(plan.bytes > 0 && plan.bytes <= 4 * 1024 * 1024);
        CHECK(plan.alignment >= alignof(double) && plan.alignment <= 4096);
        CHECK((plan.alignment & (plan.alignment - 1)) == 0);
        CHECK(plan.bytes <= std::numeric_limits<std::size_t>::max() - 2 * plan.alignment);
        bytes = static_cast<std::size_t>(plan.bytes);
        alignment = static_cast<std::size_t>(plan.alignment);
        storage = static_cast<std::byte*>(::operator new(bytes + 2 * alignment, std::align_val_t{alignment}));
        std::fill_n(storage, bytes + 2 * alignment, std::byte{0xa5});
    }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    ~Arena() { ::operator delete(storage, std::align_val_t{alignment}); }
    std::span<std::byte> memory() { return {storage + alignment, bytes}; }
    void guards() const {
        for (std::size_t i = 0; i < alignment; ++i) {
            CHECK(storage[i] == std::byte{0xa5});
            CHECK(storage[alignment + bytes + i] == std::byte{0xa5});
        }
    }
};

struct Outputs {
    std::size_t k;
    LayoutId layout;
    std::vector<double> f, g, h;

    explicit Outputs(const ObjectiveView& objective)
        : k(objective.derivative_layout().axes.size()), layout(objective.key().derivative_layout),
          f(6, sentinel), g(k + 5, sentinel), h(k * k + 5, sentinel) {}
    JetBuffers buffers() {
        return {layout, std::span<double>(f).subspan(1, 4), std::span<double>(g).subspan(1, k + 3),
            {std::span<double>(h).subspan(1, k * k + 3), k}};
    }
    void untouched(unsigned mask) const {
        const auto block = [](const std::vector<double>& values, std::size_t logical, bool requested) {
            for (std::size_t i = 0; i < values.size(); ++i)
                if (!requested || i == 0 || i > logical) CHECK(values[i] == sentinel);
        };
        block(f, 1, mask & 1); block(g, k, mask & 2); block(h, k * k, mask & 4);
    }
};

Point named_point(const Fixture& f, const ObjectiveView& objective) {
    std::vector<std::string> names(f.names);
    std::vector<NamedParameter> values;
    for (std::size_t i = f.theta.size(); i-- > 0;) values.push_back({names[i], f.theta[i]});
    auto result = take(from_theta(objective, values));
    for (auto& parameter : values) parameter.value = sentinel;
    return result;
}

Outputs public_jet(const Fixture& f, const ObjectiveView& objective, unsigned mask,
                   const Point* owned_point = nullptr) {
    const auto request = static_cast<JetRequest>(mask);
    const auto plan = take(requirements(objective, request));
    CHECK(plan.problem == objective.key().problem && plan.derivative_layout == objective.key().derivative_layout);
    Arena arena(plan);
    const auto short_arena = bind_workspace(plan, arena.memory().first(arena.bytes - 1));
    CHECK(!short_arena && short_arena.error().code == Status::insufficient_capacity);
    arena.guards();
    auto workspace = take(bind_workspace(plan, arena.memory()));
    const auto binding = from_theta(objective, f.theta, workspace);
    CHECK(binding.status == Status::ok && binding.point);
    const auto point = owned_point ? owned_point->view() : *binding.point;
    CHECK(point.theta().size() == f.theta.size());
    for (std::size_t i = 0; i < f.theta.size(); ++i) CHECK(point.theta()[i] == f.theta[i]);
    Outputs outputs(objective);
    for (unsigned bit : {1u, 2u, 4u}) {
        if (!(mask & bit)) continue;
        auto short_outputs = outputs.buffers();
        if (bit == 1) short_outputs.value = short_outputs.value.first(0);
        if (bit == 2) short_outputs.gradient = short_outputs.gradient.first(outputs.k - 1);
        if (bit == 4) short_outputs.hessian.elements = short_outputs.hessian.elements.first(outputs.k * outputs.k - 1);
        const auto rejected = evaluate(objective, point, request, workspace, short_outputs);
        CHECK(rejected.status == Status::insufficient_capacity && rejected.valid_mask == 0);
        outputs.untouched(0);
        arena.guards();
    }
    if (mask & 4) {
        auto wrong_dimension = outputs.buffers();
        wrong_dimension.hessian.dimension = outputs.k + 1;
        const auto rejected = evaluate(objective, point, request, workspace, wrong_dimension);
        CHECK(rejected.status == Status::insufficient_capacity && rejected.valid_mask == 0);
        outputs.untouched(0);
    }
    const auto valid = evaluate(objective, point, request, workspace, outputs.buffers());
    CHECK(valid.status == Status::ok && valid.requested == request && valid.valid_mask == mask);
    CHECK(valid.contains(request));
    CHECK(valid.f_reason == ((mask & 1) ? Status::ok : Status::unsupported_operation));
    CHECK(valid.g_reason == ((mask & 2) ? Status::ok : Status::unsupported_operation));
    CHECK(valid.h_reason == ((mask & 4) ? Status::ok : Status::unsupported_operation));
    outputs.untouched(mask);
    arena.guards();
    return outputs;
}

void compare_jet(const Outputs& output, unsigned mask, const Differential& expected, const Outputs& full) {
    if (mask & 1) { near(output.f[1], expected.value); near(output.f[1], full.f[1], 2e-12L, 2e-12L); }
    if (mask & 2) for (std::size_t i = 0; i < output.k; ++i) {
        near(output.g[i + 1], expected.gradient[i], 2e-9L, 2e-9L);
        near(output.g[i + 1], full.g[i + 1], 2e-11L, 2e-11L);
    }
    if (mask & 4) for (std::size_t i = 0; i < output.k * output.k; ++i) {
        near(output.h[i + 1], expected.hessian[i], 2e-8L, 3e-9L);
        near(output.h[i + 1], full.h[i + 1], 2e-10L, 2e-11L);
        near(output.h[i + 1], output.h[(i % output.k) * output.k + i / output.k + 1], 2e-12L, 2e-12L);
    }
}

void all_requests_with_excess_capacity(const Fixture& f) {
    const auto model = model_for(f);
    const auto problem = problem_for(f, model);
    verify_sample(f, problem);
    const auto objective = take(resolve_objective(problem));
    verify_layout(f, model, objective);
    const auto expected = oracle(f, default_seed(f, f.observations.size()), true);
    if (f.config.mean != Mean::zero) {
        long double cross_block = 0;
        const auto m = mean_width(f.config), k = f.theta.size();
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = m; j < k; ++j)
                cross_block = std::max(cross_block, std::abs(expected.hessian[i * k + j]));
        CHECK(cross_block > 1e-6L);
    }
    const auto owned = named_point(f, objective);
    const auto full = public_jet(f, objective, 7, &owned);
    compare_jet(full, 7, expected, full);
    for (unsigned mask = 1; mask < 7; ++mask) compare_jet(public_jet(f, objective, mask), mask, expected, full);
}

void delayed_score_preserves_startup(const Fixture& f) {
    CHECK(joint(f.config));
    const auto model = model_for(f);
    const auto seed = default_seed(f, f.observations.size());
    const auto expected = oracle(f, seed, false);
    const auto restarted = oracle(f, seed, false, true);
    CHECK(std::abs(expected.value - restarted.value) > 1e-5L);
    const auto prefix_seed = oracle(f, default_seed(f, f.stop), false);
    CHECK(std::abs(expected.value - prefix_seed.value) > 1e-6L);

    // Supplied conditioning makes excluded training-tail rows irrelevant to this evaluation.
    constexpr double supplied_seed = .77;
    const auto supplied = problem_for(f, model, supplied_seed);
    verify_sample(f, supplied, supplied_seed);
    const auto objective = take(resolve_objective(supplied));
    const auto prefix_objective = take(resolve_objective(problem_for(f, model, supplied_seed, true)));
    const auto supplied_expected = oracle(f, supplied_seed, false);
    const auto full_value = public_jet(f, objective, 1);
    const auto prefix_value = public_jet(f, prefix_objective, 1);
    near(full_value.f[1], supplied_expected.value);
    near(prefix_value.f[1], supplied_expected.value);
    near(full_value.f[1], prefix_value.f[1], 2e-12L, 2e-12L);
}

void density_block_offsets(const Case& c) {
    const Fixture f(c, true);
    const auto model = model_for(f);
    const auto problem = problem_for(f, model);
    verify_sample(f, problem);
    const auto objective = take(resolve_objective(problem));
    verify_layout(f, model, objective);
    const auto owned = named_point(f, objective);
    const auto output = public_jet(f, objective, 1, &owned);
    near(output.f[1], oracle(f, default_seed(f, f.observations.size()), false).value);
}
} // namespace

int main() {
    oracle_self_check();
    std::size_t count = 0, delayed = 0;
    for (const auto& c : cases) {
        current_case = c.name;
        const Fixture fixture(c);
        all_requests_with_excess_capacity(fixture);
        if (joint(c)) { delayed_score_preserves_startup(fixture); ++delayed; }
        ++count;
    }
    // One nonempty density block for each of the five public program alternatives.
    for (const auto index : {1u, 3u, 10u, 12u, 19u}) {
        current_case = cases[index].name;
        density_block_offsets(cases[index]);
    }
    std::printf("public composition: %zu checks, %zu Normal cases/all seven masks, %zu delayed joint cases, 5 Student-t layout/value cases\n",
        checks, count, delayed);
    return EXIT_SUCCESS;
}
