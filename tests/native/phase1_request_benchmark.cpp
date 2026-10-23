// Manual standalone target. No legacy headers, installed extension, or new math.
// Fixture values come from tests/conformance/fixtures/*_cases.py.
#include <scivolapi/scivolapi.hpp>
#include "program.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {
using namespace scivolapi;
namespace program = scivolapi::detail::program;
namespace zero = scivolapi::detail::backend;
namespace mean = scivolapi::detail::mean_backend;
namespace power = scivolapi::detail::power_backend;
namespace joint = scivolapi::detail::joint_backend;
namespace joint_power = scivolapi::detail::joint_power_backend;
using Clock = std::chrono::steady_clock;

constexpr double marker = -987654321.125, seed = 0.93;
constexpr std::array<double, 12> y_values{
    .37, -.81, 1.23, -.19, .62, -1.41, .08, -.54, .93, -.67, .28, -1.09};
constexpr std::array<const char*, 8> densities{
    "normal", "studentt", "skewt", "ged", "gh", "nig", "hyperbolic", "variance_gamma"};
constexpr std::array<unsigned, 4> timed_masks{1, 2, 4, 7};
volatile double consumed = 0;

struct Failure : std::runtime_error {
    Status status;
    Failure(Status code, const std::string& message) : std::runtime_error(message), status(code) {}
};
struct BudgetExpired : std::runtime_error {
    BudgetExpired() : std::runtime_error("Panel scheduling deadline reached") {}
};
void require(bool pass, const char* message) {
    if (!pass) throw Failure(Status::internal_error, message);
}
template<class T> T take(Outcome<T> result) {
    if (!result) throw Failure(result.error().code, result.error().message);
    return std::move(result).value();
}

struct Options {
    unsigned rounds = 5, iterations = 4, seconds = 60;
    std::vector<unsigned> selected_densities{0};
    bool check_only = false, list = false, quiet_window = false, counts = false;
};
unsigned number(const std::string& text, unsigned maximum) {
    require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
            "Expected an unsigned decimal argument");
    const auto value = std::stoul(text);
    require(value <= maximum, "Argument exceeds its bounded limit");
    return static_cast<unsigned>(value);
}
Options options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string {
            require(i + 1 < argc, "Missing option argument"); return argv[++i];
        };
        if (argument == "--check-only") result.check_only = true;
        else if (argument == "--list") result.list = true;
        else if (argument == "--quiet-window") result.quiet_window = true;
        else if (argument == "--counts") result.counts = true;
        else if (argument == "--rounds") {
            result.rounds = number(next(), 9); require(result.rounds >= 3, "At least three rounds required");
        } else if (argument == "--iterations") {
            result.iterations = number(next(), 64); require(result.iterations > 0, "Zero iterations forbidden");
        } else if (argument == "--budget-seconds") {
            result.seconds = number(next(), 120); require(result.seconds >= 5, "Budget must be at least five seconds");
        } else if (argument == "--density") {
            const auto value = next();
            result.selected_densities.clear();
            if (value == "all") for (unsigned d = 0; d < 8; ++d) result.selected_densities.push_back(d);
            else result.selected_densities.push_back(number(value, 7));
        } else if (argument == "--help") {
            std::cout << "phase1_request_benchmark [--quiet-window | --check-only | --list]\n"
                         "  [--density 0..7|all] [--rounds 3..9] [--iterations 1..64]\n"
                         "  [--budget-seconds 5..120] [--counts]\n"
                         "Default: Normal theta panel, five rounds, four calls, 60-second cap.\n"
                         "--quiet-window attests chief-controlled absence of tests/builds.\n";
            std::exit(0);
        } else throw Failure(Status::invalid_argument, "Unknown option: " + argument);
    }
    require(result.list || result.check_only || result.quiet_window,
            "Timing requires --quiet-window after the chief clears tests and builds");
    return result;
}

// A blocking native call cannot defeat the watchdog. Budget exits are never passes.
class Budget {
public:
    explicit Budget(unsigned seconds) : start_(Clock::now()), seconds_(seconds),
        watchdog_([this] {
            std::unique_lock lock(mutex_);
            if (!condition_.wait_until(lock, start_ + std::chrono::seconds(seconds_), [this] { return done_; })) {
                std::fputs("Phase1 request panel: wall-time cap; incomplete, exit 124\n", stderr);
                std::_Exit(124);
            }
        }) {}
    ~Budget() {
        { std::lock_guard lock(mutex_); done_ = true; }
        condition_.notify_one(); watchdog_.join();
    }
    void check() const {
        if (elapsed() >= seconds_ - 3) throw BudgetExpired();
    }
    double elapsed() const { return std::chrono::duration<double>(Clock::now() - start_).count(); }
private:
    Clock::time_point start_;
    unsigned seconds_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool done_ = false;
    std::thread watchdog_;
};

enum class MeanKind { zero, arma, linear };
enum class VolKind { constant, garch, gjr, egarch, aparch };
struct Case {
    std::string alternative, fixture_id, registration;
    MeanKind mean_kind;
    VolKind volatility;
    std::size_t ar, ma, p, q, columns;
    bool intercept, expected_fixed;
    unsigned density;
};

std::vector<Case> panel(const Options& opt) {
    std::vector<Case> result;
    for (unsigned d : opt.selected_densities) {
        auto add = [&](const char* alternative, const std::string& id, const std::string& registration,
                       MeanKind m, VolKind v, std::size_t ar, std::size_t ma,
                       std::size_t p, std::size_t q, std::size_t columns, bool intercept, bool fixed) {
            result.push_back({alternative, id + "-" + densities[d], registration,
                              m, v, ar, ma, p, q, columns, intercept, fixed, d});
        };
        for (VolKind v : {VolKind::garch, VolKind::gjr, VolKind::egarch, VolKind::aparch}) {
            const std::string name = v == VolKind::garch ? "garch" : v == VolKind::gjr ? "gjr"
                                   : v == VolKind::egarch ? "egarch" : "aparch";
            const std::string registration = v == VolKind::gjr ? "gjr_garch" : name;
            const char* alternative = v == VolKind::garch || v == VolKind::gjr ? "zero_garch_gjr" : "zero_power";
            for (bool fixed : {true, false}) {
                const auto p = fixed ? 1u : 3u, q = fixed ? 1u : 2u;
                // Zero-mean fixtures put the density before the order tuple.
                result.push_back({alternative, name + "-" + densities[d] + (fixed ? "-11" : "-32"),
                    "zero." + registration, MeanKind::zero, v, 0, 0, p, q, 0, false, fixed, d});
            }
            const char* joint_alternative = v == VolKind::garch || v == VolKind::gjr ? "joint_garch_gjr" : "joint_power";
            const bool is_power = v == VolKind::egarch || v == VolKind::aparch;
            add(joint_alternative, "arma11-vol11-" + name, "arma." + registration,
                MeanKind::arma, v, 1, 1, 1, 1, 0, true, true);
            if (is_power)
                add(joint_alternative, "arma02-no-intercept-vol32-" + name, "arma." + registration,
                    MeanKind::arma, v, 0, 2, 3, 2, 0, false, false);
            else
                add(joint_alternative, "arma32-vol23-" + name, "arma." + registration,
                    MeanKind::arma, v, 3, 2, 2, 3, 0, true, false);
            add(joint_alternative, "linear-three-plus-intercept-vol11-" + name, "linear." + registration,
                MeanKind::linear, v, 0, 0, 1, 1, 3, true, true);
            add(joint_alternative, "linear-wide-seventeen-vol32-" + name, "linear." + registration,
                MeanKind::linear, v, 0, 0, 3, 2, 17, false, false);
        }
        add("constant_mean", "arma11", "arma.constant_variance", MeanKind::arma,
            VolKind::constant, 1, 1, 0, 0, 0, true, true);
        add("constant_mean", "arma32", "arma.constant_variance", MeanKind::arma,
            VolKind::constant, 3, 2, 0, 0, 0, true, false);
        add("constant_mean", "linear-three-plus-intercept", "linear.constant_variance", MeanKind::linear,
            VolKind::constant, 0, 0, 0, 0, 3, true, true);
        add("constant_mean", "linear-wide-seventeen", "linear.constant_variance", MeanKind::linear,
            VolKind::constant, 0, 0, 0, 0, 17, false, false);
    }
    return result;
}

struct Fixture {
    std::vector<double> y, x, theta;
    std::vector<std::uint64_t> rows;
    std::vector<std::string> names;
    std::size_t begin = 0;
    explicit Fixture(const Case& c) {
        const auto n = c.mean_kind == MeanKind::zero ? 8u : 12u;
        y.assign(y_values.begin(), y_values.begin() + n);
        rows.resize(n); std::iota(rows.begin(), rows.end(), std::uint64_t{0});
        x.resize(n * c.columns);
        for (std::size_t t = 0; t < n; ++t) for (std::size_t j = 0; j < c.columns; ++j)
            x[t * c.columns + j] = .6 * std::sin((double(t) + 1) * (double(j) + 1) * .37)
                                   + .2 * std::cos(double(j) + double(t) * .23);
        auto parameter = [&](const std::string& name, double value) { names.push_back(name); theta.push_back(value); };
        auto lag = [&](const char* name, std::size_t i, double value) {
            parameter(std::string(name) + "[" + std::to_string(i + 1) + "]", value);
        };
        if (c.intercept) parameter("intercept", .03);
        if (c.mean_kind == MeanKind::arma) {
            for (std::size_t i = 0; i < c.ar; ++i) lag("ar", i, .09 * (i % 2 ? -1 : 1) / double(i + 1));
            for (std::size_t i = 0; i < c.ma; ++i) lag("ma", i, -.13 / double(i + 1));
            begin = c.volatility == VolKind::constant ? std::max({std::size_t{1}, c.ar, c.ma})
                                                     : std::max({c.ar, c.ma, c.p, c.q});
        } else for (std::size_t j = 0; j < c.columns; ++j)
            lag("coefficient", j, .025 * std::cos(double(j) + .3) / double(j + 1));
        if (c.volatility == VolKind::constant) parameter("sigma2", 1.3);
        else {
            parameter("omega", c.volatility == VolKind::egarch ? -.11 : .11);
            for (std::size_t i = 0; i < c.p; ++i) lag("alpha", i, .085 / double(i + 1));
            if (c.volatility != VolKind::garch)
                for (std::size_t i = 0; i < c.p; ++i)
                    lag("gamma", i, (c.volatility == VolKind::gjr ? -.025 : -.16) / double(i + 1));
            for (std::size_t i = 0; i < c.q; ++i) lag("beta", i, .38 / double(i + 1));
            if (c.volatility == VolKind::aparch) parameter("delta", 1.7);
        }
        switch (c.density) {
        case 0: break;
        case 1: parameter("nu", 7.4); break;
        case 2: parameter("nu", 7.4); parameter("lambda", -.23); break;
        case 3: parameter("nu", .8); break;
        case 4: parameter("rho", .18); parameter("zeta", 2.1); parameter("lambda", .35); break;
        case 5: case 6: parameter("rho", .18); parameter("zeta", 2.1); break;
        case 7: parameter("rho", .18); parameter("lambda", 2.4); break;
        }
    }
};

Model make_model(const Case& c) {
    ModelSpec spec;
    if (c.mean_kind == MeanKind::arma) spec.mean = ARMA{c.ar, c.ma, c.intercept};
    if (c.mean_kind == MeanKind::linear) spec.mean = LinearMean{c.intercept};
    switch (c.volatility) {
    case VolKind::constant: spec.volatility = ConstantVariance{}; break;
    case VolKind::garch: spec.volatility = GARCH{c.p, c.q}; break;
    case VolKind::gjr: spec.volatility = GJRGARCH{c.p, c.q}; break;
    case VolKind::egarch: spec.volatility = EGARCH{c.p, c.q}; break;
    case VolKind::aparch: spec.volatility = APARCH{c.p, c.q}; break;
    }
    switch (c.density) {
    case 0: spec.density = Normal{}; break; case 1: spec.density = StudentT{}; break;
    case 2: spec.density = SkewT{}; break; case 3: spec.density = GED{}; break;
    case 4: spec.density = GeneralizedHyperbolic{}; break; case 5: spec.density = NIG{}; break;
    case 6: spec.density = Hyperbolic{}; break; case 7: spec.density = VarianceGamma{}; break;
    }
    return take(compile(spec, {c.columns}));
}

program::Configuration configuration(const Case& c) {
    if (c.volatility == VolKind::constant) {
        mean::ModelConfig cfg;
        cfg.kind = c.mean_kind == MeanKind::arma ? mean::Kind::arma : mean::Kind::linear;
        cfg.p_ar = c.ar; cfg.q_ma = c.ma; cfg.design_width = c.columns;
        cfg.density = c.density; cfg.intercept = c.intercept; return cfg;
    }
    const bool is_power = c.volatility == VolKind::egarch || c.volatility == VolKind::aparch;
    if (c.mean_kind == MeanKind::zero) {
        if (!is_power) return zero::ModelConfig{c.p, c.q, c.density, c.volatility == VolKind::gjr};
        return power::ModelConfig{c.volatility == VolKind::aparch ? power::Kind::aparch : power::Kind::egarch,
                                  c.p, c.q, c.density};
    }
    if (!is_power) {
        joint::ModelConfig cfg;
        cfg.kind = c.mean_kind == MeanKind::arma ? joint::Kind::arma : joint::Kind::linear;
        cfg.p_ar = c.ar; cfg.q_ma = c.ma; cfg.design_width = c.columns;
        cfg.p_vol = c.p; cfg.q_vol = c.q; cfg.density = c.density;
        cfg.leverage = c.volatility == VolKind::gjr; cfg.intercept = c.intercept; return cfg;
    }
    joint_power::ModelConfig cfg;
    cfg.mean = c.mean_kind == MeanKind::arma ? joint_power::MeanKind::arma : joint_power::MeanKind::linear;
    cfg.kind = c.volatility == VolKind::aparch ? joint_power::Kind::aparch : joint_power::Kind::egarch;
    cfg.p_ar = c.ar; cfg.q_ma = c.ma; cfg.design_width = c.columns;
    cfg.p_vol = c.p; cfg.q_vol = c.q; cfg.density = c.density;
    cfg.intercept = c.intercept; return cfg;
}

Status parameter_status(const program::Configuration& cfg, std::span<const double> theta) {
    const auto bytes = program::workspace_bytes(cfg, false, JetRequest::FGH);
    std::vector<double> arena((bytes + sizeof(double) - 1) / sizeof(double));
    auto memory = std::as_writable_bytes(std::span<double>(arena)).first(bytes);
    return std::visit([&](auto config) {
        if constexpr (std::is_same_v<decltype(config), zero::ModelConfig>)
            return zero::parameter_status(config, theta.data());
        else if constexpr (std::is_same_v<decltype(config), detail::dependence_backend::ModelConfig>)
            return detail::dependence_backend::parameter_status(config, theta);
        else return parameter_status(config, theta, memory);
    }, cfg);
}

ObjectiveView prepare_objective(const Case& c, const Fixture& f, const Model& model) {
    require(model.layout().full_theta_width == f.theta.size(), "Independent fixture width differs from model");
    for (std::size_t i = 0; i < f.names.size(); ++i)
        require(model.layout().entries[i].name == f.names[i] && model.layout().entries[i].index == i,
                "Independent fixture parameter order differs from model");
    require(model.capabilities().registration_id == c.registration, "Wrong program alternative");
    require(model.capabilities().request_mask == 7, "Missing request capability");
    require(model.capabilities().theta_dispatch == (c.expected_fixed ? DispatchKind::fixed : DispatchKind::runtime),
            "Common fixed/runtime dispatch query changed");
    DataView data; data.observations = f.y; data.row_ids = f.rows; data.row_space = 19;
    if (c.columns) data.design = {f.x, f.y.size(), c.columns};
    Conditioning conditioning; conditioning.hold_back = 0;
    if (c.volatility != VolKind::constant) conditioning.variance_seed = seed;
    ScoreSelection selection; selection.first_original_position = f.begin; selection.stop_original_position = f.y.size();
    auto problem = take(prepare(model, data, conditioning, selection));
    const auto& sample = problem.sample();
    require(sample.original_n_obs == f.y.size() && sample.retained_begin_position == 0
        && sample.retained_n_obs == f.y.size() && sample.first_eligible_original_position == f.begin
        && sample.first_scored_original_position == f.begin && sample.n_eff == f.y.size() - f.begin,
        "Declared conditioning/scored sample changed");
    require(std::equal(sample.scored_rows.begin(), sample.scored_rows.end(), f.rows.begin() + f.begin, f.rows.end()),
            "Declared scored row identity changed");
    require(std::get<UnitDescriptor>(sample.units).public_scale == 1 && std::get<UnitDescriptor>(sample.units).internal_response_scale == 1,
            "Unexpected response unit scaling");
    require(std::get<ConditioningDescriptor>(sample.conditioning).observation_history.empty() && std::get<ConditioningDescriptor>(sample.conditioning).innovation_history.empty(),
            "Unexpected conditioning history");
    require(std::get<ConditioningDescriptor>(sample.conditioning).variance_seed == conditioning.variance_seed, "Supplied seed changed");
    auto objective = take(resolve_objective(problem));
    require(objective.key().scaling == Scaling::additive_nll && objective.key().chart == 0
        && objective.key().target == DerivativeTarget::full_theta
        && objective.key().scope == ObjectiveScope::full_data_likelihood,
        "Objective is not full-theta complete additive likelihood");
    return objective;
}

struct Counts {
    bool enabled;
    std::uint64_t binds = 0, evaluations = 0, failed_binds = 0, failed_evaluations = 0;
};
PointView bind(const ObjectiveView& objective, const Fixture& f, Workspace& workspace, Counts& counts) {
    const auto result = from_theta(objective, f.theta, workspace);
    if (counts.enabled) { ++counts.binds; if (result.status != Status::ok || !result.point) ++counts.failed_binds; }
    if (result.status != Status::ok || !result.point)
        throw Failure(result.status == Status::ok ? Status::internal_error : result.status, "Public theta bind failed");
    return *result.point;
}

struct Outputs {
    double f = marker;
    std::vector<double> g, h;
    explicit Outputs(std::size_t width) : g(width, marker), h(width * width, marker) {}
    JetBuffers buffers(LayoutId layout) { return {layout, {&f, 1}, g, {h, g.size()}}; }
    void reset() { f = marker; std::fill(g.begin(), g.end(), marker); std::fill(h.begin(), h.end(), marker); }
    void untouched(unsigned mask) const {
        if (!(mask & 1)) require(f == marker, "Unrequested F was written");
        if (!(mask & 2)) for (double x : g) require(x == marker, "Unrequested G was written");
        if (!(mask & 4)) for (double x : h) require(x == marker, "Unrequested H was written");
    }
};

#if defined(_MSC_VER)
#define PHASE1_NOINLINE __declspec(noinline)
#else
#define PHASE1_NOINLINE __attribute__((noinline))
#endif
PHASE1_NOINLINE void observe(const Outputs& out, unsigned mask) {
    double sum = 0;
    auto visit = [&](double x) { require(std::isfinite(x) && x != marker, "Invalid requested output"); sum += x; };
    if (mask & 1) visit(out.f);
    if (mask & 2) for (double x : out.g) visit(x);
    if (mask & 4) for (double x : out.h) visit(x);
    require(std::isfinite(sum), "Nonfinite output checksum"); consumed = consumed + sum;
}
PHASE1_NOINLINE void observe_point(PointView point, const Fixture& f) {
    require(point.theta().size() == f.theta.size(), "Bound theta width changed");
    double sum = 0;
    for (std::size_t i = 0; i < f.theta.size(); ++i) {
        require(point.theta()[i] == f.theta[i], "Bind changed theta"); sum += point.theta()[i];
    }
    require(std::isfinite(sum), "Nonfinite point checksum"); consumed = consumed + sum;
}
JetValidity evaluate_once(const ObjectiveView& objective, PointView point, unsigned mask,
                          Workspace& workspace, Outputs& out, Counts& counts) {
    const auto result = evaluate(objective, point, static_cast<JetRequest>(mask), workspace, out.buffers(objective.key().derivative_layout));
    if (counts.enabled) {
        ++counts.evaluations;
        if (result.status != Status::ok || result.valid_mask != mask) ++counts.failed_evaluations;
    }
    if (result.status != Status::ok || result.valid_mask != mask)
        throw Failure(result.status == Status::ok ? Status::internal_error : result.status,
                      "Requested jet invalid; valid_mask=" + std::to_string(result.valid_mask));
    require(result.requested == static_cast<JetRequest>(mask), "Wrong returned request mask");
    for (auto [bit, reason] : {std::pair{1u, result.f_reason}, std::pair{2u, result.g_reason}, std::pair{4u, result.h_reason}})
        require(reason == ((mask & bit) ? Status::ok : Status::unsupported_operation), "Wrong request-specific status");
    observe(out, mask); return result;
}

struct Agreement { double f = 0, g = 0, h = 0; };
void agree(const Outputs& out, const Outputs& full, unsigned mask, Agreement& errors) {
    auto compare = [](double a, double b, double absolute, double relative, double& largest) {
        require(std::isfinite(a) && std::isfinite(b), "Nonfinite output cannot establish agreement");
        largest = std::max(largest, std::abs(a - b));
        require(std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b)), "Sparse/full projection disagrees");
    };
    // Same-point request-plan tolerances, not independent-oracle tolerances.
    if (mask & 1) compare(out.f, full.f, 1e-9, 0, errors.f);
    if (mask & 2) for (std::size_t i = 0; i < out.g.size(); ++i) compare(out.g[i], full.g[i], 2e-9, 2e-11, errors.g);
    if (mask & 4) for (std::size_t i = 0; i < out.h.size(); ++i) compare(out.h[i], full.h[i], 5e-8, 2e-11, errors.h);
    out.untouched(mask);
}

Outputs verify_point(const ObjectiveView& objective, const Fixture& f, Counts& counts,
                     const Budget& budget, Agreement& errors, PointEvidence& binding_evidence) {
    auto workspace = take(reserve(objective, JetRequest::FGH));
    const auto point = bind(objective, f, workspace, counts); observe_point(point, f);
    binding_evidence = point.evidence();
    Outputs full(f.theta.size()); evaluate_once(objective, point, 7, workspace, full, counts);
    // Each sparse request gets its own exact reservation and a fresh bound point.
    for (unsigned mask = 1; mask <= 7; ++mask) {
        budget.check();
        auto sparse_workspace = take(reserve(objective, static_cast<JetRequest>(mask)));
        const auto sparse_point = bind(objective, f, sparse_workspace, counts);
        Outputs out(f.theta.size()); evaluate_once(objective, sparse_point, mask, sparse_workspace, out, counts);
        agree(out, full, mask, errors);
    }
    return full;
}

void quoted(std::string_view value) {
    std::cout << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') std::cout << '\\' << char(c);
        else if (c < 32) std::cout << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c)
                                   << std::dec << std::setfill(' ');
        else std::cout << char(c);
    }
    std::cout << '"';
}
template<class T> void array(const std::vector<T>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        if constexpr (std::is_same_v<T, std::string>) quoted(values[i]); else std::cout << values[i];
    }
    std::cout << ']';
}
void identity(const Case& c) {
    std::cout << "\"fixture_id\":"; quoted(c.fixture_id);
    std::cout << ",\"alternative\":"; quoted(c.alternative);
    std::cout << ",\"registration\":"; quoted(c.registration);
    std::cout << ",\"density\":"; quoted(densities[c.density]);
    std::cout << ",\"coordinate\":\"theta\",\"p_ar\":" << c.ar << ",\"q_ma\":" << c.ma
              << ",\"p_vol\":" << c.p << ",\"q_vol\":" << c.q << ",\"external_design_width\":" << c.columns
              << ",\"intercept\":" << (c.intercept ? "true" : "false");
}
void report_quality(const Case& c, const Fixture& f, const Outputs& full, const Agreement& errors,
                    const PointEvidence& binding_evidence) {
    std::cout << "{\"kind\":\"point_quality\","; identity(c);
    std::cout << ",\"status\":0,\"valid_mask\":7,\"model_owned_parameter_status\":0"
                 ",\"numerical_outputs\":\"finite_all_requested\",\"sparse_full_agreement\":\"passed_all_seven_masks\""
                 ",\"independent_math_qualification\":\"unavailable\",\"comparative_claim\":false"
                 ",\"differentiability_certificate\":\"unavailable\",\"public_binding_evidence\":{\"feasibility\":"
              << unsigned(binding_evidence.theoretical_feasibility) << ",\"numerical_support\":"
              << unsigned(binding_evidence.numerical_support) << ",\"differentiability\":"
              << unsigned(binding_evidence.differentiability) << ",\"valid_mask\":" << binding_evidence.valid_mask
              << "},\"attained_absolute_error_bound\":";
    if (binding_evidence.attained_absolute_error_bound) std::cout << *binding_evidence.attained_absolute_error_bound;
    else std::cout << "null";
    std::cout << ",\"dispatch_query\":";
    quoted(c.expected_fixed ? "fixed" : "runtime");
    std::cout << ",\"actual_dispatch_binary_evidence\":\"unavailable\",\"normalization\":\"additive_nll\""
                 ",\"scope\":\"full_data_likelihood\",\"hold_back\":0,\"units\":\"input\",\"variance_seed\":";
    if (c.volatility == VolKind::constant) std::cout << "null"; else std::cout << seed;
    std::cout << ",\"recursion_begin\":" << f.begin << ",\"first_scored\":" << f.begin
              << ",\"stop\":" << f.y.size() << ",\"n_eff\":" << f.y.size() - f.begin
              << ",\"parameter_names\":"; array(f.names);
    std::cout << ",\"theta\":"; array(f.theta); std::cout << ",\"y\":"; array(f.y);
    std::cout << ",\"design_row_major\":"; array(f.x); std::cout << ",\"row_ids\":"; array(f.rows);
    std::cout << ",\"F\":" << full.f << ",\"G\":"; array(full.g); std::cout << ",\"H_row_major\":"; array(full.h);
    std::cout << ",\"projection_max_absolute_error\":{\"F\":" << errors.f << ",\"G\":" << errors.g << ",\"H\":" << errors.h << "}}\n";
    std::cout.flush();
}

enum class Mode : unsigned { bind, evaluate, bind_evaluate };
constexpr std::array<const char*, 3> mode_names{"checked_bind", "bound_evaluate", "bind_plus_evaluate"};
void measure_request(const Case& c, const Fixture& f, const ObjectiveView& objective, const Outputs& full,
                     unsigned mask, const Options& opt, Counts& counts, const Budget& budget) {
    budget.check();
    const auto request = static_cast<JetRequest>(mask);
    const auto bytes = take(requirements(objective, request)).bytes;
    auto workspace = take(reserve(objective, request));
    auto point = bind(objective, f, workspace, counts);
    Outputs out(f.theta.size()); Agreement errors;
    std::array<std::vector<double>, 3> measurements;
    auto call = [&](Mode mode) {
        if (mode != Mode::evaluate) point = bind(objective, f, workspace, counts);
        if (mode == Mode::bind) observe_point(point, f);
        else evaluate_once(objective, point, mask, workspace, out, counts);
    };
    // Warm every mode outside timing. Alternate direction and rotate first mode.
    for (Mode mode : {Mode::bind, Mode::evaluate, Mode::bind_evaluate}) {
        budget.check(); out.reset(); call(mode);
        if (mode != Mode::bind) agree(out, full, mask, errors);
    }
    for (unsigned round = 0; round < opt.rounds; ++round) {
        for (unsigned slot = 0; slot < 3; ++slot) {
            budget.check();
            const auto index = (round + (round % 2 ? 2 - slot : slot)) % 3;
            const auto mode = static_cast<Mode>(index);
            out.reset();
            const auto start = Clock::now();
            for (unsigned i = 0; i < opt.iterations; ++i) call(mode);
            const auto stop = Clock::now();
            budget.check();
            if (mode != Mode::bind) agree(out, full, mask, errors);
            measurements[index].push_back(std::chrono::duration<double, std::nano>(stop - start).count() / opt.iterations);
        }
    }
    std::cout << "{\"kind\":\"request_cost\","; identity(c);
    std::cout << ",\"status\":0,\"valid_mask\":" << mask << ",\"request_mask\":" << mask
              << ",\"workspace_bytes\":" << bytes << ",\"iterations_per_batch\":" << opt.iterations
              << ",\"rounds\":" << opt.rounds << ",\"comparison\":\"standalone_absolute_no_comparative_claim\""
                 ",\"includes_checked_observer\":true,\"instrumented\":" << (opt.counts ? "true" : "false")
              << ",\"internal_work_counts\":null,\"internal_counts_status\":\"unavailable\",\"costs\":{";
    for (unsigned i = 0; i < 3; ++i) {
        if (i) std::cout << ','; quoted(mode_names[i]); std::cout << ":{\"raw_ns_per_call\":"; array(measurements[i]);
        auto ordered = measurements[i]; std::sort(ordered.begin(), ordered.end());
        const auto middle = ordered.size() / 2;
        const double median = ordered.size() % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
        std::cout << ",\"median_ns_per_call\":" << median << '}';
    }
    std::cout << "}}\n"; std::cout.flush();
}

int run(const Options& opt) {
    Budget budget(opt.seconds);
    const auto cases = panel(opt);
    Counts counts{opt.counts};
    std::size_t passed = 0, failed = 0, timed_requests = 0, attempted = 0;
    bool exhausted = false;
    std::cout << std::setprecision(17);
    std::cout << "{\"kind\":\"panel\",\"planned_cases\":" << cases.size()
              << ",\"budget_seconds\":" << opt.seconds << ",\"hard_max_seconds\":120"
                 ",\"normalization\":\"additive_nll\",\"coordinates\":\"theta_only\""
                 ",\"qualification\":\"prototype_incomplete\",\"baseline\":\"not_linked\",\"comparative_claim\":false,\"mode\":";
    quoted(opt.list ? "list" : opt.check_only ? "check_only" : "timing"); std::cout << "}\n";
    for (const auto& c : cases) {
        try {
            budget.check(); ++attempted;
            if (opt.list) { std::cout << "{\"kind\":\"fixture\","; identity(c); std::cout << "}\n"; ++passed; continue; }
            const Fixture fixture(c);
            const auto model = make_model(c);
            const auto cfg = configuration(c);
            require(program::parameter_count(cfg) == fixture.theta.size(), "Program layout width mismatch");
            require(program::dispatch_kind(cfg, false) == int(c.expected_fixed), "Program dispatch query differs");
            const auto domain = parameter_status(cfg, fixture.theta);
            if (domain != Status::ok) throw Failure(domain, "Model-owned theta domain rejected fixture");
            const auto objective = prepare_objective(c, fixture, model);
            Agreement errors;
            PointEvidence binding_evidence{};
            const auto full = verify_point(objective, fixture, counts, budget, errors, binding_evidence);
            report_quality(c, fixture, full, errors, binding_evidence);
            if (!opt.check_only) for (unsigned mask : timed_masks) {
                measure_request(c, fixture, objective, full, mask, opt, counts, budget); ++timed_requests;
            }
            ++passed;
        } catch (const BudgetExpired&) { exhausted = true; break; }
        catch (const Failure& error) {
            ++failed;
            std::cout << "{\"kind\":\"case_failure\","; identity(c);
            std::cout << ",\"status\":" << unsigned(error.status) << ",\"message\":"; quoted(error.what());
            std::cout << ",\"comparative_claim\":false}\n"; std::cout.flush();
        }
    }
    std::cout << "{\"kind\":\"summary\",\"status\":";
    quoted(exhausted ? "budget_incomplete" : failed ? "failed" : opt.list ? "listed_not_executed"
           : opt.check_only ? "checks_complete_not_qualified" : "standalone_complete_not_qualified");
    std::cout << ",\"planned_cases\":" << cases.size() << ",\"attempted_cases\":" << attempted
              << ",\"passed_cases\":" << passed << ",\"failed_cases\":" << failed
              << ",\"not_completed_cases\":" << cases.size() - passed - failed
              << ",\"timed_request_rows\":" << timed_requests << ",\"elapsed_seconds\":" << budget.elapsed()
              << ",\"internal_work_counts\":null,\"internal_counts_status\":\"unavailable\",\"observed_public_calls\":";
    if (!opt.counts) std::cout << "null";
    else std::cout << "{\"bind\":" << counts.binds << ",\"evaluate\":" << counts.evaluations
                   << ",\"failed_bind\":" << counts.failed_binds << ",\"failed_evaluate\":" << counts.failed_evaluations << '}';
    std::cout << ",\"counts_scope\":\"optional_external_calls_including_validation_warmup_timing\""
                 ",\"comparative_claim\":false,\"independent_math_qualification\":\"unavailable\"}\n";
    return exhausted ? 124 : failed ? 1 : 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return run(options(argc, argv)); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "Phase1 request benchmark: %s\n", error.what()); return 2;
    }
}
