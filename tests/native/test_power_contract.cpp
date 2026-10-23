#include <scivolapi/scivolapi.hpp>
#include "power_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <source_location>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace scivolapi;
namespace backend = scivolapi::detail::power_backend;
std::size_t checks = 0;
constexpr double sentinel = -0x1.23456789abcp+20;
constexpr std::array<double, 12> observations{.3, -.7, 1.1, -.2, .5, -1.3, .9, .4, -.6, .8, -.4, 1.2};
constexpr std::array<std::uint64_t, 12> rows{10, 12, 17, 23, 24, 31, 40, 55, 61, 70, 81, 93};
struct Orders { std::size_t p, q; };
constexpr std::array<Orders, 4> orders{{{1, 1}, {3, 2}, {2, 3}, {4, 3}}};

void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        std::fprintf(stderr, "power contract line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "power setup status %u: %s\n", unsigned(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}
template<class T> void rejected(const Outcome<T>& result, Status status) {
    CHECK(!result);
    if (result.error().code != status)
        std::fprintf(stderr, "expected status %u, got %u: %s\n", unsigned(status),
            unsigned(result.error().code), result.error().message.c_str());
    CHECK(result.error().code == status && !result.error().message.empty());
}
bool close(double a, double b, double absolute = 2e-10, double relative = 2e-11) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
}
void set_density(ModelSpec& spec, unsigned density) {
    switch (density) {
    case 0: spec.density = Normal{}; break;
    case 1: spec.density = StudentT{}; break;
    case 2: spec.density = SkewT{}; break;
    case 3: spec.density = GED{}; break;
    case 4: spec.density = GeneralizedHyperbolic{}; break;
    case 5: spec.density = NIG{}; break;
    case 6: spec.density = Hyperbolic{}; break;
    case 7: spec.density = VarianceGamma{}; break;
    default: CHECK(false);
    }
}
Model make_model(bool aparch, Orders order, unsigned density = 0) {
    ModelSpec spec;
    spec.mean = ZeroMean{};
    if (aparch) spec.volatility = APARCH{order.p, order.q};
    else spec.volatility = EGARCH{order.p, order.q};
    set_density(spec, density);
    return take(compile(spec));
}
DataView data_view(std::span<const double> data = observations) {
    DataView view;
    view.observations = data;
    view.row_ids = std::span<const std::uint64_t>(rows).first(data.size());
    view.row_space = 73;
    view.units.response = "returns";
    return view;
}
PreparedProblem make_problem(const Model& model, std::size_t first = 2,
    std::size_t stop = observations.size(), std::size_t hold_back = 0) {
    Conditioning conditioning;
    conditioning.variance_seed = .8;
    conditioning.hold_back = hold_back;
    ScoreSelection selection;
    selection.first_original_position = first;
    selection.stop_original_position = stop;
    return take(prepare(model, data_view(), conditioning, selection));
}
ObjectiveView objective_for(const PreparedProblem& problem, bool chart = false) {
    ObjectiveRequest request;
    request.coordinates = chart ? Coordinates::original_chart : Coordinates::theta;
    return take(resolve_objective(problem, request));
}
std::vector<double> parameters(const Model& model, bool aparch, Orders order,
    unsigned density, bool chart = false) {
    std::vector<double> result(model.layout().full_theta_width, 0.0);
    const auto shape = 1 + 2 * order.p + order.q + std::size_t(aparch);
    result[0] = chart ? -.8 : aparch ? .18 : -.07;
    for (std::size_t i = 0; i < order.p; ++i) {
        result[1 + i] = chart && aparch ? -2.8 : (aparch ? .035 : .09) / order.p;
        result[1 + order.p + i] = (aparch ? .12 : -.04) / order.p;
    }
    for (std::size_t i = 0; i < order.q; ++i)
        result[1 + 2 * order.p + i] = chart ? (aparch ? -2.6 : .25) : .35 / order.q;
    if (aparch) result[shape - 1] = chart ? .8 : 1.4;
    if (chart) {
        std::fill(result.begin() + shape, result.end(), .3);
        return result;
    }
    switch (density) {
    case 0: break;
    case 1: result[shape] = 8; break;
    case 2: result[shape] = 8; result[shape + 1] = .2; break;
    case 3: result[shape] = 1.7; break;
    case 4: result[shape] = .15; result[shape + 1] = 1.4; result[shape + 2] = .8; break;
    case 5:
    case 6: result[shape] = .15; result[shape + 1] = 1.4; break;
    case 7: result[shape] = .15; result[shape + 1] = 3; break;
    }
    return result;
}
PointBinding bind_point(const ObjectiveView& objective, std::span<const double> point,
    Workspace& workspace, bool chart = false) {
    return chart ? from_coordinates(objective, point, workspace) : from_theta(objective, point, workspace);
}
struct Outputs {
    std::size_t width;
    LayoutId layout;
    std::vector<double> f, g, h;
    explicit Outputs(const ObjectiveView& objective)
        : width(objective.derivative_layout().axes.size()), layout(objective.key().derivative_layout),
          f(3, sentinel), g(width + 2, sentinel), h(width * width + 2, sentinel) {}
    JetBuffers buffers(bool extra = false) {
        return {layout, std::span<double>(f).subspan(1, 1 + extra),
            std::span<double>(g).subspan(1, width + extra),
            {std::span<double>(h).subspan(1, width * width + extra), width}};
    }
    void reset() { for (auto* block : {&f, &g, &h}) std::fill(block->begin(), block->end(), sentinel); }
    void unchanged() const {
        for (const auto* block : {&f, &g, &h}) for (double value : *block) CHECK(value == sentinel);
    }
    void untouched(unsigned mask) const {
        for (const auto* block : {&f, &g, &h}) CHECK(block->front() == sentinel && block->back() == sentinel);
        if (!(mask & 1)) for (double value : f) CHECK(value == sentinel);
        if (!(mask & 2)) for (double value : g) CHECK(value == sentinel);
        if (!(mask & 4)) for (double value : h) CHECK(value == sentinel);
    }
};
void valid(const JetValidity& result, unsigned mask,
           std::source_location caller = std::source_location::current()) {
    if (result.status != Status::ok || result.valid_mask != mask)
        std::fprintf(stderr, "jet check at %u: request=%u valid=%u status=%u reasons=%u,%u,%u\n",
            caller.line(),mask,result.valid_mask,unsigned(result.status),
            unsigned(result.f_reason),unsigned(result.g_reason),unsigned(result.h_reason));
    CHECK(result.status == Status::ok && result.valid_mask == mask);
    CHECK(result.requested == static_cast<JetRequest>(mask) && result.contains(result.requested));
    CHECK(result.f_reason == ((mask & 1) ? Status::ok : Status::unsupported_operation));
    CHECK(result.g_reason == ((mask & 2) ? Status::ok : Status::unsupported_operation));
    CHECK(result.h_reason == ((mask & 4) ? Status::ok : Status::unsupported_operation));
}
void projection(const Outputs& selected, const Outputs& full, unsigned mask) {
    if (mask & 1) CHECK(close(selected.f[1], full.f[1], 5e-11, 2e-12));
    if (mask & 2) for (std::size_t i = 1; i <= full.width; ++i) CHECK(close(selected.g[i], full.g[i]));
    if (mask & 4) for (std::size_t i = 1; i <= full.width * full.width; ++i)
        CHECK(close(selected.h[i], full.h[i], 2e-9, 2e-10));
}
Outputs jet(const ObjectiveView& objective, std::span<const double> theta, JetRequest request = JetRequest::FGH) {
    auto workspace = take(reserve(objective, request));
    const auto point = from_theta(objective, theta, workspace);
    CHECK(point.status == Status::ok && point.point);
    Outputs output(objective);
    valid(evaluate(objective, *point.point, request, workspace, output.buffers()), unsigned(request));
    output.untouched(unsigned(request));
    return output;
}
struct Arena {
    std::size_t bytes, alignment;
    std::byte* storage;
    explicit Arena(std::size_t size, std::size_t align)
        : bytes(size), alignment(align), storage(static_cast<std::byte*>(
            ::operator new(bytes + 2 * alignment, std::align_val_t{alignment}))) {
        CHECK(bytes && alignment >= alignof(double) && !(alignment & (alignment - 1)));
        std::fill_n(storage, bytes + 2 * alignment, std::byte{0xa5});
    }
    Arena(const Arena&) = delete;
    ~Arena() { ::operator delete(storage, std::align_val_t{alignment}); }
    std::span<std::byte> memory() { return {storage + alignment, bytes}; }
    void guards() const {
        for (std::size_t i = 0; i < alignment; ++i)
            CHECK(storage[i] == std::byte{0xa5} && storage[alignment + bytes + i] == std::byte{0xa5});
    }
};

void layouts_and_admission() {
    const std::array<std::vector<std::string>, 8> shape_names{{{}, {"nu"}, {"nu", "lambda"},
        {"nu"}, {"rho", "zeta", "lambda"}, {"rho", "zeta"}, {"rho", "zeta"}, {"rho", "lambda"}}};
    for (bool aparch : {false, true}) for (auto order : orders) for (unsigned density = 0; density < 8; ++density) {
        const auto model = make_model(aparch, order, density);
        std::vector<std::string> names{"omega"};
        for (auto prefix : {"alpha", "gamma"}) for (std::size_t i = 1; i <= order.p; ++i)
            names.push_back(std::string(prefix) + "[" + std::to_string(i) + "]");
        for (std::size_t i = 1; i <= order.q; ++i) names.push_back("beta[" + std::to_string(i) + "]");
        if (aparch) names.emplace_back("delta");
        const auto volatility_width = names.size();
        names.insert(names.end(), shape_names[density].begin(), shape_names[density].end());
        CHECK(model.layout().full_theta_width == names.size() && model.layout().entries.size() == names.size());
        CHECK(model.layout().blocks.size() == 2);
        CHECK(model.layout().blocks[0].name == "volatility" && model.layout().blocks[0].offset == 0
            && model.layout().blocks[0].width == volatility_width);
        CHECK(model.layout().blocks[1].name == "density" && model.layout().blocks[1].offset == volatility_width
            && model.layout().blocks[1].width == shape_names[density].size());
        const auto dispatch = order.p == 1 && order.q == 1 ? DispatchKind::fixed : DispatchKind::runtime;
        CHECK(model.capabilities().theta_dispatch == dispatch && model.capabilities().chart_dispatch == dispatch);
        CHECK(model.capabilities().request_mask == 7 && model.capabilities().original_chart);
        CHECK(model.capabilities().registration_id == (aparch ? "zero.aparch" : "zero.egarch"));
        const auto objective = objective_for(make_problem(model));
        CHECK(objective.key().scaling == Scaling::additive_nll && objective.key().chart == 0);
        CHECK(objective.derivative_layout().target == DerivativeTarget::full_theta);
        CHECK(objective.derivative_layout().axes.size() == names.size());
        CHECK(objective.derivative_layout().full_parameter_partition.size() == names.size());
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto& entry = model.layout().entries[i];
            CHECK(entry.name == names[i] && entry.index == i && entry.id != 0);
            CHECK(objective.derivative_layout().axes[i] == entry.id);
            CHECK(objective.derivative_layout().full_parameter_partition[i] == ParameterDisposition::free);
            for (std::size_t j = 0; j < i; ++j) CHECK(entry.id != model.layout().entries[j].id);
        }
    }
    for (bool aparch : {false, true}) {
        ModelSpec spec;
        if (aparch) spec.volatility = APARCH{1, 1}; else spec.volatility = EGARCH{1, 1};
        CHECK(std::get<ModelSpec>(take(compile(spec)).spec()).density_origin == DensityOrigin::implicit_normal);
        spec.mean = ARMA{1, 1};
        CHECK(bool(compile(spec)));
        spec.mean = ZeroMean{};
        rejected(compile(spec, {1}), Status::invalid_argument);
        for (auto order : {Orders{0, 1}, Orders{1, 0}, Orders{std::numeric_limits<std::size_t>::max(), 1}}) {
            if (aparch) spec.volatility = APARCH{order.p, order.q}; else spec.volatility = EGARCH{order.p, order.q};
            rejected(compile(spec), Status::invalid_argument);
        }
    }
}

// Independent scalar equations and normalization, not a derivative oracle.
double normal_nll(bool aparch, Orders order, std::span<const double> y,
    std::span<const double> theta, long double seed, std::size_t first, std::size_t stop) {
    std::vector<long double> state(stop), standardized(stop);
    const long double pi = std::acos(-1.0L), moment = std::sqrt(2 / pi);
    const auto delta = aparch ? theta[1 + 2 * order.p + order.q] : 0.0;
    state[0] = aparch ? std::pow(seed, delta / 2.0L) : std::log(seed);
    long double total = 0;
    for (std::size_t t = 0; t < stop; ++t) {
        if (t) {
            state[t] = theta[0];
            for (std::size_t lag = 1; lag <= order.p && lag <= t; ++lag) {
                if (aparch) state[t] += theta[lag] * std::pow(std::abs(static_cast<long double>(y[t-lag]))
                    - theta[order.p + lag] * y[t-lag], delta);
                else state[t] += theta[lag] * (std::abs(standardized[t-lag]) - moment)
                    + theta[order.p + lag] * standardized[t-lag];
            }
            for (std::size_t lag = 1; lag <= order.q && lag <= t; ++lag)
                state[t] += theta[1 + 2 * order.p + lag - 1] * state[t-lag];
        }
        const long double log_h = t == 0 ? std::log(seed) : aparch ? 2 * std::log(state[t]) / delta : state[t];
        standardized[t] = y[t] * std::exp(-log_h / 2);
        if (t >= first) total += .5L * (std::log(2 * pi) + log_h + standardized[t] * standardized[t]);
    }
    return static_cast<double>(total);
}
double seed_nll(unsigned density, std::span<const double> shape, double y, double seed) {
    const double pi = std::acos(-1.0), z = y / std::sqrt(seed);
    if (density == 0) return .5 * (std::log(2*pi) + std::log(seed) + z*z);
    const double nu = shape[0];
    if (density == 3) {
        const double log_b = .5 * (std::lgamma(1/nu) - std::lgamma(3/nu));
        return .5*std::log(seed) + std::log(2/nu) + log_b + std::lgamma(1/nu)
            + std::exp(nu * (std::log(std::abs(z)) - log_b));
    }
    const double log_c = std::lgamma((nu+1)/2) - std::lgamma(nu/2) - .5*std::log(pi*(nu-2));
    if (density == 1) return .5*std::log(seed) - log_c + .5*(nu+1)*std::log1p(z*z/(nu-2));
    const double lambda = shape[1], a = 4*lambda*std::exp(log_c)*(nu-2)/(nu-1);
    const double b = std::sqrt(1+3*lambda*lambda-a*a);
    const double side = z < -a/b ? 1-lambda : 1+lambda;
    const double shifted = (b*z+a)/side;
    return .5*std::log(seed) - std::log(b) - log_c + .5*(nu+1)*std::log1p(shifted*shifted/(nu-2));
}
void conditioning_and_independent_values() {
    for (bool aparch : {false, true}) for (auto order : orders) {
        const auto model = make_model(aparch, order);
        const auto theta = parameters(model, aparch, order, 0);
        const auto selected = make_problem(model, 5, 11, 1);
        CHECK(selected.sample().retained_begin_position == 1 && selected.sample().n_eff == 6);
        CHECK(selected.sample().first_scored_original_position == 5 && std::get<ConditioningDescriptor>(selected.sample().conditioning).variance_seed == .8);
        const double value = jet(objective_for(selected), theta, JetRequest::F).f[1];
        CHECK(close(value, normal_nll(aparch, order, std::span<const double>(observations).subspan(1), theta, .8L, 4, 10)));
        const double reset = normal_nll(aparch, order, std::span<const double>(observations).subspan(5), theta, .8L, 0, 6);
        CHECK(std::abs(value-reset) > 1e-3);
        Conditioning conditioning;
        conditioning.hold_back = 1;
        const auto defaults = take(prepare(model, data_view(), conditioning));
        long double expected = 0;
        for (std::size_t t = 1; t < observations.size(); ++t)
            expected += static_cast<long double>(observations[t])*observations[t]/(observations.size()-1);
        CHECK(std::get<ConditioningDescriptor>(defaults.sample().conditioning).seed_origin == SeedOrigin::model_default);
        CHECK(close(*std::get<ConditioningDescriptor>(defaults.sample().conditioning).variance_seed, double(expected), 2e-15, 2e-15));
        ScoreSelection score{5, 9};
        const auto selected_default = take(prepare(model, data_view(), conditioning, score));
        CHECK(std::get<ConditioningDescriptor>(selected_default.sample().conditioning).variance_seed == std::get<ConditioningDescriptor>(defaults.sample().conditioning).variance_seed);
        CHECK(std::get<ConditioningDescriptor>(selected_default.sample().conditioning).source_rows.size() == observations.size()-1);
        for (std::size_t t = 0; t < observations.size()-1; ++t)
            CHECK(std::get<ConditioningDescriptor>(selected_default.sample().conditioning).source_rows[t] == rows[t+1]);
        conditioning.variance_seed = .8;
        const auto supplied = take(prepare(model, data_view(), conditioning, score));
        CHECK(std::get<ConditioningDescriptor>(supplied.sample().conditioning).seed_origin == SeedOrigin::supplied
            && std::get<ConditioningDescriptor>(supplied.sample().conditioning).variance_seed == .8);
        conditioning.observation_history = std::span<const double>(observations).first(1);
        rejected(prepare(model, data_view(), conditioning), Status::unsupported_operation);
        conditioning.observation_history = {};
        conditioning.innovation_history = std::span<const double>(observations).first(1);
        if (order.p==1) CHECK(prepare(model, data_view(), conditioning));
        else rejected(prepare(model, data_view(), conditioning), Status::invalid_argument);
        std::array<double, observations.size()> zeros{};
        rejected(prepare(model, data_view(zeros)), Status::numerical_range);
        conditioning.innovation_history = {};
        CHECK(prepare(model, data_view(zeros), conditioning));
        for (double bad : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            conditioning.variance_seed = bad;
            rejected(prepare(model, data_view(), conditioning), Status::invalid_argument);
        }
    }
    for (auto order : orders) for (unsigned density : {0u, 1u, 2u, 3u}) {
        const auto model = make_model(false, order, density);
        auto theta = parameters(model, false, order, density);
        const auto objective = objective_for(make_problem(model, 0, 1));
        const double expected = seed_nll(density, std::span<const double>(theta).subspan(1+2*order.p+order.q), observations[0], .8);
        CHECK(close(jet(objective, theta, JetRequest::F).f[1], expected, 5e-12, 2e-12));
        theta[0] = 3.0;
        theta[1] = -2.0;
        CHECK(close(jet(objective, theta, JetRequest::F).f[1], expected, 5e-12, 2e-12));
    }
}

void every_mask_and_coordinate() {
    for (bool aparch : {false, true}) for (auto order : orders)
        for (unsigned density = 0; density < 8; ++density) for (bool chart : {false, true}) {
            const auto model = make_model(aparch, order, density);
            const auto objective = objective_for(make_problem(model), chart);
            auto source = parameters(model, aparch, order, density, chart);
            auto workspace = take(reserve(objective, JetRequest::FGH));
            auto point = bind_point(objective, source, workspace, chart);
            CHECK(point.status == Status::ok && point.point);
            const auto original = source;
            if (chart) {
                CHECK(point.point->original_coordinates().size() == source.size());
                CHECK(std::equal(source.begin(), source.end(), point.point->original_coordinates().begin()));
            }
            Outputs full(objective);
            valid(evaluate(objective, *point.point, JetRequest::FGH, workspace, full.buffers()), 7);
            full.untouched(7);
            for (unsigned mask = 1; mask <= 7; ++mask) {
                const auto request = static_cast<JetRequest>(mask);
                auto exact = take(reserve(objective, request));
                const auto bound = bind_point(objective, source, exact, chart);
                CHECK(bound.status == Status::ok && bound.point);
                Outputs sparse(objective);
                valid(evaluate(objective, *bound.point, request, exact, sparse.buffers()), mask);
                sparse.untouched(mask);
                projection(sparse, full, mask);
                for (std::size_t r = 0; (mask & 4) && r < full.width; ++r)
                    for (std::size_t c = 0; c < full.width; ++c)
                        CHECK(close(sparse.h[1+r*full.width+c], sparse.h[1+c*full.width+r]));
                auto storage = take(allocate_jet_storage(objective, request));
                CHECK(storage.gradient().size() == ((mask & 2) ? full.width : 0));
                CHECK(storage.hessian().size() == ((mask & 4) ? full.width*full.width : 0));
                valid(evaluate(objective, *bound.point, request, exact, storage.outputs()), mask);
                if (mask & 1) CHECK(close(storage.value(), full.f[1])); else CHECK(std::isnan(storage.value()));
                if (mask & 2) for (std::size_t i = 0; i < full.width; ++i) CHECK(close(storage.gradient()[i], full.g[1+i]));
                if (mask & 4) for (std::size_t i = 0; i < full.width*full.width; ++i)
                    CHECK(close(storage.hessian()[i], full.h[1+i], 2e-9, 2e-10));
            }
            CHECK(source == original);
            const auto stale = *point.point;
            source[0] += .01;
            point = bind_point(objective, source, workspace, chart);
            CHECK(point.status == Status::ok && point.point);
            Outputs untouched(objective);
            const auto expired = evaluate(objective, stale, JetRequest::FGH, workspace, untouched.buffers());
            CHECK(expired.status == Status::invalid_argument && expired.valid_mask == 0);
            untouched.unchanged();
        }
}

void fixed_runtime_same_tuple() {
    for (bool aparch : {false, true}) for (unsigned density = 0; density < 8; ++density)
        for (bool chart : {false, true}) {
            const auto model = make_model(aparch, {1, 1}, density);
            const auto source = parameters(model, aparch, {1, 1}, density, chart);
            backend::ModelConfig fixed;
            CHECK(backend::configure(std::get<ModelSpec>(model.spec()), {}, fixed) == Status::ok);
            auto runtime = fixed;
            runtime.fixed = false;
            CHECK(backend::dispatch_kind(fixed) == 1 && backend::dispatch_kind(runtime) == 0);
            backend::SampleView sample{observations, 2, .8, {}, {}, 0};
            CHECK(backend::sample_status(fixed, sample) == Status::ok);
            const auto coordinates = chart ? Coordinates::original_chart : Coordinates::theta;
            for (unsigned mask = 1; mask <= 7; ++mask) {
                const auto request = static_cast<JetRequest>(mask);
                const auto fp = backend::requirements(fixed, coordinates, request);
                const auto rp = backend::requirements(runtime, coordinates, request);
                CHECK(fp.status == Status::ok && rp.status == Status::ok && fp.parameters == source.size());
                Arena fa(fp.bytes, fp.alignment), ra(rp.bytes, rp.alignment);
                double fv = sentinel, rv = sentinel;
                std::vector<double> fg(source.size(), sentinel), rg(fg), fh(source.size()*source.size(), sentinel), rh(fh);
                valid(backend::evaluate(fixed, sample, source, coordinates, request, fa.memory(), {{&fv, 1}, fg, fh}), mask);
                valid(backend::evaluate(runtime, sample, source, coordinates, request, ra.memory(), {{&rv, 1}, rg, rh}), mask);
                if (mask & 1) CHECK(close(fv, rv)); else CHECK(fv == sentinel && rv == sentinel);
                for (std::size_t i = 0; i < fg.size(); ++i)
                    if (mask & 2) CHECK(close(fg[i], rg[i])); else CHECK(fg[i] == sentinel && rg[i] == sentinel);
                for (std::size_t i = 0; i < fh.size(); ++i)
                    if (mask & 4) CHECK(close(fh[i], rh[i], 2e-9, 2e-10)); else CHECK(fh[i] == sentinel && rh[i] == sentinel);
                fa.guards(); ra.guards();
            }
        }
}

void aparch_garch_reduction() {
    for (auto order : orders) for (unsigned density = 0; density < 8; ++density) {
        const auto power = make_model(true, order, density);
        auto theta = parameters(power, true, order, density);
        std::fill(theta.begin()+1+order.p, theta.begin()+1+2*order.p, 0.0);
        theta[1+2*order.p+order.q] = 2.0;
        ModelSpec spec;
        spec.volatility = GARCH{order.p, order.q};
        set_density(spec, density);
        const auto garch = take(compile(spec));
        std::vector<std::size_t> common;
        for (std::size_t i = 0; i <= order.p; ++i) common.push_back(i);
        for (std::size_t i = 0; i < order.q; ++i) common.push_back(1+2*order.p+i);
        for (std::size_t i = 2+2*order.p+order.q; i < theta.size(); ++i) common.push_back(i);
        std::vector<double> reduced;
        for (auto index : common) reduced.push_back(theta[index]);
        const auto a = jet(objective_for(make_problem(power)), theta);
        const auto g = jet(objective_for(make_problem(garch)), reduced);
        CHECK(close(a.f[1], g.f[1], 5e-11, 2e-12));
        for (std::size_t i = 0; i < common.size(); ++i) {
            CHECK(close(a.g[1+common[i]], g.g[1+i]));
            for (std::size_t j = 0; j < common.size(); ++j)
                CHECK(close(a.h[1+common[i]*a.width+common[j]], g.h[1+i*g.width+j], 2e-9, 2e-10));
        }
    }
}

void workspace_and_buffer_admission() {
    for (bool aparch : {false, true}) for (auto order : orders) for (bool chart : {false, true}) {
        const auto model = make_model(aparch, order);
        const auto objective = objective_for(make_problem(model), chart);
        const auto source = parameters(model, aparch, order, 0, chart);
        const auto plan = take(requirements(objective, JetRequest::FGH));
        Arena arena(plan.bytes, plan.alignment);
        rejected(bind_workspace(plan, arena.memory().first(plan.bytes-1)), Status::insufficient_capacity);
        rejected(bind_workspace(plan, {}), Status::insufficient_capacity);
        rejected(bind_workspace(plan, {arena.memory().data()+1, std::size_t(plan.bytes)}), Status::invalid_argument);
        auto forged = plan;
        ++forged.work_plan_id;
        rejected(bind_workspace(forged, arena.memory()), Status::invalid_argument);
        auto workspace = take(bind_workspace(plan, arena.memory()));
        const auto point = bind_point(objective, source, workspace, chart);
        CHECK(point.status == Status::ok && point.point);
        Outputs output(objective);
        const auto reject = [&](JetBuffers buffers, JetRequest request, Status expected) {
            const auto result = evaluate(objective, *point.point, request, workspace, buffers);
            CHECK(result.status == expected && result.valid_mask == 0);
            output.unchanged();
        };
        auto buffers = output.buffers(); ++buffers.derivative_layout;
        reject(buffers, JetRequest::FGH, Status::invalid_argument);
        buffers = output.buffers(); buffers.value = {};
        reject(buffers, JetRequest::F, Status::insufficient_capacity);
        buffers = output.buffers(); buffers.gradient = buffers.gradient.first(source.size()-1);
        reject(buffers, JetRequest::G, Status::insufficient_capacity);
        buffers = output.buffers(); buffers.hessian.elements = buffers.hessian.elements.first(source.size()*source.size()-1);
        reject(buffers, JetRequest::H, Status::insufficient_capacity);
        buffers = output.buffers(); --buffers.hessian.dimension;
        reject(buffers, JetRequest::H, Status::insufficient_capacity);
        for (unsigned mask : {0u, 8u, ~0u}) reject(output.buffers(), static_cast<JetRequest>(mask), Status::invalid_argument);
        for (unsigned alias = 0; alias < 3; ++alias) {
            buffers = output.buffers();
            if (alias == 0) buffers.value = buffers.gradient.first(1);
            if (alias == 1) buffers.value = buffers.hessian.elements.first(1);
            if (alias == 2) buffers.gradient = buffers.hessian.elements.first(source.size());
            reject(buffers, JetRequest::FGH, Status::invalid_argument);
        }
        buffers = output.buffers(); buffers.value = {const_cast<double*>(point.point->theta().data()), 1};
        reject(buffers, JetRequest::F, Status::invalid_argument);
        if (chart) {
            buffers = output.buffers(); buffers.value = {const_cast<double*>(point.point->original_coordinates().data()), 1};
            reject(buffers, JetRequest::F, Status::invalid_argument);
        }
        buffers = output.buffers(); buffers.value = {reinterpret_cast<double*>(arena.memory().data())+source.size(), 1};
        reject(buffers, JetRequest::F, Status::invalid_argument);
        alignas(double) std::array<std::byte, 2*sizeof(double)> unaligned;
        unaligned.fill(std::byte{0xa5});
        buffers = output.buffers(); buffers.value = {reinterpret_cast<double*>(unaligned.data()+1), 1};
        reject(buffers, JetRequest::F, Status::invalid_argument);
        for (auto byte : unaligned) CHECK(byte == std::byte{0xa5});
        valid(evaluate(objective, *point.point, JetRequest::FGH, workspace, output.buffers()), 7);
        output.untouched(7);
        arena.guards();
        auto minimal = take(reserve(objective, JetRequest::F));
        const auto value_point = bind_point(objective, source, minimal, chart);
        CHECK(value_point.status == Status::ok && value_point.point);
        output.reset();
        const auto too_large = evaluate(objective, *value_point.point, JetRequest::G, minimal, output.buffers());
        CHECK(too_large.status == Status::insufficient_capacity && too_large.valid_mask == 0);
        output.unchanged();
        auto bad = source; bad[0] = NAN;
        const auto nonfinite = bind_point(objective, bad, workspace, chart);
        CHECK(nonfinite.status == Status::invalid_argument && !nonfinite.point);
        const auto short_point = bind_point(objective, std::span<const double>(source).first(source.size()-1), workspace, chart);
        CHECK(short_point.status == Status::invalid_argument && !short_point.point);
    }
    for (bool aparch : {false, true}) {
        const auto model = make_model(aparch, {1, 1});
        const auto base = objective_for(make_problem(model));
        const auto plan = take(requirements(base, JetRequest::F));
        Arena sample_arena(plan.bytes, plan.alignment);
        auto* data = reinterpret_cast<double*>(sample_arena.memory().data());
        data[0] = .3; data[1] = -.7;
        Conditioning conditioning; conditioning.variance_seed = .8;
        auto lease = std::shared_ptr<const void>(data, [](const void*) {});
        const auto borrowed = take(prepare_borrowed(model, data_view({data, 2}), conditioning, {lease}));
        const auto objective = objective_for(borrowed);
        const auto borrowed_plan = take(requirements(objective, JetRequest::F));
        CHECK(borrowed_plan.bytes == plan.bytes);
        rejected(bind_workspace(borrowed_plan, sample_arena.memory()), Status::invalid_argument);
        auto workspace = take(reserve(objective, JetRequest::F));
        const auto source = parameters(model, aparch, {1, 1}, 0);
        const auto point = from_theta(objective, source, workspace);
        CHECK(point.status == Status::ok && point.point);
        Outputs output(objective);
        auto buffers = output.buffers(); buffers.value = {data, 1};
        const auto result = evaluate(objective, *point.point, JetRequest::F, workspace, buffers);
        CHECK(result.status == Status::invalid_argument && result.valid_mask == 0);
        CHECK(data[0] == .3 && data[1] == -.7);
        output.unchanged(); sample_arena.guards();
    }
}

void domains_and_numerical_range() {
    for (bool aparch : {false, true}) for (unsigned density = 1; density < 8; ++density) {
        const auto model = make_model(aparch, {1, 1}, density);
        const auto objective = objective_for(make_problem(model));
        auto theta = parameters(model, aparch, {1, 1}, density);
        const auto shape = 4+std::size_t(aparch);
        if (density == 1 || density == 2) theta[shape] = 2;
        else if (density == 3) theta[shape] = 0;
        else theta[shape] = 1; // rho=1 is outside the statistical domain.
        auto workspace = take(reserve(objective, JetRequest::F));
        const auto point = from_theta(objective, theta, workspace);
        CHECK(point.status == Status::ok && point.point);
        Outputs output(objective);
        const auto result = evaluate(objective, *point.point, JetRequest::F, workspace, output.buffers());
        CHECK(result.status == Status::mathematical_domain && result.valid_mask == 0 && result.f_reason == Status::mathematical_domain);
        output.unchanged();
    }
    for (bool aparch : {false, true}) for (unsigned density : {3u, 4u, 5u, 6u}) {
        const auto model = make_model(aparch, {1, 1}, density);
        const auto objective = objective_for(make_problem(model));
        auto theta = parameters(model, aparch, {1, 1}, density);
        const auto shape = 4+std::size_t(aparch);
        if (density == 3) {
            // All positive-shape GED moments exist. Lyapunov bounds E|Z|^1.4 by one.
            // Hence alpha*(1+|gamma|)^1.4+beta < 1 proves APARCH feasibility here.
            theta[shape] = std::numeric_limits<double>::denorm_min();
            CHECK(theta[shape] > 0 && !std::isfinite(1/theta[shape]));
            if (aparch) CHECK(theta[1]*std::pow(1+std::abs(theta[2]),theta[4])+theta[3] < 1);
        } else {
            // Positive GH zeta is statistical support, below its named numerical range.
            theta[shape+1] = std::numeric_limits<double>::denorm_min();
        }
        auto workspace = take(reserve(objective, JetRequest::F));
        const auto point = from_theta(objective, theta, workspace);
        CHECK(point.status == Status::ok && point.point);
        Outputs output(objective);
        const auto result = evaluate(objective, *point.point, JetRequest::F, workspace, output.buffers());
        CHECK(result.status == Status::numerical_range && result.valid_mask == 0 && result.f_reason == Status::numerical_range);
        output.untouched(1);
    }
    for (bool aparch : {false, true}) {
        const auto model = make_model(aparch, {1, 1}, 1);
        const auto objective = objective_for(make_problem(model));
        const auto source = parameters(model, aparch, {1, 1}, 1);
        auto workspace = take(reserve(objective, JetRequest::FGH));
        for (unsigned bad_case = 0; bad_case < (aparch ? 6u : 2u); ++bad_case) {
            auto theta = source;
            if (!aparch) {
                if (bad_case == 0) theta[3] = 1.0; else theta[4] = 2.0;
            } else {
                if (bad_case == 0) theta[0] = 0;
                if (bad_case == 1) theta[1] = -.01;
                if (bad_case == 2) theta[2] = 1.0;
                if (bad_case == 3) theta[3] = -.1;
                if (bad_case == 4) theta[4] = 0;
                if (bad_case == 5) theta[4] = theta[5]; // Infinite required Student-t power moment.
            }
            const auto point = from_theta(objective, theta, workspace);
            CHECK(point.status == Status::ok && point.point);
            Outputs output(objective);
            const auto result = evaluate(objective, *point.point, JetRequest::FGH, workspace, output.buffers());
            CHECK(result.status == Status::mathematical_domain && result.valid_mask == 0);
            CHECK(result.f_reason == Status::mathematical_domain && result.g_reason == Status::mathematical_domain
                && result.h_reason == Status::mathematical_domain);
            output.unchanged();
        }
    }
    const auto egarch = make_model(false, {1, 1});
    const auto eg_objective = objective_for(make_problem(egarch, 0, 2));
    auto eg_theta = parameters(egarch, false, {1, 1}, 0);
    eg_theta[0] = -1e300; // Nonzero next residual makes the Normal NLL itself unrepresentable.
    auto eg_workspace = take(reserve(eg_objective, JetRequest::F));
    const auto eg_point = from_theta(eg_objective, eg_theta, eg_workspace);
    CHECK(eg_point.status == Status::ok && eg_point.point);
    Outputs eg_output(eg_objective);
    const auto eg_result = evaluate(eg_objective, *eg_point.point, JetRequest::F, eg_workspace, eg_output.buffers());
    CHECK(eg_result.status == Status::numerical_range && eg_result.valid_mask == 0 && eg_result.f_reason == Status::numerical_range);

    // exp(log h) overflow is not proof that a log-coordinate objective is invalid.
    eg_theta[0] = 1000;
    const auto large_variance_point = from_theta(eg_objective, eg_theta, eg_workspace);
    CHECK(large_variance_point.status == Status::ok && large_variance_point.point);
    eg_output.reset();
    const auto large_variance = evaluate(eg_objective, *large_variance_point.point, JetRequest::F, eg_workspace, eg_output.buffers());
    if (large_variance.status == Status::ok) {
        valid(large_variance, 1);
        CHECK(close(eg_output.f[1], normal_nll(false, {1, 1}, observations, eg_theta, .8L, 0, 2)));
    } else {
        CHECK(large_variance.status == Status::numerical_range && large_variance.valid_mask == 0);
        std::printf("power support debt: EGARCH omega=1000 log-coordinate Normal F is representable\n");
    }
    eg_output.untouched(1);

    // Exact binary64 coefficients give persistence 1 - 2^-107, not an infeasible point.
    const auto near_model = make_model(true, {1, 2});
    const std::array<double, 6> near_theta{.2, .5, 0, 0x1.fffffffffffffp-2, 0x1.fffffffffffffp-55, 2};
    const auto near_objective = objective_for(make_problem(near_model));
    auto near_workspace = take(reserve(near_objective, JetRequest::F));
    const auto near_point = from_theta(near_objective, near_theta, near_workspace);
    CHECK(near_point.status == Status::ok && near_point.point);
    Outputs near_output(near_objective);
    const auto near_result = evaluate(near_objective, *near_point.point, JetRequest::F, near_workspace, near_output.buffers());
    CHECK(near_result.status == Status::numerical_range && near_result.valid_mask == 0 && near_result.f_reason == Status::numerical_range);
    near_output.unchanged();

    // The raw Normal moment overflows binary64, but alpha*M is finite and below one.
    const auto weighted_model = make_model(true, {1, 1});
    const std::array<double, 5> weighted_theta{.2, 0x1p-1040, 0, .2, 304};
    const long double log_moment = 152*std::log(2.0L) + std::lgamma(152.5L) - .5L*std::log(std::acos(-1.0L));
    CHECK(log_moment > std::log(static_cast<long double>(std::numeric_limits<double>::max())));
    CHECK(std::exp(std::log(static_cast<long double>(weighted_theta[1]))+log_moment) + .2L < 1);
    const auto weighted_objective = objective_for(make_problem(weighted_model, 0, 1));
    CHECK(close(jet(weighted_objective, weighted_theta, JetRequest::F).f[1], seed_nll(0, {}, observations[0], .8)));
}

void extra_capacity_is_not_part_of_the_objective() {
    for (bool aparch : {false, true}) for (bool chart : {false, true}) {
        const auto model = make_model(aparch, {3, 2});
        const auto objective = objective_for(make_problem(model), chart);
        const auto source = parameters(model, aparch, {3, 2}, 0, chart);
        auto workspace = take(reserve(objective, JetRequest::FGH));
        const auto point = bind_point(objective, source, workspace, chart);
        CHECK(point.status == Status::ok && point.point);
        Outputs exact(objective), extra(objective);
        valid(evaluate(objective, *point.point, JetRequest::FGH, workspace, exact.buffers()), 7);
        valid(evaluate(objective, *point.point, JetRequest::FGH, workspace, extra.buffers(true)), 7);
        extra.untouched(7);
        projection(extra, exact, 7);
    }
}

void owning_points_and_sample_identity() {
    for (bool aparch : {false, true}) {
        const auto model = make_model(aparch, {3, 2});
        const auto theta = parameters(model, aparch, {3, 2}, 0);
        auto mutable_data = observations;
        Conditioning conditioning; conditioning.variance_seed = .8;
        const auto copied = take(prepare(model, data_view(mutable_data), conditioning));
        mutable_data.fill(99.0);
        const auto objective = objective_for(copied);
        std::vector<NamedParameter> named;
        for (const auto& entry : model.layout().entries) named.push_back({entry.name, theta[entry.index]});
        std::reverse(named.begin(), named.end());
        const auto owning = take(from_theta(objective, named));
        CHECK(owning.theta().size() == theta.size());
        CHECK(std::equal(theta.begin(), theta.end(), owning.theta().begin()));
        CHECK(owning.objective() == objective.key() && owning.derivative_layout().id == objective.key().derivative_layout);
        auto workspace = take(reserve(objective, JetRequest::F));
        Outputs output(objective);
        valid(evaluate(objective, owning.view(), JetRequest::F, workspace, output.buffers()), 1);
        CHECK(close(output.f[1], normal_nll(aparch, {3, 2}, observations, theta, .8L, 0, observations.size())));
        auto duplicate = named; duplicate[1].name = duplicate[0].name;
        rejected(from_theta(objective, duplicate), Status::invalid_argument);
        auto unknown = named; unknown[0].name = "unknown_parameter";
        rejected(from_theta(objective, unknown), Status::invalid_argument);
        const auto other = objective_for(make_problem(model, 0));
        auto wrong_workspace = take(reserve(other, JetRequest::F));
        output.reset();
        const auto mismatched = evaluate(other, owning.view(), JetRequest::F, wrong_workspace, output.buffers());
        CHECK(mismatched.status == Status::invalid_argument && mismatched.valid_mask == 0);
        output.unchanged();

        const auto chart = objective_for(copied, true);
        auto z = parameters(model, aparch, {3, 2}, 0, true);
        const auto original = z;
        const auto chart_point = take(from_coordinates(chart, z));
        z.assign(z.size(), 99.0);
        CHECK(chart_point.original_coordinates().size() == original.size());
        CHECK(std::equal(original.begin(), original.end(), chart_point.original_coordinates().begin()));
        CHECK(chart_point.objective() == chart.key() && chart.key().chart != 0);
    }
}
} // namespace

int main() {
    layouts_and_admission();
    conditioning_and_independent_values();
    every_mask_and_coordinate();
    fixed_runtime_same_tuple();
    aparch_garch_reduction();
    workspace_and_buffer_admission();
    owning_points_and_sample_identity();
    domains_and_numerical_range();
    extra_capacity_is_not_part_of_the_objective();
    std::printf("power contract: %zu checks\n", checks);
    return EXIT_SUCCESS;
}
