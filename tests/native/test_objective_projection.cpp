#include "objective_projection.hpp"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

namespace {
using namespace scivolapi;
using scivolapi::detail::ObjectiveProjection;

constexpr double sentinel = -0x1.23456789abcp+20;
std::size_t checks = 0;
void check(bool pass, const char* expression, int line) {
    ++checks;
    if (!pass) {
        std::fprintf(stderr, "objective projection line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "projection setup: %s\n", result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}
template<class T> void rejected(const Outcome<T>& result, Status status) {
    CHECK(!result);
    CHECK(result.error().code == status);
}
void near(double actual, double expected) {
    CHECK(std::isfinite(actual) && std::isfinite(expected));
    CHECK(std::abs(actual - expected) <= 3e-10 + 3e-11 * std::abs(expected));
}
struct Arena {
    std::byte* memory;
    std::size_t size;
    explicit Arena(std::size_t bytes) : size(bytes) {
        memory = static_cast<std::byte*>(::operator new(size + 128, std::align_val_t{64}));
        std::fill_n(memory, size + 128, std::byte{0xa5});
    }
    ~Arena() { ::operator delete(memory, std::align_val_t{64}); }
    std::span<std::byte> span() { return {memory + 64, size}; }
    void guards() const {
        for (std::size_t i = 0; i < 64; ++i) {
            CHECK(memory[i] == std::byte{0xa5});
            CHECK(memory[size + 64 + i] == std::byte{0xa5});
        }
    }
};
struct Outputs {
    std::size_t width;
    LayoutId layout;
    std::vector<double> f, g, h;
    Outputs(std::size_t d, LayoutId id) : width(d), layout(id),
        f(3, sentinel), g(d + 2, sentinel), h(d * d + 2, sentinel) {}
    explicit Outputs(const ObjectiveView& objective)
        : Outputs(objective.derivative_layout().axes.size(), objective.key().derivative_layout) {}
    JetBuffers buffers() {
        return {layout, std::span{f}.subspan(1, 1), std::span{g}.subspan(1, width),
            {std::span{h}.subspan(1, width * width), width}};
    }
    void untouched(unsigned mask) const {
        CHECK(f.front() == sentinel && f.back() == sentinel);
        CHECK(g.front() == sentinel && g.back() == sentinel);
        CHECK(h.front() == sentinel && h.back() == sentinel);
        if (!(mask & 1)) for (double x : f) CHECK(x == sentinel);
        if (!(mask & 2)) for (double x : g) CHECK(x == sentinel);
        if (!(mask & 4)) for (double x : h) CHECK(x == sentinel);
    }
};

void selection_contract() {
    const std::array<ParameterEntry, 3> entries{{{11, "a", 0}, {23, "b", 1}, {37, "c", 2}}};
    const ParameterLayout layout{71, entries, {}, 3};
    const std::array<ParameterId, 2> axes{37, 11};
    const std::array<NamedParameter, 1> fixed{{{"b", -0.0}}};
    auto projection = take(ObjectiveProjection::resolve(layout,
        DerivativeTarget::fixed_free_restriction, Coordinates::theta, {axes, fixed}));
    CHECK(projection.axes()[0] == 37 && projection.axes()[1] == 11);
    CHECK(projection.indices()[0] == 2 && projection.indices()[1] == 0);
    CHECK(projection.partition()[1] == ParameterDisposition::fixed);
    std::array<double, 3> theta{2, +0.0, 4}, reconstructed{};
    const auto before = theta;
    CHECK(projection.matches(theta));
    CHECK(std::memcmp(theta.data(), before.data(), sizeof(theta)) == 0);
    CHECK(!std::signbit(theta[1]));
    const std::array<double, 2> coordinates{4, 2};
    CHECK(projection.expand(coordinates, reconstructed) == Status::ok);
    CHECK(reconstructed[0] == 2 && reconstructed[2] == 4 && std::signbit(reconstructed[1]));
    CHECK(projection.expand(std::span{theta}.first(2), theta) == Status::invalid_argument);
    theta[1] = std::nextafter(0.0, 1.0);
    CHECK(!projection.matches(theta));
    CHECK(!projection.matches(coordinates));

    auto resolve = [&](ObjectiveRestriction restriction, DerivativeTarget target = DerivativeTarget::fixed_free_restriction,
                       Coordinates chart = Coordinates::theta) {
        return ObjectiveProjection::resolve(layout, target, chart, restriction);
    };
    rejected(resolve({axes, {}}), Status::invalid_argument);
    const std::array<ParameterId, 2> repeated{37, 37}, unknown{37, 999};
    rejected(resolve({repeated, fixed}), Status::invalid_argument);
    rejected(resolve({unknown, fixed}), Status::invalid_argument);
    const std::array<NamedParameter, 1> overlap{{{"a", 2}}}, missing{{{"unknown", 0}}},
        nan{{{"b", std::numeric_limits<double>::quiet_NaN()}}};
    rejected(resolve({axes, overlap}), Status::invalid_argument);
    rejected(resolve({axes, missing}), Status::invalid_argument);
    rejected(resolve({axes, nan}), Status::invalid_argument);
    const std::array<ParameterId, 1> one{37};
    const std::array<NamedParameter, 2> duplicate{{{"b", 0}, {"b", 0}}};
    rejected(resolve({one, duplicate}), Status::invalid_argument);
    rejected(resolve({axes, fixed}, DerivativeTarget::full_theta), Status::invalid_argument);
    rejected(resolve({axes, fixed}, DerivativeTarget::fixed_free_restriction,
        Coordinates::original_chart), Status::unsupported_operation);
    rejected(resolve({}, DerivativeTarget::exact_profile), Status::unsupported_operation);
    auto identity = take(resolve({}, DerivativeTarget::full_theta));
    CHECK(!identity.restricted() && identity.axes().size() == 3 && identity.matches(before));
    std::size_t bytes = 99;
    CHECK(identity.scratch_bytes(JetRequest::FGH, bytes) && bytes == 0);
    CHECK(projection.scratch_bytes(JetRequest::F, bytes) && bytes == 0);
    CHECK(projection.scratch_bytes(JetRequest::G, bytes) && bytes == 64);
    CHECK(projection.scratch_bytes(JetRequest::H, bytes) && bytes == 128);
    CHECK(!projection.scratch_bytes(static_cast<JetRequest>(0), bytes));
    auto bad_entries = entries;
    bad_entries[2].index = 1;
    rejected(ObjectiveProjection::resolve({71, bad_entries, {}, 3}, DerivativeTarget::full_theta,
        Coordinates::theta, {}), Status::invalid_argument);
}

// A quadratic with nonzero cross blocks makes axis transposition/order mistakes visible.
JetValidity quadratic(std::span<const double> theta, JetRequest request,
    std::span<std::byte>, JetBuffers out) noexcept {
    const auto mask = unsigned(request);
    const auto w = theta.size();
    if (mask & 1) {
        out.value[0] = 13;
        for (std::size_t i = 0; i < w; ++i)
            for (std::size_t j = 0; j < w; ++j)
                out.value[0] += 0.5 * double((i + 1) * (j + 1)) * theta[i] * theta[j];
    }
    if (mask & 2)
        for (std::size_t i = 0; i < w; ++i) {
            out.gradient[i] = 0;
            for (std::size_t j = 0; j < w; ++j)
                out.gradient[i] += double((i + 1) * (j + 1)) * theta[j];
        }
    if (mask & 4)
        for (std::size_t i = 0; i < w; ++i)
            for (std::size_t j = 0; j < w; ++j)
                out.hessian.elements[i * w + j] = double((i + 1) * (j + 1));
    return {Status::ok, request, mask, (mask & 1) ? Status::ok : Status::unsupported_operation,
        (mask & 2) ? Status::ok : Status::unsupported_operation,
        (mask & 4) ? Status::ok : Status::unsupported_operation};
}

void projection_requests_and_failures() {
    const std::array<ParameterEntry, 3> entries{{{11, "a", 0}, {23, "b", 1}, {37, "c", 2}}};
    const ParameterLayout layout{71, entries, {}, 3};
    const std::array<ParameterId, 2> axes{37, 11};
    const std::array<NamedParameter, 1> fixed{{{"b", 3}}};
    auto projection = take(ObjectiveProjection::resolve(layout,
        DerivativeTarget::fixed_free_restriction, Coordinates::theta, {axes, fixed}));
    const std::array<double, 3> theta{2, 3, 4};
    for (unsigned mask = 1; mask <= 7; ++mask) {
        const auto request = JetRequest(mask);
        std::size_t bytes;
        CHECK(projection.scratch_bytes(request, bytes));
        Arena arena(bytes);
        Outputs out(2, 99);
        unsigned calls = 0;
        const auto result = projection.evaluate(theta, request, request, arena.span(), out.buffers(),
            [&](auto point, auto received, auto work, auto outputs) noexcept {
                ++calls;
                CHECK(received == request && point.data() == theta.data());
                CHECK(work.empty());
                CHECK(outputs.gradient.size() == ((mask & 2) ? 3 : 0));
                CHECK(outputs.hessian.elements.size() == ((mask & 4) ? 9 : 0));
                return quadratic(point, received, work, outputs);
            });
        CHECK(calls == 1 && result.contains(request));
        if (mask & 1) CHECK(out.f[1] == 213);
        if (mask & 2) CHECK(out.g[1] == 60 && out.g[2] == 20);
        if (mask & 4) CHECK(out.h[1] == 9 && out.h[2] == 3 && out.h[3] == 3 && out.h[4] == 1);
        out.untouched(mask);
        arena.guards();
    }
    std::size_t bytes;
    CHECK(projection.scratch_bytes(JetRequest::FGH, bytes));
    Arena arena(bytes);
    std::byte* backend_base = nullptr;
    for (unsigned mask = 1; mask <= 7; ++mask) {
        Outputs out(2, 99);
        CHECK(projection.evaluate(theta, JetRequest(mask), JetRequest::FGH, arena.span(), out.buffers(),
            [&](auto point, auto request, auto work, auto outputs) noexcept {
                if (!backend_base) backend_base = work.data();
                CHECK(work.data() == backend_base);
                return quadratic(point, request, work, outputs);
            }).contains(JetRequest(mask)));
        out.untouched(mask);
    }
    Outputs failed(2, 99);
    const auto ambient_kink = [&](auto, auto request, auto, auto out) noexcept {
            out.value[0] = 213;
            std::fill(out.gradient.begin(), out.gradient.end(), 777);
            std::fill(out.hessian.elements.begin(), out.hessian.elements.end(), 777);
            return JetValidity{Status::nondifferentiable, request, 1, Status::ok,
                Status::nondifferentiable, Status::nondifferentiable};
        };
    const auto partial = projection.evaluate(theta, JetRequest::FGH, JetRequest::FGH, arena.span(),
        failed.buffers(), ambient_kink);
    CHECK(partial.status == Status::unsupported_operation && partial.valid_mask == 1);
    CHECK(partial.g_reason == Status::unsupported_operation && partial.h_reason == Status::unsupported_operation);
    failed.untouched(1);
    const std::array<ParameterId,3> permutation{37,11,23};
    const auto full = take(ObjectiveProjection::resolve(layout,
        DerivativeTarget::fixed_free_restriction, Coordinates::theta, {permutation,{}}));
    std::size_t full_bytes;
    CHECK(full.scratch_bytes(JetRequest::FGH,full_bytes));
    Arena full_arena(full_bytes);
    Outputs full_output(3,99);
    const auto full_failure = full.evaluate(theta,JetRequest::FGH,JetRequest::FGH,full_arena.span(),
        full_output.buffers(),ambient_kink);
    CHECK(full_failure.status == Status::nondifferentiable && full_failure.valid_mask == 1);
    CHECK(full_failure.g_reason == Status::nondifferentiable && full_failure.h_reason == Status::nondifferentiable);
    full_output.untouched(1);
    Outputs short_output(2, 99);
    auto outputs = short_output.buffers();
    outputs.gradient = outputs.gradient.first(1);
    unsigned calls = 0;
    const auto counted = [&](auto point, auto request, auto work, auto out) noexcept {
        ++calls;
        return quadratic(point, request, work, out);
    };
    CHECK(projection.evaluate(theta, JetRequest::FGH, JetRequest::FGH, arena.span(), outputs, counted).status
        == Status::insufficient_capacity);
    CHECK(projection.evaluate(theta, JetRequest::FGH, JetRequest::FGH, arena.span().first(bytes - 1),
        short_output.buffers(), counted).status == Status::insufficient_capacity);
    outputs = short_output.buffers();
    outputs.gradient = outputs.hessian.elements.first(2);
    CHECK(projection.evaluate(theta, JetRequest::G, JetRequest::FGH, arena.span(), outputs, counted).status
        == Status::invalid_argument);
    CHECK(calls == 0);
    short_output.untouched(0);
    CHECK(projection.evaluate(theta, JetRequest::H, JetRequest::F, arena.span(),
        short_output.buffers(), counted).status == Status::insufficient_capacity);
    outputs = short_output.buffers();
    outputs.value = {reinterpret_cast<double*>(arena.span().data()), 1};
    CHECK(projection.evaluate(theta, JetRequest::G, JetRequest::FGH, arena.span(), outputs, counted).status
        == Status::invalid_argument);
    CHECK(calls == 0);
    const auto nonfinite = projection.evaluate(theta, JetRequest::G, JetRequest::FGH, arena.span(), short_output.buffers(),
        [](auto point, auto request, auto work, auto out) noexcept {
            auto result = quadratic(point, request, work, out);
            out.gradient[2] = std::numeric_limits<double>::infinity();
            return result;
        });
    CHECK(nonfinite.valid_mask == 0 && nonfinite.g_reason == Status::numerical_range);

    const std::array<NamedParameter, 3> all_fixed{{{"c", 4}, {"a", 2}, {"b", 3}}};
    auto constant = take(ObjectiveProjection::resolve(layout,
        DerivativeTarget::fixed_free_restriction, Coordinates::theta, {{}, all_fixed}));
    CHECK(constant.axes().empty());
    std::array<double, 3> expanded{};
    CHECK(constant.expand({}, expanded) == Status::ok && expanded == theta);
    for (unsigned mask = 1; mask <= 7; ++mask) {
        Outputs out(0, 100);
        CHECK(constant.scratch_bytes(JetRequest(mask), bytes) && bytes == 0);
        CHECK(constant.ambient_request(JetRequest(mask)) == JetRequest::F);
        const auto result = constant.evaluate(theta, JetRequest(mask), JetRequest::FGH, {}, out.buffers(),
            [&](auto point, auto request, auto work, auto outputs) noexcept {
                CHECK(request == JetRequest::F && outputs.gradient.empty() && outputs.hessian.elements.empty());
                return quadratic(point, request, work, outputs);
            });
        CHECK(result.valid_mask == mask);
        out.untouched(mask);
    }
    Outputs out(0, 100);
    const auto invalid_domain = constant.evaluate(theta, JetRequest::GH, JetRequest::FGH, {}, out.buffers(),
        [](auto, auto request, auto, auto) noexcept {
            return JetValidity{Status::mathematical_domain, request, 0, Status::mathematical_domain};
        });
    CHECK(invalid_domain.valid_mask == 0 && invalid_domain.g_reason == Status::mathematical_domain);
    out.untouched(0);
    const auto invalid_value = constant.evaluate(theta, JetRequest::G, JetRequest::FGH, {}, out.buffers(),
        [](auto, auto request, auto, auto outputs) noexcept {
            outputs.value[0] = std::numeric_limits<double>::infinity();
            return JetValidity{Status::ok, request, 1, Status::ok};
        });
    CHECK(invalid_value.valid_mask == 0 && invalid_value.g_reason == Status::numerical_range);
    out.untouched(0);
}

// Independent two-variable forward AD of a declared Normal regression likelihood.
struct Dual {
    double f;
    std::array<double, 2> g{};
    std::array<double, 4> h{};
    static Dual variable(double value, std::size_t axis) { Dual x{value}; x.g[axis] = 1; return x; }
};
Dual operator+(Dual a, Dual b) {
    Dual r{a.f + b.f};
    for (std::size_t i = 0; i < 2; ++i) r.g[i] = a.g[i] + b.g[i];
    for (std::size_t i = 0; i < 4; ++i) r.h[i] = a.h[i] + b.h[i];
    return r;
}
Dual operator*(Dual a, Dual b) {
    Dual r{a.f * b.f};
    for (std::size_t i = 0; i < 2; ++i) {
        r.g[i] = a.g[i] * b.f + a.f * b.g[i];
        for (std::size_t j = 0; j < 2; ++j)
            r.h[2 * i + j] = a.h[2 * i + j] * b.f + a.f * b.h[2 * i + j]
                + a.g[i] * b.g[j] + b.g[i] * a.g[j];
    }
    return r;
}
Dual unary(Dual x, double value, double slope, double curvature) {
    Dual r{value};
    for (std::size_t i = 0; i < 2; ++i) {
        r.g[i] = slope * x.g[i];
        for (std::size_t j = 0; j < 2; ++j)
            r.h[2 * i + j] = slope * x.h[2 * i + j] + curvature * x.g[i] * x.g[j];
    }
    return r;
}
Dual logarithm(Dual x) { return unary(x, std::log(x.f), 1 / x.f, -1 / (x.f * x.f)); }
Dual inverse(Dual x) { return unary(x, 1 / x.f, -1 / (x.f * x.f), 2 / (x.f * x.f * x.f)); }

const std::array<double, 8> observations{.31, -.73, 1.17, -.29, .57, -1.37, .93, .43};
const std::array<std::uint64_t, 8> rows{101, 107, 111, 123, 130, 144, 167, 200};
const std::array<double, 8> design{.5, -.5, 1, .25, -.75, .125, 1.5, -.125};

double parameter_value(std::string_view name, bool egarch) {
    if (name == "intercept") return .03;
    if (name.starts_with("coefficient")) return .025;
    if (name.starts_with("ar[")) return .09;
    if (name.starts_with("ma[")) return -.13;
    if (name == "sigma2") return .93;
    if (name == "omega") return egarch ? -.11 : .11;
    if (name.starts_with("alpha[")) return .085;
    if (name.starts_with("gamma[")) return -.025;
    if (name.starts_with("beta[")) return .12;
    if (name == "delta") return 1.7;
    if (name == "nu") return 8.5;
    CHECK(false);
    return 0;
}
std::vector<NamedParameter> named(const Model& model, std::span<const double> theta) {
    std::vector<NamedParameter> values;
    for (const auto& entry : model.layout().entries) values.push_back({entry.name, theta[entry.index]});
    std::reverse(values.begin(), values.end());
    return values;
}

void public_case(unsigned backend, bool runtime) {
    ModelSpec spec;
    const auto p = runtime ? 3u : 1u, q = runtime ? 2u : 1u;
    const bool linear = backend == 1 || backend == 4;
    if (linear) spec.mean = LinearMean{};
    if (backend == 3) spec.mean = ARMA{runtime ? 2u : 1u, runtime ? 2u : 1u};
    switch (backend) {
    case 0: case 3: spec.volatility = GARCH{p, q}; break;
    case 1: spec.volatility = ConstantVariance{}; break;
    case 2: spec.volatility = EGARCH{p, q}; break;
    case 4: spec.volatility = APARCH{p, q}; break;
    default: CHECK(false);
    }
    spec.density = backend == 1 ? decltype(spec.density)::value_type{Normal{}}
                                : decltype(spec.density)::value_type{StudentT{}};
    auto model = take(compile(spec, {linear ? 1u : 0u}));
    DataView data;
    data.observations = observations; data.row_ids = rows; data.row_space = 73;
    if (linear) data.design = {design, observations.size(), 1};
    Conditioning conditioning;
    if (backend != 1) conditioning.variance_seed = .8;
    ScoreSelection selection;
    selection.first_original_position = 5;
    auto problem = take(prepare(model, data, conditioning, selection));
    auto full = take(resolve_objective(problem));
    const auto w = model.layout().entries.size();
    std::vector<double> theta;
    for (const auto& entry : model.layout().entries) theta.push_back(parameter_value(entry.name, backend == 2));
    auto full_work = take(reserve(full, JetRequest::FGH));
    auto full_point = from_theta(full, theta, full_work);
    CHECK(full_point.status == Status::ok);

    for (unsigned kind = 0; kind < 3; ++kind) {
        std::vector<std::size_t> indices;
        if (kind == 0) indices = {w - 1, 0};
        else if (kind == 1) for (std::size_t i = w; i-- > 0;) indices.push_back(i);
        std::vector<ParameterId> axes;
        std::vector<NamedParameter> fixed;
        for (auto i : indices) axes.push_back(model.layout().entries[i].id);
        for (const auto& entry : model.layout().entries)
            if (std::find(indices.begin(), indices.end(), entry.index) == indices.end())
                fixed.push_back({entry.name, theta[entry.index]});
        ObjectiveRequest request;
        request.target = DerivativeTarget::fixed_free_restriction;
        request.restriction = {axes, fixed};
        auto objective = take(resolve_objective(problem, request));
        CHECK(objective.key().scope == ObjectiveScope::full_data_likelihood);
        CHECK(objective.key().restriction_id != 0 && objective.key().scaling == Scaling::additive_nll);
        CHECK(objective.key().derivative_layout != full.key().derivative_layout);
        CHECK(objective.derivative_layout().target == DerivativeTarget::fixed_free_restriction);
        CHECK(std::equal(axes.begin(), axes.end(), objective.derivative_layout().axes.begin()));
        for (std::size_t i = 0; i < w; ++i)
            CHECK(objective.derivative_layout().full_parameter_partition[i]
                == (std::find(indices.begin(), indices.end(), i) != indices.end()
                    ? ParameterDisposition::free : ParameterDisposition::fixed));
        request.coordinates = Coordinates::original_chart;
        request.chart = 2;
        rejected(resolve_objective(problem, request), Status::unsupported_operation);

        std::vector<double> free;
        for (auto i : indices) free.push_back(theta[i]);
        auto ambient = take(from_theta(objective, named(model, theta)));
        auto coordinate_point = take(from_coordinates(objective, free));
        CHECK(ambient.theta().size() == w && ambient.original_coordinates().empty());
        CHECK(coordinate_point.theta().size() == w && coordinate_point.original_coordinates().size() == free.size());
        CHECK(std::memcmp(ambient.theta().data(), theta.data(), w * sizeof(double)) == 0);
        CHECK(std::memcmp(coordinate_point.theta().data(), theta.data(), w * sizeof(double)) == 0);
        CHECK(ambient.evidence().full_theta_layout == model.layout().id);
        // Resolved metadata must not borrow request arrays.
        if (!axes.empty()) axes[0] = 999;
        if (!fixed.empty()) fixed[0].value = 999;
        auto plan = take(requirements(objective, JetRequest::FGH));
        Arena arena(plan.bytes);
        auto work = take(bind_workspace(plan, arena.span()));
        auto binding = from_theta(objective, theta, work);
        CHECK(binding.status == Status::ok && binding.point);
        CHECK(binding.point->theta().size() == w);
        for (unsigned mask = 1; mask <= 7; ++mask) {
            Outputs output(objective), baseline(full);
            auto expected = evaluate(full, *full_point.point, JetRequest(mask), full_work, baseline.buffers());
            auto actual = evaluate(objective, *binding.point, JetRequest(mask), work, output.buffers());
            CHECK(expected.contains(JetRequest(mask)) && actual.contains(JetRequest(mask)));
            if (mask & 1) near(output.f[1], baseline.f[1]);
            for (std::size_t i = 0; i < indices.size(); ++i) {
                if (mask & 2) CHECK(output.g[i + 1] == baseline.g[indices[i] + 1]);
                if (mask & 4)
                    for (std::size_t j = 0; j < indices.size(); ++j)
                        CHECK(output.h[1 + i * indices.size() + j] == baseline.h[1 + indices[i] * w + indices[j]]);
            }
            output.untouched(mask);
            baseline.untouched(mask);
            arena.guards();
        }
        auto coordinate_binding = from_coordinates(objective, free, work);
        CHECK(coordinate_binding.status == Status::ok && coordinate_binding.point);
        CHECK(std::equal(theta.begin(), theta.end(), coordinate_binding.point->theta().begin()));
        CHECK(std::equal(free.begin(), free.end(), coordinate_binding.point->original_coordinates().begin()));
        Outputs output(objective);
        CHECK(evaluate(objective, *binding.point, JetRequest::F, work, output.buffers()).status == Status::invalid_argument);
        if (!indices.empty()) {
            free[0] = std::nextafter(free[0], 100.0);
            auto changed = from_coordinates(objective, free, work);
            CHECK(changed.status == Status::ok);
            CHECK(changed.point->theta()[indices[0]] == free[0]);
        }
        auto bad = theta;
        if (indices.size() != w) {
            for (std::size_t i = 0; i < w; ++i)
                if (std::find(indices.begin(), indices.end(), i) == indices.end()) {
                    bad[i] = std::nextafter(bad[i], 100.0);
                    break;
                }
            CHECK(from_theta(objective, bad, work).status == Status::invalid_argument);
            rejected(from_theta(objective, named(model, bad)), Status::invalid_argument);
            CHECK(from_theta(objective, std::span{theta}.first(indices.size()), work).status == Status::invalid_argument);
        }
        output.untouched(0);
        auto wrong = output.buffers();
        wrong.derivative_layout = full.key().derivative_layout;
        CHECK(evaluate(objective, ambient.view(), JetRequest::F, work, wrong).status == Status::invalid_argument);
        CHECK(evaluate(objective, *full_point.point, JetRequest::F, work, output.buffers()).status == Status::invalid_argument);
        auto f_work = take(reserve(objective, JetRequest::F));
        CHECK(evaluate(objective, ambient.view(), JetRequest::G, f_work, output.buffers()).status
            == Status::insufficient_capacity);
        rejected(from_coordinates(full, free), Status::invalid_argument);
        if (indices.size() != w)
            rejected(from_coordinates(objective, theta), Status::invalid_argument);

        // Owning sparse output storage has the free width, never the ambient width.
        auto sparse = take(allocate_jet_storage(objective, JetRequest::H));
        CHECK(sparse.gradient().empty() && sparse.hessian().size() == indices.size() * indices.size());
        CHECK(std::isnan(sparse.value()));
        CHECK(evaluate(objective, ambient.view(), JetRequest::H, work, sparse.outputs()).contains(JetRequest::H));

        if (backend == 1 && kind == 0) {
            const auto variance = Dual::variable(.93, 0), intercept = Dual::variable(.03, 1);
            Dual nll{0};
            for (std::size_t t = 5; t < observations.size(); ++t) {
                const auto residual = Dual{observations[t] - .025 * design[t]} + Dual{-1} * intercept;
                nll = nll + Dual{.5} * (Dual{std::log(2 * std::acos(-1.0))}
                    + logarithm(variance) + residual * residual * inverse(variance));
            }
            Outputs independent(objective);
            CHECK(evaluate(objective, ambient.view(), JetRequest::FGH, work, independent.buffers()).contains(JetRequest::FGH));
            near(independent.f[1], nll.f);
            for (std::size_t i = 0; i < 2; ++i) near(independent.g[i + 1], nll.g[i]);
            for (std::size_t i = 0; i < 4; ++i) near(independent.h[i + 1], nll.h[i]);
        }
    }
}

void public_invalid_fixed_domain() {
    ModelSpec spec;
    spec.volatility = GARCH{1, 1};
    auto model = take(compile(spec));
    DataView data;
    data.observations = observations; data.row_ids = rows; data.row_space = 73;
    Conditioning conditioning; conditioning.variance_seed = .8;
    auto problem = take(prepare(model, data, conditioning));
    const std::array<NamedParameter, 3> fixed{{{"omega", -1}, {"alpha[1]", .1}, {"beta[1]", .2}}};
    ObjectiveRequest request;
    request.target = DerivativeTarget::fixed_free_restriction;
    request.restriction.fixed_values = fixed;
    auto objective = take(resolve_objective(problem, request));
    auto point = take(from_coordinates(objective, {}));
    auto work = take(reserve(objective, JetRequest::FGH));
    Outputs output(objective);
    const auto result = evaluate(objective, point.view(), JetRequest::GH, work, output.buffers());
    CHECK(result.status != Status::ok && result.valid_mask == 0);
    output.untouched(0);
}
} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && (argc != 2 || std::string_view(argv[1]) != "--owner-only")) {
        std::fprintf(stderr, "Only --owner-only is accepted; default runs public integration.\n");
        return EXIT_FAILURE;
    }
    selection_contract();
    projection_requests_and_failures();
    // --owner-only is a bounded header-owner gate. Default requires merged public integration.
    if (argc == 1) {
        for (unsigned backend = 0; backend < 5; ++backend)
            for (bool runtime : {false, true}) public_case(backend, runtime);
        public_invalid_fixed_domain();
    }
    std::printf("objective projection: %zu checks\n", checks);
}
