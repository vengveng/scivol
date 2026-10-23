// Source-only public-consumer regression. Register only in an authorized build.
#include <scivolapi/jet.hpp>
#include <scivolapi/model.hpp>
#include <scivolapi/problem.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {
using namespace scivolapi;
constexpr double marker = -1234567.125;
void check(bool pass, const char* expression, int line) {
    if (!pass) {
        std::fprintf(stderr, "selected Gaussian range line %d: %s\n", line, expression);
        std::exit(1);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> result) {
    CHECK(result);
    return std::move(result).value();
}
void close_gradient(double actual, double expected) {
    CHECK(std::isfinite(actual) && actual != 0);
    CHECK(std::abs(actual - expected) <= std::abs(expected) * 2e-13);
}

void gaussian(double response_divisor, double design_divisor) {
    const std::array<double, 3> y{1, 1, 1}, x{1e-200, 1e-200, 1e-200};
    const std::array<std::uint64_t, 3> rows{1, 2, 3};
    const std::array<double, 1> divisors{design_divisor};
    const std::array<double, 2> theta{0, 1};
    ModelSpec spec;
    spec.mean = LinearMean{false};
    spec.density = Normal{};
    const auto model = take(compile(spec, {1}));
    DataView data;
    data.observations = y;
    data.design = {x, y.size(), 1};
    data.row_ids = rows;
    data.row_space = 501;
    data.scaling.internal_response_scale = response_divisor;
    data.scaling.internal_design_scales = divisors;
    ScoreSelection score;
    score.first_original_position = 2;
    const auto problem = take(prepare(model, data, {}, score));

    const auto& layout = model.layout();
    const std::array<ParameterId, 1> free{layout.entries[0].id};
    const std::array<NamedParameter, 1> fixed{{{layout.entries[1].name, theta[1]}}};
    ObjectiveRequest request;
    request.target = DerivativeTarget::fixed_free_restriction;
    request.restriction = {free, fixed};
    const auto objective = take(resolve_objective(problem, request));
    auto work = take(reserve(objective, JetRequest::FG));
    const auto point = from_theta(objective, theta, work);
    CHECK(point.point);

    // One scored Gaussian observation: d/db [.5*(y-b*x)^2] = -x at b=0.
    // H_bb=x*x is not representable here, so this fixture requests only F/G.
    const double expected_f = .5 * (std::log(2 * std::acos(-1.0)) + 1);
    for (unsigned mask : {1u, 2u, 3u, 1u, 3u}) {
        std::array<double, 3> f{marker, marker, marker};
        std::array<double, 3> g{marker, marker, marker};
        std::array<double, 3> h{marker, marker, marker};
        const JetBuffers output{objective.key().derivative_layout,
            {f.data() + 1, 1}, {g.data() + 1, 1}, {{h.data() + 1, 1}, 1}};
        const auto result = evaluate(objective, *point.point,
            static_cast<JetRequest>(mask), work, output);
        CHECK(result.status == Status::ok && result.valid_mask == mask);
        if (mask & 1) CHECK(std::isfinite(f[1]) && std::abs(f[1] - expected_f) <= 3e-12);
        else CHECK(f[1] == marker);
        if (mask & 2) close_gradient(g[1], -x[2]);
        else CHECK(g[1] == marker);
        CHECK(f.front() == marker && f.back() == marker);
        CHECK(g.front() == marker && g.back() == marker);
        CHECK(h[0] == marker && h[1] == marker && h[2] == marker);
    }

    // The qualified full-unit route is a control, not the expected-value oracle.
    const auto full = take(resolve_objective(problem));
    auto full_work = take(reserve(full, JetRequest::G));
    const auto full_point = from_theta(full, theta, full_work);
    CHECK(full_point.point);
    std::array<double, 2> full_g{marker, marker};
    const auto full_result = evaluate(full, *full_point.point, JetRequest::G,
        full_work, {full.key().derivative_layout, {}, full_g, {{}, 2}});
    CHECK(full_result.valid_mask == 2);
    close_gradient(full_g[0], -x[2]);
}
}

int main() {
    gaussian(1, 1); // Native unmapped active-basis control remains independent.
    gaussian(1e150, 1e-200);
    std::puts("selected Gaussian delayed-narrowing regression passed");
}
