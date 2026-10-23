// SCIVOL-NATIVE: TEST_ONLY; generic Newton trials and best-point retention.
#include "curvature_refinement.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

static std::size_t allocations = 0u;

void* operator new(std::size_t bytes)
{
    ++allocations;
    if (void* memory = std::malloc(bytes ? bytes : 1u))
        return memory;
    std::abort();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace curvature = vlk::experiment::curvature;
namespace kraft = vlk::experiment::kraft;
namespace runtime = vlk::experiment::runtime;

struct Quadratic {
    std::size_t width = 1u;
    double center[2] = {};
    double offset = 0.0;
    double matrix[4] = {1.0};
    double initial[2] = {};
    mutable std::size_t value_requests = 0u;
    mutable std::size_t gradient_requests = 0u;
    mutable std::size_t curvature_requests = 0u;
    mutable std::size_t unchanged_value_requests = 0u;
    mutable double first_trial[2] = {};
    mutable double last_curvature[2] = {};
    mutable double gradient_point[2] = {};

    std::size_t dimension() const noexcept { return width; }

    double compute(const double* x, double* gradient) const noexcept
    {
        const double residual[2] = {(x[0] - center[0]) + offset,
                                   x[1] - center[1]};
        double value = 0.0;
        for (std::size_t row = 0u; row < width; ++row) {
            double derivative = 0.0;
            for (std::size_t column = 0u; column < width; ++column)
                derivative += matrix[row * width + column] * residual[column];
            value += 0.5 * residual[row] * derivative;
            if (gradient)
                gradient[row] = derivative;
        }
        return value;
    }

    bool evaluate_curvature(
        const double* x, double* value, double* gradient, double* hessian) const noexcept
    {
        ++curvature_requests;
        std::copy_n(x, width, last_curvature);
        *value = compute(x, gradient);
        std::copy_n(matrix, width * width, hessian);
        return true;
    }

    bool evaluate(
        kraft::ObjectiveRequest request, const double* x, double* value,
        double* gradient, bool same_point) const noexcept
    {
        assert(!same_point);
        if (request == kraft::ObjectiveRequest::gradient) {
            ++gradient_requests;
            std::copy_n(x, width, gradient_point);
            compute(x, gradient);
            return true;
        }
        assert(request == kraft::ObjectiveRequest::value);
        if (value_requests++ == 0u)
            std::copy_n(x, width, first_trial);
        bool unchanged = true;
        for (std::size_t index = 0u; index < width; ++index)
            unchanged = unchanged && x[index] == initial[index];
        unchanged_value_requests += unchanged;
        *value = compute(x, nullptr);
        return true;
    }
};

enum class RetentionCase {
    growing_gradient,
    failed_hessian,
    failed_gradient,
    nonfinite_gradient,
    rejected_armijo,
    unrelated_convergence,
    initial_failure,
};

struct RetentionObjective : Quadratic {
    RetentionCase scenario;
    const double* continuation = nullptr;

    explicit RetentionObjective(RetentionCase selected) noexcept : scenario(selected) {}

    double compute(const double* x, double* gradient) const noexcept
    {
        double cubic = 1.0, quartic = 0.0, quintic = 0.0;
        if (scenario == RetentionCase::rejected_armijo)
            cubic = -0.5 + 1e-5;
        if (scenario == RetentionCase::unrelated_convergence) {
            const double first_decrease = 7.5e-5;
            const double second_decrease = 5.1e-5;
            cubic = -14.5 + first_decrease + 48.0 * second_decrease;
            quartic = -32.0 + 4.0 * first_decrease + 112.0 * second_decrease;
            quintic = -18.0 + 4.0 * first_decrease + 64.0 * second_decrease;
        }
        const double x2 = x[0] * x[0];
        const double x3 = x2 * x[0];
        const double x4 = x3 * x[0];
        const double x5 = x4 * x[0];
        if (gradient) {
            gradient[0] = 1.0 + x[0] + 3.0 * cubic * x2
                + 4.0 * quartic * x3 + 5.0 * quintic * x4;
        }
        return x[0] + 0.5 * x2 + cubic * x3 + quartic * x4 + quintic * x5;
    }

    bool evaluate_curvature(
        const double* x, double* value, double* gradient, double* hessian) const noexcept
    {
        ++curvature_requests;
        std::copy_n(x, width, last_curvature);
        if (scenario == RetentionCase::initial_failure
            || (x[0] != 0.0 && (scenario == RetentionCase::failed_hessian
                || scenario == RetentionCase::failed_gradient
                || scenario == RetentionCase::nonfinite_gradient))) {
            // Partial outputs from a failed combined request are not evidence.
            *value = -100.0;
            gradient[0] = hessian[0] = 0.0;
            return false;
        }
        *value = compute(x, gradient);
        hessian[0] = 1.0 + 6.0 * x[0];
        if (scenario == RetentionCase::unrelated_convergence) {
            if (x[0] == -0.5)
                assert(std::abs(gradient[0]) <= 1e-6);
            const double a = 7.5e-5, b = 5.1e-5;
            hessian[0] = 1.0 + 6.0 * (-14.5 + a + 48.0 * b) * x[0]
                + 12.0 * (-32.0 + 4.0 * a + 112.0 * b) * x[0] * x[0]
                + 20.0 * (-18.0 + 4.0 * a + 64.0 * b) * x[0] * x[0] * x[0];
        }
        return true;
    }

    bool evaluate(
        kraft::ObjectiveRequest request, const double* x, double* value,
        double* gradient, bool same_point) const noexcept
    {
        assert(!same_point);
        if (request == kraft::ObjectiveRequest::gradient) {
            ++gradient_requests;
            std::copy_n(x, width, gradient_point);
            *value = 12345.0;
            if (scenario == RetentionCase::failed_gradient) {
                gradient[0] = 0.0;
                return false;
            }
            if (scenario == RetentionCase::nonfinite_gradient) {
                gradient[0] = std::numeric_limits<double>::quiet_NaN();
                return true;
            }
            compute(x, gradient);
            return true;
        }
        assert(request == kraft::ObjectiveRequest::value);
        if (value_requests++ == 0u)
            std::copy_n(x, width, first_trial);
        unchanged_value_requests += x[0] == initial[0];
        if (scenario == RetentionCase::rejected_armijo) {
            assert(continuation[0] == 0.0);
            if (x[0] != -1.0) {
                *value = -100.0;
                return false;
            }
        }
        *value = compute(x, nullptr);
        return true;
    }
};

void print_number(double value)
{
    if (std::isnan(value))
        std::printf("NaN");
    else if (std::isinf(value))
        std::printf(value < 0.0 ? "-Infinity" : "Infinity");
    else
        std::printf("%.17g", value);
}

template<class Objective>
int run(const Objective& objective, double* coordinates,
        const runtime::RuntimeBounds& bounds, std::size_t maximum_steps)
{
    assert(runtime::runtime_kraft_detail::valid_bounds(objective.width, bounds));
    double gradient[2] = {}, accepted_coordinates[2] = {}, accepted_gradient[2] = {};
    double hessian[4] = {}, factor[4] = {}, direction[2] = {}, trial[2] = {};
    const std::size_t before = allocations;
    const auto result = curvature::refine(
        objective, coordinates, bounds, 1e-12, maximum_steps, gradient,
        accepted_coordinates, accepted_gradient, hessian, factor, direction, trial);
    const std::size_t allocated = allocations - before;
    assert(allocated == 0u);
    assert(result.function_evaluations
           == objective.curvature_requests + objective.value_requests);
    assert(result.gradient_evaluations
           == objective.curvature_requests + objective.gradient_requests);
    if (result.improved) {
        assert(result.value == objective.compute(coordinates, nullptr));
        assert(std::equal(coordinates, coordinates + objective.width, accepted_coordinates));
    }
    if (result.derivatives_current) {
        double expected_gradient[2] = {};
        objective.compute(coordinates, expected_gradient);
        assert(std::equal(gradient, gradient + objective.width, expected_gradient));
        assert(std::equal(gradient, gradient + objective.width, accepted_gradient));
    } else {
        assert(std::isinf(result.kkt_residual));
        assert(!result.converged);
        for (std::size_t index = 0u; index < objective.width; ++index)
            assert(std::isnan(gradient[index]));
    }
    std::printf("{\"coordinates\":[%.17g,%.17g],\"gradient\":[",
                coordinates[0], coordinates[1]);
    print_number(gradient[0]);
    std::printf(",");
    print_number(gradient[1]);
    std::printf("],\"value\":");
    print_number(result.value);
    std::printf(",\"kkt_residual\":");
    print_number(result.kkt_residual);
    std::printf(
        ",\"steps\":%zu,\"function_evaluations\":%zu,\"gradient_evaluations\":%zu,"
        "\"improved\":%s,\"converged\":%s,\"derivatives_current\":%s,"
        "\"value_requests\":%zu,\"gradient_requests\":%zu,"
        "\"curvature_requests\":%zu,\"unchanged_value_requests\":%zu,"
        "\"first_trial\":[%.17g,%.17g],\"last_curvature\":[%.17g,%.17g],"
        "\"gradient_point\":[%.17g,%.17g],\"allocations\":%zu}\n",
        result.steps, result.function_evaluations, result.gradient_evaluations,
        result.improved ? "true" : "false", result.converged ? "true" : "false",
        result.derivatives_current ? "true" : "false", objective.value_requests,
        objective.gradient_requests, objective.curvature_requests,
        objective.unchanged_value_requests, objective.first_trial[0], objective.first_trial[1],
        objective.last_curvature[0], objective.last_curvature[1],
        objective.gradient_point[0], objective.gradient_point[1], allocated);
    return 0;
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    Quadratic objective;
    double coordinates[2] = {};
    const double unbounded = std::numeric_limits<double>::quiet_NaN();
    double lower[2] = {unbounded, unbounded};
    double upper[2] = {unbounded, unbounded};
    std::size_t maximum_steps = 20u;

    const char* retention_cases[] = {
        "growing_gradient", "failed_hessian", "failed_gradient", "nonfinite_gradient",
        "rejected_armijo", "unrelated_convergence", "initial_failure",
    };
    for (std::size_t index = 0u; index < std::size(retention_cases); ++index) {
        if (std::strcmp(argv[1], retention_cases[index]) == 0) {
            RetentionObjective retention(static_cast<RetentionCase>(index));
            retention.continuation = coordinates;
            return run(retention, coordinates, {lower, upper}, 1u);
        }
    }

    if (std::strcmp(argv[1], "sub_ulp") == 0) {
        coordinates[0] = objective.center[0] = 1.0;
        objective.offset = 1e-20;
        objective.matrix[0] = 1e36;
    } else if (std::strcmp(argv[1], "clipped_armijo") == 0) {
        objective.center[0] = -1.0;
        lower[0] = -1e-6;
        maximum_steps = 1u;
    } else if (std::strcmp(argv[1], "clipped_unchanged") == 0) {
        objective.center[0] = -1.0;
        lower[0] = 0.0;
    } else if (std::strcmp(argv[1], "clipped_ascent") == 0
               || std::strcmp(argv[1], "clipped_zero_descent") == 0) {
        objective.width = 2u;
        const bool ascent = std::strcmp(argv[1], "clipped_ascent") == 0;
        objective.center[0] = ascent ? -0.75 : -1.0;
        objective.center[1] = ascent ? 0.5 : 1.0;
        const double matrix[4] = {2.0, 1.0, 1.0, 1.0};
        std::copy_n(matrix, 4u, objective.matrix);
        lower[0] = 0.0;
    } else if (std::strcmp(argv[1], "ordinary_newton") == 0) {
        coordinates[0] = 0.75;
        objective.center[0] = 0.25;
        objective.matrix[0] = 4.0;
    } else {
        return 2;
    }
    std::copy_n(coordinates, 2u, objective.initial);
    return run(objective, coordinates, {lower, upper}, maximum_steps);
}
