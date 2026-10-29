// The finalizer's open-bound decision: the face Newton step against the
// gradient step, on quadratic models with hand-computed answers.
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "finalize.hpp"

using scivolapi::detail::crosses_open_bound;
using scivolapi::detail::face_newton_step;
using scivolapi::optimize::Problem;

static int failures = 0;
static void check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

static constexpr double unbounded = std::numeric_limits<double>::quiet_NaN();
static constexpr double tiny = std::numeric_limits<double>::min();  // omega > 0, stored as the smallest normal

static Problem open_first(std::size_t d)
{
    Problem p;
    p.dimension = d;
    p.lower.assign(d, unbounded);
    p.upper.assign(d, unbounded);
    p.open_lower.assign(d, 0);
    p.open_upper.assign(d, 0);
    p.lower[0] = tiny;
    p.open_lower[0] = 1;
    return p;
}

int main()
{
    const std::vector<double> x = {1e-7, 0.3}, g = {1e-6, 0.0}, descent = {-1e-6, 0.0};
    std::vector<double> step;

    // A maximum 1e-7 from the bound with steep curvature: the gradient step
    // overshoots the bound, the Newton step lands inside at 9e-8.
    {
        const Problem p = open_first(2);
        check(crosses_open_bound(p, x, descent), "steep: the gradient step crosses");
        check(face_newton_step(p, x, g, std::vector<double>{100, 0, 0, 1}, 1e-8, step), "steep: definite");
        check(std::abs(step[0] + 1e-8) <= 1e-22 && !crosses_open_bound(p, x, step), "steep: Newton lands inside");
    }
    // The same slope with unit curvature: the model's minimizer is beyond the bound.
    {
        const Problem p = open_first(2);
        check(face_newton_step(p, x, g, std::vector<double>{1, 0, 0, 1}, 1e-8, step), "flat: definite");
        check(crosses_open_bound(p, x, step), "flat: Newton crosses");
    }
    // Negative curvature: no minimizer to locate.
    {
        const Problem p = open_first(2);
        check(!face_newton_step(p, x, g, std::vector<double>{-1, 0, 0, 1}, 1e-8, step), "indefinite is reported");
    }
    // A coordinate on a closed bound is held: free, the coupling would carry
    // the step away from the open bound; held, the step crosses it.
    {
        Problem p = open_first(2);
        p.lower[1] = 0.0;
        const std::vector<double> xb = {1e-7, 0.0}, gb = {1e-6, 5.0};
        check(face_newton_step(p, xb, gb, std::vector<double>{1, 0.5, 0.5, 1}, 1e-8, step), "held: definite");
        check(step[1] == 0.0 && std::abs(step[0] + 1e-6) <= 1e-20, "held: closed coordinate fixed");
        check(crosses_open_bound(p, xb, step), "held: Newton crosses");
    }
    // An active constraint x1 + x2 <= 1 is kept as an equality: the step stays
    // on the face, and only the open coordinate moves.
    {
        Problem p = open_first(3);
        p.constraint_count = 1;
        p.constraints = [](const double* v, double* c, double* jacobian) {
            c[0] = 1.0 - v[1] - v[2];
            jacobian[0] = 0.0; jacobian[1] = -1.0; jacobian[2] = -1.0;
            return true;
        };
        const std::vector<double> xc = {1e-7, 0.4, 0.6}, gc = {1e-6, -5.0, -5.0};
        const std::vector<double> h = {100, 0, 0, 0, 2, 0, 0, 0, 2};
        check(face_newton_step(p, xc, gc, h, 1e-8, step), "face: definite");
        check(std::abs(step[1]) <= 1e-15 && std::abs(step[2]) <= 1e-15, "face: the step stays on x1 + x2 = 1");
        check(std::abs(step[0] + 1e-8) <= 1e-20 && !crosses_open_bound(p, xc, step), "face: Newton lands inside");
    }

    if (failures == 0) std::printf("open bound step: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
