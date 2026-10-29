// Least-distance projection onto {x : A x >= b} against hand-computed cases.
#include <cmath>
#include <cstdio>
#include <vector>

#include "optimize/projection.hpp"

using scivolapi::optimize::ProjectionWorkspace;
using scivolapi::optimize::project_polyhedron;

static int failures = 0;
static void check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}
static bool near(double a, double b) { return std::abs(a - b) <= 1e-12 * (1.0 + std::abs(b)); }

int main()
{
    // Box x >= 0, y >= 0 and half-space x + y <= 1, i.e. -x - y >= -1.
    const std::vector<double> a = {1, 0, 0, 1, -1, -1};
    const std::vector<double> b = {0, 0, -1};
    std::vector<double> x(2), w(3);
    ProjectionWorkspace ws;
    ws.reserve(2, 0);

    // Interior point projects to itself with zero weights.
    check(project_polyhedron(a, b, 3, 2, std::vector<double>{0.2, 0.3}, x, w, ws), "interior solve");
    check(near(x[0], 0.2) && near(x[1], 0.3), "interior point unchanged");
    check(w[0] == 0.0 && w[1] == 0.0 && w[2] == 0.0, "interior weights zero");

    // (2, 2) projects onto the face x + y = 1 at (0.5, 0.5), binding only that face.
    check(project_polyhedron(a, b, 3, 2, std::vector<double>{2.0, 2.0}, x, w, ws), "face solve");
    check(near(x[0], 0.5) && near(x[1], 0.5), "face projection");
    check(w[2] > 0.0 && w[0] == 0.0 && w[1] == 0.0, "face weight");

    // (-1, 0.5) projects onto x = 0 at (0, 0.5).
    check(project_polyhedron(a, b, 3, 2, std::vector<double>{-1.0, 0.5}, x, w, ws), "bound solve");
    check(near(x[0], 0.0) && near(x[1], 0.5), "bound projection");
    check(w[0] > 0.0 && w[2] == 0.0, "bound weight");

    // (3, -2) projects onto the vertex (1, 0).
    check(project_polyhedron(a, b, 3, 2, std::vector<double>{3.0, -2.0}, x, w, ws), "vertex solve");
    check(near(x[0], 1.0) && std::abs(x[1]) <= 1e-12, "vertex projection");

    // An empty polyhedron (x >= 1 and x <= 0) is reported.
    const std::vector<double> a2 = {1, -1};
    const std::vector<double> b2 = {1, 0};
    std::vector<double> x1(1), w2(2);
    ProjectionWorkspace ws1;
    ws1.reserve(1, 0);
    check(!project_polyhedron(a2, b2, 2, 1, std::vector<double>{0.5}, x1, w2, ws1), "empty set rejected");

    if (failures == 0) std::printf("projection: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
