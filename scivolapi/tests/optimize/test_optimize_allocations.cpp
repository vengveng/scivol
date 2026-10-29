#include "test_support.hpp"

#include <cstdlib>
#include <new>

static bool counting = false;
static std::size_t allocations = 0;

void* operator new(std::size_t bytes)
{
    if (counting) ++allocations;
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

int main()
{
    for (int scenario = 0; scenario < 4; ++scenario) {
        const bool newton = scenario % 2;
        auto p = rosenbrock();
        if (scenario >= 2) {
            p = quadratic(); p.constraint_count = 1;
            p.constraints = [](const double* x, double* c, double* j) {
                c[0] = -x[0] - x[1]; j[0] = j[1] = -1; return true;
            };
        }
        const auto original = p.objective;
        p.objective = [original](const double* x, double& f, double* g) {
            counting = true;
            return original(x, f, g);
        };
        const auto r = newton ? refine_newton(p, std::array{-1.2, 1.0}, {})
            : minimize_slsqp(p, std::array{-1.2, 1.0}, {});
        counting = false;
        assert(r.termination == Termination::converged && allocations == 0);
    }
}
