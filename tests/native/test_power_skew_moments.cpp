// A local owner probe for the focused independent conformance gate.
#include "models/power/egarch_moment_policy.hpp"
#include "models/power/aparch_moment_policy.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace generated = vlk::experiment::generated;

template<std::size_t Width>
void print_jet(int order, const generated::density_moment::Jet<Width>& jet)
{
    std::printf("%d %d %.17g", order, int(jet.valid), jet.value);
    for (const double value : jet.gradient) std::printf(" %.17g", value);
    for (const double value : jet.hessian) std::printf(" %.17g", value);
    std::putchar('\n');
}

template<int Order>
void absolute_jet(const double* shape)
{
    print_jet(Order, generated::egarch_moment_detail::skew_t_finite<Order>(shape));
}

template<int Order>
void power_jet(const double* point)
{
    print_jet(Order, generated::asymmetric_power_moment<Order, generated::SkewTDensity>(
        point + 2, point[0], point[1]));
}

int main(int argc, char** argv)
{
    const bool absolute = argc == 4 && std::strcmp(argv[1], "--absolute") == 0;
    const bool power = argc == 6 && std::strcmp(argv[1], "--power") == 0;
    if (!absolute && !power) return 2;
    double point[4] {};
    for (int index = 2; index < argc; ++index) {
        char* end = nullptr;
        point[index - 2] = std::strtod(argv[index], &end);
        if (end == argv[index] || *end != '\0' || !std::isfinite(point[index - 2]))
            return 2;
    }
    if (absolute) {
        absolute_jet<0>(point);
        absolute_jet<1>(point);
        absolute_jet<2>(point);
    } else {
        power_jet<0>(point);
        power_jet<1>(point);
        power_jet<2>(point);
    }
    return 0;
}
