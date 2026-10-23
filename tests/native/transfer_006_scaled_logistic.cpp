// Standalone production-arithmetic probe; no installed extension or solver.
#include "math/power/scaled_logistic.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using vlk::experiment::positive_coordinate::ScaledLogistic;

int main()
{
    char tokens[3][128];
    while (true) {
        const int count = std::scanf("%127s %127s %127s", tokens[0], tokens[1], tokens[2]);
        if (count == EOF) return 0;
        if (count != 3) return 2;
        double point[3];
        for (int i = 0; i < 3; ++i) {
            char* end = nullptr;
            point[i] = std::strtod(tokens[i], &end);
            if (end == tokens[i] || *end != '\0' || !std::isfinite(point[i])) return 3;
        }
        const double z = point[0], a = point[1], b = point[2];
        const ScaledLogistic coordinate(z);
        const double probability = z >= 0.
            ? 1. / (1. + std::exp(-z))
            : std::exp(z) / (1. + std::exp(z));
        const double tail = std::exp(-std::abs(z));
        const double first = tail / ((1. + tail) * (1. + tail));
        const double old_first = a * first * b;
        const double outputs[] = {
            coordinate.value(a, b), coordinate.first(a, b), coordinate.second(a, b),
            coordinate.complement(a, b),
            a * probability * b, old_first, old_first * (1. - 2. * probability),
            coordinate.value(), coordinate.first(), coordinate.second(),
            coordinate.value(a), coordinate.first(a), coordinate.second(a),
        };
        for (int i = 0; i < 13; ++i)
            std::printf(i == 12 ? "%a\n" : "%a ", outputs[i]);
    }
}
