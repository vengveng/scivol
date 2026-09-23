// SCIVOL-NATIVE: TEST_ONLY; high-precision Python reference consumes these jets.
#include "student_normalization.hpp"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    for (int i=1; i<argc; ++i) {
        const double nu=std::strtod(argv[i],nullptr);
        const auto jet=vlk::experiment::generated::student_normalization<true,true,true>(nu);
        std::printf("%.17g %.17g %.17g\n",jet.value,jet.first,jet.second);
    }
}
