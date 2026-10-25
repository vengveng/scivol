#include "density/student_t.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_student_t(VolatilityKind kind, bool mean)
{
    return assembler_with<StudentTDensity>(kind, mean);
}

}  // namespace scivolapi::detail
