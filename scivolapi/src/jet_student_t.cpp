#include "density/student_t.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

Assembler assembler_student_t(VolatilityKind kind)
{
    return assembler_with<StudentTDensity>(kind);
}

}  // namespace scivolapi::detail
