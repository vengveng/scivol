#include "density/student_t.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Assembler assembler_student_t(const ModelSpec& spec)
{
    return assembler_with<StudentTDensity>(spec);
}

PathRunner path_runner_student_t(VolatilityKind kind)
{
    return path_runner_with<StudentTDensity>(kind);
}

}  // namespace scivolapi::detail
