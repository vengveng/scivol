#include "density/student_t.hpp"
#include "jet_assemble.hpp"
#include "paths.hpp"

namespace scivolapi::detail {

Kernel kernel_branches_student_t(const ModelSpec& spec)
{
    return kernel_runtime<StudentTDensity, true, true>(spec);
}

Kernel kernel_student_t(const ModelSpec& spec)
{
    return kernel_with<StudentTDensity>(spec);
}

PathRunner path_runner_student_t(VolatilityKind kind)
{
    return path_runner_with<StudentTDensity>(kind);
}

}  // namespace scivolapi::detail
