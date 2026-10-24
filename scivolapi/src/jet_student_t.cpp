#include "density/student_t.hpp"
#include "jet_assemble.hpp"

namespace scivolapi::detail {

ThetaResult assemble_student_t(const AssembleInput& in, Work& work, double& value)
{
    return assemble<StudentTDensity>(in, work, value);
}

}  // namespace scivolapi::detail
