#pragma once
#include "program.hpp"
#include "models/density_preparation.hpp"

namespace scivolapi::detail::preparation_plan {
struct Enrollment {
    Status status=Status::invalid_argument;
    unsigned mask=0;
    DensityPreparationOptions options{};
};
bool enabled(const program::Configuration&) noexcept;
Enrollment make(const program::Configuration&,unsigned density_mask,bool chart,
    const ActiveSensitivityBasis* selected=nullptr) noexcept;
}
