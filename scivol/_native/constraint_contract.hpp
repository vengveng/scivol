// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_CONSTRAINT_CONTRACT_HPP
#define SCIVOL_EXPERIMENT_CONSTRAINT_CONTRACT_HPP

#include "jet_contract.hpp"

#include <cstddef>
#include <cstdint>

extern "C" {

enum vlk_constraint_request : std::uint32_t {
    VLK_CONSTRAINT_C = 1u << 0,
    VLK_CONSTRAINT_J = 1u << 1,
};

typedef std::int32_t (*vlk_constraint_callback)(
    void* context,
    const double* parameters,
    std::size_t dimension,
    std::size_t constraints,
    std::uint32_t request,
    double* values,
    double* jacobian,
    std::uint32_t* valid_mask);

}  // extern "C"

#endif
