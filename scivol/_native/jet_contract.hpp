// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_JET_CONTRACT_HPP
#define SCIVOL_EXPERIMENT_JET_CONTRACT_HPP

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#  define VLK_EXPORT __declspec(dllexport)
#else
#  define VLK_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

enum vlk_jet_request : std::uint32_t {
    VLK_JET_F = 1u << 0,
    VLK_JET_G = 1u << 1,
    VLK_JET_H = 1u << 2,
};

enum vlk_jet_status : std::int32_t {
    VLK_JET_OK = 0,
    VLK_JET_INVALID_ARGUMENT = 1,
    VLK_JET_INSUFFICIENT_WORKSPACE = 2,
    VLK_JET_DOMAIN_ERROR = 3,
};

typedef std::int32_t (*vlk_jet_callback)(
    void* context,
    const double* parameters,
    std::size_t dimension,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

}  // extern "C"

#endif
