#pragma once

#include "models/joint/joint_garch_program.hpp"
#include "models/mean/design_adapters.hpp"
#include <vector>

namespace native_transfer_006 {

// Test transport only: layout, storage and request routing. No model equations.
template<std::size_t AR, std::size_t MA, std::size_t P, std::size_t Q,
         class Density, bool Leverage = false>
struct ArmaTransport {
    inline static constexpr auto layout =
        vlk::experiment::runtime::fixed_arma_garch_layout<AR, MA, P, Q, Leverage>;
    static constexpr std::size_t K = layout.parameter_count;
    static constexpr std::size_t omega_index = layout.omega_index;
    static constexpr std::size_t alpha_start = layout.alpha_start;
    static constexpr std::size_t gamma_start = layout.gamma_start;
    static constexpr std::size_t beta_start = layout.beta_start;

    template<unsigned Mask>
    static bool request(const double* theta, const double* y, std::size_t n,
        double seed, void* memory, std::size_t bytes, double* f, double* g, double* h)
    {
        namespace rt = vlk::experiment::runtime;
        namespace mean = vlk::experiment::mean;
        rt::WorkspaceArena arena(memory, bytes);
        rt::joint_garch::Scratch scratch;
        if (!rt::joint_garch::bind_scratch(arena, layout, Mask, scratch)) return false;
        return rt::joint_garch::evaluate<Density, Leverage,
            bool(Mask & 1), bool(Mask & 2), bool(Mask & 4),
            mean::ConditionalArmaDesign<AR, true>, P, Q, K, false, MA, &layout>(
                layout, theta, y, n, seed, scratch,
                Mask & 1 ? f : nullptr, Mask & 2 ? g : nullptr, Mask & 4 ? h : nullptr);
    }

    static bool evaluate_planned(const double* theta, const double* y, std::size_t n,
        double seed, unsigned mask, double* f, double* g, double* h)
    {
        namespace joint = vlk::experiment::runtime::joint_garch;
        std::vector<std::byte> memory(joint::workspace_bytes(layout, mask));
        switch (mask) {
#define TRANSFER_REQUEST(M) case M: return request<M>(theta, y, n, seed, memory.data(), memory.size(), f, g, h)
        TRANSFER_REQUEST(1); TRANSFER_REQUEST(2); TRANSFER_REQUEST(3);
        TRANSFER_REQUEST(4); TRANSFER_REQUEST(5); TRANSFER_REQUEST(6); TRANSFER_REQUEST(7);
#undef TRANSFER_REQUEST
        default: return false;
        }
    }
};

} // namespace native_transfer_006
