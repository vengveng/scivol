// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_JET_CONTRACT_HPP
#define SCIVOL_EXPERIMENT_JET_CONTRACT_HPP

#include <cstddef>
#include <cstdint>
#include "../math/local_density_status.hpp"
#include <cmath>
#include <limits>
#include <span>
#include <initializer_list>

namespace vlk::experiment::conditioning {

struct RequiredLags {
    std::size_t observations = 0, innovations = 0;
};
constexpr RequiredLags required_lags(std::size_t ar, std::size_t ma, std::size_t shocks = 0) noexcept {
    return {ar, ma > shocks ? ma : shocks};
}

inline bool admitted_history(std::span<const double> values) noexcept {
    if (values.empty()) return true;
    const auto address = reinterpret_cast<std::uintptr_t>(values.data());
    return address != 0 && address % alignof(double) == 0
        && values.size() <= (std::numeric_limits<std::uintptr_t>::max() - address) / sizeof(double);
}

struct History {
    // Nearest lag first, immediately before the unchanged recursion_begin.
    std::span<const double> observations, innovations;
    std::size_t recursion_begin = 0;
    // EGARCH's omitted persistence lags have canonical log state zero. Internal
    // response units translate this fixed state; it is not a fitted variance.
    double implicit_log_variance = 0;

    bool shape(RequiredLags required) const noexcept {
        return admitted_history(observations) && admitted_history(innovations)
            && (observations.empty() || observations.size() == required.observations)
            && (innovations.empty() || innovations.size() == required.innovations);
    }
    bool overlaps(const void* pointer, std::size_t bytes) const noexcept {
        if (!bytes) return false;
        const auto address=reinterpret_cast<std::uintptr_t>(pointer);
        if (!address || bytes>std::numeric_limits<std::uintptr_t>::max()-address) return true;
        for (auto values : {observations, innovations}) {
            if (values.empty()) continue;
            const auto begin=reinterpret_cast<std::uintptr_t>(values.data());
            if (address<begin+values.size()*sizeof(double) && begin<address+bytes) return true;
        }
        return false;
    }
    bool finite() const noexcept {
        if (!std::isfinite(implicit_log_variance)) return false;
        for (auto values : {observations, innovations})
            for (double value : values) if (!std::isfinite(value)) return false;
        return true;
    }
    template<class Scalar>
    Scalar presample_log_variance(Scalar log_seed, bool repeated_seed = false) const noexcept {
        return repeated_seed || !innovations.empty() ? log_seed : Scalar(implicit_log_variance);
    }
    bool initial(std::size_t time, std::size_t lag) const noexcept {
        return time - recursion_begin <= lag;
    }
    double observation(const double* retained, std::size_t time, std::size_t lag) const noexcept {
        if (!observations.empty() && initial(time, lag))
            return observations[lag - (time - recursion_begin)];
        return retained[time - 1u - lag];
    }
    double innovation(std::size_t time, std::size_t lag) const noexcept {
        return innovations[lag - (time - recursion_begin)];
    }
    static std::size_t slot_before(std::size_t begin, std::size_t lag, std::size_t slots) noexcept {
        const auto cursor = begin % slots, offset = (lag + 1u) % slots;
        return cursor >= offset ? cursor - offset : slots - (offset - cursor);
    }
    template<class Scalar, class Convert>
    void seed_innovations(Scalar* ring, std::size_t slots, std::size_t count, Convert convert) const noexcept {
        if (innovations.empty()) return;
        for (std::size_t lag = 0; lag < count; ++lag)
            ring[slot_before(recursion_begin, lag, slots)] = convert(innovations[lag]);
    }
    template<class Scalar>
    void seed_innovations(Scalar* ring, std::size_t slots, std::size_t count) const noexcept {
        seed_innovations(ring, slots, count, [](double value) { return Scalar(value); });
    }
};

} // namespace vlk::experiment::conditioning

// The extracted vlk ABI is private. Only scivolapi's public interface is exported.
#define VLK_EXPORT

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
    VLK_JET_MATHEMATICAL_DOMAIN_ERROR = 4,
    VLK_JET_UNCERTIFIED_DOMAIN = 5,
    VLK_JET_NONDIFFERENTIABLE = 6,
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

namespace vlk::experiment::density_status {
inline std::int32_t native_failure(const Report& report) noexcept {
    if (report.code==Code::nondifferentiable) return VLK_JET_NONDIFFERENTIABLE;
    if (report.code==Code::invalid_input) return VLK_JET_INVALID_ARGUMENT;
    return VLK_JET_DOMAIN_ERROR;
}
}

#endif
