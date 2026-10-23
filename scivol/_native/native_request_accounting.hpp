// SCIVOL-NATIVE: private diagnostics; no mathematical or solver policy.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef VLK_NATIVE_REQUEST_ACCOUNTING
#define VLK_NATIVE_REQUEST_ACCOUNTING 0
#endif

namespace vlk::experiment::request_accounting {
enum class Phase : std::size_t { other, optimization, refinement, filtering, inference, count };
enum class Preparation : std::size_t {
    sample, density, coordinate_map, mean_primal, residual_chart, residual_mapper,
    parameter_row, count
};
constexpr std::size_t phases = static_cast<std::size_t>(Phase::count);
constexpr std::size_t preparations = static_cast<std::size_t>(Preparation::count);
struct Requests {
    std::uint64_t requested = 0, cache_hits = 0, failures = 0;
    std::uint64_t physical = 0, physical_failures = 0, theta_passes = 0;
};
struct Counts {
    // Mask zero is an invalid-mask bucket. Coordinates: theta=0, z=1.
    Requests requests[phases][2][8]{};
    std::uint64_t preparation[phases][preparations][3]{};
    std::uint64_t phase_entries[phases]{};
    std::uint64_t solver_requests[phases][8]{}, solver_failures[phases][8]{};
};
static_assert(std::is_trivially_copyable_v<Counts>);
static_assert(sizeof(Counts) <= 8192);
constexpr bool enabled = VLK_NATIVE_REQUEST_ACCOUNTING != 0;

#if VLK_NATIVE_REQUEST_ACCOUNTING
struct Context {
    Counts* counts = nullptr;
    Phase phase = Phase::other;
    std::size_t logical_depth = 0, physical_depth = 0;
};
// Keep independent diagnostic images from sharing a sink through symbol lookup.
#if defined(__GNUC__)
inline thread_local __attribute__((visibility("hidden"))) Context context{};
#else
inline thread_local Context context{};
#endif
inline std::size_t phase_index() noexcept { return static_cast<std::size_t>(context.phase); }
inline std::size_t mask_index(std::uint32_t mask) noexcept { return mask > 0 && mask < 8 ? mask : 0; }
inline Requests* row(std::uint32_t mask, bool transformed) noexcept {
    return context.counts ? &context.counts->requests[phase_index()][transformed][mask_index(mask)] : nullptr;
}
class Binding {
    Context previous_;
public:
    explicit Binding(Counts* counts) noexcept : previous_(context) { context = {counts}; }
    ~Binding() { context = previous_; }
    Binding(const Binding&) = delete;
    Binding& operator=(const Binding&) = delete;
};
class PhaseScope {
    Phase previous_;
public:
    explicit PhaseScope(Phase phase) noexcept : previous_(context.phase) {
        context.phase = phase;
        if (context.counts) ++context.counts->phase_entries[phase_index()];
    }
    ~PhaseScope() { context.phase = previous_; }
    PhaseScope(const PhaseScope&) = delete;
    PhaseScope& operator=(const PhaseScope&) = delete;
};
class LogicalRequest {
    Requests* row_;
    std::uint32_t mask_;
public:
    LogicalRequest(std::uint32_t mask, bool transformed) noexcept : row_(row(mask, transformed)), mask_(mask) {
        ++context.logical_depth;
        if (row_) ++row_->requested;
    }
    ~LogicalRequest() { --context.logical_depth; }
    void hit() noexcept { if (row_) ++row_->cache_hits; }
    std::int32_t finish(std::int32_t status, std::uint32_t valid) noexcept {
        if (row_ && (status != 0 || valid != mask_)) ++row_->failures;
        return status;
    }
    LogicalRequest(const LogicalRequest&) = delete;
    LogicalRequest& operator=(const LogicalRequest&) = delete;
};
class PhysicalRequest {
    Requests* row_;
    std::uint32_t mask_;
    bool logical_;
public:
    PhysicalRequest(std::uint32_t mask, bool transformed) noexcept
        : row_(context.physical_depth ? nullptr : row(mask, transformed)), mask_(mask),
          logical_(context.logical_depth == 0 && context.physical_depth == 0) {
        ++context.physical_depth;
        if (row_) { ++row_->physical; if (logical_) ++row_->requested; }
    }
    ~PhysicalRequest() { --context.physical_depth; }
    std::int32_t finish(std::int32_t status, const std::uint32_t* valid) noexcept {
        if (row_ && (status != 0 || (valid && *valid != mask_))) {
            ++row_->physical_failures;
            if (logical_) ++row_->failures;
        }
        return status;
    }
    bool finish_bool(bool valid) noexcept { return finish(valid ? 0 : 1, nullptr) == 0; }
    PhysicalRequest(const PhysicalRequest&) = delete;
    PhysicalRequest& operator=(const PhysicalRequest&) = delete;
};
class SolverRequest {
    Counts* counts_;
    std::size_t phase_, mask_;
public:
    explicit SolverRequest(std::uint32_t mask) noexcept
        : counts_(context.counts), phase_(phase_index()), mask_(mask_index(mask)) {
        if (counts_) ++counts_->solver_requests[phase_][mask_];
    }
    bool finish(bool valid) noexcept {
        if (counts_ && !valid) ++counts_->solver_failures[phase_][mask_];
        return valid;
    }
};
inline void prepare(Preparation kind, std::size_t order) noexcept {
    if (context.counts && order < 3)
        ++context.counts->preparation[phase_index()][static_cast<std::size_t>(kind)][order];
}
inline void theta_pass(std::uint32_t mask) noexcept {
    if (auto* count = row(mask, false)) ++count->theta_passes;
}
#else
struct Binding { explicit Binding(Counts*) noexcept {} };
struct PhaseScope { explicit PhaseScope(Phase) noexcept {} };
struct LogicalRequest {
    LogicalRequest(std::uint32_t, bool) noexcept {}
    void hit() noexcept {}
    std::int32_t finish(std::int32_t status, std::uint32_t) noexcept { return status; }
};
struct PhysicalRequest {
    PhysicalRequest(std::uint32_t, bool) noexcept {}
    std::int32_t finish(std::int32_t status, const std::uint32_t*) noexcept { return status; }
    bool finish_bool(bool valid) noexcept { return valid; }
};
struct SolverRequest {
    explicit SolverRequest(std::uint32_t) noexcept {}
    bool finish(bool valid) noexcept { return valid; }
};
inline void prepare(Preparation, std::size_t) noexcept {}
inline void theta_pass(std::uint32_t) noexcept {}
#endif
} // namespace vlk::experiment::request_accounting
