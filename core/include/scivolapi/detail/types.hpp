#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace scivolapi {

enum class Status : std::uint32_t {
    ok, invalid_argument, insufficient_capacity, unsupported_operation,
    mathematical_domain, numerical_range, nondifferentiable,
    cancelled, allocation_failure, internal_error
};
struct Error { Status code; std::string message; };

template<class T> class Outcome {
public:
    static Outcome success(T value) { return Outcome(std::move(value)); }
    static Outcome failure(Error error) { return Outcome(std::move(error)); }
    explicit operator bool() const noexcept { return std::holds_alternative<T>(storage_); }
    const Error& error() const& { return std::get<Error>(storage_); }
    T& value() & { return std::get<T>(storage_); }
    const T& value() const& { return std::get<T>(storage_); }
    T&& value() && { return std::get<T>(std::move(storage_)); }
private:
    explicit Outcome(T value) : storage_(std::move(value)) {}
    explicit Outcome(Error error) : storage_(std::move(error)) {}
    std::variant<T, Error> storage_;
};

using LayoutId = std::uint64_t;
using ProblemId = std::uint64_t;
using ChartId = std::uint64_t;
using ParameterId = std::uint64_t;
using RowSpaceId = std::uint64_t;

enum class JetRequest : std::uint32_t { F = 1, G = 2, H = 4, FG = 3, FH = 5, GH = 6, FGH = 7 };
enum class Coordinates : std::uint32_t { theta, original_chart };
enum class ProfilePolicy : std::uint32_t { none, normal_innovation_variance };
struct DenseSymmetricView { std::span<double> elements; std::uint64_t dimension = 0; };
struct JetValidity {
    Status status = Status::invalid_argument;
    JetRequest requested = JetRequest::F;
    std::uint32_t valid_mask = 0;
    Status f_reason = Status::unsupported_operation;
    Status g_reason = Status::unsupported_operation;
    Status h_reason = Status::unsupported_operation;
    bool contains(JetRequest request) const noexcept {
        const auto mask = static_cast<std::uint32_t>(request);
        return mask != 0 && mask <= 7 && (valid_mask & mask) == mask;
    }
};

struct WorkCounts {
    std::array<std::uint64_t, 8> requested_masks{}, physical_masks{}, failed_masks{};
    std::array<std::uint64_t, 3> preparations{};
    std::uint64_t constraint_requests = 0, jacobian_requests = 0;
    std::uint64_t constraint_curvature_requests = 0, constraint_curvature_evaluations = 0;
    std::uint64_t failed_constraint_curvature_requests = 0;
    std::uint64_t cache_hits = 0, iterations = 0;
};

namespace detail {
struct Access;
struct ModelState;
struct ProblemState;
struct ObjectiveState;
struct PointState;
struct WorkspaceState;
struct JetStorageState;
}
} // namespace scivolapi
