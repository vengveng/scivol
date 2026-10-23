#include "dependence_backend.hpp"
#include "models/dependence/program.hpp"
#include <array>
#include <limits>

namespace scivolapi::detail::dependence_backend {
namespace {
namespace math = scivolapi::detail::dependence;
namespace arena = vlk::experiment::runtime;
struct Range { std::uintptr_t begin = 0, end = 0; };
template<class T>
bool range(std::span<T> values, Range& result) noexcept {
    result = {};
    if (values.empty()) return true;
    std::size_t bytes;
    const auto address = reinterpret_cast<std::uintptr_t>(values.data());
    if (!address || address % alignof(T) || !arena::checked_product(values.size(), sizeof(T), bytes)
        || bytes > std::numeric_limits<std::uintptr_t>::max() - address) return false;
    result = {address, address + bytes};
    return true;
}
bool overlap(Range a, Range b) noexcept { return a.begin < b.end && b.begin < a.end; }
Status scratch_status(std::span<std::byte> scratch, std::size_t bytes) noexcept {
    Range checked;
    if (!range(scratch, checked) || (!scratch.empty()
        && reinterpret_cast<std::uintptr_t>(scratch.data()) % alignof(double))) return Status::invalid_argument;
    return scratch.size() < bytes ? Status::insufficient_capacity : Status::ok;
}
Status sample_shape_status(math::Layout l, SampleView sample) noexcept {
    std::size_t shock_count, state_count;
    if (!l.valid || !arena::checked_product(l.p, l.series, shock_count)
        || !arena::checked_product(l.q, l.matrix, state_count)
        || sample.residuals.size() % l.series || sample.target.size() != l.matrix
        || sample.seed.size() != l.matrix
        || (!sample.residual_history.empty() && sample.residual_history.size() != shock_count)
        || (!sample.correlation_history.empty() && sample.correlation_history.size() != state_count))
        return Status::invalid_argument;
    const auto rows = sample.residuals.size() / l.series;
    if (sample.recursion_begin > sample.first_scored || sample.first_scored >= rows) return Status::invalid_argument;
    for (auto input : {sample.residuals, sample.target, sample.seed,
                       sample.residual_history, sample.correlation_history}) {
        Range checked;
        if (!range(input, checked)) return Status::invalid_argument;
    }
    return Status::ok;
}
Status symmetric(std::span<const double> matrix, math::Layout l) noexcept {
    for (std::size_t i = 0; i < l.series; ++i) {
        for (std::size_t j = 0; j < i; ++j)
            if (matrix[i * l.series + j] != matrix[j * l.series + i]) return Status::invalid_argument;
    }
    return Status::ok;
}
Status admission(ModelConfig config, SampleView sample, std::span<const double> parameters,
    Coordinates coordinates, JetRequest request, std::span<std::byte> scratch, Buffers buffers,
    const ActiveSensitivityBasis* basis) noexcept {
    const auto needed = requirements(config, coordinates, request, basis);
    if (needed.status != Status::ok) return needed.status;
    const auto l = math::layout(config);
    const auto shape = sample_shape_status(l, sample);
    if (shape != Status::ok) return shape;
    if (parameters.size() != l.parameters) return Status::invalid_argument;
    const auto capacity = scratch_status(scratch, needed.bytes);
    if (capacity != Status::ok) return capacity;
    const auto mask = static_cast<unsigned>(request);
    const auto width = needed.parameters, dense = width * width;
    if (((mask & 1) && buffers.value.empty()) || ((mask & 2) && buffers.gradient.size() < width)
        || ((mask & 4) && buffers.hessian.size() < dense)) return Status::insufficient_capacity;
    std::array<Range, 8> inputs{};
    std::array<Range, 4> writes{};
    if (!range(parameters, inputs[0]) || !range(sample.residuals, inputs[1])
        || !range(sample.target, inputs[2]) || !range(sample.seed, inputs[3])
        || !range(sample.residual_history, inputs[4]) || !range(sample.correlation_history, inputs[5])
        || (basis && (!range(std::span(basis, 1), inputs[6]) || !range(basis->indices(), inputs[7])))
        || !range(scratch, writes[0])
        || ((mask & 1) && !range(buffers.value.first(1), writes[1]))
        || ((mask & 2) && !range(buffers.gradient.first(width), writes[2]))
        || ((mask & 4) && !range(buffers.hessian.first(dense), writes[3]))) return Status::invalid_argument;
    for (std::size_t i = 0; i < writes.size(); ++i) {
        for (auto input : inputs) if (overlap(writes[i], input)) return Status::invalid_argument;
        for (std::size_t j = 0; j < i; ++j) if (overlap(writes[i], writes[j])) return Status::invalid_argument;
    }
    return parameter_status(config, parameters);
}
template<Law Density, class Basis>
JetValidity dispatch(ModelConfig config, SampleView sample, std::span<const double> theta,
    JetRequest request, std::span<std::byte> scratch, Buffers buffers, Basis basis) noexcept {
    if (dispatch_kind(config) == 1)
        return math::dispatch_request<Density>(math::FixedLayout<1, 1, 2, Density>{},
            sample, theta, request, scratch, buffers, basis);
    return math::dispatch_request<Density>(math::RuntimeLayout{math::layout(config)},
        sample, theta, request, scratch, buffers, basis);
}
JetValidity checked_evaluate(ModelConfig config, SampleView sample,
    std::span<const double> parameters, Coordinates coordinates, JetRequest request,
    std::span<std::byte> scratch, Buffers buffers, bool prevalidated,
    const ActiveSensitivityBasis* basis) noexcept {
    auto status = admission(config, sample, parameters, coordinates, request, scratch, buffers, basis);
    if (status != Status::ok) return math::failed(request, status);
    if (!prevalidated) {
        status = sample_status(config, sample);
        if (status != Status::ok) return math::failed(request, status);
    }
    const auto selected = [&](auto axes) {
        if (config.law == Law::normal)
            return dispatch<Law::normal>(config, sample, parameters, request, scratch, buffers, axes);
        return dispatch<Law::studentt>(config, sample, parameters, request, scratch, buffers, axes);
    };
    return basis ? selected(*basis) : selected(IdentitySensitivityBasis{});
}
} // namespace

std::size_t parameter_count(ModelConfig config) noexcept {
    const auto l = math::layout(config); return l.valid ? l.parameters : 0;
}
int dispatch_kind(ModelConfig config) noexcept {
    if (!math::layout(config).valid) return -1;
    return config.fixed && config.p_corr == 1 && config.q_corr == 1 && config.series == 2 ? 1 : 0;
}
Requirements requirements(ModelConfig config, Coordinates coordinates, JetRequest request,
    const ActiveSensitivityBasis* basis) noexcept {
    Requirements result;
    const auto l = math::layout(config);
    const auto mask = static_cast<unsigned>(request);
    if (!l.valid || !mask || mask > 7) return result;
    if (basis) {
        Range checked;
        if (!range(std::span(basis, 1), checked) || !basis->valid_for(l.parameters)
            || !ActiveSensitivityBasis::select(l.parameters, basis->indices())) return result;
    }
    if (coordinates != Coordinates::theta) { result.status = Status::unsupported_operation; return result; }
    result.parameters = basis ? basis->width(l.parameters) : l.parameters;
    result.dispatch = dispatch_kind(config);
    const auto bytes = [&](auto axes) {
        return (mask & 4) ? math::workspace_bytes<2>(l, axes)
            : (mask & 2) ? math::workspace_bytes<1>(l, axes) : math::workspace_bytes<0>(l, axes);
    };
    result.bytes = basis ? bytes(*basis) : bytes(IdentitySensitivityBasis{});
    result.status = result.bytes ? Status::ok : Status::invalid_argument;
    return result;
}
Status sample_status(ModelConfig config, SampleView sample) noexcept {
    const auto l = math::layout(config);
    const auto shape = sample_shape_status(l, sample);
    if (shape != Status::ok) return shape;
    for (auto input : {sample.residuals, sample.target, sample.seed,
                       sample.residual_history, sample.correlation_history}) {
        for (double value : input) if (!math::finite(value)) return Status::invalid_argument;
    }
    for (auto matrix : {sample.target, sample.seed}) {
        const auto check = symmetric(matrix, l);
        if (check != Status::ok) return check;
    }
    for (std::size_t lag = 0; lag < l.q && !sample.correlation_history.empty(); ++lag) {
        const auto check = symmetric(sample.correlation_history.subspan(lag * l.matrix, l.matrix), l);
        if (check != Status::ok) return check;
    }
    return Status::ok;
}
Status parameter_status(ModelConfig config, std::span<const double> theta) noexcept {
    Range checked;
    if (!range(theta, checked)) return Status::invalid_argument;
    return math::parameter_domain(math::layout(config), theta);
}
JetValidity evaluate(ModelConfig c, SampleView s, std::span<const double> theta, Coordinates coordinates,
    JetRequest request, std::span<std::byte> memory, Buffers buffers,
    const ActiveSensitivityBasis* basis) noexcept {
    return checked_evaluate(c, s, theta, coordinates, request, memory, buffers, false, basis);
}
JetValidity evaluate_prevalidated(ModelConfig c, SampleView s, std::span<const double> theta,
    Coordinates coordinates, JetRequest request, std::span<std::byte> memory, Buffers buffers,
    const ActiveSensitivityBasis* basis) noexcept {
    return checked_evaluate(c, s, theta, coordinates, request, memory, buffers, true, basis);
}
} // namespace scivolapi::detail::dependence_backend
