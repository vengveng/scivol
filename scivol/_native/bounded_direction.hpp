// SCIVOL-NATIVE: CORE; bounded proposal metric, never an information Hessian.
#pragma once
#include "active_face_refinement.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::bounded_direction {
namespace workspace = vlk::experiment::runtime;

struct Input { const double* data = nullptr; std::size_t count = 0; };
struct Output { double* data = nullptr; std::size_t count = 0; };

enum class Status { ok, invalid_input, short_buffer, overlap, numerical_failure };
enum class Method { none, zero_gradient, absolute_eigenvalue, cauchy };
enum class Fallback { none, unresolved_spectrum, eigensystem_failure, unusable_proposal };

struct Result {
    Status status = Status::invalid_input;
    Method method = Method::none;
    Fallback fallback = Fallback::none;
    double norm = 0, slope = 0, curvature = 0, quadratic_change = 0;
    // Dimensionless spectrum of H / max(abs(H)); H is not modified.
    // Counts are unavailable after eigensystem failure or the zero-G shortcut.
    double minimum_eigenvalue = 0, maximum_eigenvalue = 0, resolution = 0;
    std::size_t positive = 0, negative = 0, unresolved = 0;
    bool radius_limited = false;
};

struct Layout { std::size_t dimension = 0, square = 0, bytes = 0; };

constexpr Layout make_layout(std::size_t n) noexcept
{
    Layout layout;
    if (!n || n > static_cast<std::size_t>(CBLAS_INT_MAX)
        || !workspace::checked_product(n, n, layout.square)) return {};
    workspace::WorkspaceSizer size;
    size.add<double>(layout.square); size.add<double>(layout.square);
    for (unsigned i = 0; i < 3; ++i) size.add<double>(n);
    layout.dimension = n; layout.bytes = size.size();
    return layout.bytes ? layout : Layout{};
}

struct Range { std::uintptr_t begin = 0, end = 0; };

inline bool memory_range(const void* pointer, std::size_t bytes, Range& range) noexcept
{
    if (!pointer || !bytes) return false;
    range.begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - range.begin) return false;
    range.end = range.begin + bytes;
    return true;
}

inline bool intersects(Range a, Range b) noexcept
{ return a.begin < b.end && b.begin < a.end; }

inline Status validate_buffers(Layout layout, Input g, Input h, Output d,
    void* memory, std::size_t bytes) noexcept
{
    if (!layout.bytes || !g.data || !h.data || !d.data || !memory)
        return Status::invalid_input;
    if (g.count < layout.dimension || h.count < layout.square
        || d.count < layout.dimension || bytes < layout.bytes) return Status::short_buffer;
    if (reinterpret_cast<std::uintptr_t>(memory) % alignof(double)
        || reinterpret_cast<std::uintptr_t>(g.data) % alignof(double)
        || reinterpret_cast<std::uintptr_t>(h.data) % alignof(double)
        || reinterpret_cast<std::uintptr_t>(d.data) % alignof(double)) return Status::invalid_input;
    Range gr, hr, dr, arena;
    if (!memory_range(g.data, layout.dimension * sizeof(double), gr)
        || !memory_range(h.data, layout.square * sizeof(double), hr)
        || !memory_range(d.data, layout.dimension * sizeof(double), dr)
        || !memory_range(memory, bytes, arena)) return Status::invalid_input;
    if (intersects(dr, gr) || intersects(dr, hr) || intersects(arena, gr)
        || intersects(arena, hr) || intersects(arena, dr)) return Status::overlap;
    return Status::ok;
}

template<std::size_t Fixed>
inline std::size_t dimension(std::size_t runtime) noexcept
{
    if constexpr (Fixed) return Fixed;
    else return runtime;
}

template<std::size_t Fixed>
inline bool validate_values(std::size_t runtime, Input g, Input h,
    double& gscale, double& hscale) noexcept
{
    const auto n = dimension<Fixed>(runtime);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(g.data[i])) return false;
        gscale = std::max(gscale, std::abs(g.data[i]));
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(h.data[i*n+j])) return false;
            hscale = std::max(hscale, std::abs(h.data[i*n+j]));
        }
    }
    if (hscale) for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i+1; j < n; ++j)
        if (std::abs(h.data[i*n+j]/hscale - h.data[j*n+i]/hscale)
            > 8*n*std::numeric_limits<double>::epsilon()) return false;
    return true;
}

// min(cap, a*b/(c*d)) without constructing an overflowing unbounded length.
inline double bounded_length(double a, double b, double c, double d, double cap) noexcept
{
    int ea, eb, ec, ed, em, er;
    const double ma = std::frexp(a, &ea), mb = std::frexp(b, &eb);
    const double mc = std::frexp(c, &ec), md = std::frexp(d, &ed);
    const double m = std::frexp((ma*mb)/(mc*md), &em);
    const double r = std::frexp(cap, &er);
    const int exponent = ea+eb-ec-ed+em;
    if (exponent > er || (exponent == er && m >= r)) return cap;
    return std::scalbn(m, exponent);
}

template<std::size_t Fixed>
inline double vector_norm(std::size_t runtime, const double* v) noexcept
{
    double norm = 0;
    for (std::size_t i = 0; i < dimension<Fixed>(runtime); ++i) norm = std::hypot(norm, v[i]);
    return norm;
}

template<std::size_t Fixed>
inline bool bound_step(std::size_t runtime, double radius, double* d) noexcept
{
    const auto n = dimension<Fixed>(runtime);
    double norm = vector_norm<Fixed>(n, d);
    if (!std::isfinite(norm)) return false;
    // Correct outward rounding of a boundary step. This is a proposal bound,
    // not a change to the model or its feasible parameter set.
    for (unsigned retry = 0; norm > radius && retry < 4; ++retry) {
        const double scale = std::nextafter(radius/norm, 0.0);
        for (std::size_t i = 0; i < n; ++i) d[i] *= scale;
        norm = vector_norm<Fixed>(n, d);
    }
    return std::isfinite(norm) && norm <= radius;
}

template<std::size_t Fixed>
inline bool score_step(std::size_t runtime, Input g, Input h,
    const double* d, Result& result) noexcept
{
    const auto n = dimension<Fixed>(runtime);
    long double slope = 0, curvature = 0;
    for (std::size_t i = 0; i < n; ++i) {
        slope += static_cast<long double>(g.data[i])*d[i];
        for (std::size_t j = 0; j < n; ++j)
            curvature += static_cast<long double>(d[i])*h.data[i*n+j]*d[j];
    }
    result.norm = vector_norm<Fixed>(n, d);
    result.slope = static_cast<double>(slope);
    result.curvature = static_cast<double>(curvature);
    result.quadratic_change = static_cast<double>(slope + 0.5L*curvature);
    return std::isfinite(result.norm) && std::isfinite(result.slope)
        && std::isfinite(result.curvature) && std::isfinite(result.quadratic_change)
        && result.slope < 0 && result.quadratic_change < 0;
}

template<std::size_t Fixed>
inline bool propose_absolute_eigenvalue(std::size_t runtime, Input g, Input h,
    double gscale, double hscale, double radius, unsigned sweeps,
    double* matrix, double* vectors, double* values, double* coefficients,
    double* proposal, Result& result) noexcept
{
    const auto n = dimension<Fixed>(runtime);
    if (!hscale) { result.fallback = Fallback::unresolved_spectrum; result.unresolved = n; return false; }
    for (std::size_t j = 0; j < n; ++j) for (std::size_t i = 0; i < n; ++i)
        matrix[i+j*n] = h.data[i*n+j]/hscale;
    if (!curvature::symmetric_eigensystem(matrix, n, values, vectors, sweeps)) {
        result.fallback = Fallback::eigensystem_failure; return false;
    }
    result.minimum_eigenvalue = values[0]; result.maximum_eigenvalue = values[n-1];
    const double magnitude = std::max(std::abs(values[0]), std::abs(values[n-1]));
    result.resolution = 64*n*std::numeric_limits<double>::epsilon()*magnitude;
    for (std::size_t j = 0; j < n; ++j) {
        if (values[j] > result.resolution) ++result.positive;
        else if (values[j] < -result.resolution) ++result.negative;
        else ++result.unresolved;
    }
    // No eigenvalue flooring or pseudoinverse. Unresolved blocks use Cauchy.
    if (result.unresolved) { result.fallback = Fallback::unresolved_spectrum; return false; }
    for (std::size_t j = 0; j < n; ++j) {
        long double projected = 0;
        for (std::size_t i = 0; i < n; ++i) projected += vectors[i+j*n]*(g.data[i]/gscale);
        coefficients[j] = static_cast<double>(-projected/std::abs(values[j]));
    }
    for (std::size_t i = 0; i < n; ++i) {
        long double value = 0;
        for (std::size_t j = 0; j < n; ++j) value += vectors[i+j*n]*coefficients[j];
        proposal[i] = static_cast<double>(value);
    }
    const double norm = vector_norm<Fixed>(n, proposal);
    if (!(norm > 0) || !std::isfinite(norm)) return false;
    const double length = bounded_length(gscale, norm, hscale, 1.0, radius);
    result.radius_limited = length == radius;
    for (std::size_t i = 0; i < n; ++i) proposal[i] = (proposal[i]/norm)*length;
    return bound_step<Fixed>(n, radius, proposal) && score_step<Fixed>(n, g, h, proposal, result);
}

template<std::size_t Fixed>
inline bool propose_cauchy(std::size_t runtime, Input g, Input h,
    double gscale, double hscale, double radius, double* proposal, Result& result) noexcept
{
    const auto n = dimension<Fixed>(runtime);
    for (std::size_t i = 0; i < n; ++i) proposal[i] = -g.data[i]/gscale;
    const double norm = vector_norm<Fixed>(n, proposal);
    for (std::size_t i = 0; i < n; ++i) proposal[i] /= norm;
    long double slope = 0, curvature = 0;
    for (std::size_t i = 0; i < n; ++i) {
        slope += (g.data[i]/gscale)*proposal[i];
        if (hscale) for (std::size_t j = 0; j < n; ++j)
            curvature += proposal[i]*(h.data[i*n+j]/hscale)*proposal[j];
    }
    double length = radius;
    if (curvature > 0) length = bounded_length(gscale, static_cast<double>(-slope),
        hscale, static_cast<double>(curvature), radius);
    result.radius_limited = length == radius;
    for (std::size_t i = 0; i < n; ++i) proposal[i] *= length;
    return bound_step<Fixed>(n, radius, proposal) && score_step<Fixed>(n, g, h, proposal, result);
}

// G has n entries; H has n*n entries in row-major order. Inputs may alias each
// other. Their consumed prefixes and D's n-entry prefix must be disjoint from
// the entire declared, double-aligned arena; D must also be disjoint from G/H.
// Only ok commits D. Failures preserve D; numerical failure may modify scratch.
// This is an absolute-eigenvalue proposal with Cauchy fallback, not an exact
// trust-region solve or a feasibility/stationarity certificate. G=0 returns zero.
// Fixed=0 is arbitrary runtime dimension. Fixed>0 shares the algorithm with
// constant consumer loop extents and the same constexpr workspace layout.
template<std::size_t Fixed = 0>
inline Result propose(std::size_t runtime, Input g, Input h, double radius,
    Output d, void* memory, std::size_t bytes, unsigned maximum_sweeps = 64) noexcept
{
    Result result;
    const auto n = dimension<Fixed>(runtime);
    if ((Fixed && runtime != Fixed) || !std::isfinite(radius) || !(radius > 0)
        || !maximum_sweeps) return result;
    const auto layout = make_layout(n);
    result.status = validate_buffers(layout, g, h, d, memory, bytes);
    if (result.status != Status::ok) return result;
    double gscale = 0, hscale = 0;
    if (!validate_values<Fixed>(n, g, h, gscale, hscale)) {
        result.status = Status::invalid_input; return result;
    }
    workspace::WorkspaceArena arena(memory, bytes);
    double* matrix = arena.take<double>(layout.square);
    double* vectors = arena.take<double>(layout.square);
    double* values = arena.take<double>(n);
    double* coefficients = arena.take<double>(n);
    double* proposal = arena.take<double>(n);
    if (!proposal) { result.status = Status::short_buffer; return result; }
    if (!gscale) {
        std::fill_n(d.data, n, 0.0); result.method = Method::zero_gradient; return result;
    }
    if (propose_absolute_eigenvalue<Fixed>(n, g, h, gscale, hscale, radius,
            maximum_sweeps, matrix, vectors, values, coefficients, proposal, result)) {
        result.method = Method::absolute_eigenvalue;
    } else {
        if (result.fallback == Fallback::none) result.fallback = Fallback::unusable_proposal;
        if (!propose_cauchy<Fixed>(n, g, h, gscale, hscale, radius, proposal, result)) {
            result.status = Status::numerical_failure;
            result.norm = result.slope = result.curvature = result.quadratic_change = 0;
            result.radius_limited = false;
            return result;
        }
        result.method = Method::cauchy;
    }
    std::copy_n(proposal, n, d.data);
    return result;
}
} // namespace vlk::experiment::bounded_direction
