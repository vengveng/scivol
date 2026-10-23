#pragma once

#include "../../dependence_backend.hpp"
#include "../../math/runtime_workspace.hpp"

namespace scivolapi::detail::dependence {
namespace workspace = vlk::experiment::runtime;
using dependence_backend::Law;
using dependence_backend::ModelConfig;
using dependence_backend::SampleView;

struct Layout {
    std::size_t p = 0, q = 0, series = 0, matrix = 0;
    std::size_t correlation = 0, parameters = 0, packed = 0, dense = 0;
    std::size_t beta_start = 0, shape_start = 0;
    bool student = false, valid = false;
    constexpr bool operator==(const Layout&) const = default;
};

[[nodiscard]] constexpr Layout layout(std::size_t p, std::size_t q,
    std::size_t series, Law law) noexcept
{
    Layout l;
    l.p = p; l.q = q; l.series = series;
    l.student = law == Law::studentt;
    if (!p || !q || series < 2 || (law != Law::normal && law != Law::studentt)) return l;
    l.beta_start = p;
    std::size_t next, product;
    if (!workspace::checked_sum(p, q, l.correlation)
        || !workspace::checked_sum(l.correlation, std::size_t(l.student), l.parameters)
        || !workspace::checked_sum(l.correlation, 1, next)
        || !workspace::checked_product(l.correlation, next, product)
        || !workspace::checked_product(series, series, l.matrix)
        || !workspace::checked_product(l.parameters, l.parameters, l.dense)) return l;
    l.packed = product / 2;
    l.shape_start = l.correlation;
    l.valid = true;
    return l;
}
[[nodiscard]] constexpr Layout layout(ModelConfig c) noexcept {
    return layout(c.p_corr, c.q_corr, c.series, c.law);
}

struct RuntimeLayout {
    Layout dimensions;
    [[nodiscard]] constexpr Layout get() const noexcept { return dimensions; }
};
template<std::size_t P, std::size_t Q, std::size_t Series, Law Density>
struct FixedLayout {
    static constexpr auto dimensions = layout(P, Q, Series, Density);
    static_assert(dimensions.valid);
    [[nodiscard]] static constexpr Layout get() noexcept { return dimensions; }
};

// Upper triangle by rows, with overflow already excluded by layout().
[[nodiscard]] constexpr std::size_t packed_index(std::size_t i, std::size_t j,
    std::size_t width) noexcept {
    if (i > j) { const auto temporary = i; i = j; j = temporary; }
    return i * width - i * (i + 1) / 2 + j;
}

template<class Basis>
[[nodiscard]] constexpr std::size_t correlation_width(Layout l, Basis basis) noexcept {
    return basis.width(l.parameters) - std::size_t(l.student && basis.depends(l.shape_start));
}

template<int Order, class Basis = IdentitySensitivityBasis>
[[nodiscard]] constexpr std::size_t jet_width(Layout l, Basis basis = {}) noexcept {
    const auto width = correlation_width(l, basis);
    // width <= l.correlation; layout() has already checked the triangular product.
    return 1 + (Order >= 1 ? width : 0) + (Order >= 2 ? width * (width + 1) / 2 : 0);
}

template<int Order, class Basis = IdentitySensitivityBasis>
[[nodiscard]] constexpr std::size_t workspace_bytes(Layout l, Basis basis = {}) noexcept
{
    if (!l.valid) return 0;
    std::size_t matrices, jets, vectors, count, bytes;
    // q histories, current Q, R, Cholesky; forward vector and 6 scalar jets.
    if (!workspace::checked_sum(l.q, 3, matrices)
        || !workspace::checked_product(matrices, l.matrix, jets)
        || !workspace::checked_sum(l.series, 6, vectors)
        || !workspace::checked_sum(jets, vectors, count)
        || !workspace::checked_product(count, jet_width<Order>(l, basis), count)
        || !workspace::checked_product(count, sizeof(double), bytes)) return 0;
    return bytes;
}

static_assert(FixedLayout<3, 4, 5, Law::studentt>::get()
              == layout(3, 4, 5, Law::studentt));
} // namespace scivolapi::detail::dependence
