#include "family.hpp"

#include <limits>

#include "../model_layout.hpp"
#include "families.hpp"

namespace scivolapi::detail {

bool VolatilityFamily::weights(const Layout&, const double*, int, MomentWeight*) const noexcept { return true; }

std::size_t VolatilityFamily::row(const Layout& layout, std::size_t k, const MomentWeight*, std::size_t& r,
                                  SimplexTerm* t) const noexcept
{
    r = weight_count(layout);
    t[0] = {k, 1.0, 0.0, 0.0};
    return 1;
}

std::size_t VolatilityFamily::persistence_weight(const Layout& layout, std::size_t) const noexcept
{
    return weight_count(layout);
}

void VolatilityFamily::contributions(const Layout& layout, const MomentWeight* w, const double* theta,
                                     double* v) const noexcept
{
    const std::size_t count = weight_count(layout);
    for (std::size_t k = 0; k < layout.simplex_count; ++k) {
        const std::size_t r = weighted(layout) ? persistence_weight(layout, k) : count;
        v[k] = (r < count ? w[r].value : 1.0) * theta[layout.simplex_index[k]];
    }
}

double VolatilityFamily::persistence(const Layout& layout, const double* theta) const noexcept
{
    double value = 0.0;
    return layout.persistence(theta, 0, value, nullptr) ? value : std::numeric_limits<double>::quiet_NaN();
}

void VolatilityFamily::preserve_level(const Layout& layout, double* theta, double from, double to) const
{
    theta[layout.volatility] *= (1.0 - to) / (1.0 - from);
}

const VolatilityFamily& volatility_family(VolatilityKind kind) noexcept
{
    switch (kind) {
    case VolatilityKind::garch: return garch_family();
    case VolatilityKind::gjr: return gjr_family();
    case VolatilityKind::egarch: return egarch_family();
    case VolatilityKind::aparch: return aparch_family();
    case VolatilityKind::constant: return constant_family();
    }
    return garch_family();  // unreachable: the switch covers every kind
}

}  // namespace scivolapi::detail
