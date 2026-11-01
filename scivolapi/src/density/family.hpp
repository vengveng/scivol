#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <scivolapi/model.hpp>

namespace scivolapi::detail {

struct Layout;
class RestrictionBuilder;

// What generic code may ask of an innovation density beyond its likelihood
// terms (the density class), its shape domain (domain.hpp) and its moments
// and distribution functions (moments.cpp, exponential_moment.cpp,
// distribution.cpp). One instance per density, found with density_family(kind).
class DensityFamily {
public:
    static constexpr std::size_t no_tail_index = static_cast<std::size_t>(-1);

    virtual ~DensityFamily() = default;

    // The Python component's name ("StudentT").
    [[nodiscard]] virtual const char* component() const noexcept = 0;
    // Shapes reported alongside the estimated ones (Student-t: nu = 1/eta).
    [[nodiscard]] virtual std::vector<std::pair<std::string, double>> derived(const double* /*shape*/) const
    {
        return {};
    }
    // The shape t for which E|z|^r is finite exactly when r t < 1 (a
    // polynomial tail), or no_tail_index when every moment exists.
    [[nodiscard]] virtual std::size_t tail_index() const noexcept { return no_tail_index; }
    // Whether the density's log has a kink at zero that moving residuals cross.
    [[nodiscard]] virtual bool residual_kinks() const noexcept { return false; }
    // 0 for a finite kink order, NaN where the kink is not Lipschitz.
    [[nodiscard]] virtual double residual_kink_order(const double* /*shape*/) const noexcept { return 0.0; }
    // The density's immediate restrictions to nested densities.
    virtual void restrictions(const Layout&, RestrictionBuilder&) const {}
};

[[nodiscard]] const DensityFamily& density_family(DensityKind kind) noexcept;

}  // namespace scivolapi::detail
