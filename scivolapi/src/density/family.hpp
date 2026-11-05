#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <scivolapi/model.hpp>

#include "domain.hpp"
#include "exponential_moment.hpp"

namespace scivolapi::detail {

struct Layout;
class RestrictionBuilder;
struct Kernel;
struct MomentResult;
struct PathInput;
struct PathBuffers;
enum class PathStatus;
using PathRunner = PathStatus (*)(const PathInput&, PathBuffers&);

namespace moment_detail {
enum class Moment { negative_second, absolute, power };
}
namespace distribution_detail {
struct Point;
}

// Everything generic code may ask of an innovation density beyond its
// likelihood terms (the density class). One definition per density, in
// <name>_family.cpp, which also instantiates the likelihood loops for it;
// found with density_family(kind).
class DensityFamily {
public:
    static constexpr std::size_t no_tail_index = static_cast<std::size_t>(-1);

    virtual ~DensityFamily() = default;

    // The Python component's name ("StudentT").
    [[nodiscard]] virtual const char* component() const noexcept = 0;
    // The shape parameters' domains (domain.hpp), and the constants the
    // density's definition fixes (GH members with a fixed index).
    [[nodiscard]] virtual std::span<const ShapeDomain> shapes() const noexcept = 0;
    [[nodiscard]] virtual std::span<const FixedShape> fixed_shapes() const noexcept { return {}; }
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

    // The likelihood pass for a model with this density (jet_assemble.hpp;
    // `branches`: explicit residual-kink branches where the recursion has
    // them) and its forward paths (paths.hpp).
    [[nodiscard]] virtual Kernel kernel(const ModelSpec&, bool branches) const noexcept = 0;
    [[nodiscard]] virtual PathRunner path_runner(VolatilityKind) const noexcept = 0;
    // The moments of moments.hpp with their shape (and gamma, delta) derivatives.
    [[nodiscard]] virtual bool moment(moment_detail::Moment, const double* shape, double gamma, double delta,
                                      int order, MomentResult&) const noexcept = 0;
    // M(a,b) = E[exp(a |z| + b z)] at a shape inside the domain and (a, b) != 0.
    [[nodiscard]] virtual ExpectationStatus exponential_moment(const double* shape, double a, double b,
                                                               double& value) const noexcept = 0;
    // The distribution functions' point constants at a shape inside the domain.
    [[nodiscard]] virtual bool distribution_point(const double* shape, distribution_detail::Point&) const noexcept = 0;
};

// The registry: null for an integer that names no density.
[[nodiscard]] const DensityFamily* find_density_family(DensityKind kind) noexcept;
// For a known kind.
[[nodiscard]] const DensityFamily& density_family(DensityKind kind) noexcept;

[[nodiscard]] inline bool known_density(DensityKind kind) noexcept { return find_density_family(kind) != nullptr; }

[[nodiscard]] inline std::span<const ShapeDomain> shape_domains(DensityKind kind) noexcept
{
    const DensityFamily* family = find_density_family(kind);
    return family ? family->shapes() : std::span<const ShapeDomain>{};
}

[[nodiscard]] inline std::span<const FixedShape> fixed_shapes(DensityKind kind) noexcept
{
    const DensityFamily* family = find_density_family(kind);
    return family ? family->fixed_shapes() : std::span<const FixedShape>{};
}

}  // namespace scivolapi::detail
