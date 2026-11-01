#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <scivolapi/model.hpp>

namespace scivolapi::detail {

// Collects a model's immediate restrictions. Each one names the parent
// coordinates it removes (fixed at `fixed`, or zero) and the child spec whose
// parameters, matched by name, are the parent's remaining ones.
class RestrictionBuilder {
public:
    using Removed = std::vector<std::pair<std::size_t, RestrictionType>>;

    RestrictionBuilder(const Model& parent, std::vector<Restriction>& out) : parent_(parent), out_(out) {}

    [[nodiscard]] const Model& parent() const noexcept { return parent_; }
    [[nodiscard]] const ModelSpec& spec() const noexcept { return parent_.spec(); }

    Restriction& add(const ModelSpec& child, std::string name, Removed removed, double fixed = 0.0);
    Restriction& add(const ModelSpec& child, std::string name, std::size_t index, RestrictionType type,
                     double fixed = 0.0)
    {
        return add(child, std::move(name), Removed{{index, type}}, fixed);
    }

private:
    const Model& parent_;
    std::vector<Restriction>& out_;
};

// Whether two polynomials (coefficients in increasing powers) have no common
// root, by the rank of their Sylvester matrix; also false when both miss
// their highest-order term or the rank is numerically uncertain.
[[nodiscard]] bool coprime(const std::vector<double>& a, const std::vector<double>& b);

}  // namespace scivolapi::detail
