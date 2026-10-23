#pragma once
#include "math/runtime_workspace.hpp"
#include <optional>
#include <span>

namespace scivolapi::detail {
inline constexpr auto inactive_axis = std::size_t(-1);

struct IdentitySensitivityBasis {
    static constexpr bool identity = true;
    static constexpr bool mapped = false;
    constexpr std::size_t width(std::size_t ambient) const noexcept { return ambient; }
    constexpr std::size_t active_axis(std::size_t ambient) const noexcept { return ambient; }
    constexpr std::size_t ambient_axis(std::size_t active) const noexcept { return active; }
    constexpr bool valid_for(std::size_t) const noexcept { return true; }
    constexpr bool fixed_block(std::size_t begin, std::size_t end) const noexcept { return begin == end; }
    constexpr bool depends(std::size_t) const noexcept { return true; }
    constexpr std::size_t zero_factor_axis(std::size_t i) const noexcept { return i; }
    double first_product(std::size_t i,std::size_t a,double x,double unit=1) const noexcept { return i==a ? x/unit : 0; }
    double second_product(std::size_t,std::size_t,std::size_t,double) const noexcept { return 0; }
    double first_pair_product(std::size_t i,std::size_t a,std::size_t j,std::size_t b,double x,double u=1,double v=1) const noexcept { return i==a && j==b ? (x/u)/v : 0; }
};

// Borrows ObjectiveProjection's immutable ordered indices. No second partition owner.
class ActiveSensitivityBasis {
public:
    static constexpr bool identity = false;
    static constexpr bool mapped = false;
    static std::optional<ActiveSensitivityBasis> select(std::size_t ambient,
        std::span<const std::size_t> indices) noexcept {
        std::size_t bytes, adjacent, product;
        const auto address = reinterpret_cast<std::uintptr_t>(indices.data());
        if ((!indices.empty() && (!address || address % alignof(std::size_t)))
            || indices.size() > ambient
            || !vlk::experiment::runtime::checked_product(indices.size(), sizeof(std::size_t), bytes)
            || bytes > std::numeric_limits<std::uintptr_t>::max() - address
            || !vlk::experiment::runtime::checked_sum(indices.size(), 1, adjacent)
            || !vlk::experiment::runtime::checked_product(indices.size(), adjacent, product)) return {};
        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= ambient) return {};
            for (std::size_t j = 0; j < i; ++j) if (indices[i] == indices[j]) return {};
        }
        return ActiveSensitivityBasis(ambient, indices);
    }
    std::size_t width(std::size_t) const noexcept { return indices_.size(); }
    std::size_t active_axis(std::size_t ambient) const noexcept {
        for (std::size_t i = 0; i < indices_.size(); ++i) if (indices_[i] == ambient) return i;
        return inactive_axis;
    }
    std::size_t ambient_axis(std::size_t active) const noexcept { return indices_[active]; }
    bool valid_for(std::size_t ambient) const noexcept { return ambient == ambient_; }
    bool fixed_block(std::size_t begin, std::size_t end) const noexcept {
        for (auto i : indices_) if (i >= begin && i < end) return false;
        return true;
    }
    std::span<const std::size_t> indices() const noexcept { return indices_; }
    bool depends(std::size_t i) const noexcept { return active_axis(i)!=inactive_axis; }
    std::size_t zero_factor_axis(std::size_t i) const noexcept { return active_axis(i); }
    double first_product(std::size_t i,std::size_t a,double x,double unit=1) const noexcept { return ambient_axis(a)==i ? x/unit : 0; }
    double second_product(std::size_t,std::size_t,std::size_t,double) const noexcept { return 0; }
    double first_pair_product(std::size_t i,std::size_t a,std::size_t j,std::size_t b,double x,double u=1,double v=1) const noexcept { return ambient_axis(a)==i && ambient_axis(b)==j ? (x/u)/v : 0; }
private:
    friend class ObjectiveProjection;
    ActiveSensitivityBasis(std::size_t ambient, std::span<const std::size_t> indices) noexcept
        : ambient_(ambient), indices_(indices) {}
    std::size_t ambient_;
    std::span<const std::size_t> indices_;
};
} // namespace scivolapi::detail
