#pragma once
#include "../../math/runtime_workspace.hpp"

namespace vlk::experiment::mean {
struct Layout {
    std::size_t columns = 0;
    std::size_t ma_order = 0;
    std::size_t width = 0;
    std::size_t packed = 0;
    bool valid = false;
};

inline constexpr Layout layout(std::size_t columns, std::size_t q) noexcept
{
    Layout result;
    result.columns = columns;
    result.ma_order = q;
    std::size_t mean_count, adjacent, square;
    if (!runtime::checked_sum(columns, q, mean_count)
        || !runtime::checked_sum(mean_count, 1u, result.width)
        || !runtime::checked_sum(result.width, 1u, adjacent)
        || !runtime::checked_product(result.width, adjacent, square))
        return result;
    result.packed = square / 2u;
    result.valid = true;
    return result;
}

} // namespace vlk::experiment::mean
