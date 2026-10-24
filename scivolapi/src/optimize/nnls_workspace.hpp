#pragma once
#include "runtime_workspace.hpp"
#include "third_party/scipy_blas_defines.h"
#include <algorithm>
#include <climits>

namespace scivolapi::optimize::nnls {

[[nodiscard]] constexpr std::size_t workspace_bytes(std::size_t rows,
    std::size_t columns) noexcept
{
    const auto limit=static_cast<std::size_t>(CBLAS_INT_MAX);
    std::size_t matrix,work,extra;
    if (!rows || !columns || rows>limit || columns>limit
        || !detail::checked_product(rows,columns,matrix)
        || !detail::checked_product(3,columns,extra)
        || !detail::checked_sum(matrix,extra,work)
        || !detail::checked_sum(work,1,work) || work>limit) return 0;
    detail::WorkspaceSizer size;
    size.add<double>(columns);
    size.add<double>(matrix);
    size.add<double>(std::max(rows,columns));
    size.add<double>(work);
    size.add<CBLAS_INT>(columns);
    size.add<CBLAS_INT>(columns);
    size.add<unsigned char>(columns);
    return size.size();
}

// LDP adds one residual row; the feasibility QP can add one variable and two
// bounds. Equality elimination only reduces these conservative dimensions.
[[nodiscard]] constexpr std::size_t sqp_workspace_bytes(std::size_t n,
    std::size_t constraints) noexcept
{
    std::size_t rows,columns,bounds;
    if (!detail::checked_sum(n,2,rows)
        || !detail::checked_product(n,2,bounds)
        || !detail::checked_sum(bounds,2,bounds)
        || !detail::checked_sum(constraints,bounds,columns)) return 0;
    return workspace_bytes(rows,columns);
}

} // namespace scivolapi::optimize::nnls
