// SCIVOL-NATIVE: CORE
#pragma once
#include "runtime_workspace.hpp"
#include "third_party/scipy_slsqp/scipy_blas_defines.h"
#include <algorithm>

namespace vlk::experiment::nnls {

[[nodiscard]] constexpr std::size_t workspace_bytes(std::size_t rows,
    std::size_t columns) noexcept
{
    const auto limit=static_cast<std::size_t>(CBLAS_INT_MAX);
    std::size_t matrix,work,extra;
    if (!rows || !columns || rows>limit || columns>limit
        || !runtime::checked_product(rows,columns,matrix)
        || !runtime::checked_product(3,columns,extra)
        || !runtime::checked_sum(matrix,extra,work)
        || !runtime::checked_sum(work,1,work) || work>limit) return 0;
    runtime::WorkspaceSizer size;
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
    if (!runtime::checked_sum(n,2,rows)
        || !runtime::checked_product(n,2,bounds)
        || !runtime::checked_sum(bounds,2,bounds)
        || !runtime::checked_sum(constraints,bounds,columns)) return 0;
    return workspace_bytes(rows,columns);
}

} // namespace vlk::experiment::nnls
