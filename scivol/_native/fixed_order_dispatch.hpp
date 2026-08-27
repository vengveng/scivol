// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_FIXED_ORDER_DISPATCH_HPP
#define SCIVOL_EXPERIMENT_FIXED_ORDER_DISPATCH_HPP

#include <cstddef>
#include <utility>

namespace vlk::experiment::dispatch {

template<class Result, std::size_t PAr, std::size_t QMa, std::size_t PVol,
         class Callable>
Result q_vol(
    std::size_t order,
    Callable&& callable,
    Result invalid)
{
    if (order == 1u)
        return callable.template operator()<PAr, QMa, PVol, 1u>();
    if (order == 2u)
        return callable.template operator()<PAr, QMa, PVol, 2u>();
    return invalid;
}

template<class Result, std::size_t PAr, std::size_t QMa, class Callable>
Result p_vol(
    std::size_t p_order,
    std::size_t q_order,
    Callable&& callable,
    Result invalid)
{
    if (p_order == 1u)
        return q_vol<Result, PAr, QMa, 1u>(
            q_order, std::forward<Callable>(callable), invalid);
    if (p_order == 2u)
        return q_vol<Result, PAr, QMa, 2u>(
            q_order, std::forward<Callable>(callable), invalid);
    return invalid;
}

template<class Result, std::size_t PAr, class Callable>
Result q_ma(
    std::size_t q_order,
    std::size_t p_vol_order,
    std::size_t q_vol_order,
    Callable&& callable,
    Result invalid)
{
    if (q_order == 1u)
        return p_vol<Result, PAr, 1u>(
            p_vol_order, q_vol_order,
            std::forward<Callable>(callable), invalid);
    if (q_order == 2u)
        return p_vol<Result, PAr, 2u>(
            p_vol_order, q_vol_order,
            std::forward<Callable>(callable), invalid);
    return invalid;
}

template<class Result, class Callable>
Result four_orders(
    std::size_t p_ar,
    std::size_t q_ma_order,
    std::size_t p_vol_order,
    std::size_t q_vol_order,
    Callable&& callable,
    Result invalid)
{
    if (p_ar == 1u)
        return q_ma<Result, 1u>(
            q_ma_order, p_vol_order, q_vol_order,
            std::forward<Callable>(callable), invalid);
    if (p_ar == 2u)
        return q_ma<Result, 2u>(
            q_ma_order, p_vol_order, q_vol_order,
            std::forward<Callable>(callable), invalid);
    return invalid;
}

}  // namespace vlk::experiment::dispatch

#endif
