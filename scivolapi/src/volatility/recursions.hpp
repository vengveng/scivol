#pragma once

#include <cstddef>
#include <span>

#include <scivolapi/model.hpp>

#include "aparch.hpp"
#include "constant.hpp"
#include "egarch.hpp"
#include "garch.hpp"

// The recursion class of each volatility family, for the code that
// instantiates it per density (the jet, jet_assemble.hpp, and the path
// runner, paths.hpp). Each entry names its class template and the measured
// specializations it compiles:
//   Recursion<Density, Coupled, P, Q, M>  the recursion, runtime orders by default
//   compiled_widths   (1,0) and (1,1) orders with a few linear-mean regressors
//   arma11            a compiled ARMA(1,1) mean with (1,1) orders
//   Branches<Density, Coupled>  a variant with explicit residual-kink branches,
//                     when the family's shock terms have kinks the jet decomposes
namespace scivolapi::detail {

inline constexpr std::size_t runtime = std::dynamic_extent;

struct GarchRecursions {
    template<class Density, bool Coupled, std::size_t P = runtime, std::size_t Q = runtime, std::size_t M = runtime>
    using Recursion = GarchVolatility<false, Coupled, P, Q, M>;
    static constexpr bool compiled_widths = true, arma11 = true, branches = false;
};

struct GjrRecursions {
    template<class Density, bool Coupled, std::size_t P = runtime, std::size_t Q = runtime, std::size_t M = runtime>
    using Recursion = GarchVolatility<true, Coupled, P, Q, M>;
    static constexpr bool compiled_widths = true, arma11 = true, branches = false;
};

struct EgarchRecursions {
    template<class Density, bool Coupled, std::size_t P = runtime, std::size_t Q = runtime, std::size_t M = runtime>
    using Recursion = EgarchVolatility<Density, Coupled, false, P, Q, M>;
    template<class Density, bool Coupled>
    using Branches = EgarchVolatility<Density, Coupled, true>;
    static constexpr bool compiled_widths = true, arma11 = false, branches = true;
};

struct AparchRecursions {
    template<class Density, bool Coupled, std::size_t P = runtime, std::size_t Q = runtime, std::size_t M = runtime>
    using Recursion = AparchVolatility<Coupled, P, Q, M>;
    static constexpr bool compiled_widths = true, arma11 = false, branches = false;
};

struct ConstantRecursions {
    template<class Density, bool Coupled, std::size_t P = runtime, std::size_t Q = runtime, std::size_t M = runtime>
    using Recursion = ConstantVolatility<Coupled>;
    static constexpr bool compiled_widths = false, arma11 = false, branches = false;
};

// Calls visit.template operator()<Recursions>() for the kind's entry.
template<class Visitor>
decltype(auto) with_recursions(VolatilityKind kind, Visitor&& visit)
{
    switch (kind) {
    case VolatilityKind::garch: return visit.template operator()<GarchRecursions>();
    case VolatilityKind::gjr: return visit.template operator()<GjrRecursions>();
    case VolatilityKind::egarch: return visit.template operator()<EgarchRecursions>();
    case VolatilityKind::aparch: return visit.template operator()<AparchRecursions>();
    case VolatilityKind::constant: return visit.template operator()<ConstantRecursions>();
    }
    return visit.template operator()<GarchRecursions>();  // unreachable: the switch covers every kind
}

}  // namespace scivolapi::detail
