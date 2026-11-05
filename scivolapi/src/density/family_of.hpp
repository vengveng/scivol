#pragma once

#include "../jet_assemble.hpp"
#include "../paths.hpp"
#include "family.hpp"
#include "moments_dispatch.hpp"

namespace scivolapi::detail {

// The answers that follow from the density class itself: its likelihood loops
// and its moments. Included only by the density's own <name>_family.cpp, so
// each density's loops compile in one translation unit.
template<class Density>
class DensityFamilyOf : public DensityFamily {
public:
    explicit constexpr DensityFamilyOf(const char* name) : name_(name) {}

    const char* component() const noexcept override { return name_; }

    Kernel kernel(const ModelSpec& spec, bool branches) const noexcept override
    {
        return branches ? kernel_runtime<Density, true, true>(spec) : kernel_with<Density>(spec);
    }

    PathRunner path_runner(VolatilityKind kind) const noexcept override { return path_runner_with<Density>(kind); }

    bool moment(moment_detail::Moment moment, const double* shape, double gamma, double delta, int order,
                MomentResult& out) const noexcept override
    {
        return moment_detail::evaluate<Density>(moment, shape, gamma, delta, order, out);
    }

private:
    const char* name_;
};

}  // namespace scivolapi::detail
