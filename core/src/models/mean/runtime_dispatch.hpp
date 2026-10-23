#pragma once

#include "constant_mean_jet.hpp"
#include "unit_layout_dispatch.hpp"
#include "design_adapters.hpp"
#include "mean_map.hpp"
#include "../unit_map.hpp"
#include "../../selected_unit_consumer.hpp"

namespace vlk::experiment::mean {

inline constexpr auto fixed_linear_layout = layout(4, 0);
inline constexpr auto fixed_arma_layout = layout(2, 1);
inline constexpr Autoregression fixed_linear_ar {0, 0}, fixed_arma_ar {1, 1};

// Select chart metadata once; the residual engine remains ExternalLinearDesign.
template<class Function>
decltype(auto) dispatch_chart_layout(int dispatch, Autoregression ar, Function function)
{
    if (dispatch == 1 && !ar.order)
        return function.template operator()<&fixed_linear_layout, &fixed_linear_ar>();
    if (dispatch == 2)
        return function.template operator()<&fixed_arma_layout, &fixed_arma_ar>();
    return function.template operator()<nullptr, nullptr>();
}

// Transferred request planner from mean_runtime.cpp; caller checks total width.
template<const Layout* Fixed = nullptr, const Autoregression* FixedAr = nullptr,
         std::size_t Shapes = dynamic>
inline std::size_t coordinate_workspace_bytes(
    const Layout& supplied, std::size_t supplied_ar_order,
    unsigned supplied_shapes, unsigned request) noexcept
{
    static_assert((Fixed == nullptr) == (FixedAr == nullptr));
    static_assert(Fixed == nullptr || Shapes != dynamic);
    const auto& plan = arma_layout<Layout, Fixed>(supplied);
    const auto ar_order = [&] {
        if constexpr (FixedAr != nullptr) return FixedAr->order;
        else return supplied_ar_order;
    }();
    const auto shapes = extent<Shapes>(supplied_shapes);
    if (!plan.valid || request == 0 || request > 7) return 0;
    std::size_t n, square;
    if (!runtime::checked_sum(plan.width, shapes, n)
        || !runtime::checked_product(n, n, square)) return 0;
    const auto scratch_count = map_scratch_count<static_map_order<Fixed, FixedAr>>(
        std::max(ar_order, plan.ma_order), request);
    const auto kernel_bytes = workspace_bytes<Fixed>(plan, request | ((request & 4u) ? 2u : 0u));
    if (!scratch_count || !kernel_bytes) return 0;
    runtime::WorkspaceSizer size;
    size.add<double>(scratch_count);
    size.add<double>(n);
    if (request & 6u) {
        size.add<double>(square);
        size.add<double>(n);
    }
    if (request & 4u) {
        size.add<double>(square);
        size.add<double>(square);
    }
    size.add<std::byte>(kernel_bytes);
    return size.size();
}

struct Evaluation {
    Layout plan;
    Autoregression ar;
    const double* theta;
    const double* y;
    const double* x;
    std::size_t n, external_columns, first, begin;
    bool linear, intercept;
    int dispatch;
    void* workspace;
    std::size_t bytes;
    double* f;
    double* g;
    double* h;
    conditioning::History history;
    density_status::Report* density_report = nullptr;
    scivolapi::detail::DensityPreparation* persistent = nullptr;
    const scivolapi::detail::ActiveSensitivityBasis* basis = nullptr;
    const scivolapi::detail::selected_units::Point* unit_basis = nullptr;

    template<class Density, unsigned Request, bool Intercept, class Basis,class Borrowed=scivolapi::detail::NoDensityBorrow>
    bool run_basis(Basis selected,sensitivity_number_t<Basis>* numeric_g=nullptr,
        sensitivity_number_t<Basis>* numeric_h=nullptr,Borrowed borrowed={}) const noexcept
    {
        if constexpr(std::is_same_v<sensitivity_number_t<Basis>,double>) {
            if(!numeric_g)numeric_g=g;if(!numeric_h)numeric_h=h;
        }
        if (linear) {
            if (dispatch == 1)
                return evaluate<Density, Request, ExternalLinearDesign<4, Intercept>, 0, 5>(
                    fixed_linear_layout, {x, external_columns}, theta, y, n, first,
                    workspace, bytes, f, numeric_g, numeric_h, nullptr, begin, history, density_report,persistent,selected,borrowed);
            return evaluate<Density, Request>(plan,
                ExternalLinearDesign<dynamic, Intercept> {x, external_columns},
                theta, y, n, first, workspace, bytes, f, numeric_g, numeric_h, nullptr, begin, history, density_report,persistent,selected,borrowed);
        }
        if constexpr (Intercept) {
            if (dispatch == 2)
                return evaluate<Density, Request, ConditionalArmaDesign<1>, 1, 4>(
                    fixed_arma_layout, {y, 1, history}, theta, y, n, first,
                    workspace, bytes, f, numeric_g, numeric_h, nullptr, begin, history, density_report,persistent,selected,borrowed);
        }
        return evaluate<Density, Request>(plan,
            ConditionalArmaDesign<dynamic, Intercept> {y, ar.order, history},
            theta, y, n, first, workspace, bytes, f, numeric_g, numeric_h, nullptr, begin, history, density_report,persistent,selected,borrowed);
    }

    template<class Density, unsigned Request, bool Intercept>
    bool run() const noexcept {
        if (basis) return run_basis<Density,Request,Intercept>(*basis);
        return run_basis<Density,Request,Intercept>(scivolapi::detail::IdentitySensitivityBasis{});
    }

    template<class Density, unsigned Request, class Kernel, class Basis, class Borrowed>
    bool run_bound_basis(Kernel, Basis selected, sensitivity_number_t<Basis>* numeric_g,
        sensitivity_number_t<Basis>* numeric_h, Borrowed borrowed) const noexcept {
        using Design = typename Kernel::Design;
        const auto design = [&] {
            if constexpr(Kernel::linear) {
                const auto columns = Kernel::fixed ? Kernel::columns-std::size_t(Kernel::intercept) : external_columns;
                return Design{x,columns};
            } else return Design{y,extent<Kernel::p>(ar.order),history};
        }();
        return evaluate<Density,Request,Design,Kernel::q,Kernel::width>(
            Kernel::model_layout(plan),design,theta,y,n,first,workspace,bytes,f,
            numeric_g,numeric_h,nullptr,begin,history,density_report,persistent,selected,borrowed);
    }

    template<class Density, unsigned Request, class Binding, class Borrowed>
    scivolapi::JetValidity evaluate_selected_binding(Binding binding, Borrowed borrowed) const noexcept {
        using namespace scivolapi;
        using Kernel = typename Binding::kernel_type;
        constexpr bool G=bool(Request&2),H=bool(Request&4);
        const auto view=scivolapi::detail::unit_map::rebind_point(binding.owner,*unit_basis->prepared());
        if(!view)return {Status::invalid_argument,static_cast<JetRequest>(Request),0,Status::unsupported_operation,
            G?Status::invalid_argument:Status::unsupported_operation,H?Status::invalid_argument:Status::unsupported_operation};
        return scivolapi::detail::selected_units::evaluate(*unit_basis,*view,static_cast<JetRequest>(Request),
            {static_cast<std::byte*>(workspace),bytes},{{},{g,G?unit_basis->axes().width(0):0},{h,H?unit_basis->axes().width(0)*unit_basis->axes().width(0):0}},
            [&](unsigned,auto selected,auto memory,auto* ng,auto* nh) {
                if(density_report)density_report->reset();
                auto inner=*this;inner.workspace=memory.data();inner.bytes=memory.size();
                const bool ok=inner.template run_bound_basis<Density,Request>(Kernel{},selected,ng,nh,borrowed);
                return ok?Status::ok:density_report?density_report->failure():Status::numerical_range;
            });
    }

    template<class Density, bool Intercept>
    bool requests(unsigned request) const noexcept
    {
        switch (request) {
        case 1: return run<Density, 1, Intercept>();
        case 2: return run<Density, 2, Intercept>();
        case 3: return run<Density, 3, Intercept>();
        case 4: return run<Density, 4, Intercept>();
        case 5: return run<Density, 5, Intercept>();
        case 6: return run<Density, 6, Intercept>();
        case 7: return run<Density, 7, Intercept>();
        default: return false;
        }
    }

    template<class Density,unsigned Request,bool Mean>
    scivolapi::JetValidity selected_request() const noexcept {
        using namespace scivolapi;
        constexpr bool G=bool(Request&2),H=bool(Request&4);
        generated::LocalDensityAxes axes{Mean,unit_basis->axes().depends(plan.width-1),0,G,H&&Mean&&plan.ma_order!=0};
        for(std::size_t s=0;s<Density::shape_parameters;++s)if(unit_basis->axes().depends(plan.width+s))axes.shapes|=1u<<s;
        scivolapi::detail::DensityLease<Density,false,G,H,Mean> lease(persistent,theta+plan.width,n-first,axes,
            [&]{return Density::template prepare<false,G,H,Mean>(theta+plan.width,n-first,axes);});
        if(density_report)density_report->reset();
        if(!lease.accept(density_report)) {
            const auto reason=density_report ? density_report->failure():Status::numerical_range;
            return {reason,static_cast<JetRequest>(Request),0,Status::unsupported_operation,G?reason:Status::unsupported_operation,H?reason:Status::unsupported_operation};
        }
        const auto borrowed=scivolapi::detail::borrow_density(lease.get());
        const auto run_binding=[&](auto binding) -> JetValidity {
            return evaluate_selected_binding<Density,Request>(binding,borrowed);
        };
        return dispatch_selected_binding<Density>(plan,dispatch,linear,intercept,
            [&](auto binding) -> JetValidity {
                return admit_selected_binding(binding,unit_basis->plan().layout().get(),run_binding);
            });
    }
    template<class Density> scivolapi::JetValidity selected_requests(unsigned request) const noexcept {
        using namespace scivolapi;
        JetValidity result{Status::ok,static_cast<JetRequest>(request),0};
        const bool mean=!unit_basis->axes().fixed_block(0,plan.width-1);
        for(unsigned bit:{2u,4u})if(request&bit) {
            if(density_report)density_report->reset();
            const auto run=[&]<unsigned Bit>() {return mean?selected_request<Density,Bit,true>():selected_request<Density,Bit,false>();};
            const auto one=bit==2?run.template operator()<2>():run.template operator()<4>();
            result.valid_mask|=one.valid_mask;
            if(bit==2)result.g_reason=one.g_reason;else result.h_reason=one.h_reason;
            if(result.status==Status::ok && one.status!=Status::ok)result.status=one.status;
        }
        return density_status::complete_validity(result);
    }

    template<class Density>
    bool dispatch_request(unsigned request) const noexcept {
        return intercept ? requests<Density, true>(request) : requests<Density, false>(request);
    }
};

struct CoordinateEvaluation {
    Evaluation base;

    template<class Density, class Shape, const Layout* Fixed = nullptr,
             const Autoregression* FixedAr = nullptr>
    bool run_layout(unsigned request) const noexcept
    {
        static_assert((Fixed == nullptr) == (FixedAr == nullptr));
        const auto& plan = arma_layout<Layout, Fixed>(base.plan);
        const auto ar = [&] {
            if constexpr (FixedAr != nullptr) return *FixedAr;
            else return base.ar;
        }();
        const auto required = coordinate_workspace_bytes<Fixed, FixedAr, Shape::parameters>(
            plan, ar.order, Shape::parameters, request);
        if (!required || base.bytes < required || !base.workspace || !base.theta
            || ((request & 1u) && !base.f) || ((request & 2u) && !base.g)
            || ((request & 4u) && !base.h)) return false;
        const std::size_t n = plan.width + Shape::parameters;
        runtime::WorkspaceArena arena(base.workspace, base.bytes);
        double* scratch = arena.take<double>(map_scratch_count<static_map_order<Fixed, FixedAr>>(
            std::max(ar.order, plan.ma_order), request));
        double* theta = arena.take<double>(n);
        double* jacobian = (request & 6u) ? arena.take<double>(n*n) : nullptr;
        double* gradient = (request & 6u) ? arena.take<double>(n) : nullptr;
        double* hessian = (request & 4u) ? arena.take<double>(n*n) : nullptr;
        double* intermediate = (request & 4u) ? arena.take<double>(n*n) : nullptr;
        const unsigned inner_request = request | ((request & 4u) ? 2u : 0u);
        const auto inner_bytes = workspace_bytes<Fixed>(plan, inner_request);
        auto inner = base;
        inner.plan = plan;
        inner.ar = ar;
        inner.theta = theta;
        inner.workspace = arena.take<std::byte>(inner_bytes);
        inner.bytes = inner_bytes;
        inner.g = gradient;
        inner.h = hessian;
        if (!map_forward<Shape, Fixed, FixedAr>(plan, ar, base.theta, theta, jacobian, scratch)
            || !inner.dispatch_request<Density>(inner_request)) return false;
        if (!(request & 6u)) return true;
        return map_pullback<Shape, Fixed, FixedAr>(plan, ar, base.theta, theta, jacobian, gradient, hessian,
            (request & 2u) ? base.g : nullptr, (request & 4u) ? base.h : nullptr,
            scratch, intermediate);
    }

    template<class Density, class Shape>
    bool run(unsigned request) const noexcept {
        return dispatch_chart_layout(base.dispatch, base.ar,
            [&]<const Layout* Fixed, const Autoregression* FixedAr>() {
                return run_layout<Density, Shape, Fixed, FixedAr>(request);
            });
    }
};

template<class Function>
auto dispatch_shape(unsigned density, Function function)
{
    using namespace generated;
    switch (density) {
    case 0: return function.template operator()<NormalDensity, NoShapeTransform>();
    case 1: return function.template operator()<StudentTDensity, StudentTShapeTransform>();
    case 2: return function.template operator()<SkewTDensity, SkewTShapeTransform>();
    case 3: return function.template operator()<GedDensity, GedShapeTransform>();
    case 4: return function.template operator()<GeneralizedHyperbolicDensity, GhShapeTransform>();
    case 5: return function.template operator()<NormalInverseGaussianDensity, FixedLambdaGhShapeTransform>();
    case 6: return function.template operator()<HyperbolicDensity, FixedLambdaGhShapeTransform>();
    default: return function.template operator()<VarianceGammaDensity, VarianceGammaShapeTransform>();
    }
}

} // namespace vlk::experiment::mean
