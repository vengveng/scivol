#pragma once

#include "mean_map.hpp"
#include "stability_domain.hpp"
#include "../../kernel_contract.hpp"
#include "../../math/local_density_status.hpp"
#include "../unit_map.hpp"

namespace scivolapi::detail::generated_ar_chart {
namespace mean = vlk::experiment::mean;
namespace runtime = vlk::experiment::runtime;
using Block = mean::Autoregression;

struct Layout {
    std::size_t parameters=0, start=0, order=0;
    bool valid=false;
    constexpr bool operator==(const Layout&) const = default;
};
constexpr Layout layout(std::size_t parameters, Block block) noexcept {
    std::size_t square;
    return {parameters,block.start,block.order,
        parameters && block.order && block.start<=parameters && block.order<=parameters-block.start
            && runtime::checked_product(parameters,parameters,square)};
}
template<std::size_t Parameters, std::size_t Start, std::size_t Order>
struct FixedLayout {
    static constexpr Layout value=layout(Parameters,{Start,Order});
    static_assert(value.valid);
};
template<const Layout* Fixed> constexpr Layout selected_layout(std::size_t parameters,Block block) noexcept {
    if constexpr (Fixed!=nullptr) return *Fixed;
    else return layout(parameters,block);
}
template<const Layout* Fixed> inline constexpr std::size_t static_order=[] {
    if constexpr (Fixed!=nullptr) return Fixed->order;
    else return mean::dynamic;
}();

// Extend only the existing narrow fixed kernel candidate. Other AR orders and
// dimensions remain directly instantiable through FixedLayout, without a table.
inline bool common_candidate(int inner_dispatch,bool intercept,std::size_t columns,
                             std::size_t order) noexcept {
    return inner_dispatch>0 && intercept && columns==3 && order==2;
}
template<std::size_t Parameters,class Function>
decltype(auto) dispatch_common(bool selected,Function&& function) noexcept {
    if (selected) return function.template operator()<&FixedLayout<Parameters,1,2>::value>();
    return function.template operator()<nullptr>();
}

// Compose R (reflection on generated AR columns) before an existing chart B.
// B must be identity on this block and independent of it elsewhere. The linear
// recursion metadata and B's fixed layouts remain unchanged: theta = B(R(z)).
inline unsigned inner_mask(unsigned mask) noexcept { return (mask & 4) ? mask | 2u : mask; }

template<const Layout* Fixed = nullptr>
inline std::size_t workspace_bytes(std::size_t supplied_n, Block supplied_block, unsigned mask,
                                   std::size_t inner_bytes) noexcept {
    const auto plan=selected_layout<Fixed>(supplied_n,supplied_block);
    const auto n=plan.parameters;
    const Block block{plan.start,plan.order};
    std::size_t square, block_square;
    const auto scratch = mean::map_scratch_count<static_order<Fixed>>(block.order, mask);
    runtime::WorkspaceSizer size;
    if (!n || !block.order || block.start > n || block.order > n-block.start
        || !mask || mask > 7 || !scratch || !inner_bytes
        || !runtime::checked_product(n,n,square)
        || !runtime::checked_product(block.order,block.order,block_square)
        || !size.add<double>(n)
        || !size.add<double>((mask & 6) ? block_square : 0)
        || !size.add<double>((mask & 6) ? n : 0)
        || !size.add<double>((mask & 4) ? square : 0)
        || !size.add<double>(scratch) || !size.add<std::byte>(inner_bytes)) return 0;
    return size.size();
}

struct Scratch {
    std::span<double> intermediate, jacobian, gradient, hessian;
    double* reflection = nullptr;
    std::span<std::byte> inner;
};
template<const Layout* Fixed = nullptr>
inline bool bind(std::size_t supplied_n, Block supplied_block, unsigned mask, std::size_t inner_bytes,
                 std::span<std::byte> memory, Scratch& result) noexcept {
    const auto plan=selected_layout<Fixed>(supplied_n,supplied_block);
    const auto n=plan.parameters;
    const Block block{plan.start,plan.order};
    const auto needed = workspace_bytes<Fixed>(n,block,mask,inner_bytes);
    if (!needed || memory.size() < needed) return false;
    runtime::WorkspaceArena arena(memory.data(),memory.size());
    const auto take = [&](std::size_t count) -> std::span<double> {
        return count ? std::span<double>{arena.take<double>(count),count} : std::span<double>{};
    };
    result.intermediate = take(n);
    result.jacobian = take((mask & 6) ? block.order*block.order : 0);
    result.gradient = take((mask & 6) ? n : 0);
    result.hessian = take((mask & 4) ? n*n : 0);
    result.reflection = arena.take<double>(mean::map_scratch_count<static_order<Fixed>>(block.order,mask));
    result.inner = {arena.take<std::byte>(inner_bytes),inner_bytes};
    return result.intermediate.data() && result.reflection && result.inner.data()
        && (!(mask & 6) || (result.jacobian.data() && result.gradient.data()))
        && (!(mask & 4) || result.hessian.data());
}
template<const Layout* Fixed = nullptr>
inline Status prepare(std::span<const double> z, Block supplied_block, Scratch& scratch) noexcept {
    const auto plan=selected_layout<Fixed>(z.size(),supplied_block);
    if (!plan.valid || z.size()!=plan.parameters) return Status::invalid_argument;
    const auto start=mean::map_extent_tag<Fixed,&Layout::start>(plan);
    const auto order=mean::map_extent_tag<Fixed,&Layout::order>(plan);
    std::copy(z.begin(),z.end(),scratch.intermediate.begin());
    if (!mean::reflection_block(z.data()+start,
        scratch.intermediate.data()+start, scratch.jacobian.data(),
        order,std::integral_constant<std::size_t,0>{},order,1.0,scratch.reflection)) return Status::numerical_range;
    // Finite coordinates can round the polynomial onto its excluded boundary.
    const auto status=mean::polynomial_status(scratch.intermediate.subspan(start,order),
        1.0,{scratch.reflection,2*std::size_t(order)},order);
    return status==Status::mathematical_domain ? Status::numerical_range : status;
}
inline bool finite(std::span<const double> values) noexcept {
    return std::all_of(values.begin(),values.end(),[](double x) { return std::isfinite(x); });
}
inline JetValidity failure(Status reason, JetRequest request) noexcept {
    const auto mask = static_cast<unsigned>(request);
    return {reason,request,0,(mask & 1) ? reason : Status::unsupported_operation,
        (mask & 2) ? reason : Status::unsupported_operation,
        (mask & 4) ? reason : Status::unsupported_operation};
}

template<const Layout* Fixed = nullptr>
inline void gradient(Block block, const Scratch& scratch, std::span<double> output) noexcept {
    const auto plan=selected_layout<Fixed>(scratch.intermediate.size(),block);
    const auto n=plan.parameters, r=plan.order, a=plan.start;
    std::copy_n(scratch.gradient.data(),n,output.data());
    for (std::size_t col=0; col<r; ++col) {
        double sum=0;
        for (std::size_t row=0; row<r; ++row)
            sum += scratch.gradient[a+row]*scratch.jacobian[row*r+col];
        output[a+col]=sum;
    }
}
template<const Layout* Fixed = nullptr>
inline bool hessian(std::span<const double> z, Block block, Scratch& scratch,
                    std::span<double> output) noexcept {
    const auto plan=selected_layout<Fixed>(z.size(),block);
    const auto n=plan.parameters, r=plan.order, a=plan.start;
    std::copy_n(scratch.hessian.data(),n*n,output.data());
    // Right multiplication changes only AR columns. Stage each left-multiplied
    // column before overwriting its AR rows, retaining every joint cross block.
    for (std::size_t row=0; row<n; ++row) for (std::size_t col=0; col<r; ++col) {
        double sum=0;
        for (std::size_t k=0; k<r; ++k)
            sum += scratch.hessian[row*n+a+k]*scratch.jacobian[k*r+col];
        output[row*n+a+col]=sum;
    }
    for (std::size_t col=0; col<n; ++col) {
        for (std::size_t row=0; row<r; ++row) {
            double sum=0;
            for (std::size_t k=0; k<r; ++k)
                sum += scratch.jacobian[k*r+row]*output[(a+k)*n+col];
            scratch.reflection[row]=sum;
        }
        for (std::size_t row=0; row<r; ++row) output[(a+row)*n+col]=scratch.reflection[row];
    }
    return runtime::reflection_add_contracted_curvature(z.data(),
        mean::map_extent_tag<Fixed,&Layout::start>(plan),mean::map_extent_tag<Fixed,&Layout::order>(plan),1.0,
        scratch.gradient.data(),mean::map_extent_tag<Fixed,&Layout::parameters>(plan),output.data(),scratch.reflection);
}
template<const Layout* Fixed = nullptr>
inline JetValidity pullback(std::span<const double> z, Block block, JetRequest request,
    Scratch& scratch, JetValidity upstream, KernelBuffers output) noexcept {
    const auto mask=static_cast<unsigned>(request);
    const auto n=selected_layout<Fixed>(z.size(),block).parameters;
    auto result=failure(Status::unsupported_operation,request);
    const unit_map::InternalJet source{{},scratch.gradient,scratch.hessian,upstream};
    const auto record=[&](unsigned bit,Status reason) {
        if (bit==1) result.f_reason=reason;
        if (bit==2) result.g_reason=reason;
        if (bit==4) result.h_reason=reason;
        if (reason==Status::ok) result.valid_mask|=bit;
    };
    if (mask & 1) record(1,source.reason(1));
    if (mask & 2) {
        auto status=source.reason(2);
        if (status==Status::ok) {
            gradient<Fixed>(block,scratch,output.gradient);
            if (!finite(output.gradient.first(n))) status=Status::numerical_range;
        }
        record(2,status);
    }
    if (mask & 4) {
        auto status=source.reason(4);
        if (status==Status::ok) status=source.reason(2);
        if (status==Status::ok && (!hessian<Fixed>(z,block,scratch,output.hessian)
            || !finite(output.hessian.first(n*n)))) status=Status::numerical_range;
        record(4,status);
    }
    return vlk::experiment::density_status::complete_validity(result);
}

template<const Layout* Fixed = nullptr,class InnerMap>
Status map(std::span<const double> z, Block supplied_block, std::span<double> theta,
    std::span<double> jacobian, std::size_t inner_bytes, std::span<std::byte> memory,
    InnerMap&& inner_map) noexcept {
    Scratch scratch;
    const auto plan=selected_layout<Fixed>(z.size(),supplied_block);
    const Block block{plan.start,plan.order};
    if (!plan.valid || z.size()!=plan.parameters) return Status::invalid_argument;
    const auto mask=jacobian.empty() ? 1u : 2u;
    if (!bind<Fixed>(plan.parameters,block,mask,inner_bytes,memory,scratch)) return Status::insufficient_capacity;
    if (const auto status=prepare<Fixed>(z,block,scratch); status!=Status::ok) return status;
    const auto status=inner_map(scratch.intermediate,scratch.inner);
    if (status!=Status::ok) return status;
    for (std::size_t row=0; row<block.order; ++row) {
        if (theta[block.start+row]!=scratch.intermediate[block.start+row]) return Status::internal_error;
        if (!jacobian.empty()) for (std::size_t col=0; col<block.order; ++col)
            jacobian[(block.start+row)*plan.parameters+block.start+col]=scratch.jacobian[row*block.order+col];
    }
    return Status::ok;
}

template<const Layout* Fixed = nullptr,class InnerEvaluate>
JetValidity evaluate(std::span<const double> z, Block block, JetRequest request,
    std::size_t inner_bytes, std::span<std::byte> memory, KernelBuffers output,
    InnerEvaluate&& inner_evaluate) noexcept {
    Scratch scratch;
    const auto mask=static_cast<unsigned>(request);
    if (z.size()!=selected_layout<Fixed>(z.size(),block).parameters) return failure(Status::invalid_argument,request);
    if (!bind<Fixed>(z.size(),block,mask,inner_bytes,memory,scratch)) return failure(Status::insufficient_capacity,request);
    if (const auto status=prepare<Fixed>(z,block,scratch); status!=Status::ok) return failure(status,request);
    const auto upstream=inner_evaluate(scratch.intermediate,static_cast<JetRequest>(inner_mask(mask)),
        scratch.inner,KernelBuffers{(mask & 1) ? output.value : std::span<double>{},scratch.gradient,scratch.hessian});
    return pullback<Fixed>(z,block,request,scratch,upstream,output);
}

template<const Layout* Fixed = nullptr,class InnerPullback>
Status chart_pullback(std::span<const double> z, Block block, JetRequest request,
    std::size_t inner_bytes, std::span<std::byte> memory, KernelBuffers output,
    InnerPullback&& inner_pullback) noexcept {
    return evaluate<Fixed>(z,block,request,inner_bytes,memory,output,
        [&](auto intermediate,JetRequest inner_request,auto work,KernelBuffers buffers) {
            const auto status=inner_pullback(intermediate,inner_request,work,buffers);
            auto result=failure(status,inner_request);
            if (status==Status::ok) result.valid_mask=static_cast<unsigned>(inner_request);
            return result;
        }).status;
}
} // namespace scivolapi::detail::generated_ar_chart
