#include "profile_ordinary_probe.hpp"
#include "models/profile/scaled_normal_moments.hpp"
#include "models/profile/boost_profile_precision.hpp"
#include <array>

namespace profile_ordinary_test {
namespace {
using namespace scivolapi;
namespace range = detail::profile::range;
namespace wide = vlk::experiment::wide;
void checked_product_sum(range::BoostProfilePrecision::Mantissa& sum, double a, double b) noexcept {
    if (a == 0.0 || b == 0.0) return;
    sum = wide::add(wide::Real(sum),wide::multiply(wide::Real(a),wide::Real(b))).number();
}
struct CountedPrecision : range::BoostProfilePrecision {
    using Base = range::BoostProfilePrecision;
    static inline bool force_scaled = false;
    static inline bool checked_ordinary = false;
    static inline std::size_t updates = 0, normalizations = 0;
    static bool ordinary_product_input(double value) noexcept {
        return !force_scaled && Base::ordinary_product_input(value);
    }
    static void accumulate_ordinary_product(Mantissa& sum, double a, double b) noexcept {
        ++updates;
        if (checked_ordinary) checked_product_sum(sum,a,b);
        else Base::accumulate_ordinary_product(sum,a,b);
    }
    static Mantissa normalize(const Mantissa& value, int& exponent) noexcept {
        ++normalizations;
        return Base::normalize(value,exponent);
    }
};

template<unsigned Request, std::size_t Width>
Result run(const Input& input, std::span<std::byte> scratch,
    std::span<double> gradient, std::span<double> hessian) noexcept {
    using Moments = range::NormalMoments<CountedPrecision,Request,Width>;
    Result result;
    Moments moments;
    result.status = moments.bind(input.width,scratch);
    if (result.status != Status::ok) return result;
    CountedPrecision::force_scaled = input.force_scaled;
    CountedPrecision::checked_ordinary = input.checked_ordinary;
    CountedPrecision::updates = CountedPrecision::normalizations = 0;
    const auto k = input.width, packed = k*(k+1)/2;
    for (std::size_t t=0;t<input.rows;++t) {
        const auto before = CountedPrecision::normalizations;
        result.status = moments.observe(input.residual[t], k ? input.first.data()+t*k : nullptr,
            k ? input.second.data()+t*packed : nullptr);
        result.observe_normalizations += CountedPrecision::normalizations-before;
        if (result.status != Status::ok) return result;
        if (input.finalize_after == t+1 && t+1 < input.rows) {
            const auto intermediate = moments.nuisance(t+1,input.reporting_response_scale);
            if (intermediate.status != Status::ok) { result.status = intermediate.status; return result; }
        }
    }
    result.ordinary_updates = CountedPrecision::updates;
    const auto before = CountedPrecision::normalizations;
    const auto nuisance = moments.nuisance(input.rows,input.reporting_response_scale);
    result.final_normalizations = CountedPrecision::normalizations-before;
    result.status = nuisance.status;
    result.zero_sse_unbounded = nuisance.zero_sse_unbounded;
    result.has_sse = nuisance.sse.has_value();
    result.has_variance = nuisance.variance.has_value();
    if (result.has_sse) result.sse = *nuisance.sse;
    if (result.has_variance) result.variance = *nuisance.variance;
    result.log_variance = nuisance.log_variance;
    if (result.status != Status::ok) return result;
    if constexpr (Request&1u) result.valid |= 1u;
    if constexpr (Request&2u) {
        bool valid = true;
        for (std::size_t i=0;i<k;++i)
            valid = (moments.gradient(i,input.rows,gradient[i]) == Status::ok) && valid;
        if (valid) result.valid |= 2u;
    }
    if constexpr (Request&4u) {
        bool valid = true;
        for (std::size_t i=0;i<k;++i) for (std::size_t j=0;j<k;++j)
            valid = (moments.hessian(i,j,input.rows,hessian[i*k+j]) == Status::ok) && valid;
        if (valid) result.valid |= 4u;
    }
    return result;
}
template<unsigned Request>
Result select(const Input& in, std::span<std::byte> scratch,
    std::span<double> g, std::span<double> h) noexcept {
    switch (in.fixed_width) {
    case 0: return run<Request,0>(in,scratch,g,h);
    case 1: return run<Request,1>(in,scratch,g,h);
    case 2: return run<Request,2>(in,scratch,g,h);
    case 3: return run<Request,3>(in,scratch,g,h);
    case 4: return run<Request,4>(in,scratch,g,h);
    case 17: return run<Request,17>(in,scratch,g,h);
    case dynamic_width: return run<Request,dynamic_width>(in,scratch,g,h);
    default: return {};
    }
}
} // namespace

std::size_t workspace_bytes(const Input& in) noexcept {
    return range::NormalMoments<CountedPrecision,7>::requirements(in.width).bytes;
}
Result evaluate(const Input& in, std::span<std::byte> scratch,
    std::span<double> g, std::span<double> h) noexcept {
    if (in.residual.size()!=in.rows || in.first.size()!=in.rows*in.width
        || in.second.size()!=in.rows*(in.width*(in.width+1)/2)
        || g.size()<in.width || h.size()<in.width*in.width) return {};
    switch (in.request) {
#define REQUEST(R) case R: return select<R>(in,scratch,g,h);
    REQUEST(1) REQUEST(2) REQUEST(3) REQUEST(4) REQUEST(5) REQUEST(6) REQUEST(7)
#undef REQUEST
    default: return {};
    }
}
bool trace_products(std::span<const Product> products, std::span<ProductTrace> traces) noexcept {
    using Precision = range::BoostProfilePrecision;
    if (traces.size() != products.size()) return false;
    for (const auto& p : products)
        if (!Precision::ordinary_product_input(p.left) || !Precision::ordinary_product_input(p.right))
            return false;
    Precision::Mantissa checked(0), admitted(0);
    for (std::size_t i = 0; i < products.size(); ++i) {
        const auto [a,b] = products[i];
        checked_product_sum(checked,a,b);
        Precision::accumulate_ordinary_product(admitted,a,b);
        traces[i] = {{checked.backend().my_first(),checked.backend().my_second()},
            {admitted.backend().my_first(),admitted.backend().my_second()}};
    }
    return true;
}
bool entry_guards_preserve_prefix() noexcept {
    using Moments = range::NormalMoments<CountedPrecision,7,0>;
    CountedPrecision::force_scaled = CountedPrecision::checked_ordinary = false;
    CountedPrecision::updates = CountedPrecision::normalizations = 0;
    Moments moments;
    if (moments.observe(1) != Status::invalid_argument || moments.bind(0,{}) != Status::ok
        || moments.observe(1) != Status::ok) return false;
    const auto updates = CountedPrecision::updates;
    if (moments.nuisance(0).status != Status::invalid_argument
        || moments.nuisance(2).status != Status::invalid_argument) return false;
    for (double r : {0.0,-1.0,std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN()})
        if (moments.nuisance(1,r).status != Status::invalid_argument) return false;
    if (moments.observe(std::numeric_limits<double>::infinity()) != Status::numerical_range
        || moments.observe(std::numeric_limits<double>::quiet_NaN()) != Status::numerical_range
        || CountedPrecision::normalizations != 0 || CountedPrecision::updates != updates) return false;
    if (moments.observe(1) != Status::ok || CountedPrecision::updates != updates+1
        || CountedPrecision::normalizations != 0) return false;
    const auto nuisance = moments.nuisance(2,3);
    if (nuisance.status != Status::ok || nuisance.variance != 9 || nuisance.sse != 18) return false;
    if (moments.observe(1) != Status::ok || CountedPrecision::updates != updates+1) return false;
    return moments.nuisance(3,3).variance == 9;
}
bool checked_fallbacks_preserved() noexcept {
    struct Observer {
        unsigned& calls;
        unsigned& copies;
        Observer(unsigned& calls,unsigned& copies) : calls(calls),copies(copies) {}
        Observer(const Observer& other) : calls(other.calls),copies(other.copies) { ++copies; }
        void operator()(wide::Real,wide::Real) { ++calls; }
    };
    unsigned calls = 0, copies = 0;
    Observer observer(calls,copies);
    (void)wide::multiply_observed(wide::Real(0),wide::Real(1),observer);
    if (calls != 1 || copies != 1) return false;
    for (const auto [a,b] : std::array<Product,3>{{{0x1p500,0x1p401},
            {0x1p-500,0x1p-401},{0,1}}}) {
        unsigned observed = 0;
        const auto product = wide::multiply_observed(wide::Real(a),wide::Real(b),
            [&](wide::Real,wide::Real) { ++observed; });
        if (observed != 1 || product.high != a*b || product.low != 0) return false;
    }
    const wide::Real a(std::numeric_limits<double>::max());
    const auto sum = wide::add(a,a), expected = wide::Real(a.number()+a.number());
    const auto equal = [](double x,double y) { return x == y || (std::isnan(x) && std::isnan(y)); };
    return equal(sum.high,expected.high) && equal(sum.low,expected.low);
}
} // namespace profile_ordinary_test
