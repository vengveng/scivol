#include "profile_ordinary_probe.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/constants/constants.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
namespace probe = profile_ordinary_test;
using scivolapi::Status;
using MP = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
constexpr double marker = -123456789.25;
std::size_t prefixes = 0, requests = 0;
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr,"%s\n",message); std::exit(1); }
}
bool same(double a,double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}
void trace(const std::vector<probe::Product>& products,bool exact_prefixes = false) {
    std::vector<probe::ProductTrace> output(products.size());
    require(probe::trace_products(products,output),"Admitted input rejected by trace probe");
    MP exact = 0;
    for (std::size_t i = 0; i < products.size(); ++i) {
        const auto& p = output[i];
        require(same(p.checked.high,p.admitted.high) && same(p.checked.low,p.admitted.low),
            "Admitted update changed a DD limb, including signed zero");
        if (exact_prefixes) {
            exact += MP(products[i].left)*MP(products[i].right);
            require(MP(p.admitted.high)+MP(p.admitted.low) == exact,
                "Exactly DD-representable prefix differs from independent MP product sum");
        }
        ++prefixes;
    }
}
void arithmetic_parity() {
    const double next = std::nextafter(1.0,2.0);
    trace({{next,next},{-next,1},{-0x1p-52,1},{-0x1p-104,1},{-0.0,1},{1,-0.0}},true);
    trace({{1,0x1p53},{1,1},{1,-0x1p53}},true);
    trace({{0x1p200,0x1p200},{-0x1p200,0x1p200},
        {0x1p-200,0x1p-200},{-0x1p-200,0x1p-200}},true);
    const std::array factors{0.0,-0.0,1.0,-1.0,next,-next,
        0x1p-200,-0x1p-200,std::nextafter(0x1p-200,1.0),
        0x1p200,-0x1p200,std::nextafter(0x1p200,0.0),0x1p-99,0x1p99};
    std::vector<probe::Product> products;
    for (double a : factors) for (double b : factors) {
        products.push_back({a,b});
        products.push_back({-a,b});
    }
    trace(products);
    products.clear();
    for (std::size_t i = 0; i < 4096; ++i)
        products.push_back({i%3 ? next : -1.0,i%5 ? 0x1p200 : 0x1p-200});
    trace(products);
    // Admission must inspect the other factor even when multiplication by zero
    // would let the admitted arithmetic return without touching it.
    for (double bad : {std::nextafter(0x1p-200,0.0),std::nextafter(0x1p200,INFINITY),
            std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        for (const auto p : {probe::Product{0,bad},probe::Product{bad,-0.0}}) {
            const std::array input{p};
            std::array<probe::ProductTrace,1> output{{{{marker,marker},{marker,marker}}}};
            require(!probe::trace_products(input,output),"Invalid zero-product factor admitted");
            require(output[0].admitted.high == marker && output[0].admitted.low == marker,
                "Admission failed after mutating the prefix");
        }
    }
    require(probe::checked_fallbacks_preserved(),"Generic checked wide fallback changed");
    require(probe::entry_guards_preserve_prefix(),"Invalid entry finalized or mutated the admitted prefix");
}
struct Fixture {
    std::size_t k;
    std::vector<double> residual,first,second;
    double reporting_response_scale = 1;
};
enum class Accumulation { admitted, checked, scaled };
struct Output { probe::Result result; std::vector<double> g,h; };
Output run(const Fixture& f,unsigned mask,Accumulation mode,std::size_t width,
    std::size_t finalize_after = 0) {
    probe::Input input{mask,f.k,f.residual.size(),f.residual,f.first,f.second,
        mode == Accumulation::scaled,width,finalize_after,
        mode == Accumulation::checked,f.reporting_response_scale};
    const auto bytes = probe::workspace_bytes(input);
    std::vector<std::max_align_t> storage((bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
    Output output{{},std::vector<double>(f.k,marker),std::vector<double>(f.k*f.k,marker)};
    output.result = probe::evaluate(input,std::as_writable_bytes(std::span{storage}).first(bytes),
        output.g,output.h);
    return output;
}
void compare_requests(const Fixture& f,std::size_t finalize_after = 0) {
    for (unsigned mask = 1; mask <= 7; ++mask) for (auto width : {probe::dynamic_width,f.k}) {
        const auto a = run(f,mask,Accumulation::admitted,width,finalize_after);
        const auto c = run(f,mask,Accumulation::checked,width,finalize_after);
        const auto& x = a.result;
        const auto& y = c.result;
        require(x.status == y.status && x.valid == y.valid && x.has_sse == y.has_sse
            && x.has_variance == y.has_variance && x.zero_sse_unbounded == y.zero_sse_unbounded,
            "Admitted request changed validity or nuisance availability");
        require(same(x.sse,y.sse) && same(x.variance,y.variance) && same(x.log_variance,y.log_variance)
            && x.ordinary_updates == y.ordinary_updates && x.observe_normalizations == y.observe_normalizations
            && x.final_normalizations == y.final_normalizations,"Admitted nuisance or path differs from checked control");
        for (std::size_t i = 0; i < a.g.size(); ++i)
            require(same(a.g[i],c.g[i]),"Admitted G differs from checked-DD G");
        for (std::size_t i = 0; i < a.h.size(); ++i)
            require(same(a.h[i],c.h[i]),"Admitted H differs from checked-DD H");
        ++requests;
    }
}
void exact_cancellation() {
    // These are linear residuals at beta=0 with design=-first. A double-only
    // accumulator loses G=1 in the first case and H=-2^148 in the second.
    const std::array fixtures{Fixture{1,{1,1,1},{0x1p53,1,-0x1p53},{0,0,0}},
        Fixture{1,{1,1},{0x1p100,0x1p47},{0,0}}};
    for (std::size_t i = 0; i < fixtures.size(); ++i) {
        const auto& f = fixtures[i];
        compare_requests(f);
        for (unsigned mask = 1; mask <= 7; ++mask)
            for (auto mode : {Accumulation::admitted,Accumulation::checked,Accumulation::scaled})
                for (auto width : {probe::dynamic_width,f.k}) {
                    const auto out = run(f,mask,mode,width);
                    require(out.result.status == Status::ok && out.result.valid == mask
                        && out.result.variance == 1 && out.result.sse == double(f.residual.size()),
                        "Exact ordinary cancellation request failed");
                    require(mask&2 ? out.g[0] == (i ? 0x1p100 : 1.0) : out.g[0] == marker,
                        "Lost ordinary-band gradient low bits or wrote unrequested G");
                    require(mask&4 ? out.h[0] == (i ? -0x1p148 : 0x1p107) : out.h[0] == marker,
                        "Lost exact Schur cancellation or wrote unrequested H");
                    if (mask&1) {
                        const double expected = static_cast<double>(MP(f.residual.size())/2
                            *(log(2*boost::math::constants::pi<MP>())+1));
                        const double actual = double(f.residual.size())/2
                            *(std::log(2*std::acos(-1.0))+1+out.result.log_variance);
                        require(std::abs(actual-expected) <= 3e-12+5e-14*std::abs(expected),
                            "Independent additive F gate failed");
                    }
                    ++requests;
                }
    }
}
void requests_and_units() {
    for (std::size_t k : {0u,1u,2u,4u,17u}) {
        Fixture f{k};
        for (std::size_t t = 0; t < 19; ++t) {
            f.residual.push_back(double(int(t%7)-3)/8+0.03125);
            for (std::size_t i = 0; i < k; ++i) f.first.push_back(double(int((t+3*i)%13)-6)/16);
            for (std::size_t i = 0; i < k*(k+1)/2; ++i) f.second.push_back(double(int((2*t+i)%5)-2)/32);
        }
        compare_requests(f);
        compare_requests(f,7);
        f.reporting_response_scale = 3;
        compare_requests(f);
    }
    const double next = std::nextafter(1.0,2.0);
    compare_requests({1,{next,1,1,0x1p-201},{next,-next,0x1p-52,0},{0,0,0,0}});
    for (double bad : {std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        compare_requests({1,{1,0},{1,bad},{0,0}});
        compare_requests({1,{1,0},{1,0},{0,bad}});
    }
    for (double r : {.5,2.0}) {
        Fixture f{0,{0x1p-537/r,0x1p-538/r},{},{},r};
        compare_requests(f);
        for (unsigned mask = 1; mask <= 7; ++mask) {
            const auto out = run(f,mask,Accumulation::admitted,0);
            require(out.result.status == Status::ok && out.result.valid == mask
                && out.result.variance == 0x1p-1074 && out.result.sse == 0x1p-1074,
                "Reporting nuisance was narrowed before unit restoration");
        }
    }
    const Fixture overflow{0,{1,0,0,0},{},{},0x1p512};
    compare_requests(overflow);
    for (unsigned mask = 1; mask <= 7; ++mask) {
        const auto out = run(overflow,mask,Accumulation::admitted,0);
        require(out.result.status == Status::ok && out.result.valid == mask
            && !out.result.has_sse && out.result.variance == 0x1p1022,
            "Overflowing reporting SSE invalidated representable reporting variance");
    }
    compare_requests({0,{0,0},{},{}});
    compare_requests({0,{0x1p512},{},{}});
    compare_requests({0,{0x1p-538,0x1p-538},{},{}});
}
} // namespace
int main() {
    arithmetic_parity();
    exact_cancellation();
    requests_and_units();
    std::printf("admitted DD: %zu limb prefixes, %zu request comparisons/oracles; entry and units gates passed\n",
        prefixes,requests);
}
