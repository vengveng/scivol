#include "profile_ordinary_probe.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/constants/constants.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

namespace {
bool allocation_scope = false;
std::size_t allocations = 0;
}
void* operator new(std::size_t size) {
    if (allocation_scope) ++allocations;
    if (void* p=std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }

namespace {
namespace probe = profile_ordinary_test;
using scivolapi::Status;
using Integer = boost::multiprecision::cpp_int;
using Rational = boost::multiprecision::cpp_rational;
using MP = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
constexpr double marker = -123456789.25;
std::size_t checked = 0;
void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr,"%s\n",what); std::exit(1); }
}
Rational exact(double x) {
    require(std::isfinite(x),"Nonfinite exact oracle input");
    const auto bits=std::bit_cast<std::uint64_t>(x), e=(bits>>52)&0x7ff;
    const auto sig=(bits&((std::uint64_t(1)<<52)-1)) | (e ? std::uint64_t(1)<<52 : 0);
    const int shift=e ? int(e)-1075 : -1074;
    Rational result(sig);
    if (shift>=0) result*=Integer(1)<<shift;
    else result/=Integer(1)<<-shift;
    return bits>>63 ? -result : result;
}
MP mp(const Rational& x) {
    return MP(numerator(x))/MP(denominator(x));
}
double rounded(const Rational& x) { return static_cast<double>(mp(x)); }
bool close(double actual,double expected) {
    if (expected==0 || std::abs(expected)<std::numeric_limits<double>::min()) return actual==expected;
    return std::isfinite(actual) && std::isfinite(expected)
        && std::abs(actual-expected)<=4e-14*std::abs(expected);
}
struct Fixture {
    std::size_t k;
    std::vector<double> e,d,h;
    bool ordinary = true;
};
struct Expected {
    Rational s;
    std::vector<Rational> a,b;
    Expected(const Fixture& f):a(f.k),b(f.k*f.k) {
        const auto k=f.k,p=k*(k+1)/2;
        for (std::size_t t=0;t<f.e.size();++t) {
            const auto e=exact(f.e[t]);
            s+=e*e;
            for (std::size_t i=0;i<k;++i) a[i]+=e*exact(f.d[t*k+i]);
            std::size_t u=0;
            for (std::size_t i=0;i<k;++i) for (std::size_t j=i;j<k;++j)
                b[i*k+j]+=exact(f.d[t*k+i])*exact(f.d[t*k+j])+e*exact(f.h[t*p+u++]);
        }
    }
};
struct Output { probe::Result result; std::vector<double> g,h; };
Output run(const Fixture& f,unsigned mask,bool forced=false,
    std::size_t fixed=probe::dynamic_width,std::size_t finalize_after=0) {
    probe::Input in{mask,f.k,f.e.size(),f.e,f.d,f.h,forced,fixed,finalize_after};
    const auto bytes=probe::workspace_bytes(in);
    std::vector<std::max_align_t> storage((bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
    Output out{{},std::vector<double>(f.k,marker),std::vector<double>(f.k*f.k,marker)};
    auto scratch=std::as_writable_bytes(std::span{storage}).first(bytes);
    const auto before=allocations;
    allocation_scope=true;
    out.result=probe::evaluate(in,scratch,out.g,out.h);
    allocation_scope=false;
    require(allocations==before,"Moment probe allocated inside request");
    return out;
}
void check(const Fixture& f,std::size_t finalize_after=0) {
    const Expected expected(f);
    const Rational n(f.e.size());
    const double variance=rounded(expected.s/n),sse=rounded(expected.s);
    const MP full_value=MP(f.e.size())/2*(log(2*boost::math::constants::pi<MP>())+1+log(mp(expected.s/n)));
    for (unsigned mask=1;mask<=7;++mask) for (bool forced:{false,true})
        for (auto width:{probe::dynamic_width,f.k}) {
            const auto out=run(f,mask,forced,width,finalize_after);
            const auto& r=out.result;
            require(r.status==Status::ok && r.valid==mask,"Sparse/fixed request failed");
            require(r.has_variance && close(r.variance,variance),"Exact rational variance mismatch");
            require(std::isfinite(sse) ? r.has_sse && close(r.sse,sse) : !r.has_sse,"SSE diagnostic mismatch");
            if (mask&1) {
                const double actual=double(f.e.size())/2*(std::log(2*std::acos(-1.0))+1+r.log_variance);
                const double value=static_cast<double>(full_value);
                require(std::abs(actual-value)<=3e-12+5e-14*std::abs(value),"Independent MP additive F mismatch");
            }
            for (std::size_t i=0;i<f.k;++i) {
                require(mask&2 ? close(out.g[i],rounded(n*expected.a[i]/expected.s)) : out.g[i]==marker,
                    "Gradient mismatch/unrequested output write");
                for (std::size_t j=0;j<f.k;++j) {
                    const auto u=std::min(i,j)*f.k+std::max(i,j);
                    const auto exact_h=n*(expected.b[u]*expected.s-2*expected.a[i]*expected.a[j])/(expected.s*expected.s);
                    require(mask&4 ? close(out.h[i*f.k+j],rounded(exact_h)) : out.h[i*f.k+j]==marker,
                        "Exact Schur mismatch/unrequested output write");
                }
            }
            if (f.ordinary && !forced && !finalize_after) {
                require(r.observe_normalizations==0,"Ordinary observation normalized a moment");
                const auto products=1+((mask&6) ? f.k : 0)+((mask&4) ? f.k*(f.k+1) : 0);
                require(r.ordinary_updates==f.e.size()*products,"Ordinary request did not use admitted product sums");
            }
            if (forced) require(r.ordinary_updates==0,"Forced scaled control used ordinary arithmetic");
            ++checked;
        }
}
Fixture generic(std::size_t k) {
    Fixture f{k};
    for (std::size_t t=0;t<19;++t) {
        f.e.push_back(double(int(t%7)-3)/8+0.03125);
        for (std::size_t i=0;i<k;++i) f.d.push_back(double(int((t+3*i)%13)-6)/16);
        for (std::size_t i=0;i<k*(k+1)/2;++i) f.h.push_back(double(int((2*t+i)%5)-2)/32);
    }
    return f;
}
void transitions_and_cancellation() {
    const double next=std::nextafter(1.0,2.0);
    // Product low bits survive signed accumulation, transition and Schur contraction.
    check({1,{next,1,1},{next,-next,0x1p-52},{0,0,0}});
    auto late=Fixture{1,{next,1,1,0x1p-201},{next,-next,0x1p-52,0},{0,0,0,0},false};
    check(late);
    require(run(late,7).result.ordinary_updates==12,"Late fallback did not preserve the admitted prefix");
    for (double x:{0x1p-200,0x1p200,std::nextafter(0x1p-200,0.0),std::nextafter(0x1p200,INFINITY)}) {
        const bool ordinary=x>=0x1p-200 && x<=0x1p200;
        check({1,{1,1,1},{1,-1,x},{0,0,0},ordinary});
        check({1,{1,1,1},{1,-1,0},{0,0,x},ordinary});
        check({0,{x,x},{},{},ordinary});
    }
    for (double e:{1.0,std::nextafter(1.0,2.0),std::nextafter(1.0,0.0)})
        check({1,{1,e},{-0x1p200,0},{0,0}});
    check({1,{0,1,1},{0,-0x1p513,0},{0,0,0},false});
    check({0,{0,0x1p-537,0x1p-537,0x1p-538},{},{},false});
    check(generic(2),7); // Finalize, then append through the existing scaled path.
}
void failures() {
    const auto zero=run({0,{0,0},{},{}},7).result;
    require(zero.status==Status::mathematical_domain && zero.zero_sse_unbounded && !zero.has_variance,
        "Zero SSE was floored");
    for (double bad:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        Fixture f{1,{1,0},{1,bad},{0,0},false};
        auto r=run(f,7).result;
        require(r.status==Status::ok && r.valid==1 && r.has_variance,"Zero times invalid first derivative escaped");
        require(run(f,1).result.valid==1,"Unrequested invalid derivative poisoned F");
        f.d[1]=0; f.h[1]=bad;
        r=run(f,7).result;
        require(r.status==Status::ok && r.valid==3,"Invalid second derivative poisoned F/G or passed H");
    }
    require(run({0,{0x1p-538,0x1p-538},{},{},false},7).result.status==Status::numerical_range,
        "Positive variance underflow accepted");
    require(run({0,{0x1p512},{},{},false},7).result.status==Status::numerical_range,
        "Variance overflow accepted");
    require(run(generic(2),7,false,1).result.status==Status::invalid_argument,"Wrong fixed width accepted");
}
} // namespace
int main() {
    for (auto k:{0u,1u,2u,4u,17u}) check(generic(k));
    transitions_and_cancellation();
    failures();
    std::printf("ordinary accumulator: %zu mask/width/mode oracle checks; failure and allocation gates passed\n",checked);
}
