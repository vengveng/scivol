#include "density/student_t.hpp"
#include "density/skew_t.hpp"
#include "density/moments.hpp"
#include "density/distribution.hpp"
#include "density/exponential_moment.hpp"
#include "correlation/density.hpp"
#include <cstdio>

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
struct TailReference { double eta,u,expected[9]; };
struct SkewReference { double eta,expected[12]; };
struct ProbabilityReference { double eta,z,upper; };
#include "student_reference.inc"
int failures=0;
void check(bool ok,const char* name) {
    if(!ok) { ++failures; std::printf("FAIL %s\n",name); }
}
void close(double x,double y,double tolerance,const char* name) {
    if(!std::isfinite(x) || std::abs(x-y)>tolerance*std::max(1e-300,std::abs(y))) {
        ++failures; std::printf("FAIL %s actual=%.17g expected=%.17g relative=%.3g\n",name,x,y,std::abs((x-y)/y));
    }
}
void references() {
    for(const auto& r:tail_references) {
        const auto t=t_kernel(r.u,r.eta,2);
        const auto n=t_log_normalizer(r.eta);
        const double actual[]={t.t,t.u,t.uu,t.eta,t.ueta,t.etaeta,n.value,n.d1,n.d2};
        for(int j=0;j<9;++j) close(actual[j],r.expected[j],4e-14,"mpmath eta reference");
        for(int order=0;order<2;++order) {
            const auto sparse=t_kernel(r.u,r.eta,order);
            check(sparse.t==t.t,"sparse kernel value");
            if(order) check(sparse.u==t.u && sparse.eta==t.eta,"sparse kernel gradient");
        }
    }
}
void skew_references_test() {
    for(const auto& r:skew_references) {
        const double shape[]={r.eta,.3}; SkewTDensity::Prepared p;
        check(SkewTDensity::prepare(shape,2,p),"mpmath skew prepare");
        const double actual[]={p.a,p.b,p.a1[0],p.a1[1],p.b1[0],p.b1[1],
            p.a2[0],p.a2[1],p.a2[2],p.b2[0],p.b2[1],p.b2[2]};
        for(int j=0;j<12;++j) close(actual[j],r.expected[j],4e-14,"mpmath skew prepare reference");
    }
}
void probabilities() {
    for(const auto& r:probability_references) {
        double lower,upper,z; const double shape[]={r.eta};
        check(distribution(DensityKind::student_t,shape,r.z,lower,upper)==ExpectationStatus::ok,"mpmath eta CDF status");
        close(upper,r.upper,2e-13,"mpmath eta survival");
        check(quantile_upper(DensityKind::student_t,shape,r.upper,z)==ExpectationStatus::ok,"mpmath eta quantile status");
        close(z,r.z,2e-13,"mpmath eta quantile");
    }
}
void limits() {
    for(double eta:{0.,1e-300,1e-16}) {
        const double shapes[]={eta,.3};
        SkewTDensity::Prepared p;
        check(SkewTDensity::prepare(shapes,2,p),"skew Normal prepare");
        close(p.a,1.2/std::sqrt(2.*3.14159265358979323846),1e-14,"skew Normal a");
        close(p.b,std::sqrt(1.27-p.a*p.a),1e-14,"skew Normal b");
        MomentResult m;
        check(absolute_moment(DensityKind::student_t,shapes,2,m),"Normal t absolute moment");
        close(m.value,std::sqrt(2./3.14159265358979323846),1e-14,"Normal absolute value");
        close(m.gradient[0],-m.value/4,1e-14,"Normal absolute eta derivative");
        check(negative_second_moment(DensityKind::skew_t,shapes,2,m),"Normal skew kappa");
        check(power_moment(DensityKind::skew_t,shapes,.2,1.3,2,m),"Normal skew power moment");
        const auto c=correlation::student_constant(eta,4,2);
        close(c.first,-3.,1e-14,"Normal copula constant gradient");
        close(c.second,4.,1e-14,"Normal copula constant Hessian");
        for(double z:{-10.,-1.,0.,1.,10.}) {
            double l,r,nl,nr,q;
            check(distribution(DensityKind::student_t,shapes,z,l,r)==ExpectationStatus::ok,"Normal t CDF status");
            check(distribution(DensityKind::normal,nullptr,z,nl,nr)==ExpectationStatus::ok,"Normal CDF status");
            close(l,nl,1e-12,"Normal t lower tail"); close(r,nr,1e-12,"Normal t upper tail");
            if(l>0. && l<1.) {
                check(quantile(DensityKind::student_t,shapes,l,q)==ExpectationStatus::ok,"Normal t quantile status");
                check(std::abs(q-z)<1e-12,"Normal t quantile value");
            }
        }
    }
    const double zero[]={0.,0.}; double a,b;
    check(exponential_moment(DensityKind::normal,nullptr,.2,-.1,a)==ExpectationStatus::ok,"Normal exponential moment");
    check(exponential_moment(DensityKind::student_t,zero,.2,-.1,b)==ExpectationStatus::ok && a==b,"t Normal exponential identity");
    check(exponential_moment(DensityKind::skew_t,zero,.2,-.1,b)==ExpectationStatus::ok,"skew Normal exponential moment");
    close(a,b,1e-13,"symmetric skew Normal exponential identity");
}
}
int main() { references(); skew_references_test(); probabilities(); limits(); return failures ? 1 : 0; }
