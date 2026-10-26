#include "density/moments.hpp"
#include "density/gh.hpp"
#include "density/skew_t.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {
bool watching = false;
std::size_t allocations = 0;
}
void* operator new(std::size_t size) {
    if (watching) ++allocations;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
constexpr const char* names[] = {"Normal", "Student-t", "Skew-t", "GED", "GH", "NIG", "Hyperbolic", "VG"};
int failures = 0;
void check(bool ok, const char* message);
struct Reference {
    int kind, moment;
    double shape[3], gamma, delta;
    double expected[21];
};
#include "moments_reference.inc"

double worst[8][3][3]{};

void reference_tests() {
    for (const auto& r : references) {
        const auto kind = static_cast<DensityKind>(r.kind);
        MomentResult full;
        const auto evaluate = [&](int order, MomentResult& d) {
            return r.moment==0 ? negative_second_moment(kind,r.shape,order,d)
                : r.moment==1 ? absolute_moment(kind,r.shape,order,d)
                : power_moment(kind,r.shape,r.gamma,r.delta,order,d);
        };
        watching=true;
        const bool ok = evaluate(2,full);
        watching=false;
        if (!ok) std::printf("REF REJECT %s moment=%d shape=%g,%g,%g\n",names[r.kind],r.moment,r.shape[0],r.shape[1],r.shape[2]);
        check(ok,"reference convergence");
        if (!ok) continue;
        const auto n = full.dimension;
        for (int order=0;order<=2;++order) {
            MomentResult d;
            d.gradient.fill(12345.0);
            d.hessian.fill(12345.0);
            watching=true;
            const bool sparse_ok=evaluate(order,d);
            watching=false;
            check(sparse_ok,"sparse order convergence");
            check(d.dimension==n,"runtime result dimension");
            const auto compare = [&](double x, double y, int derivative) {
                const double error = std::abs(x-y)/(derivative ? std::max(1.0,std::abs(y)) : std::abs(y));
                worst[r.kind][r.moment][derivative] = std::max(worst[r.kind][r.moment][derivative],error);
                if (error > (derivative ? 1e-12 : 1e-14)) {
                    std::printf("REF ERROR %s moment=%d order=%d derivative=%d shape=%g,%g,%g actual=%.17g expected=%.17g scaled=%.3g\n",
                        names[r.kind],r.moment,order,derivative,r.shape[0],r.shape[1],r.shape[2],x,y,error);
                    ++failures;
                }
            };
            if (r.moment == 0 && (r.kind == 4 || r.kind == 5 || r.kind == 7)) {
                check(d.value == full.value,"mixture sparse value bits");
                if (order >= 1) for (std::size_t j=0;j<n;++j)
                    check(d.gradient[j] == full.gradient[j],"mixture sparse gradient bits");
            }
            compare(d.value,r.expected[0],0);
            check(std::abs(d.value-full.value)<=1e-14*std::abs(full.value),"sparse value equality");
            if (order>=1) for(std::size_t j=0;j<n;++j) compare(d.gradient[j],r.expected[1+j],1);
            else for(double x:d.gradient) check(x==12345.0,"unrequested gradient untouched");
            if (order>=2) for(std::size_t j=0;j<packed_size(n);++j) compare(d.hessian[j],r.expected[1+n+j],2);
            else for(double x:d.hessian) check(x==12345.0,"unrequested Hessian untouched");
            for(std::size_t j=n;j<d.gradient.size();++j) check(d.gradient[j]==12345,"unused gradient untouched");
            for(std::size_t j=packed_size(n);j<d.hessian.size();++j) check(d.hessian[j]==12345,"unused Hessian untouched");
        }
    }
    for(int k=0;k<8;++k) for(int m=0;m<3;++m)
        std::printf("worst %s moment=%d value=%.3g gradient=%.3g hessian=%.3g\n",names[k],m,worst[k][m][0],worst[k][m][1],worst[k][m][2]);
}

void check(bool ok, const char* message) {
    if (!ok) { ++failures; std::printf("FAIL %s\n", message); }
}

template<class Density>
void sanity(const double* shape) {
    Dual2<Density::shapes> mean, variance, mass;
    const bool a = expectation<Density, 0>(shape, 2, [](double x, int) { return Dual2<0>::constant(x); }, mean);
    const bool b = expectation<Density, 0>(shape, 2, [](double x, int) { return Dual2<0>::constant(x*x); }, variance);
    const bool c = expectation<Density, 0>(shape, 2, [](double, int) { return Dual2<0>::constant(1); }, mass);
    std::printf("sanity S=%zu ok=%d%d%d mean=%.17g variance=%.17g mass=%.17g\n", Density::shapes, a,b,c,mean.v,variance.v,mass.v);
    check(a && b && c, "generic sanity convergence");
    if (a) check(std::abs(mean.v) < 1e-14, "zero mean");
    if (b) check(std::abs(variance.v-1) < 1e-14, "unit variance");
    if (c) check(std::abs(mass.v-1) < 1e-14, "unit mass");
    if (a && b && c) {
        for(std::size_t j=0;j<Density::shapes;++j) {
            check(std::abs(mean.g[j])<1e-12,"mean gradient");
            check(std::abs(variance.g[j])<1e-12,"variance gradient");
            check(std::abs(mass.g[j])<1e-12,"mass gradient");
        }
        for(std::size_t j=0;j<packed_size(Density::shapes);++j) {
            check(std::abs(mean.h[j])<1e-12,"mean Hessian");
            check(std::abs(variance.h[j])<1e-12,"variance Hessian");
            check(std::abs(mass.h[j])<1e-12,"mass Hessian");
        }
    }
}

template<class Density>
void closed_against_quadrature(const double* shape) {
    constexpr auto S=Density::shapes;
    Dual2<S> k,m,kq,mq;
    Dual2<S+2> a,aq;
    check(negative_second_moment<Density>(shape,2,k) && quadrature_negative_second_moment<Density>(shape,2,kq),"kappa quadrature");
    check(absolute_moment<Density>(shape,2,m) && quadrature_absolute_moment<Density>(shape,2,mq),"absolute quadrature");
    check(power_moment<Density>(shape,.2,1.3,2,a) && quadrature_power_moment<Density>(shape,.2,1.3,2,aq),"power quadrature");
    auto compare=[](const auto& x,const auto& y) {
        check(std::abs(x.v-y.v)<1e-14*std::abs(x.v),"closed/quadrature value");
        for(std::size_t j=0;j<x.g.size();++j) check(std::abs(x.g[j]-y.g[j])<1e-12*std::max(1.0,std::abs(x.g[j])),"closed/quadrature gradient");
        for(std::size_t j=0;j<x.h.size();++j) check(std::abs(x.h[j]-y.h[j])<1e-12*std::max(1.0,std::abs(x.h[j])),"closed/quadrature Hessian");
    };
    compare(k,kq); compare(m,mq); compare(a,aq);
}

void identities_and_domains() {
    const double shapes[8][3]={{},{.125},{.125,.3},{1.5},{.2,1.3,.8},{.2,1.3},{.2,1.3},{.2,2.3}};
    closed_against_quadrature<NormalDensity>(shapes[0]);
    closed_against_quadrature<StudentTDensity>(shapes[1]);
    closed_against_quadrature<GedDensity>(shapes[3]);
    closed_against_quadrature<SkewTDensity>(shapes[2]);
    for(int k=0;k<8;++k) {
        MomentResult a,kappa;
        auto kind=static_cast<DensityKind>(k);
        check(power_moment(kind,shapes[k],0,2,2,a),"second moment identity convergence");
        check(negative_second_moment(kind,shapes[k],2,kappa),"identity kappa");
        check(std::abs(a.value-1)<1e-14,"A(0,2)=1");
        const auto S=kappa.dimension;
        for(std::size_t j=0;j<S;++j) {
            check(std::abs(a.gradient[j])<1e-12,"A(0,2) shape gradient");
            check(std::abs(a.hessian[packed(S+2,j,S)]-4*kappa.gradient[j])<1e-12,"gamma/shape cross derivative");
            for(std::size_t l=j;l<S;++l) check(std::abs(a.hessian[packed(S+2,j,l)])<1e-12,"A(0,2) shape Hessian");
        }
        check(std::abs(a.gradient[S]-(4*kappa.value-2))<1e-12,"gamma derivative identity");
        check(std::abs(a.hessian[packed(S+2,S,S)]-2)<1e-12,"gamma curvature identity");
        for(double gamma:{-1.0,1.0,std::numeric_limits<double>::quiet_NaN()})
            check(!power_moment(kind,shapes[k],gamma,1.3,2,a),"gamma domain");
        for(double delta:{0.0,-1.0,std::numeric_limits<double>::infinity()})
            check(!power_moment(kind,shapes[k],0,delta,2,a),"delta domain");
        for(int order:{-1,3}) check(!absolute_moment(kind,shapes[k],order,a),"order domain");
        if(k) check(!absolute_moment(kind,nullptr,2,a),"null shape");
    }
    MomentResult d;
    const double boundary[]={1/2.00001};
    check(power_moment(DensityKind::student_t,boundary,0,2,2,d),"near-boundary second moment");
    check(std::abs(d.value-1)<1e-14 && std::abs(d.gradient[0])<1e-12 && std::abs(d.hessian[0])<1e-12,"near-boundary exact second moment");
    for(auto kind:{DensityKind::student_t,DensityKind::skew_t}) {
        check(!power_moment(kind,shapes[static_cast<int>(kind)],0,8,2,d),"nonexistent t moment");
        check(!power_moment(kind,shapes[static_cast<int>(kind)],0,9,0,d),"nonexistent t value");
    }
    for(int k:{2,4,5,6,7}) {
        double symmetric[3]={0,1.3,.8};
        if(k==2) {symmetric[0]=.125;symmetric[1]=0;}
        if(k==7) symmetric[1]=2.3;
        check(negative_second_moment(static_cast<DensityKind>(k),symmetric,2,d),"zero-skew convergence");
        check(std::abs(d.value-.5)<1e-14,"zero-skew kappa");
        if(k==2) closed_against_quadrature<SkewTDensity>(symmetric);
    }
    Dual2<1> shifted;
    check(expectation<NormalDensity,1>(nullptr,2,[](double x,int order) {
        auto e=order ? Dual2<1>::variable(.3,0):Dual2<1>::constant(.3);
        auto u=e+x;
        return u*u;
    },shifted),"generic integrand parameters");
    check(std::abs(shifted.v-1.09)<1e-14 && std::abs(shifted.g[0]-.6)<1e-12 && std::abs(shifted.h[0]-2)<1e-12,"generic integrand derivatives");
    Dual2<2> mixed;
    Dual2<1> m;
    check(absolute_moment<StudentTDensity>(shapes[1],2,m),"mixed reference");
    check(expectation<StudentTDensity,1>(shapes[1],2,[](double x,int order) {
        const auto e=order ? Dual2<1>::variable(.7,0):Dual2<1>::constant(.7);
        return std::abs(x)*e;
    },mixed),"generic shape/integrand cross derivatives");
    check(std::abs(mixed.v-.7*m.v)<1e-14 && std::abs(mixed.g[0]-.7*m.g[0])<1e-12
        && std::abs(mixed.g[1]-m.v)<1e-12 && std::abs(mixed.h[1]-m.g[0])<1e-12,"mixed derivative identity");
    d.value=12345;
    d.dimension=12345;
    d.gradient.fill(12345);
    d.hessian.fill(12345);
    check(!absolute_moment(static_cast<DensityKind>(99),nullptr,2,d),"unknown density");
    check(!power_moment(DensityKind::student_t,shapes[1],0,8,2,d),"failure preserves buffers");
    check(d.value==12345 && d.dimension==12345,"failure preserves value and dimension");
    for(double x:d.gradient) check(x==12345,"failure preserves gradient");
    for(double x:d.hessian) check(x==12345,"failure preserves Hessian");
    Dual2<1> value_only;
    value_only.g.fill(12345);
    value_only.h.fill(12345);
    check(absolute_moment<StudentTDensity>(shapes[1],0,value_only),"template value-only request");
    check(value_only.g[0]==12345 && value_only.h[0]==12345,"template unrequested derivatives untouched");
}

void boundary_stress() {
    const double points[8][3]={{},{std::nextafter(0.5,0.0)},
        {std::nextafter(0.5,0.0),.999999},{.01},{.9999,.001,-.5},{.9999,.001},{.9999,.001},{.9999,1.50000001}};
    for(int k=0;k<8;++k) {
        int status[3]{};
        for(int m=0;m<3;++m) {
            MomentResult d;
            const auto kind=static_cast<DensityKind>(k);
            status[m]=m==0 ? negative_second_moment(kind,points[k],2,d)
                : m==1 ? absolute_moment(kind,points[k],2,d) : power_moment(kind,points[k],.2,1.3,2,d);
            if(status[m]) {
                check(std::isfinite(d.value) && d.value>0,"finite positive stress value");
                if(m<2) check(d.value<1,"stress moment bound");
            }
        }
        std::printf("boundary stress %s shape=%.17g,%.17g,%.17g status=%d%d%d\n",names[k],points[k][0],points[k][1],points[k][2],status[0],status[1],status[2]);
    }
}

void smoke() {
    const double shapes[8][3] = {{}, {.125}, {.125,.3}, {1.5}, {.2,1.3,.8}, {.2,1.3}, {.2,1.3}, {.2,2.3}};
    for (int k=0; k<8; ++k) {
        for (int m=0; m<3; ++m) {
            MomentResult d;
            const int repetitions = k==0 ? 10000 : k==1 || k==3 ? 2000 : k==2 ? 50 : 10;
            watching = true;
            const auto start = std::chrono::steady_clock::now();
            const auto kind = static_cast<DensityKind>(k);
            bool ok = true;
            for(int repeat=0;repeat<repetitions;++repeat) ok &= m==0 ? negative_second_moment(kind, shapes[k],2,d)
                : m==1 ? absolute_moment(kind,shapes[k],2,d) : power_moment(kind,shapes[k],.2,1.3,2,d);
            const double us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repetitions;
            watching = false;
            std::printf("%s moment=%d ok=%d value=%.17g time=%.3f us\n", names[k],m,ok,d.value,us);
            check(ok,"moment convergence");
        }
    }
    check(allocations==0,"no allocations");
    sanity<NormalDensity>(shapes[0]);
    sanity<StudentTDensity>(shapes[1]);
    sanity<SkewTDensity>(shapes[2]);
    sanity<GedDensity>(shapes[3]);
    sanity<GhFamilyDensity<GeneralizedHyperbolic>>(shapes[4]);
    sanity<GhFamilyDensity<NormalInverseGaussian>>(shapes[5]);
    sanity<GhFamilyDensity<Hyperbolic>>(shapes[6]);
    sanity<GhFamilyDensity<VarianceGamma>>(shapes[7]);
}
}
int main() { reference_tests(); identities_and_domains(); boundary_stress(); smoke(); return failures ? 1 : 0; }
