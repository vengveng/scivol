#include "density/distribution.hpp"
#include "density/domain.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

namespace { bool watching = false; std::size_t allocations = 0; }
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
using Status = ExpectationStatus;
constexpr const char* names[] = {"Normal","Student-t","Skew-t","GED","GH","NIG","Hyperbolic","VG"};
constexpr double shapes[8][3] = {{},{.125},{.125,.3},{1.5},{.2,1.3,.8},{.2,1.3},{.2,1.3},{.2,2.3}};
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { ++failures; std::printf("FAIL %s\n",message); }
}
struct ShapePoint { int kind; double shape[3]; };
constexpr ShapePoint points[] = {
    {0,{}}, {1,{0.49997500124993743}}, {1,{0.48780487804878053}}, {1,{0.125}}, {1,{0.02}}, {1,{0.002}}, {1,{0.001}}, {1,{0.0001}}, {1,{1e-08}},
    {2,{0.48780487804878053,-.7}}, {2,{0.48780487804878053,.7}}, {2,{0.125,-.7}}, {2,{0.125,.7}}, {2,{1e-08,.7}},
    {3,{.7}}, {3,{1}}, {3,{1.5}}, {3,{2}}, {3,{4}},
    {4,{.2,1.3,.8}}, {4,{.9,.03,-.5}}, {4,{-.9,.05,-2}},
    {4,{.9,40,2}}, {4,{-.9,60,.25}}, {4,{0,1,-1}},
    {5,{.2,1.3}}, {5,{.9,.03}}, {5,{-.9,.05}}, {5,{.9,40}},
    {6,{.2,1.3}}, {6,{.9,.03}}, {6,{-.9,.05}}, {6,{-.9,60}},
    {7,{.2,2.3}}, {7,{.9,1.5001}}, {7,{-.9,2}}, {7,{.9,20}}
};
struct Reference { int kind; double shape[3], z, lower, upper; };
struct QuantileReference { int kind; double shape[3], probability; bool upper; double expected; };
#include "distribution_reference.inc"
void reference_tests() {
    double worst[8]{};
    double inverse_worst[8]{};
    int accuracy_rejections = 0;
    for (const auto& ref : references) {
        double l = 12345, r = 6789;
        const auto kind = static_cast<DensityKind>(ref.kind);
        if (distribution(kind,ref.shape,ref.z,l,r) != Status::ok) {
            if (ref.kind == 3 && ref.shape[0] > 8.0 && std::min(ref.lower,ref.upper) > 0.0
                && std::min(ref.lower,ref.upper) < 1e-10) {
                ++accuracy_rejections;
                check(l == 12345 && r == 6789,"unresolved flat-GED reference preserves outputs");
                continue;
            }
            std::printf("REFERENCE REJECT %s shape=%g,%g,%g z=%.17g\n",names[ref.kind],ref.shape[0],ref.shape[1],ref.shape[2],ref.z);
            ++failures; continue;
        }
        const double error_l = ref.lower ? std::abs(l/ref.lower-1) : l;
        const double error_r = ref.upper ? std::abs(r/ref.upper-1) : r;
        const double error = std::max(error_l,error_r);
        worst[ref.kind] = std::max(worst[ref.kind],error);
        if (error > 1e-12) {
            std::printf("REFERENCE ERROR %s shape=%g,%g,%g z=%.17g lower=%.17g expected=%.17g upper=%.17g expected=%.17g relative=%.3g\n",
                names[ref.kind],ref.shape[0],ref.shape[1],ref.shape[2],ref.z,l,ref.lower,r,ref.upper,error);
            ++failures;
        }
        const bool upper = ref.upper < ref.lower;
        const double u = upper ? ref.upper : ref.lower;
        if (u > 0.0 && u <= .5 && !(ref.kind == 3 && ref.shape[0] > 8.0)) {
            double x = 12345;
            const auto status = upper ? quantile_upper(kind,ref.shape,u,x) : quantile(kind,ref.shape,u,x);
            if (status != Status::ok) { check(false,"inverse reference status"); continue; }
            const double dx = std::abs(x-ref.z)/std::max(1.,std::abs(ref.z));
            inverse_worst[ref.kind] = std::max(inverse_worst[ref.kind],dx);
            if (dx > 1e-12) {
                std::printf("INVERSE REFERENCE ERROR %s shape=%g,%g,%g u=%g upper=%d expected=%.17g actual=%.17g relative=%.3g\n",
                    names[ref.kind],ref.shape[0],ref.shape[1],ref.shape[2],u,upper,ref.z,x,dx);
                ++failures;
            }
        }
    }
    for (int k = 0; k < 8; ++k) std::printf("reference %s worst CDF=%.3g quantile=%.3g\n",names[k],worst[k],inverse_worst[k]);
    std::printf("reference grid=%zu accuracy rejections=%d\n",std::size(references),accuracy_rejections);
}
void inverse_reference_tests() {
    double worst[8]{};
    for (const auto& ref : quantile_references) {
        double x = 12345;
        const auto kind = static_cast<DensityKind>(ref.kind);
        const auto status = ref.upper ? quantile_upper(kind,ref.shape,ref.probability,x)
                                      : quantile(kind,ref.shape,ref.probability,x);
        if (status != Status::ok) { check(false,"independent inverse reference status"); continue; }
        const double error = std::abs(x-ref.expected)/std::max(1.,std::abs(ref.expected));
        worst[ref.kind] = std::max(worst[ref.kind],error);
        if (error > 1e-12) {
            std::printf("INDEPENDENT INVERSE ERROR %s shape=%g,%g,%g u=%g upper=%d actual=%.17g expected=%.17g relative=%g\n",
                names[ref.kind],ref.shape[0],ref.shape[1],ref.shape[2],ref.probability,ref.upper,x,ref.expected,error);
            ++failures;
        }
    }
    for (int k = 0; k < 8; ++k) std::printf("independent inverse %s worst=%.3g\n",names[k],worst[k]);
    std::printf("independent inverse grid=%zu\n",std::size(quantile_references));
}
void roundtrips() {
    double all_worst[8]{};
    for (const auto& point : points) {
        const int k = point.kind;
        const auto* shape = point.shape;
        double worst = 0.0;
        const auto kind = static_cast<DensityKind>(k);
        for (double u : {1e-300,1e-100,1e-20,1e-8,.01,.1,.5,.9,1.0-1e-16}) {
            for (bool upper : {false,true}) {
                double x = 12345.0, l = 12345.0, r = 12345.0;
                const auto status = upper ? quantile_upper(kind,shape,u,x) : quantile(kind,shape,u,x);
                if (status != Status::ok || distribution(kind,shape,x,l,r) != Status::ok) {
                    std::printf("REJECT %s shape=%g,%g,%g u=%g upper=%d x=%.17g\n",names[k],shape[0],shape[1],shape[2],u,upper,x); ++failures; continue;
                }
                const double error = std::abs((upper ? r : l)/u-1.0);
                worst = std::max(worst,error);
                if (error > 1e-12) {
                    std::printf("ROUNDTRIP %s u=%.17g upper=%d x=%.17g p=%.17g rel=%g\n",names[k],u,upper,x,upper?r:l,error);
                    ++failures;
                }
            }
        }
        all_worst[k] = std::max(all_worst[k],worst);
    }
    for (int k = 0; k < 8; ++k) std::printf("roundtrip %s worst=%.3g\n",names[k],all_worst[k]);
}
void batch_tests() {
    constexpr std::size_t n = 4096;
    std::vector<double> u(n), out(n);
    for (std::size_t i = 0; i < n; ++i) u[i] = (i+0.5)/n;
    for (int k = 0; k < 8; ++k) {
        const auto kind = static_cast<DensityKind>(k);
        watching = true;
        const auto status = quantiles(kind,shapes[k],u.data(),n,out.data());
        watching = false;
        check(status == Status::ok,"batch inversion");
        if (status != Status::ok) { std::printf("BATCH REJECT %s\n",names[k]); continue; }
        double mean = 0.0, second = 0.0, worst = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            mean += out[i]/n; second += out[i]*out[i]/n;
            if (i) check(out[i] > out[i-1],"monotonicity");
            if (i%31 == 0) {
                double l, r;
                check(distribution(kind,shapes[k],out[i],l,r) == Status::ok,"batch CDF status");
                const double error = u[i] <= .5 ? std::abs(l/u[i]-1) : std::abs(r/(1-u[i])-1);
                worst = std::max(worst,error);
            }
        }
        check(worst < 1e-12,"batch exact CDF roundtrip");
        check(std::abs(mean) < .002,"quantile integral mean");
        check(std::abs(second-1) < .02,"quantile integral variance");
        std::printf("batch %s worst=%.3g mean=%.6g variance=%.6g\n",names[k],worst,mean,second-mean*mean);
        // Deliberately unsorted, with exact in-place input/output aliasing.
        const auto expected = out;
        for (std::size_t i = 0; i < n; ++i) out[i] = u[(i*1543)%n];
        watching = true;
        check(quantiles(kind,shapes[k],out.data(),n,out.data()) == Status::ok,"in-place unsorted batch");
        watching = false;
        for (std::size_t i = 0; i < n; ++i)
            check(std::abs(out[i]-expected[(i*1543)%n]) < 1e-12*std::max(1.0,std::abs(out[i])),"in-place permutation");
    }
}
void shaped_batch_tests() {
    constexpr std::size_t n = 512;
    std::vector<double> u(n), out(n);
    for (std::size_t i = 0; i < n; ++i) u[i] = (i+0.5)/n;
    u.front() = 1e-300; u.back() = 1.0-1e-16;
    double worst = 0.0;
    for (const auto& point : points) {
        if (point.kind < 4) continue;
        const auto kind = static_cast<DensityKind>(point.kind);
        watching = true;
        const auto status = quantiles(kind,point.shape,u.data(),n,out.data());
        watching = false;
        check(status == Status::ok,"GH shape-grid batch inversion");
        if (status != Status::ok) continue;
        for (std::size_t i = 0; i < n; ++i) {
            if (i) check(out[i] > out[i-1],"GH shape-grid batch monotonicity");
            if (i%29 == 0 || i == n-1) {
                double l, r;
                check(distribution(kind,point.shape,out[i],l,r) == Status::ok,"GH shape-grid batch CDF");
                const double error = u[i] <= .5 ? std::abs(l/u[i]-1) : std::abs(r/(1-u[i])-1);
                worst = std::max(worst,error);
                if (error > 1e-12) {
                    std::printf("SHAPED BATCH ERROR %s shape=%g,%g,%g u=%g relative=%g\n",
                        names[point.kind],point.shape[0],point.shape[1],point.shape[2],u[i],error);
                    ++failures;
                }
            }
        }
    }
    std::printf("GH shape-grid batch worst=%.3g\n",worst);
}
void domains() {
    constexpr double infinity = std::numeric_limits<double>::infinity();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    for (int k = 0; k < 8; ++k) {
        const auto kind = static_cast<DensityKind>(k);
        for (double x : {nan,infinity,-infinity}) {
            double l = 12345, r = 6789;
            check(distribution(kind,shapes[k],x,l,r) == Status::numerical_failure,"nonfinite z");
            check(l == 12345 && r == 6789,"CDF failure preserves outputs");
        }
        for (double u : {nan,infinity,-infinity,-1.,0.,1.,2.}) {
            double x = 12345;
            check(quantile(kind,shapes[k],u,x) == Status::numerical_failure,"invalid lower probability");
            check(quantile_upper(kind,shapes[k],u,x) == Status::numerical_failure,"invalid upper probability");
            check(x == 12345,"inverse failure preserves output");
            const double probabilities[] = {.3,u}; double out[] = {12345,6789};
            check(quantiles(kind,shapes[k],probabilities,2,out) == Status::numerical_failure,"invalid batch probability");
            check(out[0] == 12345 && out[1] == 6789,"invalid batch is not written");
        }
        for (std::size_t j = 0; j < shape_domains(kind).size(); ++j) {
            const auto d = shape_domains(kind)[j];
            for (double invalid : {d.closed_lower ? std::nextafter(d.lower,-infinity) : d.lower,d.upper,nan,infinity,-infinity}) {
                double s[3]; std::copy_n(shapes[k],3,s); s[j] = invalid;
                double l = 12345, r = 6789, x = 4321;
                check(distribution(kind,s,0,l,r) == Status::numerical_failure,"invalid shape CDF");
                check(quantile(kind,s,.3,x) == Status::numerical_failure,"invalid shape inverse");
                check(quantile_upper(kind,s,.3,x) == Status::numerical_failure,"invalid shape upper inverse");
                check(l == 12345 && r == 6789 && x == 4321,"invalid shape preserves outputs");
            }
        }
        double l, r, x;
        if (k) check(distribution(kind,nullptr,0,l,r) == Status::numerical_failure,"missing shapes");
        check(quantiles(kind,shapes[k],nullptr,0,nullptr) == Status::ok,"empty batch");
        check(quantiles(kind,shapes[k],nullptr,1,&x) == Status::numerical_failure,"missing probabilities");
        const double u[] = {.2};
        check(quantiles(kind,shapes[k],u,1,nullptr) == Status::numerical_failure,"missing output");
    }
    for (int k : {-1,99}) {
        double l, r, x;
        check(distribution(static_cast<DensityKind>(k),nullptr,0,l,r) == Status::numerical_failure,"invalid kind CDF");
        check(quantile(static_cast<DensityKind>(k),nullptr,.3,x) == Status::numerical_failure,"invalid kind inverse");
    }
}
void reflection() {
    for (const auto& point : points) {
        const auto kind = static_cast<DensityKind>(point.kind);
        double mirror[3]; std::copy_n(point.shape,3,mirror);
        if (point.kind == 2) mirror[1] *= -1;
        else if (point.kind >= 4) mirror[0] *= -1;
        for (double z : {-10.,-1.,0.,1.,10.}) {
            double l, r, ml, mr;
            check(distribution(kind,point.shape,z,l,r) == Status::ok,"reflection CDF");
            check(distribution(kind,mirror,-z,ml,mr) == Status::ok,"reflected CDF");
            check(std::abs(l-mr) <= 1e-12*l && std::abs(r-ml) <= 1e-12*r,"reflected tails");
        }
    }
}
void underflow_and_flat_shapes() {
    double lower = 12345, upper = 6789;
    check(distribution(DensityKind::normal,nullptr,38.0,lower,upper) == Status::numerical_failure,
          "subnormal relative rounding budget");
    check(lower == 12345 && upper == 6789,"subnormal failure preserves outputs");
    check(distribution(DensityKind::normal,nullptr,100.0,lower,upper) == Status::ok
          && lower == 1.0 && upper == 0.0,"true tail underflow");
    for (double nu : {10.,1000.}) {
        const double shape[] = {nu};
        for (double u : {.1,.3,.49,.7,.9}) {
            double x, l, r;
            check(quantile(DensityKind::ged,shape,u,x) == Status::ok,"flat GED quantile");
            check(distribution(DensityKind::ged,shape,x,l,r) == Status::ok,"flat GED roundtrip CDF");
            check(std::abs(l/u-1) < 1e-12,"flat GED roundtrip");
        }
    }
    const double flat[] = {1000.}; double x = 12345;
    check(quantile(DensityKind::ged,flat,1e-300,x) == Status::numerical_failure
          && x == 12345,"unrepresentable flat GED extreme quantile rejected");
}
void timing(bool batch) {
    constexpr std::size_t n = 1000000;
    std::vector<double> u(batch ? n : 0), out(batch ? n : 0);
    if (batch) for (std::size_t i = 0; i < n; ++i) u[i] = (i+0.5)/n;
    for (int k = 0; k < 8; ++k) {
        const auto kind = static_cast<DensityKind>(k);
        double x, l, r;
        bool ok = true;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) ok &= distribution(kind,shapes[k],.7,l,r) == Status::ok;
        const auto middle = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) ok &= quantile(kind,shapes[k],.1,x) == Status::ok;
        const auto end = std::chrono::steady_clock::now();
        std::printf("timing %s distribution=%.3f us quantile=%.3f us",names[k],
            std::chrono::duration<double,std::micro>(middle-start).count()/10,
            std::chrono::duration<double,std::micro>(end-middle).count()/10);
        if (batch) {
            watching = true;
            const auto bs = std::chrono::steady_clock::now();
            ok &= quantiles(kind,shapes[k],u.data(),n,out.data()) == Status::ok;
            const auto be = std::chrono::steady_clock::now();
            watching = false;
            std::printf(" batch=%.3f us/value",std::chrono::duration<double,std::micro>(be-bs).count()/n);
        }
        std::printf("\n"); check(ok,"timing succeeds"); std::fflush(stdout);
    }
}
}
int main(int argc, char**) {
    watching = true;
    reference_tests();
    inverse_reference_tests();
    roundtrips();
    domains();
    reflection();
    underflow_and_flat_shapes();
    watching = false;
    batch_tests();
    shaped_batch_tests();
    if (argc > 1) timing(true);
    check(allocations == 0,"no allocations during evaluation");
    return failures ? 1 : 0;
}
