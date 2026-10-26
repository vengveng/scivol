#include "math/bessel.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

namespace {
bool watch_allocations = false;
std::size_t allocations = 0;
}

void* operator new(std::size_t size) {
    if (watch_allocations) ++allocations;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
using namespace scivolapi::detail::bessel;
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
struct Reference { int function; double nu, z; std::array<double, 6> expected; };
#include "bessel_reference.inc"

constexpr const char* names[] = {"L", "M", "S", "E", "G"};
constexpr const char* quantities[] = {"value", "z", "zz", "nu", "nunu", "nuz"};
constexpr const char* regions[] = {"tiny z < 1e-50", "small z <= 2", "2 < z < 64", "z >= 64", "|nu| > 128", "z = 0"};
struct Worst { double error = 0.0, nu = 0.0, z = 0.0; int function = 0; };
Worst worst[5][6][6]{};
int failures = 0;

std::array<double, 6> fields(LogK x) noexcept {
    return {x.value, x.z, x.zz, x.nu, x.nunu, x.nuz};
}

bool requested(int i, unsigned want) noexcept {
    return i == 0 || (i <= 2 ? bool(want & argument) : bool(want & order));
}

void fail(const char* what, const Reference& row, unsigned want, int field = -1,
          double actual = nan, double expected = nan) noexcept {
    if (failures++ < 25)
        std::printf("FAIL %s %s nu=%.17g z=%.17g want=%u %s actual=%.17g expected=%.17g\n",
            what, names[row.function], row.nu, row.z, want,
            field < 0 ? "" : quantities[field], actual, expected);
}

int region(const Reference& row) noexcept {
    if (row.z == 0) return 5;
    if (std::abs(row.nu) > 128) return 4;
    return row.z < 1e-50 ? 0 : row.z <= 2 ? 1 : row.z < 64 ? 2 : 3;
}

using Function = bool (*)(double, double, unsigned, LogK&) noexcept;
constexpr Function functions[] = {log_k, log_zk, log_k_scaled, log_zk_scaled, log_ratio_gap};

bool prepared_zk(double nu, double z, unsigned want, LogK& out) noexcept {
    PreparedOrder p;
    return prepare(nu, want, p) && log_zk_scaled(p, z, out);
}

void check_reference(const Reference& row, bool prepared = false) noexcept {
    const auto function = prepared ? prepared_zk : functions[row.function];
    LogK complete;
    const bool complete_ok = function(row.nu, row.z, argument|order, complete);
    const auto full = fields(complete);
    for (unsigned want = 0; want <= (argument|order); ++want) {
        bool expected_ok = true;
        for (int i = 0; i < 6; ++i)
            if (requested(i, want) && !std::isfinite(row.expected[i])) expected_ok = false;
        LogK output;
        const bool ok = function(row.nu, row.z, want, output);
        if (ok != expected_ok) fail("return status", row, want);
        const auto actual = fields(output);
        for (int i = 0; i < 6; ++i) {
            if (!ok || !requested(i, want)) {
                if (!std::isnan(actual[i])) fail("expected NaN", row, want, i, actual[i]);
                continue;
            }
            if (!std::isfinite(actual[i])) { fail("nonfinite output", row, want, i, actual[i]); continue; }
            const double scaled_error = std::abs(actual[i]-row.expected[i])/std::max(1.0, std::abs(row.expected[i]));
            auto& w = worst[row.function][region(row)][i];
            if (scaled_error > w.error) w = {scaled_error, row.nu, row.z, row.function};
            if (scaled_error > (i == 0 ? 1e-14 : 1e-12))
                fail("reference error", row, want, i, actual[i], row.expected[i]);
            if (!row.function && row.z >= 64 && i == 2 && row.expected[i] > 0
                && std::abs(actual[i]/row.expected[i]-1.0) > 1e-10)
                fail("relative tail curvature", row, want, i, actual[i], row.expected[i]);
            if (complete_ok && std::bit_cast<std::uint64_t>(actual[i]) != std::bit_cast<std::uint64_t>(full[i]))
                fail("sparse bits", row, want, i, actual[i], full[i]);
        }
        if (want) {
            LogK first;
            const auto first_fields = [&] {
                function(row.nu, row.z, want | first_only, first);
                return fields(first);
            }();
            for (int i : {0, 1, 3}) {
                if (!requested(i, want) || !std::isfinite(full[i])) continue;
                if (complete_ok && first_fields[i] != full[i])
                    fail("first derivative sparse bits", row, want | first_only, i, first_fields[i], full[i]);
            }
        }
    }
}

struct FixedReference { double z; std::array<double, 3> expected; };
#include "nig_reference.inc"

void check_fixed_nig() noexcept {
    double worst[3] = {};
    for (const auto& r : fixed_references) {
        LogK full;
        const bool full_ok = log_zk_scaled_nig(r.z, argument, full);
        for (unsigned want : {0u, unsigned(argument), unsigned(argument | first_only)}) {
            LogK output;
            const bool expected_ok = want == 0 || ((want & first_only) ? std::isfinite(r.expected[1])
                : std::isfinite(r.expected[2]));
            const bool ok = log_zk_scaled_nig(r.z, want, output);
            const Reference row{3, -1.0, r.z, {}};
            if (ok != expected_ok) fail("fixed NIG status", row, want);
            const auto actual = fields(output), complete = fields(full);
            for (int i = 0; i < 6; ++i) {
                const bool requested = i == 0 || (i == 1 && want & argument)
                    || (i == 2 && want == argument);
                if (!ok || !requested) {
                    if (!std::isnan(actual[i])) fail("fixed NIG NaN", row, want, i);
                    continue;
                }
                const double error = std::abs(actual[i] - r.expected[i]) / (i == 0 ? std::max(1.0, std::abs(r.expected[i])) : std::abs(r.expected[i]));
                worst[i] = std::max(worst[i], error);
                if (error > 1e-14) fail("fixed NIG relative error", row, want, i, actual[i], r.expected[i]);
                if (full_ok && actual[i] != complete[i]) fail("fixed NIG sparse bits", row, want, i);
            }
        }
    }
    for (double z : {-1.0, 0.0, inf, nan}) {
        LogK output;
        if (log_zk_scaled_nig(z, 0u, output)) fail("fixed NIG domain", {3, -1.0, z, {}}, 0u);
    }
    std::printf("Fixed NIG worst errors log value scaled, derivatives relative: value=%.3g z=%.3g zz=%.3g, %zu points\n",
        worst[0], worst[1], worst[2], std::size(fixed_references));
}

void check_ladders() noexcept {
    for (double nu : {-37.5, -3.2, -1.0, -0.5, -1e-8, 0.0, 0.5, 1.0, 7.3, 100.0, 1000.0})
        for (double z : {1e-300, 1e-100, 1e-3, 0.1, 2.0, 2.1, 20.0, 100.0, 1e8})
            for (unsigned want = 0; want <= (argument|order); ++want)
                for (std::size_t count = 0; count <= 4; ++count) {
                    LogK ladder[4], separate[4];
                    bool expected_ok = true;
                    for (std::size_t i = 0; i < count; ++i)
                        expected_ok = log_k(nu+static_cast<double>(i), z, want, separate[i]) && expected_ok;
                    const bool ok = log_k_ladder(nu, z, count, want, ladder);
                    const Reference row{false, nu, z, {}};
                    if (ok != expected_ok) fail("ladder status", row, want);
                    for (std::size_t i = 0; i < count; ++i) {
                        const auto a = fields(ladder[i]), b = fields(separate[i]);
                        for (int j = 0; j < 6; ++j) {
                            if (!ok || !requested(j, want)) {
                                if (!std::isnan(a[j])) fail("ladder NaN", row, want, j);
                            } else if (std::abs(a[j]-b[j]) > 1e-14*std::max(1.0, std::abs(b[j])))
                                fail("ladder mismatch", row, want, j, a[j], b[j]);
                        }
                    }
                }
}

void check_domains() noexcept {
    for (bool product : {false, true})
        for (double nu : {-inf, -1.0, 0.0, 0.5, 1.0, inf, nan})
            for (double z : {-inf, -1.0, 0.0, 1.0, inf, nan}) {
                const bool valid = std::isfinite(nu) && std::isfinite(z)
                    && (product ? nu > 0.0 && z >= 0.0 : z > 0.0);
                LogK out;
                const bool ok = product ? log_zk(nu, z, value_only, out) : log_k(nu, z, value_only, out);
                if (ok != valid) fail("domain", {product, nu, z, {}}, value_only);
                if (!ok)
                    for (double f : fields(out))
                        if (!std::isnan(f)) fail("domain NaN", {product, nu, z, {}}, value_only);
            }
    LogK out[4];
    if (log_k_ladder(1, 1, 5, 0, out) || log_k_ladder(1, 1, 1, 0, nullptr)
        || log_k(1, 1, 4, out[0]) || log_zk(1, 1, 4, out[0]))
        fail("invalid request", {false, 1, 1, {}}, 4);
    // These curvatures overflow, although value and order requests are valid.
    if (log_k(20, 1e-300, argument, out[0]) || !log_k(20, 1e-300, order, out[0])
        || log_zk(0.25, 1e-300, argument, out[0]) || !log_zk(0.25, 1e-300, order, out[0]))
        fail("representability", {false, 20, 1e-300, {}}, argument);
    if (!log_zk(0.75, 0.0, argument | first_only, out[0]) || out[0].z != 0.0
        || !std::isnan(out[0].zz) || !log_zk(0.25, 0.0, order | first_only, out[0])
        || !std::isfinite(out[0].nu) || !std::isnan(out[0].nunu)
        || log_zk(0.25, 0.0, argument | first_only, out[0])
        || !log_k(20.0, 1e-300, argument | first_only, out[0]))
        fail("first derivative limits", {true, 0.75, 0.0, {}}, argument | first_only);
}

void check_extreme_limits() noexcept {
    const auto close = [](double actual, double expected) {
        return std::isfinite(actual) && std::abs(actual/expected-1.0) < 2e-14;
    };
    // Debye's discarded terms are smaller than one ulp at these orders.
    // Check tiny representable derivatives relatively, so zero cannot pass.
    for (double a : {1e100, 1e300, 1.7e308}) {
        LogK out;
        const double root = std::sqrt(2.0), inverse = (1.0/a)/root;
        if (!log_k(a, a, argument|order, out)
            || !close(out.value/a, std::asinh(1.0)-root)
            || !close(out.z, -root) || !close(out.zz, inverse)
            || !close(out.nu, std::asinh(1.0)) || !close(out.nunu, inverse)
            || !close(out.nuz, -inverse))
            fail("large-order limits", {false, a, a, {}}, argument|order);
    }
    LogK out, zero;
    if (!log_zk(1e100, 1e-200, argument|order, out)
        || !close(out.z, -5e-301) || !close(out.zz, -5e-101)
        || !close(out.nunu, 1e-100))
        fail("scaled extreme limit", {true, 1e100, 1e-200, {}}, argument|order);
    if (!log_k(0, 1, argument|order, zero)
        || !log_k(1e-300, 1, argument|order, out)
        || !close(out.nu/1e-300, zero.nunu))
        fail("small-order parity", {false, 1e-300, 1, {}}, argument|order);
}

void check_scaled_identities() noexcept {
    for (double nu : {-60.0, -1.0, -0.5, -1e-8, 0.0, 0.5, 1.0, 20.0, 60.0, 1000.0})
        for (double z : {1e-10, 0.1, 2.0, 2.1, 20.0, 64.0, 100.0, 1e4, 1e6})
            for (int f : {2, 3}) {
                if (f == 3 && nu <= 0.0) continue;
                LogK plain, scaled;
                if (!functions[f-2](nu, z, argument|order, plain)
                    || !functions[f](nu, z, argument|order, scaled)) {
                    fail("scaled identity status", {f, nu, z, {}}, argument|order);
                    continue;
                }
                const auto a = fields(plain), b = fields(scaled);
                for (int q = 0; q < 6; ++q) {
                    const double expected = a[q]+(q == 0 ? z : q == 1 ? 1.0 : 0.0);
                    const double rounding = 4.0*std::numeric_limits<double>::epsilon()
                        * std::max(std::abs(a[q]), q == 0 ? z : q == 1 ? 1.0 : 0.0);
                    const double tolerance = (q == 0 ? 1e-14 : 1e-12)*std::max(1.0, std::abs(b[q]));
                    if (rounding <= tolerance && std::abs(b[q]-expected) > tolerance)
                        fail("scaled identity", {f, nu, z, {}}, argument|order, q, b[q], expected);
                }
            }
    for (int f : {2, 3, 4})
        for (double nu : {-inf, -1.0, 0.0, 0.5, 1.0, inf, nan})
            for (double z : {-inf, -1.0, 0.0, 1.0, inf, nan}) {
                const bool valid = std::isfinite(nu) && std::isfinite(z)
                    && (z > 0.0 || (f == 3 && z == 0.0 && nu > 0.0));
                LogK out;
                const bool ok = functions[f](nu, z, value_only, out);
                if (ok != valid) fail("scaled domain", {f, nu, z, {}}, value_only);
                if (!ok)
                    for (double x : fields(out))
                        if (!std::isnan(x)) fail("scaled domain NaN", {f, nu, z, {}}, value_only);
            }
    for (int f : {2, 3, 4}) {
        LogK out;
        if (functions[f](1, 1, 4, out)) fail("scaled invalid request", {f, 1, 1, {}}, 4);
        for (double x : fields(out))
            if (!std::isnan(x)) fail("scaled invalid request NaN", {f, 1, 1, {}}, 4);
    }
    // An unscaled slope rounds to -1 here. The scaled result must retain
    // its small, representable correction, including the product at nu=1/2.
    for (double z : {1e6, 1e20, 1e100, 1e300}) {
        LogK s, e;
        if (!log_k_scaled(0.5, z, argument, s)
            || std::abs(s.z/(-0.5/z)-1.0) > 2e-15
            || !log_zk_scaled(1.5, z, argument, e)
            || std::abs(e.z/(1.0/(1.0+z))-1.0) > 2e-15
            || !log_zk_scaled(0.5, z, argument, e) || e.z != 0.0 || e.zz != 0.0)
            fail("scaled tail correction", {2, 0.5, z, {}}, argument);
    }
}

void check_gap_extreme_limits() noexcept {
    const auto relative = [](double actual, double expected) {
        return std::isfinite(actual) && std::abs(actual/expected-1.0) < 3e-14;
    };
    for (double a : {1e100, 1e300, 1.7e308}) {
        LogK g;
        const double t = 1.0/std::sqrt(2.0), invh = t/a;
        const double expected_nu = (1.0-t)*invh;
        if (!log_ratio_gap(a, a, argument|order, g)
            || std::abs((g.value+std::log(a))-std::log1p(t)) > 2e-13
            || !relative(g.z, (-1.0-t+t*t)/a)
            || !relative(g.nu, expected_nu))
            fail("gap large-order limits", {4, a, a, {}}, argument|order);
        if (a == 1e100
            && (!relative(g.nunu, -(1.0+t-2.0*t*t)*invh*invh)
                || !relative(g.nuz, (2.0*t-1.0)*t*invh*invh)
                || !relative(g.zz, (1.0+t-t*t+(1.0-2.0*t)*t*(1.0-t*t))/a/a)))
            fail("gap large-order curvature", {4, a, a, {}}, argument|order);
    }
}

void print_errors() noexcept {
    for (int f = 0; f < 5; ++f) {
        std::printf("Maximum |error| / max(1, |reference|), %s\n", names[f]);
        std::printf("%-18s %11s %11s %11s %11s %11s %11s\n", "region", "value", "z", "zz", "nu", "nunu", "nuz");
        for (int r = 0; r < 6; ++r) {
            std::printf("%-18s", regions[r]);
            for (int q = 0; q < 6; ++q) std::printf(" %11.3e", worst[f][r][q].error);
            std::puts("");
        }
        for (int r = 0; r < 6; ++r)
            for (int q = 0; q < 6; ++q)
                if (worst[f][r][q].error > (q == 0 ? 1e-14 : 1e-12)) {
                    const auto& w = worst[f][r][q];
                    std::printf("Worst %s %s: %s nu=%.17g z=%.17g\n", regions[r], quantities[q], names[f], w.nu, w.z);
                }
    }
}

void print_timings() noexcept {
    constexpr int repetitions = 1500;
    constexpr double order_panels[2][5] = {{-3.0, -0.5, 0.25, 1.0, 5.0}, {-60.0, -1.0, -0.5, 0.5, 60.0}};
    constexpr double argument_panels[2][5] = {{0.05, 0.5, 2.0, 5.0, 50.0}, {64.0, 100.0, 1e3, 1e6, 1e8}};
    volatile double sink = 0.0;
    for (int panel = 0; panel < 2; ++panel) {
        std::printf("Timing ns/call, median of 3, 25 points, %s panel; M/E use |nu|\n", panel ? "large argument" : "ordinary");
        std::printf("%-12s %12s %12s %12s %12s %12s\n", "want", "log_k", "log_zk", "K scaled", "zK scaled", "ratio gap");
        for (unsigned want : {0u, unsigned(argument), unsigned(argument|order)}) {
            std::printf("%-12s", want == 0 ? "value" : want == argument ? "argument" : "arg|order");
            for (int f = 0; f < 5; ++f) {
                std::array<double, 3> samples{};
                for (double& ns : samples) {
                    const auto start = std::chrono::steady_clock::now();
                    for (int i = 0; i < repetitions; ++i)
                        for (double nu : order_panels[panel])
                            for (double z : argument_panels[panel]) {
                                LogK output;
                                const bool ok = functions[f](f == 1 || f == 3 ? std::abs(nu) : nu, z, want, output);
                                sink = sink+(ok ? output.value : 0.0);
                            }
                    const auto stop = std::chrono::steady_clock::now();
                    ns = std::chrono::duration<double, std::nano>(stop-start).count()/(25*repetitions);
                }
                std::sort(samples.begin(), samples.end());
                std::printf(" %12.1f", samples[1]);
            }
            std::puts("");
        }
    }
    if (!std::isfinite(sink)) ++failures;
}

}  // namespace

int main() {
    watch_allocations = true;
    for (const auto& row : references) check_reference(row);
    for (const auto& row : extended_references) {
        check_reference(row);
        if (row.function == 3) check_reference(row, true);
    }
    check_fixed_nig();
    check_ladders();
    check_domains();
    check_extreme_limits();
    check_scaled_identities();
    check_gap_extreme_limits();
    watch_allocations = false;
    if (allocations) {
        std::printf("FAIL evaluation allocated %zu times\n", allocations);
        ++failures;
    }
    print_errors();
    print_timings();
    std::printf("%zu original + %zu extended reference rows; %d failures\n", std::size(references), std::size(extended_references), failures);
    return failures ? 1 : 0;
}
