#include "bounded_direction.hpp"

namespace bd = vlk::experiment::bounded_direction;

#ifdef VLK_BOUNDED_DIRECTION_AUDIT_ONLY
// External callers observe the complete result and output buffer in this audit.
bd::Result bounded_runtime(std::size_t n, bd::Input g, bd::Input h, double radius,
    bd::Output d, void* memory, std::size_t bytes) noexcept
{ return bd::propose(n, g, h, radius, d, memory, bytes); }

bd::Result bounded_fixed2(bd::Input g, bd::Input h, double radius,
    bd::Output d, void* memory, std::size_t bytes) noexcept
{ return bd::propose<2>(2, g, h, radius, d, memory, bytes); }

bd::Result bounded_fixed5(bd::Input g, bd::Input h, double radius,
    bd::Output d, void* memory, std::size_t bytes) noexcept
{ return bd::propose<5>(5, g, h, radius, d, memory, bytes); }

bd::Result bounded_fixed9(bd::Input g, bd::Input h, double radius,
    bd::Output d, void* memory, std::size_t bytes) noexcept
{ return bd::propose<9>(9, g, h, radius, d, memory, bytes); }
#else
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <string>
#include <vector>

static std::size_t allocations = 0;
void* operator new(std::size_t bytes)
{
    ++allocations;
    if (void* memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }
void* operator new(std::size_t bytes, std::align_val_t alignment)
{
    ++allocations;
    void* memory = nullptr;
    if (!posix_memalign(&memory, static_cast<std::size_t>(alignment), bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes, std::align_val_t alignment)
{ return ::operator new(bytes, alignment); }
void operator delete(void* memory, std::align_val_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::align_val_t) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t, std::align_val_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t, std::align_val_t) noexcept { std::free(memory); }

namespace {
constexpr double sentinel = 817.25;

double read_number()
{
    std::string token;
    assert(bool(std::cin >> token));
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    assert(end && *end == '\0');
    return value;
}

bd::Result dispatch(unsigned fixed, std::size_t n, bd::Input g, bd::Input h,
    double radius, bd::Output d, void* memory, std::size_t bytes, unsigned sweeps = 64)
{
    switch (fixed) {
    case 0: return bd::propose(n, g, h, radius, d, memory, bytes, sweeps);
    case 1: return bd::propose<1>(n, g, h, radius, d, memory, bytes, sweeps);
    case 2: return bd::propose<2>(n, g, h, radius, d, memory, bytes, sweeps);
    case 3: return bd::propose<3>(n, g, h, radius, d, memory, bytes, sweeps);
    case 5: return bd::propose<5>(n, g, h, radius, d, memory, bytes, sweeps);
    case 9: return bd::propose<9>(n, g, h, radius, d, memory, bytes, sweeps);
    default: std::abort();
    }
}

void observe(const bd::Result& result, const double* output, std::size_t n,
    std::size_t added)
{
    std::cout.precision(17);
    std::cout << "{\"status\":" << int(result.status) << ",\"method\":" << int(result.method)
        << ",\"fallback\":" << int(result.fallback) << ",\"norm\":" << result.norm
        << ",\"slope\":" << result.slope << ",\"curvature\":" << result.curvature
        << ",\"quadratic_change\":" << result.quadratic_change
        << ",\"positive\":" << result.positive << ",\"negative\":" << result.negative
        << ",\"unresolved\":" << result.unresolved << ",\"resolution\":" << result.resolution
        << ",\"minimum_eigenvalue\":" << result.minimum_eigenvalue
        << ",\"maximum_eigenvalue\":" << result.maximum_eigenvalue
        << ",\"radius_limited\":" << (result.radius_limited ? "true" : "false")
        << ",\"allocations\":" << added << ",\"direction\":[";
    for (std::size_t i = 0; i < n; ++i) {
        if (i) std::cout << ',';
        std::cout << output[i];
    }
    std::cout << "]}\n";
}

void case_from_stream()
{
    std::size_t n; unsigned fixed, sweeps;
    assert(bool(std::cin >> n >> fixed >> sweeps));
    assert(n > 0 && n <= 256);
    const double radius = read_number();
    std::vector<double> g(n), h(n*n), output(n+2, sentinel);
    for (auto& value : g) value = read_number();
    for (auto& value : h) value = read_number();
    const auto original_g = g, original_h = h;
    const auto bytes = bd::make_layout(n).bytes;
    std::vector<double> arena(bytes/sizeof(double)+2, sentinel);
    const auto before = allocations;
    const auto result = dispatch(fixed, n, {g.data(), n}, {h.data(), n*n}, radius,
        {output.data()+1, n}, arena.data()+1, bytes, sweeps);
    const auto added = allocations-before;
    assert(!added && !std::memcmp(g.data(), original_g.data(), n*sizeof(double))
        && !std::memcmp(h.data(), original_h.data(), n*n*sizeof(double)));
    assert(output.front() == sentinel && output.back() == sentinel);
    assert(arena.front() == sentinel && arena.back() == sentinel);
    if (result.status != bd::Status::ok)
        for (auto value : output) assert(value == sentinel);
    if (result.status == bd::Status::invalid_input || result.method == bd::Method::zero_gradient)
        for (auto value : arena) assert(value == sentinel);
    observe(result, output.data()+1, n, added);
}

void guard_controls()
{
    constexpr auto fixed_layout = bd::make_layout(5);
    static_assert(fixed_layout.dimension == 5 && fixed_layout.square == 25 && fixed_layout.bytes == 520);
    assert(bd::make_layout(0).bytes == 0);
    assert(bd::make_layout(std::numeric_limits<std::size_t>::max()).bytes == 0);
    assert(bd::make_layout(static_cast<std::size_t>(CBLAS_INT_MAX)).bytes == 0);
    double g[2] = {1, -2}, h[4] = {2, 0, 0, 3}, d[4];
    const auto bytes = bd::make_layout(2).bytes;
    std::vector<double> arena(bytes/sizeof(double)+4, sentinel);
    bd::Input gv{g, 2}, hv{h, 4}; bd::Output dv{d+1, 2};
    unsigned checks = 0;
    const auto reject = [&](std::size_t n, bd::Input gi, bd::Input hi,
        bd::Output di, void* memory, std::size_t size, double radius,
        bd::Status expected, unsigned fixed = 0, unsigned sweeps = 64) {
        std::fill_n(d, 4, sentinel);
        const auto saved_arena = arena;
        double saved_g[2], saved_h[4];
        std::copy_n(g, 2, saved_g); std::copy_n(h, 4, saved_h);
        const auto before = allocations;
        const auto result = dispatch(fixed, n, gi, hi, radius, di, memory, size, sweeps);
        assert(allocations == before && result.status == expected);
        assert(result.method == bd::Method::none && result.fallback == bd::Fallback::none);
        assert(result.norm == 0 && result.slope == 0 && result.curvature == 0
            && result.quadratic_change == 0 && result.minimum_eigenvalue == 0
            && result.maximum_eigenvalue == 0 && result.resolution == 0
            && result.positive == 0 && result.negative == 0 && result.unresolved == 0
            && !result.radius_limited);
        for (auto value : d) assert(value == sentinel);
        assert(arena == saved_arena);
        assert(!std::memcmp(g, saved_g, sizeof(g)) && !std::memcmp(h, saved_h, sizeof(h)));
        ++checks;
    };
    reject(0, gv, hv, dv, arena.data()+1, bytes, 1, bd::Status::invalid_input);
    reject(std::numeric_limits<std::size_t>::max(), gv, hv, dv, arena.data()+1, bytes, 1, bd::Status::invalid_input);
    reject(2, {nullptr, 2}, hv, dv, arena.data()+1, bytes, 1, bd::Status::invalid_input);
    reject(2, gv, {nullptr, 4}, dv, arena.data()+1, bytes, 1, bd::Status::invalid_input);
    reject(2, gv, hv, {nullptr, 2}, arena.data()+1, bytes, 1, bd::Status::invalid_input);
    reject(2, gv, hv, dv, nullptr, bytes, 1, bd::Status::invalid_input);
    reject(2, {g, 1}, hv, dv, arena.data()+1, bytes, 1, bd::Status::short_buffer);
    reject(2, gv, {h, 3}, dv, arena.data()+1, bytes, 1, bd::Status::short_buffer);
    reject(2, gv, hv, {d+1, 1}, arena.data()+1, bytes, 1, bd::Status::short_buffer);
    reject(2, gv, hv, dv, arena.data()+1, bytes-1, 1, bd::Status::short_buffer);
    reject(2, gv, hv, dv, reinterpret_cast<char*>(arena.data()+1)+1, bytes, 1, bd::Status::invalid_input);
    reject(2, gv, hv, {g, 2}, arena.data()+1, bytes, 1, bd::Status::overlap);
    reject(2, gv, hv, {h, 2}, arena.data()+1, bytes, 1, bd::Status::overlap);
    reject(2, gv, hv, {arena.data()+1, 2}, arena.data()+1, bytes, 1, bd::Status::overlap);
    reject(2, {arena.data()+1, 2}, hv, dv, arena.data()+1, bytes, 1, bd::Status::overlap);
    reject(2, gv, {arena.data()+1, 4}, dv, arena.data()+1, bytes, 1, bd::Status::overlap);
    reject(2, gv, hv, dv, arena.data()+1, bytes, 0, bd::Status::invalid_input);
    reject(2, gv, hv, dv, arena.data()+1, bytes, -1, bd::Status::invalid_input);
    reject(2, gv, hv, dv, arena.data()+1, bytes, std::numeric_limits<double>::infinity(), bd::Status::invalid_input);
    reject(2, gv, hv, dv, arena.data()+1, bytes, std::numeric_limits<double>::quiet_NaN(), bd::Status::invalid_input);
    reject(2, gv, hv, dv, arena.data()+1, bytes, 1, bd::Status::invalid_input, 3);
    reject(2, gv, hv, dv, arena.data()+1, bytes, 1, bd::Status::invalid_input, 0, 0);
    auto* overflow = reinterpret_cast<double*>(std::numeric_limits<std::uintptr_t>::max()-7);
    reject(2, {overflow, 2}, hv, dv, arena.data()+1, bytes, 1, bd::Status::invalid_input);

    reject(2, gv, {overflow, 4}, dv, arena.data()+1, bytes, 1, bd::Status::invalid_input);
    reject(2, gv, hv, {overflow, 2}, arena.data()+1, bytes, 1, bd::Status::invalid_input);
    reject(2, gv, hv, dv, overflow, bytes, 1, bd::Status::invalid_input);
    reject(2, {reinterpret_cast<double*>(reinterpret_cast<char*>(g)+1), 2},
        hv, dv, arena.data()+1, bytes, 1, bd::Status::invalid_input);
    reject(2, gv, {reinterpret_cast<double*>(reinterpret_cast<char*>(h)+1), 4},
        dv, arena.data()+1, bytes, 1, bd::Status::invalid_input);
    reject(2, gv, hv, {reinterpret_cast<double*>(reinterpret_cast<char*>(d+1)+1), 2},
        arena.data()+1, bytes, 1, bd::Status::invalid_input);
    reject(2, gv, hv, {g+1, 2}, arena.data()+1, bytes, 1, bd::Status::overlap);
    reject(2, gv, hv, {h+3, 2}, arena.data()+1, bytes, 1, bd::Status::overlap);
    reject(2, {d+2, 2}, hv, dv, arena.data()+1, bytes, 1, bd::Status::overlap);
    reject(2, gv, {d, 4}, dv, arena.data()+1, bytes, 1, bd::Status::overlap);
    // Both ends of the arena, plus its unused declared suffix, are protected.
    auto* start = arena.data()+1;
    const auto length = bytes/sizeof(double);
    for (auto* edge : {start-1, start+length-1}) {
        reject(2, {edge, 2}, hv, dv, start, bytes, 1, bd::Status::overlap);
        reject(2, gv, {edge, 4}, dv, start, bytes, 1, bd::Status::overlap);
        reject(2, gv, hv, {edge, 2}, start, bytes, 1, bd::Status::overlap);
    }
    reject(2, {start+length, 2}, hv, dv, start, bytes+sizeof(double), 1, bd::Status::overlap);
    reject(2, gv, {start+length, 4}, dv, start, bytes+sizeof(double), 1, bd::Status::overlap);
    reject(2, gv, hv, {start+length, 2}, start, bytes+sizeof(double), 1, bd::Status::overlap);
    std::cout << "{\"guard_checks\":" << checks << ",\"allocations\":0}\n";
}

void exact_arena_controls()
{
    unsigned checks = 0;
    for (std::size_t n : {1u, 2u, 5u, 9u, 13u, 37u}) {
        const auto layout = bd::make_layout(n);
        assert(layout.dimension == n && layout.square == n*n
            && layout.bytes == sizeof(double)*(2*n*n+3*n));
        for (unsigned fixed : {0u, static_cast<unsigned>(n <= 9 ? n : 0)}) {
            for (bool short_arena : {false, true}) {
                // The allocation itself, not just the passed count, is exact/short.
                std::vector<std::byte> arena(layout.bytes-short_arena, std::byte{0x5a});
                const auto saved_arena = arena;
                std::vector<double> g(n, 1), h(n*n, 0), d(n, sentinel);
                for (std::size_t i = 0; i < n; ++i) h[i*n+i] = 2;
                assert(reinterpret_cast<std::uintptr_t>(arena.data()) % alignof(double) == 0);
                const auto before = allocations;
                const auto result = dispatch(fixed, n, {g.data(), n}, {h.data(), n*n}, 1,
                    {d.data(), n}, arena.data(), arena.size());
                assert(allocations == before);
                if (short_arena) {
                    assert(result.status == bd::Status::short_buffer && arena == saved_arena);
                    for (double value : d) assert(value == sentinel);
                } else {
                    assert(result.status == bd::Status::ok && result.norm <= 1
                        && result.slope < 0 && result.quadratic_change < 0);
                }
                observe(result, d.data(), n, 0);
                ++checks;
            }
        }
    }
    std::cout << "{\"exact_arena_checks\":" << checks << ",\"allocations\":0}\n";
}

void allowed_alias_controls()
{
    const auto bytes = bd::make_layout(2).bytes;
    const auto length = bytes/sizeof(double);
    unsigned checks = 0;
    // G is H's first row; both read-only prefixes may overlap exactly or partly.
    for (unsigned fixed : {0u, 2u}) for (std::size_t offset : {0u, 1u}) {
        double h[] = {4, 1, 1, 3}, d[2];
        std::vector<double> arena(length);
        const auto before = allocations;
        const auto result = dispatch(fixed, 2, {h+offset, 2}, {h, 4}, 1,
            {d, 2}, arena.data(), bytes);
        assert(allocations == before && result.status == bd::Status::ok);
        assert(h[0] == 4 && h[1] == 1 && h[2] == 1 && h[3] == 3);
        observe(result, d, 2, 0);
        ++checks;
    }
    // Adjacent active prefixes are allowed even when declared counts are larger.
    // Exercise inputs/output immediately before and immediately after the arena.
    for (bool before_arena : {false, true}) {
        std::vector<double> storage(length+8, sentinel);
        double* arena = storage.data()+(before_arena ? 8 : 0);
        double* g = storage.data()+(before_arena ? 0 : length);
        double* h = g+2;
        double* d = h+4;
        g[0] = 1; g[1] = -2;
        h[0] = 4; h[1] = h[2] = 1; h[3] = 3;
        const auto before = allocations;
        const auto result = bd::propose(2, {g, 8}, {h, 6}, 1, {d, 2}, arena, bytes);
        assert(allocations == before && result.status == bd::Status::ok);
        assert(g[0] == 1 && g[1] == -2 && h[0] == 4 && h[1] == 1 && h[2] == 1 && h[3] == 3);
        observe(result, d, 2, 0);
        ++checks;
    }
    std::cout << "{\"alias_checks\":" << checks << ",\"allocations\":0}\n";
}
} // namespace

int main(int argc, char** argv)
{
    assert(argc == 2);
    const std::string mode(argv[1]);
    if (mode == "guards") guard_controls();
    else if (mode == "arenas") exact_arena_controls();
    else if (mode == "aliases") allowed_alias_controls();
    else { assert(mode == "case"); case_from_stream(); }
}
#endif
