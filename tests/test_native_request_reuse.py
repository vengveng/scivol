"""Exact-request cache contracts, sanitizer checks and native solver integration."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import struct

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "scivol" / "_native"

# Keep the standalone source here: these tests need neither localdev nor an extension.
SOURCE = r'''
#include "exact_request_reuse.hpp"
namespace reuse = vlk::experiment::request_reuse;

#ifdef VLK_REUSE_AUDIT_ONLY
// Runtime inputs and externally visible outputs prevent dead-code elimination.
bool cache_bind(reuse::Cache& c, reuse::Layout l, void* arena, std::size_t bytes)
{ return c.bind(l, arena, bytes); }
bool cache_get(reuse::Cache& c, const double* p, unsigned m, unsigned coordinate,
    reuse::Outputs out)
{ return c.get(p, m, coordinate, out); }
void cache_put(reuse::Cache& c, const double* p, unsigned m, unsigned coordinate,
    reuse::Values out, int status, unsigned valid)
{ c.put(p, m, coordinate, out, status, valid); }
bool bank_bind(reuse::Bank& b, std::size_t width, std::size_t buckets,
    void* arena, std::size_t bytes)
{ return b.bind(width, buckets, arena, bytes); }
bool bank_get(reuse::Bank& b, const double* p, unsigned m, unsigned coordinate,
    reuse::Outputs out)
{ return b.get(p, m, coordinate, out); }
void bank_put(reuse::Bank& b, const double* p, unsigned m, unsigned coordinate,
    reuse::Values out, int status, unsigned valid)
{ b.put(p, m, coordinate, out, status, valid); }
std::size_t bank_bytes(std::size_t width, std::size_t buckets)
{ return reuse::Bank::bytes(width, buckets); }
reuse::Layout cache_layout(std::size_t width, std::size_t buckets, bool curvature)
{ return reuse::layout(width, buckets, curvature); }
#else
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

static std::size_t allocations = 0;
void* operator new(std::size_t bytes)
{
    ++allocations;
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    std::abort();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t bytes, std::align_val_t alignment)
{
    ++allocations;
    void* p = nullptr;
    if (!posix_memalign(&p, static_cast<std::size_t>(alignment), bytes ? bytes : 1)) return p;
    std::abort();
}
void* operator new[](std::size_t bytes, std::align_val_t alignment)
{ return ::operator new(bytes, alignment); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

struct NoHeap {
    std::size_t before = allocations;
    ~NoHeap() { assert(allocations == before); }
};

constexpr std::uint64_t poison_bits = 0x7ff8000000000042ull;
constexpr std::uint64_t guard_bits = 0xa55aa55aa55aa55aull;
double poison() { return std::bit_cast<double>(poison_bits); }
bool same(double a, double b)
{ return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

struct Arena {
    std::size_t bytes;
    std::vector<std::uint64_t> words;
    explicit Arena(std::size_t size) : bytes(size), words(size/8u + 2u, guard_bits)
    { assert(size && size % 8u == 0); }
    void* data() { return words.data() + 1; }
    void guards() const { assert(words.front() == guard_bits && words.back() == guard_bits); }
};

struct Block {
    std::vector<double> data;
    explicit Block(std::size_t size) : data(size + 2u, poison()) {}
    double* begin() { return data.data() + 1; }
    const double* begin() const { return data.data() + 1; }
    std::size_t size() const { return data.size() - 2u; }
    void clear() { std::fill(data.begin(), data.end(), poison()); }
    void guards() const { assert(same(data.front(), poison()) && same(data.back(), poison())); }
    void expect(const Block& source, bool requested) const
    {
        guards(); source.guards();
        for (std::size_t i = 0; i < size(); ++i)
            assert(same(begin()[i], requested ? source.begin()[i] : poison()));
    }
};

struct Buffers {
    Block f, g, h;
    explicit Buffers(std::size_t width) : f(1), g(width), h(width*width) {}
    void fill(unsigned mask, double tag = 0)
    {
        clear();
        for (auto [block, bit] : {std::pair{&f, 1u}, {&g, 2u}, {&h, 4u}})
            if (mask & bit)
                for (std::size_t i = 0; i < block->size(); ++i)
                    block->begin()[i] = (i + bit) % 3u == 0
                        ? -0. : tag + double(i + bit) + 0.25;
    }
    void clear() { f.clear(); g.clear(); h.clear(); }
    reuse::Values values(unsigned mask, bool null_unused = true) const
    { return {mask&1u || !null_unused ? f.begin() : nullptr,
        mask&2u || !null_unused ? g.begin() : nullptr,
        mask&4u || !null_unused ? h.begin() : nullptr}; }
    reuse::Outputs outputs(unsigned mask, bool null_unused = true)
    { return {mask&1u || !null_unused ? f.begin() : nullptr,
        mask&2u || !null_unused ? g.begin() : nullptr,
        mask&4u || !null_unused ? h.begin() : nullptr}; }
    void expect(const Buffers& source, unsigned mask) const
    { f.expect(source.f, mask&1u); g.expect(source.g, mask&2u); h.expect(source.h, mask&4u); }
};

void all_masks(std::size_t width)
{
    Arena arena(reuse::Bank::bytes(width, 2));
    Block point(width);
    Buffers source(width), target(width), original(width);
    for (std::size_t i = 0; i < width; ++i) point.begin()[i] = double(i) + 0.5;
    const auto point_snapshot = point.data;
    reuse::Bank bank;
    NoHeap measured;
    assert(bank.bind(width, 2, arena.data(), arena.bytes));
    for (unsigned mask = 1; mask <= 7; ++mask) {
        source.fill(mask, 10*mask);
        original = source;
        assert(!bank.get(point.begin(), mask, 0, target.outputs(mask)));
        target.expect(source, 0);
        bank.put(point.begin(), mask, 0, source.values(mask), 0, mask);
        assert(bank.get(point.begin(), mask, 0, target.outputs(mask)));
        target.expect(source, mask);
        source.expect(original, mask);
        target.clear();
    }
    assert(bank.queries() == 14 && bank.hits() == 7);
    for (unsigned mask = 1; mask <= 7; ++mask) {
        source.fill(mask, 10*mask);
        assert(bank.get(point.begin(), mask, 0, target.outputs(mask, false)));
        target.expect(source, mask);
        target.clear();
    }
    assert(bank.queries() == 21 && bank.hits() == 14);
    assert(std::memcmp(point.data.data(), point_snapshot.data(), point.data.size()*sizeof(double)) == 0);
    point.guards(); arena.guards();
}

void exact_keys()
{
    constexpr std::size_t width = 3;
    Arena arena(reuse::Bank::bytes(width, 1));
    Buffers source(width), target(width);
    reuse::Bank bank;
    std::array<double, width> point{0., 2., -3.}, changed;
    source.fill(7);
    NoHeap measured;
    for (unsigned mask = 1; mask <= 7; ++mask) {
        assert(bank.bind(width, 1, arena.data(), arena.bytes));
        bank.put(point.data(), mask, 0, source.values(mask), 0, mask);
        for (std::size_t i = 0; i < width; ++i) {
            changed = point;
            changed[i] = std::nextafter(changed[i], std::numeric_limits<double>::infinity());
            assert(!bank.get(changed.data(), mask, 0, target.outputs(mask)));
            target.expect(source, 0);
        }
        changed = point; changed[0] = -0.;
        assert(!bank.get(changed.data(), mask, 0, target.outputs(mask)));
        assert(!bank.get(point.data(), mask, 1, target.outputs(mask)));
        target.expect(source, 0);
        bank.put(changed.data(), mask, 1, source.values(mask), 0, mask);
        assert(bank.get(changed.data(), mask, 1, target.outputs(mask)));
        target.expect(source, mask); target.clear();
        assert(!bank.get(changed.data(), mask, 0, target.outputs(mask)));
        changed = point;
        assert(bank.get(changed.data(), mask, 0, target.outputs(mask)));
        target.expect(source, mask); target.clear();
        assert(bank.queries() == 8 && bank.hits() == 2);
    }
    arena.guards();
}

void mask_isolation()
{
    Arena arena(reuse::Bank::bytes(3, 1));
    Buffers source(3), target(3);
    reuse::Bank bank;
    double point[]{1., 2., 3.};
    NoHeap measured;
    for (unsigned stored = 1; stored <= 7; ++stored) {
        assert(bank.bind(3, 1, arena.data(), arena.bytes));
        source.fill(stored);
        bank.put(point, stored, 0, source.values(stored), 0, stored);
        assert(bank.queries() == 0 && bank.hits() == 0);
        for (unsigned queried = 1; queried <= 7; ++queried) {
            assert(bank.get(point, queried, 0, target.outputs(queried)) == (stored == queried));
            target.expect(source, stored == queried ? stored : 0);
            target.clear();
        }
        assert(bank.queries() == 7 && bank.hits() == 1);
    }
    arena.guards();
}

void rejected_results()
{
    Arena arena(reuse::Bank::bytes(3, 1));
    Buffers source(3), target(3);
    reuse::Bank bank;
    double point[]{1., 2., 3.}, resident[]{4., 5., 6.};
    source.fill(7);
    NoHeap measured;
    for (unsigned mask = 1; mask <= 7; ++mask) {
        for (int status : {-1, 1, 17}) {
            assert(bank.bind(3, 1, arena.data(), arena.bytes));
            bank.put(point, mask, 0, {nullptr, nullptr, nullptr}, status, mask);
            assert(!bank.get(point, mask, 0, target.outputs(mask)));
            target.expect(source, 0);
        }
        for (unsigned valid = 0; valid < 16; ++valid) {
            if (valid == mask) continue;
            assert(bank.bind(3, 1, arena.data(), arena.bytes));
            bank.put(resident, mask, 0, source.values(mask), 0, mask);
            bank.put(point, mask, 0, {nullptr, nullptr, nullptr}, 0, valid);
            assert(!bank.get(point, mask, 0, target.outputs(mask)));
            target.expect(source, 0);
            assert(bank.get(resident, mask, 0, target.outputs(mask)));
            target.expect(source, mask); target.clear();
        }
    }
    assert(bank.bind(3, 1, arena.data(), arena.bytes));
    for (unsigned mask : {0u, 8u, 9u, 15u, ~0u}) {
        bank.put(nullptr, mask, 0, {nullptr, nullptr, nullptr}, 0, mask);
        assert(!bank.get(nullptr, mask, 0, {nullptr, nullptr, nullptr}));
    }
    for (unsigned coordinate : {2u, ~0u}) {
        bank.put(nullptr, 7, coordinate, {nullptr, nullptr, nullptr}, 0, 7);
        assert(!bank.get(nullptr, 7, coordinate, {nullptr, nullptr, nullptr}));
    }
    assert(bank.queries() == 7 && bank.hits() == 0);
    arena.guards();
}

void nonfinite_results()
{
    Arena arena(reuse::Bank::bytes(3, 1));
    Buffers source(3), target(3);
    reuse::Bank bank;
    double point[]{1., 2., 3.};
    const double bad[]{poison(), std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    NoHeap measured;
    for (unsigned mask = 1; mask <= 7; ++mask) {
        for (double invalid : bad) {
            for (std::size_t i = 0; i < 3; ++i) {
                source.fill(mask);
                assert(bank.bind(3, 1, arena.data(), arena.bytes));
                const double saved = point[i]; point[i] = invalid;
                bank.put(point, mask, 0, source.values(mask), 0, mask);
                assert(!bank.get(point, mask, 0, target.outputs(mask)));
                point[i] = saved;
                assert(!bank.get(point, mask, 0, target.outputs(mask)));
                target.expect(source, 0);
            }
            for (auto [block, bit] : {std::pair{&source.f, 1u},
                    {&source.g, 2u}, {&source.h, 4u}}) {
                if (!(mask & bit)) continue;
                for (std::size_t i = 0; i < block->size(); ++i) {
                    source.fill(mask);
                    assert(bank.bind(3, 1, arena.data(), arena.bytes));
                    block->begin()[i] = invalid;
                    bank.put(point, mask, 0, source.values(mask), 0, mask);
                    assert(!bank.get(point, mask, 0, target.outputs(mask)));
                    target.expect(source, 0);
                }
            }
        }
        // Unrequested poison must neither invalidate a result nor be read/written.
        source.fill(mask);
        assert(bank.bind(3, 1, arena.data(), arena.bytes));
        bank.put(point, mask, 0, source.values(mask, false), 0, mask);
        assert(bank.get(point, mask, 0, target.outputs(mask, false)));
        target.expect(source, mask); target.clear();
    }
    arena.guards();
}

void copied_lifetime()
{
    Arena arena(reuse::Bank::bytes(3, 1)), other(reuse::Bank::bytes(3, 1));
    Buffers source(3), expected(3), target(3);
    reuse::Bank bank, independent;
    double point[]{1., 2., 3.}, copy[]{1., 2., 3.};
    source.fill(7, 10); expected = source;
    NoHeap measured;
    assert(bank.bind(3, 1, arena.data(), arena.bytes));
    bank.put(point, 7, 0, source.values(7), 0, 7);
    point[0] = 4.; source.fill(7, 20);
    assert(bank.get(copy, 7, 0, target.outputs(7)));
    target.expect(expected, 7); target.clear();
    assert(!bank.get(point, 7, 0, target.outputs(7)));
    assert(independent.bind(3, 1, other.data(), other.bytes));
    assert(!independent.get(copy, 7, 0, target.outputs(7)));
    independent.put(copy, 7, 0, source.values(7), 0, 7);
    assert(bank.get(copy, 7, 0, target.outputs(7)));
    target.expect(expected, 7); target.clear();
    assert(bank.bind(3, 1, arena.data(), arena.bytes));
    assert(bank.queries() == 0 && bank.hits() == 0);
    assert(!bank.get(copy, 7, 0, target.outputs(7)));
    bank.put(copy, 7, 0, source.values(7), 0, 7);
    assert(bank.get(copy, 7, 0, target.outputs(7)));
    target.expect(source, 7); target.clear();
    assert(bank.bind(1, 1, arena.data(), arena.bytes));
    assert(!bank.get(copy, 7, 0, target.outputs(7)));
    assert(bank.bind(3, 1, other.data(), other.bytes));
    assert(!bank.get(copy, 7, 0, target.outputs(7)));
    arena.guards(); other.guards();
}

void eviction()
{
    const auto layout = reuse::layout(1, 1);
    Arena arena(layout.bytes), bank_arena(reuse::Bank::bytes(1, 2));
    reuse::Cache cache;
    reuse::Bank bank;
    Buffers source(1), target(1);
    double point[]{0.};
    source.fill(7);
    NoHeap measured;
    assert(cache.bind(layout, arena.data(), arena.bytes));
    for (unsigned i = 0; i < reuse::ways; ++i) {
        point[0] = double(i); cache.put(point, 1, 0, source.values(1), 0, 1);
    }
    point[0] = 0.; assert(cache.get(point, 1, 0, target.outputs(1)));
    target.expect(source, 1); target.clear();
    point[0] = 8.; cache.put(point, 1, 0, source.values(1), 0, 1);
    point[0] = 1.; assert(!cache.get(point, 1, 0, target.outputs(1)));
    target.expect(source, 0);
    for (unsigned i : {0u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
        point[0] = double(i); assert(cache.get(point, 1, 0, target.outputs(1)));
        target.expect(source, 1); target.clear();
    }
    // Curvature traffic has its own eight-way cache; first-order traffic cannot evict it.
    assert(bank.bind(1, 2, bank_arena.data(), bank_arena.bytes));
    point[0] = -1.;
    for (unsigned mask : {1u, 2u, 3u}) bank.put(point, mask, 0, source.values(mask), 0, mask);
    for (unsigned i = 0; i < 9; ++i) {
        point[0] = double(i); bank.put(point, 4, 0, source.values(4), 0, 4);
    }
    point[0] = 0.; assert(!bank.get(point, 4, 0, target.outputs(4)));
    target.expect(source, 0);
    point[0] = -1.;
    for (unsigned mask : {1u, 2u, 3u}) {
        assert(bank.get(point, mask, 0, target.outputs(mask)));
        target.expect(source, mask); target.clear();
    }
    for (unsigned i = 0; i < 128; ++i) {
        point[0] = double(i); bank.put(point, 3, 0, source.values(3), 0, 3);
    }
    point[0] = 8.; assert(bank.get(point, 4, 0, target.outputs(4)));
    target.expect(source, 4);
    assert(bank.queries() == 5 && bank.hits() == 4);
    arena.guards(); bank_arena.guards();
}

void unbound_and_failed_rebind()
{
    const auto layout = reuse::layout(3, 1);
    Arena arena(reuse::Bank::bytes(3, 1)), cache_arena(layout.bytes);
    const auto untouched = arena.words;
    auto bank_snapshot = arena.words, cache_snapshot = cache_arena.words;
    Buffers source(3), target(3);
    reuse::Cache cache;
    reuse::Bank bank;
    double point[]{1., 2., 3.};
    source.fill(7);
    NoHeap measured;
    for (unsigned mask = 1; mask <= 7; ++mask) {
        assert(!cache.get(nullptr, mask, 0, target.outputs(mask)));
        cache.put(nullptr, mask, 0, {nullptr, nullptr, nullptr}, 0, mask);
        assert(!bank.get(nullptr, mask, 0, target.outputs(mask)));
        bank.put(nullptr, mask, 0, {nullptr, nullptr, nullptr}, 0, mask);
        target.expect(source, 0);
    }
    assert(arena.words == untouched);
    assert(bank.queries() == 7 && bank.hits() == 0);
    for (unsigned failure = 0; failure < 4; ++failure) {
        assert(cache.bind(layout, cache_arena.data(), cache_arena.bytes));
        assert(bank.bind(3, 1, arena.data(), arena.bytes));
        cache.put(point, 7, 0, source.values(7), 0, 7);
        assert(cache.get(point, 7, 0, target.outputs(7)));
        target.expect(source, 7); target.clear();
        bank.put(point, 1, 0, source.values(1), 0, 1);
        bank.put(point, 4, 0, source.values(4), 0, 4);
        assert(bank.get(point, 1, 0, target.outputs(1))); target.clear();
        assert(bank.get(point, 4, 0, target.outputs(4))); target.clear();
        assert(bank.queries() == 2 && bank.hits() == 2);
        bank_snapshot = arena.words; cache_snapshot = cache_arena.words;
        void* memory = failure == 0 ? nullptr : failure == 1
            ? static_cast<std::byte*>(arena.data()) + 1 : arena.data();
        const std::size_t size = failure == 2 ? 1u : arena.bytes;
        void* cache_memory = failure == 0 ? nullptr : failure == 1
            ? static_cast<std::byte*>(cache_arena.data()) + 1 : cache_arena.data();
        assert(!cache.bind(failure == 3 ? reuse::Layout{} : layout, cache_memory,
            failure == 2 ? 1u : cache_arena.bytes));
        assert(!bank.bind(failure == 3 ? 0u : 3u, 1, memory, size));
        assert(bank.queries() == 0 && bank.hits() == 0);
        assert(arena.words == bank_snapshot && cache_arena.words == cache_snapshot);
        for (unsigned mask = 1; mask <= 7; ++mask) {
            assert(!cache.get(nullptr, mask, 0, target.outputs(mask)));
            cache.put(nullptr, mask, 0, {nullptr, nullptr, nullptr}, 0, mask);
            assert(!bank.get(nullptr, mask, 0, target.outputs(mask)));
            bank.put(nullptr, mask, 0, {nullptr, nullptr, nullptr}, 0, mask);
            target.expect(source, 0);
        }
        assert(bank.queries() == 7 && bank.hits() == 0);
        assert(arena.words == bank_snapshot && cache_arena.words == cache_snapshot);
    }
    arena.guards(); cache_arena.guards();
}

void packed_null(const char* operation)
{
    const auto layout = reuse::layout(3, 1);
    Arena arena(layout.bytes);
    reuse::Cache cache;
    double point[]{1., 2., 3.}, packed[13]{};
    const bool get = !std::strcmp(operation, "get");
    assert(get || !std::strcmp(operation, "put"));
    NoHeap measured;
    for (unsigned stage = 0; stage < 3; ++stage) {
        if (stage) {
            assert(cache.bind(layout, arena.data(), arena.bytes));
            cache.put(point, 7, 0, packed, 0, 7);
            if (stage == 1) assert(!cache.bind(layout, nullptr, arena.bytes));
        }
        const double* input = stage == 2 ? point : nullptr;
        for (unsigned mask = 1; mask <= 7; ++mask) {
            if (get) assert(!cache.get(input, mask, 0, static_cast<double*>(nullptr)));
            else cache.put(input, mask, 0, static_cast<const double*>(nullptr), 0, mask);
        }
        if (stage == 2) assert(cache.get(point, 7, 0, packed));
    }
    arena.guards();
}

void planner_and_arenas()
{
    constexpr auto fixed = reuse::layout(3, 2);
    static_assert(fixed.width == 3 && fixed.square == 9 && fixed.outputs == 13
        && fixed.stride == 16 && fixed.buckets == 2 && fixed.entries == 16
        && fixed.supported == 7);
    constexpr auto first = reuse::layout(3, 2, false);
    static_assert(first.outputs == 4 && first.stride == 7 && first.supported == 3);
    const auto max = std::numeric_limits<std::size_t>::max();
    for (auto [width, buckets] : {std::pair{0u, std::size_t{1}}, {1u, std::size_t{0}},
            {3u, std::size_t{3}}, {1u, max/2u + 1u}, {1u, max/16u + 1u},
            {1u, max/128u + 1u}}) {
        assert(reuse::layout(width, buckets).bytes == 0);
        assert(reuse::layout(width, buckets, false).bytes == 0);
        assert(reuse::Bank::bytes(width, buckets) == 0);
    }
    for (std::size_t width : {max, max/2u, std::size_t{1} << (sizeof(std::size_t)*4u)}) {
        assert(reuse::layout(width, 1).bytes == 0);
        assert(reuse::layout(width, 1, false).bytes == 0);
        assert(reuse::Bank::bytes(width, 1) == 0);
    }
    for (std::size_t width : {1u, 3u, 17u}) {
        Buffers source(width), target(width);
        Block point(width); source.fill(7);
        std::fill_n(point.begin(), width, 1.);
        for (std::size_t buckets : {1u, 2u, 8u}) {
            const auto full = reuse::layout(width, buckets);
            const auto small = reuse::layout(width, buckets, false);
            const auto bytes = reuse::Bank::bytes(width, buckets);
            assert(full.bytes == full.entries*(sizeof(reuse::Entry) + full.stride*sizeof(double)));
            assert(small.bytes == small.entries*(sizeof(reuse::Entry) + small.stride*sizeof(double)));
            assert(bytes == small.bytes + reuse::layout(width, 1).bytes);
            std::vector<std::byte> exact(bytes), short_arena(bytes - 1u);
            std::vector<std::byte> exact_first(small.bytes), short_first(small.bytes - 1u);
            std::vector<std::byte> exact_full(full.bytes), short_full(full.bytes - 1u);
            Arena guarded(bytes), guarded_first(small.bytes), guarded_full(full.bytes);
            const auto untouched = guarded.words;
            const auto untouched_first = guarded_first.words, untouched_full = guarded_full.words;
            reuse::Cache cache;
            reuse::Bank bank;
            NoHeap measured;
            assert(!bank.bind(width, buckets, short_arena.data(), short_arena.size()));
            assert(!bank.bind(width, buckets, guarded.data(), bytes - 1u));
            assert(!bank.bind(width, buckets, static_cast<std::byte*>(guarded.data()) + 1, bytes));
            assert(guarded.words == untouched);
            assert(bank.bind(width, buckets, exact.data(), exact.size()));
            for (unsigned mask = 1; mask <= 7; ++mask) {
                bank.put(point.begin(), mask, 0, source.values(mask), 0, mask);
                assert(bank.get(point.begin(), mask, 0, target.outputs(mask)));
                target.expect(source, mask); target.clear();
            }
            for (auto layout : {small, full}) {
                auto& exact_cache = layout.supported == 3 ? exact_first : exact_full;
                auto& short_cache = layout.supported == 3 ? short_first : short_full;
                auto& guarded_cache = layout.supported == 3 ? guarded_first : guarded_full;
                const auto& untouched_cache = layout.supported == 3 ? untouched_first : untouched_full;
                assert(!cache.bind(layout, short_cache.data(), short_cache.size()));
                assert(!cache.bind(layout, guarded_cache.data(), layout.bytes - 1u));
                assert(!cache.bind(layout, static_cast<std::byte*>(guarded_cache.data()) + 1, layout.bytes));
                assert(guarded_cache.words == untouched_cache);
                assert(cache.bind(layout, exact_cache.data(), exact_cache.size()));
                for (unsigned mask = 1; mask <= 7; ++mask) {
                    cache.put(point.begin(), mask, 0, source.values(mask), 0, mask);
                    const bool supported = !(mask & ~layout.supported);
                    assert(cache.get(point.begin(), mask, 0, target.outputs(mask)) == supported);
                    target.expect(source, supported ? mask : 0); target.clear();
                }
                assert(cache.bind(layout, guarded_cache.data(), layout.bytes));
                cache.put(point.begin(), 3, 0, source.values(3), 0, 3);
                assert(cache.get(point.begin(), 3, 0, target.outputs(3)));
                target.expect(source, 3); target.clear();
                guarded_cache.guards();
            }
        }
    }
}

void packed_compatibility()
{
    const auto layout = reuse::layout(3, 1);
    Arena arena(layout.bytes);
    reuse::Cache cache;
    double point[]{1., 2., 3.};
    Block source(layout.outputs), target(layout.outputs);
    for (std::size_t i = 0; i < source.size(); ++i)
        source.begin()[i] = i % 3u == 0 ? -0. : double(i);
    NoHeap measured;
    for (unsigned mask = 1; mask <= 7; ++mask) {
        assert(cache.bind(layout, arena.data(), arena.bytes));
        cache.put(point, mask, 0, source.begin(), 0, mask);
        assert(cache.get(point, mask, 0, target.begin()));
        for (std::size_t i = 0; i < source.size(); ++i) {
            const unsigned bit = i == 0 ? 1u : i < 4 ? 2u : 4u;
            assert(same(target.begin()[i], mask & bit ? source.begin()[i] : poison()));
        }
        source.guards(); target.guards(); target.clear();
    }
    arena.guards();
}

void finite_extremes()
{
    Arena arena(reuse::Bank::bytes(3, 1));
    Buffers source(3), target(3);
    reuse::Bank bank;
    const double maximum = std::numeric_limits<double>::max();
    const double minimum = std::numeric_limits<double>::denorm_min();
    double point[]{maximum, -maximum, minimum};
    const double finite[]{0., -0., minimum, -minimum, maximum, -maximum};
    NoHeap measured;
    assert(bank.bind(3, 1, arena.data(), arena.bytes));
    for (unsigned mask = 1; mask <= 7; ++mask) {
        source.fill(mask);
        for (auto [block, bit] : {std::pair{&source.f, 1u},
                {&source.g, 2u}, {&source.h, 4u}})
            if (mask & bit)
                for (std::size_t i = 0; i < block->size(); ++i)
                    block->begin()[i] = finite[(i + mask) % 6u];
        bank.put(point, mask, 0, source.values(mask), 0, mask);
        assert(bank.get(point, mask, 0, target.outputs(mask)));
        target.expect(source, mask); target.clear();
    }
    assert(bank.queries() == 7 && bank.hits() == 7);
    arena.guards();
}

int main(int argc, char** argv)
{
    assert(argc >= 2);
    if (!std::strcmp(argv[1], "masks")) {
        assert(argc == 3); all_masks(std::strtoul(argv[2], nullptr, 10));
    }
    else if (!std::strcmp(argv[1], "keys")) exact_keys();
    else if (!std::strcmp(argv[1], "isolation")) mask_isolation();
    else if (!std::strcmp(argv[1], "rejected")) rejected_results();
    else if (!std::strcmp(argv[1], "nonfinite")) nonfinite_results();
    else if (!std::strcmp(argv[1], "lifetime")) copied_lifetime();
    else if (!std::strcmp(argv[1], "eviction")) eviction();
    else if (!std::strcmp(argv[1], "unbound")) unbound_and_failed_rebind();
    else if (!std::strcmp(argv[1], "packed_null")) {
        assert(argc == 3); packed_null(argv[2]);
    }
    else if (!std::strcmp(argv[1], "planner")) planner_and_arenas();
    else if (!std::strcmp(argv[1], "packed")) packed_compatibility();
    else if (!std::strcmp(argv[1], "extremes")) finite_extremes();
    else std::abort();
    std::puts("contract passed; request allocations=0");
}
#endif
'''


def compiler_and_flags():
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required for request-reuse contracts")
    frame_flags = (
        ["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
        if "clang" in Path(compiler).name
        else ["-Werror=frame-larger-than=16384"]
    )
    return compiler, ["-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
        "-march=native", "-fno-exceptions", "-fno-rtti", *frame_flags,
        "-fstack-usage", "-I", str(NATIVE)]


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def check_frames(directory):
    frames = []
    for path in directory.glob("*.su"):
        for line in path.read_text().splitlines():
            fields = line.split("\t")
            assert len(fields) == 3 and fields[2] == "static", line
            assert int(fields[1]) <= 16384, line
            frames.append(int(fields[1]))
    assert frames, "Compiler did not emit the required stack audit"


@pytest.fixture(scope="module", params=["production", "sanitized"])
def native(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"request_reuse_{request.param}")
    source = directory / "contracts.cpp"
    source.write_text(SOURCE)
    executable = directory / "contracts"
    compiler, flags = compiler_and_flags()
    if request.param == "sanitized":
        flags += ["-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer"]
    run([compiler, *flags, str(source), "-o", str(executable)], cwd=directory)
    if request.param == "production":
        check_frames(directory)
    return executable


def invoke(native, contract, *arguments):
    environment = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0",
        "UBSAN_OPTIONS": "halt_on_error=1"}
    output = run([str(native), contract, *map(str, arguments)], env=environment)
    assert output.strip() == "contract passed; request allocations=0"


@pytest.mark.parametrize("width", [1, 3, 17])
def test_all_seven_masks_with_disjoint_outputs_and_null_unrequested(native, width):
    invoke(native, "masks", width)


def test_bit_exact_keys_signed_zero_and_coordinate_identity(native):
    invoke(native, "keys")


def test_no_cross_mask_reuse_and_exact_bank_counters(native):
    invoke(native, "isolation")


def test_failed_partial_and_invalid_requests_are_not_cached(native):
    invoke(native, "rejected")


def test_nonfinite_requested_data_and_unrequested_poison(native):
    invoke(native, "nonfinite")


def test_copied_keys_outputs_independent_bindings_and_rebind_lifetime(native):
    invoke(native, "lifetime")


def test_lru_eviction_and_first_order_curvature_separation(native):
    invoke(native, "eviction")


def test_unbound_and_failed_rebind_discard_entries_and_reset_counters(native):
    invoke(native, "unbound")


@pytest.mark.parametrize("operation", ["get", "put"])
def test_packed_overloads_do_not_offset_null_in_any_binding_state(native, operation):
    invoke(native, "packed_null", operation)


def test_checked_planner_exact_short_and_misaligned_arenas(native):
    invoke(native, "planner")


def test_packed_overloads_preserve_sparse_outputs(native):
    invoke(native, "packed")


def test_finite_extremes_and_output_signed_zero_are_preserved_bit_exactly(native):
    invoke(native, "extremes")


def test_binary_has_no_allocator_or_dynamic_stack(tmp_path):
    compiler, flags = compiler_and_flags()
    source = tmp_path / "request_reuse.cpp"
    source.write_text(SOURCE)
    obj = tmp_path / "request_reuse.o"
    run([compiler, *flags, "-DVLK_REUSE_AUDIT_ONLY", "-c", str(source),
        "-o", str(obj)], cwd=tmp_path)
    symbols = run(["nm", "-u", str(obj)])
    assert not re.search(
        r"\b_?(malloc|calloc|realloc|free|alloca|aligned_alloc|posix_memalign)\b|_Zn[aw]",
        symbols,
    ), symbols
    check_frames(tmp_path)


def _native_problem(orders, density):
    import numpy as np
    import scivol
    from scivol import _native_core as core, _native_fit as fit

    spec = (scivol.ARMA(*orders[:2]) + scivol.APARCH(*orders[2:])
            + (scivol.Normal() if density == 0 else scivol.StudentT()))
    model = fit._describe_model(spec)
    data = np.random.default_rng(742).normal(size=128)
    theta, _ = fit._initial_theta(spec,model,data,None)
    point = fit._theta_to_coordinates(model,theta)
    lower, upper = fit._coordinate_bounds(model)
    optimizer = fit._aligned_workspace(core._optimizer_workspace_bytes(model.n_params))
    jet = fit._jet_workspace(model,transformed=True)
    return core, fit, model, data, point, lower, upper, optimizer, jet


def _solver_bits(value):
    if isinstance(value, float):
        return struct.pack("!d",value)
    if isinstance(value, tuple):
        return tuple(_solver_bits(item) for item in value)
    return value


def _native_request(problem, data, cache=None, buckets=0):
    import ctypes
    import numpy as np

    core, fit, model, _, point, lower, upper, optimizer, jet = problem
    current = point.copy()
    before = [array.tobytes() for array in (data,lower,upper)]
    arguments = (current.ctypes.data,data.ctypes.data,data.size,fit._initial_variance(data),
        model.p_ar,model.q_ma,model.p_vol,model.q_vol,model.density_code,
        lower.ctypes.data,upper.ctypes.data,optimizer.address,optimizer.size,
        jet.address,jet.size,1e-8,8)
    if cache is None:
        result = core._fit_arma_aparch(*arguments)
        counts = None
    else:
        memory, size = cache
        guard = np.ctypeslib.as_array((ctypes.c_uint8*(size+128)).from_address(memory.address))
        guard[:64] = guard[-64:] = 0xA5
        result, queries, hits = core._fit_arma_aparch_reuse(*arguments,
            memory.address+64 if buckets else 0,size if buckets else 0,buckets)
        assert np.all(guard[:64] == 0xA5) and np.all(guard[-64:] == 0xA5)
        assert 0 <= hits <= queries
        counts = queries,hits
    assert before == [array.tobytes() for array in (data,lower,upper)]
    return (_solver_bits(result),current.tobytes()),counts


@pytest.mark.parametrize("orders,density", [((1,1,1,1),0),((2,1,3,2),1),((3,2,2,3),1)])
def test_native_solver_cache_preserves_results_and_resets_between_samples(orders, density):
    problem = _native_problem(orders,density)
    core, fit, model, data = problem[:4]
    size = core._request_reuse_workspace_bytes(model.n_params,8)
    cache = fit._aligned_workspace(size+128),size
    for sample in (data,1.17*data+.03,data):
        ordinary, _ = _native_request(problem,sample)
        disabled, counts = _native_request(problem,sample,cache,0)
        assert disabled == ordinary and counts == (0,0)
        cached, counts = _native_request(problem,sample,cache,8)
        assert cached == ordinary and counts[0] > 0
