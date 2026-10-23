// Exercise static LSQ and the actual reverse-communication SQP without a test API.
#define BLAS_SYMBOL_PREFIX vlk_
#define NO_APPEND_FORTRAN
#include "active_face_refinement.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

static bool counting = false;
static std::size_t allocations = 0;

void* operator new(std::size_t bytes)
{
    if (counting) ++allocations;
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    std::abort();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

static void* counted_malloc(std::size_t bytes)
{
    if (counting) ++allocations;
    return std::malloc(bytes);
}
static void* counted_calloc(std::size_t n, std::size_t bytes)
{
    if (counting) ++allocations;
    return std::calloc(n, bytes);
}
static void* counted_realloc(void* p, std::size_t bytes)
{
    if (counting) ++allocations;
    return std::realloc(p, bytes);
}

// Headers are loaded before these macros; count C allocations in the shared QP too.
#define malloc counted_malloc
#define calloc counted_calloc
#define realloc counted_realloc
#include "slsqp_translation.cpp"
#undef malloc
#undef calloc
#undef realloc

constexpr double sentinel = 917.25;

template<class T> struct Guarded {
    std::vector<T> storage;
    explicit Guarded(std::size_t n) : storage(n + 2, static_cast<T>(sentinel)) {}
    T* data() { return storage.data() + 1; }
    bool intact() const {
        return storage.front() == static_cast<T>(sentinel)
            && storage.back() == static_cast<T>(sentinel);
    }
    bool untouched() const {
        return std::all_of(storage.begin(), storage.end(), [](T x) {
            return x == static_cast<T>(sentinel);
        });
    }
};

static void number(double x)
{
    if (std::isfinite(x)) std::printf("%.17g", x);
    else std::printf("null");
}

static void array(const char* name, const double* x, std::size_t n)
{
    std::printf(",\"%s\":[", name);
    for (std::size_t i = 0; i < n; ++i) {
        if (i) std::printf(",");
        number(x[i]);
    }
    std::printf("]");
}

int main(int argc, char** argv)
{
    if (argc < 2) return 2;
    const char* name = argv[1];
    const bool first = std::strcmp(name, "first") == 0;
    const bool full = std::strcmp(name, "full") == 0;
    const bool body = first || full;
    const bool augmented = std::strncmp(name, "augmented", 9) == 0;
    const bool equality = std::strcmp(name, "equality") == 0
        || std::strcmp(name, "augmented_equality") == 0;
    const bool inequality = std::strcmp(name, "inequality") == 0
        || std::strcmp(name, "augmented_inequality") == 0;
    const bool dense = std::strncmp(name, "dense", 5) == 0;
    const bool factor_bad = std::strncmp(name, "factor_", 7) == 0;
    const bool mixed = std::strcmp(name, "mixed") == 0;
    const int n = first ? (argc > 2 ? std::atoi(argv[2]) : 1)
        : full || augmented ? 1 : dense || equality || inequality || mixed ? 2 : 8;
    if (n != 1 && n != 2 && n != 8) return 2;
    const int m = mixed ? 2 : equality || inequality ? 1 : 0;
    const int meq = equality || mixed ? 1 : 0;
    const std::size_t packed = n * (n + 1) / 2;
    std::size_t scratch_count = 0;
    if (!vlk::experiment::runtime::runtime_kraft_buffer_count(n, m, scratch_count, meq))
        return 2;
    const auto nnls_bytes = vlk::experiment::nnls::sqp_workspace_bytes(n, m);
    Guarded<double> factor(packed), gradient(n), normals(std::max(1, m) * n);
    Guarded<double> gaps(std::max(1, m)), lower(n + 1), upper(n + 1), x(n + 1);
    Guarded<double> multipliers(m + 2 * (n + 1) + 2), scratch(scratch_count);
    Guarded<double> nnls((nnls_bytes + sizeof(double) - 1) / sizeof(double));
    Guarded<CBLAS_INT> indices(m + 2 * (n + 1));
    std::vector<double> diagonal(n, 1.0);
    std::fill_n(factor.data(), packed, 0.0);
    std::fill_n(normals.data(), std::max(1, m) * n, 0.0);
    std::fill_n(gaps.data(), std::max(1, m), 0.0);
    for (int i = 0; i < n; ++i) {
        gradient.data()[i] = (i + 1) / 10.0;
        lower.data()[i] = -1.0;
        upper.data()[i] = 1.0;
        x.data()[i] = 0.0;
    }
    if (body) gradient.data()[0] = full ? 1e6 : (argc > 3 ? std::strtod(argv[3], nullptr) : 1.0);
    if (!body && !augmented && !dense && !equality && !inequality && !mixed && !factor_bad) {
        const double d[] = {0.25, 2, 4, 0.5, 16, 8, 1, 32};
        const double g[] = {4, -8, 2, -0.125, 0, -16, 1e15, -1e15};
        std::copy_n(d, n, diagonal.data());
        std::copy_n(g, n, gradient.data());
    }
    if (std::strcmp(name, "unbounded") == 0) {
        std::fill_n(lower.data(), n, NAN);
        std::fill_n(upper.data(), n, NAN);
    } else if (std::strcmp(name, "one_sided") == 0) {
        for (int i = 0; i < n; ++i) (i % 2 ? lower : upper).data()[i] = NAN;
    } else if (std::strcmp(name, "signed_bounds") == 0) {
        const double lo[] = {2, -5, 0, -0.0, 0, -3, 2, -5};
        const double hi[] = {5, -2, 0, 0.0, 0, -1, 2, -5};
        std::copy_n(lo, n, lower.data());
        std::copy_n(hi, n, upper.data());
    } else if (std::strncmp(name, "overflow", 8) == 0) {
        for (int i = 0; i < n; ++i) {
            diagonal[i] = i % 2 ? std::numeric_limits<double>::min()
                : std::numeric_limits<double>::denorm_min();
            gradient.data()[i] = (i % 2 ? -1.0 : 1.0) * std::numeric_limits<double>::max();
        }
        if (std::strcmp(name, "overflow_finite") == 0) {
            const double lo[] = {-1, NAN, 1, -4, 0, -0.0, 2, -5};
            const double hi[] = {NAN, 1, 5, -1, 0, 0, 2, -5};
            std::copy_n(lo, n, lower.data());
            std::copy_n(hi, n, upper.data());
        } else {
            // Fail on the last coordinate, after earlier bounded coordinates succeed.
            lower.data()[n - 1] = std::strcmp(name, "overflow_wrong_side") == 0 ? -1 : NAN;
            upper.data()[n - 1] = NAN;
            if (std::strcmp(name, "overflow_negative") == 0) {
                gradient.data()[n - 1] = std::numeric_limits<double>::max();
                lower.data()[n - 1] = NAN;
                upper.data()[n - 1] = 1;
            }
        }
    } else if (std::strcmp(name, "inconsistent") == 0) {
        lower.data()[n - 1] = 2;
        upper.data()[n - 1] = -2;
    } else if (std::strncmp(name, "gradient_", 9) == 0) {
        gradient.data()[n - 1] = std::strcmp(name, "gradient_nan") == 0 ? NAN
            : std::strcmp(name, "gradient_inf") == 0 ? INFINITY : -INFINITY;
    } else if (std::strncmp(name, "bound_", 6) == 0) {
        const bool low = std::strstr(name, "lower") != nullptr;
        (low ? lower : upper).data()[n - 1] = std::strstr(name, "negative") ? -INFINITY : INFINITY;
    }
    if (dense) {
        diagonal = {2, 3};
        gradient.data()[0] = -2;
        gradient.data()[1] = -4;
        lower.data()[0] = lower.data()[1] = -2;
        upper.data()[0] = std::strcmp(name, "dense_box") == 0 ? 0.25 : 2;
        upper.data()[1] = 2;
    }
    if (m && !augmented) {
        diagonal = {2, 4};
        gradient.data()[0] = gradient.data()[1] = 0;
        normals.data()[0] = normals.data()[m] = 1;
        gaps.data()[0] = inequality ? -1 : -0.5;
        if (mixed) {
            normals.data()[1] = 1;
            gaps.data()[1] = -0.4;
        }
    }
    if (augmented) {
        diagonal[0] = 4;
        gradient.data()[0] = 1;
        if (m) {
            normals.data()[0] = 1;
            gaps.data()[0] = -2;
        }
    }
    for (int i = 0, cursor = 0; i < n; cursor += n - i, ++i)
        factor.data()[cursor] = diagonal[i];
    if (dense) factor.data()[1] = 0.5;
    if (std::strcmp(name, "tiny_offdiagonal") == 0)
        factor.data()[packed - 2] = std::numeric_limits<double>::denorm_min();
    if (std::strcmp(name, "nan_offdiagonal") == 0) factor.data()[packed - 2] = NAN;
    if (std::strcmp(name, "signed_zero_factor") == 0)
        for (int j = 0, cursor = 0; j < n; ++j) {
            ++cursor;
            for (int i = j + 1; i < n; ++i) factor.data()[cursor++] = -0.0;
        }
    if (factor_bad) factor.data()[packed - 1] = std::strcmp(name, "factor_zero") == 0 ? 0
        : std::strcmp(name, "factor_negative") == 0 ? -1
        : std::strcmp(name, "factor_nan") == 0 ? NAN : INFINITY;
    const auto saved_factor = factor.storage, saved_gradient = gradient.storage;
    const auto saved_normals = normals.storage, saved_gaps = gaps.storage;
    const auto saved_lower = lower.storage, saved_upper = upper.storage;

    SLSQP_vars state {};
    state.acc = 1e-12;
    state.itermax = 100;
    state.n = n;
    double value = 0;
    std::int64_t mode = 0;
    std::size_t f_trials = 0, g_trials = 0, body_calls = 0;
    double minimum_trial = INFINITY;
    const ScivolNnlsWorkspace workspace {nnls.data(), nnls_bytes};
    counting = true;
    if (body) {
        do {
            __slsqp_body(&state, &value, gradient.data(), normals.data(), gaps.data(),
                x.data(), multipliers.data(), lower.data(), upper.data(), scratch.data(),
                indices.data(), workspace);
            ++body_calls;
            if (!full) break;
            if (state.mode == 1) {
                ++f_trials;
                minimum_trial = std::min(minimum_trial, x.data()[0]);
                value = 0.5 * x.data()[0] * x.data()[0] + 1e6 * x.data()[0];
            } else if (state.mode == -1) {
                ++g_trials;
                gradient.data()[0] = x.data()[0] + 1e6;
            } else break;
        } while (body_calls < 500);
        mode = state.mode;
    } else {
        lsq(m, meq, n, augmented, 100, factor.data(), gradient.data(), normals.data(),
            gaps.data(), lower.data(), upper.data(), x.data(), multipliers.data(),
            scratch.data(), indices.data(), &mode, workspace);
    }
    counting = false;
    const auto unchanged = [](const auto& current, const auto& before) {
        return std::memcmp(current.data(), before.data(), before.size() * sizeof(double)) == 0;
    };
    const bool inputs_intact = unchanged(factor.storage, saved_factor)
        && (full || unchanged(gradient.storage, saved_gradient))
        && unchanged(normals.storage, saved_normals) && unchanged(gaps.storage, saved_gaps)
        && std::memcmp(lower.data(), saved_lower.data() + 1, n * sizeof(double)) == 0
        && std::memcmp(upper.data(), saved_upper.data() + 1, n * sizeof(double)) == 0;
    const bool guards = factor.intact() && gradient.intact() && normals.intact()
        && gaps.intact() && lower.intact() && upper.intact() && x.intact()
        && multipliers.intact() && scratch.intact() && nnls.intact() && indices.intact()
        && (augmented || (x.data()[n] == sentinel && lower.data()[n] == sentinel
            && upper.data()[n] == sentinel));
    bool nan_multipliers = true;
    for (int i = 0; i < 2 * (n + augmented); ++i)
        nan_multipliers &= std::isnan(multipliers.data()[m + i]);
    const bool tails_intact = multipliers.data()[m + 2 * (n + augmented)] == sentinel
        && multipliers.data()[m + 2 * (n + augmented) + 1] == sentinel;
    std::printf("{\"mode\":%lld,\"allocations\":%zu,\"guards_intact\":%s,"
        "\"inputs_intact\":%s,\"scratch_touched\":%s,\"nnls_touched\":%s,"
        "\"bound_multipliers_nan\":%s,\"multiplier_tails_intact\":%s,"
        "\"iterations\":%lld,\"resets\":%lld,\"f_trials\":%zu,\"g_trials\":%zu,"
        "\"body_calls\":%zu,\"minimum_trial\":",
        static_cast<long long>(mode), allocations, guards ? "true" : "false",
        inputs_intact ? "true" : "false", scratch.untouched() ? "false" : "true",
        nnls.untouched() ? "false" : "true", nan_multipliers ? "true" : "false",
        tails_intact ? "true" : "false", static_cast<long long>(state.iter),
        static_cast<long long>(state.reset), f_trials, g_trials, body_calls);
    number(minimum_trial);
    array("x", x.data(), n + augmented);
    array("diagonal", diagonal.data(), n);
    array("factor", factor.data(), packed);
    array("gradient", gradient.data(), n);
    array("lower", lower.data(), n);
    array("upper", upper.data(), n);
    array("multipliers", multipliers.data(), m);
    std::printf(",\"value\":");
    number(value);
    std::printf("}\n");
    return 0;
}
