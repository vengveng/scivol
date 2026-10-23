#include "causal_mean.hpp"

#include "../math/runtime_workspace.hpp"
#include "../models/jet_contract.hpp"
#include <array>
#include <bit>

namespace scivolapi::detail::causal_mean {
namespace {
using vlk::experiment::runtime::checked_product;
using vlk::experiment::runtime::checked_sum;

// Exact binary64 rolling sum, including removal. 2240 bits cover the 2098-bit
// binary64 exponent range plus any size_t observation count, without overflow.
constexpr std::size_t limbs = 70;
constexpr std::int64_t radix = std::int64_t{1} << 32;
constexpr std::size_t scratch_bytes = limbs * (sizeof(std::int64_t) + 2 * sizeof(std::uint32_t));
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));

class RollingSum {
public:
    explicit RollingSum(std::span<std::byte> memory) noexcept {
        vlk::experiment::runtime::WorkspaceArena arena(memory.data(), memory.size());
        digits_ = arena.take<std::int64_t>(limbs);
        magnitude_ = arena.take<std::uint32_t>(limbs);
        quotient_ = arena.take<std::uint32_t>(limbs);
        std::fill_n(digits_, limbs, 0);
    }

    void add(double value, bool remove = false) noexcept {
        const auto bits = std::bit_cast<std::uint64_t>(value);
        const auto exponent = (bits >> 52) & 0x7ff;
        const auto mantissa = (bits & ((std::uint64_t{1} << 52) - 1))
            | (exponent ? std::uint64_t{1} << 52 : 0);
        if (!mantissa) return;
        const auto shift = exponent ? exponent - 1 : 0;
        const auto word = static_cast<std::size_t>(shift / 32);
        const auto offset = static_cast<unsigned>(shift % 32);
        const auto low = mantissa << offset;
        const auto high = offset ? mantissa >> (64 - offset) : 0;
        const std::int64_t sign = (bool(bits >> 63) != remove) ? -1 : 1;
        add_digit(word, sign * static_cast<std::uint32_t>(low));
        add_digit(word + 1, sign * static_cast<std::uint32_t>(low >> 32));
        add_digit(word + 2, sign * static_cast<std::uint32_t>(high));
    }

    double mean(std::size_t count) noexcept {
        std::size_t top = limbs;
        while (top && digits_[top - 1] == 0) --top;
        if (!top) return 0;
        const bool negative = digits_[top - 1] < 0;
        const std::int64_t sign = negative ? -1 : 1;
        std::int64_t carry = 0;
        for (std::size_t i = 0; i < top; ++i) {
            const auto value = sign * digits_[i] + carry;
            carry = value / radix;
            auto remainder = value % radix;
            if (remainder < 0) { remainder += radix; --carry; }
            magnitude_[i] = static_cast<std::uint32_t>(remainder);
        }
        while (top && magnitude_[top - 1] == 0) --top;
        const auto remainder = divide(top, static_cast<std::uint64_t>(count));
        while (top && quotient_[top - 1] == 0) --top;
        const auto highest = top ? (top - 1) * 32
            + (31 - std::countl_zero(quotient_[top - 1])) : 0;
        auto shift = highest > 52 ? highest - 52 : 0;
        std::uint64_t mantissa = 0;
        for (unsigned bit = 0; bit < 53; ++bit)
            if (set(shift + bit, top)) mantissa |= std::uint64_t{1} << bit;
        bool round_up;
        if (shift) {
            const bool sticky = remainder != 0 || lower_bits(shift - 1, top);
            round_up = set(shift - 1, top) && (sticky || (mantissa & 1));
        } else {
            const auto complement = static_cast<std::uint64_t>(count) - remainder;
            round_up = remainder > complement || (remainder == complement && (mantissa & 1));
        }
        mantissa += round_up;
        if (mantissa == (std::uint64_t{1} << 53)) { mantissa >>= 1; ++shift; }
        const auto representation = mantissa < (std::uint64_t{1} << 52) ? mantissa
            : ((static_cast<std::uint64_t>(shift) + 1) << 52)
                | (mantissa - (std::uint64_t{1} << 52));
        return std::bit_cast<double>(representation | (std::uint64_t(negative) << 63));
    }

private:
    void add_digit(std::size_t word, std::int64_t value) noexcept {
        while (value) {
            const auto total = digits_[word] + value;
            digits_[word++] = total % radix;
            value = total / radix;
        }
    }

    std::uint64_t divide(std::size_t top, std::uint64_t count) noexcept {
        std::uint64_t remainder = 0;
        for (std::size_t i = top; i-- > 0;) {
            std::uint32_t word = 0;
            if (count <= std::numeric_limits<std::uint32_t>::max()) {
                const auto value = (remainder << 32) | magnitude_[i];
                word = static_cast<std::uint32_t>(value / count);
                remainder = value % count;
            } else {
                // Portable long division, including counts wider than 32 bits.
                for (unsigned bit = 32; bit-- > 0;) {
                    const bool overflow = remainder >> 63;
                    remainder = (remainder << 1) | ((magnitude_[i] >> bit) & 1);
                    if (overflow || remainder >= count) {
                        remainder -= count;
                        word |= std::uint32_t{1} << bit;
                    }
                }
            }
            quotient_[i] = word;
        }
        return remainder;
    }

    bool set(std::size_t bit, std::size_t top) const noexcept {
        return bit / 32 < top && ((quotient_[bit / 32] >> (bit % 32)) & 1);
    }
    bool lower_bits(std::size_t stop, std::size_t top) const noexcept {
        const auto words = std::min(stop / 32, top);
        for (std::size_t i = 0; i < words; ++i) if (quotient_[i]) return true;
        const auto remainder = static_cast<unsigned>(stop % 32);
        return words < top && remainder && (quotient_[words] & ((std::uint32_t{1} << remainder) - 1));
    }
    std::int64_t* digits_;
    std::uint32_t* magnitude_;
    std::uint32_t* quotient_;
};

bool same_design(const DesignPlan& a, const DesignPlan& b) noexcept {
    return a.generated_feature_width == b.generated_feature_width && a.design_width == b.design_width
        && a.coefficient_width == b.coefficient_width && a.required_hold_back == b.required_hold_back
        && a.polynomial_order == b.polynomial_order && a.polynomial_width == b.polynomial_width
        && a.polynomial_offset == b.polynomial_offset;
}
bool same_sample(const SamplePlan& a, const SamplePlan& b) noexcept {
    return a.original_n == b.original_n && a.requested_hold_back == b.requested_hold_back
        && a.retained_begin == b.retained_begin && a.retained_n == b.retained_n
        && a.recursion_begin == b.recursion_begin && a.first_eligible_original == b.first_eligible_original
        && a.first_original_score == b.first_original_score && a.stop_original_score == b.stop_original_score
        && a.first_scored == b.first_scored && a.scored_stop == b.scored_stop && a.n_eff == b.n_eff;
}
bool valid_design(const DesignPlan& plan) noexcept {
    DesignPlan expected;
    return plan_design(plan.spec, expected) == Status::ok && same_design(plan, expected);
}
bool valid_sample(const DesignPlan& design, const SamplePlan& sample) noexcept {
    SamplePlan expected;
    return plan_sample(design, sample.original_n,
        {sample.requested_hold_back, sample.first_original_score, sample.stop_original_score}, expected)
        == Status::ok && same_sample(sample, expected);
}

struct Range { std::uintptr_t begin = 0, end = 0; };
template<class T> bool range(std::span<T> values, Range& result) noexcept {
    if (values.empty()) { result = {}; return true; }
    const auto begin = reinterpret_cast<std::uintptr_t>(values.data());
    std::size_t bytes;
    if (!begin || begin % alignof(T) || !checked_product(values.size(), sizeof(T), bytes)
        || bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
    result = {begin, begin + bytes};
    return true;
}
bool overlaps(Range a, Range b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}

Status validate_storage(const Requirements& needs, const DesignPlan& plan, const SamplePlan& sample,
                        InputView input, Buffers buffers, PreparedView& output) noexcept {
    if (buffers.observations.size() < needs.observations || buffers.design.size() < needs.design_elements
        || buffers.columns.size() < needs.columns || buffers.polynomial_basis.size() < needs.basis_elements
        || buffers.workspace.size() < needs.workspace_bytes) return Status::insufficient_capacity;
    std::array<Range, 8> sources{};
    std::array<Range, 5> targets{};
    if (!range(input.observations, sources[0]) || !range(input.exogenous, sources[1])
        || !range(input.row_ids, sources[2]) || !range(plan.spec.har_horizons, sources[3])
        || !range(std::span{&plan, 1}, sources[4]) || !range(std::span{&sample, 1}, sources[5])
        || !range(input.observation_history, sources[6])
        || !range(std::span{&output, 1}, sources[7])
        || !range(buffers.observations.first(needs.observations), targets[0])
        || !range(buffers.design.first(needs.design_elements), targets[1])
        || !range(buffers.columns.first(needs.columns), targets[2])
        || !range(buffers.polynomial_basis.first(needs.basis_elements), targets[3])
        || !range(buffers.workspace.first(needs.workspace_bytes), targets[4])) return Status::invalid_argument;
    if (needs.workspace_bytes && targets[4].begin % needs.workspace_alignment) return Status::invalid_argument;
    for (std::size_t i = 0; i + 1 < sources.size(); ++i)
        if (overlaps(sources[i], sources.back())) return Status::invalid_argument;
    for (std::size_t i = 0; i < targets.size(); ++i) {
        for (auto source : sources) if (overlaps(targets[i], source)) return Status::invalid_argument;
        for (std::size_t j = 0; j < i; ++j)
            if (overlaps(targets[i], targets[j])) return Status::invalid_argument;
    }
    return Status::ok;
}

Status validate_inputs(const DesignPlan& plan, const SamplePlan& sample, InputView input) noexcept {
    const vlk::experiment::conditioning::History history{input.observation_history, {}, 0};
    if (!history.shape({plan.required_hold_back, 0}) || !history.finite()) return Status::invalid_argument;
    std::size_t exogenous_elements;
    Range admitted;
    if (input.observations.size() != sample.original_n
        || !checked_product(sample.original_n, plan.spec.input_exogenous_width, exogenous_elements)
        || input.exogenous.size() != exogenous_elements
        || (!input.row_ids.empty() && input.row_ids.size() != sample.original_n)
        || !range(input.observations, admitted) || !range(input.exogenous, admitted)
        || !range(input.row_ids, admitted)) return Status::invalid_argument;
    for (double value : input.observations) if (!std::isfinite(value)) return Status::invalid_argument;
    for (double value : input.exogenous) if (!std::isfinite(value)) return Status::invalid_argument;
    for (std::size_t i = 1; i < input.row_ids.size(); ++i)
        if (input.row_ids[i] <= input.row_ids[i - 1]) return Status::invalid_argument;
    return Status::ok;
}

void describe_columns(const DesignPlan& plan, Buffers buffers) noexcept {
    const auto intercept = std::size_t(plan.spec.intercept);
    if (intercept) buffers.columns[0] = {ColumnRole::intercept, 0, no_column, 0, no_column};
    for (std::size_t j = 0; j < plan.generated_feature_width; ++j) {
        const bool ar = plan.spec.kind == Kind::arx;
        buffers.columns[intercept + j] = {ar ? ColumnRole::ar_lag : ColumnRole::har_mean,
            intercept + j, j, ar ? j + 1 : plan.spec.har_horizons[j], no_column};
    }
    for (std::size_t j = 0; j < plan.spec.input_exogenous_width; ++j) {
        const auto column = plan.generated_feature_width + j;
        buffers.columns[intercept + column] = {ColumnRole::exogenous, intercept + column, column, 0, j};
    }
    if (plan.spec.kind == Kind::harx)
        for (std::size_t l = 0; l < plan.polynomial_order; ++l)
            for (std::size_t j = 0; j < plan.polynomial_width; ++j)
                buffers.polynomial_basis[l * plan.polynomial_width + j] = l < plan.spec.har_horizons[j]
                    ? 1.0 / static_cast<double>(plan.spec.har_horizons[j]) : 0.0;
}

void prepare_features(const DesignPlan& plan, const SamplePlan& sample, InputView input,
                      Buffers buffers) noexcept {
    const auto y = input.observations.subspan(sample.retained_begin);
    const vlk::experiment::conditioning::History history{input.observation_history, {}, 0};
    std::copy(y.begin(), y.end(), buffers.observations.begin());
    for (std::size_t t = 0; t < y.size(); ++t) {
        for (std::size_t j = 0; j < plan.generated_feature_width; ++j)
            buffers.design[t * plan.design_width + j] = plan.spec.kind == Kind::arx
                && (t > j || !history.observations.empty()) ? history.observation(y.data(), t, j) : 0;
        for (std::size_t j = 0; j < plan.spec.input_exogenous_width; ++j)
            buffers.design[t * plan.design_width + plan.generated_feature_width + j]
                = input.exogenous[(sample.retained_begin + t) * plan.spec.input_exogenous_width + j];
    }
    if (plan.spec.kind != Kind::harx) return;
    for (std::size_t j = 0; j < plan.generated_feature_width; ++j) {
        const auto horizon = plan.spec.har_horizons[j];
        if (horizon == 1) {
            if (!history.observations.empty()) buffers.design[j] = history.observations[0];
            for (std::size_t t = 1; t < y.size(); ++t)
                buffers.design[t * plan.design_width + j] = y[t - 1];
            continue;
        }
        RollingSum window(buffers.workspace);
        if (!history.observations.empty()) {
            for (std::size_t lag = 0; lag < horizon; ++lag) window.add(history.observations[lag]);
            buffers.design[j] = window.mean(horizon);
        }
        for (std::size_t t = 1; t < y.size(); ++t) {
            if (t > horizon || !history.observations.empty())
                window.add(history.observation(y.data(), t, horizon), true);
            window.add(y[t - 1]);
            buffers.design[t * plan.design_width + j] = window.mean(history.observations.empty()
                ? std::min(t, horizon) : horizon);
        }
    }
}
} // namespace

Status plan_design(DesignSpec spec, DesignPlan& output) noexcept {
    DesignPlan result;
    result.spec = spec;
    result.polynomial_offset = std::size_t(spec.intercept);
    switch (spec.kind) {
    case Kind::linear:
        if (spec.ar_lags || !spec.har_horizons.empty()) return Status::invalid_argument;
        break;
    case Kind::arx:
        if (!spec.har_horizons.empty()) return Status::invalid_argument;
        result.generated_feature_width = result.required_hold_back = spec.ar_lags;
        result.polynomial_order = result.polynomial_width = spec.ar_lags;
        break;
    case Kind::harx: {
        Range admitted;
        if (spec.ar_lags || spec.har_horizons.empty() || !range(spec.har_horizons, admitted))
            return Status::invalid_argument;
        std::size_t previous = 0;
        for (auto h : spec.har_horizons) {
            if (h <= previous) return Status::invalid_argument;
            previous = h;
        }
        result.generated_feature_width = result.polynomial_width = spec.har_horizons.size();
        result.polynomial_order = result.required_hold_back = previous;
        break;
    }
    default: return Status::invalid_argument;
    }
    std::size_t basis, bytes;
    if (!checked_sum(result.generated_feature_width, spec.input_exogenous_width, result.design_width)
        || !checked_sum(result.design_width, std::size_t(spec.intercept), result.coefficient_width)
        || !checked_product(result.coefficient_width, sizeof(Column), bytes)
        || (spec.kind == Kind::harx && (!checked_product(result.polynomial_order, result.polynomial_width, basis)
            || !checked_product(basis, sizeof(double), bytes)))) return Status::invalid_argument;
    output = result;
    return Status::ok;
}

Status plan_sample(const DesignPlan& design, std::size_t n, SampleRequest request, SamplePlan& output) noexcept {
    if (!valid_design(design)) return Status::invalid_argument;
    const auto begin = std::max(request.requested_hold_back, design.required_hold_back);
    if (n < 2 || begin >= n || n - begin < 2) return Status::invalid_argument;
    const auto first = request.first_original_score.value_or(begin);
    const auto stop = request.stop_original_score.value_or(n);
    if (first < begin || first >= stop || stop > n) return Status::invalid_argument;
    output = {n, request.requested_hold_back, begin, n - begin, 0, begin, first, stop,
              first - begin, stop - begin, stop - first};
    return Status::ok;
}

Requirements requirements(const DesignPlan& design, const SamplePlan& sample) noexcept {
    Requirements result;
    if (!valid_design(design) || !valid_sample(design, sample)) return result;
    result.observations = sample.retained_n;
    result.columns = design.coefficient_width;
    std::size_t bytes;
    if (!checked_product(sample.retained_n, design.design_width, result.design_elements)
        || !checked_product(result.design_elements, sizeof(double), bytes)
        || !checked_product(result.observations, sizeof(double), bytes)) return result;
    if (design.spec.kind == Kind::harx) {
        if (!checked_product(design.polynomial_order, design.polynomial_width, result.basis_elements)) return result;
        if (design.spec.har_horizons.back() > 1) {
            result.workspace_bytes = scratch_bytes;
            result.workspace_alignment = alignof(std::int64_t);
        }
    }
    result.status = Status::ok;
    return result;
}

Status compute(const DesignPlan& design, const SamplePlan& sample, InputView input,
               Buffers buffers, PreparedView& output) noexcept {
    const auto needs = requirements(design, sample);
    if (needs.status != Status::ok) return needs.status;
    const auto inputs = validate_inputs(design, sample, input);
    if (inputs != Status::ok) return inputs;
    const auto storage = validate_storage(needs, design, sample, input, buffers, output);
    if (storage != Status::ok) return storage;
    describe_columns(design, buffers);
    prepare_features(design, sample, input, buffers);
    PreparedView result;
    result.sample = sample;
    result.observations = buffers.observations.first(needs.observations);
    result.design = buffers.design.first(needs.design_elements);
    result.design_width = design.design_width;
    result.input_exogenous_width = design.spec.input_exogenous_width;
    result.generated_feature_width = design.generated_feature_width;
    result.columns = buffers.columns.first(needs.columns);
    if (!input.row_ids.empty()) {
        result.retained_row_ids = input.row_ids.subspan(sample.retained_begin);
        result.scored_row_ids = input.row_ids.subspan(sample.first_original_score, sample.n_eff);
    }
    if (design.polynomial_order)
        result.polynomial = PolynomialView{design.polynomial_offset, {design.polynomial_order,
            design.polynomial_width, needs.basis_elements ? buffers.polynomial_basis.data() : nullptr, 1.0}};
    output = result;
    return Status::ok;
}
} // namespace scivolapi::detail::causal_mean
