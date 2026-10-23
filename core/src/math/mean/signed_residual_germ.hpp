#pragma once
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::mean {
// Bounded exact fixed-term proof, following positive_simplex.hpp's binary64
// decoding precedent. No rounded FP operations, expression tree or heap storage.
// Capacity failure loses the proof, never changes the statistical domain.
struct FixedResidualDyadic {
    std::uint64_t magnitude=0;
    int exponent=0;
    bool negative=false, valid=true;
    void normalize() noexcept {
        if (!magnitude) { exponent=0; negative=false; return; }
        const auto shift=std::countr_zero(magnitude);
        magnitude>>=shift;
        exponent+=shift;
    }
    static FixedResidualDyadic decode(double value) noexcept {
        static_assert(sizeof(double)==sizeof(std::uint64_t)
            && std::numeric_limits<double>::is_iec559
            && std::numeric_limits<double>::digits==53
            && std::numeric_limits<double>::radix==2
            && std::numeric_limits<double>::min_exponent==-1021
            && std::numeric_limits<double>::max_exponent==1024
            && std::bit_cast<std::uint64_t>(1.0)==UINT64_C(0x3ff0000000000000));
        const auto bits=std::bit_cast<std::uint64_t>(value);
        const auto field=unsigned((bits>>52)&2047);
        FixedResidualDyadic result;
        result.valid=field!=2047;
        result.magnitude=(bits&UINT64_C(0x000fffffffffffff))
            | (field && result.valid ? UINT64_C(1)<<52 : 0);
        result.exponent=field ? int(field)-1075 : -1074;
        result.negative=(bits>>63)!=0;
        result.normalize();
        return result;
    }
    static FixedResidualDyadic product(double a,double b) noexcept {
        auto result=decode(a);
        const auto right=decode(b);
        result.valid=result.valid && right.valid;
        if (!result.valid) return result;
        if (right.magnitude && result.magnitude>UINT64_MAX/right.magnitude) {
            result.valid=false; return result;
        }
        result.magnitude*=right.magnitude;
        result.exponent+=right.exponent;
        result.negative=result.negative!=right.negative;
        result.normalize();
        return result;
    }
    void subtract(FixedResidualDyadic right) noexcept {
        if (!valid || !right.valid) { valid=false; return; }
        if (!right.magnitude) return;
        right.negative=!right.negative;
        if (!magnitude) { *this=right; return; }
        const auto common=std::min(exponent,right.exponent);
        const auto left_shift=unsigned(exponent-common), right_shift=unsigned(right.exponent-common);
        if (left_shift>=64 || right_shift>=64
            || magnitude>(UINT64_MAX>>left_shift)
            || right.magnitude>(UINT64_MAX>>right_shift)) { valid=false; return; }
        const auto a=magnitude<<left_shift, b=right.magnitude<<right_shift;
        if (negative==right.negative) {
            if (a>UINT64_MAX-b) { valid=false; return; }
            magnitude=a+b;
        } else {
            magnitude=a>=b ? a-b : b-a;
            if (b>a) negative=right.negative;
        }
        exponent=common;
        normalize();
    }
    int sign() const noexcept { return !valid ? 2 : !magnitude ? 0 : negative ? -1 : 1; }
    bool equals(double value) const noexcept {
        const auto other=decode(value);
        return valid && other.valid && magnitude==other.magnitude
            && exponent==other.exponent && negative==other.negative;
    }
};

// Certificate: sign * product(odd free zero-coordinate factors) * a nonnegative
// analytic function. Fixed exact terms collect separately until finish(). Other
// addition retains only equal-parity, equal-sign proofs.
// Unknown is not a zero derivative, and no pointwise derivative certifies sign.
struct SignedResidualGerm {
    static constexpr int unknown=2;
    static constexpr auto infinite_order=std::numeric_limits<std::size_t>::max();
    struct CenterEvidence {
        int fixed_sign=0;
        std::size_t order_lower=0,odd_factors=0;
        bool known_leading_sign=false;
    };
    int sign=unknown;
    std::size_t order_lower=0;
    unsigned char* odd=nullptr;
    std::size_t width=0;
    FixedResidualDyadic fixed;
    int fixed_sign=0;
    bool exact_constant=false;
    static int scalar_sign(double x) noexcept { return x>0 ? 1 : x<0 ? -1 : 0; }
    void reset(double constant) noexcept {
        sign=0;
        order_lower=infinite_order;
        fixed=FixedResidualDyadic::decode(constant);
        fixed_sign=fixed.sign();
        exact_constant=false;
        if (odd) std::fill_n(odd,width,0);
    }
    template<class Basis>
    void subtract_product(double coefficient,std::size_t parameter,double lagged,
        int lagged_sign,const unsigned char* lagged_odd,Basis basis,
        std::size_t lagged_order=0,bool lagged_exact_constant=false) noexcept {
        if (!odd) { sign=unknown; return; }
        const bool dependent=basis.depends(parameter);
        if (!dependent && lagged_exact_constant) {
            const auto term=FixedResidualDyadic::product(coefficient,lagged);
            const int term_sign=-scalar_sign(coefficient)*scalar_sign(lagged);
            const int fallback=term_sign==0 ? fixed_sign : fixed_sign==0 ? term_sign
                : fixed_sign==term_sign ? fixed_sign : unknown;
            fixed.subtract(term);
            fixed_sign=fixed.valid ? fixed.sign() : fallback;
            return;
        }
        const auto term_order=lagged_sign==0 ? infinite_order : coefficient==0
            ? !dependent ? infinite_order : lagged_order==infinite_order ? infinite_order : lagged_order+1
            : lagged_order;
        order_lower=std::min(order_lower,term_order);
        const auto factor=dependent && coefficient==0 ? basis.zero_factor_axis(parameter) : std::size_t(-1);
        const int cs=coefficient!=0 ? scalar_sign(coefficient)
            : !dependent ? 0 : factor!=std::size_t(-1) ? 1 : unknown;
        const int product=(cs==0 || lagged_sign==0) ? 0
            : cs==unknown || lagged_sign==unknown ? unknown : -cs*lagged_sign;
        if (product==0) return;
        if (product==unknown || sign==unknown) { sign=unknown; return; }
        if (sign!=0 && sign!=product) { sign=unknown; return; }
        for (std::size_t a=0;a<width;++a) {
            const auto bit=static_cast<unsigned char>((lagged_odd ? lagged_odd[a] : 0) ^ (a==factor));
            if (sign!=0 && odd[a]!=bit) { sign=unknown; return; }
        }
        if (sign==0) for (std::size_t a=0;a<width;++a)
            odd[a]=static_cast<unsigned char>((lagged_odd ? lagged_odd[a] : 0) ^ (a==factor));
        sign=product;
        (void)lagged;
    }
    // Call once after the shared traversal, before retaining the public germ.
    void finish(double value) noexcept {
        exact_constant=sign==0 && fixed.equals(value);
        if (fixed_sign==0) return;
        order_lower=0;
        if (!odd || fixed_sign==unknown || sign==unknown) { sign=unknown; return; }
        if (sign==0) {
            std::fill_n(odd,width,0);
            sign=fixed_sign;
            return;
        }
        if (sign!=fixed_sign) { sign=unknown; return; }
        for (std::size_t a=0;a<width;++a) if (odd[a]) { sign=unknown; return; }
    }
    int fixed_sign_at_zero(double value) const noexcept {
        if (value!=0 || !odd || !fixed.valid || fixed.magnitude || (sign!=1 && sign!=-1)) return 0;
        for (std::size_t a=0;a<width;++a) if (odd[a]) return 0;
        return sign;
    }
    CenterEvidence center_evidence(double value) const noexcept {
        if (value!=0 || !odd || !fixed.valid || fixed.magnitude) return {};
        std::size_t count=0;
        for (std::size_t a=0;a<width;++a) count+=odd[a]!=0;
        return {fixed_sign_at_zero(value),order_lower,count,sign==1 || sign==-1};
    }
};
} // namespace vlk::experiment::mean
