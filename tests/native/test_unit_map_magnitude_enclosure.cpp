#include "models/unit_map.hpp"
#include <cstdio>
#include <cstdlib>

namespace {
namespace u = scivolapi::detail::unit_map;
void check(bool pass, int line) {
    if (!pass) { std::fprintf(stderr, "magnitude enclosure line %d\n", line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__), __LINE__)

template<class Number> void rounded_down_stream() {
    using std::nextafter;
    const Number epsilon = u::implementation::epsilon<Number>();
    const Number increment = epsilon / 4;
    Number rounded = 1;
    for (unsigned i = 0; i < 128; ++i) rounded += increment;
    // All terms and this exact binary-power reference are representable.
    const Number exact = Number(1) + 32 * epsilon;
    CHECK(rounded == 1);
    CHECK(nextafter(rounded, Number(std::numeric_limits<double>::infinity())) < exact);
    u::implementation::Evaluation<Number> e;
    e.value = 1; e.magnitude = rounded; e.uses_preparation = false;
    CHECK(u::enclose_magnitude(e, 128) == u::ContractionAdmission::ready);
    CHECK(e.magnitude >= exact && e.value == 1 && e.valid && !e.uses_preparation);

    auto no_rounding = e;
    CHECK(u::enclose_magnitude(no_rounding, 0) == u::ContractionAdmission::ready);
    CHECK(no_rounding.magnitude == e.magnitude);
    auto invalid = e; invalid.magnitude = -1;
    CHECK(u::enclose_magnitude(invalid, 128) == u::ContractionAdmission::invalid);
    CHECK(invalid.magnitude == -1);
    invalid = e; invalid.magnitude = Number(std::numeric_limits<double>::infinity());
    CHECK(u::enclose_magnitude(invalid, 128) == u::ContractionAdmission::invalid);
    CHECK(!u::implementation::finite_number(invalid.magnitude));
    invalid = e; invalid.valid = false;
    CHECK(u::enclose_magnitude(invalid, 128) == u::ContractionAdmission::invalid);
    CHECK(invalid.magnitude == e.magnitude);
    u::implementation::Evaluation<Number> zero;
    CHECK(u::enclose_magnitude(zero, 128) == u::ContractionAdmission::ready);
    CHECK(zero.value == 0 && zero.magnitude == 0);
    zero.value = 1;
    CHECK(u::enclose_magnitude(zero, 128) == u::ContractionAdmission::invalid);
}

void ordinary_depth_limit() {
    u::implementation::Evaluation<long double> e;
    e.value = e.magnitude = 1;
    if (static_cast<long double>(std::numeric_limits<std::size_t>::max())
        * std::numeric_limits<long double>::epsilon() >= 1) {
        CHECK(u::enclose_magnitude(e, std::numeric_limits<std::size_t>::max())
            == u::ContractionAdmission::insufficient_precision);
        CHECK(e.magnitude == 1);
    }
}
}

int main() {
    rounded_down_stream<long double>();
    rounded_down_stream<u::implementation::Quad>();
    rounded_down_stream<u::implementation::Octuple>();
    ordinary_depth_limit();
    std::puts("counted magnitude enclosure properties passed");
}
