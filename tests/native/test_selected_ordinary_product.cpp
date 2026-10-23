#include "../../core/src/selected_unit_consumer.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace scivolapi;
using namespace scivolapi::detail;
namespace u=unit_map;
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"ordinary product %d\n",__LINE__);std::abort();}}while(false)
struct LifetimeProbe {
    static inline unsigned constructions=0;
    LifetimeProbe() noexcept {++constructions;}
};
void scratch_lifetime() {
    std::array<std::max_align_t,4> memory;
    for(unsigned replay=0;replay<3;++replay){
        vlk::experiment::runtime::WorkspaceArena arena(memory.data(),sizeof(memory));
        CHECK(vlk::experiment::take_sensitivities<LifetimeProbe>(arena,3));
        CHECK(LifetimeProbe::constructions==3*(replay+1));
        CHECK(!vlk::experiment::take_sensitivities<LifetimeProbe>(arena,sizeof(memory)+1));
        CHECK(LifetimeProbe::constructions==3*(replay+1));
    }
}
template<class Number,class Prepared>void products(const Prepared& point,ActiveSensitivityBasis axes) {
    const auto actions=u::map_actions<Number,false>(point);CHECK(actions);
    selected_units::Accounting accounting;
    using Actions=std::remove_cvref_t<decltype(*actions)>;
    const selected_units::Basis<Number,Actions> basis(axes,*actions,accounting,nullptr,nullptr);
    auto operations=accounting.operations;
    for(Number c:{Number(-3),Number(0),Number(2)})for(bool reverse:{false,true}) {
        const auto result=reverse?basis.product(Number(0),c):basis.product(c,Number(0));
        operations+=16;
        CHECK(result==0 && accounting.valid && accounting.operations==operations && !accounting.uses_preparation);
    }
    for(Number bad:{Number(std::numeric_limits<double>::infinity()),Number(std::numeric_limits<double>::quiet_NaN())})
        for(bool reverse:{false,true}) {
            selected_units::Accounting rejected;
            const selected_units::Basis<Number,Actions> invalid(axes,*actions,rejected,nullptr,nullptr);
            const auto result=reverse?invalid.product(Number(0),bad):invalid.product(bad,Number(0));
            CHECK(!vlk::experiment::finite_sensitivity(result) && !rejected.valid && rejected.operations==48);
        }
    u::implementation::Evaluation<Number> strict;
    (void)u::implementation::contraction_product(strict,Number(2),Number(0),Number(1));
    CHECK(!strict.valid); // The map-factor helper still rejects an active zero factor.
}
int main() {
    scratch_lifetime();
    constexpr auto layout=u::make_layout({3,u::no_axis,{},u::ScaleCoordinate::variance,0,u::no_axis,{}});
    u::Plan<> plan;CHECK(u::make_plan(u::RuntimeLayout{layout},{3,{},1},plan)==Status::ok);
    const std::array<double,3> theta{1,.125,.25};const std::array<std::size_t,1> selected{0};
    const auto axes=ActiveSensitivityBasis::select(3,selected);CHECK(axes);
    const auto size=u::requirements(plan,JetRequest::G);CHECK(size.status==Status::ok);
    std::vector<std::max_align_t> memory((size.bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
    u::PreparedPoint<1> point;CHECK(u::prepare(plan,theta,{reinterpret_cast<std::byte*>(memory.data()),size.bytes},point)==Status::ok);
    products<long double>(point,*axes);products<u::implementation::Quad>(point,*axes);products<u::implementation::Octuple>(point,*axes);
    const auto fixed=u::rebind_point(u::FixedLayout<layout.roles>{},point);CHECK(fixed);
    products<long double>(*fixed,*axes);products<u::implementation::Quad>(*fixed,*axes);products<u::implementation::Octuple>(*fixed,*axes);
    std::puts("all-tier ordinary zero products passed");
}
