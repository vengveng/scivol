#include "../../core/src/selected_unit_consumer.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace scivolapi;
using namespace scivolapi::detail;
namespace u=unit_map;
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"selected tier retry %d\n",__LINE__);std::abort();}}while(false)

// An exact tier-control witness, not qualification of a model recurrence.
void retry(bool report_only) {
    constexpr auto layout=u::make_layout({1,u::no_axis,{},u::ScaleCoordinate::variance,0,u::no_axis,{}});
    u::Plan<> plan;CHECK(u::make_plan(u::RuntimeLayout{layout},{1,{},1},plan)==Status::ok);
    const std::array<double,1> theta{1};const std::array<std::size_t,1> indices{0};
    const auto axes=ActiveSensitivityBasis::select(1,indices);CHECK(axes);
    const auto r=u::requirements(plan,JetRequest::G);CHECK(r.status==Status::ok);
    std::vector<std::max_align_t> cache((r.bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
    u::PreparedPoint<1> prepared;
    CHECK(u::prepare(plan,theta,{reinterpret_cast<std::byte*>(cache.data()),r.bytes},prepared)==Status::ok);
    const auto fixed=u::rebind_point(u::FixedLayout<layout.roles>{},prepared);CHECK(fixed);
    const auto size=selected_units::workspace_bytes(*axes,2,[](auto){return std::size_t(1);});CHECK(size);
    std::vector<std::max_align_t> work((size+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
    unit_layout_roles::DispatchCounts counts;
    const selected_units::Point point(*axes,prepared,theta,nullptr,&counts);
    double output=-7;
    const auto result=selected_units::evaluate(point,*fixed,JetRequest::G,
        {reinterpret_cast<std::byte*>(work.data()),size},{{},{&output,1},{}},
        [&](unsigned,auto basis,auto,auto* g,auto*) {
            using Number=typename decltype(basis)::number_type;
            Number value(1);
            for(unsigned i=0;i<17;++i){
                value=report_only?value*Number(0x1p1000):basis.product(value,Number(0x1p1000));
                if(!vlk::experiment::normal_sensitivity(value))return Status::numerical_range;
            }
            for(unsigned i=0;i<17;++i)value=basis.product(value,Number(0x1p-1000));
            basis.record_operations(34);
            g[0]=value;basis.gradient_magnitude()[0]=vlk::experiment::sensitivity_magnitude(value);
            return Status::ok;
        });
    CHECK(result.status==Status::ok && result.valid_mask==2 && output==1);
    CHECK(counts.fixed_selected_tiers[0]==1 && counts.fixed_selected_tiers[1]==1 && counts.fixed_selected_tiers[2]==1);
    CHECK(counts.runtime_selected_tiers==std::array<std::size_t,3>{});
}
int main(){retry(false);retry(true);std::puts("exact selected range-retry control passed");}
