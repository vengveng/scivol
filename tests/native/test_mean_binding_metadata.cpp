#include "unit_layout_dispatch.hpp"
#if defined(SCIVOL_CONSTANT_MEAN_JET_HPP) || defined(SCIVOL_MEAN_EQUATION_HPP)
#error "Binding metadata must not include the mean numerical kernel."
#endif

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace mean = vlk::experiment::mean;
namespace u = scivolapi::detail::unit_map;
namespace roles = scivolapi::detail::unit_layout_roles;
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"mean binding metadata %d\n",__LINE__);std::abort();}} while(false)
template<std::size_t Shapes> struct Density {static constexpr auto shape_parameters=Shapes;};

template<std::size_t Shapes>
void check_case(bool linear,bool intercept,int dispatch,std::size_t columns,std::size_t q) {
    const auto plan=mean::layout(columns,q);
    const auto base=roles::variance(plan.width+Shapes,plan.width-1,intercept,
        linear ? u::Block{std::size_t(intercept),columns-std::size_t(intercept)} : u::Block{});
    for(std::size_t prefix=0;prefix<=base.roles.exogenous.count;++prefix) {
        const auto admitted=roles::without_generated(base,prefix);
        unsigned visits=0;
        mean::dispatch_selected_binding<Density<Shapes>>(plan,dispatch,linear,intercept,[&](auto binding) {
            using Kernel=typename decltype(binding)::kernel_type;
            CHECK(Kernel::linear==linear && Kernel::intercept==intercept);
            CHECK(Kernel::fixed==(dispatch!=0));
            if constexpr(Kernel::fixed) {
                constexpr auto metadata=Kernel::model_layout(mean::Layout{});
                static_assert(metadata.valid && metadata.width==Kernel::width && metadata.ma_order==Kernel::q);
                CHECK(metadata.columns==columns && metadata.ma_order==q);
                if constexpr(Kernel::linear) {
                    static_assert(std::is_same_v<typename Kernel::Design,mean::ExternalLinearDesign<4,Kernel::intercept>>);
                    static_assert(Kernel::width==5 && Kernel::q==0);
                } else {
                    static_assert(std::is_same_v<typename Kernel::Design,mean::ConditionalArmaDesign<1,true>>);
                    static_assert(Kernel::p==1 && Kernel::q==1 && Kernel::width==4);
                }
            } else {
                static_assert(Kernel::width==std::dynamic_extent && Kernel::q==std::dynamic_extent);
                CHECK(Kernel::model_layout(plan).width==plan.width);
            }
            mean::admit_selected_binding(binding,admitted,[&](auto refined) {
                static_assert(std::is_same_v<Kernel,typename decltype(refined)::kernel_type>);
                static_assert(Kernel::fixed != std::is_same_v<decltype(refined.owner),u::RuntimeLayout>);
                CHECK(refined.owner.get()==admitted);
                ++visits;
            });
        });
        CHECK(visits==1);
        mean::dispatch_unit_layout<Density<Shapes>>(plan,dispatch,linear,intercept,[&](auto owner) {
            CHECK(owner.get()==base);
        });
    }
}

template<std::size_t Shapes> void cases() {
    for(bool intercept:{false,true}) {
        check_case<Shapes>(true,intercept,1,4,0);
        check_case<Shapes>(true,intercept,0,6,0);
        check_case<Shapes>(false,intercept,0,3+std::size_t(intercept),2);
    }
    check_case<Shapes>(false,true,2,2,1);
    check_case<Shapes>(true,false,0,0,0);
    check_case<Shapes>(false,false,0,0,0);
}
int main() {
    cases<0>();cases<1>();cases<2>();cases<3>();
    std::puts("mean binding metadata and generated-prefix routing passed");
}
